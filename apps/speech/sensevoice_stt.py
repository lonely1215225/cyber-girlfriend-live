"""SenseVoiceSmall adapter for the speech-to-speech Paraformer slot.

speech-to-speech 0.2.10 exposes FunASR through its ``paraformer`` backend,
but its generic adapter drops the ModelScope namespace and returns SenseVoice
control tags as conversation text.  This tracked adapter keeps the upstream
pipeline untouched while providing the SenseVoice-specific setup and cleanup.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Any, Iterator

import numpy as np
import torch
from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

LOG = logging.getLogger(__name__)
CONSOLE = Console()
_CONTROL_TOKEN = re.compile(r"<\|[^|>]+\|>")
_PRIVATE_LANGUAGE_PREFIX = "sensevoice:"
_LANGUAGES = {"zh", "en", "yue", "ja", "ko"}
_EMOTIONS = {
    "HAPPY": "开心",
    "SAD": "难过",
    "ANGRY": "生气",
    "FEARFUL": "害怕",
    "DISGUSTED": "厌恶",
    "SURPRISED": "惊讶",
}
_EVENTS = {
    "Laughter": "笑声",
    "Cry": "哭声",
    "Applause": "掌声",
}
_TURN_CONTEXT_MAX = 256
_TURN_CONTEXT_TTL_SECONDS = 900.0
_turn_context_lock = Lock()
_turn_contexts: OrderedDict[tuple[str, int | None], tuple[float, "SenseVoiceMetadata"]] = OrderedDict()


@dataclass(frozen=True)
class SenseVoiceMetadata:
    language: str | None = None
    emotion: str | None = None
    event: str | None = None


def remember_turn_context(
    turn_id: str | None,
    turn_revision: int | None,
    metadata: SenseVoiceMetadata,
) -> None:
    """Keep short-lived acoustic context for the matching TTS response."""

    if not turn_id:
        return
    now = monotonic()
    key = (turn_id, turn_revision)
    with _turn_context_lock:
        _turn_contexts[key] = (now, metadata)
        _turn_contexts.move_to_end(key)
        while _turn_contexts:
            first_key, (created_at, _) = next(iter(_turn_contexts.items()))
            if len(_turn_contexts) <= _TURN_CONTEXT_MAX and now - created_at <= _TURN_CONTEXT_TTL_SECONDS:
                break
            _turn_contexts.pop(first_key, None)


def get_turn_context(turn_id: str | None, turn_revision: int | None) -> SenseVoiceMetadata | None:
    if not turn_id:
        return None
    now = monotonic()
    with _turn_context_lock:
        item = _turn_contexts.get((turn_id, turn_revision))
        if item is None:
            return None
        created_at, metadata = item
        if now - created_at > _TURN_CONTEXT_TTL_SECONDS:
            _turn_contexts.pop((turn_id, turn_revision), None)
            return None
        return metadata


def parse_sensevoice_metadata(raw_text: str) -> SenseVoiceMetadata:
    """Extract stable SenseVoice categories while ignoring unknown/noisy tags."""

    tags = _CONTROL_TOKEN.findall(raw_text)
    names = [tag[2:-2] for tag in tags]
    language = next((name for name in names if name in _LANGUAGES), None)

    emotion_counts = Counter(name for name in names if name in _EMOTIONS)
    event_counts = Counter(name for name in names if name in _EVENTS)
    emotion_key = emotion_counts.most_common(1)[0][0] if emotion_counts else None
    event_key = event_counts.most_common(1)[0][0] if event_counts else None
    return SenseVoiceMetadata(
        language=language,
        emotion=_EMOTIONS.get(emotion_key) if emotion_key else None,
        event=_EVENTS.get(event_key) if event_key else None,
    )


def encode_sensevoice_context(metadata: SenseVoiceMetadata) -> str | None:
    """Transport private acoustic context through the existing language field."""

    if not (metadata.language or metadata.emotion or metadata.event):
        return None
    values = (metadata.language or "", metadata.emotion or "", metadata.event or "")
    return _PRIVATE_LANGUAGE_PREFIX + "|".join(values)


def decode_sensevoice_context(value: str | None) -> SenseVoiceMetadata | None:
    if not value or not value.startswith(_PRIVATE_LANGUAGE_PREFIX):
        return None
    parts = value[len(_PRIVATE_LANGUAGE_PREFIX) :].split("|", 2)
    parts += [""] * (3 - len(parts))
    return SenseVoiceMetadata(*(part or None for part in parts))


def build_emotion_aware_prompt(transcript: str, metadata: SenseVoiceMetadata) -> str:
    """Add non-display acoustic hints to the user turn consumed by the LLM."""

    hints = []
    if metadata.emotion:
        hints.append(f"用户语音情绪可能是{metadata.emotion}")
    if metadata.event:
        hints.append(f"语音中检测到{metadata.event}")
    if not hints:
        return transcript
    context = "，".join(hints)
    return (
        f"[内部声学线索：{context}。这是模型推测，仅用于调整回复的共情程度和语气；"
        "不要直接复述标签，也不要声称你进行了情绪识别。]\n"
        f"用户原话：{transcript}"
    )


def clean_sensevoice_text(raw_text: str) -> str:
    """Return only spoken text, without language/emotion/event control tags."""

    # Use FunASR's official postprocessor first.  Keep a small fallback so a
    # packaging mismatch never leaks raw model tags into the LLM prompt.
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        text = rich_transcription_postprocess(raw_text)
    except (ImportError, KeyError, TypeError, ValueError):
        text = raw_text

    text = _CONTROL_TOKEN.sub("", text)
    # The official rich postprocessor renders the task tokens as these leading
    # annotations. They are useful in an ASR console, but not as user speech.
    text = text.strip(" \t\r\n☕☹☺❤️🎼😂😡😭😱😊😐😍")
    return " ".join(text.split()).strip()


class SenseVoiceSTTHandler(BaseSTTHandler):
    """FunASR SenseVoiceSmall handler for short, VAD-segmented live audio."""

    def setup(
        self,
        model_name: str = "iic/SenseVoiceSmall",
        device: str = "cuda",
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        try:
            from funasr import AutoModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SenseVoice STT requires FunASR. Run ./install.sh or install funasr==1.4.2."
            ) from exc

        self.language = os.environ.get("SENSEVOICE_LANGUAGE", "auto")
        self.use_itn = os.environ.get("SENSEVOICE_USE_ITN", "1") != "0"
        self.emotion_enabled = os.environ.get("SENSEVOICE_EMOTION_ENABLED", "1") != "0"
        self.gen_kwargs = dict(gen_kwargs or {})
        self.device = device
        LOG.info(
            "Loading SenseVoice STT model=%s device=%s language=%s itn=%s emotion=%s",
            model_name,
            device,
            self.language,
            self.use_itn,
            self.emotion_enabled,
        )
        self.model = AutoModel(
            model=model_name,
            device=device,
            disable_update=True,
        )
        self.warmup()

    def _transcribe(self, audio: np.ndarray) -> tuple[str, SenseVoiceMetadata]:
        result = self.model.generate(
            input=audio,
            cache={},
            language=self.language,
            use_itn=self.use_itn,
            ban_emo_unk=True,
            **self.gen_kwargs,
        )
        if not result:
            return "", SenseVoiceMetadata()
        raw_text = str(result[0].get("text", ""))
        metadata = parse_sensevoice_metadata(raw_text)
        if not self.emotion_enabled:
            metadata = SenseVoiceMetadata(language=metadata.language)
        return clean_sensevoice_text(raw_text), metadata

    def warmup(self) -> None:
        LOG.info("Warming up %s", self.__class__.__name__)
        self._transcribe(np.zeros(1600, dtype=np.float32))

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        pred_text, metadata = self._transcribe(vad_audio.audio)
        if self.device.startswith("mps"):
            torch.mps.empty_cache()

        LOG.info(
            "SenseVoice transcription: %s (language=%s emotion=%s event=%s)",
            pred_text,
            metadata.language,
            metadata.emotion,
            metadata.event,
        )
        CONSOLE.print(f"[yellow]USER: {pred_text}")
        if vad_audio.mode == "progressive":
            yield PartialTranscription(
                text=pred_text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
            )
        else:
            remember_turn_context(vad_audio.turn_id, vad_audio.turn_revision, metadata)
            yield Transcription(
                text=pred_text,
                language_code=encode_sensevoice_context(metadata),
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
                speech_stopped_at_s=vad_audio.created_at_s,
            )


def install_sensevoice_adapter() -> None:
    """Replace only the generic Paraformer class used by the upstream CLI."""

    import speech_to_speech.STT.paraformer_handler as paraformer_handler

    paraformer_handler.ParaformerSTTHandler = SenseVoiceSTTHandler
    _install_emotion_context_bridge()
    LOG.info("SenseVoice adapter installed for speech-to-speech paraformer backend")


def _install_emotion_context_bridge() -> None:
    """Keep acoustic hints private while enriching the LLM conversation turn."""

    from speech_to_speech.api.openai_realtime import service as service_module
    from speech_to_speech.LLM.chat import make_user_message
    from speech_to_speech.pipeline.messages import GenerateResponseRequest

    if getattr(service_module.RealtimeService, "_sensevoice_context_installed", False):
        return

    def on_transcription_completed(self, conn_id, event):
        metadata = decode_sensevoice_context(event.language_code)
        public_event = event
        llm_transcript = event.transcript
        if metadata is not None:
            public_event = event.model_copy(update={"language_code": metadata.language})
            llm_transcript = build_emotion_aware_prompt(event.transcript, metadata)

        st = self._state(conn_id)
        same_speculative_turn = event.turn_id is not None and event.turn_id == st.speculative_user_turn_id
        if same_speculative_turn:
            st.response_usage.audio_duration_s -= st.speculative_audio_duration_s
        else:
            st.speculative_audio_duration_s = 0.0

        # Only the clean transcript and real language code reach the browser.
        events = self.conversation.on_transcription_completed(conn_id, public_event)
        if event.turn_id is not None:
            st.speculative_audio_duration_s = st.input_audio_duration_s

        cfg = st.runtime_config
        transcript = event.transcript
        if transcript:
            if same_speculative_turn and st.speculative_user_item_id:
                replaced = cfg.chat.replace_user_message_text(st.speculative_user_item_id, llm_transcript)
                if not replaced:
                    item = cfg.chat.add_item(make_user_message(llm_transcript))
                    st.speculative_user_item_id = item.id
            else:
                item = cfg.chat.add_item(make_user_message(llm_transcript))
                st.speculative_user_item_id = item.id
        elif same_speculative_turn and st.speculative_user_item_id:
            cfg.chat.remove_user_message(st.speculative_user_item_id)
            st.speculative_user_item_id = None
        elif event.turn_id is not None and event.turn_id != st.speculative_user_turn_id:
            st.speculative_user_item_id = None

        if event.turn_id is not None:
            st.speculative_user_turn_id = event.turn_id
            st.speculative_user_turn_revision = event.turn_revision
            st.speculative_user_speech_stopped_at_s = event.speech_stopped_at_s

        queue = self.text_prompt_queue
        if queue and transcript:
            st.response_pending = True
            queue.put(
                GenerateResponseRequest(
                    runtime_config=cfg,
                    language_code=public_event.language_code,
                    turn_id=event.turn_id,
                    turn_revision=event.turn_revision,
                    speech_stopped_at_s=event.speech_stopped_at_s,
                )
            )
        return events

    service_module.RealtimeService._on_transcription_completed = on_transcription_completed
    service_module.RealtimeService._sensevoice_context_installed = True
