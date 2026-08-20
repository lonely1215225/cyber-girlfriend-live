from __future__ import annotations

import sys
import unittest
from pathlib import Path
from queue import Queue
from types import SimpleNamespace


PROXY = Path(__file__).resolve().parents[1] / "proxy"
if str(PROXY) not in sys.path:
    sys.path.insert(0, str(PROXY))

from sensevoice_stt import (  # noqa: E402
    SenseVoiceMetadata,
    build_emotion_aware_prompt,
    clean_sensevoice_text,
    decode_sensevoice_context,
    encode_sensevoice_context,
    parse_sensevoice_metadata,
    _install_emotion_context_bridge,
)


class SenseVoiceTextTests(unittest.TestCase):
    def test_clean_control_tokens(self) -> None:
        self.assertEqual(
            clean_sensevoice_text("<|zh|><|NEUTRAL|><|Speech|><|withitn|>今天天气很好。"),
            "今天天气很好。",
        )

    def test_clean_rich_annotations(self) -> None:
        self.assertEqual(clean_sensevoice_text("😊今天天气很好。☕"), "今天天气很好。")

    def test_extracts_emotion_language_and_event(self) -> None:
        raw = "<|zh|><|HAPPY|><|Laughter|><|withitn|>今天真开心。"
        self.assertEqual(
            parse_sensevoice_metadata(raw),
            SenseVoiceMetadata(language="zh", emotion="开心", event="笑声"),
        )

    def test_neutral_and_unknown_are_not_injected(self) -> None:
        raw = "<|en|><|NEUTRAL|><|Speech|><|withitn|>Hello."
        metadata = parse_sensevoice_metadata(raw)
        self.assertEqual(metadata, SenseVoiceMetadata(language="en"))
        self.assertEqual(build_emotion_aware_prompt("Hello.", metadata), "Hello.")

    def test_private_context_round_trip_and_prompt(self) -> None:
        metadata = SenseVoiceMetadata(language="zh", emotion="难过", event="哭声")
        encoded = encode_sensevoice_context(metadata)
        self.assertEqual(decode_sensevoice_context(encoded), metadata)
        prompt = build_emotion_aware_prompt("我没事。", metadata)
        self.assertIn("情绪可能是难过", prompt)
        self.assertIn("检测到哭声", prompt)
        self.assertIn("用户原话：我没事。", prompt)

    def test_realtime_bridge_keeps_public_text_clean(self) -> None:
        from speech_to_speech.api.openai_realtime.service import RealtimeService
        from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
        from speech_to_speech.LLM.chat import Chat
        from speech_to_speech.pipeline.events import TranscriptionCompletedEvent

        _install_emotion_context_bridge()

        class Conversation:
            received = None

            def on_transcription_completed(self, _conn_id, event):
                self.received = event
                return ["public-event"]

        state = SimpleNamespace(
            speculative_user_turn_id=None,
            speculative_audio_duration_s=0.0,
            input_audio_duration_s=1.0,
            response_usage=SimpleNamespace(audio_duration_s=1.0),
            runtime_config=RuntimeConfig(chat=Chat(12)),
            speculative_user_item_id=None,
            speculative_user_turn_revision=None,
            speculative_user_speech_stopped_at_s=None,
            response_pending=False,
        )
        conversation = Conversation()
        service = SimpleNamespace(
            _state=lambda _conn_id: state,
            conversation=conversation,
            text_prompt_queue=Queue(),
        )
        metadata = SenseVoiceMetadata(language="zh", emotion="难过")
        event = TranscriptionCompletedEvent(
            transcript="我没事。",
            language_code=encode_sensevoice_context(metadata),
            turn_id="turn-1",
            turn_revision=0,
        )

        result = RealtimeService._on_transcription_completed(service, "conn-1", event)
        self.assertEqual(result, ["public-event"])
        self.assertEqual(conversation.received.transcript, "我没事。")
        self.assertEqual(conversation.received.language_code, "zh")
        llm_text = state.runtime_config.chat.buffer[0].content[0].text
        self.assertIn("情绪可能是难过", llm_text)
        self.assertIn("用户原话：我没事。", llm_text)
        request = service.text_prompt_queue.get_nowait()
        self.assertEqual(request.language_code, "zh")


if __name__ == "__main__":
    unittest.main()
