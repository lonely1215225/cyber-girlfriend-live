import asyncio
import math
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "s2s" / "hf-realtime-voice"))

from avatar_profiles import AvatarProfileStore  # noqa: E402


class AvatarProfileStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        reference = root / "reference.wav"
        with wave.open(str(reference), "wb") as output:
            output.setparams((1, 2, 24000, 0, "NONE", ""))
            output.writeframes(struct.pack("<h", 400) * (24000 * 4))
        self.store = AvatarProfileStore(root / "room.sqlite3", root / "data", reference)
        await self.store.initialize(
            [{"id": "one", "label": "角色一"}, {"id": "two", "label": "角色二"}],
            "one", {"blink_enabled": True},
        )

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def test_profiles_are_isolated_and_activation_is_atomic(self):
        await self.store.update_profile(
            "two",
            view={"size": 110, "position": -2, "vertical": 3, "fade": 20},
            persona_prompt="你叫阿月，性格活泼，喜欢音乐。",
        )
        profiles = await self.store.profiles()
        one, two = profiles["profiles"]
        self.assertEqual(one["view"]["size"], 100)
        self.assertEqual(two["view"]["size"], 110)
        self.assertNotEqual(one["persona_prompt"], two["persona_prompt"])
        self.assertEqual(two["persona_prompt"], "你叫阿月，性格活泼，喜欢音乐。")
        self.assertEqual(two["tts"]["provider"], "fish")
        await self.store.set_pending("two")
        self.assertEqual((await self.store.active())["pending_avatar_id"], "two")
        await self.store.activate("two")
        active = await self.store.active()
        self.assertEqual(active["avatar_id"], "two")
        self.assertEqual(active["pending_avatar_id"], "")
        self.assertEqual(await self.store.active_persona(), "你叫阿月，性格活泼，喜欢音乐。")

    async def test_empty_persona_uses_safe_default_and_length_is_limited(self):
        await self.store.update_profile("one", persona_prompt="")
        persona = await self.store.active_persona()
        self.assertIn("你叫小麻", persona)
        self.assertIn("既然", persona)
        self.assertIn("磕绊", persona)
        with self.assertRaises(ValueError):
            await self.store.update_profile("one", persona_prompt="太" * 12001)

    async def test_shipped_persona_upgrades_old_builtin_wording(self):
        from avatar_profiles import DEFAULT_PERSONA_PROMPT

        with self.store._connect() as db:
            db.execute(
                "UPDATE avatar_profiles SET persona_prompt=? WHERE avatar_id='one'",
                (
                    "你叫小麻，是直播间里甜甜又有点坏的女孩。"
                    "用自然标点，只输出能直接说出口的话；不用Markdown、列表、表情符号或思考过程。",
                ),
            )
        await self.store.initialize(
            [{"id": "one", "label": "角色一"}, {"id": "two", "label": "角色二"}],
            "one",
            {"blink_enabled": True},
        )
        self.assertEqual(await self.store.active_persona(), DEFAULT_PERSONA_PROMPT)

    async def test_custom_persona_is_not_replaced_by_builtin_refresh(self):
        custom = "你叫小麻，说话短一点。不要用“既然……那就……”这类书面转折。只陪对方聊天。"
        await self.store.update_profile("one", persona_prompt=custom)
        await self.store.initialize(
            [{"id": "one", "label": "角色一"}, {"id": "two", "label": "角色二"}],
            "one",
            {"blink_enabled": True},
        )
        self.assertEqual(await self.store.active_persona(), custom)

    async def test_system_voice_cannot_be_archived_and_view_is_validated(self):
        with self.assertRaises(ValueError):
            await self.store.archive_voice("system-default-xiaoya")
        with self.assertRaises(ValueError):
            await self.store.update_profile("one", view={"size": 999})

    async def test_uploaded_voice_is_normalized_bound_and_protected(self):
        source = Path(self.tmp.name) / "upload.wav"
        with wave.open(str(source), "wb") as output:
            output.setparams((1, 2, 16000, 0, "NONE", ""))
            frames = [int(5000 * math.sin(2 * math.pi * 220 * index / 16000)) for index in range(16000 * 4)]
            output.writeframes(struct.pack(f"<{len(frames)}h", *frames))
        voice = await self.store.create_voice(source.read_bytes(), name="测试", ref_text="测试参考文本", source="upload", suffix=".wav")
        self.assertEqual(voice["status"], "ready")
        self.assertEqual(voice["sample_rate"], 48000)
        await self.store.update_profile("one", voice_asset_id=voice["id"])
        with self.assertRaises(ValueError):
            await self.store.archive_voice(voice["id"])

    async def test_role_emotion_references_are_isolated_and_protected(self):
        source = Path(self.tmp.name) / "emotion.wav"
        with wave.open(str(source), "wb") as output:
            output.setparams((1, 2, 16000, 0, "NONE", ""))
            frames = [int(4200 * math.sin(2 * math.pi * 260 * index / 16000)) for index in range(16000 * 4)]
            output.writeframes(struct.pack(f"<{len(frames)}h", *frames))
        emotion = await self.store.create_voice(
            source.read_bytes(), name="轻笑参考", ref_text="哈哈，被你发现了。",
            source="upload", suffix=".wav",
        )
        updated = await self.store.update_profile(
            "one", emotion_references={"soft_laugh": emotion["id"], "playful": emotion["id"]}
        )
        self.assertEqual(updated["tts"]["emotion_references"]["soft_laugh"], emotion["id"])
        profiles = await self.store.profiles()
        other = next(item for item in profiles["profiles"] if item["avatar_id"] == "two")
        self.assertEqual(other["tts"]["emotion_references"], {})
        with self.assertRaises(ValueError):
            await self.store.archive_voice(emotion["id"])
        with self.assertRaises(ValueError):
            await self.store.update_profile("one", emotion_references={"unknown": emotion["id"]})


if __name__ == "__main__":
    unittest.main()
