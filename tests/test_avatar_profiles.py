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
        await self.store.set_pending("two")
        self.assertEqual((await self.store.active())["pending_avatar_id"], "two")
        await self.store.activate("two")
        active = await self.store.active()
        self.assertEqual(active["avatar_id"], "two")
        self.assertEqual(active["pending_avatar_id"], "")
        self.assertEqual(await self.store.active_persona(), "你叫阿月，性格活泼，喜欢音乐。")

    async def test_empty_persona_uses_safe_default_and_length_is_limited(self):
        await self.store.update_profile("one", persona_prompt="")
        self.assertIn("你叫小雅", await self.store.active_persona())
        with self.assertRaises(ValueError):
            await self.store.update_profile("one", persona_prompt="太" * 12001)

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
        self.assertEqual(voice["sample_rate"], 24000)
        await self.store.update_profile("one", voice_asset_id=voice["id"])
        with self.assertRaises(ValueError):
            await self.store.archive_voice(voice["id"])


if __name__ == "__main__":
    unittest.main()
