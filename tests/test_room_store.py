import tempfile
import unittest
from pathlib import Path
import sys


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "s2s" / "hf-realtime-voice"
sys.path.insert(0, str(FRONTEND_DIR))

from room_store import RoomStore  # noqa: E402


class RoomStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = RoomStore(Path(self.tempdir.name) / "room.sqlite3")
        await self.store.initialize()

    async def asyncTearDown(self):
        self.tempdir.cleanup()

    async def test_user_session_name_messages_and_private_memory_survive(self):
        created = await self.store.create_user(
            user_id="user-a",
            display_name="林知夏",
            name_zh="林知夏",
            name_en="",
            token="secret-browser-token",
            ip_address="203.0.113.8",
            user_agent="test-browser",
        )
        self.assertTrue(created)
        restored = await self.store.resolve_user(
            "secret-browser-token", ip_address="203.0.113.9", user_agent="test-browser"
        )
        self.assertEqual(restored["display_name"], "林知夏")
        self.assertTrue(await self.store.rename_user("user-a", "小林"))

        await self.store.save_message(
            {
                "id": "m-user",
                "kind": "voice",
                "role": "user",
                "speaker": "小林",
                "text": "我特别喜欢橘猫，也喜欢看科幻动漫。",
                "created_at": 100.0,
            },
            user_id="user-a",
        )
        await self.store.save_message(
            {
                "id": "m-assistant",
                "kind": "voice",
                "role": "assistant",
                "speaker": "小雅",
                "text": "那以后可以多聊聊猫和科幻作品呀。",
                "created_at": 101.0,
            },
            user_id="user-a",
        )
        await self.store.save_message(
            {
                "id": "m-old-rag",
                "kind": "voice",
                "role": "user",
                "speaker": "小林",
                "text": "上次我们认真聊到了天文学和脉冲星。",
                "created_at": 102.0,
            },
            user_id="user-a",
        )
        for index in range(12):
            await self.store.save_message(
                {
                    "id": f"m-filler-{index}",
                    "kind": "chat",
                    "role": "viewer",
                    "speaker": "小林",
                    "text": f"这是后来的普通聊天内容第{index}条。",
                    "created_at": 200.0 + index,
                },
                user_id="user-a",
            )
        self.assertTrue(await self.store.create_user(
            user_id="user-b", display_name="Robin Sage", name_zh="", name_en="Robin Sage",
            token="other-browser-token",
        ))
        await self.store.save_message(
            {"id": "m-private-b", "kind": "chat", "role": "viewer", "speaker": "Robin Sage",
             "text": "我喜欢养蛇。", "created_at": 300.0},
            user_id="user-b",
        )
        memory = await self.store.memory_context("user-a", "你还记得我喜欢什么猫吗")
        self.assertIn("橘猫", memory)
        self.assertIn("科幻", memory)
        self.assertNotIn("养蛇", memory)

        rag_memory = await self.store.memory_context("user-a", "我们以前聊过脉冲星吗")
        self.assertIn("脉冲星", rag_memory)

        history = await self.store.load_recent_messages()
        self.assertIn("m-private-b", [item["id"] for item in history])

    async def test_admin_sessions_are_independent_and_revocable(self):
        first = await self.store.create_admin_session(
            ip_address="203.0.113.10", user_agent="browser-a", ttl_seconds=1800
        )
        second = await self.store.create_admin_session(
            ip_address="203.0.113.11", user_agent="browser-b", ttl_seconds=1800
        )
        self.assertIsNotNone(await self.store.admin_session(first))
        self.assertIsNotNone(await self.store.admin_session(second))
        await self.store.revoke_admin_session(first)
        self.assertIsNone(await self.store.admin_session(first))
        self.assertIsNotNone(await self.store.admin_session(second))


if __name__ == "__main__":
    unittest.main()
