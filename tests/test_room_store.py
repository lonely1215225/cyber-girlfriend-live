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

    async def test_legacy_welcome_is_backfilled_into_its_users_context(self):
        self.assertTrue(await self.store.create_user(
            user_id="viewer-123", display_name="林清欢", name_zh="林清欢", name_en="",
            token="welcome-owner-token",
        ))
        welcome = "林清欢，你一来，今晚的月亮都像偷偷调亮了一格呀。"
        await self.store.save_message({
            "id": "bot:welcome:viewer-123:1000:speech:1:0",
            "kind": "mention_reply", "role": "assistant", "speaker": "小麻",
            "text": welcome, "created_at": 1000.0,
        })
        before = await self.store.memory_context("viewer-123", "刚才怎么欢迎我的")
        self.assertNotIn(welcome, before)

        # Reopening the store runs the compatibility migration.
        restored = RoomStore(self.store.path)
        await restored.initialize()
        after = await restored.memory_context("viewer-123", "刚才怎么欢迎我的")
        self.assertIn(welcome, after)

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

    async def test_room_setting_is_global_persistent_and_versioned(self):
        missing = await self.store.room_setting("avatar_transport", "webrtc")
        self.assertEqual(missing["value"], "webrtc")
        self.assertEqual(missing["revision"], 0)
        first = await self.store.set_room_setting("avatar_transport", "http-flv")
        second = await self.store.set_room_setting("avatar_transport", "webrtc")
        self.assertEqual(first["revision"], 1)
        self.assertEqual(second["revision"], 2)
        restored = RoomStore(self.store.path)
        await restored.initialize()
        current = await restored.room_setting("avatar_transport", "http-flv")
        self.assertEqual(current["value"], "webrtc")
        self.assertEqual(current["revision"], 2)

    async def test_agent_jobs_are_persistent_recoverable_and_keep_context(self):
        base = {
            "id": "aj-one", "message_id": "chat-one", "participant_id": "user-a",
            "speaker": "林知夏", "prompt": "现在比特币多少钱", "phase": "researching",
            "status_text": "正在核对最新价格", "terminal": False,
        }
        saved = await self.store.save_agent_job(base)
        self.assertEqual(saved["phase"], "researching")
        recovering = await self.store.load_agent_jobs(recoverable_only=True)
        self.assertEqual([item["id"] for item in recovering], ["aj-one"])

        await self.store.save_agent_job({
            **base, "phase": "completed", "status_text": "现在大约是65,000美元。",
            "final_text": "现在大约是65,000美元。", "terminal": True,
        })
        self.assertEqual(await self.store.load_agent_jobs(recoverable_only=True), [])
        context = await self.store.recent_agent_context("user-a")
        self.assertIn("比特币", context)
        self.assertIn("65,000美元", context)

    async def test_failed_agent_output_is_not_reused_as_private_memory(self):
        self.assertTrue(await self.store.create_user(
            user_id="user-a", display_name="林知夏", name_zh="林知夏", name_en="",
            token="failed-memory-token",
        ))
        await self.store.save_message({
            "id": "chat-failed", "kind": "chat", "role": "viewer", "speaker": "林知夏",
            "text": "@小麻 你是机器人吗", "created_at": 100.0,
        }, user_id="user-a")
        await self.store.save_message({
            "id": "bot-failed", "kind": "mention_reply", "role": "assistant", "speaker": "小麻",
            "text": "正在核对实时行情和币种。", "created_at": 101.0,
            "reply_to": {"id": "chat-failed", "speaker": "林知夏", "text": "@小麻 你是机器人吗"},
        }, user_id="user-a")
        await self.store.save_agent_job({
            "id": "aj-failed", "message_id": "chat-failed", "participant_id": "user-a",
            "speaker": "林知夏", "prompt": "你是机器人吗", "phase": "failed",
            "status_text": "本轮失败", "terminal": True, "error": "off-topic",
        })
        memory = await self.store.memory_context("user-a", "刚才机器人怎么回事")
        self.assertNotIn("实时行情", memory)
        self.assertIn("你是机器人吗", memory)

    async def test_conversation_focus_is_private_and_persistent(self):
        spec = {
            "subject": "bitcoin", "subject_label": "比特币",
            "aliases": ["比特币", "bitcoin", "btc"], "intent": "price_lookup",
            "resolved_question": "现在比特币多少钱",
        }
        await self.store.set_conversation_focus(
            "user-a", spec, "现在约65000美元", "2026-08-24T00:00:00Z"
        )
        focus = await self.store.get_conversation_focus("user-a")
        self.assertEqual(focus["subject"], "bitcoin")
        self.assertEqual(focus["last_intent"], "price_lookup")
        self.assertIn("65000", focus["last_answer"])
        self.assertEqual(await self.store.get_conversation_focus("user-b"), {})

    async def test_only_one_full_news_topic_and_small_history_are_kept(self):
        await self.store.set_active_news_topic({
            "fingerprint": "event-one", "title": "比特币突破八万美元",
            "title_normalized": "比特币突破80000美元", "category": "新闻",
            "source": "人民日报", "summary": "市场价格出现上涨。",
            "evidence": "可信的第一条资料", "locked_until": 9_999_999_999,
        })
        context = await self.store.active_news_context("这个为什么会上涨")
        self.assertIn("比特币突破八万美元", context)
        self.assertIn("可信的第一条资料", context)
        self.assertEqual(await self.store.active_news_context("你喜欢什么动物"), "")

        await self.store.set_active_news_topic({
            "fingerprint": "event-two", "title": "新款芯片正式发布",
            "title_normalized": "新款芯片正式发布", "category": "科技",
            "source": "IT之家", "evidence": "可信的第二条资料",
            "locked_until": 9_999_999_999,
        })
        current = await self.store.active_news_context("刚才那条新闻怎么样")
        self.assertIn("新款芯片正式发布", current)
        self.assertNotIn("比特币突破八万美元", current)
        titles = await self.store.recent_news_titles()
        self.assertEqual(set(titles), {"比特币突破八万美元", "新款芯片正式发布"})


if __name__ == "__main__":
    unittest.main()
