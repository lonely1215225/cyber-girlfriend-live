import asyncio
import sys
import unittest
from pathlib import Path


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "apps" / "web"
sys.path.insert(0, str(FRONTEND_DIR))

from room_manager import MESSAGE_LIMIT, LiveRoom, RoomError  # noqa: E402


class LiveRoomTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.room = LiveRoom(queue_limit=2)
        self.alice, _ = await self.room.identify(None)
        self.bob, _ = await self.room.identify(None)
        self.cara, _ = await self.room.identify(None)

    async def test_generated_identity_and_rename(self):
        self.assertNotEqual(bool(self.alice.name_zh), bool(self.alice.name_en))
        generated = self.alice.name_zh or self.alice.name_en
        self.assertEqual(self.alice.display_name, generated)

        renamed = await self.room.rename(self.alice.token, "  小 明  ")
        self.assertEqual(renamed["name"], "小 明")
        await self.room.rename(self.bob.token, "独一无二")
        with self.assertRaises(RoomError) as caught:
            await self.room.rename(self.cara.token, "独一无二")
        self.assertEqual(caught.exception.code, "name_taken")

    async def test_proactive_news_needs_a_watching_viewer(self):
        self.assertFalse(await self.room.can_start_proactive())
        channel, arrived = await self.room.subscribe_presence(self.alice.token)
        self.assertTrue(arrived)
        self.assertTrue(await self.room.can_start_proactive())
        await self.room.unsubscribe(channel)
        self.assertFalse(await self.room.can_start_proactive())

    async def test_agent_updates_block_proactive_news_cooldown(self):
        await self.room.subscribe_presence(self.alice.token)
        self.assertTrue(await self.room.can_start_proactive())
        await self.room.publish_agent_job({
            "id": "job-one", "message_id": "msg-one", "participant_id": self.alice.id,
            "speaker": self.alice.display_name, "prompt": "查一下新闻", "phase": "planning",
            "status_text": "正在查询", "terminal": False,
        }, reply_to=None)
        self.assertFalse(await self.room.can_start_proactive())

    async def test_only_one_caller_and_fifo_queue(self):
        first = await self.room.request_session(self.alice.token)
        second = await self.room.request_session(self.bob.token)
        third = await self.room.request_session(self.cara.token)

        self.assertEqual(first["state"], "granted")
        self.assertEqual(second["position"], 1)
        self.assertEqual(third["position"], 2)

        still_waiting = await self.room.poll_queue(self.bob.token, second["queue_id"])
        self.assertEqual(still_waiting["state"], "queued")

        session_id, caller_name = await self.room.claim_websocket(
            self.alice.token, first["session_token"]
        )
        self.assertEqual(session_id, first["session_id"])
        self.assertEqual(caller_name, self.alice.display_name)
        self.assertTrue(await self.room.is_active_caller(self.alice.token))
        self.assertFalse(await self.room.is_active_caller(self.bob.token))
        with self.assertRaises(RoomError):
            await self.room.claim_websocket(self.alice.token, first["session_token"])

        self.assertTrue(await self.room.end_session(self.alice.token, session_id))
        self.assertFalse(await self.room.is_active_caller(self.alice.token))
        # Polling out of order cannot jump the first person in line.
        cara_waiting = await self.room.poll_queue(self.cara.token, third["queue_id"])
        self.assertEqual(cara_waiting["position"], 2)
        bob_grant = await self.room.poll_queue(self.bob.token, second["queue_id"])
        self.assertEqual(bob_grant["state"], "granted")

        cara_state = await self.room.snapshot(self.cara.token)
        self.assertEqual(cara_state["me"]["queue_position"], 1)
        self.assertEqual(cara_state["active"]["id"], self.bob.id)
        bob_viewer = next(viewer for viewer in cara_state["viewers"] if viewer["id"] == self.bob.id)
        self.assertEqual(bob_viewer["status"], "ready")

    async def test_queue_limit_and_explicit_leave(self):
        fourth, _ = await self.room.identify(None)
        await self.room.request_session(self.alice.token)
        bob_ticket = await self.room.request_session(self.bob.token)
        await self.room.request_session(self.cara.token)

        with self.assertRaises(RoomError) as caught:
            await self.room.request_session(fourth.token)
        self.assertEqual(caught.exception.code, "at_capacity")

        self.assertTrue(await self.room.leave_queue(self.bob.token, bob_ticket["queue_id"]))
        fourth_ticket = await self.room.request_session(fourth.token)
        self.assertEqual(fourth_ticket["position"], 2)

    async def test_presence_counts_event_stream_connections(self):
        alice_events = await self.room.subscribe(self.alice.token)
        bob_events = await self.room.subscribe(self.bob.token)
        try:
            state = await self.room.snapshot(self.alice.token)
            self.assertEqual(state["viewer_count"], 2)
            self.assertEqual(
                {viewer["name"] for viewer in state["viewers"]},
                {self.alice.display_name, self.bob.display_name},
            )
            self.assertTrue(all(viewer["status"] == "watching" for viewer in state["viewers"]))
            self.assertEqual(state["decor"]["mode"], "auto")
            self.assertEqual(state["decor"]["weather"], "unknown")
            self.assertEqual(state["decor"]["source"], "clock")
            self.assertIn(state["decor"]["scene"], {"sun", "night", "overcast", "rain", "rainbow"})
        finally:
            await self.room.unsubscribe(alice_events)
            await self.room.unsubscribe(bob_events)
            for task in tuple(self.room._disconnect_tasks.values()):
                task.cancel()
            await asyncio.gather(*self.room._disconnect_tasks.values(), return_exceptions=True)

    async def test_presence_marks_arrival_but_not_sse_reconnect(self):
        first, arrived = await self.room.subscribe_presence(self.alice.token)
        self.assertTrue(arrived)
        await self.room.unsubscribe(first)

        reconnect, arrived = await self.room.subscribe_presence(self.alice.token)
        self.assertFalse(arrived)
        await self.room.unsubscribe(reconnect)
        task = self.room._disconnect_tasks.pop(self.alice.token, None)
        self.assertIsNotNone(task)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        returned, arrived = await self.room.subscribe_presence(self.alice.token)
        self.assertTrue(arrived)
        await self.room.unsubscribe(returned)
        for pending in tuple(self.room._disconnect_tasks.values()):
            pending.cancel()
        await asyncio.gather(*self.room._disconnect_tasks.values(), return_exceptions=True)

    async def test_public_transcript_upserts_and_is_bounded(self):
        for index in range(MESSAGE_LIMIT + 2):
            await self.room.publish_transcript(
                session_id="call-1",
                event_id=f"line-{index}",
                role="assistant",
                speaker="小雅",
                text=f"第 {index} 句",
                partial=True,
            )
        await self.room.publish_transcript(
            session_id="call-1",
            event_id=f"line-{MESSAGE_LIMIT + 1}",
            role="assistant",
            speaker="小雅",
            text="最后一句完整内容",
            partial=False,
        )
        state = await self.room.snapshot(self.alice.token)
        self.assertEqual(len(state["messages"]), MESSAGE_LIMIT)
        self.assertEqual(state["messages"][-1]["text"], "最后一句完整内容")
        self.assertFalse(state["messages"][-1]["partial"])

    async def test_public_transcript_hides_reasoning_markup(self):
        await self.room.publish_transcript(
            session_id="call-think",
            event_id="answer",
            role="assistant",
            speaker="小雅",
            text="<think>不能公开的推理</think> 这是最终回答",
        )
        state = await self.room.snapshot(self.alice.token)
        self.assertEqual(state["messages"][-1]["text"], "这是最终回答")

        # A partial stream that has opened but not closed the reasoning block
        # should not publish anything yet.
        before = len(state["messages"])
        await self.room.publish_transcript(
            session_id="call-think",
            event_id="partial",
            role="assistant",
            speaker="小雅",
            text="<think>仍在推理",
            partial=True,
        )
        state = await self.room.snapshot(self.alice.token)
        self.assertEqual(len(state["messages"]), before)

    async def test_assistant_delivery_controls_never_reach_public_messages(self):
        await self.room.publish_transcript(
            session_id="call-expression",
            event_id="answer",
            role="assistant",
            speaker="小麻",
            text=(
                "smirk 0.6 cheerful 哟，敢不敢比？ "
                "happy 0.75 gentle 输了可别跑。"
            ),
        )
        state = await self.room.snapshot(self.alice.token)
        self.assertEqual(state["messages"][-1]["text"], "哟，敢不敢比？ 输了可别跑。")

        before = len(state["messages"])
        await self.room.publish_transcript(
            session_id="call-expression",
            event_id="partial-control",
            role="assistant",
            speaker="小麻",
            text="smirk 0.6 cheer",
            partial=True,
        )
        state = await self.room.snapshot(self.alice.token)
        self.assertEqual(len(state["messages"]), before)

        await self.room.publish_transcript(
            session_id="call-expression",
            event_id="partial-compact",
            role="assistant",
            speaker="小麻",
            text="happy 0.58 playful 0.56",
            partial=True,
        )
        state = await self.room.snapshot(self.alice.token)
        self.assertEqual(len(state["messages"]), before)

        compact = await self.room.publish_bot_reply(
            message_id="compact-expression-reply",
            text="happy 0.58 neutral 0.08 none 1.00 先说正事。",
            reply_to=None,
        )
        self.assertEqual(compact["text"], "先说正事。")

        reply = await self.room.publish_bot_reply(
            message_id="expression-reply",
            text="<e profile=wink intensity=0.7 style=cheerful>欢迎你呀。",
            reply_to=None,
        )
        self.assertEqual(reply["text"], "欢迎你呀。")

        protocol_reply = await self.room.publish_bot_reply(
            message_id="tool-protocol-reply",
            text=(
                "我先看一眼。<tool_call>web_fetch?url=https://example.com"
                "</tool_call>这是最终结论。"
            ),
            reply_to=None,
        )
        self.assertEqual(protocol_reply["text"], "我先看一眼。这是最终结论。")

    async def test_viewers_can_chat_and_are_rate_limited(self):
        message = await self.room.publish_chat(self.alice.token, "  大家 好  ")
        self.assertEqual(message["text"], "大家 好")
        self.assertEqual(message["speaker"], self.alice.display_name)
        self.assertEqual(message["role"], "viewer")
        state = await self.room.snapshot(self.bob.token)
        self.assertEqual(state["messages"][-1]["participant_id"], self.alice.id)

        for index in range(4):
            await self.room.publish_chat(self.alice.token, f"消息 {index}")
        with self.assertRaises(RoomError) as caught:
            await self.room.publish_chat(self.alice.token, "太快了")
        self.assertEqual(caught.exception.code, "rate_limited")

    async def test_bot_reply_is_quoted_and_yields_to_calls(self):
        message = await self.room.publish_chat(self.alice.token, "@小麻 你喜欢猫吗")
        self.assertTrue(await self.room.can_bot_reply())
        grant = await self.room.request_session(self.bob.token)
        self.assertFalse(await self.room.can_bot_reply())
        await self.room.end_session(self.bob.token, grant["session_id"])
        self.assertTrue(await self.room.can_bot_reply())
        reply = await self.room.publish_bot_reply(
            message_id=message["id"],
            text="喜欢呀，猫咪很可爱。",
            reply_to=message,
        )
        self.assertEqual(reply["speaker"], "小麻")
        self.assertEqual(reply["reply_to"]["id"], message["id"])

        status = await self.room.publish_bot_reply(
            message_id=message["id"],
            text="正在查询相关新闻…",
            reply_to=message,
            partial=True,
        )
        self.assertTrue(status["partial"])
        final = await self.room.publish_bot_reply(
            message_id=message["id"],
            text="这是查询后的最终回答",
            reply_to=message,
        )
        self.assertFalse(final["partial"])
        snapshot = await self.room.snapshot(self.alice.token)
        matching = [item for item in snapshot["messages"] if item["id"] == final["id"]]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["text"], "这是查询后的最终回答")

        proactive = await self.room.publish_bot_reply(
            message_id="proactive:1",
            text="刚看到一条很有意思的热点新闻。",
            reply_to=None,
        )
        self.assertNotIn("reply_to", proactive)

        interrupted = await self.room.publish_bot_reply(
            message_id="proactive:interrupted",
            text="已经实际播出的内容",
            reply_to=None,
            interrupted=True,
        )
        self.assertTrue(interrupted["interrupted"])


if __name__ == "__main__":
    unittest.main()
