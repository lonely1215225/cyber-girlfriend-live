import asyncio
import sys
import unittest
from pathlib import Path


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "s2s" / "hf-realtime-voice"
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
        with self.assertRaises(RoomError):
            await self.room.claim_websocket(self.alice.token, first["session_token"])

        self.assertTrue(await self.room.end_session(self.alice.token, session_id))
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
        finally:
            await self.room.unsubscribe(alice_events)
            await self.room.unsubscribe(bob_events)
            for task in tuple(self.room._disconnect_tasks.values()):
                task.cancel()
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


if __name__ == "__main__":
    unittest.main()
