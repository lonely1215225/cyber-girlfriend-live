from __future__ import annotations

import asyncio
import unittest

from tests.import_paths import add_service_paths

add_service_paths()

from avatar_tee import LocalAvatarTee  # noqa: E402


class AvatarTeeFinishTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tee = LocalAvatarTee("http://127.0.0.1:9")

    async def asyncTearDown(self):
        if self.tee.segment_gap_task is not None:
            self.tee.segment_gap_task.cancel()
        if self.tee._finish_task is not None:
            self.tee._finish_task.cancel()
        if self.tee.pump_task is not None:
            self.tee.pump_task.cancel()

    async def test_schedule_finish_closes_the_turn_after_the_hold(self):
        self.tee.schedule_finish(complete=False, delay=0.02)
        self.assertFalse(self.tee.done)
        await asyncio.sleep(0.05)
        self.assertTrue(self.tee.done)

    async def test_new_pcm_cancels_a_scheduled_finish(self):
        self.tee.schedule_finish(complete=False, delay=0.03)
        self.tee.feed(b"\x00\x00")
        await asyncio.sleep(0.06)
        self.assertFalse(self.tee.done)
        self.assertIsNone(self.tee._finish_task)

    async def test_keep_turn_open_cancels_finish_and_holds_the_gap(self):
        self.tee.started_at = 1.0
        self.tee.last_feed_at = 1.0
        self.tee.schedule_finish(complete=False, delay=0.02)
        self.tee.keep_turn_open(delay=0.2)
        await asyncio.sleep(0.05)
        self.assertFalse(self.tee.done)
        self.assertIsNone(self.tee._finish_task)
        self.assertIsNotNone(self.tee.segment_gap_task)
