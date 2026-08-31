from __future__ import annotations

import unittest

from tests.import_paths import add_service_paths

add_service_paths()

from avtr1_gateway import FlvMuxer, SAMPLE_RATE  # noqa: E402


def _tag_timestamp(tag: bytes) -> int:
    return (tag[4] << 16) | (tag[5] << 8) | tag[6] | (tag[7] << 24)


class FlvAudioTimestampTests(unittest.TestCase):
    def test_aac_tags_follow_the_sample_clock_not_the_video_tick(self) -> None:
        muxer = FlvMuxer()
        frame_bytes = 1024 * 2
        stamps: list[int] = []
        for _ in range(4):
            muxer.timestamp_ms += 40
            for tag in muxer.audio_tags(bytes(frame_bytes)):
                if tag[0] == 8 and len(tag) > 13 and tag[11] == 0xAE and tag[12] == 0x01:
                    stamps.append(_tag_timestamp(tag))
        self.assertGreaterEqual(len(stamps), 3)
        step = int(1024 * 1000 / SAMPLE_RATE)
        gaps = [stamps[index + 1] - stamps[index] for index in range(len(stamps) - 1)]
        self.assertTrue(all(gap == step for gap in gaps), gaps)
        self.assertNotIn(40, gaps)


if __name__ == "__main__":
    unittest.main()
