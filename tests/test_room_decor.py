import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "apps" / "web"
sys.path.insert(0, str(FRONTEND_DIR))

from room_decor import (  # noqa: E402
    RoomDecorError,
    RoomDecorStore,
    after_rain_from_hourly,
    parse_mode,
    resolve_scene,
    weather_from_wmo,
)
from room_manager import LiveRoom  # noqa: E402


class RoomDecorTests(unittest.IsolatedAsyncioTestCase):
    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(RoomDecorError) as caught:
            parse_mode("cats")
        self.assertEqual(caught.exception.status, 422)
        store = RoomDecorStore(Path(tempfile.mkdtemp()) / "room_decor.json")
        with self.assertRaises(RoomDecorError) as store_caught:
            store.set_mode("wallpaper")
        self.assertEqual(store_caught.exception.status, 422)
        self.assertEqual(store.public()["mode"], "auto")

    def test_mode_persists_and_weather_maps(self):
        path = Path(tempfile.mkdtemp()) / "room_decor.json"
        store = RoomDecorStore(path)
        store.set_mode("night")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["mode"], "night")
        reloaded = RoomDecorStore(path)
        self.assertEqual(reloaded.public()["mode"], "night")
        self.assertEqual(reloaded.public()["scene"], "night")
        self.assertEqual(weather_from_wmo(0), "clear")
        self.assertEqual(weather_from_wmo(61), "rain")
        self.assertEqual(weather_from_wmo(3), "overcast")

    def test_clock_uses_shanghai_night(self):
        from room_decor import period_from_clock
        noon_utc = datetime(2026, 9, 1, 13, 0, tzinfo=timezone.utc)
        self.assertEqual(period_from_clock(noon_utc), "night")

    def test_auto_scene_rules(self):
        self.assertEqual(
            resolve_scene(mode="auto", weather="rain", period="night"),
            "rain",
        )
        self.assertEqual(
            resolve_scene(mode="auto", weather="clear", period="night"),
            "night",
        )
        self.assertEqual(
            resolve_scene(mode="auto", weather="clear", period="day"),
            "sun",
        )
        self.assertEqual(
            resolve_scene(mode="auto", weather="clear", period="day", after_rain=True),
            "rainbow",
        )
        self.assertEqual(
            resolve_scene(mode="auto", weather="overcast", period="day"),
            "overcast",
        )
        self.assertEqual(
            resolve_scene(mode="auto", weather="unknown", period="night"),
            "night",
        )
        self.assertEqual(
            resolve_scene(mode="rainbow", weather="rain", period="night"),
            "rainbow",
        )

    def test_after_rain_from_hourly(self):
        now = datetime(2026, 9, 1, 15, 0, tzinfo=timezone.utc)
        payload = {
            "hourly": {
                "time": [
                    "2026-09-01T12:00",
                    "2026-09-01T13:00",
                    "2026-09-01T14:00",
                    "2026-09-01T15:00",
                ],
                "precipitation": [0.4, 0.0, 0.0, 0.0],
            }
        }
        self.assertTrue(after_rain_from_hourly(payload, now=now))
        payload["hourly"]["precipitation"] = [0.0, 0.0, 0.0, 0.0]
        self.assertFalse(after_rain_from_hourly(payload, now=now))

    async def test_snapshot_uses_decor_provider(self):
        room = LiveRoom(
            decor_provider=lambda: {
                "mode": "night",
                "weather": "clear",
                "scene": "night",
                "source": "open-meteo",
            }
        )
        alice, _ = await room.identify(None)
        state = await room.snapshot(alice.token)
        self.assertEqual(
            state["decor"],
            {"mode": "night", "weather": "clear", "scene": "night", "source": "open-meteo"},
        )

    async def test_weather_failure_falls_back_to_clock(self):
        store = RoomDecorStore(Path(tempfile.mkdtemp()) / "room_decor.json")
        with patch("room_decor.httpx.AsyncClient") as client_cls:
            client = AsyncMock()
            client.__aenter__.return_value = client
            client.get.side_effect = OSError("dns down")
            client_cls.return_value = client
            decor = await store.refresh_weather(force=True)
        self.assertEqual(decor["weather"], "unknown")
        self.assertEqual(decor["source"], "clock")
        self.assertIn(decor["scene"], {"sun", "night"})
        with patch("room_decor.httpx.AsyncClient") as client_cls:
            cached = await store.refresh_weather()
            client_cls.assert_not_called()
        self.assertEqual(cached["source"], "clock")


if __name__ == "__main__":
    unittest.main()
