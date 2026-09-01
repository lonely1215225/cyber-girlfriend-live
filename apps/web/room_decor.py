"""Shared live-room weather scene.

The host may force a scene; otherwise Open-Meteo plus the local clock
pick sun / rain / night / rainbow / overcast. Failures cache as clock
so the settings page does not stall on a dead outbound HTTPS.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx

MODES = ("auto", "sun", "rain", "night", "rainbow")
SCENES = ("sun", "rain", "night", "rainbow", "overcast")
WEATHERS = ("clear", "rain", "overcast", "unknown")
SOURCES = ("open-meteo", "clock")
DEFAULT_MODE = "auto"
WEATHER_CACHE_SECONDS = 30 * 60
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_LAT = 30.67
DEFAULT_LON = 104.07
DEFAULT_TZ = "Asia/Shanghai"
_RAIN_CODES = frozenset(
    {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99}
)
_OLD_MOOD_TO_MODE = {
    "cats": "auto",
    "warm": "auto",
    "rain": "rain",
    "night": "night",
}


class RoomDecorError(Exception):
    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def parse_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode not in MODES:
        raise RoomDecorError("无效的天气场景", status=422)
    return mode


def weather_from_wmo(code: Any) -> str:
    try:
        value = int(code)
    except (TypeError, ValueError):
        return "unknown"
    if value in (0, 1):
        return "clear"
    if value in _RAIN_CODES:
        return "rain"
    return "overcast"


def _room_tz():
    name = os.environ.get("ROOM_DECOR_TZ", DEFAULT_TZ)
    try:
        return ZoneInfo(name)
    except Exception:
        return timezone(timedelta(hours=8))


def period_from_clock(now: datetime | None = None) -> str:
    tz = _room_tz()
    stamp = now or datetime.now(tz)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=tz)
    else:
        stamp = stamp.astimezone(tz)
    if stamp.hour < 6 or stamp.hour >= 20:
        return "night"
    return "day"


def after_rain_from_hourly(payload: Any, *, now: datetime | None = None) -> bool:
    hourly = payload.get("hourly") if isinstance(payload, dict) else None
    if not isinstance(hourly, dict):
        return False
    times = hourly.get("time") or []
    precips = hourly.get("precipitation") or []
    if not times or not precips:
        return False
    stamp = now or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    total = 0.0
    counted = 0
    for raw_time, raw_precip in zip(times, precips):
        try:
            when = datetime.fromisoformat(str(raw_time))
        except ValueError:
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=stamp.tzinfo)
        age_hours = (stamp - when).total_seconds() / 3600.0
        if 0 < age_hours <= 3:
            try:
                total += float(raw_precip or 0)
            except (TypeError, ValueError):
                continue
            counted += 1
    return counted > 0 and total >= 0.1


def resolve_scene(
    *,
    mode: str,
    weather: str,
    period: str,
    after_rain: bool = False,
) -> str:
    if mode in SCENES and mode != "auto":
        return mode
    if weather == "rain":
        return "rain"
    if period == "night":
        return "night"
    if weather == "clear" and after_rain:
        return "rainbow"
    if weather == "clear":
        return "sun"
    if weather == "unknown":
        return "night" if period == "night" else "sun"
    return "overcast"


def normalize_decor(payload: Any) -> dict[str, str]:
    data = payload if isinstance(payload, dict) else {}
    mode = data.get("mode")
    if mode not in MODES:
        mode = _OLD_MOOD_TO_MODE.get(str(data.get("mood") or ""), DEFAULT_MODE)
        if mode not in MODES:
            mode = DEFAULT_MODE
    weather = data.get("weather") if data.get("weather") in WEATHERS else "unknown"
    source = data.get("source") if data.get("source") in SOURCES else "clock"
    after_rain = bool(data.get("after_rain"))
    scene = data.get("scene") if data.get("scene") in SCENES else None
    if scene is None:
        scene = resolve_scene(
            mode=mode,
            weather=weather,
            period=period_from_clock(),
            after_rain=after_rain,
        )
    return {"mode": mode, "weather": weather, "scene": scene, "source": source}


def default_decor() -> dict[str, str]:
    return normalize_decor({"mode": DEFAULT_MODE, "weather": "unknown", "source": "clock"})


class RoomDecorStore:
    def __init__(self, path: str | Path | None = None) -> None:
        root = Path(__file__).resolve().parents[2]
        self.path = Path(path) if path else root / "data" / "room_decor.json"
        self._mode = DEFAULT_MODE
        self._weather = "unknown"
        self._source = "clock"
        self._after_rain = False
        self._fetched_at = 0.0
        self.load()

    def public(self) -> dict[str, str]:
        return normalize_decor(
            {
                "mode": self._mode,
                "weather": self._weather,
                "source": self._source,
                "after_rain": self._after_rain,
            }
        )

    def load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return
        if not isinstance(raw, dict):
            return
        try:
            if raw.get("mode") in MODES:
                self._mode = parse_mode(raw.get("mode"))
            elif raw.get("mood") in _OLD_MOOD_TO_MODE:
                self._mode = _OLD_MOOD_TO_MODE[str(raw.get("mood"))]
        except RoomDecorError:
            self._mode = DEFAULT_MODE

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"mode": self._mode}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def set_mode(self, mode: Any) -> dict[str, str]:
        self._mode = parse_mode(mode)
        self.save()
        return self.public()

    def _coords(self) -> tuple[float, float]:
        try:
            lat = float(os.environ.get("ROOM_DECOR_LAT", DEFAULT_LAT))
            lon = float(os.environ.get("ROOM_DECOR_LON", DEFAULT_LON))
        except (TypeError, ValueError):
            return DEFAULT_LAT, DEFAULT_LON
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return DEFAULT_LAT, DEFAULT_LON
        return lat, lon

    async def refresh_weather(self, *, force: bool = False) -> dict[str, str]:
        now = time.monotonic()
        age = now - self._fetched_at
        if not force and self._fetched_at and age < WEATHER_CACHE_SECONDS:
            return self.public()
        lat, lon = self._coords()
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                response = await client.get(
                    OPEN_METEO_URL,
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "current": "weather_code",
                        "hourly": "precipitation",
                        "past_hours": 3,
                        "forecast_hours": 1,
                        "timezone": "auto",
                    },
                )
            response.raise_for_status()
            payload = response.json()
            code = (payload.get("current") or {}).get("weather_code")
            weather = weather_from_wmo(code)
            if weather == "unknown":
                raise ValueError("missing weather code")
            self._weather = weather
            self._after_rain = weather != "rain" and after_rain_from_hourly(payload)
            self._source = "open-meteo"
            self._fetched_at = now
        except Exception:
            self._weather = "unknown"
            self._after_rain = False
            self._source = "clock"
            self._fetched_at = now
        return self.public()
