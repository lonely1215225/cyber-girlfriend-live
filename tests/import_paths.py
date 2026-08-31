from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"
SERVICE_DIRS = (
    ROOT / "apps" / "speech",
    ROOT / "services" / "tts",
    ROOT / "services" / "avatar",
    ROOT / "services" / "llm",
    WEB,
)


def add_service_paths() -> None:
    for path in SERVICE_DIRS:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
