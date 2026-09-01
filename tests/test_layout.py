from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LayoutTests(unittest.TestCase):
    def test_runtime_entrypoints_exist(self):
        required = (
            "apps/web/server.py",
            "apps/web/room_manager.py",
            "apps/speech/s2s_with_avatar_tee.py",
            "services/avatar/avtr1_gateway.py",
            "services/tts/emotion_aware_tts.py",
            "services/tts/voxcpm_shared_client.py",
            "services/llm/ollama_thinkless.py",
            "deploy/nginx/nginx.conf.tpl",
            "deploy/mediamtx/mediamtx.yml.tpl",
            "deploy/docker/Dockerfile",
            "deploy/docker/entrypoint.sh",
            "docker-compose.yml",
            "scripts/start.sh",
            "scripts/stop.sh",
            "scripts/docker-up.sh",
            "install.sh",
            "assets/expressions/xiaoya_locket/reference-laugh.png",
            "apps/web/room_decor.py",
            "apps/web/room-decor.js",
        )
        missing = [rel for rel in required if not (ROOT / rel).is_file()]
        self.assertEqual(missing, [])

    def test_old_space_path_is_gone(self):
        self.assertFalse((ROOT / "s2s" / "hf-realtime-voice" / "server.py").exists())
        self.assertFalse((ROOT / "proxy" / "avtr1_gateway.py").exists())
        self.assertFalse((ROOT / "proxy" / "nginx.conf.tpl").exists())
