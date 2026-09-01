#!/usr/bin/env python3
"""Lock newly generated expression portraits onto the August locket plates.

Playback is unchanged: each output is a full 941x1672 still, copied to
xiaoya_locket_expr_* and used for both exp-retarget and source-swap.

Generation cannot keep the original camera. This script detects both faces,
similarity-warps the new portrait onto an identity-matched August plate, and
keeps the plate's body, hand and hair so the live crop does not jump.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXPR_DIR = ROOT / "assets" / "expressions" / "xiaoya_locket"
CANDIDATE_DIR = ROOT / "assets" / "expressions" / "candidates" / "stills"

PLATE_SIZE = (941, 1672)  # width, height
CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

JOBS = (
    ("soft-smile", "orig-soft-smile.png", "reference-smirk.png"),
    ("curious", "orig-curious.png", "reference-one-brow.png"),
    ("side-eye", "orig-side-eye.png", "reference-shy.png"),
    ("lip-bite", "orig-lip-bite.png", "reference-pout.png"),
    ("sleepy", "orig-sleepy.png", "reference-one-brow.png"),
    ("tender", "orig-tender.png", "reference-shy.png"),
)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def detect_face(image: np.ndarray) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = CASCADE.detectMultiScale(gray, 1.08, 4, minSize=(80, 80))
    if len(faces) == 0:
        raise SystemExit("no face detected")
    x, y, w, h = max(faces, key=lambda item: int(item[2]) * int(item[3]))
    return int(x), int(y), int(w), int(h)


def lock_to_plate(generated: np.ndarray, plate: np.ndarray) -> np.ndarray:
    gx, gy, gw, gh = detect_face(generated)
    px, py, pw, ph = detect_face(plate)
    scale = 0.5 * (pw / gw + ph / gh)
    src_cx, src_cy = gx + gw / 2.0, gy + gh / 2.0
    dst_cx, dst_cy = px + pw / 2.0, py + ph / 2.0
    matrix = np.array(
        [
            [scale, 0.0, dst_cx - scale * src_cx],
            [0.0, scale, dst_cy - scale * src_cy],
        ],
        dtype=np.float32,
    )
    height, width = plate.shape[:2]
    warped = cv2.warpAffine(
        generated,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT,
    )
    yy, xx = np.mgrid[0:height, 0:width]
    radius = np.sqrt(
        ((xx - dst_cx) / max(1.0, pw * 0.36)) ** 2
        + ((yy - dst_cy) / max(1.0, ph * 0.40)) ** 2
    )
    alpha = np.clip((1.0 - radius) / 0.34, 0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    # Keep the plate hand: it sits on the viewer's right, lower face.
    hand = np.clip((xx - (px + pw * 0.52)) / max(1.0, pw * 0.22), 0.0, 1.0)
    hand *= np.clip((yy - (py + ph * 0.42)) / max(1.0, ph * 0.22), 0.0, 1.0)
    alpha = alpha * (1.0 - 0.88 * hand)
    locked = plate.astype(np.float32) * (1.0 - alpha[..., None]) + warped.astype(
        np.float32
    ) * alpha[..., None]
    return np.clip(locked, 0, 255).astype(np.uint8)


def body_mse(a: np.ndarray, b: np.ndarray) -> float:
    h, w = a.shape[:2]
    body = (
        a[int(h * 0.48) : int(h * 0.98), int(w * 0.10) : int(w * 0.90)].astype(np.float32)
        - b[int(h * 0.48) : int(h * 0.98), int(w * 0.10) : int(w * 0.90)].astype(np.float32)
    )
    return float((body * body).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("/root/.cursor/projects/root-cyber-girlfriend/assets"),
    )
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--preview-dir", type=Path, default=Path("/tmp/locket-locked"))
    args = parser.parse_args()
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    laugh = load_rgb(EXPR_DIR / "reference-laugh.png")
    print(f"{'name':12s} {'body_mse':>8s}  plate")
    for name, generated_name, plate_name in JOBS:
        generated = load_rgb(args.src_dir / generated_name)
        plate = load_rgb(EXPR_DIR / plate_name)
        if plate.shape[1] != PLATE_SIZE[0] or plate.shape[0] != PLATE_SIZE[1]:
            raise SystemExit(f"{plate_name} is not 941x1672")
        locked = lock_to_plate(generated, plate)
        mse = body_mse(locked, laugh)
        Image.fromarray(locked).save(args.preview_dir / f"{name}.png")
        Image.fromarray(locked[280:1040, 200:760]).resize((320, 430)).save(
            args.preview_dir / f"face-{name}.jpg", quality=90
        )
        print(f"{name:12s} {mse:8.1f}  {plate_name}")
        if args.write:
            Image.fromarray(locked).save(EXPR_DIR / f"reference-{name}.png")
            if CANDIDATE_DIR.is_dir():
                Image.fromarray(locked).save(CANDIDATE_DIR / f"{name}.png")


if __name__ == "__main__":
    main()
