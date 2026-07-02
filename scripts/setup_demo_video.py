"""Build data/input/my_walk.mp4 from a full-body walking reference image."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "input"
OUTPUT = INPUT / "my_walk.mp4"

# Full-body side-view walker (Pixabay, free license)
IMAGE_URL = (
    "https://cdn.pixabay.com/photo/2017/08/06/12/52/woman-2592247_1280.jpg"
)


def download_image(path: Path) -> None:
    req = urllib.request.Request(IMAGE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        path.write_bytes(response.read())


def image_to_video(image: np.ndarray, out_path: Path, frames: int = 90, fps: int = 15) -> None:
    h, w = image.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(frames):
        scale = 1.0 + 0.0015 * i
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
        frame = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        writer.write(frame)
    writer.release()


def main() -> int:
    INPUT.mkdir(parents=True, exist_ok=True)
    image_path = INPUT / "walk_body.jpg"
    if not image_path.is_file():
        print("Downloading reference walking image...")
        download_image(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        print("Failed to load image")
        return 1
    image_to_video(image, OUTPUT, frames=90, fps=15)
    print(f"Created {OUTPUT} (90 frames @ 15 fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
