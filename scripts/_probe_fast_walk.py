from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

src = Path(r"data/demo_videos/_fast_walk_candidates/extract/walk_54_side_anonymized.avi")
cap = cv2.VideoCapture(str(src))
print("opened", cap.isOpened())
print("fps", cap.get(cv2.CAP_PROP_FPS))
print("frames", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("w", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "h", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("fourcc", int(cap.get(cv2.CAP_PROP_FOURCC)))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
print("duration", (n / fps) if fps else None)
samples = sorted({0, n // 4, n // 2, (3 * n) // 4, max(0, n - 1)}) if n else []
ok = fail = 0
tiles = []
for idx in samples:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    good, frame = cap.read()
    if good and frame is not None:
        ok += 1
        h, w = frame.shape[:2]
        scale = 240 / max(h, 1)
        tile = cv2.resize(frame, (int(w * scale), int(h * scale)))
        cv2.putText(tile, f"f{idx}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(tile)
    else:
        fail += 1
cap.release()
print("decoded_ok", ok, "decoded_fail", fail)
print("sha256", sha256(src.read_bytes()).hexdigest())
if tiles:
    height = max(t.shape[0] for t in tiles)
    width = sum(t.shape[1] for t in tiles)
    sheet = np.zeros((height, width, 3), dtype=np.uint8)
    x = 0
    for tile in tiles:
        sheet[: tile.shape[0], x : x + tile.shape[1]] = tile
        x += tile.shape[1]
    dest = Path(r"data/demo_videos/_fast_walk_candidates/walk_54_side_contact_sheet.png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), sheet)
    print("contact_sheet", dest)
