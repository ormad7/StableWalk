"""Display-only video sizing. Never crops or resamples analysis input."""

from __future__ import annotations

PORTRAIT_ASPECT = 0.92


def is_portrait_aspect(aspect: float | None) -> bool:
    if aspect is None:
        return False
    try:
        return float(aspect) > 0.0 and float(aspect) < PORTRAIT_ASPECT
    except (TypeError, ValueError):
        return False


def contain_size(
    src_w: int, src_h: int, box_w: int, box_h: int
) -> tuple[int, int]:
    """Largest size that fits in the box without stretching or cropping."""
    if src_w <= 0 or src_h <= 0 or box_w <= 0 or box_h <= 0:
        return 1, 1
    scale = min(box_w / src_w, box_h / src_h)
    width = max(1, int(round(src_w * scale)))
    height = max(1, int(round(src_h * scale)))
    if width > box_w or height > box_h:
        scale *= 0.999
        width = max(1, int(round(src_w * scale)))
        height = max(1, int(round(src_h * scale)))
    return width, height


def cover_size_and_crop(
    src_w: int, src_h: int, box_w: int, box_h: int
) -> tuple[int, int, tuple[int, int, int, int]]:
    """Fill the box; return resize size and a centered display crop.

    The crop applies only to the shown image. Callers must not use it on
    pose-estimation frames.
    """
    if src_w <= 0 or src_h <= 0 or box_w <= 0 or box_h <= 0:
        return 1, 1, (0, 0, 1, 1)
    scale = max(box_w / src_w, box_h / src_h)
    width = max(1, int(round(src_w * scale)))
    height = max(1, int(round(src_h * scale)))
    left = max(0, (width - box_w) // 2)
    top = max(0, (height - box_h) // 2)
    right = min(width, left + max(1, box_w))
    bottom = min(height, top + max(1, box_h))
    if right <= left:
        right = left + 1
    if bottom <= top:
        bottom = top + 1
    return width, height, (left, top, right, bottom)


def click_to_normalized_xy(
    event_x: float,
    event_y: float,
    *,
    label_w: int,
    label_h: int,
    src_w: int,
    src_h: int,
    zoom_mode: str = "Fit",
    crop_box: tuple[int, int, int, int] | None = None,
    fitted_w: int | None = None,
    fitted_h: int | None = None,
) -> tuple[float, float] | None:
    """Map a click on the letterboxed video label into source-normalized xy."""
    if label_w <= 1 or label_h <= 1 or src_w <= 0 or src_h <= 0:
        return None
    mode = (zoom_mode or "Fit").strip()
    if mode == "Fill":
        if crop_box is None or fitted_w is None or fitted_h is None:
            fitted_w, fitted_h, crop_box = cover_size_and_crop(
                src_w, src_h, label_w, label_h
            )
        left, top, right, bottom = crop_box
        displayed_w = max(1, right - left)
        displayed_h = max(1, bottom - top)
        ox = (label_w - displayed_w) // 2
        oy = (label_h - displayed_h) // 2
        px = float(event_x) - ox + left
        py = float(event_y) - oy + top
        nx = px / max(int(fitted_w), 1)
        ny = py / max(int(fitted_h), 1)
    else:
        new_w, new_h = contain_size(src_w, src_h, label_w, label_h)
        if fitted_w is not None and fitted_h is not None:
            new_w, new_h = int(fitted_w), int(fitted_h)
        ox = (label_w - new_w) // 2
        oy = (label_h - new_h) // 2
        px = float(event_x) - ox
        py = float(event_y) - oy
        if px < 0 or py < 0 or px >= new_w or py >= new_h:
            return None
        nx = px / max(new_w, 1)
        ny = py / max(new_h, 1)
    if nx < 0.0 or ny < 0.0 or nx > 1.0 or ny > 1.0:
        return None
    return (min(1.0, max(0.0, nx)), min(1.0, max(0.0, ny)))


__all__ = [
    "PORTRAIT_ASPECT",
    "click_to_normalized_xy",
    "contain_size",
    "cover_size_and_crop",
    "is_portrait_aspect",
]
