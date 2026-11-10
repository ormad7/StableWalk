"""Display-only video fit/fill math — analysis frames are never cropped."""

from stablewalk.ui.video_display_fit import (
    contain_size,
    cover_size_and_crop,
    is_portrait_aspect,
)


def test_contain_preserves_aspect_and_stays_inside_box() -> None:
    w, h = contain_size(1080, 1920, 400, 600)
    assert w <= 400 and h <= 600
    assert abs((w / h) - (1080 / 1920)) < 0.02
    landscape = contain_size(1920, 1080, 400, 600)
    assert landscape[0] <= 400 and landscape[1] <= 600
    assert abs((landscape[0] / landscape[1]) - (1920 / 1080)) < 0.02


def test_fill_covers_box_and_only_crops_display() -> None:
    rw, rh, crop = cover_size_and_crop(1080, 1920, 400, 300)
    left, top, right, bottom = crop
    assert right - left == 400
    assert bottom - top == 300
    assert rw >= 400 and rh >= 300
    assert abs((rw / rh) - (1080 / 1920)) < 0.02


def test_portrait_gate() -> None:
    assert is_portrait_aspect(9 / 16)
    assert not is_portrait_aspect(16 / 9)
    assert not is_portrait_aspect(None)
