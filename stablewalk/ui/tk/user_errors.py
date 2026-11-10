"""Short user-facing error text — never show raw stack traces in dialogs."""

from __future__ import annotations


def format_user_error(
    exc: BaseException | str | None,
    *,
    fallback: str = "Something went wrong. See the log for details.",
    max_len: int = 240,
) -> str:
    """Return a single-line message suitable for messagebox / status labels."""
    if exc is None:
        return fallback
    text = str(exc).strip() if not isinstance(exc, str) else exc.strip()
    if not text:
        return fallback
    # Prefer the first non-traceback line.
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("traceback") or low.startswith('file "'):
            continue
        if len(line) > max_len:
            return line[: max_len - 3] + "..."
        return line
    return fallback


__all__ = ["format_user_error"]
