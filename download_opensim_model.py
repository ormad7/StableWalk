#!/usr/bin/env python3
"""
Download official OpenSim sample models into models/opensim/.

Source repository (official):
  https://github.com/opensim-org/opensim-models

Usage:
  python download_opensim_model.py                  # Gait2392_Simbody (default)
  python download_opensim_model.py --model gait2354
  python download_opensim_model.py --model gait2392
  python download_opensim_model.py --list           # show discovered local models

If automatic download fails (network/firewall), the script prints manual steps.
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path

from stablewalk import config
from stablewalk.opensim_models import (
    OFFICIAL_MODEL_BUNDLES,
    OPENSIM_MODELS_RAW,
    OPENSIM_MODELS_REPO,
    discover_opensim_models,
    ensure_models_dir,
    find_preferred_opensim_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("download_opensim_model")


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", dest.name)
    logger.info("  -> %s", dest)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "StableWalk/1.0 (OpenSim model downloader)"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    dest.write_bytes(data)
    logger.info("  Saved %d bytes", len(data))


def download_official_bundle(model_key: str) -> list[Path]:
    """
    Download an official model bundle from opensim-org/opensim-models.

    Returns list of written ``.osim`` paths.
    """
    key = model_key.lower()
    if key not in OFFICIAL_MODEL_BUNDLES:
        raise ValueError(
            f"Unknown model {model_key!r}. Choose: {', '.join(OFFICIAL_MODEL_BUNDLES)}"
        )

    bundle = OFFICIAL_MODEL_BUNDLES[key]
    folder = str(bundle["folder"])
    files = list(bundle["files"])  # type: ignore[arg-type]
    note = str(bundle.get("note", ""))

    dest_root = ensure_models_dir() / folder
    dest_root.mkdir(parents=True, exist_ok=True)

    logger.info("Official OpenSim models: %s", OPENSIM_MODELS_REPO)
    logger.info("Target folder: %s", dest_root)
    if note:
        logger.info("Note: %s", note)

    written: list[Path] = []
    errors: list[str] = []

    for filename in files:
        url = f"{OPENSIM_MODELS_RAW}/Models/{folder}/{filename}"
        dest = dest_root / filename
        if dest.is_file() and dest.stat().st_size > 0:
            logger.info("Already present: %s", dest)
            written.append(dest.resolve())
            continue
        try:
            _download_file(url, dest)
            written.append(dest.resolve())
        except (OSError, urllib.error.URLError) as exc:
            errors.append(f"{filename}: {exc}")

    if errors:
        print_manual_instructions(model_key)
        raise RuntimeError(
            "Some downloads failed:\n  " + "\n  ".join(errors)
        )

    preferred = find_preferred_opensim_model()
    if preferred:
        logger.info("Usable model with markers (manual selection): %s", preferred)
    else:
        logger.info(
            "No auto-loaded model — use OpenSim panel to select a model, "
            "or run OpenSim Demo IK (Gait2392_Pipeline)."
        )
    return written


def print_manual_instructions(model_key: str = "gait2392") -> None:
    """Print steps to obtain models manually from the official repository."""
    key = model_key.lower()
    bundle = OFFICIAL_MODEL_BUNDLES.get(key, OFFICIAL_MODEL_BUNDLES["gait2392"])
    folder = bundle["folder"]
    files = bundle["files"]

    print(
        "\n"
        "=== Manual OpenSim model setup ===\n"
        f"1. Open the official repository:\n"
        f"   {OPENSIM_MODELS_REPO}\n"
        "\n"
        "2. Download the repository (Code -> Download ZIP) or clone with git:\n"
        f"   git clone {OPENSIM_MODELS_REPO}.git\n"
        "\n"
        f"3. Copy the folder Models/{folder} into:\n"
        f"   {config.OPENSIM_MODELS_DIR / folder}\n"
        "\n"
        "4. Verify at least one of these files exists:\n"
    )
    for name in files:
        print(f"   {config.OPENSIM_MODELS_DIR / folder / name}")
    print(
        "\n"
        "5. For Gait2392: the official repo provides gait2392_millard2012muscle.osim\n"
        "   (there is no gait2392_simbody.osim in opensim-models).\n"
        "   For Gait2354: use gait2354_simbody.osim\n"
        "\n"
        "6. Restart StableWalk or click 'Select Local Model' in the OpenSim panel.\n"
        "   Run OpenSim IK only after the model loads successfully in the GUI.\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download official OpenSim gait models into models/opensim/",
    )
    parser.add_argument(
        "--model",
        choices=tuple(OFFICIAL_MODEL_BUNDLES),
        default="gait2392",
        help="Model bundle to download (default: gait2392)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List .osim files already under models/opensim/ and exit",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Print manual download instructions and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_models_dir()

    if args.manual:
        print_manual_instructions(args.model)
        return 0

    if args.list:
        models = discover_opensim_models()
        if not models:
            logger.info("No .osim files under %s", config.OPENSIM_MODELS_DIR)
            logger.info("Run: python download_opensim_model.py")
            return 0
        logger.info("Discovered %d model(s) under %s:", len(models), config.OPENSIM_MODELS_DIR)
        for path in models:
            logger.info("  %s", path)
        preferred = find_preferred_opensim_model()
        if preferred:
            logger.info("Preferred default: %s", preferred)
        return 0

    try:
        written = download_official_bundle(args.model)
    except (RuntimeError, ValueError, OSError, urllib.error.URLError) as exc:
        logger.error("%s", exc)
        print_manual_instructions(args.model)
        return 1

    logger.info("Download complete — %d file(s)", len(written))
    for path in written:
        logger.info("  %s", path)

    preferred = find_preferred_opensim_model()
    if preferred:
        logger.info("Usable model with markers (manual selection): %s", preferred)
    else:
        logger.info(
            "StableWalk does not auto-load models. For IK demo, ensure "
            "models/opensim/Gait2392_Pipeline/ is present."
        )

    logger.info(
        "Next: open StableWalk GUI -> OpenSim panel -> Select Local Model, "
        "or Load .osim Model for another file."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
