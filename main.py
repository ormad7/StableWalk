"""
StableWalk pipeline entry point.

Steps implemented:
  1. Stream/extract video frames → JPG
  2. Full-body pose + 12–14 DOF angles + velocities → JSON

Usage:
  python main.py                              # streams default walking video URL
  python main.py --url https://example.com/walk.mp4
  python main.py data/input/my_walk.mp4       # local file fallback
  python main.py --max-frames 30 --draw-overlays
  python main.py --view                         # matplotlib viewer after processing
  python main.py --gui                          # desktop GUI (Tkinter)
  python main.py --gui walk_stream              # GUI with specific pose output
  python gui.py                                 # GUI only (same as --gui)
  python main.py --view-only walk_stream        # matplotlib viewer only
  python main.py --simulate walk_stream         # robotic walk simulation
  python main.py --run-opensim-demo-ik          # run Gait2392 demo IK (requires OpenSim SDK)
  python main.py --export-motion-reference VIDEO_PATH  # Real-to-Sim motion .npz export
  python main.py --real-to-sim VIDEO_OR_RUN            # Full 4-stage Real-to-Sim pipeline
  python main.py --frames-only data/output/frames/walk_stream
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from stablewalk import config
from stablewalk.pose_estimation import PoseEstimator
from stablewalk.gait_analysis import GaitCycleAnalyzer
from stablewalk.pose_report import print_frame_report, print_sequence_summary
from stablewalk.video_processing import VideoProcessor, is_video_url
from stablewalk.video_validation import validate_video_source
from stablewalk.gui_app import launch_gui
from stablewalk.robot_simulation_viz import launch_robot_simulation
from stablewalk.visualization import launch_viewer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stablewalk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="StableWalk: video → frames → pose + joint angles",
    )
    parser.add_argument(
        "video",
        nargs="?",
        help="Local walking video path (optional if --url or default URL is used).",
    )
    parser.add_argument(
        "--url",
        nargs="?",
        const=config.DEFAULT_WALKING_VIDEO_URL,
        default=None,
        metavar="URL",
        help=(
            "Stream video from URL (no full-file download). "
            f"Use --url alone for default: {config.DEFAULT_WALKING_VIDEO_URL[:50]}..."
        ),
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip pre-flight full-body pose check on video source",
    )
    parser.add_argument(
        "--frames-only",
        metavar="DIR",
        help="Skip Step 1; run pose estimation on existing frame directory",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit frames processed (useful for quick tests)",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Subfolder name under data/output/frames (default: video stem)",
    )
    parser.add_argument(
        "--draw-overlays",
        action="store_true",
        help="Save skeleton overlay images for the first detected frame",
    )
    parser.add_argument(
        "--export-opensim",
        action="store_true",
        help="Export OpenSim-compatible files (.trc markers, .mot angles, JSON)",
    )
    parser.add_argument(
        "--list-videos",
        action="store_true",
        help="List videos in data/input/ and exit",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open interactive skeleton viewer after processing",
    )
    parser.add_argument(
        "--view-only",
        metavar="NAME",
        nargs="?",
        const="",
        help="Open matplotlib viewer only (uses data/output/poses/NAME_poses.json)",
    )
    parser.add_argument(
        "--gui",
        metavar="NAME",
        nargs="?",
        const="",
        help="Open desktop GUI (optional pose run name, e.g. walk_stream)",
    )
    parser.add_argument(
        "--simulate",
        metavar="NAME",
        nargs="?",
        const="",
        help="Open robotic walk simulator (DOF angles → mechanical joints)",
    )
    parser.add_argument(
        "--run-ik",
        action="store_true",
        help=(
            "Run StableWalk experimental IK on the mapped TRC "
            "(requires --opensim-model and exported walk_stream.trc)"
        ),
    )
    parser.add_argument(
        "--opensim-model",
        metavar="PATH",
        help="Path to an OpenSim .osim model (for --run-ik or validation)",
    )
    parser.add_argument(
        "--run-opensim-demo-ik",
        action="store_true",
        help=(
            "Run the official Gait2392 OpenSim Demo IK immediately "
            "(models/opensim/Gait2392_Pipeline/subject01_Setup_IK.xml)"
        ),
    )
    parser.add_argument(
        "--export-motion-reference",
        metavar="VIDEO_OR_RUN",
        nargs="?",
        const="",
        help=(
            "Export stablewalk_motion.npz for Real-to-Sim retargeting "
            "(video path, pose run name, or latest poses if omitted)"
        ),
    )
    parser.add_argument(
        "--real-to-sim",
        metavar="VIDEO_OR_RUN",
        nargs="?",
        const="",
        help=(
            "Run full Real-to-Sim pipeline: gait style, retargeting, "
            "AMP reference export, virtual GRF (video, run name, or latest)"
        ),
    )
    return parser.parse_args()


def run_step1(
    source: str,
    frames_dir: Path,
    max_frames: int | None,
    *,
    from_url: bool,
) -> tuple[float, list[str], str]:
    processor = VideoProcessor(jpeg_quality=config.DEFAULT_JPEG_QUALITY)
    if from_url:
        result = processor.extract_frames_from_url(
            source,
            frames_dir,
            max_frames=max_frames,
        )
    else:
        result = processor.extract_frames(
            source,
            frames_dir,
            max_frames=max_frames,
        )
    return result.fps, result.frame_paths, result.video_path


def run_step2(
    frames_dir: Path,
    poses_path: Path,
    source_video: str,
    fps: float,
    max_frames: int | None,
    draw_overlays: bool,
) -> None:
    with PoseEstimator(
        model_variant=config.DEFAULT_POSE_MODEL_VARIANT,
        min_detection_confidence=config.DEFAULT_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.DEFAULT_MIN_TRACKING_CONFIDENCE,
        video_mode=not config.DEFAULT_POSE_IMAGE_MODE,
    ) as estimator:
        sequence = estimator.process_directory(
            frames_dir,
            source_video=source_video,
            fps=fps,
            max_frames=max_frames,
        )
        estimator.save_sequence(sequence, poses_path)

        overlay_dir = frames_dir.parent / "overlays" / frames_dir.name
        overlay_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for pf in sequence.frames:
            if pf.detected and pf.keypoints:
                out = overlay_dir / f"overlay_{pf.frame_index:06d}.jpg"
                if PoseEstimator.draw_keypoints_overlay(pf.image_path, pf.keypoints, out):
                    saved += 1
        if saved:
            logger.info("Saved %d skeleton overlay images → %s", saved, overlay_dir)

        detected_frames = [f for f in sequence.frames if f.detected]
        print_sequence_summary(sequence.frames)
        if detected_frames:
            logger.info("%s", GaitCycleAnalyzer().event_summary(sequence))
        if not detected_frames:
            logger.warning(
                "No valid full-body gait poses in this video.\n"
                "  Try: python main.py --url\n"
                "  Or pass a local clip: python main.py data/input/my_walk.mp4",
            )
        else:
            print_frame_report(detected_frames[0])
            if len(detected_frames) > 1:
                print_frame_report(detected_frames[len(detected_frames) // 2])


def _auto_export_opensim_after_analysis(poses_path: Path, run_name: str) -> None:
    """Export TRC/MOT/JSON after MediaPipe and log honest OpenSim status."""
    from stablewalk.opensim_integration import export_from_pose_json
    from stablewalk.opensim_sdk import log_post_analysis_opensim_status

    try:
        written = export_from_pose_json(
            poses_path, config.OPENSIM_DIR / run_name, name=run_name
        )
        log_post_analysis_opensim_status(written, logger)
        from stablewalk.analysis.opensim_id_readiness import assess_opensim_id_readiness

        id_report = assess_opensim_id_readiness(run_name)
        if not id_report.ready_for_traditional_id:
            logger.info(
                "OpenSim Inverse Dynamics not ready for %s (missing measured external loads). "
                "See docs/VIRTUAL_GRF.md for vGRF research architecture.",
                run_name,
            )
    except (OSError, ValueError) as exc:
        logger.error("OpenSim export failed: %s", exc)
        log_post_analysis_opensim_status(None, logger)


def _run_opensim_demo_ik_cli() -> int:
    """Run Gait2392 demo IK from the CLI and print the output ``*ik*.mot`` path."""
    from stablewalk.opensim_sdk import (
        check_opensim_sdk,
        run_opensim_demo_ik,
        update_opensim_status_md,
    )

    status = check_opensim_sdk(refresh=True)
    if not status.available:
        logger.error("%s", status.message)
        update_opensim_status_md(demo_ik_result=None)
        return 1

    result = run_opensim_demo_ik()
    update_opensim_status_md(demo_ik_result=result)

    if result.ran and result.output_motion_path and Path(result.output_motion_path).is_file():
        logger.info("OpenSim Demo IK completed successfully")
        logger.info("Demo IK output file: %s", result.output_motion_path)
        print(result.output_motion_path)
        return 0

    logger.error("OpenSim Demo IK failed: %s", result.message)
    return 1


def _export_motion_reference_cli(
    target: str | None,
    *,
    max_frames: int | None = None,
    output_name: str | None = None,
) -> int:
    """
    Export ``stablewalk_motion.npz`` for Isaac Lab / imitation-learning pipelines.

    Accepts a video path, pose run name, or uses the latest pose JSON when omitted.
    """
    from stablewalk.io.motion_reference_export import export_motion_reference_from_poses

    if target:
        # Pose run name
        poses_candidate = config.POSES_DIR / f"{target}_poses.json"
        if poses_candidate.is_file():
            run_name = output_name or target
            result = export_motion_reference_from_poses(
                poses_candidate,
                config.MOTION_REFERENCE_EXPORT_DIR,
                run_name=run_name,
            )
            logger.info("Motion reference exported → %s", result.npz_path)
            print(result.npz_path)
            return 0

        # Video path
        try:
            video_path = config.resolve_video_path(target)
        except FileNotFoundError:
            video_path = Path(target)
        if video_path.is_file():
            run_name = output_name or video_path.stem
            frames_dir = config.FRAMES_DIR / run_name
            poses_path = config.POSES_DIR / f"{run_name}_poses.json"
            if not poses_path.is_file():
                logger.info("Running pipeline for motion reference export…")
                fps, _, source_video = run_step1(
                    str(video_path),
                    frames_dir,
                    max_frames,
                    from_url=False,
                )
                run_step2(
                    frames_dir,
                    poses_path,
                    source_video,
                    fps,
                    max_frames,
                    draw_overlays=False,
                )
            result = export_motion_reference_from_poses(
                poses_path,
                config.MOTION_REFERENCE_EXPORT_DIR,
                run_name=run_name,
            )
            logger.info("Motion reference exported → %s", result.npz_path)
            print(result.npz_path)
            return 0

        logger.error(
            "Could not resolve --export-motion-reference target: %s "
            "(expected video path or pose run name)",
            target,
        )
        return 1

    candidates = sorted(
        config.POSES_DIR.glob("*_poses.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        logger.error("No pose JSON found. Run: python main.py --url")
        return 1
    poses_path = candidates[0]
    run_name = output_name or poses_path.stem.replace("_poses", "")
    result = export_motion_reference_from_poses(
        poses_path,
        config.MOTION_REFERENCE_EXPORT_DIR,
        run_name=run_name,
    )
    logger.info("Motion reference exported → %s", result.npz_path)
    print(result.npz_path)
    return 0


def _run_real_to_sim_cli(
    target: str | None,
    *,
    max_frames: int | None = None,
    output_name: str | None = None,
) -> int:
    """Run the 4-stage Real-to-Sim pipeline offline."""
    from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
    from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
    from stablewalk.io.pose_loader import load_pose_sequence
    from stablewalk.real_to_sim.pipeline import run_real_to_sim_pipeline

    poses_path: Path | None = None
    run_name: str | None = output_name

    if target:
        poses_candidate = config.POSES_DIR / f"{target}_poses.json"
        if poses_candidate.is_file():
            poses_path = poses_candidate
            run_name = run_name or target
        else:
            try:
                video_path = config.resolve_video_path(target)
            except FileNotFoundError:
                video_path = Path(target)
            if video_path.is_file():
                run_name = run_name or video_path.stem
                frames_dir = config.FRAMES_DIR / run_name
                poses_path = config.POSES_DIR / f"{run_name}_poses.json"
                if not poses_path.is_file():
                    logger.info("Running video pipeline for Real-to-Sim…")
                    fps, _, source_video = run_step1(
                        str(video_path),
                        frames_dir,
                        max_frames,
                        from_url=False,
                    )
                    run_step2(
                        frames_dir,
                        poses_path,
                        source_video,
                        fps,
                        max_frames,
                        draw_overlays=False,
                    )
            else:
                logger.error("Could not resolve --real-to-sim target: %s", target)
                return 1
    else:
        candidates = sorted(
            config.POSES_DIR.glob("*_poses.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            logger.error("No pose JSON found. Run: python main.py --url")
            return 1
        poses_path = candidates[0]
        run_name = run_name or poses_path.stem.replace("_poses", "")

    assert poses_path is not None and run_name is not None
    sequence = load_pose_sequence(poses_path)
    recording = pose_sequence_to_gait_motion(sequence)
    cycles = analyze_gait_cycles(recording)
    report = run_real_to_sim_pipeline(
        recording,
        config.MOTION_REFERENCE_EXPORT_DIR,
        run_name=run_name,
        sequence=sequence,
        cycles=cycles,
    )
    logger.info("Real-to-Sim pipeline report → %s", report.report_path)
    for stage in report.stages:
        logger.info("  %s: %s — %s", stage.stage, stage.status, stage.detail)
    if report.gait_style:
        logger.info("Gait style: %s", report.gait_style.style_summary)
    print(report.report_path)
    return 0


def _run_opensim_ik_cli(
    *,
    run_name: str,
    model_path: str | None,
    run_ik: bool,
) -> int:
    """Run StableWalk experimental IK using the mapped TRC (not raw MediaPipe names)."""
    from stablewalk.opensim_sdk import (
        NO_MODEL_LOADED_MESSAGE,
        check_opensim_sdk,
        run_stablewalk_ik_experimental,
        update_opensim_status_md,
    )

    if not run_ik:
        if model_path:
            from stablewalk.opensim_sdk import load_opensim_model

            resolved_model = Path(model_path)
            if resolved_model.is_file():
                info = load_opensim_model(resolved_model)
                if info.valid:
                    logger.info("Model loaded: %s", info.name or resolved_model.name)
                else:
                    logger.error("Model load failed: %s", info.message)
                    return 1
        return 0

    status = check_opensim_sdk(refresh=True)
    if not status.available:
        logger.error("%s", status.message)
        return 1

    trc_path = config.OPENSIM_DIR / run_name / f"{run_name}.trc"
    if not trc_path.is_file():
        logger.error("StableWalk TRC not found for IK: %s", trc_path)
        logger.error("Run the pipeline first: python main.py")
        return 1

    if not model_path:
        logger.error(NO_MODEL_LOADED_MESSAGE)
        logger.error(
            "Pass --opensim-model PATH, e.g. "
            "models/opensim/Gait2392_Pipeline/subject01_simbody.osim"
        )
        return 1

    result = run_stablewalk_ik_experimental(
        trc_path, model_path=model_path, run_name=run_name
    )
    update_opensim_status_md()

    if result.ran and result.output_motion_path and Path(result.output_motion_path).is_file():
        print(result.output_motion_path)
        return 0

    logger.error("StableWalk IK failed: %s", result.message)
    return 1


def ensure_sample_alias() -> None:
    """Create my_walk.mp4 from sample_walk.mp4 if the user has not added their own."""
    my_walk = config.INPUT_DIR / "my_walk.mp4"
    sample = config.INPUT_DIR / "sample_walk.mp4"
    if my_walk.is_file() or not sample.is_file():
        return
    shutil.copy2(sample, my_walk)
    logger.info("Created %s from sample_walk.mp4 (replace with your own video anytime)", my_walk.name)


def _resolve_poses_path(name: str) -> Path:
    """Resolve pose JSON from run name or stem."""
    stem = name.strip()
    if not stem:
        stem = "walk_stream"
    path = config.POSES_DIR / f"{stem}_poses.json"
    if path.is_file():
        return path
    # Allow passing full path
    direct = Path(name)
    if direct.is_file():
        return direct
    raise FileNotFoundError(
        f"Pose data not found: {path}\n"
        f"Run the pipeline first: python main.py"
    )


def main() -> int:
    args = parse_args()
    config.ensure_output_dirs()

    from stablewalk.opensim_sdk import log_opensim_startup_status

    log_opensim_startup_status(logger)

    if args.run_opensim_demo_ik:
        return _run_opensim_demo_ik_cli()

    if args.export_motion_reference is not None:
        target = args.export_motion_reference if args.export_motion_reference else None
        return _export_motion_reference_cli(
            target,
            max_frames=args.max_frames,
            output_name=args.output_name,
        )

    if args.real_to_sim is not None:
        target = args.real_to_sim if args.real_to_sim else None
        return _run_real_to_sim_cli(
            target,
            max_frames=args.max_frames,
            output_name=args.output_name,
        )

    if args.run_ik:
        run_name = args.output_name or "walk_stream"
        return _run_opensim_ik_cli(
            run_name=run_name,
            model_path=args.opensim_model,
            run_ik=True,
        )

    if args.simulate is not None:
        name = args.simulate if args.simulate else None
        try:
            path = _resolve_poses_path(name) if name else None
        except FileNotFoundError:
            path = None
        if path is None:
            candidates = sorted(
                config.POSES_DIR.glob("*_poses.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                logger.error("No pose JSON found. Run: python main.py --url")
                return 1
            path = candidates[0]
        try:
            launch_robot_simulation(path)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("%s", exc)
            return 1
        return 0

    if args.gui is not None:
        name = args.gui if args.gui else None
        try:
            path = _resolve_poses_path(name) if name else None
        except FileNotFoundError:
            path = None
        launch_gui(path)
        return 0

    if args.view_only is not None:
        try:
            launch_viewer(_resolve_poses_path(args.view_only))
        except (FileNotFoundError, ValueError) as exc:
            logger.error("%s", exc)
            return 1
        return 0

    if args.list_videos:
        videos = config.list_input_videos()
        if not videos:
            print(f"No videos in {config.INPUT_DIR}")
            print(f"Add a file such as: {config.INPUT_DIR / 'my_walk.mp4'}")
        else:
            print(f"Videos in {config.INPUT_DIR}:")
            for v in videos:
                print(f"  {v.name}")
        return 0

    if args.frames_only:
        frames_dir = Path(args.frames_only)
        if not frames_dir.is_dir():
            logger.error("Frames directory not found: %s", frames_dir)
            return 1
        source_video = ""
        fps = 30.0
        run_name = frames_dir.name
    else:
        if args.url is not None:
            video_source = args.url
            from_url = True
            run_name = args.output_name or "walk_stream"
            logger.info("Using video URL: %s", video_source)
        elif args.video and is_video_url(args.video):
            video_source = args.video
            from_url = True
            run_name = args.output_name or "walk_stream"
            logger.info("Using video URL: %s", video_source)
        elif args.video:
            try:
                video_path = config.resolve_video_path(args.video)
            except FileNotFoundError as exc:
                logger.error("%s", exc)
                return 1
            video_source = str(video_path)
            from_url = False
            run_name = args.output_name or video_path.stem
            logger.info("Using local video: %s", video_path)
        else:
            video_source = config.DEFAULT_WALKING_VIDEO_URL
            from_url = True
            run_name = args.output_name or "walk_stream"
            logger.info("Using default walking video URL: %s", video_source)

        if from_url and not args.no_validate:
            passed, ratio, msg = validate_video_source(
                video_source,
                sample_count=config.DEFAULT_VIDEO_VALIDATION_SAMPLES,
                min_valid_ratio=config.DEFAULT_MIN_VALID_FRAME_RATIO,
                model_variant=config.DEFAULT_POSE_MODEL_VARIANT,
            )
            if not passed:
                logger.error(msg)
                return 1
            logger.info(msg)

        frames_dir = config.FRAMES_DIR / run_name
        logger.info("Step 1: extracting frames → %s", frames_dir)
        fps, _, source_video = run_step1(
            video_source,
            frames_dir,
            args.max_frames,
            from_url=from_url,
        )

    poses_path = config.POSES_DIR / f"{run_name}_poses.json"
    logger.info("Step 2: pose estimation → %s", poses_path)
    run_step2(
        frames_dir,
        poses_path,
        source_video,
        fps,
        args.max_frames,
        args.draw_overlays,
    )

    logger.info("Done. Frames: %s | Poses: %s", frames_dir, poses_path)

    # Always prepare OpenSim-compatible files after MediaPipe analysis.
    _auto_export_opensim_after_analysis(poses_path, run_name)

    if args.run_ik or args.opensim_model:
        ik_code = _run_opensim_ik_cli(
            run_name=run_name,
            model_path=args.opensim_model,
            run_ik=bool(args.run_ik),
        )
        if args.run_ik and ik_code != 0:
            return ik_code

    if args.export_opensim:
        logger.info(
            "Note: --export-opensim is now the default after analysis; "
            "files were already exported above."
        )

    if args.view:
        logger.info("Opening matplotlib viewer...")
        launch_viewer(poses_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
