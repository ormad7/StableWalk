# Research gait-video validation and comparison

StableWalk can import **local** walking videos into `data/research_dataset/`,
run pose and analysis **separately** for Video A and Video B, and export a
comparison report.

The Compare tab **Research dataset** page is the workspace. The existing
Laboratory Compare (session cache) remains on the second inner page and is
built only when opened.

## Import rules

- Only local files (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.m4v`).
- Remote URLs are stored as citation metadata if you type them; they are
  **never downloaded**.
- Clinical diagnosis (including dementia) is shown only when
  `clinical_verified` is true **and** `clinical_source` is non-empty.

## Categories

Normal, hemiplegic, limping, ataxic, Parkinsonian, shuffling, other abnormal,
unknown/unverified, and optional `dementia_research` (organizational tag, not
a diagnosis).

## Independence

Each Analyze A / Analyze B call owns its `PoseSequence` and
`GaitMotionRecording`. Gait-phase playback maps 0–100% through **that clip’s**
cycles. The live Overview session is not reused as both panels.

## Comparison

Δ = B − A on cadence, stride time, knee ROM, ASI, dominant frequency, ACF
stride lag, and phase-aligned knee RMSE. Reports include data-quality counts
and monocular-3D limitations.

## Exports

JSON / CSV / Markdown from the Research page. Existing session CSV/JSON/PDF
and OpenSim export paths are unchanged.
