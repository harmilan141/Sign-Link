"""
extract_landmarks.py
=====================

Batch-extracts MediaPipe landmarks from every video in the
ISL-CSLRT "Videos_Sentence_Level" corpus and saves one compressed
``.npz`` file per video (via :mod:`save_npz`).

Pipeline overview
------------------
1.  Recursively discover every video file under the dataset root.
2.  Load ``ISL Corpus sign glosses.csv`` and build a normalized
    ``Sentence -> Gloss`` lookup table.
3.  For every video (in parallel, across worker processes):
      a. Skip it if an output ``.npz`` already exists (resume support).
      b. Open it with OpenCV to read ``fps`` / ``frame_count``, then run
         the whole video through the MediaPipe Holistic pipeline via
         :mod:`mediapipe_utils` to obtain a fixed-length landmark
         vector per frame, building a ``(T, D)`` sequence.
      c. Resolve the video's sentence/gloss label from the CSV mapping.
      d. Save ``landmarks``, ``sentence``, ``gloss``, ``video_name``,
         ``fps``, ``frame_count`` and ``feature_dim`` to a single
         ``.npz`` file via :func:`save_npz.save_landmarks_npz`.
4.  Report a final summary (processed / skipped / failed counts).

``mediapipe_utils`` interface
------------------------------
This script uses the following functions exposed by ``mediapipe_utils.py``::

    initialize_holistic() -> <holistic model instance>
    process_frame(...)
    extract_landmarks(...)
    normalize_landmarks(...)
    flatten_landmarks(...)
    extract_frame_features(...)
    extract_video_landmark_sequence(video_path, holistic) -> np.ndarray

Only :func:`initialize_holistic` and :func:`extract_video_landmark_sequence`
are called directly from this script; the remaining functions are used
internally by ``mediapipe_utils`` itself.

Usage
-----
.. code-block:: bash

    python extract_landmarks.py \\
        --video-root /workspace/datasets/ISL_CSLRT/ISL_CSLRT_Corpus/Videos_Sentence_Level \\
        --csv-path "/workspace/datasets/ISL_CSLRT/ISL_CSLRT_Corpus/ISL Corpus sign glosses.csv" \\
        --output-dir preprocessed \\
        --workers 4
"""

from __future__ import annotations

import argparse
import atexit
import csv
import logging
import multiprocessing as mp
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from mediapipe_utils import initialize_holistic, extract_video_landmark_sequence
from save_npz import npz_exists, save_landmarks_npz

logger = logging.getLogger("extract_landmarks")

#: Video file extensions considered part of the corpus.
VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI")

#: Per-process global Holistic model instance (initialized once per worker
#: via ``_init_worker`` to avoid re-creating expensive MediaPipe graphs for
#: every single video).
_HOLISTIC = None


@dataclass(frozen=True)
class VideoTask:
    """A single unit of work handed to a worker process.

    Attributes:
        video_path: Absolute path to the source video file.
        video_name: File stem (no extension) used as archive identifier.
        sentence: Resolved sentence text (may be ``"UNKNOWN"``).
        gloss: Resolved gloss text (may be ``"UNKNOWN"``).
    """

    video_path: Path
    video_name: str
    sentence: str
    gloss: str


@dataclass(frozen=True)
class ProcessResult:
    """Outcome of processing a single :class:`VideoTask`."""

    video_name: str
    status: str  # one of: "saved", "skipped", "failed"
    message: str = ""


def _normalize_text(text: str) -> str:
    """Normalize a string for robust lookup/matching.

    Lower-cases, strips surrounding whitespace, collapses internal
    whitespace, and removes punctuation so that minor formatting
    differences between filenames/folder names and CSV entries do not
    prevent a match.

    Args:
        text: Raw input string.

    Returns:
        Normalized string suitable for dictionary-key comparison.
    """
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _parse_csv(handle, csv_path: Path) -> Dict[str, Tuple[str, str]]:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        raise ValueError(f"Could not read header row from {csv_path}")

    field_lookup = {name.strip().lower(): name for name in reader.fieldnames}
    sentence_col = field_lookup.get("sentence")
    gloss_col = field_lookup.get("sign glosses")

    if sentence_col is None or gloss_col is None:
        raise ValueError(
            "CSV must contain 'Sentence' and 'SIGN GLOSSES' columns; "
            f"found columns: {reader.fieldnames}"
        )

    sentence_to_gloss: Dict[str, Tuple[str, str]] = {}
    for row in reader:
        sentence_raw = (row.get(sentence_col) or "").strip()
        gloss_raw = (row.get(gloss_col) or "").strip()
        if not sentence_raw:
            continue
        sentence_to_gloss[_normalize_text(sentence_raw)] = (sentence_raw, gloss_raw)
    return sentence_to_gloss


def load_sentence_gloss_map(csv_path: Path) -> Dict[str, Tuple[str, str]]:
    """Load ``ISL Corpus sign glosses.csv`` into a normalized lookup table.

    The CSV is expected to contain (at least) a ``Sentence`` column and a
    ``SIGN GLOSSES`` column (column name matching is case-insensitive and
    whitespace-tolerant).

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Mapping from normalized sentence text to a tuple of (original sentence, gloss).

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If the required columns cannot be found.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Gloss CSV not found: {csv_path}")

    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            return _parse_csv(handle, csv_path)
    except UnicodeDecodeError:
        with csv_path.open("r", newline="", encoding="cp1252") as handle:
            return _parse_csv(handle, csv_path)


def discover_videos(video_root: Path) -> List[Path]:
    """Recursively find every video file under ``video_root``.

    Args:
        video_root: Root directory of the sentence-level video corpus.

    Returns:
        Sorted list of video file paths.

    Raises:
        FileNotFoundError: If ``video_root`` does not exist.
    """
    if not video_root.is_dir():
        raise FileNotFoundError(f"Video root directory not found: {video_root}")

    videos = [
        path
        for path in video_root.rglob("*")
        if path.is_file() and path.suffix in VIDEO_EXTENSIONS
    ]
    videos.sort()
    logger.info("Discovered %d video files under %s", len(videos), video_root)
    return videos


def resolve_sentence_gloss(
    video_path: Path,
    video_root: Path,
    sentence_to_gloss: Dict[str, Tuple[str, str]],
) -> Tuple[str, str]:
    """Resolve the (sentence, gloss) label pair for a video.

    Matching strategy: the ISL-CSLRT sentence-level corpus typically
    organizes videos in sub-folders named after the sentence they depict.
    This function tries, in order, the video's parent directory name(s)
    (from deepest to shallowest, up to ``video_root``) and finally the
    video's own filename stem, normalizing each candidate and looking it
    up in ``sentence_to_gloss``.

    Args:
        video_path: Path to the video file.
        video_root: Root directory of the corpus (used as a search bound).
        sentence_to_gloss: Normalized sentence -> (original sentence, gloss) lookup table.

    Returns:
        Tuple of ``(sentence, gloss)``. If no match is found, both values
        fall back to ``"UNKNOWN"`` and a warning is logged by the caller.
    """
    video_path = video_path.resolve()
    video_root = video_root.resolve()
    candidates: List[str] = []

    parent = video_path.parent
    while parent != video_root and video_root in parent.parents:
        candidates.append(parent.name)
        parent = parent.parent
    if parent == video_root:
        pass  # already covered all intermediate folder names

    candidates.append(video_path.stem)

    for candidate in candidates:
        normalized = _normalize_text(candidate)
        if normalized in sentence_to_gloss:
            return sentence_to_gloss[normalized]

    return "UNKNOWN", "UNKNOWN"


def build_tasks(
    video_root: Path,
    sentence_to_gloss: Dict[str, Tuple[str, str]],
    output_dir: Path,
    resume: bool,
) -> List[VideoTask]:
    """Build the full list of :class:`VideoTask` objects to process.

    Videos whose output archive already exists are excluded up-front when
    ``resume`` is ``True``, avoiding unnecessary work being dispatched to
    worker processes at all.

    Args:
        video_root: Root directory of the video corpus.
        sentence_to_gloss: Normalized sentence -> (original sentence, gloss) lookup table.
        output_dir: Directory where ``.npz`` archives are written.
        resume: If ``True``, skip videos already present in ``output_dir``.

    Returns:
        List of tasks to be processed.
    """
    tasks: List[VideoTask] = []
    unmatched = 0

    for video_path in discover_videos(video_root):
        video_name = video_path.stem

        if resume and npz_exists(output_dir, video_name):
            continue

        sentence, gloss = resolve_sentence_gloss(video_path, video_root, sentence_to_gloss)
        if sentence == "UNKNOWN":
            unmatched += 1
            logger.warning("No sentence/gloss match found for video: %s", video_path)

        tasks.append(
            VideoTask(
                video_path=video_path,
                video_name=video_name,
                sentence=sentence,
                gloss=gloss,
            )
        )

    if unmatched:
        logger.warning(
            "%d/%d videos had no matching CSV entry and were labeled 'UNKNOWN'",
            unmatched,
            len(tasks),
        )

    return tasks


def _close_worker_resources() -> None:
    """Explicitly close worker MediaPipe resources on exit."""
    global _HOLISTIC
    if _HOLISTIC is not None:
        try:
            _HOLISTIC.close()
        except Exception:
            pass
        _HOLISTIC = None


def _init_worker() -> None:
    """Multiprocessing pool initializer: create one Holistic model per worker.

    Creating the MediaPipe Holistic model once per worker process (rather
    than once per video) avoids the substantial overhead of repeatedly
    constructing/tearing down MediaPipe's internal graphs.
    """
    global _HOLISTIC
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(processName)s: %(message)s",
    )
    _HOLISTIC = initialize_holistic()
    atexit.register(_close_worker_resources)


def _extract_video_landmarks(video_path: Path) -> Tuple[np.ndarray, float, int]:
    """Read a video's fps/frame_count via OpenCV and extract its landmark sequence.

    Args:
        video_path: Path to the video file.

    Returns:
        Tuple of ``(landmarks, fps, frame_count)`` where ``landmarks`` has
        shape ``(T, D)``, ``fps`` is the source video's frame rate as
        reported by OpenCV, and ``frame_count`` is the number of frames
        reported by OpenCV (falling back to ``landmarks.shape[0]`` if
        OpenCV cannot report a usable frame count).

    Raises:
        RuntimeError: If the video cannot be opened, or no landmark
            sequence could be extracted from it.
    """
    assert _HOLISTIC is not None, "Worker Holistic model was not initialized"

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()

    # Fallback to standard 25.0 fps if OpenCV yields invalid/empty fps
    if np.isnan(fps) or fps <= 0:
        fps = 25.0

    landmarks = extract_video_landmark_sequence(str(video_path), _HOLISTIC)
    landmarks = np.asarray(landmarks, dtype=np.float32)

    if landmarks.ndim != 2 or landmarks.shape[0] == 0:
        raise RuntimeError(f"No usable landmark sequence extracted from video: {video_path}")

    if frame_count <= 0:
        frame_count = landmarks.shape[0]

    return landmarks, fps, frame_count


def process_task(task: VideoTask, output_dir: Path, overwrite: bool) -> ProcessResult:
    """Process a single video end-to-end: extract landmarks and save.

    Args:
        task: The video task to process.
        output_dir: Directory where the ``.npz`` archive should be saved.
        overwrite: Whether to overwrite an existing archive.

    Returns:
        A :class:`ProcessResult` describing the outcome.
    """
    try:
        landmarks, fps, frame_count = _extract_video_landmarks(task.video_path)
        saved_path = save_landmarks_npz(
            output_dir=output_dir,
            video_name=task.video_name,
            landmarks=landmarks,
            sentence=task.sentence,
            gloss=task.gloss,
            fps=fps,
            frame_count=frame_count,
            overwrite=overwrite,
        )
        if saved_path is None:
            return ProcessResult(task.video_name, "failed", "save_landmarks_npz returned None")
        return ProcessResult(task.video_name, "saved", str(saved_path))
    except Exception as exc:  # noqa: BLE001 - broad on purpose, one bad video must not kill the run
        logger.exception("Failed to process video: %s", task.video_path)
        return ProcessResult(task.video_name, "failed", str(exc))


def _worker_entrypoint(args: Tuple[VideoTask, Path, bool]) -> ProcessResult:
    """Adapter so :func:`process_task` can be used with ``Pool.imap_unordered``.

    Args:
        args: Tuple of ``(task, output_dir, overwrite)``.

    Returns:
        A :class:`ProcessResult` describing the outcome.
    """
    task, output_dir, overwrite = args
    return process_task(task, output_dir, overwrite)


def run_pipeline(
    video_root: Path,
    csv_path: Path,
    output_dir: Path,
    workers: int,
    resume: bool,
    overwrite: bool,
) -> None:
    """Run the full extraction pipeline over the whole corpus.

    Args:
        video_root: Root directory of the sentence-level video corpus.
        csv_path: Path to ``ISL Corpus sign glosses.csv``.
        output_dir: Directory to write ``.npz`` archives to.
        workers: Number of worker processes to use.
        resume: If ``True``, skip videos already processed.
        overwrite: If ``True``, re-save even if an archive already exists
            (only meaningful when ``resume`` is ``False`` for a given
            video, or when re-running without resume).
    """
    video_root = video_root.resolve()
    csv_path = csv_path.resolve()
    output_dir = output_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    sentence_to_gloss = load_sentence_gloss_map(csv_path)
    tasks = build_tasks(video_root, sentence_to_gloss, output_dir, resume)

    total_videos = len(discover_videos(video_root))
    already_done = total_videos - len(tasks)
    if resume and already_done:
        logger.info("Resuming: %d/%d videos already processed, skipping them", already_done, total_videos)

    if not tasks:
        logger.info("Nothing to do -- all videos already processed.")
        return

    logger.info("Processing %d videos using %d worker process(es)", len(tasks), workers)

    saved_count = 0
    failed_count = 0
    work_items = [(task, output_dir, overwrite) for task in tasks]

    with mp.Pool(processes=workers, initializer=_init_worker) as pool:
        results_iter = pool.imap_unordered(_worker_entrypoint, work_items)
        for result in tqdm(results_iter, total=len(work_items), desc="Extracting landmarks", unit="video"):
            if result.status == "saved":
                saved_count += 1
            elif result.status == "failed":
                failed_count += 1
                logger.error("FAILED: %s (%s)", result.video_name, result.message)

    logger.info(
        "Done. saved=%d failed=%d skipped_before_dispatch=%d total_discovered=%d",
        saved_count,
        failed_count,
        already_done,
        total_videos,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract MediaPipe landmarks from every ISL-CSLRT sentence-level "
            "video and save one compressed .npz archive per video."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path(
            "/workspace/datasets/ISL_CSLRT/ISL_CSLRT_Corpus/Videos_Sentence_Level"
        ),
        help="Root directory containing sentence-level videos (searched recursively).",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path(
            "/workspace/datasets/ISL_CSLRT/ISL_CSLRT_Corpus/corpus_csv_files/ISL Corpus sign glosses.csv"
        ),
        help="Path to the CSV mapping Sentence -> SIGN GLOSSES.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preprocessed"),
        help="Directory to write per-video .npz archives to.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (mp.cpu_count() or 2) - 1),
        help="Number of worker processes to use for extraction.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable automatic resume; reprocess every video even if its archive already exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing archives instead of leaving them untouched.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity for the main process.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Script entry point.

    Args:
        argv: Optional argument list, primarily for testing.

    Returns:
        Process exit code (``0`` on success, non-zero on fatal error).
    """
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        run_pipeline(
            video_root=args.video_root,
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            workers=args.workers,
            resume=not args.no_resume,
            overwrite=args.overwrite,
        )
    except Exception:
        logger.exception("Pipeline failed with a fatal error")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
