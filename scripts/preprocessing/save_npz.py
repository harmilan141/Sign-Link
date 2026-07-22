"""
save_npz.py
===========

Reusable helper module for persisting per-video landmark sequences to
compressed ``.npz`` files, and for the small set of filesystem checks
(existence / resume) that the extraction pipeline needs around them.

This module intentionally has **no** dependency on OpenCV, MediaPipe, or
multiprocessing so that it can be imported cheaply from worker processes,
unit tests, or downstream dataset-loading code without pulling in heavy
video/vision libraries.

Each ``.npz`` file produced by :func:`save_landmarks_npz` contains exactly
seven arrays/scalars:

``landmarks``
    ``np.ndarray`` of shape ``(T, D)`` and dtype ``float32`` -- the per-frame
    flattened landmark vectors for the whole video.
``sentence``
    0-d ``np.ndarray`` (via ``np.array(str)``) holding the original sentence
    text associated with the video.
``gloss``
    0-d ``np.ndarray`` holding the ISL gloss sequence mapped to ``sentence``.
``video_name``
    0-d ``np.ndarray`` holding the video's file stem (no extension),
    used as a stable identifier.
``fps``
    0-d ``np.ndarray`` (``float32``) holding the source video's frame rate.
``frame_count``
    0-d ``np.ndarray`` (``int64``) holding the source video's frame count.
``feature_dim``
    0-d ``np.ndarray`` (``int64``) holding ``D``, the per-frame landmark
    vector dimensionality (i.e. ``landmarks.shape[1]``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

#: Standard extension used for all saved landmark archives.
NPZ_EXTENSION: str = ".npz"


def build_output_path(output_dir: Union[str, Path], video_name: str) -> Path:
    """Build the destination ``.npz`` path for a given video.

    Args:
        output_dir: Directory in which preprocessed archives are stored.
        video_name: File stem (no extension) identifying the source video.

    Returns:
        The ``Path`` where the archive for ``video_name`` should be
        read from / written to.
    """
    output_dir = Path(output_dir)
    safe_name = video_name.strip()
    return output_dir / f"{safe_name}{NPZ_EXTENSION}"


def npz_exists(output_dir: Union[str, Path], video_name: str) -> bool:
    """Check whether a landmark archive already exists for ``video_name``.

    Used by the extraction pipeline to implement automatic resume: videos
    whose archive already exists are skipped without being reprocessed.

    Args:
        output_dir: Directory in which preprocessed archives are stored.
        video_name: File stem (no extension) identifying the source video.

    Returns:
        ``True`` if a non-empty archive already exists on disk.
    """
    path = build_output_path(output_dir, video_name)
    return path.is_file() and path.stat().st_size > 0


def save_landmarks_npz(
    output_dir: Union[str, Path],
    video_name: str,
    landmarks: np.ndarray,
    sentence: str,
    gloss: str,
    fps: float,
    frame_count: int,
    overwrite: bool = False,
) -> Optional[Path]:
    """Save a single video's landmark sequence to a compressed ``.npz`` file.

    The array is written atomically: it is first written to a temporary
    ``.tmp`` file in the same directory and then renamed into place, so a
    process interrupted mid-write can never leave behind a corrupt archive
    that would be mistaken for a completed one by :func:`npz_exists`.

    Args:
        output_dir: Directory in which to store the archive. Created if it
            does not already exist.
        video_name: File stem (no extension) identifying the source video.
        landmarks: Array of shape ``(T, D)`` with per-frame landmark
            vectors, where ``T`` is the number of frames and ``D`` is the
            flattened landmark dimensionality.
        sentence: The original sentence text spoken/signed in the video.
        gloss: The gloss sequence corresponding to ``sentence``.
        fps: Frame rate of the source video, as read by OpenCV.
        frame_count: Number of frames in the source video, as read by
            OpenCV.
        overwrite: If ``False`` (default) and the archive already exists,
            the save is skipped and the existing path is returned. If
            ``True``, any existing archive is replaced.

    Returns:
        The path to the saved (or pre-existing) ``.npz`` file, or ``None``
        if saving failed.

    Raises:
        ValueError: If ``landmarks`` is not a 2-D array.
    """
    if landmarks.ndim != 2:
        raise ValueError(
            f"'landmarks' must have shape (T, D), got shape {landmarks.shape}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = build_output_path(output_dir, video_name)

    if final_path.is_file() and not overwrite:
        logger.debug("Archive already exists, skipping save: %s", final_path)
        return final_path

    tmp_path = final_path.with_name(final_path.stem + ".tmp.npz")

    try:
        feature_dim = landmarks.shape[1]
        np.savez_compressed(
            tmp_path,
            landmarks=landmarks.astype(np.float32),
            sentence=np.array(sentence),
            gloss=np.array(gloss),
            video_name=np.array(video_name),
            fps=np.array(fps, dtype=np.float32),
            frame_count=np.array(frame_count, dtype=np.int64),
            feature_dim=np.array(feature_dim, dtype=np.int64),
        )

        # Validate saved temporary file before marking complete
        with np.load(tmp_path, allow_pickle=False) as data:
            required_keys = {
                "landmarks",
                "sentence",
                "gloss",
                "video_name",
                "fps",
                "frame_count",
                "feature_dim",
            }
            missing = required_keys - set(data.files)
            if missing:
                raise KeyError(f"Archive is missing keys: {missing}")
            
            saved_landmarks = data["landmarks"]
            if saved_landmarks.dtype != np.float32:
                raise TypeError(f"landmarks must be float32, got {saved_landmarks.dtype}")
            if saved_landmarks.ndim != 2:
                raise ValueError(f"landmarks must be 2D, got shape {saved_landmarks.shape}")
            if saved_landmarks.shape[1] != feature_dim:
                raise ValueError(f"landmarks feature dimension mismatch: expected {feature_dim}, got {saved_landmarks.shape[1]}")
            
            # Ensure scalar keys are 0-dimensional arrays
            for key in ["sentence", "gloss", "video_name", "fps", "frame_count", "feature_dim"]:
                if data[key].ndim != 0:
                    raise ValueError(f"Scalar field {key} must have 0 dimensions, got {data[key].ndim}")

        tmp_path.replace(final_path)
        logger.debug(
            "Saved landmarks to %s (shape=%s, fps=%s, frame_count=%s, feature_dim=%s)",
            final_path,
            landmarks.shape,
            fps,
            frame_count,
            feature_dim,
        )
        return final_path
    except Exception:
        logger.exception("Failed to save landmarks for '%s'", video_name)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return None


def load_landmarks_npz(path: Union[str, Path]) -> dict:
    """Load a previously saved landmark archive.

    Args:
        path: Path to the ``.npz`` file to load.

    Returns:
        A dictionary with keys ``landmarks`` (``np.ndarray`` of shape
        ``(T, D)``), ``sentence`` (``str``), ``gloss`` (``str``),
        ``video_name`` (``str``), ``fps`` (``float``), ``frame_count``
        (``int``), and ``feature_dim`` (``int``).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If the archive is missing one of the expected keys.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such archive: {path}")

    with np.load(path, allow_pickle=False) as data:
        required_keys = {
            "landmarks",
            "sentence",
            "gloss",
            "video_name",
            "fps",
            "frame_count",
            "feature_dim",
        }
        missing = required_keys - set(data.files)
        if missing:
            raise KeyError(f"Archive '{path}' is missing keys: {missing}")

        return {
            "landmarks": data["landmarks"],
            "sentence": str(data["sentence"]),
            "gloss": str(data["gloss"]),
            "video_name": str(data["video_name"]),
            "fps": float(data["fps"]),
            "frame_count": int(data["frame_count"]),
            "feature_dim": int(data["feature_dim"]),
        }
