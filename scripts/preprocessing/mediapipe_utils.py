"""
mediapipe_utils.py

Reusable utility functions for extracting, normalizing, and flattening
MediaPipe Holistic landmarks (pose, face, left hand, right hand) from
video frames.

Intended for use in an Indian Sign Language (ISL) translation
preprocessing pipeline built on top of the ISL-CSLTR dataset, where raw
RGB video is converted into compact landmark-based feature sequences
instead of being fed directly into a video model.

This module contains ONLY reusable, importable functions/classes. It
does not implement a dataset-processing script, CLI entry point, or
`__main__` block.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

mp_holistic = mp.solutions.holistic

#: Number of landmarks produced by MediaPipe Holistic for each body part.
NUM_POSE_LANDMARKS: int = 33
NUM_FACE_LANDMARKS: int = 468
NUM_HAND_LANDMARKS: int = 21

#: Number of coordinate dimensions stored per landmark.
#: Pose landmarks include (x, y, z, visibility); all others use (x, y, z).
POSE_DIMS: int = 4
OTHER_DIMS: int = 3

#: Flattened feature dimensionality for each body part.
POSE_FEATURE_DIM: int = NUM_POSE_LANDMARKS * POSE_DIMS
FACE_FEATURE_DIM: int = NUM_FACE_LANDMARKS * OTHER_DIMS
HAND_FEATURE_DIM: int = NUM_HAND_LANDMARKS * OTHER_DIMS

#: Total flattened feature dimensionality for a single frame
#: (pose + face + left_hand + right_hand).
TOTAL_FEATURE_DIM: int = (
    POSE_FEATURE_DIM + FACE_FEATURE_DIM + 2 * HAND_FEATURE_DIM
)

#: Indices (within the 33 pose landmarks) of the left/right shoulder,
#: used as a stable reference for scale normalization.
LEFT_SHOULDER_INDEX: int = 11
RIGHT_SHOULDER_INDEX: int = 12

#: Type alias for a dictionary holding one landmark array per body part.
LandmarkDict = Dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------


def initialize_holistic(
    static_image_mode: bool = False,
    model_complexity: int = 1,
    smooth_landmarks: bool = True,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> mp_holistic.Holistic:
    """
    Initialize and return a MediaPipe Holistic model instance.

    Args:
        static_image_mode: If True, treats every frame as an independent
            image (no tracking between frames). Use False for video
            streams so MediaPipe can track landmarks across frames.
        model_complexity: Complexity of the pose landmark model
            (0, 1, or 2). Higher values are more accurate but slower.
        smooth_landmarks: Whether to temporally smooth landmarks to
            reduce jitter across video frames.
        min_detection_confidence: Minimum confidence value ([0.0, 1.0])
            for person/part detection to be considered successful.
        min_tracking_confidence: Minimum confidence value ([0.0, 1.0])
            for landmark tracking to be considered successful.

    Returns:
        A configured `mediapipe.solutions.holistic.Holistic` instance.
        The caller is responsible for closing it (e.g. via `.close()`
        or a `with` block) once processing is complete.
    """
    return mp_holistic.Holistic(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


# ---------------------------------------------------------------------------
# Frame processing
# ---------------------------------------------------------------------------


def process_frame(frame: np.ndarray, holistic_model: mp_holistic.Holistic) -> Any:
    """
    Run MediaPipe Holistic inference on a single BGR video frame.

    Args:
        frame: A single video frame in BGR format (as read by OpenCV),
            with shape (height, width, 3).
        holistic_model: An initialized MediaPipe Holistic model
            instance, typically created via `initialize_holistic`.

    Returns:
        The MediaPipe Holistic results object. Its `pose_landmarks`,
        `face_landmarks`, `left_hand_landmarks`, and
        `right_hand_landmarks` attributes are each either a MediaPipe
        landmark list or `None` if that part was not detected.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = holistic_model.process(rgb_frame)
    rgb_frame.flags.writeable = True
    return results


# ---------------------------------------------------------------------------
# Landmark extraction
# ---------------------------------------------------------------------------


def _landmark_list_to_array(
    landmark_list: Any, num_landmarks: int, num_dims: int
) -> np.ndarray:
    """
    Convert a MediaPipe landmark list into a fixed-size NumPy array.

    If `landmark_list` is `None` (i.e. that body part was not detected
    in the frame), a zero-filled array of the expected shape is
    returned instead, so downstream code never has to special-case
    missing detections.

    Args:
        landmark_list: A MediaPipe `NormalizedLandmarkList` (or `None`).
        num_landmarks: Expected number of landmarks for this body part.
        num_dims: Number of coordinate dimensions per landmark
            (4 for pose: x, y, z, visibility; 3 otherwise: x, y, z).

    Returns:
        np.ndarray of shape (num_landmarks, num_dims), dtype float32.
    """
    if landmark_list is None:
        return np.zeros((num_landmarks, num_dims), dtype=np.float32)

    coords = np.zeros((num_landmarks, num_dims), dtype=np.float32)
    for idx, landmark in enumerate(landmark_list.landmark):
        if idx >= num_landmarks:
            break
        if num_dims == POSE_DIMS:
            coords[idx] = (
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility,
            )
        else:
            coords[idx] = (landmark.x, landmark.y, landmark.z)
    return coords


def extract_landmarks(results: Any) -> LandmarkDict:
    """
    Extract raw (un-normalized) landmarks from MediaPipe Holistic results.

    Any body part that was not detected in the frame is represented as
    an all-zero array of the correct shape, so the returned dictionary
    always has consistent, fixed-size arrays regardless of detection
    success.

    Args:
        results: The output of `process_frame` (a MediaPipe Holistic
            results object).

    Returns:
        Dictionary with keys:
            - "pose": np.ndarray of shape (33, 4)
            - "face": np.ndarray of shape (468, 3)
            - "left_hand": np.ndarray of shape (21, 3)
            - "right_hand": np.ndarray of shape (21, 3)
    """
    pose = _landmark_list_to_array(
        getattr(results, "pose_landmarks", None), NUM_POSE_LANDMARKS, POSE_DIMS
    )
    face = _landmark_list_to_array(
        getattr(results, "face_landmarks", None), NUM_FACE_LANDMARKS, OTHER_DIMS
    )
    left_hand = _landmark_list_to_array(
        getattr(results, "left_hand_landmarks", None),
        NUM_HAND_LANDMARKS,
        OTHER_DIMS,
    )
    right_hand = _landmark_list_to_array(
        getattr(results, "right_hand_landmarks", None),
        NUM_HAND_LANDMARKS,
        OTHER_DIMS,
    )

    return {
        "pose": pose,
        "face": face,
        "left_hand": left_hand,
        "right_hand": right_hand,
    }


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _normalize_pose(pose: np.ndarray, reference_index: int) -> np.ndarray:
    """
    Normalize pose landmarks to be translation- and scale-invariant.

    Coordinates are centered on a reference landmark (default: nose)
    and scaled by the shoulder-to-shoulder distance, which is
    reasonably stable across signers and camera distances. Visibility
    scores are passed through unchanged. Arrays that are entirely zero
    (i.e. pose was not detected) are returned unchanged.

    Args:
        pose: np.ndarray of shape (33, 4) -> (x, y, z, visibility).
        reference_index: Index of the pose landmark used as the origin.

    Returns:
        np.ndarray of shape (33, 4) with normalized (x, y, z) and
        unmodified visibility.
    """
    if not np.any(pose):
        return pose.copy()

    coords = pose[:, :3]
    visibility = pose[:, 3:4]

    origin = coords[reference_index].copy()
    centered = coords - origin

    scale = float(
        np.linalg.norm(
            coords[LEFT_SHOULDER_INDEX] - coords[RIGHT_SHOULDER_INDEX]
        )
    )
    if scale < 1e-6:
        scale = 1.0
    centered = centered / scale

    return np.concatenate([centered, visibility], axis=1).astype(np.float32)


def _normalize_part(part: np.ndarray) -> np.ndarray:
    """
    Normalize a face/hand landmark array to be translation- and
    scale-invariant.

    Coordinates are centered on the part's own first landmark and
    scaled by the maximum distance from that origin to any other
    landmark in the part. Arrays that are entirely zero (i.e. the part
    was not detected) are returned unchanged so missing landmarks stay
    zero after normalization.

    Args:
        part: np.ndarray of shape (num_landmarks, 3) -> (x, y, z).

    Returns:
        np.ndarray of shape (num_landmarks, 3) with normalized
        coordinates.
    """
    if not np.any(part):
        return part.copy()

    origin = part[0].copy()
    centered = part - origin

    max_extent = float(np.max(np.linalg.norm(centered, axis=1)))
    scale = max_extent if max_extent > 1e-6 else 1.0
    centered = centered / scale

    return centered.astype(np.float32)


def normalize_landmarks(
    landmarks: LandmarkDict,
    pose_reference_index: int = 0,
) -> LandmarkDict:
    """
    Normalize all landmark groups to be translation- and scale-invariant.

    - Pose landmarks are centered on `pose_reference_index` (default:
      the nose landmark) and scaled by shoulder-to-shoulder distance.
    - Face, left-hand, and right-hand landmarks are each independently
      centered on their own first landmark and scaled by their own
      maximum point-to-origin distance.
    - Any body part with all-zero (missing) landmarks remains all-zero
      after normalization.

    Args:
        landmarks: Dictionary as returned by `extract_landmarks`, with
            keys "pose", "face", "left_hand", "right_hand".
        pose_reference_index: Index of the pose landmark used as the
            normalization origin for the pose (default: 0, the nose).

    Returns:
        A new dictionary with the same keys, containing normalized
        landmark arrays of the same shapes as the input.
    """
    return {
        "pose": _normalize_pose(landmarks["pose"], pose_reference_index),
        "face": _normalize_part(landmarks["face"]),
        "left_hand": _normalize_part(landmarks["left_hand"]),
        "right_hand": _normalize_part(landmarks["right_hand"]),
    }


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------


def flatten_landmarks(landmarks: LandmarkDict) -> np.ndarray:
    """
    Flatten a landmark dictionary into a single 1D feature vector.

    The parts are concatenated in a fixed order (pose, face, left_hand,
    right_hand) so the resulting vector layout is consistent across
    frames and videos.

    Args:
        landmarks: Dictionary with keys "pose", "face", "left_hand",
            "right_hand", each mapping to a NumPy array (as produced by
            `extract_landmarks` or `normalize_landmarks`).

    Returns:
        1D np.ndarray of length `TOTAL_FEATURE_DIM`, dtype float32.
    """
    part_order = ("pose", "face", "left_hand", "right_hand")
    flattened_parts = [landmarks[part].flatten() for part in part_order]
    feature_vector = np.concatenate(flattened_parts, axis=0)
    return feature_vector.astype(np.float32)


# ---------------------------------------------------------------------------
# High-level convenience functions
# ---------------------------------------------------------------------------


def extract_frame_features(
    frame: np.ndarray,
    holistic_model: mp_holistic.Holistic,
    normalize: bool = True,
) -> np.ndarray:
    """
    Run the full single-frame pipeline: process -> extract -> normalize
    -> flatten.

    Args:
        frame: A single video frame in BGR format (as read by OpenCV).
        holistic_model: An initialized MediaPipe Holistic model
            instance, typically created via `initialize_holistic`.
        normalize: Whether to apply translation/scale normalization
            before flattening.

    Returns:
        1D np.ndarray of length `TOTAL_FEATURE_DIM` representing this
        frame's landmark feature vector.
    """
    results = process_frame(frame, holistic_model)
    landmarks = extract_landmarks(results)
    if normalize:
        landmarks = normalize_landmarks(landmarks)
    return flatten_landmarks(landmarks)


def extract_video_landmark_sequence(
    video_path: str,
    holistic_model: Optional[mp_holistic.Holistic] = None,
    normalize: bool = True,
    max_frames: Optional[int] = None,
) -> np.ndarray:
    """
    Extract a sequence of per-frame landmark feature vectors from a
    video file.

    This is a reusable helper for turning one sentence-level video into
    a `(num_frames, TOTAL_FEATURE_DIM)` array of MediaPipe Holistic
    features, suitable as input to a downstream sequence model.

    Args:
        video_path: Path to the video file to read.
        holistic_model: Optional pre-initialized Holistic model. If
            `None`, a new instance is created internally (with default
            settings suited to video) and closed automatically after
            processing.
        normalize: Whether to apply translation/scale normalization to
            each frame's landmarks before flattening.
        max_frames: Optional maximum number of frames to process. If
            `None`, the entire video is processed.

    Returns:
        np.ndarray of shape (num_frames, TOTAL_FEATURE_DIM), dtype
        float32. Returns an array with `num_frames == 0` if the video
        contains no readable frames.

    Raises:
        FileNotFoundError: If the video file cannot be opened by OpenCV.
    """
    owns_model = holistic_model is None
    if owns_model:
        holistic_model = initialize_holistic(static_image_mode=False)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        if owns_model:
            holistic_model.close()
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    feature_sequence: list[np.ndarray] = []
    frame_count = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            feature_vector = extract_frame_features(
                frame, holistic_model, normalize=normalize
            )
            feature_sequence.append(feature_vector)
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        capture.release()
        if owns_model:
            holistic_model.close()

    if not feature_sequence:
        return np.zeros((0, TOTAL_FEATURE_DIM), dtype=np.float32)

    return np.stack(feature_sequence, axis=0)
