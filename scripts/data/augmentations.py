import numpy as np

def unpack_landmarks(frame: np.ndarray):
    """Unpack flat 1662-dim vector into pose, face, left_hand, and right_hand coordinate arrays."""
    pose = frame[0:132].reshape((33, 4))
    face = frame[132:1536].reshape((468, 3))
    left_hand = frame[1536:1599].reshape((21, 3))
    right_hand = frame[1599:1662].reshape((21, 3))
    return pose, face, left_hand, right_hand

def pack_landmarks(pose, face, left_hand, right_hand) -> np.ndarray:
    """Pack pose, face, left_hand, and right_hand coordinate arrays into flat 1662-dim vector."""
    return np.concatenate([
        pose.flatten(),
        face.flatten(),
        left_hand.flatten(),
        right_hand.flatten()
    ])

def speed_warp(landmarks: np.ndarray, warp_factor: float) -> np.ndarray:
    """Resamples the sequence length T by warp_factor using linear interpolation."""
    T, D = landmarks.shape
    new_T = int(round(T * warp_factor))
    if new_T < 3:
        return landmarks
        
    indices = np.linspace(0, T - 1, new_T)
    warped = np.zeros((new_T, D), dtype=landmarks.dtype)
    for i, idx in enumerate(indices):
        low = int(np.floor(idx))
        high = int(np.ceil(idx))
        weight = idx - low
        warped[i] = landmarks[low] * (1.0 - weight) + landmarks[high] * weight
    return warped

def frame_dropout(landmarks: np.ndarray, dropout_rate: float) -> np.ndarray:
    """Drops random frames and re-interpolates the sequence to preserve length T."""
    T, D = landmarks.shape
    if T < 5:
        return landmarks
        
    # Always keep first and last frames
    keep_indices = [0]
    for t in range(1, T - 1):
        if np.random.rand() > dropout_rate:
            keep_indices.append(t)
    keep_indices.append(T - 1)
    
    # Re-interpolate to size T
    new_indices = np.linspace(0, len(keep_indices) - 1, T)
    dropped = np.zeros((T, D), dtype=landmarks.dtype)
    for i, idx in enumerate(new_indices):
        low = int(np.floor(idx))
        high = int(np.ceil(idx))
        weight = idx - low
        
        orig_low = keep_indices[low]
        orig_high = keep_indices[high]
        dropped[i] = landmarks[orig_low] * (1.0 - weight) + landmarks[orig_high] * weight
    return dropped

def augment_landmarks_sequence(landmarks: np.ndarray, config: dict) -> np.ndarray:
    """
    Applies landspace-space and temporal-space augmentations on landmark sequence based on config parameters.
    
    Args:
        landmarks: np.ndarray of shape (T, 1662)
        config: dict containing probabilities and scales for each transform.
        
    Returns:
        np.ndarray of augmented landmarks.
    """
    T, D = landmarks.shape
    aug_seq = landmarks.copy()
    
    # --- 1. Temporal Speed Warping ---
    if config.get("speed_warp_prob", 0.0) > 0.0 and np.random.rand() < config["speed_warp_prob"]:
        warp_min = config.get("speed_warp_min", 0.8)
        warp_max = config.get("speed_warp_max", 1.2)
        warp_factor = np.random.uniform(warp_min, warp_max)
        aug_seq = speed_warp(aug_seq, warp_factor)
        T = aug_seq.shape[0] # T changes after speed warping
        
    # --- 2. Temporal Frame Dropout ---
    if config.get("frame_dropout_prob", 0.0) > 0.0 and np.random.rand() < config["frame_dropout_prob"]:
        dropout_rate = config.get("frame_dropout_rate", 0.05)
        aug_seq = frame_dropout(aug_seq, dropout_rate)
        
    # --- 3. Left-Right Mirroring ---
    if config.get("mirror_prob", 0.0) > 0.0 and np.random.rand() < config["mirror_prob"]:
        for t in range(T):
            pose, face, left_hand, right_hand = unpack_landmarks(aug_seq[t])
            
            # Negate x-coordinate for all landmarks (mirrors horizontally)
            pose[:, 0] = -pose[:, 0]
            face[:, 0] = -face[:, 0]
            
            # Swap hands: left hand gets negated right hand, right hand gets negated left hand
            temp_left = left_hand.copy()
            left_hand = -right_hand
            right_hand = -temp_left
            
            aug_seq[t] = pack_landmarks(pose, face, left_hand, right_hand)
            
    # --- 4. Spatial Scaling ---
    if config.get("scale_prob", 0.0) > 0.0 and np.random.rand() < config["scale_prob"]:
        scale_min = config.get("scale_min", 0.95)
        scale_max = config.get("scale_max", 1.05)
        scale_factor = np.random.uniform(scale_min, scale_max)
        
        for t in range(T):
            pose, face, left_hand, right_hand = unpack_landmarks(aug_seq[t])
            pose[:, :3] *= scale_factor
            face *= scale_factor
            left_hand *= scale_factor
            right_hand *= scale_factor
            aug_seq[t] = pack_landmarks(pose, face, left_hand, right_hand)
            
    # --- 5. Spatial Rotation (2D) ---
    if config.get("rotate_prob", 0.0) > 0.0 and np.random.rand() < config["rotate_prob"]:
        max_angle = config.get("rotate_max_angle_deg", 10.0)
        theta = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=aug_seq.dtype)
        
        for t in range(T):
            pose, face, left_hand, right_hand = unpack_landmarks(aug_seq[t])
            
            pose[:, :2] = np.dot(pose[:, :2], R.T)
            face[:, :2] = np.dot(face[:, :2], R.T)
            left_hand[:, :2] = np.dot(left_hand[:, :2], R.T)
            right_hand[:, :2] = np.dot(right_hand[:, :2], R.T)
            
            aug_seq[t] = pack_landmarks(pose, face, left_hand, right_hand)
            
    # --- 6. Gaussian Jitter ---
    if config.get("jitter_prob", 0.0) > 0.0 and np.random.rand() < config["jitter_prob"]:
        jitter_std = config.get("jitter_std", 0.005)
        for t in range(T):
            frame = aug_seq[t]
            # Add noise only to non-zero values (preserving MediaPipe origin-paddings)
            noise = np.random.normal(0, jitter_std, frame.shape).astype(frame.dtype)
            frame += noise * (frame != 0.0)
            aug_seq[t] = frame
            
    return aug_seq

def get_augmented_target_sentence(sentence: str, paraphrase_hook: callable = None) -> str:
    """
    Optional text-side augmentation hook to map/paraphrase the target sentence.
    
    Args:
        sentence: Input target English string
        paraphrase_hook: An optional external function mapping sentence -> paraphrase
        
    Returns:
        Augmented or original English target sentence
    """
    if paraphrase_hook is not None:
        try:
            return paraphrase_hook(sentence)
        except Exception:
            pass
    return sentence
