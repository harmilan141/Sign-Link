import numpy as np

def normalize_landmarks_sequence(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalizes a sequence of flat 1662-dim landmark features.
    For each frame:
    1. Extract pose, face, left hand, and right hand coordinates.
    2. Compute the shoulder midpoint: average of left shoulder (p11) and right shoulder (p12).
    3. Compute the hip midpoint: average of left hip (p23) and right hip (p24).
    4. Compute torso length: distance between shoulder and hip midpoints.
    5. Subtract shoulder midpoint from all coordinates, and scale by torso length.
    6. Re-flatten and return the sequence.
    
    Args:
        landmarks: np.ndarray of shape (T, 1662) or (1662,)
        
    Returns:
        np.ndarray of same shape as input, containing normalized coordinates.
    """
    is_1d = (landmarks.ndim == 1)
    if is_1d:
        seq = landmarks[np.newaxis, :]
    else:
        seq = landmarks.copy()
        
    T = seq.shape[0]
    normalized_seq = np.zeros_like(seq)
    
    for t in range(T):
        frame = seq[t]
        
        # Unpack landmark groups
        pose = frame[0:132].reshape((33, 4))
        face = frame[132:1536].reshape((468, 3))
        left_hand = frame[1536:1599].reshape((21, 3))
        right_hand = frame[1599:1662].reshape((21, 3))
        
        # Check if pose is non-zero
        pose_visible = np.any(pose[:, :3] != 0.0)
        
        if pose_visible:
            # Left shoulder: 11, Right shoulder: 12
            shoulder_midpoint = (pose[11, :3] + pose[12, :3]) / 2.0
            # Left hip: 23, Right hip: 24
            hip_midpoint = (pose[23, :3] + pose[24, :3]) / 2.0
            
            torso_length = np.linalg.norm(shoulder_midpoint - hip_midpoint)
            scale = torso_length if torso_length > 1e-5 else 1.0
            
            # Normalize Pose (x, y, z only; keep visibility unchanged)
            pose_norm = pose.copy()
            pose_norm[:, :3] = (pose[:, :3] - shoulder_midpoint) / scale
            
            # Normalize Face
            face_norm = (face - shoulder_midpoint) / scale
            
            # Normalize Left Hand (if detected/non-zero)
            if np.any(left_hand != 0.0):
                left_hand_norm = (left_hand - shoulder_midpoint) / scale
            else:
                left_hand_norm = left_hand.copy()
                
            # Normalize Right Hand (if detected/non-zero)
            if np.any(right_hand != 0.0):
                right_hand_norm = (right_hand - shoulder_midpoint) / scale
            else:
                right_hand_norm = right_hand.copy()
        else:
            # Fallback if no pose detected (e.g. all zero landmarks)
            pose_norm = pose
            face_norm = face
            left_hand_norm = left_hand
            right_hand_norm = right_hand
            
        # Re-pack and flatten
        normalized_seq[t] = np.concatenate([
            pose_norm.flatten(),
            face_norm.flatten(),
            left_hand_norm.flatten(),
            right_hand_norm.flatten()
        ])
        
    if is_1d:
        return normalized_seq[0]
    return normalized_seq

def get_visibility_mask(landmarks: np.ndarray) -> np.ndarray:
    """
    Computes a binary visibility/detection mask for Pose, Left Hand, and Right Hand landmark groups.
    
    Args:
        landmarks: np.ndarray of shape (T, 1662) or (1662,)
        
    Returns:
        np.ndarray of shape (T, 3) or (3,) containing:
        - index 0: Pose detected (1.0) or not (0.0)
        - index 1: Left hand detected (1.0) or not (0.0)
        - index 2: Right hand detected (1.0) or not (0.0)
    """
    is_1d = (landmarks.ndim == 1)
    if is_1d:
        seq = landmarks[np.newaxis, :]
    else:
        seq = landmarks
        
    T = seq.shape[0]
    mask = np.zeros((T, 3), dtype=np.float32)
    
    for t in range(T):
        frame = seq[t]
        pose = frame[0:132].reshape((33, 4))
        left_hand = frame[1536:1599]
        right_hand = frame[1599:1662]
        
        # Check if pose contains any non-zero coordinate values
        if np.any(pose[:, :3] != 0.0):
            mask[t, 0] = 1.0
        # Check if hands contain any non-zero coordinate values
        if np.any(left_hand != 0.0):
            mask[t, 1] = 1.0
        if np.any(right_hand != 0.0):
            mask[t, 2] = 1.0
            
    if is_1d:
        return mask[0]
    return mask
