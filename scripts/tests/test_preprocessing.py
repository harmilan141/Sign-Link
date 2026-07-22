import numpy as np
import sys
from pathlib import Path

# Add scripts directory to path to import local modules
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from preprocessing.normalize_landmarks import normalize_landmarks_sequence, get_visibility_mask

def test_landmark_normalization_and_masking():
    # Create empty flat frame sequence: shape (1, 1662)
    landmarks = np.zeros((1, 1662), dtype=np.float32)
    
    # 1. Populate Pose shoulders & hips to define midpoint and scale
    # Pose indices in 1662 feature vector: 0 to 132 (33 joints * 4 floats)
    # Left shoulder is joint 11 -> flat index range [11*4 : 11*4 + 4] = [44:48]
    # Right shoulder is joint 12 -> flat index range [12*4 : 12*4 + 4] = [48:52]
    # Left hip is joint 23 -> flat index range [23*4 : 23*4 + 4] = [92:96]
    # Right hip is joint 24 -> flat index range [24*4 : 24*4 + 4] = [96:100]
    
    # Left Shoulder (x=0.0, y=1.0, z=0.0, visibility=1.0)
    landmarks[0, 44:48] = [0.0, 1.0, 0.0, 1.0]
    # Right Shoulder (x=2.0, y=1.0, z=0.0, visibility=1.0)
    landmarks[0, 48:52] = [2.0, 1.0, 0.0, 1.0]
    # Left Hip (x=0.0, y=-1.0, z=0.0, visibility=1.0)
    landmarks[0, 92:96] = [0.0, -1.0, 0.0, 1.0]
    # Right Hip (x=2.0, y=-1.0, z=0.0, visibility=1.0)
    landmarks[0, 96:100] = [2.0, -1.0, 0.0, 1.0]
    
    # Populate Left Hand (dim 1536:1599) with non-zero coordinates
    # First joint is index 1536 (x=5.0, y=5.0, z=5.0)
    landmarks[0, 1536:1539] = [5.0, 5.0, 5.0]
    
    # Keep Right Hand (dim 1599:1662) as all zeros (not detected)
    
    # Expected midpoints and scale:
    # Shoulder Midpoint = ([0, 1, 0] + [2, 1, 0]) / 2 = [1.0, 1.0, 0.0]
    # Hip Midpoint = ([0, -1, 0] + [2, -1, 0]) / 2 = [1.0, -1.0, 0.0]
    # Torso Length (scale) = distance([1, 1, 0], [1, -1, 0]) = 2.0
    
    # Call normalization
    normalized = normalize_landmarks_sequence(landmarks)
    
    # Assert normalized Pose left shoulder:
    # coord = ([0, 1, 0] - [1, 1, 0]) / 2.0 = [-0.5, 0.0, 0.0]
    normalized_pose = normalized[0, 0:132].reshape((33, 4))
    np.testing.assert_almost_equal(normalized_pose[11, :3], [-0.5, 0.0, 0.0], decimal=5)
    assert normalized_pose[11, 3] == 1.0 # visibility unchanged
    
    # Assert normalized Pose right shoulder:
    # coord = ([2, 1, 0] - [1, 1, 0]) / 2.0 = [0.5, 0.0, 0.0]
    np.testing.assert_almost_equal(normalized_pose[12, :3], [0.5, 0.0, 0.0], decimal=5)
    
    # Assert normalized Left Hand first joint:
    # coord = ([5, 5, 5] - [1, 1, 0]) / 2.0 = [2.0, 2.0, 2.5]
    normalized_left_hand = normalized[0, 1536:1599].reshape((21, 3))
    np.testing.assert_almost_equal(normalized_left_hand[0], [2.0, 2.0, 2.5], decimal=5)
    
    # Assert Right Hand is still all zeros (preserved zero-fill padding)
    normalized_right_hand = normalized[0, 1599:1662]
    np.testing.assert_almost_equal(normalized_right_hand, np.zeros(63), decimal=5)
    
    # Test Visibility Masking
    mask = get_visibility_mask(landmarks)
    assert mask.shape == (1, 3)
    assert mask[0, 0] == 1.0 # Pose detected
    assert mask[0, 1] == 1.0 # Left hand detected
    assert mask[0, 2] == 0.0 # Right hand NOT detected
    
    print("Pre-processing unit tests passed successfully!")

if __name__ == "__main__":
    test_landmark_normalization_and_masking()
