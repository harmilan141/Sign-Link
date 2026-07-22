import numpy as np
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from data.augmentations import augment_landmarks_sequence, unpack_landmarks

def test_augmentations():
    # Create a dummy sequence of shape (10, 1662)
    landmarks = np.zeros((10, 1662), dtype=np.float32)
    
    # Populate Left Hand (indices 1536:1599) on the first frame
    landmarks[0, 1536:1539] = [3.0, 4.0, 5.0]
    # Populate Pose left shoulder (indices 44:47)
    landmarks[0, 44:47] = [1.0, 1.0, 1.0]
    
    # Define config to enable only speed warping
    config_warp = {
        "speed_warp_prob": 1.0,
        "speed_warp_min": 1.2,
        "speed_warp_max": 1.2
    }
    warped = augment_landmarks_sequence(landmarks, config_warp)
    # Warped sequence shape: 10 * 1.2 = 12 frames
    assert warped.shape == (12, 1662)
    
    # Define config to enable scaling
    config_scale = {
        "scale_prob": 1.0,
        "scale_min": 1.5,
        "scale_max": 1.5
    }
    scaled = augment_landmarks_sequence(landmarks, config_scale)
    pose_scaled, _, left_scaled, _ = unpack_landmarks(scaled[0])
    np.testing.assert_almost_equal(pose_scaled[11, :3], [1.5, 1.5, 1.5], decimal=5)
    np.testing.assert_almost_equal(left_scaled[0], [4.5, 6.0, 7.5], decimal=5)
    
    # Define config to enable mirroring
    config_mirror = {
        "mirror_prob": 1.0
    }
    mirrored = augment_landmarks_sequence(landmarks, config_mirror)
    pose_mirrored, _, left_mirrored, right_mirrored = unpack_landmarks(mirrored[0])
    # x coordinate negated for shoulder: x = -1.0
    assert pose_mirrored[11, 0] == -1.0
    # Left hand is now zero (since original right hand was zero)
    np.testing.assert_almost_equal(left_mirrored[0], [0.0, 0.0, 0.0])
    # Right hand has the mirrored coordinates of the original left hand: x=-3.0, y=-4.0, z=-5.0 (since hand swap mirrors all)
    np.testing.assert_almost_equal(right_mirrored[0], [-3.0, -4.0, -5.0])
    
    # Define config to enable jitter
    config_jitter = {
        "jitter_prob": 1.0,
        "jitter_std": 0.1
    }
    jittered = augment_landmarks_sequence(landmarks, config_jitter)
    # Ensure zero values (MediaPipe tracking pad flags) are NOT jittered/modified
    assert jittered[0, 1600] == 0.0
    # Ensure non-zero values got jittered
    assert jittered[0, 44] != 1.0
    
    print("Data augmentation unit tests passed successfully!")

if __name__ == "__main__":
    test_augmentations()
