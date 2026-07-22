import cv2
import mediapipe as mp
import csv
import os

# ====================================================
# MediaPipe Holistic
# ====================================================

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ====================================================
# Webcam
# ====================================================

cap = cv2.VideoCapture(0)

csv_file = "landmarks.csv"

# ====================================================
# Create CSV Header
# ====================================================

if not os.path.exists(csv_file):

    header = ["label"]

    # ---------------- Face ----------------
    for i in range(468):
        header.append(f"face_x{i}")
        header.append(f"face_y{i}")
        header.append(f"face_z{i}")

    # ---------------- Pose ----------------
    for i in range(33):
        header.append(f"pose_x{i}")
        header.append(f"pose_y{i}")
        header.append(f"pose_z{i}")

    # ---------------- Left Hand ----------------
    for i in range(21):
        header.append(f"left_x{i}")
        header.append(f"left_y{i}")
        header.append(f"left_z{i}")

    # ---------------- Right Hand ----------------
    for i in range(21):
        header.append(f"right_x{i}")
        header.append(f"right_y{i}")
        header.append(f"right_z{i}")

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

print("\n==========================================")
print("Press S -> Save Current Frame")
print("Press Q -> Quit")
print("==========================================\n")

label = "UNKNOWN"

# ====================================================
# Main Loop
# ====================================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = holistic.process(rgb)

    # ====================================================
    # Draw Landmarks
    # ====================================================

    if results.face_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.face_land_landmarks if False else results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION
        )

    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    # ====================================================
    # Create One Row
    # ====================================================

    row = [label]

    # ---------------- Face ----------------

    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])
    else:
        row.extend([0.0] * (468 * 3))

    # ---------------- Pose ----------------

    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])
    else:
        row.extend([0.0] * (33 * 3))

    # ---------------- Left Hand ----------------

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])
    else:
        row.extend([0.0] * (21 * 3))

    # ---------------- Right Hand ----------------

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])
    else:
        row.extend([0.0] * (21 * 3))

    # ====================================================
    # Display
    # ====================================================

    cv2.imshow("Sign-Link Landmark Extractor", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save Sample

    if key == ord("s"):

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print("✅ Sample Saved")

    # Quit

    if key == ord("q"):
        break

# ====================================================
# Cleanup
# ====================================================

cap.release()
cv2.destroyAllWindows()