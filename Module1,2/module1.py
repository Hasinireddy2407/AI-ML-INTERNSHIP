import cv2
import mediapipe as mp
import math

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def calculate_distance(p1, p2, w, h):
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return round(math.sqrt((x2 - x1)**2 + (y2 - y1)**2), 1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

print("Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:

        lm = results.pose_landmarks.landmark

        landmarks = {
    "Nose"       : mp_pose.PoseLandmark.NOSE,
    "L.Eye"      : mp_pose.PoseLandmark.LEFT_EYE,
    "R.Eye"      : mp_pose.PoseLandmark.RIGHT_EYE,
    "L.Ear"      : mp_pose.PoseLandmark.LEFT_EAR,
    "R.Ear"      : mp_pose.PoseLandmark.RIGHT_EAR,
    "L.Shoulder" : mp_pose.PoseLandmark.LEFT_SHOULDER,
    "R.Shoulder" : mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "L.Elbow"    : mp_pose.PoseLandmark.LEFT_ELBOW,
    "R.Elbow"    : mp_pose.PoseLandmark.RIGHT_ELBOW,
    "L.Hip"      : mp_pose.PoseLandmark.LEFT_HIP,
    "R.Hip"      : mp_pose.PoseLandmark.RIGHT_HIP,
    "L.Knee"     : mp_pose.PoseLandmark.LEFT_KNEE,
    "R.Knee"     : mp_pose.PoseLandmark.RIGHT_KNEE,
    "L.Wrist"    : mp_pose.PoseLandmark.LEFT_WRIST,
    "R.Wrist"    : mp_pose.PoseLandmark.RIGHT_WRIST,
    "L.Ankle"    : mp_pose.PoseLandmark.LEFT_ANKLE,
    "R.Ankle"    : mp_pose.PoseLandmark.RIGHT_ANKLE,
}

        for label, point in landmarks.items():
            p = lm[point]
            cx, cy = int(p.x * w), int(p.y * h)
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
            cv2.putText(frame, label, (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        shoulder_width = calculate_distance(
            lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
            lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], w, h)

        hip_width = calculate_distance(
            lm[mp_pose.PoseLandmark.LEFT_HIP],
            lm[mp_pose.PoseLandmark.RIGHT_HIP], w, h)

        arm_length = calculate_distance(
            lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
            lm[mp_pose.PoseLandmark.LEFT_WRIST], w, h)

        cv2.putText(frame, f"Shoulder : {shoulder_width} px", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Hip      : {hip_width} px", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Arm      : {arm_length} px", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, "Landmark Detected", (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "No Landmark Detected", (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Module 1 - Landmark Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()