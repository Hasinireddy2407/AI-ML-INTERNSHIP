import cv2
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=5)
connection_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
LANDMARK_LABELS = {
    mp_pose.PoseLandmark.LEFT_SHOULDER: "L.Shoulder",
    mp_pose.PoseLandmark.RIGHT_SHOULDER: "R.Shoulder",
    mp_pose.PoseLandmark.LEFT_ELBOW: "L.Elbow",
    mp_pose.PoseLandmark.RIGHT_ELBOW: "R.Elbow",
    mp_pose.PoseLandmark.LEFT_WRIST: "L.Wrist",
    mp_pose.PoseLandmark.RIGHT_WRIST: "R.Wrist",
    mp_pose.PoseLandmark.LEFT_HIP: "L.Hip",
    mp_pose.PoseLandmark.RIGHT_HIP: "R.Hip",
    mp_pose.PoseLandmark.LEFT_KNEE: "L.Knee",
    mp_pose.PoseLandmark.RIGHT_KNEE: "R.Knee",
    mp_pose.PoseLandmark.LEFT_ANKLE: "L.Ankle",
    mp_pose.PoseLandmark.RIGHT_ANKLE: "R.Ankle",
}
cap = cv2.VideoCapture(0)

print("Pose Detection Running... Press 'Q' to quit.")

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = pose.process(rgb_frame)
        rgb_frame.flags.writeable = True
        pose_detected = results.pose_landmarks is not None
        status_text = "Pose Detected" if pose_detected else "No Pose"
        status_color = (0, 255, 0) if pose_detected else (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (w, 45), (0, 0, 0), -1)
        cv2.putText(frame, "MediaPipe Pose Detection", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"| {status_text}", (320, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        if pose_detected:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_style,
                connection_drawing_spec=connection_style
            )
            for landmark_id, label in LANDMARK_LABELS.items():
                lm = results.pose_landmarks.landmark[landmark_id]
                if lm.visibility > 0.5:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.putText(frame, label, (cx - 20, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 230, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, "Press Q to quit", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Pose Detection - AI/ML Intern Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()