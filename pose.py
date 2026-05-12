import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Pose Detection Running... Press 'Q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB (MediaPipe needs RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False

    # Run pose detection
    results = pose.process(rgb_frame)

    # Convert back to BGR for OpenCV display
    rgb_frame.flags.writeable = True
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    # Draw skeleton if landmarks detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # Label key landmarks on screen
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        key_points = {
            "L.Shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
            "R.Shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
            "L.Elbow":    mp_pose.PoseLandmark.LEFT_ELBOW,
            "R.Elbow":    mp_pose.PoseLandmark.RIGHT_ELBOW,
            "L.Hip":      mp_pose.PoseLandmark.LEFT_HIP,
            "R.Hip":      mp_pose.PoseLandmark.RIGHT_HIP,
            "L.Knee":     mp_pose.PoseLandmark.LEFT_KNEE,
            "R.Knee":     mp_pose.PoseLandmark.RIGHT_KNEE,
        }

        for label, point in key_points.items():
            lm = landmarks[point]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.putText(frame, label, (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        status = "Pose Detected"
        color = (0, 255, 0)
    else:
        status = "No Pose Detected"
        color = (0, 0, 255)

    # Status bar at top
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (30, 30, 30), -1)
    cv2.putText(frame, f"MediaPipe Pose Detection  |  {status}", (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    # Quit hint at bottom
    cv2.putText(frame, "Press Q to quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.imshow("Pose Detection - AI/ML Intern Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
print("Closed.")
