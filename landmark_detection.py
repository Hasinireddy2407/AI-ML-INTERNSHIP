import cv2
import mediapipe as mp
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5
)

def calculate_distance(p1, p2, w, h):
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return round(math.sqrt((x2 - x1)**2 + (y2 - y1)**2), 1)

image_path = r"C:\Users\user\Downloads\person1.png"

image = cv2.imread(image_path)
image = cv2.resize(image, (600, 900))

if image is None:
    print("Image not found. Check your path.")
    exit()

h, w, _ = image.shape
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(rgb)
image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

if results.pose_landmarks:

    
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )

    lm = results.pose_landmarks.landmark


    landmark_names = [
        "Nose", "L.Eye.In", "L.Eye", "L.Eye.Out",
        "R.Eye.In", "R.Eye", "R.Eye.Out",
        "L.Ear", "R.Ear", "Mouth.L", "Mouth.R",
        "L.Shoulder", "R.Shoulder",
        "L.Elbow", "R.Elbow",
        "L.Wrist", "R.Wrist",
        "L.Pinky", "R.Pinky",
        "L.Index", "R.Index",
        "L.Thumb", "R.Thumb",
        "L.Hip", "R.Hip",
        "L.Knee", "R.Knee",
        "L.Ankle", "R.Ankle",
        "L.Heel", "R.Heel",
        "L.Foot", "R.Foot"
    ]

    for i, name in enumerate(landmark_names):
        p = lm[i]
        cx, cy = int(p.x * w), int(p.y * h)
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(image, name, (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 255), 1)

    shoulder = calculate_distance(
        lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
        lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], w, h)

    hip = calculate_distance(
        lm[mp_pose.PoseLandmark.LEFT_HIP],
        lm[mp_pose.PoseLandmark.RIGHT_HIP], w, h)

    arm = calculate_distance(
        lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
        lm[mp_pose.PoseLandmark.LEFT_WRIST], w, h)

    cv2.rectangle(image, (0, 0), (280, 100), (20, 20, 20), -1)
    cv2.putText(image, f"Shoulder : {shoulder} px", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(image, f"Hip      : {hip} px", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(image, f"Arm      : {arm} px", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.putText(image, "All 33 Landmarks Detected", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    output_path = r"C:\Users\user\Downloads\output_module1.png"
    cv2.imwrite(output_path, image)
    print(f"Output saved to: {output_path}")

else:
    print("No pose detected in the image.")
    cv2.putText(image, "No Pose Detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow("Module 1 ", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
pose.close()
print("Done.")