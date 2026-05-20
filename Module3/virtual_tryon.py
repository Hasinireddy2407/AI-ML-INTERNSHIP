import cv2
import mediapipe as mp
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
Tk().withdraw()
print("Select PERSON image...")
person_path = askopenfilename(
    title="Select Person Image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)
print("Select SHIRT/CLOTHING image...")
shirt_path = askopenfilename(
    title="Select Shirt/Clothing Image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)
if not person_path or not shirt_path:
    print("Please select both images.")
    exit()
person = cv2.imread(person_path)
shirt  = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
if person is None or shirt is None:
    print("Could not load images.")
    exit()
h, w, _ = person.shape
mp_pose    = mp.solutions.pose
pose       = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5
)
rgb        = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
results    = pose.process(rgb)
pose.close()
if not results.pose_landmarks:
    print("No pose detected. Use a clear full body image.")
    exit()

lm         = results.pose_landmarks.landmark
l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
l_hip      = lm[mp_pose.PoseLandmark.LEFT_HIP]
r_hip      = lm[mp_pose.PoseLandmark.RIGHT_HIP] 
ls_x = int(l_shoulder.x * w)
ls_y = int(l_shoulder.y * h)
rs_x = int(r_shoulder.x * w)
rs_y = int(r_shoulder.y * h)
lh_y = int(l_hip.y * h)
shirt_width  = int(abs(ls_x - rs_x) * 2.8)
shirt_height = int(abs(lh_y - ls_y) * 1.8)
shirt_x = int((rs_x + ls_x) / 2) - shirt_width // 2
shirt_y = int((ls_y + rs_y) / 2) - int(shirt_height * 0.25)
shirt_x = max(0, shirt_x)
shirt_y = max(0, shirt_y)
if shirt_x + shirt_width > w:
    shirt_width = w - shirt_x
if shirt_y + shirt_height > h:
    shirt_height = h - shirt_y
shirt_resized = cv2.resize(shirt, (shirt_width, shirt_height))
output = person.copy()
if shirt_resized.shape[2] == 4:
    shirt_rgb   = shirt_resized[:, :, :3]
    shirt_alpha = shirt_resized[:, :, 3] / 255.0
    for c in range(3):
        output[shirt_y:shirt_y + shirt_height,
               shirt_x:shirt_x + shirt_width, c] = (
            shirt_alpha * shirt_rgb[:, :, c] +
            (1 - shirt_alpha) * person[shirt_y:shirt_y + shirt_height,
                                       shirt_x:shirt_x + shirt_width, c]
        )
else:
    shirt_gray  = cv2.cvtColor(shirt_resized, cv2.COLOR_BGR2GRAY)
    _, shirt_mask = cv2.threshold(shirt_gray, 240, 255, cv2.THRESH_BINARY_INV)
    shirt_mask  = shirt_mask / 255.0
    for c in range(3):
        output[shirt_y:shirt_y + shirt_height,
               shirt_x:shirt_x + shirt_width, c] = (
            shirt_mask * shirt_resized[:, :, c] +
            (1 - shirt_mask) * person[shirt_y:shirt_y + shirt_height,
                                      shirt_x:shirt_x + shirt_width, c]
        )
cv2.circle(output, (ls_x, ls_y), 6, (0, 255, 0), -1)
cv2.circle(output, (rs_x, rs_y), 6, (0, 255, 0), -1)
cv2.putText(output, "L.Shoulder", (ls_x + 5, ls_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
cv2.putText(output, "R.Shoulder", (rs_x + 5, rs_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
combined = cv2.hconcat([person, output])
cv2.putText(combined, "Original", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.putText(combined, "Try-On Output", (w + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imwrite("tryon_output.jpg", combined)
cv2.imshow("Virtual Try-On Demo", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Done! Output saved as tryon_output.jpg")