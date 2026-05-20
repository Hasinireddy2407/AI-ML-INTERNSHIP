import cv2
import mediapipe as mp
import math
from tkinter import Tk
from tkinter.filedialog import askopenfilename

pose=mp.solutions.pose

def angle(a,b,c):

    ax,ay=a
    bx,by=b
    cx,cy=c

    ab=(ax-bx,ay-by)
    cb=(cx-bx,cy-by)

    dot=ab[0]*cb[0]+ab[1]*cb[1]

    mag1=math.hypot(ab[0],ab[1])
    mag2=math.hypot(cb[0],cb[1])

    if mag1*mag2==0:
        return 0

    cos_angle=dot/(mag1*mag2)

    cos_angle=max(-1,min(1,cos_angle))

    ang=math.degrees(math.acos(cos_angle))

    return int(ang)

Tk().withdraw()

path=askopenfilename(
    title="Select Image",
    filetypes=[("Image Files","*.jpg *.png *.jpeg")]
)

img=cv2.imread(path)

if img is None:
    print("Image not found")

else:

    H,W=img.shape[:2]

    with pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9
    ) as model:

        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        result=model.process(rgb)

        if result.pose_landmarks:

            lm=result.pose_landmarks.landmark

            pts={}

            ids=[11,12,13,14,15,16,23,24,25,26,27,28]

            for i in ids:

                x=int(lm[i].x*W)
                y=int(lm[i].y*H)

                pts[i]=(x,y)

            left_elbow=angle(
                pts[11],
                pts[13],
                pts[15]
            )

            right_elbow=angle(
                pts[12],
                pts[14],
                pts[16]
            )

            left_shoulder=angle(
                pts[13],
                pts[11],
                pts[23]
            )

            right_shoulder=angle(
                pts[14],
                pts[12],
                pts[24]
            )

            left_knee=angle(
                pts[23],
                pts[25],
                pts[27]
            )

            right_knee=angle(
                pts[24],
                pts[26],
                pts[28]
            )

            action="Standing"

            if left_knee<130 or right_knee<130:
                action="Sitting"

            elif left_shoulder>80 and left_shoulder<110 and right_shoulder>80 and right_shoulder<110:
                action="T Pose"

            elif lm[15].y<lm[11].y and lm[16].y<lm[12].y:
                action="Hands Raised"

            elif abs(lm[11].z-lm[12].z)>0.15:
                action="Side Pose"

            lines=[
                (11,12),(11,23),(12,24),(23,24),
                (11,13),(13,15),(12,14),(14,16),
                (23,25),(25,27),(24,26),(26,28)
            ]

            for a,b in lines:

                cv2.line(
                    img,
                    pts[a],
                    pts[b],
                    (255,0,0),
                    3
                )

            for i in ids:

                cv2.circle(
                    img,
                    pts[i],
                    7,
                    (0,255,255),
                    -1
                )

            overlay=img.copy()

            cv2.rectangle(
                overlay,
                (10,10),
                (460,360),
                (0,0,0),
                -1
            )

            img=cv2.addWeighted(
                overlay,
                0.5,
                img,
                0.5,
                0
            )

            info=[
                f"Left Elbow Angle : {left_elbow}",
                f"Right Elbow Angle : {right_elbow}",
                f"Left Shoulder Angle : {left_shoulder}",
                f"Right Shoulder Angle : {right_shoulder}",
                f"Left Knee Angle : {left_knee}",
                f"Right Knee Angle : {right_knee}",
                f"Detected Action : {action}"
            ]

            y=45

            for t in info:

                cv2.putText(
                    img,
                    t,
                    (20,y),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.75,
                    (0,255,255),
                    2
                )

                y+=45

            output_path=r"C:/Users/user/Downloads/pose_output2.jpg"

            cv2.imwrite(output_path,img)

            print(f"\nOutput saved to:\n{output_path}")

        else:
            print("No pose detected")

        cv2.imshow("Angle Pose Detection",img)

        cv2.waitKey(0)

cv2.destroyAllWindows()