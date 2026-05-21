import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
PERSON_IMAGE_PATH=r"C:\Users\user\Downloads\person3.png"
CLOTH_IMAGE_PATH=r"C:\Users\user\Downloads\cloth1.png"
OUTPUT_FOLDER=r"C:\Users\user\AI-ML-INTERNSHIP\Computervision\output"
Path(OUTPUT_FOLDER).mkdir(parents=True,exist_ok=True)
def save(name,img):
    path=str(Path(OUTPUT_FOLDER)/name)
    cv2.imwrite(path,img)
    print(f"Saved: {name}")
    return path
person_bgr=cv2.imread(PERSON_IMAGE_PATH)
cloth_bgr=cv2.imread(CLOTH_IMAGE_PATH)
if person_bgr is None:
    print(f"ERROR: Cannot load person image from: {PERSON_IMAGE_PATH}")
    exit(1)
H,W=person_bgr.shape[:2]
save("1_image.png",person_bgr)
try:
    from transformers import SegformerImageProcessor,SegformerForSemanticSegmentation
    import torch
    from torch.nn import functional as F

    PARSE_COLORS={
        0:(0,0,0),
        1:(0,0,128),
        2:(0,0,255),
        3:(51,170,221),
        4:(0,85,255),
        5:(0,128,0),
        6:(85,85,0),
        7:(85,0,85),
        8:(0,51,85),
        9:(0,255,255),
        10:(0,170,255),
        11:(255,0,0),
        12:(170,255,85),
        13:(85,255,170),
        14:(255,255,0),
        15:(221,170,51),
        16:(128,128,128),
        17:(128,0,128),
    }

    UPPER_LABELS=[4,7]
    processor=SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model=SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model.eval()
    pil_img=Image.fromarray(cv2.cvtColor(person_bgr,cv2.COLOR_BGR2RGB))
    inputs=processor(images=pil_img,return_tensors="pt")
    with torch.no_grad():
        outputs=model(**inputs)
        logits=outputs.logits
    upsampled=F.interpolate(logits,size=(H,W),mode="bilinear",align_corners=False)
    seg_map=upsampled.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

    parse_color=np.zeros((H,W,3),dtype=np.uint8)

    for label_id,color in PARSE_COLORS.items():
        parse_color[seg_map==label_id]=color

    save("4_parse.png",parse_color)

    agnostic_mask=np.zeros((H,W),dtype=np.uint8)

    for ul in UPPER_LABELS:
        agnostic_mask[seg_map==ul]=255

    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    agnostic_mask=cv2.dilate(agnostic_mask,kernel,iterations=2)

    save("9_agnostic_mask.png",agnostic_mask)

    parse_agnostic=parse_color.copy()
    parse_agnostic[agnostic_mask>0]=(0,0,0)

    save("3_parse_agnostic.png",parse_agnostic)

    agnostic_img=person_bgr.copy()

    blurred=cv2.GaussianBlur(person_bgr,(51,51),0)

    agnostic_img[agnostic_mask>0]=blurred[agnostic_mask>0]

    agnostic_img[agnostic_mask>0]=(
        agnostic_img[agnostic_mask>0].astype(np.float32)*0.6
    ).astype(np.uint8)

    save("2_agnostic.png",agnostic_img)

except ImportError:
    print("Install transformers torch torchvision")

mp_pose=mp.solutions.pose

pose_model=mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5,
)

rgb=cv2.cvtColor(person_bgr,cv2.COLOR_BGR2RGB)

pose_res=pose_model.process(rgb)

landmarks=pose_res.pose_landmarks

openpose_img=np.zeros((H,W,3),dtype=np.uint8)

if landmarks:
    lms=landmarks.landmark

    CONNECTIONS=[
        (11,12),(11,13),(13,15),
        (12,14),(14,16),
        (11,23),(12,24),
        (23,24),
        (23,25),(25,27),
        (24,26),(26,28),
        (0,11),(0,12),
    ]

    for a,b in CONNECTIONS:
        ax,ay=int(lms[a].x*W),int(lms[a].y*H)
        bx,by=int(lms[b].x*W),int(lms[b].y*H)

        cv2.line(openpose_img,(ax,ay),(bx,by),(200,200,200),3)

    KP_COLORS=[
        (255,0,0),
        (255,85,0),
        (255,170,0),
        (255,255,0),
        (170,255,0),
        (85,255,0),
        (0,255,0),
        (0,255,85),
        (0,255,170),
        (0,255,255),
        (0,170,255),
        (0,85,255),
        (0,0,255),
        (85,0,255),
        (170,0,255),
        (255,0,255),
        (255,0,170),
    ]

    for i,lm in enumerate(lms[:17]):
        if lm.visibility>0.4:
            cx,cy=int(lm.x*W),int(lm.y*H)
            color=KP_COLORS[i%len(KP_COLORS)]

            cv2.circle(openpose_img,(cx,cy),8,color,-1)
            cv2.circle(openpose_img,(cx,cy),8,(255,255,255),2)

save("6_open_pose.png",openpose_img)

dense_pose_img=np.zeros((H,W,3),dtype=np.uint8)

if landmarks and pose_res.segmentation_mask is not None:
    seg_float=pose_res.segmentation_mask
    body_mask=(seg_float>0.5).astype(np.uint8)

    lms=landmarks.landmark

    def px(idx):
        return int(lms[idx].x*W),int(lms[idx].y*H)

    def fill_region(pts,color):
        if len(pts)>=3:
            poly=np.array(pts,dtype=np.int32)
            cv2.fillPoly(dense_pose_img,[poly],color)

    if all(lms[i].visibility>0.3 for i in [11,12,23,24]):
        ls,rs=px(11),px(12)
        lh,rh=px(23),px(24)

        fill_region([ls,rs,rh,lh],(200,100,30))

        head_cy=min(ls[1],rs[1])-int(abs(rs[0]-ls[0])*0.6)
      head_cx=(ls[0]+rs[0])//2

        cv2.ellipse(
            dense_pose_img,
            (head_cx,head_cy),
            (int(abs(rs[0]-ls[0])*0.35),int(abs(rs[0]-ls[0])*0.45)),
            0,
            0,
            360,
            (0,215,255),
            -1
        )

    for side,shoulder_idx,elbow_idx,wrist_idx,color in [
        ("L",11,13,15,(180,80,0)),
        ("R",12,14,16,(0,180,120)),
    ]:
        if all(lms[i].visibility>0.3 for i in [shoulder_idx,elbow_idx]):
            s,e=px(shoulder_idx),px(elbow_idx)
            thick=max(int(abs(s[0]-e[0])*0.5+abs(s[1]-e[1])*0.1),15)
            cv2.line(dense_pose_img,s,e,color,thick)
        if all(lms[i].visibility>0.3 for i in [elbow_idx,wrist_idx]):
            e2,w=px(elbow_idx),px(wrist_idx)
            thick=max(int(abs(e2[0]-w[0])*0.5+abs(e2[1]-w[1])*0.1),12)
            cv2.line(dense_pose_img,e2,w,color,thick-4)
    for side,hip_idx,knee_idx,ankle_idx,color in [
        ("L",23,25,27,(0,100,200)),
        ("R",24,26,28,(50,180,50)),
    ]:
        if all(lms[i].visibility>0.3 for i in [hip_idx,knee_idx]):
            h2p,k=px(hip_idx),px(knee_idx)
            cv2.line(dense_pose_img,h2p,k,color,30)
        if all(lms[i].visibility>0.3 for i in [knee_idx,ankle_idx]):
            k2,a=px(knee_idx),px(ankle_idx)
            cv2.line(dense_pose_img,k2,a,color,25)
    body_3ch=np.stack([body_mask]*3,axis=2)
    dense_pose_img=dense_pose_img*body_3ch
save("5_dense_pose.png",dense_pose_img)
try:
    from rembg import remove as rembg_remove
    cloth_pil=Image.open(CLOTH_IMAGE_PATH).convert("RGBA")
    cloth_no_bg=rembg_remove(cloth_pil)
    cloth_arr=np.array(cloth_no_bg)
    alpha=cloth_arr[:,:,3:4]
    cloth_on_black=np.zeros((cloth_arr.shape[0],cloth_arr.shape[1],3),np.uint8)

    cloth_on_black[:,:,0]=(cloth_arr[:,:,2]*(alpha[:,:,0]/255.0)).astype(np.uint8)
    cloth_on_black[:,:,1]=(cloth_arr[:,:,1]*(alpha[:,:,0]/255.0)).astype(np.uint8)
    cloth_on_black[:,:,2]=(cloth_arr[:,:,0]*(alpha[:,:,0]/255.0)).astype(np.uint8)
    cloth_on_black=cv2.resize(cloth_on_black,(W,H))
    save("7_cloth.png",cloth_on_black)
    cloth_mask_gray=cv2.resize(alpha[:,:,0],(W,H))
    _,cloth_mask_bin=cv2.threshold(cloth_mask_gray,127,255,cv2.THRESH_BINARY)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cloth_mask_bin=cv2.morphologyEx(
        cloth_mask_bin,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )
    cloth_mask_bgr=cv2.cvtColor(cloth_mask_bin,cv2.COLOR_GRAY2BGR)

    save("8_cloth_mask.png",cloth_mask_bgr)
except ImportError:
    print("Install rembg")
