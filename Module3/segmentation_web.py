import cv2
import mediapipe as mp
mp_selfie = mp.solutions.selfie_segmentation
segmentation = mp_selfie.SelfieSegmentation(model_selection=1)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
print("Segmentation Running... Press 'Q' to quit.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = segmentation.process(rgb) 
    mask = results.segmentation_mask > 0.5  
    bg = (0, 255, 0)
    output = frame.copy()
    output[~mask] = bg
    combined = cv2.hconcat([frame, output])
    cv2.putText(combined, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Segmented", (frame.shape[1] + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Segmentation Demo", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
segmentation.close()
print("Closed.")