import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
Tk().withdraw()
image_path = askopenfilename(
    title="Select image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)
if not image_path:
    print("No image selected.")
    exit()
image = cv2.imread(image_path)
h, w, _ = image.shape
mask = np.zeros(image.shape[:2], np.uint8)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)
rect = (20, 20, w - 40, h - 40)
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
white_bg = np.ones_like(image) * 255
white_output = np.where(final_mask[:, :, np.newaxis] == 1, image, white_bg)
black_bg = np.zeros_like(image)
black_output = np.where(final_mask[:, :, np.newaxis] == 1, image, black_bg)
green_bg = np.zeros_like(image)
green_bg[:] = (0, 255, 0)
green_output = np.where(final_mask[:, :, np.newaxis] == 1, image, green_bg)
mask_visual = (final_mask * 255).astype(np.uint8)
mask_colored = cv2.cvtColor(mask_visual, cv2.COLOR_GRAY2BGR)
top_row    = cv2.hconcat([image, white_output])
bottom_row = cv2.hconcat([green_output, mask_colored])
combined   = cv2.vconcat([top_row, bottom_row])
cv2.putText(combined, "Original",  (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.putText(combined, "White BG",  (w + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.putText(combined, "Green BG",  (10, h + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.putText(combined, "Mask Only", (w + 10, h + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imwrite("grabcut_output2.jpg", combined)
cv2.imshow("GrabCut Segmentation", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Done!.jpg")