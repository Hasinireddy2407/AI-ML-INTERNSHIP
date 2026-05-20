import cv2
import mediapipe as mp
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

mp_selfie = mp.solutions.selfie_segmentation
segmentation = mp_selfie.SelfieSegmentation(model_selection=1)

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

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = segmentation.process(rgb)

mask = results.segmentation_mask
binary_mask = (mask > 0.5).astype(np.uint8)

white_bg = np.ones_like(image, dtype=np.uint8) * 255
white_bg_output = np.where(
    binary_mask[:, :, np.newaxis] == 1,
    image,
    white_bg
)
black_bg = np.zeros_like(image, dtype=np.uint8)
black_bg_output = np.where(
    binary_mask[:, :, np.newaxis] == 1,
    image,
    black_bg
)

green_bg = np.zeros_like(image, dtype=np.uint8)
green_bg[:] = (0, 255, 0)
green_bg_output = np.where(
    binary_mask[:, :, np.newaxis] == 1,
    image,
    green_bg
)

mask_visual = (binary_mask * 255).astype(np.uint8)
mask_colored = cv2.cvtColor(mask_visual, cv2.COLOR_GRAY2BGR)

top_row    = cv2.hconcat([image, white_bg_output])
bottom_row = cv2.hconcat([green_bg_output, mask_colored])
combined   = cv2.vconcat([top_row, bottom_row])

cv2.putText(combined, "Original",        (10, 30),      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(combined, "White BG",        (w + 10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(combined, "Green BG",        (10, h + 30),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(combined, "Mask Only",       (w + 10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imwrite("output_combined3.jpg",   combined)

print("All outputs saved!")
print("  output_white_bg.jpg  - Person with white background")
print("  output_black_bg.jpg  - Person with black background")
print("  output_green_bg.jpg  - Person with green background")
print("  output_mask.jpg      - Segmentation mask only")
print("  output_combined.jpg  - All 4 outputs combined")


cv2.imshow("Segmentation Test - All Outputs", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
segmentation.close()
print("Done.")