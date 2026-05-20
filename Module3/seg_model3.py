import cv2
import numpy as np
from rembg import remove
from PIL import Image
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
print("Processing... please wait")
input_image = Image.open(image_path)
output_image = remove(input_image)
output_image.save("rembg_transparent.png")
print("Transparent output saved!")
original_cv = cv2.imread(image_path)
output_cv   = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGBA2BGRA)

h, w, _ = original_cv.shape
white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
alpha    = output_cv[:, :, 3] / 255.0
for c in range(3):
    white_bg[:, :, c] = (alpha * output_cv[:, :, c] +
                         (1 - alpha) * white_bg[:, :, c]).astype(np.uint8)
green_bg = np.zeros((h, w, 3), dtype=np.uint8)
green_bg[:] = (0, 255, 0)
for c in range(3):
    green_bg[:, :, c] = (alpha * output_cv[:, :, c] +
                         (1 - alpha) * green_bg[:, :, c]).astype(np.uint8)

mask_visual  = (alpha * 255).astype(np.uint8)
mask_colored = cv2.cvtColor(mask_visual, cv2.COLOR_GRAY2BGR)
top_row    = cv2.hconcat([original_cv, white_bg])
bottom_row = cv2.hconcat([green_bg, mask_colored])
combined   = cv2.vconcat([top_row, bottom_row])
cv2.putText(combined, "Original",  (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.putText(combined, "White BG",  (w + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.putText(combined, "Green BG",  (10, h + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.putText(combined, "Mask Only", (w + 10, h + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imwrite("rembg_output1.jpg", combined)
cv2.imshow("rembg Segmentation", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Done! Output saved as rembg_output.jpg")