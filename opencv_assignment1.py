# --------------------------------------------------------------
# Assignment 1: Analyse Multiple Features of OpenCV Library
# Lecturer Demonstration Version – 10 image processing commands
# --------------------------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------------------------------------
# STEP 1: Load the image (must be in the same folder as this .py file)
# --------------------------------------------------------------
print("Current working directory:", os.getcwd())  # shows current folder

img = cv2.imread('sample.jpeg')   # updated file extension

# Verify that the image loaded correctly
if img is None:
    print("❌ Error: 'sample.jpeg' not found in this folder.")
    print("Place your image and this file in the same folder, e.g.:")
    print("   assignment1/")
    print("      ├── opencv_assignment1.py")
    print("      └── sample.jpeg")
    exit()
else:
    print("✅ Image loaded successfully!")
    print(f"Image shape: {img.shape}")

# --------------------------------------------------------------
# STEP 2: Define Functions for Each OpenCV Operation
# --------------------------------------------------------------

# 1. Convert image to Grayscale
def to_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Image", gray)

# 2. Apply Gaussian Blur
def apply_blur(image):
    blur = cv2.GaussianBlur(image, (15, 15), 0)
    cv2.imshow("Blurred Image", blur)

# 3. Detect Edges using Canny
def detect_edges(image):
    edges = cv2.Canny(image, 100, 200)
    cv2.imshow("Canny Edge Detection", edges)

# 4. Draw Shapes and Text
def draw_shapes(image):
    img_copy = image.copy()
    cv2.rectangle(img_copy, (50, 50), (200, 200), (0, 255, 0), 3)
    cv2.circle(img_copy, (300, 150), 50, (255, 0, 0), -1)
    cv2.putText(img_copy, "OpenCV Demo", (50, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Shapes and Text", img_copy)

# 5. Apply Binary Thresholding
def threshold_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("Thresholded Image", thresh)

# 6. Resize Image
def resize_image(image):
    resized = cv2.resize(image, (300, 200))
    cv2.imshow("Resized Image (300x200)", resized)

# 7. Rotate Image
def rotate_image(image):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), 45, 1)
    rotated = cv2.warpAffine(image, M, (w, h))
    cv2.imshow("Rotated Image (45°)", rotated)

# 8. Convert to HSV Color Space
def convert_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV Color Space", hsv)

# 9. Apply Color Masking (Red color)
def mask_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 120, 70])    # lower range for red
    upper = np.array([10, 255, 255])  # upper range for red
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow("Masked Image (Red Region)", result)

# 10. Display Histogram of Grayscale Image
def show_histogram(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

# --------------------------------------------------------------
# STEP 3: Create a Command Dictionary
# --------------------------------------------------------------
commands = {
    "gray": to_gray,
    "blur": apply_blur,
    "edges": detect_edges,
    "shapes": draw_shapes,
    "threshold": threshold_image,
    "resize": resize_image,
    "rotate": rotate_image,
    "hsv": convert_hsv,
    "mask": mask_image,
    "hist": show_histogram
}

# --------------------------------------------------------------
# STEP 4: Interactive Command Execution
# --------------------------------------------------------------
print("\n===== OpenCV Feature Analysis =====")
print("Available Commands:")
for key in commands:
    print(f" - {key}")
print("Type 'exit' to quit.\n")

while True:
    cmd = input("Enter command: ").strip().lower()
    if cmd == "exit":
        print("👋 Exiting program... Goodbye!")
        break
    elif cmd in commands:
        cv2.destroyAllWindows()
        commands[cmd](img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ Invalid command. Try again.")

cv2.destroyAllWindows()
