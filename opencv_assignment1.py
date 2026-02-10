# --------------------------------------------------------------
# Streamlit Web App: Analyse Multiple Features of OpenCV Library
# Lecturer Demonstration Version – 10 image processing commands
# --------------------------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
from PIL import Image

# --------------------------------------------------------------
# Streamlit Page Setup
# --------------------------------------------------------------
st.set_page_config(page_title="OpenCV Feature Analysis", layout="wide")

st.title("🎯 OpenCV Feature Analysis – 10 Image Processing Operations")
st.markdown("Upload an image and explore different OpenCV features interactively!")

# --------------------------------------------------------------
# STEP 1: Upload Image
# --------------------------------------------------------------
uploaded_file = st.file_uploader("📸 Upload an image (JPEG or PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image from uploaded file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)

    st.subheader("Select an operation to apply 👇")

    # --------------------------------------------------------------
    # STEP 2: Define Functions
    # --------------------------------------------------------------

    def to_gray(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_blur(image):
        return cv2.GaussianBlur(image, (15, 15), 0)

    def detect_edges(image):
        return cv2.Canny(image, 100, 200)

    def draw_shapes(image):
        img_copy = image.copy()
        cv2.rectangle(img_copy, (50, 50), (200, 200), (0, 255, 0), 3)
        cv2.circle(img_copy, (300, 150), 50, (255, 0, 0), -1)
        cv2.putText(img_copy, "OpenCV Demo", (50, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img_copy

    def threshold_image(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        return thresh

    def resize_image(image):
        return cv2.resize(image, (300, 200))

    def rotate_image(image):
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), 45, 1)
        return cv2.warpAffine(image, M, (w, h))

    def convert_hsv(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def mask_image(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 120, 70])
        upper = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        return cv2.bitwise_and(image, image, mask=mask)

    def show_histogram(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        fig, ax = plt.subplots()
        ax.plot(hist, color='black')
        ax.set_title("Grayscale Histogram")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    # --------------------------------------------------------------
    # STEP 3: Buttons for Operations
    # --------------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("Grayscale"):
            result = to_gray(img)
            st.image(result, caption="Grayscale Image", use_container_width=True)
    with col2:
        if st.button("Blur"):
            result = apply_blur(img)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Blurred Image", use_container_width=True)
    with col3:
        if st.button("Edges"):
            result = detect_edges(img)
            st.image(result, caption="Edge Detection", use_container_width=True)
    with col4:
        if st.button("Shapes"):
            result = draw_shapes(img)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Shapes & Text", use_container_width=True)
    with col5:
        if st.button("Threshold"):
            result = threshold_image(img)
            st.image(result, caption="Thresholded Image", use_container_width=True)

    col6, col7, col8, col9, col10 = st.columns(5)
    with col6:
        if st.button("Resize"):
            result = resize_image(img)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Resized Image", use_container_width=True)
    with col7:
        if st.button("Rotate"):
            result = rotate_image(img)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Rotated Image", use_container_width=True)
    with col8:
        if st.button("HSV"):
            result = convert_hsv(img)
            st.image(cv2.cvtColor(result, cv2.COLOR_HSV2RGB), caption="HSV Color Space", use_container_width=True)
    with col9:
        if st.button("Mask Red"):
            result = mask_image(img)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Red Color Masked", use_container_width=True)
    with col10:
        if st.button("Histogram"):
            show_histogram(img)

else:
    st.info("👆 Please upload an image file to get started.")
