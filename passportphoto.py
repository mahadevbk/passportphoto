import streamlit as st
from PIL import Image
import cv2
import numpy as np
import io

# --- Passport sizes in mm (width, height) ---
passport_sizes = {
    "United States": (51, 51),
    "United Kingdom": (35, 45),
    "Europe (Schengen)": (35, 45),
    "India": (35, 45),
    "China": (33, 48),
    "Hong Kong": (40, 50),
    "Singapore": (35, 45),
    "Malaysia": (35, 50),
    "Australia": (35, 45),
    "Middle East (General)": (40, 50),
    "Other": (35, 45)
}

# --- MM to pixels conversion ---
def mm_to_pixels(mm, dpi=300):
    return int((mm / 25.4) * dpi)

# --- Detect and crop around face ---
def detect_and_crop_face(pil_image):
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return pil_image, False

    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    padding = 2.0
    cx, cy = x + w // 2, y + h // 2
    half_side = int(max(w, h) * padding / 2)

    left = max(cx - half_side, 0)
    top = max(cy - half_side, 0)
    right = min(cx + half_side, cv_image.shape[1])
    bottom = min(cy + half_side, cv_image.shape[0])

    face_crop = pil_image.crop((left, top, right, bottom))
    return face_crop, True

# --- Streamlit UI ---
st.set_page_config(page_title="Passport Photo Generator", layout="centered")
st.title("📸 Passport Photo Generator with Auto Face Centering")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Settings")
dpi = st.sidebar.slider("DPI (dots per inch)", 200, 600, 300)
border_mm = st.sidebar.slider("White Border (mm)", 2, 5, 2)

uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    original_image = Image.open(uploaded_file).convert("RGB")
    st.image(original_image, caption="Original Image", use_column_width=True)

    # Display countries with size in dropdown
    country_options = [
        f"{country} ({w}x{h} mm)" for country, (w, h) in passport_sizes.items()
    ]
    country_options.append("Custom")

    selection = st.selectbox("Select Country or Custom Size", country_options)

    if selection == "Custom":
        width_mm = st.number_input("Custom Width (mm)", min_value=25, max_value=100, value=35)
        height_mm = st.number_input("Custom Height (mm)", min_value=25, max_value=100, value=45)
    else:
        # Extract the country name from the dropdown selection
        selected_country = selection.split(" (")[0]  # This removes the size part
        width_mm, height_mm = passport_sizes[selected_country]

    # Convert dimensions
    photo_width_px = mm_to_pixels(width_mm, dpi)
    photo_height_px = mm_to_pixels(height_mm, dpi)
    border_px = mm_to_pixels(border_mm, dpi)

    st.info("🔍 Detecting and centering face...")
    cropped_image, found = detect_and_crop_face(original_image)

    if found:
        st.success("✅ Face detected and centered.")
    else:
        st.warning("⚠️ Face not detected. Using original image.")

    # Resize with aspect ratio preserved
    img_ratio = cropped_image.width / cropped_image.height
    target_ratio = photo_width_px / photo_height_px

    if img_ratio > target_ratio:
        new_width = photo_width_px
        new_height = int(photo_width_px / img_ratio)
    else:
        new_height = photo_height_px
        new_width = int(photo_height_px * img_ratio)

    resized_image = cropped_image.resize((new_width, new_height), Image.LANCZOS)

    # --- Create final image with white border ---
    # Create the passport-sized canvas for image (without border)
    passport_canvas = Image.new("RGB", (photo_width_px, photo_height_px), "white")
    paste_position = (
        (photo_width_px - new_width) // 2,
        (photo_height_px - new_height) // 2
    )
    passport_canvas.paste(resized_image, paste_position)

    # Add the white border around the image (both top and bottom should be equal)
    final_width = passport_canvas.width + 2 * border_px
    final_height = passport_canvas.height + 2 * border_px

    final_image = Image.new("RGB", (final_width, final_height), "white")
    final_image.paste(passport_canvas, (border_px, border_px))

    st.subheader("🖼️ Final Passport Photo Preview")
    st.image(final_image, caption="Centered and Bordered", width=300)

    custom_filename = st.text_input("Enter the file name to download (without extension):", value="passport_photo")

    if st.button("Download Photo"):
        img_buffer = io.BytesIO()
        final_image.save(img_buffer, format="JPEG")
        st.download_button(
            label="📥 Click to Download",
            data=img_buffer.getvalue(),
            file_name=f"{custom_filename}.jpg",
            mime="image/jpeg"
        )
