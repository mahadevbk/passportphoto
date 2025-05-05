import streamlit as st
from PIL import Image, ImageDraw, ImageFont
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

st.sidebar.header("⚙️ Settings")
dpi = st.sidebar.slider("DPI (dots per inch)", 200, 600, 300)
border_mm = st.sidebar.slider("White Border (mm)", 0, 5, 2)
corner_radius = st.sidebar.slider("Polaroid Corner Radius (px)", 0, 300, 20)

uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    original_image = Image.open(uploaded_file).convert("RGB")
    st.image(original_image, caption="Original Image", use_column_width=True)

    auto_center = st.radio("Auto Face Centering", ["Enabled", "Disabled"]) == "Enabled"

    country_options = [f"{country} ({w}x{h} mm)" for country, (w, h) in passport_sizes.items()]
    country_options.append("Custom")
    selection = st.selectbox("Select Country or Custom Size", country_options)

    if selection == "Custom":
        width_mm = st.number_input("Custom Width (mm)", min_value=25, max_value=100, value=35)
        height_mm = st.number_input("Custom Height (mm)", min_value=25, max_value=100, value=45)
    else:
        selected_country = selection.split(" (")[0]
        width_mm, height_mm = passport_sizes[selected_country]

    photo_width_px = mm_to_pixels(width_mm, dpi)
    photo_height_px = mm_to_pixels(height_mm, dpi)
    border_px = mm_to_pixels(border_mm, dpi)

    if auto_center:
        st.info("🔍 Detecting and centering face...")
        cropped_image, found = detect_and_crop_face(original_image)

        if found:
            st.success("✅ Face detected and centered.")
        else:
            st.warning("⚠️ Face not detected. Using original image.")
    else:
        cropped_image = original_image

    img_width, img_height = cropped_image.size
    scale = max(photo_width_px / img_width, photo_height_px / img_height)
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)

    resized_image = cropped_image.resize((new_width, new_height), Image.LANCZOS)

    left = (new_width - photo_width_px) // 2
    top = (new_height - photo_height_px) // 2
    cropped_resized = resized_image.crop((left, top, left + photo_width_px, top + photo_height_px))

    if border_mm > 0:
        final_width = cropped_resized.width + 2 * border_px
        final_height = cropped_resized.height + 2 * border_px
        final_image = Image.new("RGB", (final_width, final_height), "white")
        final_image.paste(cropped_resized, (border_px, border_px))
    else:
        final_image = cropped_resized

    st.subheader("🖼️ Final Passport Photo Preview")
    st.image(final_image, caption="Centered and Bordered", width=300)

    custom_filename = st.text_input("Enter the file name to download (without extension):", value="passport_photo")

    download_option = st.radio("Select Download Option", ["1 Photo", "6 Photos (3x2 grid)", "Polaroid Style"])

    if download_option == "6 Photos (3x2 grid)":
        grid_width = final_image.width * 3
        grid_height = final_image.height * 2
        grid_image = Image.new("RGB", (grid_width, grid_height), "white")

        for i in range(3):
            for j in range(2):
                grid_image.paste(final_image, (i * final_image.width, j * final_image.height))

        img_buffer = io.BytesIO()
        grid_image.save(img_buffer, format="JPEG")
        st.download_button(
            label="🗕️ Download 6 Photos",
            data=img_buffer.getvalue(),
            file_name=f"{custom_filename}_6_photos.jpg",
            mime="image/jpeg"
        )

    elif download_option == "Polaroid Style":
        st.markdown("✏️ Optional: Add a caption below the photo.")
        caption_text = st.text_input("Caption (leave blank for no text):", "")
        caption_font_mm = st.slider("Caption Font Size (mm)", min_value=2, max_value=15, value=12)

        top_border = side_border = border_px
        bottom_border = int(border_px * 6)  # doubled bottom space

        polaroid_width = final_image.width + 2 * side_border
        polaroid_height = final_image.height + top_border + bottom_border

        transparent_bg = Image.new("RGBA", (polaroid_width, polaroid_height), (255, 255, 255, 0))

        # High-resolution mask for anti-aliasing
        scale_factor = 4
        large_size = (rounded_image.size[0]*scale_factor, rounded_image.size[1]*scale_factor)
        large_mask = Image.new("L", large_size, 0)
        draw = ImageDraw.Draw(large_mask)
        draw.rounded_rectangle([(0, 0), large_size], radius=corner_radius*scale_factor, fill=255)

        # Downsample with anti-aliasing
        rounded_mask = large_mask.resize(rounded_image.size, Image.LANCZOS)
        rounded_image.putalpha(rounded_mask)

        #rounded_image = final_image.convert("RGBA")
        #rounded_mask = Image.new("L", rounded_image.size, 0)
        #draw = ImageDraw.Draw(rounded_mask)
        #draw.rounded_rectangle([(0, 0), rounded_image.size], radius=corner_radius, fill=255)
        #rounded_image.putalpha(rounded_mask)

        transparent_bg.paste(rounded_image, (side_border, top_border), mask=rounded_mask)

        if caption_text.strip():
            draw_text = ImageDraw.Draw(transparent_bg)
            try:
                font = ImageFont.truetype("CoveredByYourGrace-Regular.ttf", size=mm_to_pixels(caption_font_mm, dpi))
            except:
                font = ImageFont.load_default()

            text_bbox = draw_text.textbbox((0, 0), caption_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = (transparent_bg.width - text_width) // 2
            text_y = final_image.height + top_border + ((bottom_border - text_height) // 2)
            draw_text.text((text_x, text_y), caption_text, fill="black", font=font)

        polaroid_img = Image.new("RGB", (polaroid_width, polaroid_height), "white")
        alpha_mask = transparent_bg.getchannel("A")
        polaroid_img.paste(transparent_bg.convert("RGB"), (0, 0), mask=alpha_mask)

        st.subheader("🖼️ Polaroid-Style Preview")
        st.image(polaroid_img, caption="Polaroid Output", width=300)

        img_buffer = io.BytesIO()
        polaroid_img.save(img_buffer, format="JPEG")
        st.download_button(
            label="🗕️ Download Polaroid Image",
            data=img_buffer.getvalue(),
            file_name=f"{custom_filename}_polaroid.jpg",
            mime="image/jpeg"
        )

    else:
        img_buffer = io.BytesIO()
        final_image.save(img_buffer, format="JPEG")
        st.download_button(
            label="🗕️ Click to Download",
            data=img_buffer.getvalue(),
            file_name=f"{custom_filename}.jpg",
            mime="image/jpeg"
        )
