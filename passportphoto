import streamlit as st
from PIL import Image
import io

# Country-specific passport photo sizes in mm (width, height)
passport_sizes = {
    "United States": (51, 51),  # 2 x 2 inches
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

def mm_to_pixels(mm, dpi=300):
    return int((mm / 25.4) * dpi)

st.title("📸 Passport Photo Generator")

uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="Original Image", use_column_width=True)

    country = st.selectbox("Select Country", list(passport_sizes.keys()))
    width_mm, height_mm = passport_sizes[country]
    width_px = mm_to_pixels(width_mm)
    height_px = mm_to_pixels(height_mm)

    st.write(f"Passport photo size for **{country}**: {width_mm}mm x {height_mm}mm")

    resized_image = original_image.copy().resize((width_px, height_px))
    st.image(resized_image, caption="Resized Passport Photo Preview", width=250)

    custom_filename = st.text_input("Enter the file name to download (without extension):", value="passport_photo")

    if st.button("Download Photo"):
        img_buffer = io.BytesIO()
        resized_image.save(img_buffer, format="JPEG")
        st.download_button(
            label="Click to Download",
            data=img_buffer.getvalue(),
            file_name=f"{custom_filename}.jpg",
            mime="image/jpeg"
        )
