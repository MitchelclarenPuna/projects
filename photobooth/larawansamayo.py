import streamlit as st
import io
import time
import hashlib
from typing import Optional, List
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="Larawan sa Mayo", layout="centered")

#session state initialization
defstate = {
    "photos": [],
    "cur_photo_hash": None,
}
for key, value in defstate.items():
    if key not in st.session_state:
        st.session_state[key] = value

#templates for strips
templates = {
    "White": {
        "title_color": "black",
        "bg_color": "white",
        "border_color": "black",
        "footer_color": "black",
        "frame_width": 420,
        "frame_height": 280,
        "margin": 25,
        "gap": 18,
        "top": 80,
        "bottom": 90,
    },
    "Burgundy": {
        "title_color": "white",
        "bg_color": "#480202",
        "border_color": "black",
        "footer_color": "black",
        "frame_width": 420,
        "frame_height": 280,
        "margin": 25,
        "gap": 18,
        "top": 80,
        "bottom": 90,
    },
    "Pale Pink": {
        "title_color": "black",
        "bg_color": "#993366",
        "border_color": "black",
        "footer_color": "black",
        "frame_width": 420,
        "frame_height": 280,
        "margin": 25,
        "gap": 18,
        "top": 80,
        "bottom": 90,
    },
    "Retro": {
        "title_color": "#f5e6c8",
        "bg_color": "#3b2f2f",
        "border_color": "#3b2f2f",
        "footer_color": "3b2f2f",
        "frame_width": 420,
        "frame_height": 280,
        "margin": 25,
        "gap": 18,
        "top": 80,
        "bottom": 90,
    }
}
def reset_photos():
    st.session_state.photos = []

def get_image_hash(uploaded_image) -> Optional[str]:
    if uploaded_image is None:
        return None
    try:
        file_bytes = uploaded_image.getvalue()
        return hashlib.md5(file_bytes).hexdigest()
    except Exception:
        return None

def lfont(size: int):
    font_options = [
        "monograph-Regular.ttf",
        "Montserrat-Regular.ttf",
    ]
    for font_name in font_options:
        try:
            return ImageFont.truetype(font_name, size)
        except Exception:
            continue
        return ImageFont.load_default()

def pil_convert(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    if fmt.upper() == "JPEG":
        image = image.convert("RGB")
        image.save(buffer, format=fmt)
        return buffer.getvalue()
    
@st.cache_data(show_spinner=False)
def decode(file_bytes: bytes) -> Optional[np.ndarray]:
    try:
        array = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            return None
        return image
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def filter_cache(image_bytes: bytes, filter_name: str) -> Optional[np.ndarray]:
    filt_img = decode(image_bytes)
    if filt_img is None:
        return None
    if filter_name == "Normal":
        return filt_img.copy()
    if filter_name == "Blur":
        return cv2.GaussianBlur(filt_img, (15, 15,), 0)
    if filter_name == "Sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(filt_img, -1, kernel)
    if filter_name == "Edgy":
        gray = cv2.cvtColor(filt_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if filter_name == "Sepia":
        kernel = np.array(
            [[0.272, 0.534, 0.131],
             [0.349, 0.686, 0.168],
             [0.393, 0.769, 0.189],
            ],
            dtype=np.float32,
        )
        sepia = cv2.transform(filt_img, kernel)
        return np.clip(sepia, 0, 255).astype(np.uint8)
    if filter_name == "Retro":
        img = filt_img.astype(np.float32)

        img[:, :, 2] *= 1.2
        img[:, :, 1] *= 1.05
        img[:, :, 0] *= 0.9

        img = img * 0.9 + 20
        
        noise = np.random.normal(0, 8, img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)

        return img
    if filter_name == "Mirror":
        return cv2.flip(filt_img, 1)
    
    return filt_img.copy()

@st.cache_data(show_spinner=False)
def adjust_image(filt_img: np.ndarray, brightness: float, contrast: float)-> np.ndarray:
    img = filt_img.astype(np.float32)
    img *= brightness
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    img = (img - mean)*contrast + mean
    return np.clip(img, 0, 255).astype(np.uint8)

def cv2_convert(filt_img: np.ndarray)-> Image.Image:
    imgrgb = cv2.cvtColor(filt_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(imgrgb)

def caption(img_pil: Image.Image, text: str)-> Image.Image:
    if not text.strip():
        return img_pil
    img_copy = img_pil.copy()
    draw = ImageDraw.Draw(img_copy)
    font = lfont(22)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = max((img_copy.width - text_width)// 2, 10)
    y = max(img_copy.height - text_height - 24, 10)
    pad = 10

    draw.rounded_rectangle(
        (x - pad, y - pad, x+text_width + pad, y + text_height + pad),
        radius = 5,
        fill = (0, 0, 0)
    )
    draw.text((x, y), text, fill="white", font=font)
    return img_copy

@st.cache_data(show_spinner=False)
def crop_img(img_bytes: bytes, target_width: int, target_height: int)-> Image.Image:
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    src_width, src_height = image.size

    src_ratio = src_width / src_height
    target_ratio = target_width / target_height

    if src_ratio > target_ratio:
        new_height = target_height
        new_width = int(new_height * src_ratio)
    else: 
        new_width = target_width
        new_height = int(new_width / src_ratio)

    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left = (new_width - target_width)// 2
    top = (new_height - target_height)// 2
    return image.crop((left, top, left + target_width, top + target_height))

@st.cache_data(show_spinner=False)
def fstrip(
    img_bytes_ls: tuple[bytes, ...],
    strip_title: str,
    footer_text: str,
    template_name: str,
) -> Image.Image:
    tmps = templates[template_name]

    frame_width = tmps["frame_width"]
    frame_height = tmps["frame_height"]
    margin = tmps["margin"]
    gap = tmps["gap"]
    top = tmps["top"]
    bottom = tmps["bottom"]

    okimgs = [
        crop_img(image_bytes, frame_width, frame_height)
        for image_bytes in img_bytes_ls
    ]
    num_imgs = len(okimgs)
    strip_width = frame_width + (margin * 2)
    strip_height = top + (frame_height * num_imgs) + (gap * (num_imgs - 1)) + bottom

    strip = Image.new("RGB", (strip_width, strip_height), tmps["bg_color"])
    draw = ImageDraw.Draw(strip)

    title_font = lfont(28)
    footer_font = lfont(18)

    if strip_title.strip():
        title_bbox = draw.textbbox((0, 0), strip_title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text(
            ((strip_width - title_width)// 2, 24),
            strip_title, fill=tmps["title_color"], 
            font=title_font,
        )
        y = top
        for img in okimgs:
            strip.paste(img, (margin, y))
            draw.rectangle(
                (margin, y, margin + frame_width, y + frame_height),
                outline=tmps["border_color"],
                width=3,
            )
            y += frame_height + gap

            if footer_text.strip():
                footer_bbox = draw.textbbox((0, 0), footer_text, font=footer_font)
                footer_width = footer_bbox[2] - footer_bbox[0]
                draw.text(
                    ((strip_width - footer_width) // 2, strip_height - 40),
                    footer_text,
                    fill=tmps["footer_color"],
                    font=footer_font,
                )
    return strip

def okimgs_cur(
        file_bytes: bytes,
        filter_name: str,
        brightness: float,
        contrast: float,
        text: str,
)-> Optional[Image.Image]:
    filtered = filter_cache(file_bytes, filter_name)
    if filtered is None:
        return None
    
    adjusted = adjust_image(filtered, brightness, contrast)
    editedimg = cv2_convert(adjusted)
    editedimg = caption(editedimg, text)
    return editedimg

def img_convert(image: Image.Image)-> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

with st.sidebar:
    st.header("Captured in May")
    source = st.radio("Which will you use?", ["Camera", "Upload Image"])
    
    no_shots = st.number_input("Number of shots", min_value=1, max_value=8, value=4)

#here's the header and sidebar
#the photobooth main
page = st.sidebar.selectbox("Go to", ["Photobooth", "About"])
if page == "Photobooth":
    st.title("Larawan Sa Mayo")
    st.caption("Captured from your lens")
    
    with st.sidebar:
        st.header("Filter Window")
    
        no_shots = st.selectbox("How many shots?", [2, 3, 4], index=1)
        template_name = st.selectbox("Which template do you like?", list(templates.keys()), index=0)

        with st.form("filter_form"):
            source = st.radio("Which will you use?", ["Camera", "Upload Image"])

            filter_options = st.selectbox(
            "Filters",
            ["Normal", "Blur", "Sharpen", "Edgy", "Sepia", "Retro", "Mirror"],
            )
            brightness = st.slider("Brightness:", 0.5, 2.0, 1.0, 0.1)
            contrast = st.slider("Contrast:", 0.5, 2.0, 1.0, 0.1)

            text = st.text_input(
                "Write a caption",
                placeholder="Feel free to put anything you like!"
            )
            strip_title = st.text_input("Title", value="Larawan sa Mayo")
            footer_text = st.text_input("Footer", value="From your lens")

            submitted = st.form_submit_button("Apply Settings", use_container_width=True)

        if st.button("Reset All Photos", use_container_width=True, type="secondary"):
            reset_photos()
            st.success("All saved photos have been cleared.")
            st.rerun()

        st.divider()
        st.markdown("### Preview your vibe")
        chosen_tmp = templates[template_name]
        st.write(f"**Background:** `{chosen_tmp['bg_color']}`")
        st.write(f"**Frame size:** `{chosen_tmp['frame_width']}px` × `{chosen_tmp['frame_height']}px`")

#my about page section
else:
    st.title("About Larawan Sa Mayo")
    
    st.markdown("""
    ### The Project
    **Larawan Sa Mayo** is built as an activity for Integrative Programming & Technologies course.
    This web application is inspired by the series of photobooths me and my partner went to before we got together.
    The app acts as a camera that incorporates features of a photobooth and captures moments spent together without the need to go out, 
    making the photobooth experience more accessible wherever you are.
    This web application  targets various users be it couples, friends, or solo individuals who are interested in taking pictures or uploading images to add a vintage photobooth feel to it,
    while also being able to include captions that preserve their moments. This web application accepts for camera, and image input. 
    After processing, it displays the final image and allows the user to download it.

    I used the following libraries to create a simple photobooth web application:
    * **Streamlit:** This serves as the front-end framework of my project. I used this for the controls, user inputs, and displaying result.
    * **OpenCV:** As this is a photobooth web app, I had to use a computer vision for converting the color and applying filters on the images.
    * **Pillow (PIL):** This library is useful for typography and stitching multiple images together.
    * **NumPy:** is for efficient matrix operations on image data such as pixel manipulation. An example of this is adjusting the brightness of the image.
    
    Mitchelclaren G. Puna
    ICS-01-402A
    ---
    """)
    
    st.info("Version 1.0.0 | Developed in 2026")

#inputs
if source == "Camera":
    cam_key = f"cam_input_{len(st.session_state.photos)}"
    uploaded_source = st.camera_input("Capture the moment", key=cam_key)
else:
    uploaded_source = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="file_upload"
    )

if uploaded_source is not None:
    current_file_bytes = uploaded_source.getvalue()
    
    current_preview = okimgs_cur(
        current_file_bytes,
        filter_options,
        brightness,
        contrast,
        text,
    )

    if current_preview is not None:
        st.subheader("Preview")
        st.image(current_preview, use_container_width=True)
        
        st.info(f"Shots saved: {len(st.session_state.photos)} / {no_shots}")

        if len(st.session_state.photos) < no_shots:
            if st.button(f"Save Shot {len(st.session_state.photos) + 1}", use_container_width=True):
                st.session_state.photos.append(current_preview.copy())
                st.success("Shot saved successfully!")
                st.rerun()
        else:
            st.warning("You have reached the limit. Scroll down to see your final strip!")

#strip generation
if len(st.session_state.photos) == no_shots:
    with st.spinner("Stitching the look..."):
        photo_bytes = tuple(img_convert(photos) for photos in st.session_state.photos)
        fin_look = fstrip(
            photo_bytes,
            strip_title=strip_title,
            footer_text=footer_text,
            template_name=template_name,
        )
        
        st.subheader("Final Look")
        st.image(fin_look, use_container_width=True)
        st.success("Behold! Your best angles.")
        
        import io
        st.divider()
        dl_format = st.radio("Download format", ["PNG", "JPEG"], horizontal=True)

        buf = io.BytesIO()
        if dl_format == "PNG":
            file_name = "larawan_sa_mayo.png"
            mime = "image/png"
            fin_look.save(buf, format="PNG")
        else:
            file_name = "larawan_sa_mayo.jpg"
            mime = "image/jpeg"
            if fin_look.mode in ("RGBA", "P"):
                fin_look = fin_look.convert("RGB")
            fin_look.save(buf, format="JPEG")
        
        file_data = buf.getvalue()

        st.download_button(
            label=f"Download as {dl_format}",
            data=file_data, 
            file_name=file_name,
            mime=mime,
            use_container_width=True,
        )
else:
    st.warning(f"Save your {no_shots} best looks to see them all together.")