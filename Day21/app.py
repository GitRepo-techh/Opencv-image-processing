import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from image_processing import (
    to_grayscale,
    apply_blur,
    detect_edges,
    rotate_image,
    adjust_brightness_contrast,
    detect_contours,
    detect_shapes,
    scan_document,
    sharpen,
    flip_image,
    apply_threshold,
    apply_sepia,
    detect_color,
    detect_faces,
    detect_faces_and_eyes
)

from analytics import (
    get_image_statistics,
    grayscale_histogram,
    rgb_histogram,
    compare_histograms,
    get_rgb_channels,
    get_hsv_channels,
    calculate_edge_density,
    get_color_means
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Aperture",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# SESSION STATE
# ============================================================

if "original_image" not in st.session_state:
    st.session_state.original_image = None

if "processed_image" not in st.session_state:
    st.session_state.processed_image = None

if "undo_stack" not in st.session_state:
    st.session_state.undo_stack = []

if "redo_stack" not in st.session_state:
    st.session_state.redo_stack = []

if "history" not in st.session_state:
    st.session_state.history = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = []


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def image_to_bytes(img, extension=".png"):

    success, encoded = cv2.imencode(
        extension,
        img
    )

    if not success:
        return None

    return encoded.tobytes()


def load_uploaded_image(uploaded_file):

    image_bytes = uploaded_file.read()

    pil_image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    rgb = np.array(
        pil_image
    )

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    return bgr


def save_state_for_undo():

    if st.session_state.processed_image is not None:

        st.session_state.undo_stack.append(
            st.session_state.processed_image.copy()
        )

        st.session_state.redo_stack = []


def add_history(operation):

    st.session_state.history.append(
        operation
    )


def reset_image():

    if st.session_state.original_image is not None:

        st.session_state.processed_image = (
            st.session_state.original_image.copy()
        )

        st.session_state.undo_stack = []
        st.session_state.redo_stack = []
        st.session_state.history = []


def undo():

    if len(st.session_state.undo_stack) == 0:
        return

    current = st.session_state.processed_image.copy()

    st.session_state.redo_stack.append(
        current
    )

    st.session_state.processed_image = (
        st.session_state.undo_stack.pop()
    )


def redo():

    if len(st.session_state.redo_stack) == 0:
        return

    current = st.session_state.processed_image.copy()

    st.session_state.undo_stack.append(
        current
    )

    st.session_state.processed_image = (
        st.session_state.redo_stack.pop()
    )


def display_image(img, caption):

    rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    st.image(
        rgb,
        caption=caption,
        use_container_width=True
    )


# ============================================================
# HEADER
# ============================================================

st.title("◈ Aperture")

st.subheader(
    "Computer Vision Studio"
)

st.write(
    "Analyze, transform, and understand images using Python and OpenCV."
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.header("📥 Input")

input_tab1, input_tab2 = st.tabs(
    [
        "📁 Upload Image",
        "📷 Camera"
    ]
)


with input_tab1:

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
            "bmp"
        ]
    )

    if uploaded_file is not None:

        file_id = (
            uploaded_file.name
            + str(uploaded_file.size)
        )

        if st.session_state.get(
            "loaded_file_id"
        ) != file_id:

            image = load_uploaded_image(
                uploaded_file
            )

            st.session_state.original_image = (
                image.copy()
            )

            st.session_state.processed_image = (
                image.copy()
            )

            st.session_state.undo_stack = []
            st.session_state.redo_stack = []
            st.session_state.history = []
            st.session_state.pipeline = []

            st.session_state.loaded_file_id = (
                file_id
            )

            st.success(
                "Image loaded successfully."
            )


with input_tab2:

    camera_image = st.camera_input(
        "Take a picture"
    )

    if camera_image is not None:

        file_id = (
            "camera_"
            + str(camera_image.size)
        )

        if st.session_state.get(
            "camera_file_id"
        ) != file_id:

            image = load_uploaded_image(
                camera_image
            )

            st.session_state.original_image = (
                image.copy()
            )

            st.session_state.processed_image = (
                image.copy()
            )

            st.session_state.undo_stack = []
            st.session_state.redo_stack = []
            st.session_state.history = []
            st.session_state.pipeline = []

            st.session_state.camera_file_id = (
                file_id
            )

            st.success(
                "Camera image captured."
            )


# ============================================================
# CHECK IMAGE
# ============================================================

if st.session_state.original_image is None:

    st.info(
        "Upload an image or capture one with your camera to begin."
    )

    st.stop()


original = st.session_state.original_image
processed = st.session_state.processed_image


# ============================================================
# IMAGE INFORMATION
# ============================================================

stats = get_image_statistics(
    processed
)

st.header("📋 Image Information")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Width",
    f"{stats['Width']} px"
)

c2.metric(
    "Height",
    f"{stats['Height']} px"
)

c3.metric(
    "Channels",
    stats["Channels"]
)

c4.metric(
    "Brightness",
    f"{stats['Mean Brightness']:.1f}"
)


# ============================================================
# OPERATIONS
# ============================================================

st.header("🛠️ Operations")

operation = st.selectbox(
    "Select an operation",
    [
        "Grayscale",
        "Blur",
        "Edge Detection",
        "Rotation",
        "Brightness/Contrast",
        "Contour Detection",
        "Shape Detection",
        "Document Scanner",
        "Sharpen",
        "Flip",
        "Threshold",
        "Sepia",
        "Color Detection",
        "Face Detection",
        "Face + Eye Detection"
    ]
)


# ============================================================
# PARAMETERS
# ============================================================

st.subheader("⚙️ Parameters")


# ---------- GRAYSCALE ----------

if operation == "Grayscale":

    st.write(
        "Convert the image to grayscale."
    )


# ---------- BLUR ----------

elif operation == "Blur":

    blur_method = st.selectbox(
        "Blur method",
        [
            "gaussian",
            "median",
            "bilateral"
        ]
    )

    blur_kernel = st.slider(
        "Kernel size",
        min_value=1,
        max_value=31,
        value=5,
        step=2
    )

    blur_sigma = st.slider(
        "Gaussian sigma",
        min_value=0,
        max_value=20,
        value=0
    )


# ---------- EDGES ----------

elif operation == "Edge Detection":

    edge_threshold1 = st.slider(
        "Lower threshold",
        0,
        500,
        100
    )

    edge_threshold2 = st.slider(
        "Upper threshold",
        0,
        500,
        200
    )


# ---------- ROTATION ----------

elif operation == "Rotation":

    rotation_angle = st.slider(
        "Angle",
        -180,
        180,
        0
    )

    rotation_scale = st.slider(
        "Scale",
        0.1,
        3.0,
        1.0,
        0.1
    )


# ---------- BRIGHTNESS / CONTRAST ----------

elif operation == "Brightness/Contrast":

    contrast = st.slider(
        "Contrast",
        0.1,
        3.0,
        1.0,
        0.1
    )

    brightness = st.slider(
        "Brightness",
        -255,
        255,
        0
    )


# ---------- CONTOURS ----------

elif operation == "Contour Detection":

    contour_threshold = st.slider(
        "Threshold",
        0,
        255,
        127
    )

    draw_all_contours = st.checkbox(
        "Detect all contours",
        value=True
    )


# ---------- SHAPES ----------

elif operation == "Shape Detection":

    shape_threshold = st.slider(
        "Threshold",
        0,
        255,
        127
    )


# ---------- SCANNER ----------

elif operation == "Document Scanner":

    adaptive_threshold = st.checkbox(
        "Apply adaptive threshold",
        value=True
    )


# ---------- FLIP ----------

elif operation == "Flip":

    flip_direction = st.selectbox(
        "Direction",
        [
            "horizontal",
            "vertical",
            "both"
        ]
    )


# ---------- THRESHOLD ----------

elif operation == "Threshold":

    threshold_method = st.selectbox(
        "Method",
        [
            "binary",
            "otsu"
        ]
    )

    threshold_value = st.slider(
        "Threshold value",
        0,
        255,
        127
    )


# ---------- COLOR DETECTION ----------

elif operation == "Color Detection":

    color_choice = st.selectbox(
        "Quick color preset",
        [
            "Red",
            "Green",
            "Blue",
            "Custom"
        ]
    )

    if color_choice == "Red":

        lower_hsv = [0, 100, 100]
        upper_hsv = [10, 255, 255]

    elif color_choice == "Green":

        lower_hsv = [35, 50, 50]
        upper_hsv = [85, 255, 255]

    elif color_choice == "Blue":

        lower_hsv = [90, 50, 50]
        upper_hsv = [140, 255, 255]

    else:

        st.write("Custom HSV range")

        lower_h = st.slider(
            "Lower Hue",
            0,
            179,
            0
        )

        lower_s = st.slider(
            "Lower Saturation",
            0,
            255,
            50
        )

        lower_v = st.slider(
            "Lower Value",
            0,
            255,
            50
        )

        upper_h = st.slider(
            "Upper Hue",
            0,
            179,
            179
        )

        upper_s = st.slider(
            "Upper Saturation",
            0,
            255,
            255
        )

        upper_v = st.slider(
            "Upper Value",
            0,
            255,
            255
        )

        lower_hsv = [
            lower_h,
            lower_s,
            lower_v
        ]

        upper_hsv = [
            upper_h,
            upper_s,
            upper_v
        ]


# ============================================================
# APPLY OPERATION
# ============================================================

if st.button(
    "▶ Apply Operation",
    type="primary",
    use_container_width=True
):

    save_state_for_undo()

    try:

        with st.spinner(
            f"Applying {operation}..."
        ):

            if operation == "Grayscale":

                result = to_grayscale(
                    processed
                )

                history_text = (
                    "Grayscale"
                )


            elif operation == "Blur":

                result = apply_blur(
                    processed,
                    method=blur_method,
                    ksize=blur_kernel,
                    sigma=blur_sigma
                )

                history_text = (
                    f"Blur | "
                    f"{blur_method} | "
                    f"kernel={blur_kernel}"
                )


            elif operation == "Edge Detection":

                result = detect_edges(
                    processed,
                    edge_threshold1,
                    edge_threshold2
                )

                history_text = (
                    f"Edge Detection | "
                    f"{edge_threshold1}/{edge_threshold2}"
                )


            elif operation == "Rotation":

                result = rotate_image(
                    processed,
                    rotation_angle,
                    rotation_scale
                )

                history_text = (
                    f"Rotation | "
                    f"{rotation_angle}° | "
                    f"scale={rotation_scale}"
                )


            elif operation == "Brightness/Contrast":

                result = adjust_brightness_contrast(
                    processed,
                    contrast,
                    brightness
                )

                history_text = (
                    f"Brightness/Contrast | "
                    f"brightness={brightness} | "
                    f"contrast={contrast}"
                )


            elif operation == "Contour Detection":

                result, contour_count = detect_contours(
                    processed,
                    contour_threshold,
                    draw_all_contours
                )

                history_text = (
                    f"Contour Detection | "
                    f"{contour_count} contours"
                )


            elif operation == "Shape Detection":

                result, shape_counts = detect_shapes(
                    processed,
                    shape_threshold
                )

                history_text = (
                    "Shape Detection"
                )

                st.session_state.last_shape_counts = (
                    shape_counts
                )


            elif operation == "Document Scanner":

                result = scan_document(
                    processed,
                    adaptive_threshold
                )

                history_text = (
                    "Document Scanner"
                )


            elif operation == "Sharpen":

                result = sharpen(
                    processed
                )

                history_text = (
                    "Sharpen"
                )


            elif operation == "Flip":

                result = flip_image(
                    processed,
                    flip_direction
                )

                history_text = (
                    f"Flip | {flip_direction}"
                )


            elif operation == "Threshold":

                result = apply_threshold(
                    processed,
                    threshold_value,
                    threshold_method
                )

                history_text = (
                    f"Threshold | "
                    f"{threshold_method}"
                )


            elif operation == "Sepia":

                result = apply_sepia(
                    processed
                )

                history_text = (
                    "Sepia"
                )


            elif operation == "Color Detection":

                result, mask = detect_color(
                    processed,
                    lower_hsv,
                    upper_hsv
                )

                history_text = (
                    f"Color Detection | "
                    f"{color_choice}"
                )

                st.session_state.last_mask = mask


            elif operation == "Face Detection":

                result, face_count = detect_faces(
                    processed
                )

                history_text = (
                    f"Face Detection | "
                    f"{face_count} faces"
                )

                st.session_state.last_face_count = (
                    face_count
                )


            elif operation == "Face + Eye Detection":

                (
                    result,
                    face_count,
                    eye_count
                ) = detect_faces_and_eyes(
                    processed
                )

                history_text = (
                    f"Face + Eye Detection | "
                    f"{face_count} faces | "
                    f"{eye_count} eyes"
                )

                st.session_state.last_face_count = (
                    face_count
                )

                st.session_state.last_eye_count = (
                    eye_count
                )


            st.session_state.processed_image = (
                result
            )

            add_history(
                history_text
            )

        st.success(
            f"{operation} applied successfully."
        )

    except Exception as error:

        st.error(
            f"Processing error: {error}"
        )


# Refresh processed image reference
processed = st.session_state.processed_image


# ============================================================
# SPECIAL RESULTS
# ============================================================

if "last_shape_counts" in st.session_state:

    if operation == "Shape Detection":

        st.subheader(
            "🔺 Shape Analysis"
        )

        shape_counts = (
            st.session_state.last_shape_counts
        )

        columns = st.columns(5)

        columns[0].metric(
            "Triangles",
            shape_counts["Triangle"]
        )

        columns[1].metric(
            "Squares",
            shape_counts["Square"]
        )

        columns[2].metric(
            "Rectangles",
            shape_counts["Rectangle"]
        )

        columns[3].metric(
            "Pentagons",
            shape_counts["Pentagon"]
        )

        columns[4].metric(
            "Circles",
            shape_counts["Circle"]
        )


if operation == "Face Detection":

    if "last_face_count" in st.session_state:

        st.metric(
            "Faces Detected",
            st.session_state.last_face_count
        )


if operation == "Face + Eye Detection":

    if "last_face_count" in st.session_state:

        c1, c2 = st.columns(2)

        c1.metric(
            "Faces",
            st.session_state.last_face_count
        )

        c2.metric(
            "Eyes",
            st.session_state.last_eye_count
        )


# ============================================================
# BEFORE / AFTER
# ============================================================

st.divider()

st.header("🖼️ Before / After")

before_col, after_col = st.columns(2)

with before_col:

    display_image(
        original,
        "Original"
    )

with after_col:

    display_image(
        processed,
        "Processed"
    )


# ============================================================
# ANALYTICS
# ============================================================

st.divider()

st.header("📊 Image Analytics")

analytics_tab1, analytics_tab2, analytics_tab3, analytics_tab4 = st.tabs(
    [
        "📈 Histograms",
        "📋 Statistics",
        "🎨 Color Analysis",
        "🔬 Edge Analysis"
    ]
)


# ============================================================
# HISTOGRAMS
# ============================================================

with analytics_tab1:

    st.subheader(
        "Grayscale Histogram"
    )

    gray_fig = grayscale_histogram(
        processed
    )

    st.pyplot(
        gray_fig,
        use_container_width=True
    )

    st.subheader(
        "RGB Histogram"
    )

    rgb_fig = rgb_histogram(
        processed
    )

    st.pyplot(
        rgb_fig,
        use_container_width=True
    )

    st.subheader(
        "Original vs Processed"
    )

    comparison_fig = compare_histograms(
        original,
        processed
    )

    st.pyplot(
        comparison_fig,
        use_container_width=True
    )


# ============================================================
# STATISTICS
# ============================================================

with analytics_tab2:

    stats = get_image_statistics(
        processed
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Resolution",
        f"{stats['Width']} × {stats['Height']}"
    )

    c2.metric(
        "Mean Brightness",
        f"{stats['Mean Brightness']:.2f}"
    )

    c3.metric(
        "Contrast",
        f"{stats['Standard Deviation']:.2f}"
    )

    st.subheader(
        "Pixel Statistics"
    )

    stat_data = {
        "Metric": [
            "Width",
            "Height",
            "Channels",
            "Mean Brightness",
            "Standard Deviation",
            "Minimum Pixel",
            "Maximum Pixel",
            "Mean Blue",
            "Mean Green",
            "Mean Red"
        ],
        "Value": [
            stats["Width"],
            stats["Height"],
            stats["Channels"],
            round(
                stats["Mean Brightness"],
                2
            ),
            round(
                stats["Standard Deviation"],
                2
            ),
            stats["Minimum Pixel"],
            stats["Maximum Pixel"],
            round(
                stats["Mean Blue"],
                2
            ),
            round(
                stats["Mean Green"],
                2
            ),
            round(
                stats["Mean Red"],
                2
            )
        ]
    }

    st.table(
        stat_data
    )


# ============================================================
# COLOR ANALYSIS
# ============================================================

with analytics_tab3:

    st.subheader(
        "RGB Channel Analysis"
    )

    blue, green, red = get_rgb_channels(
        processed
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.image(
            blue,
            caption="Blue Channel",
            use_container_width=True
        )

    with c2:
        st.image(
            green,
            caption="Green Channel",
            use_container_width=True
        )

    with c3:
        st.image(
            red,
            caption="Red Channel",
            use_container_width=True
        )

    means = get_color_means(
        processed
    )

    st.subheader(
        "Average Channel Values"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Blue",
        f"{means['Blue']:.2f}"
    )

    c2.metric(
        "Green",
        f"{means['Green']:.2f}"
    )

    c3.metric(
        "Red",
        f"{means['Red']:.2f}"
    )

    st.subheader(
        "HSV Channels"
    )

    hue, saturation, value = get_hsv_channels(
        processed
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.image(
            hue,
            caption="Hue",
            use_container_width=True
        )

    with c2:
        st.image(
            saturation,
            caption="Saturation",
            use_container_width=True
        )

    with c3:
        st.image(
            value,
            caption="Value",
            use_container_width=True
        )


# ============================================================
# EDGE ANALYSIS
# ============================================================

with analytics_tab4:

    edge_density = calculate_edge_density(
        processed
    )

    st.metric(
        "Edge Density",
        f"{edge_density:.2f}%"
    )

    st.write(
        "Edge density represents the percentage of pixels "
        "identified as edges using Canny edge detection."
    )


# ============================================================
# PROCESSING HISTORY
# ============================================================

st.divider()

st.header("📜 Processing History")

if len(st.session_state.history) == 0:

    st.info(
        "No operations have been applied yet."
    )

else:

    for index, item in enumerate(
        reversed(st.session_state.history),
        start=1
    ):

        st.write(
            f"✓ {item}"
        )


# ============================================================
# UNDO / REDO / RESET
# ============================================================

st.divider()

st.header("↩️ Image Controls")

c1, c2, c3 = st.columns(3)

with c1:

    if st.button(
        "↶ Undo",
        use_container_width=True
    ):

        undo()
        st.rerun()


with c2:

    if st.button(
        "↷ Redo",
        use_container_width=True
    ):

        redo()
        st.rerun()


with c3:

    if st.button(
        "🔄 Reset",
        use_container_width=True
    ):

        reset_image()
        st.rerun()


# ============================================================
# PROCESSING PIPELINE
# ============================================================

st.divider()

st.header("🧬 Processing Pipeline")

st.write(
    "Build a sequence of operations and run them together."
)

pipeline_operation = st.selectbox(
    "Add operation to pipeline",
    [
        "Grayscale",
        "Blur",
        "Edge Detection",
        "Sharpen",
        "Sepia",
        "Flip"
    ],
    key="pipeline_operation"
)

if st.button(
    "➕ Add to Pipeline"
):

    st.session_state.pipeline.append(
        pipeline_operation
    )

    st.success(
        f"{pipeline_operation} added."
    )


if len(st.session_state.pipeline) > 0:

    st.subheader(
        "Current Pipeline"
    )

    for index, item in enumerate(
        st.session_state.pipeline,
        start=1
    ):

        st.write(
            f"{index}. {item}"
        )

    p1, p2 = st.columns(2)

    with p1:

        if st.button(
            "▶ Run Pipeline",
            type="primary",
            use_container_width=True
        ):

            save_state_for_undo()

            result = (
                st.session_state.processed_image.copy()
            )

            for item in st.session_state.pipeline:

                if item == "Grayscale":

                    result = to_grayscale(
                        result
                    )

                elif item == "Blur":

                    result = apply_blur(
                        result,
                        "gaussian",
                        5,
                        0
                    )

                elif item == "Edge Detection":

                    result = detect_edges(
                        result,
                        100,
                        200
                    )

                elif item == "Sharpen":

                    result = sharpen(
                        result
                    )

                elif item == "Sepia":

                    result = apply_sepia(
                        result
                    )

                elif item == "Flip":

                    result = flip_image(
                        result,
                        "horizontal"
                    )

            st.session_state.processed_image = result

            add_history(
                "Pipeline: "
                + " → ".join(
                    st.session_state.pipeline
                )
            )

            st.success(
                "Pipeline executed successfully."
            )

            st.rerun()

    with p2:

        if st.button(
            "🗑 Clear Pipeline",
            use_container_width=True
        ):

            st.session_state.pipeline = []

            st.rerun()


# ============================================================
# DOWNLOAD
# ============================================================

st.divider()

st.header("💾 Export")

download_format = st.selectbox(
    "Download format",
    [
        "PNG",
        "JPG",
        "WEBP"
    ]
)

extension_map = {
    "PNG": ".png",
    "JPG": ".jpg",
    "WEBP": ".webp"
}

extension = extension_map[
    download_format
]

download_bytes = image_to_bytes(
    st.session_state.processed_image,
    extension
)

if download_bytes is not None:

    st.download_button(
        label="⬇️ Download Processed Image",
        data=download_bytes,
        file_name=(
            "aperture_processed"
            + extension
        ),
        mime=(
            "image/png"
            if download_format == "PNG"
            else
            "image/jpeg"
            if download_format == "JPG"
            else
            "image/webp"
        ),
        type="primary",
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Aperture • Computer Vision Studio • "
    "Built with Python, Streamlit, OpenCV and NumPy"
)