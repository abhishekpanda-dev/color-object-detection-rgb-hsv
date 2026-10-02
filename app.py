import streamlit as st
import numpy as np
from PIL import Image

from detector import detect_colored_objects
from processing import (
    clean_binary_mask,
    convert_rgb_to_hsv,
    create_hsv_mask,
    create_red_hsv_mask,
    create_rgb_mask,
)
from webcam import WebcamCapture


RGB_PRESETS = {
    "Red": {"r_min": 150, "r_max": 255, "g_min": 0, "g_max": 120, "b_min": 0, "b_max": 120},
    "Green": {"r_min": 0, "r_max": 120, "g_min": 120, "g_max": 255, "b_min": 0, "b_max": 120},
    "Blue": {"r_min": 0, "r_max": 120, "g_min": 0, "g_max": 120, "b_min": 150, "b_max": 255},
    "Yellow": {"r_min": 150, "r_max": 255, "g_min": 150, "g_max": 255, "b_min": 0, "b_max": 140},
}

HSV_PRESETS = {
    "Red": {
        "lower_h_min": 0,
        "lower_h_max": 10,
        "upper_h_min": 170,
        "upper_h_max": 179,
        "s_min": 70,
        "s_max": 255,
        "v_min": 50,
        "v_max": 255,
    },
    "Green": {"h_min": 35, "h_max": 85, "s_min": 50, "s_max": 255, "v_min": 50, "v_max": 255},
    "Blue": {"h_min": 90, "h_max": 130, "s_min": 50, "s_max": 255, "v_min": 50, "v_max": 255},
    "Yellow": {"h_min": 20, "h_max": 35, "s_min": 80, "s_max": 255, "v_min": 80, "v_max": 255},
}

CAMERA_STATE_KEY = "_webcam_camera"
CAMERA_RUNNING_KEY = "_webcam_running"
CAMERA_ERROR_KEY = "_webcam_error"


def load_selected_preset() -> None:
    """Load both color-space presets when the target color changes."""
    target_color = st.session_state["target_color"]

    for preset in (RGB_PRESETS[target_color], HSV_PRESETS[target_color]):
        for key, value in preset.items():
            st.session_state[key] = value


def keep_minimum_valid(
    minimum_widget_key: str,
    maximum_widget_key: str,
    minimum_key: str,
    maximum_key: str,
) -> None:
    """Raise a channel maximum when its minimum is moved above it."""
    if st.session_state[minimum_widget_key] > st.session_state[maximum_widget_key]:
        st.session_state[maximum_widget_key] = st.session_state[minimum_widget_key]
    st.session_state[minimum_key] = st.session_state[minimum_widget_key]
    st.session_state[maximum_key] = st.session_state[maximum_widget_key]


def keep_maximum_valid(
    minimum_widget_key: str,
    maximum_widget_key: str,
    minimum_key: str,
    maximum_key: str,
) -> None:
    """Lower a channel minimum when its maximum is moved below it."""
    if st.session_state[maximum_widget_key] < st.session_state[minimum_widget_key]:
        st.session_state[minimum_widget_key] = st.session_state[maximum_widget_key]
    st.session_state[minimum_key] = st.session_state[minimum_widget_key]
    st.session_state[maximum_key] = st.session_state[maximum_widget_key]


def prepare_slider_state(
    minimum_key: str,
    maximum_key: str,
) -> tuple[str, str]:
    """Restore persistent thresholds into the currently visible widgets."""
    minimum_widget_key = f"_widget_{minimum_key}"
    maximum_widget_key = f"_widget_{maximum_key}"
    st.session_state[minimum_widget_key] = st.session_state[minimum_key]
    st.session_state[maximum_widget_key] = st.session_state[maximum_key]
    return minimum_widget_key, maximum_widget_key


def channel_sliders(channel: str) -> tuple[int, int]:
    """Display the minimum and maximum sliders for one RGB channel."""
    minimum_key = f"{channel.lower()}_min"
    maximum_key = f"{channel.lower()}_max"
    minimum_widget_key, maximum_widget_key = prepare_slider_state(
        minimum_key,
        maximum_key,
    )

    minimum = st.slider(
        f"{channel} Minimum",
        min_value=0,
        max_value=255,
        key=minimum_widget_key,
        on_change=keep_minimum_valid,
        args=(minimum_widget_key, maximum_widget_key, minimum_key, maximum_key),
    )
    maximum = st.slider(
        f"{channel} Maximum",
        min_value=0,
        max_value=255,
        key=maximum_widget_key,
        on_change=keep_maximum_valid,
        args=(minimum_widget_key, maximum_widget_key, minimum_key, maximum_key),
    )
    return minimum, maximum


def hsv_range_sliders(
    label: str,
    minimum_key: str,
    maximum_key: str,
    channel_maximum: int,
) -> tuple[int, int]:
    """Display a validated pair of HSV threshold sliders."""
    minimum_widget_key, maximum_widget_key = prepare_slider_state(
        minimum_key,
        maximum_key,
    )

    minimum = st.slider(
        f"{label} Minimum",
        min_value=0,
        max_value=channel_maximum,
        key=minimum_widget_key,
        on_change=keep_minimum_valid,
        args=(minimum_widget_key, maximum_widget_key, minimum_key, maximum_key),
    )
    maximum = st.slider(
        f"{label} Maximum",
        min_value=0,
        max_value=channel_maximum,
        key=maximum_widget_key,
        on_change=keep_maximum_valid,
        args=(minimum_widget_key, maximum_widget_key, minimum_key, maximum_key),
    )
    return minimum, maximum


def display_detection_result(
    annotated_image: np.ndarray,
    object_count: int,
    detections: list[dict[str, object]],
    contour_count: int,
) -> None:
    """Display the annotated image, count, and accepted contour information."""
    st.subheader("Final Detection Result")
    st.image(annotated_image, use_container_width=True)
    st.write(f"Objects Detected: {object_count}")
    st.caption(f"External contours found before area filtering: {contour_count}")

    if object_count == 0:
        st.info("No colored objects passed the current detection thresholds.")
        return

    result_rows = [
        {
            "Object": index,
            "Area (px)": round(float(detection["area"]), 2),
            "Bounding Box (x, y, w, h)": str(detection["bounding_box"]),
        }
        for index, detection in enumerate(detections, start=1)
    ]
    st.table(result_rows)


def display_comparison_detection_result(
    method: str,
    annotated_image: np.ndarray,
    object_count: int,
    detections: list[dict[str, object]],
    contour_count: int,
) -> None:
    """Display one independent branch of the RGB-versus-HSV comparison."""
    st.markdown(f"#### {method} Detection Result")
    st.image(annotated_image, use_container_width=True)
    st.write(f"{method} Objects Detected: {object_count}")
    st.caption(f"{method} External Contours: {contour_count}")

    if object_count == 0:
        st.info(f"No {method} detections passed the current settings.")
        return

    result_rows = [
        {
            "Object": index,
            "Area (px)": round(float(detection["area"]), 2),
            "Bounding Box (x, y, w, h)": str(detection["bounding_box"]),
        }
        for index, detection in enumerate(detections, start=1)
    ]
    st.markdown(f"**{method} Detection Details**")
    st.table(result_rows)


def release_webcam_camera() -> None:
    """Stop and remove the current session's webcam resource."""
    camera = st.session_state.pop(CAMERA_STATE_KEY, None)
    if camera is not None:
        camera.release()
    st.session_state[CAMERA_RUNNING_KEY] = False


def start_webcam_camera() -> None:
    """Open the default webcam and record a user-facing failure message."""
    release_webcam_camera()
    st.session_state[CAMERA_ERROR_KEY] = None

    try:
        camera = WebcamCapture(device_index=0, width=640, height=480)
    except Exception:
        st.session_state[CAMERA_ERROR_KEY] = (
            "Unable to access webcam. Check camera permissions and make sure "
            "another application is not using the camera."
        )
        return

    if not camera.is_opened():
        camera.release()
        st.session_state[CAMERA_ERROR_KEY] = (
            "Unable to access webcam. Check camera permissions and make sure "
            "another application is not using the camera."
        )
        return

    st.session_state[CAMERA_STATE_KEY] = camera
    st.session_state[CAMERA_RUNNING_KEY] = True


st.set_page_config(
    page_title="Color-Based Object Detection Using RGB and HSV",
)


@st.fragment(run_every=0.1)
def display_live_webcam(
    detection_mode: str,
    target_color: str,
    kernel_size: int,
    minimum_area: int,
    thresholds: dict[str, int],
) -> None:
    """Read and process one webcam frame per non-blocking fragment rerun."""
    if not st.session_state.get(CAMERA_RUNNING_KEY, False):
        return

    camera = st.session_state.get(CAMERA_STATE_KEY)
    if camera is None or not camera.is_opened():
        release_webcam_camera()
        st.session_state[CAMERA_ERROR_KEY] = (
            "The webcam became unavailable and was stopped."
        )
        st.rerun(scope="app")

    try:
        frame_ok, rgb_frame = camera.read_rgb_frame()
        if not frame_ok or rgb_frame is None:
            release_webcam_camera()
            st.session_state[CAMERA_ERROR_KEY] = (
                "Unable to read a webcam frame. The camera was stopped."
            )
            st.rerun(scope="app")

        if detection_mode == "RGB Detection":
            raw_mask = create_rgb_mask(
                rgb_frame,
                min_rgb=(
                    thresholds["r_min"],
                    thresholds["g_min"],
                    thresholds["b_min"],
                ),
                max_rgb=(
                    thresholds["r_max"],
                    thresholds["g_max"],
                    thresholds["b_max"],
                ),
            )
        else:
            hsv_frame = convert_rgb_to_hsv(rgb_frame)
            if target_color == "Red":
                raw_mask = create_red_hsv_mask(
                    hsv_frame,
                    lower_hue_range=(
                        thresholds["lower_h_min"],
                        thresholds["lower_h_max"],
                    ),
                    upper_hue_range=(
                        thresholds["upper_h_min"],
                        thresholds["upper_h_max"],
                    ),
                    saturation_range=(thresholds["s_min"], thresholds["s_max"]),
                    value_range=(thresholds["v_min"], thresholds["v_max"]),
                )
            else:
                raw_mask = create_hsv_mask(
                    hsv_frame,
                    min_hsv=(
                        thresholds["h_min"],
                        thresholds["s_min"],
                        thresholds["v_min"],
                    ),
                    max_hsv=(
                        thresholds["h_max"],
                        thresholds["s_max"],
                        thresholds["v_max"],
                    ),
                )

        cleaned_mask = clean_binary_mask(raw_mask, kernel_size)
        annotated_image, object_count, _, contour_count = detect_colored_objects(
            rgb_frame,
            cleaned_mask,
            minimum_area,
            target_color,
        )

        st.subheader("Live Detection Result")
        st.image(annotated_image, channels="RGB", width="stretch")
        st.write(f"Objects Detected: {object_count}")
        st.caption(f"External contours found before area filtering: {contour_count}")
    except Exception:
        release_webcam_camera()
        st.session_state[CAMERA_ERROR_KEY] = (
            "Webcam processing stopped because the current frame could not be processed."
        )
        st.rerun(scope="app")

st.title("Color-Based Object Detection Using RGB and HSV")
st.write(
    "Detect and compare colored objects using RGB and HSV color-space thresholding."
)

input_source = st.selectbox(
    "Input Source",
    options=["Upload Image", "Webcam / Live Detection"],
    key="input_source",
)

if input_source == "Upload Image":
    release_webcam_camera()

detection_mode_options = ["RGB Detection", "HSV Detection"]
if input_source == "Upload Image":
    detection_mode_options.append("RGB vs HSV Comparison")

if st.session_state.get("detection_mode") not in detection_mode_options:
    st.session_state["detection_mode"] = "RGB Detection"

detection_mode = st.selectbox(
    "Detection Mode",
    options=detection_mode_options,
    key="detection_mode",
)

target_color = st.selectbox(
    "Target Color",
    options=list(RGB_PRESETS),
    key="target_color",
    on_change=load_selected_preset,
)

for active_preset in (RGB_PRESETS[target_color], HSV_PRESETS[target_color]):
    for threshold_name, threshold_value in active_preset.items():
        st.session_state.setdefault(threshold_name, threshold_value)

st.write(f"Selected Target Color: {target_color}")
st.write(f"Detection Mode: {detection_mode}")

kernel_size = st.selectbox(
    "Morphological Kernel Size",
    options=[3, 5, 7],
    index=1,
    format_func=lambda size: f"{size} x {size}",
    help=(
        "Controls the neighborhood size used for morphological opening "
        "and closing."
    ),
)

if detection_mode in ("RGB Detection", "RGB vs HSV Comparison"):
    st.subheader("RGB Thresholds")
    r_min, r_max = channel_sliders("R")
    g_min, g_max = channel_sliders("G")
    b_min, b_max = channel_sliders("B")

if detection_mode in ("HSV Detection", "RGB vs HSV Comparison"):
    st.subheader("HSV Thresholds")

    if target_color == "Red":
        st.info(
            "Red wraps around the OpenCV Hue scale, so two Hue ranges are combined."
        )
        lower_h_min, lower_h_max = hsv_range_sliders(
            "Lower Red H", "lower_h_min", "lower_h_max", 179
        )
        upper_h_min, upper_h_max = hsv_range_sliders(
            "Upper Red H", "upper_h_min", "upper_h_max", 179
        )
    else:
        h_min, h_max = hsv_range_sliders("H", "h_min", "h_max", 179)

    s_min, s_max = hsv_range_sliders("S", "s_min", "s_max", 255)
    v_min, v_max = hsv_range_sliders("V", "v_min", "v_max", 255)

minimum_area = st.slider(
    "Minimum Object Area",
    min_value=50,
    max_value=10000,
    value=500,
    step=50,
    help="Contours smaller than this area in pixels are ignored.",
)

st.write(f"Minimum Object Area: {minimum_area} px")
st.write(f"Morphological Kernel: {kernel_size}x{kernel_size}")

uploaded_file = None
if input_source == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
    )

if uploaded_file is not None:
    # PIL supplies RGB data. Keeping that order is essential because OpenCV
    # often uses BGR when it reads an image itself.
    image = Image.open(uploaded_file).convert("RGB")
    rgb_image = np.asarray(image)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if detection_mode in ("RGB Detection", "RGB vs HSV Comparison"):
        rgb_mask = create_rgb_mask(
            rgb_image,
            min_rgb=(r_min, g_min, b_min),
            max_rgb=(r_max, g_max, b_max),
        )

    if detection_mode in ("HSV Detection", "RGB vs HSV Comparison"):
        hsv_image = convert_rgb_to_hsv(rgb_image)

        if target_color == "Red":
            hsv_mask = create_red_hsv_mask(
                hsv_image,
                lower_hue_range=(lower_h_min, lower_h_max),
                upper_hue_range=(upper_h_min, upper_h_max),
                saturation_range=(s_min, s_max),
                value_range=(v_min, v_max),
            )
        else:
            hsv_mask = create_hsv_mask(
                hsv_image,
                min_hsv=(h_min, s_min, v_min),
                max_hsv=(h_max, s_max, v_max),
            )

    if detection_mode == "RGB Detection":
        cleaned_rgb_mask = clean_binary_mask(rgb_mask, kernel_size)

        raw_column, cleaned_column = st.columns(2)
        with raw_column:
            st.subheader("Raw RGB Binary Mask")
            st.image(rgb_mask, use_container_width=True, clamp=True)
        with cleaned_column:
            st.subheader("Cleaned RGB Binary Mask")
            st.image(cleaned_rgb_mask, use_container_width=True, clamp=True)

        annotated_image, object_count, detections, contour_count = (
            detect_colored_objects(
                rgb_image,
                cleaned_rgb_mask,
                minimum_area,
                target_color,
            )
        )
        display_detection_result(
            annotated_image,
            object_count,
            detections,
            contour_count,
        )
    elif detection_mode == "HSV Detection":
        cleaned_hsv_mask = clean_binary_mask(hsv_mask, kernel_size)

        raw_column, cleaned_column = st.columns(2)
        with raw_column:
            st.subheader("Raw HSV Binary Mask")
            st.image(hsv_mask, use_container_width=True, clamp=True)
        with cleaned_column:
            st.subheader("Cleaned HSV Binary Mask")
            st.image(cleaned_hsv_mask, use_container_width=True, clamp=True)

        annotated_image, object_count, detections, contour_count = (
            detect_colored_objects(
                rgb_image,
                cleaned_hsv_mask,
                minimum_area,
                target_color,
            )
        )
        display_detection_result(
            annotated_image,
            object_count,
            detections,
            contour_count,
        )
    else:
        cleaned_rgb_mask = clean_binary_mask(rgb_mask, kernel_size)
        cleaned_hsv_mask = clean_binary_mask(hsv_mask, kernel_size)

        rgb_annotated, rgb_count, rgb_detections, rgb_contours = (
            detect_colored_objects(
                rgb_image,
                cleaned_rgb_mask,
                minimum_area,
                target_color,
            )
        )
        hsv_annotated, hsv_count, hsv_detections, hsv_contours = (
            detect_colored_objects(
                rgb_image,
                cleaned_hsv_mask,
                minimum_area,
                target_color,
            )
        )

        st.subheader("Raw Binary Mask Comparison")
        rgb_raw_column, hsv_raw_column = st.columns(2)
        with rgb_raw_column:
            st.markdown("#### RGB Raw Binary Mask")
            st.image(rgb_mask, use_container_width=True, clamp=True)
        with hsv_raw_column:
            st.markdown("#### HSV Raw Binary Mask")
            st.image(hsv_mask, use_container_width=True, clamp=True)

        st.subheader("Cleaned Binary Mask Comparison")
        rgb_cleaned_column, hsv_cleaned_column = st.columns(2)
        with rgb_cleaned_column:
            st.markdown("#### RGB Cleaned Binary Mask")
            st.image(cleaned_rgb_mask, use_container_width=True, clamp=True)
        with hsv_cleaned_column:
            st.markdown("#### HSV Cleaned Binary Mask")
            st.image(cleaned_hsv_mask, use_container_width=True, clamp=True)

        st.subheader("Final Detection Comparison")
        rgb_result_column, hsv_result_column = st.columns(2)
        with rgb_result_column:
            display_comparison_detection_result(
                "RGB",
                rgb_annotated,
                rgb_count,
                rgb_detections,
                rgb_contours,
            )
        with hsv_result_column:
            display_comparison_detection_result(
                "HSV",
                hsv_annotated,
                hsv_count,
                hsv_detections,
                hsv_contours,
            )

        st.subheader("Comparison Summary")
        st.write(f"Target Color: {target_color}")
        st.write(f"Morphological Kernel: {kernel_size}x{kernel_size}")
        st.write(f"Minimum Object Area: {minimum_area} px")
        st.write(f"RGB Objects Detected: {rgb_count}")
        st.write(f"HSV Objects Detected: {hsv_count}")

if input_source == "Webcam / Live Detection":
    start_column, stop_column = st.columns(2)
    with start_column:
        start_camera = st.button(
            "Start Camera",
            disabled=st.session_state.get(CAMERA_RUNNING_KEY, False),
            width="stretch",
        )
    with stop_column:
        stop_camera = st.button(
            "Stop Camera",
            disabled=not st.session_state.get(CAMERA_RUNNING_KEY, False),
            width="stretch",
        )

    if start_camera:
        start_webcam_camera()
        st.rerun()
    if stop_camera:
        release_webcam_camera()
        st.session_state[CAMERA_ERROR_KEY] = None
        st.rerun()

    camera_error = st.session_state.get(CAMERA_ERROR_KEY)
    if camera_error:
        st.error(camera_error)

    if st.session_state.get(CAMERA_RUNNING_KEY, False):
        if detection_mode == "RGB Detection":
            live_thresholds = {
                "r_min": r_min,
                "r_max": r_max,
                "g_min": g_min,
                "g_max": g_max,
                "b_min": b_min,
                "b_max": b_max,
            }
        elif target_color == "Red":
            live_thresholds = {
                "lower_h_min": lower_h_min,
                "lower_h_max": lower_h_max,
                "upper_h_min": upper_h_min,
                "upper_h_max": upper_h_max,
                "s_min": s_min,
                "s_max": s_max,
                "v_min": v_min,
                "v_max": v_max,
            }
        else:
            live_thresholds = {
                "h_min": h_min,
                "h_max": h_max,
                "s_min": s_min,
                "s_max": s_max,
                "v_min": v_min,
                "v_max": v_max,
            }

        display_live_webcam(
            detection_mode,
            target_color,
            kernel_size,
            minimum_area,
            live_thresholds,
        )
    elif not camera_error:
        st.info("Select Start Camera to begin live color-based object detection.")
