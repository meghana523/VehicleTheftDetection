
import streamlit as st
import cv2
import tempfile
import os
import time
from pathlib import Path

from ultralytics import YOLO


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Vehicle Theft Detection",
    page_icon="🚘",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: bold;
    color: #ffffff;
}

.subtitle {
    font-size: 20px;
    color: #b0b0b0;
}

.module-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #172b43;
    color: white;
    margin-bottom: 20px;
}

.result-box {
    padding: 18px;
    border-radius: 12px;
    background-color: #123d2b;
    color: #50e68b;
    font-size: 20px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD YOLO MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = YOLO("yolov8n.pt")

    return model


model = load_model()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("📋 Project Modules")

st.sidebar.success("✅ Vehicle Detection")

st.sidebar.info("🚘 CCTV Vehicle Detection")

st.sidebar.warning("⏳ Vehicle Re-identification")

st.sidebar.warning("⏳ Similarity Search")

st.sidebar.warning("⏳ Theft Timestamp")

st.sidebar.warning("⏳ Evidence Clip")

st.sidebar.divider()

st.sidebar.write("Team: Vehicle Theft Detection")
st.sidebar.write("Review 2 Prototype")


# --------------------------------------------------
# MAIN TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🚘 Vehicle Theft Detection and Localization</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Deep Learning-Based CCTV Vehicle Analysis</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a vehicle image or CCTV video to detect vehicles using YOLOv8."
)

st.info(
    "Review 2 Prototype | Module 1: Image Detection + Module 2: CCTV Video Detection"
)


# --------------------------------------------------
# INPUT SELECTION
# --------------------------------------------------

input_type = st.radio(
    "Select Input Type",
    ["Vehicle Image", "CCTV Video"],
    horizontal=True
)


# ==================================================
# MODULE 1: IMAGE DETECTION
# ==================================================

if input_type == "Vehicle Image":

    st.header("📷 Upload Vehicle Image")

    uploaded_image = st.file_uploader(
        "Choose a vehicle image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        st.subheader("Uploaded Image")

        st.image(
            uploaded_image,
            use_container_width=True
        )

        detect_button = st.button(
            "🔍 Detect Vehicles",
            type="primary"
        )

        if detect_button:

            with st.spinner("Detecting vehicles..."):

                results = model.predict(
                    source=uploaded_image.getvalue(),
                    conf=0.25
                )

                result = results[0]

                annotated_image = result.plot()

                st.subheader("Detection Result")

                st.image(
                    annotated_image,
                    channels="BGR",
                    use_container_width=True
                )

                vehicle_count = 0

                for box in result.boxes:

                    class_id = int(box.cls[0])

                    class_name = model.names[class_id]

                    confidence = float(box.conf[0])

                    if class_name in [
                        "car",
                        "motorcycle",
                        "bus",
                        "truck"
                    ]:

                        vehicle_count += 1

                        st.write(
                            f"🚘 Vehicle {vehicle_count}: "
                            f"{class_name.upper()} | "
                            f"Confidence: {confidence:.2f}"
                        )

                st.success(
                    f"Total vehicles detected: {vehicle_count}"
                )


# ==================================================
# MODULE 2: CCTV VIDEO DETECTION
# ==================================================

else:

    st.header("🎥 CCTV Video Vehicle Detection")

    st.write(
        "Upload a CCTV video to detect vehicles frame by frame."
    )

    uploaded_video = st.file_uploader(
        "Choose a CCTV video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    confidence_threshold = st.slider(
        "Detection Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05
    )

    process_video_button = st.button(
        "🎥 Process CCTV Video",
        type="primary"
    )

    if uploaded_video is not None:

        st.subheader("Uploaded CCTV Video")

        st.video(uploaded_video)

    if uploaded_video is not None and process_video_button:

        input_video_path = None
        output_video_path = None

        try:

            # ------------------------------------------
            # SAVE INPUT VIDEO
            # ------------------------------------------

            input_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded_video.name).suffix
            )

            input_file.write(
                uploaded_video.getbuffer()
            )

            input_file.close()

            input_video_path = input_file.name

            # ------------------------------------------
            # OPEN VIDEO
            # ------------------------------------------

            cap = cv2.VideoCapture(input_video_path)

            if not cap.isOpened():

                st.error(
                    "Unable to open the uploaded video."
                )

                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0:

                fps = 25

            width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            # ------------------------------------------
            # OUTPUT VIDEO
            # ------------------------------------------

            output_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            output_file.close()

            output_video_path = output_file.name

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            out = cv2.VideoWriter(
                output_video_path,
                fourcc,
                fps,
                (width, height)
            )

            # ------------------------------------------
            # PROCESSING VARIABLES
            # ------------------------------------------

            vehicle_classes = {
                "car",
                "motorcycle",
                "bus",
                "truck"
            }

            total_detections = 0

            max_confidence = 0.0

            processed_frames = 0

            unique_vehicle_types = set()

            progress_bar = st.progress(0)

            status_text = st.empty()

            st.subheader(
                "🔍 Processing CCTV Video"
            )

            start_time = time.time()

            # ------------------------------------------
            # FRAME-BY-FRAME DETECTION
            # ------------------------------------------

            while True:

                success, frame = cap.read()

                if not success:

                    break

                results = model.predict(
                    source=frame,
                    conf=confidence_threshold,
                    verbose=False
                )

                result = results[0]

                annotated_frame = result.plot()

                frame_vehicle_count = 0

                for box in result.boxes:

                    class_id = int(box.cls[0])

                    class_name = model.names[class_id]

                    confidence = float(box.conf[0])

                    if class_name in vehicle_classes:

                        frame_vehicle_count += 1

                        total_detections += 1

                        unique_vehicle_types.add(
                            class_name
                        )

                        max_confidence = max(
                            max_confidence,
                            confidence
                        )

                out.write(annotated_frame)

                processed_frames += 1

                if total_frames > 0:

                    progress = min(
                        processed_frames / total_frames,
                        1.0
                    )

                    progress_bar.progress(progress)

                status_text.write(
                    f"Processing frame "
                    f"{processed_frames} / {total_frames} | "
                    f"Vehicles in current frame: "
                    f"{frame_vehicle_count}"
                )

            # ------------------------------------------
            # RELEASE RESOURCES
            # ------------------------------------------

            cap.release()

            out.release()

            elapsed_time = time.time() - start_time

            progress_bar.progress(1.0)

            status_text.success(
                "CCTV video processing completed."
            )

            # ------------------------------------------
            # DISPLAY PROCESSED VIDEO
            # ------------------------------------------

            st.subheader(
                "🎬 Processed CCTV Video"
            )

            with open(
                output_video_path,
                "rb"
            ) as video_file:

                video_bytes = video_file.read()

            st.video(video_bytes)

            # ------------------------------------------
            # RESULTS
            # ------------------------------------------

            st.subheader(
                "📊 Detection Statistics"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Processed Frames",
                    processed_frames
                )

            with col2:

                st.metric(
                    "Total Vehicle Detections",
                    total_detections
                )

            with col3:

                st.metric(
                    "Processing Time",
                    f"{elapsed_time:.1f} sec"
                )

            st.success(
                "CCTV vehicle detection completed successfully."
            )

            if unique_vehicle_types:

                st.write(
                    "**Vehicle Types Detected:** "
                    + ", ".join(
                        sorted(unique_vehicle_types)
                    )
                )

            if max_confidence > 0:

                st.write(
                    f"**Highest Detection Confidence:** "
                    f"{max_confidence:.2f}"
                )

            st.info(
                "Note: This module detects vehicles in CCTV "
                "frames. It does not yet identify stolen "
                "vehicles or prove theft."
            )

        except Exception as error:

            st.error(
                f"Error while processing video: {error}"
            )

        finally:

            if input_video_path and os.path.exists(
                input_video_path
            ):

                os.remove(input_video_path)

            if output_video_path and os.path.exists(
                output_video_path
            ):

                # Keep the output during this run.
                # Streamlit has already loaded the bytes.
                pass


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Vehicle Theft Detection | YOLOv8 Prototype | Review 2"
)