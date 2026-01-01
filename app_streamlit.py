
import streamlit as st
import cv2
import numpy as np
import tempfile
import subprocess
import sys
import os
import time
import mediapipe as mp
import os

IS_RENDER = os.environ.get("RENDER", False)


st.set_page_config(layout="wide", page_title="Pose App (Image / Video / Live Webcam Launcher)")
st.title("🧍 Pose & Mesh App — Image / Video / Launch Live Webcam Runner")

# Path to the external runner script (assumed in same folder)
RUNNER_SCRIPT = os.path.join(os.path.dirname(__file__), "webcam_runner.py")

# Sidebar controls
mode = st.sidebar.radio(
    "Mode",
    ["Image", "Video"] if IS_RENDER else ["Image", "Video", "Live Webcam"]
)


# session_state to keep track of subprocess
if "webcam_proc" not in st.session_state:
    st.session_state.webcam_proc = None

# ------------ Image mode ------------
if mode == "Image":
    st.write("Upload an image — MediaPipe Holistic will draw pose / face / hands (red).")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # process once with MediaPipe Holistic
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils

        with mp_holistic.Holistic(static_image_mode=True,
                                  model_complexity=1,
                                  refine_face_landmarks=True) as holistic:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(img_rgb)
            out = img.copy()
            # draw red meshes
            red = (0, 0, 255)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(out, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=red, thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=red, thickness=2))
            if results.face_landmarks:
                mp_drawing.draw_landmarks(out, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                          mp_drawing.DrawingSpec(color=red, thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=red, thickness=1))
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(out, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=red, thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=red, thickness=2))
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(out, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=red, thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=red, thickness=2))

            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)

# ------------ Video mode ------------
elif mode == "Video":
    st.write("Upload a video. The app will process frames and display results.")
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded:
        # save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.flush()
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils

        with mp_holistic.Holistic(static_image_mode=False,
                                  model_complexity=1,
                                  refine_face_landmarks=True) as holistic:
            prev = time.time()
            last_out = None
            frame_idx = 0
            # Processing loop: process every frame (you can skip frames for speed)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                # optional resize for speed (uncomment if required)
                # frame = cv2.resize(frame, (640, 480))
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(img_rgb)
                out = frame.copy()
                red = (0, 0, 255)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(out, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=red, thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=red, thickness=2))
                if results.face_landmarks:
                    mp_drawing.draw_landmarks(out, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                              mp_drawing.DrawingSpec(color=red, thickness=1, circle_radius=1),
                                              mp_drawing.DrawingSpec(color=red, thickness=1))
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(out, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=red, thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=red, thickness=2))
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(out, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=red, thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=red, thickness=2))

                # FPS
                now = time.time()
                fps = 1.0 / (now - prev) if prev else 0.0
                prev = now
                cv2.putText(out, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                stframe.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)

        cap.release()
        try:
            os.remove(tfile.name)
        except Exception:
            pass

# ------------ Live Webcam (native runner) ------------
else:
    st.write("Live Webcam will run in a separate native Python process (OpenCV window) for smooth realtime.")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Launch Live Webcam Runner"):
            # only launch if script exists
            if not os.path.exists(RUNNER_SCRIPT):
                st.error(f"Runner script not found: {RUNNER_SCRIPT}")
            else:
                # if a previous process exists, do not start another
                if st.session_state.webcam_proc is None:
                    # start subprocess using same Python executable
                    proc = subprocess.Popen([sys.executable, RUNNER_SCRIPT], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    st.session_state.webcam_proc = proc
                    st.success("Launched webcam runner. A new OpenCV window should open on your desktop.")
                else:
                    st.warning("Webcam runner already launched.")

    with col2:
        if st.button("Stop Live Webcam Runner"):
            proc = st.session_state.get("webcam_proc", None)
            if proc is None:
                st.info("No webcam runner is running.")
            else:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
                st.session_state.webcam_proc = None
                st.success("Stopped webcam runner.")

    st.markdown(
        """
        **Notes**
        - The webcam window (OpenCV) runs outside the browser — allow popups / accept camera permission if prompted.
        - To stop, use the "Stop Live Webcam Runner" button or close the OpenCV window (the process may still be running; stop via the button).
        - If the runner doesn't start, run `python webcam_runner.py` manually from the same folder to see error messages.
        """
    )

