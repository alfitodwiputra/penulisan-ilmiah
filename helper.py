from ultralytics import YOLO
import streamlit as st
import cv2
import settings

def load_model(model_path):
    # Memuat model YOLO dari path yang diberikan
    return YOLO(model_path)

def _display_detected_frames(conf, model, st_frame, image):
    # Menampilkan frame yang terdeteksi
    image = cv2.resize(image, (720, int(720 * (9 / 16))))
    res = model.predict(image, conf=conf)
    res_plotted = res[0].plot()
    st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)

def play_webcam(conf, model):
    # Memutar webcam dan mendeteksi objek
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(CAP_V4L2)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
