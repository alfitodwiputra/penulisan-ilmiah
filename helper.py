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

def display_webrtc_frames(conf, model, st_frame, image):
    # Menampilkan frame yang terdeteksi untuk webrtc
    _display_detected_frames(conf, model, st_frame, image)
