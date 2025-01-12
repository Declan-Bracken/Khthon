# minimal_app.py
import streamlit as st
import torch
from ultralytics import YOLO
model = YOLO("Computer_Vision/models/v2/best.pt")

st.title("Minimal App Test")
st.write("YOLO Model:", model)
st.write(f"Torch version: {torch.__version__}")
st.write(f"Ultralytics version: {YOLO._version}")




