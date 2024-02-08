"""Pneumonia detection app"""

import pickle

import numpy as np
import pydicom
import streamlit as st
import torch

from pnm.pneumonia_classifier import PneumoniaClassifier

MODEL_PATH = 'models/resnet_2024-02-06.pth'

st.title("Pneumonia Detection")

threshold = st.sidebar.slider("Probability threshold", 0.0, 1.0, 0.25)

model_load_state = st.text('Loading pneumonia detector...')
standard_params = pickle.load(open('models/standard_params.pkl', 'rb'))
pnm_model = PneumoniaClassifier()
pnm_model.load_state_dict(torch.load(MODEL_PATH))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pnm_model.to(device)
model_load_state.text('')

uploaded_file = st.file_uploader("Upload DICOM or NPY file",
                                 type=["dcm", "npy"])


if uploaded_file is not None:
    if uploaded_file.name.endswith('.npy'):
        img = np.load(uploaded_file)
    else:
        dcm = pydicom.dcmread(uploaded_file)
        img = dcm.pixel_array

    st.image(img, caption='chest x-ray', width=300)

    if st.button('Run Pneumonia Detection'):
        img_batch = torch.tensor(img).unsqueeze(0).unsqueeze(0)
        img_batch = (img_batch.to(device).float() -
                     standard_params[0] / standard_params[1])
        with torch.no_grad():
            logit = pnm_model(img_batch)

        prob = torch.sigmoid(logit).item()
        st.write(f'Pneumonia probability: {prob:.2f}')
        if prob > threshold:
            result = f':( Pneumonia detected with probability {prob:.2f} :('
        else:
            result = 'Congrats! No pneumonia detected.'

        st.write(result)
