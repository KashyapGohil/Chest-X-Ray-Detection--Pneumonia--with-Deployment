import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import streamlit as st
from PIL import Image

st.title("Chest X-Ray Detection")
st.header("Identify Pneumonia")

@st.cache(allow_output_mutation=True)
def model_loading():
    model = load_model('models/model.h5')
    return model    

model_load_state = st.text('Loading Model...')
model= model_loading()
model_load_state.text('Loading Model...done!')


def make_prediction(img):
    img_size =150
    img = img.resize((img_size,img_size))
    resized_arr = np.array(img) / 255
    resized_arr = resized_arr.reshape(-1, img_size, img_size, 1)
    pred=model.predict_classes(resized_arr)[0][0]
    if (pred==0):
        pred='Pneumonia'
    else:
        pred='Normal'
    return img,pred

uploaded_file = st.file_uploader(label="Upload an image.. ",
                                type=["png","jpeg","jpg"])
                                

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.text("Press Predict Button ")
    pred_button = st.button("Predict")
else:
    st.warning("Please upload image !!!")
    st.stop()

if pred_button:
    img,pred = make_prediction(img)
    st.write(f"Prediction: {pred}")