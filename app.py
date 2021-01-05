import numpy as np, pandas as pd 
from flask import Flask, render_template, request
import cv2
from werkzeug.utils import secure_filename
import os
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('models/model.h5')

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(file.filename))
        file.save(file_path)
    
    path = os.path.join(file_path)
    print(path)
    img_size=150
    img_arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    resized_arr = cv2.resize(img_arr, (img_size, img_size))
    resized_arr = np.array(resized_arr) / 255
    resized_arr = resized_arr.reshape(-1, img_size, img_size, 1)
    pred=model.predict_classes(resized_arr)[0][0]
    if (pred==0):
        pred='Pneumonia'
    else:
        pred='Normal'
    return render_template('index.html', predict = pred)

if __name__ == '__main__':
    app.run(debug=True)
s
