from __future__ import division, print_function
import numpy as np
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
import os
import glob
import re
import sys
from gevent.pywsgi import WSGIServer
from tensorflow.keras.models import model_from_json
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
app = Flask(__name__)

model = load_model("/Users/tushararora/Documents/Fashion items prediction/fashion_item_prediction.h5")


def pro(filename, model):
    img = image.load_img(filename, color_mode = "grayscale" ,target_size=(28,28))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    pro_pred = model.predict(images, batch_size=128)
    return pro_pred

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        upload_file = request.files['file']


        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(upload_file.filename))
        upload_file.save(file_path)

        class_prediction = pro(file_path, model)
        class_prediction = np.argmax(class_prediction, axis = 1)
        # pred_class = decode_predictions(class_prediction, top=1)
        if class_prediction[0] == 0:
            product = "T-shirt/top"
        elif class_prediction[0] == 1:
            product = "Trouser"
        elif class_prediction[0] == 2:
            product = "Pullover"
        elif class_prediction[0] == 3:
            product = "Dress"
        elif class_prediction[0] == 4:
            product = "Coat"
        elif class_prediction[0] == 5:
            product = "Sandal"
        elif class_prediction[0] == 6:
            product = "Shirt"
        elif class_prediction[0] == 7:
            product = "Sneaker"
        elif class_prediction[0] == 8:
            product = "Bag"
        elif class_prediction[0] == 9:
            product = "Ankle boot"
        else:
            product = "Can't be Recognized!!!"
    return render_template('index.html', Prediction_text = f"According to predcition the image is related to {product}")


if __name__ == "__main__":
    app.run(debug=True, threaded=False)
