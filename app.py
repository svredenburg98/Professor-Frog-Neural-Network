from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
COOL_PATH = 'models/cool_model'
COLOR_PATH = 'models/color_model'
DETAIL_PATH = 'models/detail_model'
FORM_PATH = 'models/form_model'
# Load your trained model
cool_model = load_model(COOL_PATH)
color_model = load_model(COLOR_PATH)
detail_model = load_model(DETAIL_PATH)
form_model = load_model(FORM_PATH)
#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def cool_predict(img_path, model):
    img = image.load_img(img_path, target_size=(180, 180))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    cool_preds = cool_model.predict(x)
    return cool_preds

def color_predict(img_path, model):
    img = image.load_img(img_path, target_size=(180, 180))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    color_preds = color_model.predict(x)
    return color_preds

def detail_predict(img_path, model):
    img = image.load_img(img_path, target_size=(180, 180))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    detail_preds = detail_model.predict(x)
    return detail_preds

def form_predict(img_path, model):
    img = image.load_img(img_path, target_size=(180, 180))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    form_preds = form_model.predict(x)
    return form_preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        cool_preds = cool_predict(file_path, cool_model)
        color_preds = color_predict(file_path, color_model)
        detail_preds = detail_predict(file_path, detail_model)
        form_preds = form_predict(file_path, form_model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)  
        cool_score = tf.nn.softmax(cool_preds[0])
        color_score = tf.nn.softmax(color_preds[0])
        detail_score = tf.nn.softmax(detail_preds[0])
        form_score = tf.nn.softmax(form_preds[0])
        class_names = ['1', '2', '3', '4']
        result = "Cool Score: {} with {:.2f} percent confidence \n Color Score: {} with {:.2f} percent confidence \n Detail Score: {} with {:.2f} percent confidence \n Form Score: {} with {:.2f} percent confidence".format(class_names[np.argmax(cool_score)], 100 * np.max(cool_score), class_names[np.argmax(color_score)], 100 * np.max(color_score), class_names[np.argmax(detail_score)], 100 * np.max(detail_score), class_names[np.argmax(form_score)], 100 * np.max(form_score))
        #pred_class = decode_predictions(preds, top=1)   
        #result = str(pred_class[0][0][1])              
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
        