from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np


from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf

from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# define flask app
app = Flask(__name__)

#models
COOL_PATH = 'models/cool_model'
COLOR_PATH = 'models/color_model'
DETAIL_PATH = 'models/detail_model'
FORM_PATH = 'models/form_model'
# Load trained models
cool_model = load_model(COOL_PATH)
color_model = load_model(COLOR_PATH)
detail_model = load_model(DETAIL_PATH)
form_model = load_model(FORM_PATH)

print('Model loaded. Check http://127.0.0.1:5000/')


def cool_predict(img_path, model):
    img = image.load_img(img_path, target_size=(180, 180))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    cool_preds = cool_model.predict(x)
    return cool_preds

def color_predict(img_path, model):
    img = image.load_img(img_path, target_size=(180, 180))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)



    color_preds = color_model.predict(x)
    return color_preds

def detail_predict(img_path, model):
    img = image.load_img(img_path, target_size=(180, 180))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)


    detail_preds = detail_model.predict(x)
    return detail_preds

def form_predict(img_path, model):
    img = image.load_img(img_path, target_size=(180, 180))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)


    form_preds = form_model.predict(x)
    return form_preds


@app.route('/', methods=['GET'])
def index():
    
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # get the file from post request
        f = request.files['file']

        # save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # run model
        cool_preds = cool_predict(file_path, cool_model)
        color_preds = color_predict(file_path, color_model)
        detail_preds = detail_predict(file_path, detail_model)
        form_preds = form_predict(file_path, form_model)

        #translate results
        cool_score = tf.nn.softmax(cool_preds[0])
        color_score = tf.nn.softmax(color_preds[0])
        detail_score = tf.nn.softmax(detail_preds[0])
        form_score = tf.nn.softmax(form_preds[0])
        class_names = ['1', '2', '3', '4']
        #get just the number
        cool = class_names[np.argmax(cool_score)]
        color = class_names[np.argmax(color_score)]
        detail = class_names[np.argmax(detail_score)]
        form = class_names[np.argmax(form_score)]
        #determine professor response
        if color == '4':
            result = ["Wow, incredible colors on this frog!", int(cool), int(color), int(detail), int(form)]
        elif detail == '4':
            result = ["This must have took you hours! Are you ok?", int(cool), int(color), int(detail), int(form)]
        elif form == '4':
            result = ["Oh my god this almost looks like a real frog!", int(cool), int(color), int(detail), int(form)]
        elif cool == '4':
            result = ["I love this frog. You get an A+", int(cool), int(color), int(detail), int(form)]
        elif cool == '1':
            result = ["No I do not trust this frog at all", int(cool), int(color), int(detail), int(form)]
        elif detail == '1':
            result = ["Did you even try?", int(cool), int(color), int(detail), int(form)]
        elif color == '1':
            result = ["This is depressing to look at", int(cool), int(color), int(detail), int(form)]
        elif form == '1':
            result = ["Ooh, very abstract!", int(cool), int(color), int(detail), int(form)]
        
        else:
            result = ["This is passing, I guess", int(cool), int(color), int(detail), int(form)]
        
        
       
        #send list with response text and number values for plotly
        return jsonify(result)
    return None


if __name__ == '__main__':
    app.run(debug=True)
        