# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import Flask, render_template, request, render_template, send_file, request, jsonify, json, make_response
from flask_login import login_required
from jinja2 import TemplateNotFound

from re import S

# from flask import Flask, render_template, send_file, request, jsonify, json, make_response
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
# import os
from wtforms.validators import InputRequired


import json
from threading import active_count
# import matplotlib.pyplot as plt
# import numpy as np
import PIL
import random
import tensorflow as tf
# import tensorflow as tfp
from tensorflow.python.keras import layers
# import tensorflow.layers
from tensorflow.python.keras.layers import Flatten, Input, Softmax
from tensorflow.python.keras.callbacks import TensorBoard
import keras

from dis import show_code
import json
from operator import truediv
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pprint as pp
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from copy import deepcopy


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'files'


@blueprint.route('/index')
@login_required
def index():
    return render_template('home/index.html', segment='index')


@blueprint.route('/upload')
# @login_required
def data_upload():
    form = UploadFileFrom()
    return render_template('home/uploadImage.html', form=form, segment='index')


# @blueprint.route('/upload')
# # @login_required
# def data_upload():
#     form = UploadFileFrom()
#     return render_template('home/uploadImage.html', form=form, segment='index')


@blueprint.route('/<template>')
@login_required
def route_template(template):
    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):
    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None


class UploadFileFrom(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")


@blueprint.route('/detect', methods=['POST'])
# @login_required
def detect():

    form = UploadFileFrom()
    if form.validate_on_submit():
        file = form.file.data  # grab file
        # return os.path.join(os.path.abspath(os.path.dirname(__file__)), 'files', secure_filename(file.filename))
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))
        fileName = file.filename
        # return "File Have Been Uploaded"
        img = Image.open(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
        img.load()
        print(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
        data = np.asarray(img, dtype="float32")
        datacpy = deepcopy(data)
        data = data / 255.0

        maxy = len(data)
        maxx = len(data[0])
        imgArr = []

        for y in range(0, maxy, 96):
            for x in range(0, maxx, 96):
                imgArr.append(deepcopy(data[y:y + 96, x:x + 96]))

        imgArr = np.array(imgArr, dtype=np.float32)
        # imgArr = tf.convert_to_tensor(imgArr, dtype=tf.float32)

        # imgArr = np.asarray(imgArr).astype(np.float32)

        print(data.shape)
        print(imgArr.shape)
        # input()

        # a = np.array([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]])
        # print('true is')
        # print(a)
        # print('shape is')
        # print(a.shape)
        # a = a.reshape(4, 3)
        # print('flattened is')
        # print(a)
        # print('shape is')
        # print(a.shape)
        # input()

        modelFile = f'apps/static/Models/AcaciaMaxAvg0.84.h5'

        # def createModel():
        #    model = tfp.keras.Sequential([
        #         tfp.keras.layers.InputLayer(input_shape=(96, 96, 3)),
        #         tfp.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        #         tfp.keras.layers.MaxPooling2D(),
        #         tfp.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        #         tfp.keras.layers.MaxPooling2D(),
        #         tfp.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        #         tfp.keras.layers.AveragePooling2D(),
        #         tfp.keras.layers.Flatten(),
        #         tfp.keras.layers.Dense(128, activation='relu'),
        #         tfp.keras.layers.Dense(2)
        #     ])
        #     return model

        model = keras.models.load_model(modelFile, compile=False)

        # model.summary()

        probability_model = keras.Sequential([model, tf.keras.layers.Softmax()])

        predictions = probability_model.predict(imgArr)

        # print(predictions[:10])
        # for i in predictions:
        #     print(np.argmax(i))
        # print(test_labels[0])

        linewidth = 5
        i = 0
        for y in range(0, maxy, 96):
            for x in range(0, maxx, 96):
                if np.argmax(predictions[i]) == 0:

                    for yi in range(y, y + linewidth):
                        for xi in range(x, x + 95):
                            datacpy[yi, xi] = [255.0, 0.0, 0.0]

                    for yi in range(y + 96 - linewidth, y + 96):
                        for xi in range(x, x + 95):
                            datacpy[yi, xi] = [255.0, 0.0, 0.0]

                    for xi in range(x, x + linewidth):
                        for yi in range(y, y + 96):
                            datacpy[yi, xi] = [255.0, 0.0, 0.0]

                    for xi in range(x + 96 - linewidth, x + 96):
                        for yi in range(y, y + 96):
                            datacpy[yi, xi] = [255.0, 0.0, 0.0]
                i += 1

        datacpy = np.asarray(datacpy, dtype='float32')
        datacpy = datacpy / 255.0

        plt.imsave('apps/static/files/test1234.jpg', datacpy)
        imgplot = plt.imshow(datacpy)
        plt.tight_layout()
        plt.savefig('apps/static/files/plt_save.jpg')
        # plt.show()

        # return send_file('test1234.jpg', mimetype='image/jpg')
        return render_template('home/viewResults.html', image_filename='test1234.jpg')
