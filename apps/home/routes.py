# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import Flask, render_template, request, render_template, send_file, request, jsonify, json, make_response, \
    redirect, url_for, flash
from flask_login import login_required, current_user
from jinja2 import TemplateNotFound
from python.apicode import detect_images
import time
from flask_sqlalchemy import SQLAlchemy
import base64
from re import S

# from flask import Flask, render_template, send_file, request, jsonify, json, make_response
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, StringField, IntegerField, HiddenField, TextAreaField, MultipleFileField
from werkzeug.utils import secure_filename
# import os
from wtforms.validators import InputRequired
from apps import db
# from apps.pictures import blueprint
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
from apps.pictures.models import Pictures
import xlsxwriter
from io import BytesIO
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'files'


@blueprint.route('/index')
@login_required
def index():
    # curr_user = current_user.id
    return render_template('home/index.html', segment='index')


@blueprint.route('/upload')
@login_required
def data_upload():
    form = UploadFileFrom()
    return render_template('home/uploadImage.html', form=form, segment='index')


@blueprint.route("/export", methods=['GET'])
@login_required
def export():
    form = ExportFrom()
    return render_template('home/export.html', form=form, segment='index')


@blueprint.route("/download", methods=['POST'])
@login_required
def download_xls():
    form = ExportFrom()
    tags = form.tags.data
    print(tags + 'ho')
    if not tags:
        pictures = Pictures.query.all()
    else:
        # SQLAlchemy query to get all data from User table
        pictures = Pictures.query.filter_by(tags=tags).all()

    # Create a list of dictionaries where each dictionary represents a user record
    data = [{'ID': picture.id, 'User': picture.user_id, 'Picture': picture.path, 'Coverage in Percentage': picture.area_coverage*100, 'Total Coverage(approx) Sq Meter': picture.total_area_covered, 'Height': picture.height, 'Detection Accuracy': 86} for picture in pictures]

    # Convert to pandas DataFrame
    df = pd.DataFrame(data)

    # Write DataFrame to Excel
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.close()
    output.seek(0)

    return send_file(output, attachment_filename='data.xlsx', as_attachment=True)


# @blueprint.route('/upload')
# # @login_required
# def data_upload():
#     form = UploadFileFrom()
#     return render_template('home/uploadImage.html', form=form, segment='index')


# @blueprint.route('/<template>')
# @login_required
# def route_template(template):
#     try:
#
#         if not template.endswith('.html'):
#             template += '.html'
#
#         # Detect the current page
#         segment = get_segment(request)
#
#         # Serve the file (if exists) from app/templates/home/FILE.html
#         return render_template("home/" + template, segment=segment)
#
#     except TemplateNotFound:
#         return render_template('home/page-404.html'), 404
#
#     except:
#         return render_template('home/page-500.html'), 500


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
    files = MultipleFileField('File(s) Upload', validators=[InputRequired()])
    description = TextAreaField("description")
    tags = StringField("tags")
    height = IntegerField("height")
    submit = SubmitField("Upload File")


class ExportFrom(FlaskForm):
    tags = StringField("tags")
    submit = SubmitField("Export")


@blueprint.route('/detect', methods=['POST'])
@login_required
def detect():
    form = UploadFileFrom()
    if form.validate_on_submit():
        files_filenames = []
        description = form.description.data
        tags = form.tags.data
        try:
            for file in form.files.data:
                # file = form.files.data  # grab file
                # description = form.description.data
                # return os.path.join(os.path.abspath(os.path.dirname(__file__)), 'files', secure_filename(file.filename))
                file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                       secure_filename(file.filename)))
                # return "File Have Been Uploaded"
                img = Image.open(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                              secure_filename(file.filename)))
                img.load()
                print(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                   secure_filename(file.filename)))
                full_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                         secure_filename(file.filename))

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

                new_name = str(int(time.time())) + '.jpg'
                new_name_plt = 'PLT' + str(int(time.time())) + '.jpg'
                save_name = f'apps/static/files/' + new_name
                file_path = f'/static/files/' + new_name
                plt.imsave(save_name, datacpy)
                imgplot = plt.imshow(datacpy)
                plt.tight_layout()
                plt_save = f'apps/static/files/' + new_name_plt
                plt.savefig(plt_save)
                plt.close()
                # plt.show()
                detection_response = detect_images(full_path)
                area_coverage = detection_response['a']['sem4']['coverage']
                # return str(detection_response['b'][0][2])
                latitude = detection_response['b'][0][2]
                longitude = detection_response['b'][0][3]
                # geo_codes = jsonify(detection_response['b'])
                geo_codes = {}
                count = 0
                for i in range(len(detection_response['b'])):
                    geo_codes[i] = {'0': detection_response['b'][i][2], '1': detection_response['b'][i][3]}

                geo_codes_db = json.dumps(geo_codes)
                # return jsonify(geo_codes)
                curr_user = current_user.id
                height = form.height.data
                total_area_coverage  = detection_response['a']['sem4Length']['length']*1.25*1.25

                picture = Pictures(file_path, latitude, longitude, user_id=curr_user, area_coverage=area_coverage,
                                   geo_codes=geo_codes_db, height=height, total_area_covered=total_area_coverage,
                                   description=description, tags=tags)
                db.session.add(picture)
                db.session.commit()
            return redirect(url_for('pictures_blueprint.data', id=picture.id))
        except Exception as ex:
            # return ex
            print("something wrong", ex)
            flash('An error occurred.')
            return redirect(url_for('home_blueprint.data_upload'))

        # return send_file('test1234.jpg', mimetype='image/jpg')
        # return render_template('home/viewResults.html', image_filename='test1234.jpg')
