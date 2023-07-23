# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
import json
import os

from apps.pictures import blueprint
from flask import Flask, render_template, redirect, url_for
from flask_login import login_required

from apps import db
from apps.pictures import blueprint
from apps.pictures.models import Pictures

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'files'


@blueprint.route('/index2')
@login_required
def index():
    return render_template('home/index.html', segment='index')


@blueprint.route('/data/<int:id>')
@login_required
def data(id):
    data = Pictures.query.get_or_404(id)
    geo_codes = json.loads(data.geo_codes, parse_int=True)

    # return str(geo_codes)
    return render_template('home/examples-project-detail.html', picture=data, geo_codes=geo_codes)


@blueprint.route('/all_data')
@login_required
def all_data():
    pictures = Pictures.query.order_by(Pictures.id)

    return render_template('home/tables-data.html', pictures=pictures)


@blueprint.route('/delete_data/<int:id>', methods=['POST'])
@login_required
def delete_data(id):
    picture_data = Pictures.query.get_or_404(id)

    if picture_data is None:
        return "Error: No such data"

    # remove the associated file
    file_path = os.path.join(os.path.dirname(app.root_path), os.path.normpath(picture_data.path.lstrip("/")))
    if os.path.exists(file_path):
        os.remove(file_path)
        print('file found', file_path)
    else:
        print(file_path)

    # remove the data from database
    db.session.delete(picture_data)
    db.session.commit()

    return redirect(url_for('pictures_blueprint.all_data'))






