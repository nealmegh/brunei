# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.pictures import blueprint
from flask import Flask, render_template
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





