# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask_login import UserMixin

from apps import db
# , login_manager

# from apps.authentication.util import hash_pass


class Pictures(db.Model):

    __tablename__ = 'Pictures'

    id = db.Column(db.Integer, primary_key=True)
    path = db.Column(db.String(255))
    latitude = db.Column(db.String(255))
    longitude = db.Column(db.String(255))
    user_id = db.Column(db.Integer)
    area_coverage = db.Column(db.Integer)
    geo_codes = db.Column(db.Text())
    description = db.Column(db.Text())
    total_area_covered = db.Column(db.Integer)
    height = db.Column(db.Integer)
    tags = db.Column(db.String(255))

    def __init__(self, path, latitude, longitude, user_id, area_coverage, geo_codes, description, total_area_covered, height, tags):
        self.path = path
        self.latitude = latitude
        self.longitude = longitude
        self.user_id = user_id
        self.area_coverage = area_coverage
        self.geo_codes = geo_codes
        self.description = description
        self.total_area_covered = total_area_covered
        self.height = height
        self.tags = tags

    def __repr__(self):
        return f"('{self.id}', '{self.path}', '{self.area_coverage}')"
