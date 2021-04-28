# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

from flask_login import UserMixin
from sqlalchemy import Binary, Column, Integer, String

from app import db, login_manager

from app.base.util import hash_pass

class User(db.Model, UserMixin):

    __tablename__ = 'User'

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    email = Column(String, unique=True)
    password = Column(Binary)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            # depending on whether value is an iterable or not, we must
            # unpack it's value (when **kwargs is request.form, some values
            # will be a 1-element list)
            if hasattr(value, '__iter__') and not isinstance(value, str):
                # the ,= unpack of a singleton fails PEP8 (travis flake8 test)
                value = value[0]

            if property == 'password':
                value = hash_pass( value ) # we need bytes here (not plain str)
                
            setattr(self, property, value)

    def __repr__(self):
        return str(self.username)


@login_manager.user_loader
def user_loader(id):
    return User.query.filter_by(id=id).first()

@login_manager.request_loader
def request_loader(request):
    username = request.form.get('username')
    user = User.query.filter_by(username=username).first()
    return user if user else None

class ClaimData(db.Model):
    """
    Create a donors table
    """
    __tablename__='claims'
    id = db.Column(db.Integer, primary_key=True)
    PATIENT_ID = db.Column(db.String(60))
    CLAIM_ID = db.Column(db.String(60))
    Pvd = db.Column(db.String(60))
    CLM_Start_DT = db.Column(db.String(60))
    CLM_End_DT = db.Column(db.String(60))
    Diagnosis = db.Column(db.String(60))
    ICD10_Diagnosis = db.Column(db.String(60))
    Claim_Amnt = db.Column(db.String(60))  
    Status = db.Column(db.String(60),default="0") 
    def __repr__(self):
        return '<ClaimData: {}>'.format(self.id)

# claim seeder
class ClaimDataSeeder(db.Model):
    """
    Create a donors table
    """
    __tablename__='claims_seeder'
    id = db.Column(db.String(60), primary_key=True)
    PATIENT_ID = db.Column(db.String(60))
    CLAIM_ID = db.Column(db.String(60))
    Pvd = db.Column(db.String(60))
    CLM_Start_DT = db.Column(db.String(60))
    CLM_End_DT = db.Column(db.String(60))
    Diagnosis = db.Column(db.String(60))
    ICD10_Diagnosis = db.Column(db.String(60))
    Claim_Amnt = db.Column(db.String(60))  
    Status = db.Column(db.String(60),default="0") 
    def __repr__(self):
        return '<ClaimDataSeeder: {}>'.format(self.id)