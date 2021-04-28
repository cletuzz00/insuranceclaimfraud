# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

from app.home import blueprint
from flask import render_template, redirect, url_for
from flask_login import login_required, current_user
from app import login_manager,db
from jinja2 import TemplateNotFound
from app.base.models import User, ClaimData, ClaimDataSeeder
# model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir('/Users/CLETUS/Projects/python/mundoz/ClaimFraud')
heart = pd.read_csv('mun.csv', sep=',', header=0)
# print(heart.head())

le = preprocessing.LabelEncoder()
heart['PATIENT_ID'] = le.fit_transform(heart['PATIENT_ID'].astype(str))
heart['Pvd'] = le.fit_transform(heart['Pvd'].astype(str))
heart['Diagnosis'] = le.fit_transform(heart['Diagnosis'].astype(str))
heart['ICD10 Diagnosis'] = le.fit_transform(heart['ICD10 Diagnosis'].astype(str))
heart.drop(['Attending Physician'], axis=1, inplace=True)
# print(heart.head())

heart['Status'].replace(to_replace='Y',value=1,inplace=True)
heart['Status'].replace(to_replace='N',value=0,inplace=True)

# print(heart.head())

Y = heart["Status"].values
X = heart.drop(labels = ["Status"],axis = 1)
# Create Train & Test Data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)

model = GaussianNB()    
# Train the model
model.fit(X_train, y_train)
# # Test Model
prediction_test = model.predict(X_test)
# # Print the prediction results
# print(prediction_test)

print(accuracy_score(y_test, prediction_test))
# # %%
# Predict = model.predict([[595,657,198738000,20101110,20101115,2,13,91]])
# print(Predict)
# model end
@blueprint.route('/index')
@login_required
def index():
    
    if not current_user.is_authenticated:
        return redirect(url_for('base_blueprint.login'))
    all_claims = db.session.query(ClaimData).all()
    return render_template('index.html',all_claims=all_claims)

@blueprint.route('/<template>')
def route_template(template):
    
    if not current_user.is_authenticated:
        return redirect(url_for('base_blueprint.login'))

    try:

        return render_template(template + '.html')

    except TemplateNotFound:
        return render_template('page-404.html'), 404
    
    except:
        return render_template('page-500.html'), 500

@blueprint.route('/fraud', methods = ['GET', 'POST'])
def fraud():
    claims = db.session.query(ClaimData).all()
    for claim in claims:
        #distribute
        p_id = le.fit_transform([claim.PATIENT_ID]).astype(str)
        cl_id = le.fit_transform([claim.CLAIM_ID]).astype(str)
        pvd = le.fit_transform([claim.Pvd]).astype(str)
        clms = int(claim.CLM_Start_DT)
        clme = int(claim.CLM_End_DT)
        dg = le.fit_transform([claim.Diagnosis]).astype(str)
        icd = le.fit_transform([claim.ICD10_Diagnosis]).astype(str)
        camount = int(claim.Claim_Amnt)
        fraud = model.predict([[p_id,cl_id,pvd,clms,clme,dg,icd,camount]])
        claim.Status = int(fraud) 
        db.session.commit()  
    all_claims = db.session.query(ClaimData).all()
    return render_template('index.html',all_claims=all_claims)
