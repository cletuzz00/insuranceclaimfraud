# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")
import os
from sklearn.linear_model import LogisticRegression
from pylab import rcParams
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

info = pd.read_csv('/Users/CLETUS/projects/python/mundoz/insurance_claims.csv')
print(info.head())
plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(18,6))
ax = info.groupby('incident_date').total_claim_amount.count().plot.bar(ylim=0)
ax.set_ylabel('Claim amount ($)')
plt.show()

plt.rcParams['figure.figsize']=[15,8]
plt.style.use('fivethirtyeight')
table = pd.crosstab(info.age, info.fraud_reported)
table.div(table.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.title('Stacked Bar Chart of Age vs Fraud Reported', fontsize=12)
plt.xlabel('Age')
plt.ylabel('Fraud Reported')
plt.show()

info['fraud_reported'].replace(to_replace='Y',value=1,inplace=True)
info['fraud_reported'].replace(to_replace='N',value=0,inplace=True)

info = info.drop(columns=[
    'policy_number',
    'policy_csl',
    'insured_zip',
    'policy_bind_date',
    'incident_date',
    'incident_location',
    'auto_year',
    'incident_hour_of_the_day'])
info.head(2)
# heart['ICD10_Diagnosis'] = le.fit_transform(heart['ICD10_Diagnosis'])
# heart['Diagnosis'] = le.fit_transform(heart['Diagnosis'])
# heart.drop(['PVD_NUM'], axis=1, inplace=True)
# heart.drop(['Attending_Physician'], axis=1, inplace=True)

# # heart['PRVDR_NUM'] = le.fit_transform(heart['PRVDR_NUM'])
# heart.Status.map(dict(Y=1, N=0))
# heart['Claim_ID'] = pd.to_numeric(heart['Claim_ID'])
# heart['CLM_Start_DT'] = pd.to_numeric(heart['CLM_Start_DT'])
# heart['Diagnosis'] = pd.to_numeric(heart['Diagnosis'])
# heart['ICD10_Diagnosis'] = pd.to_numeric(heart['ICD10_Diagnosis'])
# heart['Claim_Amnt'] = pd.to_numeric(heart['Claim_ID'])
# Y = heart["Status"].values
# X = heart.drop(labels = ["Status"],axis = 1)
# # Create Train & Test Data
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)


# # %%
# model = LogisticRegression()
# result = model.fit(X_train, y_train)


# # %%
# prediction_test = model.predict(X_test)
# # Print the prediction accuracy
# print (metrics.accuracy_score(y_test, prediction_test))
# weights = pd.Series(model.coef_[0],
#  index=X.columns.values)
# weights.sort_values(ascending = False)
# print (weights)
# feat_importances = pd.Series(model.coef_[0],index=X.columns.values)
# feat_importances.nlargest(10).plot(kind='barh')
# plt.show()
# # # %%
# # heart['sexualActivity'] = pd.to_numeric(heart['sexualActivity'])
# # heart['serviceExperience'] = pd.to_numeric(heart['serviceExperience'])
# # heart['numberOfPreviousDonations'] = pd.to_numeric(heart['numberOfPreviousDonations'])


# # # %%
# # dummy = pd.get_dummies(heart['gender'])
# dummy.head()


# # %%
# heart  = pd.concat([heart,dummy],axis=1)
# heart.head()


# # %%
# heart.drop(['gender'], axis=1, inplace=True)
# heart.head()


# # %%
# heart["churn"] = heart["churn"].astype(int)
# Y = heart["churn"].values
# X = heart.drop(labels = ["churn"],axis = 1)
# # Create Train & Test Data
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)


# # %%
# model = LogisticRegression()
# result = model.fit(X_train, y_train)


# # %%
# prediction_test = model.predict(X_test)
# # Print the prediction accuracy
# print (metrics.accuracy_score(y_test, prediction_test))


# # %%

# # RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(X_train, y_train)
# # model3 = RF.predict(X_test)
# # round(RF.score(X_test, y_test), 4)

# # NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1).fit(X_train, y_train)
# # model2 = NN.predict(X_test)
# # round(NN.score(X_test, y_test), 4)
# # print (metrics.accuracy_score(y_test, model3))
# # print (metrics.accuracy_score(y_test, model2))


# # %%
# # To get the weights of all the variables
# weights = pd.Series(model.coef_[0],
#  index=X.columns.values)
# weights.sort_values(ascending = False)
# print (weights)
# feat_importances = pd.Series(model.coef_[0],index=X.columns.values)
# feat_importances.nlargest(10).plot(kind='barh')
# plt.show()
# # %%
# # Data to plot
# sizes = heart['churn'].value_counts(sort = True)
# colors = ["grey","purple"] 
# labels = ["yes","no"]
# explode = (0, 0.1)  # only "explode" no label
# rcParams['figure.figsize'] = 5,5
# # Plot
# plt.pie(sizes, colors=colors,labels=labels,explode=explode,
#         autopct='%1.1f%%', shadow=True, startangle=270,)
# plt.title('Percentage of Churn in Dataset ')
# plt.show()

# # from sklearn.datasets import make_blobs
# # # create the inputs and outputs
# # T, w = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=2)
# # # define model
# # model = LogisticRegression(solver='lbfgs')
# # # fit model
# # model.fit(T, w)
# # # make predictions on the entire training dataset
# # yhat = model.predict(T)
# # # connect predictions with outputs
# # for i in range(10):
# # 	print(T[i], yhat[i])
