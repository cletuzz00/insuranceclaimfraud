import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir('/Users/CLETUS/Projects/python/mundoz')
heart = pd.read_csv('mun.csv', sep=',', header=0)
print(heart.head())

le = preprocessing.LabelEncoder()
heart['PATIENT_ID'] = le.fit_transform(heart['PATIENT_ID'].astype(str))
heart['Pvd'] = le.fit_transform(heart['Pvd'].astype(str))
heart['Diagnosis'] = le.fit_transform(heart['Diagnosis'].astype(str))
heart['ICD10 Diagnosis'] = le.fit_transform(heart['ICD10 Diagnosis'].astype(str))
heart.drop(['Attending Physician'], axis=1, inplace=True)
print(heart.head())

heart['Status'].replace(to_replace='Y',value=1,inplace=True)
heart['Status'].replace(to_replace='N',value=0,inplace=True)

print(heart.head())

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
print(prediction_test)

print(accuracy_score(y_test, prediction_test))
# # %%
Predict = model.predict([[595,657,198738000,20101110,20101115,2,13,91]])
print(Predict)
# prediction_test = model.predict(X_test)
# # Print the prediction accuracy
# print (metrics.accuracy_score(y_test, prediction_test))
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
# # plot X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
# # plt.scatter(X_test, y_test,  color='gray')
# # plt.plot(X_test, prediction_test, color='red', linewidth=2)
# # plt.show()

# # %%

# # RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(X_train, y_train)
# # model3 = RF.predict(X_test)
# # round(RF.score(X_test, y_test), 4)

# # NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1).fit(X_train, y_train)
# # model2 = NN.predict(X_test)
# # round(NN.score(X_test, y_test), 4)
# # print (metrics.accuracy_score(y_test, model3))
# # print (metrics.accuracy_score(y_test, model2))

