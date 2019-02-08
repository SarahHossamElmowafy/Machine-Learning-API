#import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Loading data
data = pd.read_csv('dataset.csv')

features = ['gender','PhoneService','InternetService','tenure','TotalCharges','MonthlyCharges','Churn']
data = data[features]

data['tenure'].fillna(data['tenure'].mean(),inplace=True)

# for gender
data['gender'].replace('Male',0,inplace=True)
data['gender'].replace('Female',1,inplace=True)
# for InternetService
data['InternetService'].replace('DSL',0,inplace=True)
data['InternetService'].replace('Fiber optic',1,inplace=True)
data['InternetService'].replace('No',2,inplace=True)

LE = LabelEncoder()
data['PhoneService'] = LE.fit_transform(data['PhoneService'])
data['Churn'] = LE.fit_transform(data['Churn'])

X = data[['gender','PhoneService','InternetService','tenure','TotalCharges','MonthlyCharges']]
y = data.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

svc_classifier = SVC(kernel = 'rbf', random_state = 0)
svc_classifier.fit(X_train, y_train)
y_pred = svc_classifier.predict(X_test)


knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)

NB_classifier = GaussianNB()
NB_classifier.fit(X_train, y_train)
y_pred = NB_classifier.predict(X_test)

#Saving the model: Serialization and Deserialization
from sklearn.externals import joblib
joblib.dump(classifier, 'logit_model.pkl')
joblib.dump(knn_classifier, 'knn_model.pkl')
joblib.dump(NB_classifier, 'nb_model.pkl')
joblib.dump(svc_classifier, 'svc_model.pkl')
