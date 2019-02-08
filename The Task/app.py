from flask import Flask,render_template,url_for,request
from flask_material import Material
from flask_wtf import Form

#import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ML Pkg
from sklearn.externals import joblib

app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/preview')
def preview():
    return render_template("preview.html")

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
X = data[['gender','PhoneService','InternetService','SeniorCitizen','tenure','TotalCharges','MonthlyCharges']]
y = data.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression

# seed of random generator when shuffle the data
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


@app.route('/', methods=['POST'])

def analyze():
    if request.method == 'POST':
        gender = request.form['gender']
        PhoneService = request.form['PhoneService']
        InternetService = request.form['InternetService']
		tenure = request.form['tenure']
        month = request.form['month']
		total = request.form['total']

    sample = [gender, PhoneService, InternetService,tenure, month, total]
    ex1 = np.array(sample).reshape(1,-1)
    result_prediction = classifier.predict(ex1)

    # Reloading the Model
    if model_choice == 'logitmodel':
        logit_model = joblib.load('data/logit_model.pkl')
        result_prediction = logit_model.predict(ex1)
    elif model_choice == 'knnmodel':
        knn_model = joblib.load('data/knn_model.pkl')
        result_prediction = knn_model.predict(ex1)
    elif model_choice == 'naiveb':
        knn_model = joblib.load('data/nb_model.pkl')
        result_prediction = knn_model.predict(ex1)
    elif model_choice == 'supportv':
        knn_model = joblib.load('data/svc_model.pkl')
        result_prediction = knn_model.predict(ex1)

	return render_template('index.html', result_prediction=result_prediction)


if __name__ == '__main__':
	app.run(debug=True, port=12345)
