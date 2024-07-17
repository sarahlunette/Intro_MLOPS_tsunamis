from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score as acc

from mlflow import MlflowClient
import mlflow

def run_mlflow_experiment():
	medical_experiment = mlflow.set_experiment("Medical_experiment")
	run_name = "run_medical"
	artifact_path = "models"
	
	# This will have to go into a separate file
	df = pd.read_csv('../data/processed/diabetes.csv')
	df.dropna(subset = 'Outcome', inplace = True)
	df = df.fillna(df.mean()) # L'enlever quand on update
	df.drop_duplicates(inplace = True) # L'enlever quand on update
	
	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	
	X = df.drop('Outcome', axis = 1)
	y = df['Outcome']
	
	from sklearn.model_selection import train_test_split
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, stratify = y)

	X_train_standard = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)
	
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.model_selection import GridSearchCV
	
	names = ['RandomForest', 'LogisticRegression']
	estimators = [RandomForestClassifier(), LogisticRegression()]
	parameters = [{
	            'n_estimators': [50, 100, 200],
	            'max_depth': [None, 5, 10],
	            'min_samples_split': [2, 5, 10]
	        },
	              {
	            'C': [0.1, 1.0, 10.0],
	            'penalty': ['l2']
	        }
	             ]
	
	clf1 = GridSearchCV(estimators[0], parameters[0], cv = 5, n_jobs = -1)
	clf2 = GridSearchCV(estimators[1], parameters[1], cv = 5, n_jobs = -1)
	
	
	clf1.fit(X_train_standard, y_train)
	clf2.fit(X_train_standard, y_train)

	print(clf1.best_params_)
	print(clf2.best_params_)
	
	
	y_pred1 = clf1.predict(X_test)
	y_pred2 = clf2.predict(X_test)
	
	models = [clf1, clf2]

	print(len(pd.Series(y_test.unique())),len(pd.Series(y_pred1).unique()))#, len(pd.Series(y_pred2).unique()))
	
	def selection(models):
		if clf1.score(X_test, y_test) > clf2.score(X_test, y_test):
			return clf1
		else:
			return clf2

	model = selection(models)
	
	params = clf1.best_params_
	print(params)

	metrics = {'acc' : clf1.score(X_test, y_test)}

	#joblib.dump(model,'../apps/models/model.py')
	
	# Store information in tracking server
	with mlflow.start_run(run_name=run_name) as run:
	    mlflow.set_tag("mlflow.runName", run_name)
	    mlflow.log_metrics(metrics)
	    mlflow.log_params(params)
	    mlflow.sklearn.log_model(sk_model=model, artifact_path = artifact_path)

run_mlflow_experiment()
	    

