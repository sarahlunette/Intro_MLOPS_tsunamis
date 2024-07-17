import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

# set the experiment id
mlflow.set_experiment("other_experiment")

mlflow.autolog()
db = load_diabetes()

X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models.
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)

# Store information in tracking server
with mlflow.start_run(run_name='experiment') as run:
	mlflow.set_tag("mlflow.runName", 'experiment')
	mlflow.sklearn.log_model(sk_model=rf, artifact_path = 'model')
