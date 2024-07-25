import dagshub
import sys
import os
sys.path.append('/Users/sarahlenet/Desktop/MLOPS/Intro_MLOPS_tsunamis/Intro_MLOPS_tsunamis/src/scripts/')
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import mlflow
from mlflow import MlflowClient
from preprocessing import preprocess
from scale import scale
from tqdm import tqdm

path = '../../data/raw/'


# Adding noise for augmentation
def add_noise(X, noise_level=0.001):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

# predicting the with the output of a kmeans
def predict_kmeans(df, i, k):

  # Preprocess
  # X, y
  X = df.drop('human_damages', axis = 1)
  y = df['human_damages']

  # Kmeans
  km = KMeans(n_clusters = k + 1,  n_init=10)
  centroids = km.fit(X)

  # Adding column
  X['clustering'] = pd.Series(km.labels_)
  df_processed = pd.concat([X,y], axis = 1).dropna()
  X = df_processed.drop('human_damages', axis = 1)
  y = df_processed['human_damages']

  # Augment data
  X_noisy = pd.concat([X, X.apply(add_noise)], axis = 0).reset_index(drop = True)
  y_noisy = pd.concat([y, y], axis = 0).reset_index(drop = True)

  # train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X_noisy, y_noisy, test_size=0.2, random_state=42)


  # Train the regression model
  gbr = GradientBoostingRegressor()
  gbr.fit(X_train, y_train)

  # Make predictions and evaluate the model
  y_pred = gbr.predict(X_test)
  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)
  return km, gbr, r2


# experiment on best model with parameters perplexity and n_neigbors (we could also go up 80/90 if number of records went up)
def run_mlflow_experiment():
  mlflow.set_tracking_uri('https://dagshub.com/sarahlunette/Intro_MLOPS_tsunamis.mlflow')
  artifact_path = 'models'

  human_damages, houses_damages = preprocess(path) #rassembler les deux preprocessing dans un seul fichier
  human_damages_scaled, houses_damages_scaled = scale(human_damages, houses_damages)

  run_name = f"tsunamis_n_perplexity_{10}_n_clusters_{10}"
  dagshub.init(repo_name = 'Intro_MLOPS_tsunamis', repo_owner = 'sarahlunette', mlflow = True)
  with mlflow.start_run(run_name = run_name) as run:
    human_damages_scaled.dropna(inplace = True)
    km, gbr,r2 = predict_kmeans(human_damages_scaled, 10,10)
    mlflow.log_metric('r2', r2)
    mlflow.log_params({'perplexity' : 10, 'n_neighbors' : 10})
    mlflow.set_tag("mlflow.runName", run_name)
    mlflow.sklearn.log_model(km, 'kmeans')
    mlflow.sklearn.log_model(gbr, 'GBR')


run_mlflow_experiment()