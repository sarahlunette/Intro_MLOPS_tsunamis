import sys
import os
sys.path.append('/Users/sarahlenet/Desktop/MLOPS/Intro_MLOPS_tsunamis/Intro_MLOPS_tsunamis/src/scripts/')
import pandas as pd
import numpy as np
import sklearn
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import mlflow
from mlflow import MlflowClient
from preprocessing import preprocess
from scale import scale


path = '../../data/raw/'


# Adding noise for augmentation
def add_noise(X, noise_level=0.001):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

# predicting the with the output of a tsne
def predict_tsne(df, i, k):
  # Instantiate TSNE
  tsne = TSNE(n_components=2,perplexity=i + 1, random_state=42)
  X = df.drop('human_damages', axis = 1)
  X_tsne = tsne.fit_transform(X)

  y = df['human_damages']

  # KMeans
  km = KMeans(n_clusters= k + 1)
  centroids = km.fit(X_tsne)
  X['clustering'] = pd.Series(km.labels_)

  # Dropping nan values
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
  tsunamis_experiment = mlflow.set_experiment('tsunamis_experiment_perplexity_n_neighbors')
  artifact_path = 'models'

  human_damages, houses_damages = preprocess(path) #rassembler les deux preprocessing dans un seul fichier
  human_damages_scaled, houses_damages_scaled = scale(human_damages, houses_damages)

  score = pd.DataFrame(columns = range(80), index = range(90))
  mlflow.set_tracking_uri('https://dagshub.com/sarahlunette/Intro_MLOPS_tsunamis.mlflow')
  for k in tqdm(range(2, 90, 1), desc="Outer loop"):
    for i in tqdm(range(80), desc="Inner loop"):
      run_name = f"tsunamis_n_perplexity_{str(i + 1)}_n_clusters_{str(k + 1)}"
      with mlflow.start_run(run_name = run_name) as run:
        human_damages_scaled.dropna(inplace = True)
        km, gbr,r2 = predict_tsne(human_damages_scaled, i, k)
        score.iloc[k, i] = r2
        mlflow.log_metrics(r2)
        mlflow.log_params({'perplexity' : i + 1, 'n_neighbors' : k + 1})
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.sklearn.log_model(km, 'kmeans')
        mlflow.sklearn.log_model(gbr, 'GBR')

  run_name = f"tsunamis_best_p"
  with mlflow.start_run(run_name = run_name) as run:
    max_value = score.values.max()
    flat_index = score.values.argmax()
    row_index, col_index = np.unravel_index(flat_index, score.shape)
    gbr, r2 = predict_tsne(human_damages_scaled, col_index + 1, row_index + 1)

    mlflow.log_metric('r2', r2)
    mlflow.log_params({'perplexity' : col_index + 1, 'n_neighbors' : row_index + 1})
    mlflow.set_tag("mlflow.runName", run_name)
    mlflow.sklearn.log_model(gbr, 'GBR')

run_mlflow_experiment()