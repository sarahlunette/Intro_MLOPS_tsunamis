import sys
import os
sys.path.append('/Users/sarahlenet/Desktop/MLOPS/Intro_MLOPS_tsunamis/Intro_MLOPS_tsunamis/src/scripts/')
from fastapi import FastAPI
from predict import predict_record
from pydantic import BaseModel
import pickle as pkl
import numpy as np
from sklearn.neighbors import NearestNeighbors


api = FastAPI()

import pandas as pd
import pickle as pkl

# Load the model
try:
    with open('model/model.pkl', 'rb') as model_file:
        model = pkl.load(model_file)
except FileNotFoundError:
    raise FileNotFoundError("Model file not found. Please check the file path.")

# Load the KMeans
try:
    with open('model/kmeans_model.pkl', 'rb') as model_file:
        km = pkl.load(model_file)
except FileNotFoundError:
    raise FileNotFoundError("Model file not found. Please check the file path.")

# Load the TSNE
try:
    with open('model/tsne.pkl', 'rb') as model_file:
        tsne_model = pkl.load(model_file)
except FileNotFoundError:
    raise FileNotFoundError("Model file not found. Please check the file path.")

X_tsne = np.load('tsne_embeddings.npy')

# Function to transform new data based on the saved TSNE model
def transform_new_data(tsne_model, X_tsne, new_data):
    # Use nearest neighbors to find the closest points in the original space
    nbrs = NearestNeighbors(n_neighbors=1).fit(tsne_model.embedding_)
    distances, indices = nbrs.kneighbors(new_data)
    transformed_data = X_tsne[indices.flatten()]
    return transformed_data


class InputData(BaseModel):
  month : int
  day : int
  country : str
  period : int
  latitude : float
  longitude : float
  runup_ht : float
  runup_ht_r : float
  runup_hori : float
  dist_from_ : float
  hour : float
  cause_code : float
  event_vali : float
  eq_mag_unk : float
  eq_mag_mb : float
  eq_mag_ms : float
  eq_mag_mw : float
  eq_mag_mfa : float
  eq_magnitu : float
  eq_magni_1 : float
  eq_depth : float
  max_event_ : float
  ts_mt_ii : float
  ts_intensi : float
  num_runup : float
  num_slides : float
  map_slide_ : float
  map_eq_id : float

columns = [
 'month',
 'day',
 'country',
 'period',
 'latitude',
 'longitude',
 'runup_ht',
 'runup_ht_r',
 'runup_hori',
 'dist_from_',
 'hour',
 'cause_code',
 'event_vali',
 'eq_mag_unk',
 'eq_mag_mb',
 'eq_mag_ms',
 'eq_mag_mw',
 'eq_mag_mfa',
 'eq_magnitu',
 'eq_magni_1',
 'eq_depth',
 'max_event_',
 'ts_mt_ii',
 'ts_intensi',
 'num_runup',
 'num_slides',
 'map_slide_',
 'map_eq_id',
]

columns_final = ['month', 'day', 'period', 'latitude', 'longitude', 'runup_ht',
       'runup_ht_r', 'runup_hori', 'dist_from_', 'hour', 'cause_code',
       'event_vali', 'eq_mag_unk', 'eq_mag_mb', 'eq_mag_ms', 'eq_mag_mw',
       'eq_mag_mfa', 'eq_magnitu', 'eq_magni_1', 'eq_depth', 'max_event_',
       'ts_mt_ii', 'ts_intensi', 'num_runup', 'num_slides', 'map_slide_',
       'map_eq_id', 'country_bangladesh', 'country_canada', 'country_chile',
       'country_china', 'country_dominican republic', 'country_egypt',
       'country_el salvador', 'country_fiji', 'country_greece',
       'country_haiti', 'country_india', 'country_indonesia',
       'country_jamaica', 'country_japan', 'country_kenya',
       'country_madagascar', 'country_malaysia', 'country_maldives',
       'country_mexico', 'country_myanmar', 'country_new caledonia',
       'country_nicaragua', 'country_pakistan', 'country_papua new guinea',
       'country_peru', 'country_philippines', 'country_samoa',
       'country_solomon islands', 'country_somalia', 'country_sri lanka',
       'country_taiwan', 'country_tanzania', 'country_tonga', 'country_turkey',
       'country_united kingdom', 'country_united states', 'country_vanuatu',
       'country_venezuela', 'country_yemen']

@api.post('/predict/')
async def predict(input_data:InputData):
    # Change the country name key
    country = 'country_' + input_data.country
    print(input_data.country)
    data = input_data.dict()
    print(data)
    del data['country']
    for key in data.keys():
        data[key] = [data[key]]
    data[country] = 1  # Set the corresponding country column to 1

    # Create a DataFrame with the input data
    record = pd.DataFrame.from_dict(data)

    # Ensure all columns are present, filling missing columns with 0
    record = record.reindex(columns=columns_final, fill_value=0)

    # Add KMeans for filling clustering column and the rest (save Kmeans and load model)
    record['clustering'] = km.predict(transform_new_data(tsne, X_tsne, record)).labels_

    print(record)
    print(record.shape)

    try:
        prediction = model.predict(record)
        # Assuming predict_record function is not needed if model.predict() works directly
        # result = predict_record(record, model)
        return {"prediction": str(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to make prediction: {str(e)}")

  
