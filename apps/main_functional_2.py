from fastapi import FastAPI
from predict import predict_record
from pydantic import BaseModel


api = FastAPI()

import pandas as pd
import pickle as pkl

model = pkl.load(open('model/model.pkl', 'rb'))

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
  data = input_data.dict()
  data[country] = data[input_data.country]
  del data[input_data.country]

  ## Convert the entire record to a pandas Series
  # Change the values to lists of values
  for key in data.keys():
    data[key] = [data[key]] # You have to remove country from this

  ## create the dataframe for predictions
  record_predict = pd.DataFrame(columns = columns_final, index = [0])
  # Fill the values with the user input
  record_predict[record.columns] = record
  # Fill the rest with 0 because they are not in these countries
  record_predict.fillna(0, inplace = True) # That is correct only if the other values are nan

  ## TO-DO: Add KMeans for filling clustering column and the rest

  print(record)
  print(record.shape)
  result = predict_record(record, model)
  return {"prediction": str(result)}
  


  
