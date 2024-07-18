from fastapi import FastAPI
from predict import predict_record

api = FastAPI()

import pandas as pd
import pickle as pkl

model = pkl.load(open('model/model.pkl', 'rb'))

Class User_Input(BaseModel):
  month : int
  day : int
  country : int
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

@api.post('/predict/')
async def predict(input:User_Input):
  # Load JSON data
  data = json.loads(input)
  # Convert the entire record to a pandas Series
  record = pd.Series(data)
  result = predict_record(record, model)
  return {"prediction": str(result)}
  


  
