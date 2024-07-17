from fastapi import FastAPI
api = FastAPI()

import pandas as pd
import pickle as pkl

model = pkl.load(open('model/model.pkl', 'rb'))

@api.get('/')
def hello_world():
    return {'message':'hello_world'}

@api.post('/predict/')
async def predict(Pregnancies:float,Glucose:float,BloodPressure:float,SkinThickness:float,Insulin:float,BMI:float,DiabetesPedigreeFunction:float,Age:float):
  record = pd.DataFrame(columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
  record.loc[0] = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
  results = model.predict(record)[0]
  print(results)
  print(type(results))
  return {"prediction": str(results)}
  


  
