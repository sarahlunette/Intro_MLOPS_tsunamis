import pandas as pd
import numpy as np
from skleanr.preprocessing import StandardScaler

def standardize(df):
  sc = StandardScaler()
  df = sc.fit_transform(df)
  return df

def extreme_values(df, threshold = 0.00001):
  for col in df.columns:
    df[col] = np.where(df[col] > threshold, threshold, df[col])
  return df
