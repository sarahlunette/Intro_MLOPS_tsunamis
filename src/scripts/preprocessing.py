import pandas as pd
import numpy as np

df = pd.read_csv('/data/diabetes.csv')

# Drop nan
# Drop duplicates
df.dropna(inplace = True)
df.drop_duplicates(inplace = True)