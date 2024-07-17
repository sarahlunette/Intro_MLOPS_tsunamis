def cleaning(df):
  df = df.fillna(df.mean())
  df.drop_duplicates(inplace = True)
  return df
