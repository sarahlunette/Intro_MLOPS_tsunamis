# import necessary packages
import pandas as pd
import numpy as np

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

path = '../src/scripts/'
def preprocess(df):
	df.columns = df.columns.str.lower()

	gdp_per_capita_dict = {
	    "afghanistan": 547,
	    "bhutan": 3162,
	    "channel islands": 37235,
	    "cuba": 8616,
	    "eritrea": 617,
	    "gibraltar": 81000,
	    "greenland": 45167,
	    "guam": 34529,
	    "isle of man": 105375,
	    "Not classified": 30000,
	    "lebanon": 7556,
	    "liechtenstein": 160750,
	    "st. martin (french part)": 14000,
	    "northern mariana islands": 16200,
	    "palau": 14500,
	    "korea, dem. people's rep.": 1140,
	    "san marino": 69000,
	    "south sudan": 268,
	    "syrian arab republic": 919,
	    "tonga": 5000,
	    "venezuela, rb": 6800,
	    "british virgin islands": 33333,
	    "virgin islands (u.s.)": 38000,
	    "taiwan":35129,
	    "venezuela": 15975,
	    "turkey": 10674,
	    "yemen": 650,
	    "egypt": 4295,
	    "south korea": 32422, 
	    "russia":15262,
	    "micronesia": 3714
	}

	dict_replace = {'usa': 'united states', 'usa territory':'united states', 'myanmar (burma)':'myanmar', 'uk territory':'united kingdom', 'east china sea':'china',
	 'micronesia, fed. states of':'micronesia', 'east timor': 'indonesia', 'azores (portugal)':'portugal', 'uk':'united kingdom', 'east china sea':'china',
	 'cook islands': 'france', 'martinique (french territory)':'france'}

	# We acquire the data and then rename the columns
	gdp = pd.read_csv(path + 'gdp_per_capita.csv', on_bad_lines ='skip', sep = ',')
	gdp = gdp[['Country Name', '2022']].rename({'Country Name':'country', '2022':'gdp_per_capita'}, axis = 1)

	# Then we convert all the values to lowercase
	gdp['country'] = gdp['country'].str.lower()
	df['country'] = df['country'].str.lower()

	# We handle outliers and missing values
	gdp = gdp[gdp['country'] != 'Not classified']
	gdp['country'] = gdp['country'].replace(dict_replace)
	# On merge
	df['country'] = df['country'].replace(dict_replace)
	data = df.merge(gdp, on = 'country', how = 'left')

	# We do the same thing with population
	population = pd.read_csv(path + 'countries-by-population-density-_-countries-by-density-2024.csv')
	population['country'] = population['country'].str.lower()
	population['country'] = population['country'].replace(dict_replace)

	data = data.merge(population, on = 'country', how = 'left')

	data['GDP_per_capita'] = data['country'].apply(lambda x: gdp_per_capita_dict.get(x))
	data['gdp_per_capita'] = data['gdp_per_capita'].fillna(data['GDP_per_capita'])
	data.drop('GDP_per_capita', axis = 1, inplace = True)


	data = data.reset_index(drop = True)

	X = pd.get_dummies(data.select_dtypes('object'))
	data = pd.concat([data.drop('country', axis = 1), X], axis = 1)
	print(len(data.columns), data.columns)

	return data
