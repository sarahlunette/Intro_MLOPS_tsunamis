import streamlit as st
import requests
import json

st.title('Simple Predictions of Tsunamis Casualties')

columns_final = ['month', 'day', 'period', 'latitude', 'longitude', 'runup_ht',
       'runup_ht_r', 'runup_hori', 'dist_from_', 'hour', 'cause_code',
       'event_vali', 'eq_mag_unk', 'eq_mag_mb', 'eq_mag_ms', 'eq_mag_mw',
       'eq_mag_mfa', 'eq_magnitu', 'eq_magni_1', 'eq_depth', 'max_event_',
       'ts_mt_ii', 'ts_intensi', 'num_runup', 'num_slides', 'map_slide_',
       'map_eq_id', 'country_bangladesh', 'country_canada', 'country_chile',
       'country_china', 'country_colombia', 'country_costa rica',
       'country_dominican republic', 'country_ecuador', 'country_egypt',
       'country_el salvador', 'country_fiji', 'country_france',
       'country_french polynesia', 'country_greece', 'country_haiti',
       'country_india', 'country_indonesia', 'country_italy',
       'country_jamaica', 'country_japan', 'country_kenya',
       'country_madagascar', 'country_malaysia', 'country_maldives',
       'country_mexico', 'country_micronesia', 'country_myanmar',
       'country_new caledonia', 'country_new zealand', 'country_nicaragua',
       'country_norway', 'country_pakistan', 'country_panama',
       'country_papua new guinea', 'country_peru', 'country_philippines',
       'country_portugal', 'country_russia', 'country_samoa',
       'country_solomon islands', 'country_somalia', 'country_south korea',
       'country_spain', 'country_sri lanka', 'country_taiwan',
       'country_tanzania', 'country_tonga', 'country_turkey',
       'country_united kingdom', 'country_united states', 'country_vanuatu',
       'country_venezuela', 'country_yemen']

countries =  ['country_bangladesh', 'country_canada', 'country_chile',
       'country_china', 'country_colombia', 'country_costa rica',
       'country_dominican republic', 'country_ecuador', 'country_egypt',
       'country_el salvador', 'country_fiji', 'country_france',
       'country_french polynesia', 'country_greece', 'country_haiti',
       'country_india', 'country_indonesia', 'country_italy',
       'country_jamaica', 'country_japan', 'country_kenya',
       'country_madagascar', 'country_malaysia', 'country_maldives',
       'country_mexico', 'country_micronesia', 'country_myanmar',
       'country_new caledonia', 'country_new zealand', 'country_nicaragua',
       'country_norway', 'country_pakistan', 'country_panama',
       'country_papua new guinea', 'country_peru', 'country_philippines',
       'country_portugal', 'country_russia', 'country_samoa',
       'country_solomon islands', 'country_somalia', 'country_south korea',
       'country_spain', 'country_sri lanka', 'country_taiwan',
       'country_tanzania', 'country_tonga', 'country_turkey',
       'country_united kingdom', 'country_united states', 'country_vanuatu',
       'country_venezuela', 'country_yemen']

columns = ['month', 'day', 'period', 'latitude', 'longitude', 'runup_ht',
       'runup_ht_r', 'runup_hori', 'dist_from_', 'hour', 'cause_code',
       'event_vali', 'eq_mag_unk', 'eq_mag_mb', 'eq_mag_ms', 'eq_mag_mw',
       'eq_mag_mfa', 'eq_magnitu', 'eq_magni_1', 'eq_depth', 'max_event_',
       'ts_mt_ii', 'ts_intensi', 'num_runup', 'num_slides', 'map_slide_',
       'map_eq_id']

# Possible to put the analysis and solutions (do simulations)

option = st.selectbox(
    "Country",
    tuple(countries)) # to change later

st.write("You selected:", option)

# User input
data = dict()
data['country'] = option

for column in column:
	input_ = st.number_input('Input for ' + column, value=0.0)
	data[column] = input_ 

headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}

response = requests.post(url = "http://127.0.0.1:8000/predict/", headers = headers, data = json.dumps(inputs)).json()


st.write('Prediction of the number of casualties')
st.write(response['prediction'])
