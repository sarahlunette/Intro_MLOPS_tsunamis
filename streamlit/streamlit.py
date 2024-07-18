import streamlit as st
import requests
import json

st.title('Simple Predictions of Tsunamis Casualties')



# Possible to put the analysis and solutions (do simulations)

data = # entered by the user
requests.post(url = "http://127.0.0.1:8000/"+data, data = json.dumps(inputs))
