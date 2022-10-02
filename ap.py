import streamlit as st 
import numpy as np
import pandas as pd
import pickle
from pickle import load
from PIL import Image

c1,c2, = st.columns(2)
with c1:
    st.title("Diamond Price Prediction Application")
with c2:
    image = Image.open('diamond.jpg')
    st.image(image)

xg_model = pickle.load(open('xgboost_regressor.pkl','rb'))

clarity_enc = {'I1':0,'SI2':3,'SI1':2,'VS2':5,'VS1':4,'VVS2':7,'VVS1':6,'IF':1}
color_enc = {'J':6,'I':5,'H':4,'G':3,'F':2,'E':1,'D':0}
cut_enc = {'Fair':0,'Good':1,'Very Good':4,'Ideal': 2,'Premium':3}

column1, column2 = st.columns(2)

with column1:
    carat = st.number_input('Enter Carat value in mm (range 0.2-5.01)')
    x = st.number_input('Enter x length value in mm (range 3.73-10.74)')
    y = st.number_input('Enter y width value in mm (range 3.68-58.90)')
    z = st.number_input('Enter z depth value in mm (range 1.07-31.80)')

with column2:
    cut = st.selectbox(
        'How would you like the cut of Diamond to be?',
        ('select option','Fair', 'Good', 'Very Good', 'Ideal', 'Premium'))

    color = st.selectbox(
        'How would you like the color of Diamond to be?',
        ('select option','J', 'I', 'H', 'G', 'F', 'E', 'D'))

    clarity = st.selectbox(
        'How would you like the clarity of diamond to be?',
        ('select option','I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'))



btn_click=st.button("Click button to Predict the Diamond Price")
# order of inputs is carat	cut	color	clarity	x	y	z

if btn_click == True:
    x_sample = np.array([[float(carat), cut_enc[cut], color_enc[color], clarity_enc[clarity], float(x), float(y), float(z)]], dtype = float)

    predicted_price = xg_model.predict(x_sample)
    st.write('The price of diamond is going to be {} $'. format(predicted_price[0]))
    st.success('The prediction was success', icon="✅")
    st.snow()

# for xgboost we got error of Unicode-3 is not supported. to overcome this we added dtypes in numpy array
# the error of ran out of input was takled by using pickle_out.close() in python script