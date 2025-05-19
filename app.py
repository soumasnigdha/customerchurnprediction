import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle


# Load the trained model
model = tf.keras.models.load_model('models/model.keras')

# Load the scaler and encoders
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)
with open('models/label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)


# Streamlit App
st.title("Customer Churn Prediction")
st.write("This app predicts whether a customer will churn based on their information.")
st.write("Please enter the customer information below:")

## Input fields
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 19, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

## Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender],
    'Geography': [geography],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

## One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform(input_data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Label encode 'Gender'
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

# Concat input_data and geo_encoded_df
input_data = pd.concat([input_data.reset_index(drop=True).drop(columns='Geography', axis=1), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

## Predict Churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn')

st.write('Churn Probability: ', prediction_proba)