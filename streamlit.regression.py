import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# Load the trained model
model = tf.keras.models.load_model('salary_regression_model.h5')

# Load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit App
# User input

geography = st.selectbox(
    'Geography',
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    'Gender',
    label_encoder_gender.classes_
)

age = st.slider('Age', 18, 92)

balance = st.number_input('Balance')

credit_score = st.number_input('Credit Score')

tenure = st.slider('Tenure', 0, 10)

num_of_products = st.slider('Number of Products', 1, 4)

has_cr_card = st.selectbox(
    'Has Credit Card',
    [0, 1]
)

is_active_member = st.selectbox(
    'Is Active Member',
    [0, 1]
)

exited = st.selectbox(
    'Exited',
    [0, 1]
)

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Combine encoded geography with input data
input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df],
    axis=1
)

# Reorder columns to match scaler training
input_data = input_data[
    [
        'CreditScore',
        'Gender',
        'Age',
        'Tenure',
        'Balance',
        'NumOfProducts',
        'HasCrCard',
        'IsActiveMember',
        'Exited',
        'Geography_France',
        'Geography_Germany',
        'Geography_Spain'
    ]
]

# Scale input data
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):

    prediction = model.predict(input_scaled)
    prediction_value = prediction[0][0]

    st.write(f'Prediction: {prediction_value:.4f}')

    st.success("Prediction completed successfully.")
