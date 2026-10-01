import os
import pickle
import warnings

# Must be set before importing TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import pandas as pd
import streamlit as st
import tensorflow as tf

warnings.filterwarnings(
    "ignore",
    message=".*does not have valid feature names.*",
)


@st.cache_resource
def load_artifacts():
    # compile=False avoids "compiled metrics have yet to be built" on load
    model = tf.keras.models.load_model("model.h5", compile=False)

    with open("encoder_geo.pkl", "rb") as file:
        label_encoder_geo = pickle.load(file)

    with open("lable_encoder_gender.pkl", "rb") as file:
        label_encoder_gender = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    return model, label_encoder_geo, label_encoder_gender, scaler


model, label_encoder_geo, label_encoder_gender, scaler = load_artifacts()

st.title("Customer Churn Prediction")

geography = st.selectbox("Geography", label_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance", min_value=0.0)
credit_score = st.number_input("Credit Score", min_value=0)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0)
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary],
})

# Pass a DataFrame so feature names match how OneHotEncoder was fitted
geo_df = pd.DataFrame({"Geography": [geography]})
geo_encoded = label_encoder_geo.transform(geo_df).toarray()
geo_columns = label_encoder_geo.get_feature_names_out(["Geography"])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_columns)

input_data = pd.concat([input_data, geo_encoded_df], axis=1)
input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled, verbose=0)
prediction_proba = float(prediction[0][0])

st.write(f"Churn Probability: {prediction_proba:.2%}")

if prediction_proba > 0.5:
    st.warning("The customer is likely to churn.")
else:
    st.success("The customer is not likely to churn.")
