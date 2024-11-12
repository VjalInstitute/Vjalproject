import streamlit as st
import pandas as pd
import joblib

# Load the trained Decision Tree model
model = joblib.load('assembled_model.pkl')

# Title of the web app
st.title("Decision Tree Model Deployment with Streamlit")

# Input form for user data
st.subheader("Provide input for prediction")

# Reduced feature set
features = [
    'DESCRIPTION', 'REFERENCE', 'TYPE', 'TRANSACTION_TOTAL',
    'IS_RECONCILED', 'ACCOUNT_CLASS', 'CODE', 'ACCOUNT_TYPE'
]

# Collect user input for each feature
user_data = {}
for feature in features:
    user_data[feature] = st.text_input(feature) if feature in ['DESCRIPTION', 'REFERENCE', 'TYPE', 'ACCOUNT_CLASS', 'ACCOUNT_TYPE'] else st.number_input(feature, value=0.0)

# Convert user input to DataFrame
input_df = pd.DataFrame([user_data])

# Predict and display results
if st.button("Predict"):
    # Encode categorical inputs the same way as in training
    for col in input_df.columns:
        if input_df[col].dtype == 'object':
            input_df[col] = input_df[col].astype(str).apply(lambda x: x)  # Dummy logic if necessary
    
    # Predict and display the result
    prediction = model.predict(input_df)
    st.write(f"Predicted ACCOUNT_NAME: {prediction[0]}")
