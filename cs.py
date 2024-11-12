# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load and preprocess the dataset
st.title("ML Project: Classification with Streamlit")

@st.cache_data
def load_data(file):
    data = pd.read_csv(file, encoding='latin-1')
    data.dropna(inplace=True)
    encoders = {}
    columns_to_encode = ['DESCRIPTION', 'REFERENCE', 'TYPE', 'IS_RECONCILED',
                         'ACCOUNT_CLASS', 'ACCOUNT_TYPE', 'TRANSACTION_TOTAL',
                         'CODE', 'ACCOUNT_NAME']
    for col in columns_to_encode:
        encoders[col] = LabelEncoder()
        data[col] = encoders[col].fit_transform(data[col])
    data = data.drop('TRANSACTION_DATE', axis=1)
    return data, encoders

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    data, encoders = load_data(uploaded_file)
    st.write("### Dataset Preview", data.head())

    # Split data into features and target
    X = data[['DESCRIPTION', 'REFERENCE', 'TYPE', 'TRANSACTION_TOTAL', 
              'IS_RECONCILED', 'ACCOUNT_CLASS', 'CODE', 'ACCOUNT_TYPE']]
    y = data['ACCOUNT_NAME']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection
    model_choice = st.selectbox("Select a Model", ["Decision Tree", "Logistic Regression"])
    
    if model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)

    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Display accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"### Model Accuracy: {accuracy:.2f}")

    # User input for prediction
    st.write("## Predict New Data")

    with st.form("prediction_form"):
        description = st.selectbox("DESCRIPTION", encoders['DESCRIPTION'].inverse_transform(data['DESCRIPTION'].unique()))
        reference = st.selectbox("REFERENCE", encoders['REFERENCE'].inverse_transform(data['REFERENCE'].unique()))
        type_ = st.selectbox("TYPE", encoders['TYPE'].inverse_transform(data['TYPE'].unique()))
        transaction_total = st.selectbox("TRANSACTION_TOTAL", encoders['TRANSACTION_TOTAL'].inverse_transform(data['TRANSACTION_TOTAL'].unique()))
        is_reconciled = st.selectbox("IS_RECONCILED", encoders['IS_RECONCILED'].inverse_transform(data['IS_RECONCILED'].unique()))
        account_class = st.selectbox("ACCOUNT_CLASS", encoders['ACCOUNT_CLASS'].inverse_transform(data['ACCOUNT_CLASS'].unique()))
        code = st.selectbox("CODE", encoders['CODE'].inverse_transform(data['CODE'].unique()))
        account_type = st.selectbox("ACCOUNT_TYPE", encoders['ACCOUNT_TYPE'].inverse_transform(data['ACCOUNT_TYPE'].unique()))

        # Submit button
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            new_data = pd.DataFrame({
                'DESCRIPTION': [encoders['DESCRIPTION'].transform([description])[0]],
                'REFERENCE': [encoders['REFERENCE'].transform([reference])[0]],
                'TYPE': [encoders['TYPE'].transform([type_])[0]],
                'TRANSACTION_TOTAL': [encoders['TRANSACTION_TOTAL'].transform([transaction_total])[0]],
                'IS_RECONCILED': [encoders['IS_RECONCILED'].transform([is_reconciled])[0]],
                'ACCOUNT_CLASS': [encoders['ACCOUNT_CLASS'].transform([account_class])[0]],
                'CODE': [encoders['CODE'].transform([code])[0]],
                'ACCOUNT_TYPE': [encoders['ACCOUNT_TYPE'].transform([account_type])[0]]
            })
            
            # Get probabilities for all classes
            probabilities = model.predict_proba(new_data)[0]
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            top_3_classes = encoders['ACCOUNT_NAME'].inverse_transform(top_3_indices)
            top_3_probs = probabilities[top_3_indices]

            st.write("### Top 3 Predictions with Confidence Scores")
            for i in range(3):
                st.write(f"{i+1}. **{top_3_classes[i]}** with confidence **{top_3_probs[i]:.2f}**")
