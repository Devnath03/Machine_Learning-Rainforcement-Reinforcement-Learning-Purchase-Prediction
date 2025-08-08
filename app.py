# ==========================================
# iPhone Purchase Prediction Web App
# ==========================================

# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------
# Load the trained model
# ------------------------------------------
# Make sure 'model.joblib' was trained with the same feature names: ['Age', 'Salary']
model = joblib.load('model.joblib')

# ------------------------------------------
# App title and description
# ------------------------------------------
st.title('📱 iPhone Purchase Prediction')
st.write(
    'This machine learning model predicts the likelihood of a customer purchasing '
    'an iPhone based on their **Age** and **Estimated Salary**.'
)

# ------------------------------------------
# Sidebar - User Inputs
# ------------------------------------------
st.sidebar.header('User Input Features')

def user_input():
    """
    Collects user input for Age and Salary from sliders in the sidebar
    and returns a DataFrame with the correct column names.
    """
    age = st.sidebar.slider('Age', 18, 70, 30)
    salary = st.sidebar.slider('Salary', 30000, 150000, 60000)
    
    # Create dataframe with the same structure as training data
    input_df = pd.DataFrame([[age, salary]], columns=['Age', 'Salary'])
    
    # Reorder columns to match the model's training features
    if hasattr(model, "feature_names_in_"):
        input_df = input_df[model.feature_names_in_]
    
    return input_df

# Get user data
df = user_input()

# Display the entered data
st.subheader('User Input')
st.write(df)

# ------------------------------------------
# Make Predictions
# ------------------------------------------
prediction = model.predict(df)            # Returns class label (0 or 1)
prediction_proba = model.predict_proba(df)  # Returns probabilities

# Define label mapping
purchase_labels = ['Not Purchased', 'Purchased']

# ------------------------------------------
# Display Prediction
# ------------------------------------------
st.subheader('Prediction')
st.write(f"**{purchase_labels[int(prediction[0])]}**")

# ------------------------------------------
# Display Prediction Probabilities
# ------------------------------------------
st.subheader('Prediction Probability')
st.write(f'Purchased Probability: {prediction_proba[0][1] * 100:.2f}%')
st.write(f'Not Purchased Probability: {prediction_proba[0][0] * 100:.2f}%')

# ------------------------------------------
# Data Visualization (Example dataset)
# ------------------------------------------
# Load your historical dataset (must have same features + target column)
# Change 'data.csv' to your actual dataset file path
try:
    data = pd.read_csv("iphone_purchase_records.csv")
    
    # Ensure the target column exists; rename if needed
    # If your CSV has 'Purchased' instead of 'Purchase IPhone', rename it
    if 'Purchased' in data.columns and 'Purchase Iphone' not in data.columns:
        data.rename(columns={'Purchased': 'Purchase Iphone'}, inplace=True)

    if 'Purchase Iphone' in data.columns:
        st.subheader("📊 Data Distribution by Purchase Status")
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Age distribution
        sns.histplot(data, x='Age', hue='Purchase Iphone', kde=True, ax=ax[0])
        ax[0].set_title('Age Distribution by Purchase')
        
        # Salary distribution
        sns.histplot(data, x='Salary', hue='Purchase Iphone', kde=True, ax=ax[1])
        ax[1].set_title('Salary Distribution by Purchase')
        
        st.pyplot(fig)
    else:
        st.warning("⚠ The dataset does not contain a 'Purchase Iphone' column for visualization.")
except FileNotFoundError:
    st.info("ℹ No dataset file found for visualization. Upload 'iphone_purchase_records.csv' to enable charts.")
    

