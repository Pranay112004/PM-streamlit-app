# STREAMLIT APP: Mumbai House Price Prediction (Based on Locality + BHK)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

# Title
st.title("🏙️ Mumbai House Price Prediction App")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('mumbai_cleaned_data_original.csv')
    return df

df = load_data()

# Display raw data
if st.checkbox('Show raw data'):
    st.write(df)

# Prepare data
df['bhk_num'] = df['bhk'].str.extract('(\d+)').astype(int)

# Train model
X = df[['locality', 'bhk_num']]
y = df['price_lakhs']

# One-Hot Encoding for Locality
X_encoded = pd.get_dummies(X, columns=['locality'], drop_first=True)

# Train model
model = LinearRegression()
model.fit(X_encoded, y)

# Sidebar Inputs
st.sidebar.header('Provide Input Details:')

# Get list of unique localities
locality_list = sorted(df['locality'].unique())

selected_locality = st.sidebar.selectbox('Select Locality', locality_list)
selected_bhk = st.sidebar.number_input('Enter number of Bedrooms (BHK)', min_value=1, max_value=10, value=2)

# Prediction
if st.sidebar.button('Predict Price'):
    # Create input dataframe
    input_dict = {col: 0 for col in X_encoded.columns}
    input_dict['bhk_num'] = selected_bhk
    
    locality_col = f'locality_{selected_locality}'
    if locality_col in input_dict:
        input_dict[locality_col] = 1
    else:
        st.error("Selected locality not found in training data. Please try another.")

    input_df = pd.DataFrame([input_dict])

    predicted_price = model.predict(input_df)[0]
    st.success(f"🏡 Predicted Price: ₹{predicted_price:.2f} Lakhs")

# Footer
st.caption('Developed by You 💻')
