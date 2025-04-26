# app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import joblib

# Matplotlib backend for Streamlit
matplotlib.use('Agg')

st.set_page_config(layout="wide", page_title="Mumbai House Price Analysis")

# Title
st.title("🏠 Mumbai House Price Prediction App")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('mumbai-house-price-data-cleaned.csv')
    return df

df = load_data()

# Sidebar Menu
menu = ['Home', 'EDA (Exploratory Data Analysis)', 'Model Training', 'Predict Price']
choice = st.sidebar.selectbox("Navigation", menu)

# --- HOME PAGE ---
if choice == 'Home':
    st.subheader("📋 About the Project")
    st.write("""
    This project analyzes real estate price trends in Mumbai 📈.
    We clean the data, visualize trends, train ML models, and make price predictions.
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# --- EDA PAGE ---
elif choice == 'EDA (Exploratory Data Analysis)':
    st.header("📊 Exploratory Data Analysis")

    if st.checkbox("Show Dataset"):
        st.dataframe(df)

    st.subheader("Heatmap of Correlations")
    if st.button('Show Heatmap'):
        numeric_df = df.select_dtypes(include='number')
        fig, ax = plt.subplots(figsize=(12,8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    st.subheader("Boxplots for Numerical Features")
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    selected_feature = st.selectbox("Select Feature for Boxplot", numerical_features)

    fig, ax = plt.subplots()
    sns.boxplot(x=df[selected_feature], ax=ax)
    st.pyplot(fig)

    st.subheader("Distribution Plots")
    feature = st.selectbox("Select Feature for Distribution Plot", numerical_features)
    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, bins=30, ax=ax)
    st.pyplot(fig)

# --- MODEL TRAINING PAGE ---
elif choice == 'Model Training':
    st.header("🤖 Model Training and Evaluation")

    st.info("We train Linear Regression, Random Forest, and XGBoost models")

    # Data Preprocessing
    df_clean = df.copy()
    df_clean['title'].fillna('Unknown', inplace=True)
    necessary_columns = ['price_per_sqft', 'price', 'locality', 'bedroom_num', 'area']
    df_clean = df_clean[necessary_columns]
    df_clean['price_lakhs'] = (df_clean['price'] / 100000).round(0).astype(int)
    df_clean.drop(['price'], axis=1, inplace=True)

    df_clean['bedroom_num'] = df_clean['bedroom_num'].fillna(0).astype(int).astype(str)

    df_clean = df_clean.dropna(subset=['price_per_sqft', 'locality', 'bedroom_num', 'area'])

    df_clean['locality'] = df_clean['locality'].str.title()

    df_clean = df_clean.rename(columns={
        'price_per_sqft': 'price_per_sqft',
        'locality': 'locality',
        'bedroom_num': 'bhk',
        'area': 'area_sqft'
    })

    df_clean['price_per_sqft'] = df_clean['price_per_sqft'].round(0).astype(int)

    X = df_clean[['locality', 'bhk', 'area_sqft', 'price_per_sqft']]
    y = np.log1p(df_clean['price_lakhs'])

    # Preprocessing Pipeline
    numerical_cols = ['area_sqft', 'price_per_sqft']
    categorical_cols = ['locality', 'bhk']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(random_state=42)
    }

    for name, model in models.items():
        st.subheader(name)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_exp = np.expm1(y_pred)
        y_test_exp = np.expm1(y_test)
        r2 = r2_score(y_test_exp, y_pred_exp)

        st.write(f"**R² Score:** {r2:.4f}")
        joblib.dump(pipeline, f'{name.lower().replace(" ", "_")}_model.pkl')

# --- PREDICT PAGE ---
elif choice == 'Predict Price':
    st.header("🔮 Predict Mumbai House Price")

    # Load model
    model_choice = st.selectbox("Choose Model", ['Linear Regression', 'Random Forest', 'XGBoost'])
    model_file = model_choice.lower().replace(" ", "_") + "_model.pkl"
    model = joblib.load(model_file)

    # Input fields
    st.subheader("Input Features")
    locality = st.text_input("Locality (Title Case Example: Andheri East)")
    bhk = st.text_input("Number of Bedrooms (e.g., 2)")
    area_sqft = st.number_input("Area (sqft)", min_value=100, max_value=10000, value=500)
    price_per_sqft = st.number_input("Price per sqft", min_value=1000, max_value=100000, value=10000)

    if st.button('Predict'):
        input_df = pd.DataFrame([[locality, bhk, area_sqft, price_per_sqft]],
                                columns=['locality', 'bhk', 'area_sqft', 'price_per_sqft'])
        pred = model.predict(input_df)
        predicted_price = np.expm1(pred)[0]
        st.success(f"🏡 Predicted Price: ₹ {predicted_price:.2f} Lakhs")
