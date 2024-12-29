import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st

# Load the dataset
df = pd.read_csv('/content/kc_house_data.csv', usecols=['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'price', 'yr_built'])

# Data Preprocessing
df['bathrooms'] = df['bathrooms'].astype('int')
df['bedrooms'] = df['bedrooms'].replace(33,3)

# Streamlit header
st.title('House Price Predictor')
st.write("This app predicts house prices based on features like the number of bedrooms, bathrooms, living area, and more.")

# Show basic info and statistical data
if st.checkbox('Show Data Info'):
    st.subheader('Data Info')
    st.write(df.info())

if st.checkbox('Show Statistical Description'):
    st.subheader('Statistical Description')
    st.write(df.describe())

# Visualizations
st.subheader('Univariate Analysis')

# Bedrooms distribution
st.write('### Bedrooms Distribution')
fig1, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(df['bedrooms'], ax=axes[0])
axes[1].boxplot(df['bedrooms'])
st.pyplot(fig1)

# Bathrooms distribution
st.write('### Bathrooms Distribution')
fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(df['bathrooms'], ax=axes[0])
axes[1].boxplot(df['bathrooms'])
st.pyplot(fig2)

# Sqft Living distribution
st.write('### Sqft Living Distribution')
fig3, axes = plt.subplots(1, 2, figsize=(12, 4))
df['sqft_living'].plot(kind='kde', ax=axes[0])
axes[1].boxplot(df['sqft_living'])
st.pyplot(fig3)

# Grade distribution
st.write('### Grade Distribution')
fig4, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(df['grade'], ax=axes[0])
axes[1].boxplot(df['grade'])
st.pyplot(fig4)

# Yr Built distribution
st.write('### Year Built Distribution')
fig5, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(df['yr_built'], ax=axes[0])
axes[1].boxplot(df['yr_built'])
st.pyplot(fig5)

# Correlation
st.subheader('Correlation Matrix')
st.write(df.corr().style.background_gradient().format("{:.2f}"))

# Model Training and Prediction
st.subheader('Model Training')
x = df.drop(columns='price')
y = df['price']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

# Train the model
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

# Display coefficients
coef_dict = {
    'features': x.columns,
    'coef_value': lin_reg.coef_
}
coef = pd.DataFrame(coef_dict, columns=['features', 'coef_value'])
st.write('### Coefficients of the Model')
st.write(coef)

# Model score
score = lin_reg.score(x_test, y_test)
st.write(f'### Model Accuracy: {score:.2f}')

# User Input for Prediction
st.subheader('Predict House Price')
bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=3)
bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2)  # Fixed this line
sqft_living = st.number_input('Square Footage of Living Area', min_value=100, max_value=10000, value=1800)
grade = st.number_input('Grade of the House', min_value=1, max_value=13, value=7)
yr_built = st.number_input('Year the House Was Built', min_value=1900, max_value=2024, value=1990)

# Make prediction
prediction = lin_reg.predict([[bedrooms, bathrooms, sqft_living, grade, yr_built]])
st.write(f'Predicted House Price: ${prediction[0]:,.2f}')

