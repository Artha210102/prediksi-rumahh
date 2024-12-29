import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the data
@st.cache
def load_data():
    df = pd.read_csv('kc_house_data.csv', usecols=['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'price', 'yr_built'])
    # Data cleaning
    df['bathrooms'] = df['bathrooms'].astype('int')
    df['bedrooms'] = df['bedrooms'].replace(33, 3)
    return df

# Application layout
def main():
    st.title("House Price Prediction App")
    st.sidebar.title("Navigation")
    options = ["Overview", "Data Analysis", "Model Training", "Predict House Price"]
    choice = st.sidebar.selectbox("Choose an option", options)

    df = load_data()

    if choice == "Overview":
        st.subheader("Overview of the Dataset")
        st.write(df.head())
        st.write("Shape of the dataset:", df.shape)
        st.write("Dataset Information:")
        st.write(df.info())
        st.write("Statistical Summary:")
        st.write(df.describe())

    elif choice == "Data Analysis":
        st.subheader("Univariate Analysis")
        st.write("### Bedrooms Distribution")
        f, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.countplot(df['bedrooms'], ax=axes[0])
        axes[1].boxplot(df['bedrooms'])
        st.pyplot(f)

        st.write("### Bathrooms Distribution")
        f, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.countplot(df['bathrooms'], ax=axes[0])
        axes[1].boxplot(df['bathrooms'])
        st.pyplot(f)

        st.write("### Sqft Living Distribution")
        f, axes = plt.subplots(1, 2, figsize=(12, 4))
        df['sqft_living'].plot(kind='kde', ax=axes[0])
        axes[1].boxplot(df['sqft_living'])
        st.pyplot(f)

        st.write("### Grade Distribution")
        f, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.countplot(df['grade'], ax=axes[0])
        axes[1].boxplot(df['grade'])
        st.pyplot(f)

        st.write("### Year Built Distribution")
        f, axes = plt.subplots(1, 2, figsize=(20, 8))
        sns.countplot(df['yr_built'], ax=axes[0])
        axes[1].boxplot(df['yr_built'])
        st.pyplot(f)

        st.write("### Pairplot Analysis")
        sns.pairplot(data=df, x_vars=['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'yr_built'], y_vars=['price'], height=4, aspect=1)
        st.pyplot()

        st.write("### Correlation Matrix")
        st.write(df.corr().style.background_gradient(cmap="coolwarm"))

    elif choice == "Model Training":
        st.subheader("Linear Regression Model Training")
        x = df.drop(columns='price')
        y = df['price']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
        st.write(f"Training set shape: {x_train.shape}, Testing set shape: {x_test.shape}")

        lin_reg = LinearRegression()
        lin_reg.fit(x_train, y_train)

        st.write("Model Coefficients:")
        coef_dict = {'Feature': x.columns, 'Coefficient': lin_reg.coef_}
        st.write(pd.DataFrame(coef_dict))

        st.write(f"Model Intercept: {lin_reg.intercept_:.2f}")

        st.write("### Model Evaluation")
        accuracy = lin_reg.score(x_test, y_test)
        st.write(f"Model Accuracy: {accuracy:.2f}")

    elif choice == "Predict House Price":
        st.subheader("Predict Your House Price")
        bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
        sqft_living = st.number_input("Living Area (sqft)", min_value=500, max_value=10000, value=1800)
        grade = st.number_input("Grade", min_value=1, max_value=13, value=7)
        yr_built = st.number_input("Year Built", min_value=1900, max_value=2024, value=1990)

        user_input = np.array([[bedrooms, bathrooms, sqft_living, grade, yr_built]])
        predicted_price = lin_reg.predict(user_input)

        st.write(f"Predicted Price: ${predicted_price[0]:,.2f}")

if __name__ == "__main__":
    main()
