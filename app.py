import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
data = pd.read_csv('kc_house_data.csv')

# Selecting features and target variable
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
target = 'price'
X = data[features]
y = data[target]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app title
st.title("House Price Prediction App")

# Sidebar for user input
st.sidebar.header("Enter House Features")

# Input fields for house features
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)
sqft_living = st.sidebar.number_input("Square Footage of Living Area", 500, 10000, 2000)
sqft_lot = st.sidebar.number_input("Square Footage of Lot", 500, 50000, 5000)
floors = st.sidebar.slider("Number of Floors", 1, 3, 1)
zipcode = st.sidebar.number_input("Zipcode", 98001, 98199, 98004)

# Button to predict
if st.sidebar.button("Predict Price"):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'sqft_living': [sqft_living],
        'sqft_lot': [sqft_lot],
        'floors': [floors],
        'zipcode': [zipcode]
    })

    # Predict the house price using the model
    predicted_price = model.predict(input_data)

    # Display the predicted price
    st.write(f"### Predicted House Price: ${predicted_price[0]:,.2f}")

    # Evaluate the model performance
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    st.write(f"### Model Performance:")
    st.write(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    st.write(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")

