import streamlit as st
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

# Load the trained model
with open('logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app layout
st.title("Mine vs Rock Prediction")
st.write("Enter the Mean SONAR Reading to predict if it's a Mine or a Rock.")

# Input field for the mean sonar reading
mean_reading = st.number_input("Mean Sonar Reading", min_value=0.0, max_value=1.0, format="%.4f")

# When the user clicks the predict button
if st.button("Predict"):
    # Create a numpy array with 60 identical features based on the mean reading
    input_data_as_numpy_array = np.array([mean_reading] * 60).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_data_as_numpy_array)

    # Display the prediction
    if prediction[0] == 1:
        st.success('The object is a Rock')
    else:
        st.success('The object is a Mine')
