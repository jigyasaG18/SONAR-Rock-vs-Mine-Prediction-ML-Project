import streamlit as st
import numpy as np
import pickle

# Load the trained Support Vector Classifier model
with open('svc_model.pkl', 'rb') as file:  # Change the file name if needed
    model = pickle.load(file)

# Streamlit app layout
st.title("Mine vs Rock Prediction")
st.write("Enter the Mean Sonar Reading to predict whether it's a Mine or a Rock.")

# Single input field for the mean sonar reading
mean_reading = st.number_input("Mean Sonar Reading", min_value=0.0, max_value=1.0, format="%.4f")

# When the user clicks the predict button
if st.button("Predict"):
    # Create a numpy array with 60 identical features based on the mean reading
    input_data_as_numpy_array = np.array([mean_reading] * 60).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data_as_numpy_array)

    # Display the prediction result
    if prediction[0] == 'R':
        st.success('The object is a Rock')
    else:
        st.success('The object is a Mine')