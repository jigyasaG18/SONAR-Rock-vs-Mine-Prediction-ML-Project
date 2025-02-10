import streamlit as st
import numpy as np
import pickle

# Load the trained Support Vector Classifier model
with open('svc_model.pkl', 'rb') as file:  # Change the file name if needed
    model = pickle.load(file)

# Streamlit app layout
st.title("Mine vs Rock Prediction")
st.write("Enter the SONAR Readings to predict whether it's a Mine or a Rock.")

input_data = st.text_input('Enter comma-separated SONAR Reading Values here')
# When the user clicks the predict button
if st.button("Predict"):
    # Prepare input data
    input_data_np_array = np.asarray(input_data.split(','), dtype=float)
    reshaped_input = input_data_np_array.reshape(1, -1)
    # Predict and show result
    prediction = model.predict(reshaped_input)
    if prediction[0] == 'R':
        st.write('This Object is Rock')
    else:
        st.write('The Object is Mine')
