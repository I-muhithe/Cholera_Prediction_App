# Importing the necessary libraries
import streamlit as st
import joblib
import numpy as np

# Loading your saved model. Be sure to change the path if needed
model = joblib.load("cholera_model.joblib") # Changed path

st.title('Cholera Prediction App') # Changed title
st.write("This app predicts the probability of someone having cholera based on their age and test result.") # Changed description

# Input fields
#test_result = st.selectbox('Test_result', ['0', '1'])
test_result = st.selectbox('Test_result', options=[0, 1], index=0)
#sex = st.selectbox('Sex', options=[0, 1], index=0)
#sex= st.selectbox('Sex', ['1', '2'])
sex= st.selectbox('Sex', options=[1, 2], index=1)
age= st.number_input('Age', min_value=0, max_value=100, value=20) # Changed min/max to integer
#subcounty= st.selectbox('Subcounty', ['0','1','2','3','4','5','6'])
subcounty= st.selectbox('Subcounty', options=[1,2,3,4,5,6], index=1)
#water= st.selectbox('Water', ['1','2','3','4'])
water= st.selectbox('Water', options=[1,2,3,4], index=1)
#sanitation= st.selectbox('Sanitation', ['1','2','3','4'])
sanitation= st.selectbox('Sanitation', options=[1,2,3,4], index=1)
#income= st.selectbox('Income', ['1','2','3','4','5'])
income= st.selectbox('Income', options=[1,2,3,4,5], index=1)
#informal_settlement= st.selectbox('Informal_settlement', ['1','2'])
informal_settlement= st.selectbox('Informal_settlement', options=[1, 2], index=1)

if st.button('Predict'):
    # Prepare input features for prediction
    input_features = np.array([[test_result, sex,age, subcounty, water, sanitation,income,informal_settlement]])  # Ensure this is a 2D array

    # Make prediction
    prediction = model.predict(input_features)
    st.write(f'Predicted Outcome: {prediction[0]:.2f}')
    #st.write(f'Predicted Outcome: {prediction[0]}')
    # Display prediction
    # Assuming the model outputs a probability or a value that needs interpretation for classification
    # For a binary classification model, model.predict_proba might be more appropriate if available.
    # For simplicity, if the model is a regression model trained on binary outcomes,
    # we can interpret the output as a score and potentially threshold it.
    # However, since the goal is predicting probability, a classification model is more suitable.
    # Given the linear regression model, we'll display the predicted value and note it's a score.

    ##st.write(f'Predicted score (interpretation needed for probability): {prediction[0]:.2f}')






