import os
import streamlit as st
import pandas as pd
from pycaret.classification import predict_model, load_model

model_dir = "best_model"
# load the model
model = load_model(model_dir)

# Create the streamlit app
st.title("Income Prediction App")

# Create a form for user input
st.sidebar.header("User Input Features")

# load the original features
df = pd.read_csv(r"income.csv")

def clean_marital_status(status):
    married_list = ["Married-civ-spouse", "Married-AF-spouse"]
    if status in married_list:
        return "Married"
    else:
        return "Not_married"
    
df["marital-status"] = df["marital-status"].apply(lambda x: clean_marital_status(x))
# rename the target
df.rename(columns={"income >50K": "income"}, inplace=True)

# define the input fields
features = df.drop("income", axis = 1)
num_features = features.select_dtypes(include = "number").columns.tolist()
cat_features = features.select_dtypes(include="object").columns.tolist()

input_fields = {}
for feature in num_features:
    input_fields[feature] = st.sidebar.slider(f"Select {feature}", df[feature].min(), 
                                              df[feature].max(), 0)
for feature in cat_features:
    input_fields[feature] = st.sidebar.selectbox(f"Select {feature}", df[feature].unique())


# Create a dataframe for the user input
user_input = pd.DataFrame([input_fields])
income_group = ["<50K", ">50K"]

# Make predictions
if st.sidebar.button("Predict"):
    prediction = predict_model(model, data = user_input, raw_score=True)
    st.write(prediction)
    predicted_label = prediction["prediction_label"].iloc[0]
    st.write(f"The Income Group is: {income_group[predicted_label]}")
    #st.write(f"Predicted Income: {prediction["prediction_label"]}")
                                             