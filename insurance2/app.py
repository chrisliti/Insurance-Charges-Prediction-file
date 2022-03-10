import streamlit as st
import pandas as pd
import numpy as np
import pickle

## Load model
#filename = 'random_forest_regressor_model.sav'
model = pickle.load(open('rf-model.pkl', 'rb'))

## Load Dataset

pred_df = st.file_uploader("upload file", type={"csv", "txt"})
if pred_df is not None:
    pred_df = pd.read_csv(pred_df)

## Main function

def make_predictions():
  
  categorical_features = ["sex","smoker","region"]

  pred_df['sex_male'] = np.where(pred_df['sex'] == "male", 1,0)
  pred_df['smoker_yes'] = np.where(pred_df['smoker'] == "yes", 1,0)

  pred_df['region_northwest'] = np.where(pred_df['region'] == "region_northwest", 1,0)
  pred_df['region_southeast'] = np.where(pred_df['region'] == "region_southeast", 1,0)
  pred_df['region_southwest'] = np.where(pred_df['region'] == "region_southwest", 1,0)

  pred_df.drop(categorical_features,axis=1,inplace=True)
   
  predictions = model.predict(pred_df)

  pred_df.loc[:,'predicted_expense'] = predictions

  return pred_df


if st.button('predict'):
    predicted_file = make_predictions()
    st.dataframe(predicted_file)







