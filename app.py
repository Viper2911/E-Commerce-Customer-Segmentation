import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Segmentation AI", layout="centered")

@st.cache_resource
def load_models():
    model=joblib.load('kmeans_segmentation_model.joblib')
    scaler=joblib.load('rfm_scaler.joblib')
    return model,scaler

model,scaler=load_models()

st.title("Customer Segmentation AI")
st.markdown("Enter a customer's purchase behaviour below to instantly categorize them into a marketing segment.")

with st.form("customer_data"):
    st.subheader("Customer RFM Metrics")

    recency=st.number_input("Recency: Days since last purchase",min_value=1,max_value=500,value=15)
    frequency=st.number_input("Frequency: Total number of orders",min_value=1,max_value=200,value=5)
    monetary=st.number_input("Monetary: Total amount spent (in $)",min_value=1.0,max_value=50000.0,value=250.0)

    submit_button=st.form_submit_button(label="Analyze Customer Segment")
