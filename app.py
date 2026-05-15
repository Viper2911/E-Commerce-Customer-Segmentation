import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Segmentation AI", layout="centered")

@st.cache_resource
def load_models():
    model = joblib.load('kmeans_segmentation_model.joblib')
    scaler = joblib.load('rfm_scaler.joblib')
    return model, scaler

model, scaler = load_models()

st.title("Customer Segmentation AI")
st.markdown("Enter a customer's purchase behaviour below to instantly categorize them into a marketing segment.")

with st.form("customer_data"):
    st.subheader("Customer RFM Metrics")

    recency = st.number_input("Recency: Days since last purchase", min_value=1, max_value=500, value=15)
    frequency = st.number_input("Frequency: Total number of orders", min_value=1, max_value=200, value=5)
    monetary = st.number_input("Monetary: Total amount spent (in $)", min_value=1.0, max_value=50000.0, value=250.0)

    submit_button = st.form_submit_button(label="Analyze Customer Segment")

if submit_button:
    log_r = np.log(recency)
    log_f = np.log(frequency)
    log_m = np.log(monetary)

    input_data = pd.DataFrame([[log_r, log_f, log_m]], columns=['Recency', 'Frequency', 'Monetary'])
    input_scaled = scaler.transform(input_data)
    
    cluster_id = model.predict(input_scaled)[0]
    st.divider()

    if cluster_id == 0:
        st.success("Segment: Loyal VIPs")
        st.write("These customers buy often, spend a lot, and purchased recently.")
        st.info("Marketing Action: Send exclusive early-access products and premium loyalty rewards. Do not discount heavily.")
        
    elif cluster_id == 1:
        st.warning("Segment: Churn-Risk / Slipping Away")
        st.write("These customers used to spend money but haven't visited the site in a long time.")
        st.info("Marketing Action: Send aggressive 'We miss you' discount codes to win them back.")
        
    elif cluster_id == 2:
        st.info("Segment: Recent Shoppers / Potential Loyalists")
        st.write("They bought recently but have low frequency and spend.")
        st.info("Marketing Action: Nurture them. Recommend related products based on their first purchase to encourage a second order.")
        
    else:
        st.error("Segment: Low Value / Lost Customers")
        st.write("They spent very little, bought once, and haven't returned in months.")
        st.info("Marketing Action: Do not spend high ad budget here. Keep them on standard automated email newsletters.")