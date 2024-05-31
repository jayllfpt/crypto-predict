import streamlit as st
import matplotlib.pyplot as plt
from trends import TrendPredict
from prices import PricePredict
from data.APIs import get_data

import json
import numpy as np
from datetime import datetime

timesteps = 10
trend_predictor = TrendPredict(timesteps=timesteps,
                                   model_path="models/trend_model.h5",
                                   scaler_path="models/trend_scaler.dump")
price_predictor = PricePredict(timesteps=timesteps,
                                model_path="models/price_model.h5",
                                scaler_path="models/price_scaler.dump")


def inference_page():
    st.title("inference")
    st.write("Data range config")
    start_date = st.date_input("Start Date", value=datetime(2024, 1, 1), key = "startdate")
    end_date = st.date_input("End Date", value=datetime.today(), key = "enddate")

    inference_button = st.button("Run models", key="inference_btn")
    if inference_button:
        
        data = get_data(start_date, end_date, "data/test_data.csv")
        labels = trend_predictor(data)
        predictions = price_predictor(data[timesteps:], labels)
        st.write("Result")
        placeholder = st.empty()
        with placeholder.container():
            kpi1, kpi2, kpi3 = st.columns(3)
            if labels[-1] == 1: _trend = "UP" 
            else: _trend = "DOWN"
            kpi1.metric(label="trend", value=_trend)
            kpi2.metric(label="yesterday close price", value=data[-1])
            kpi3.metric(label="yesterday predicted price", value=predictions[-1])

            x = np.array([x for x in range(len(predictions))])
            fig, ax = plt.subplots()
            ax.plot(x, data[-len(predictions):])
            ax.plot(x, predictions)
            st.pyplot(fig)


def training_page():
    st.title("training")
    st.write("Data range config")
    start_date = st.date_input("Start Date", value=datetime(2020, 1, 1), key = "startdate")
    end_date = st.date_input("End Date", value=datetime.today(), key = "enddate")
    placeholder = st.empty()
    with placeholder.container():
        trend_model, price_model = st.columns(2)
        with trend_model:
            st.subheader("Trend Prediction Model Training Configuration")
            epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=10,key="trend_train_epochs")
            batch_size = st.number_input("Batch Size", min_value=1, max_value=1024, value=32,key="trend_training_batchsize")
            save_path = st.text_input("Save Path", value="models/trend_model.h5", key = "trend_model_path")
            train_button = st.button("Start Training", key="trend_train_btn")

            if train_button:
                st.write(f"get data from {start_date} to {end_date}")
                data = get_data(start_date, end_date, "data/trend_train_data.csv")
                trend_predictor.train(data, 0.8, epochs, batch_size)
                st.write(f"trend prediction model are saved at {save_path}")

        with price_model:
            st.subheader("Price Prediction Model Training Configuration")
            epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=10,key="price_training_epochs")
            batch_size = st.number_input("Batch Size", min_value=1, max_value=1024, value=32,key="price_training_batchsize")
            save_path = st.text_input("Save Path", value="models/price_model.h5", key = "price_model_path")
            train_button = st.button("Start Training", key="price_train_btn")

            if train_button:
                st.write(f"get data from {start_date} to {end_date}")
                data = get_data(start_date, end_date, "data/price_train_data.csv")
                trend_predictor.train(data, 0.8, epochs, batch_size)
                st.write(f"trend prediction model are saved at {save_path}")


pages = {
    "inference": inference_page,
    "training": training_page,
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))

page = pages[selection]
page()
