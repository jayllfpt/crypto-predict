import yfinance as yf
import datetime
import numpy as np

def get_data(startdate, enddate, save_path):
    data = yf.download('BTC-USD', start=startdate, end=enddate)
    data.to_csv(save_path)
    return data['Close'].values
