from trends import TrendPredict
from prices import PricePredict

import json
import numpy as np

if __name__ == "__main__":
    timesteps=10
    with open('data/test_data.json') as f:
        data = json.load(f)
    data = np.array(data['close'])

    trend_predictor = TrendPredict(timesteps=timesteps,
                             model_path="models/trend_model.h5",
                             scaler_path="models/trend_scaler.dump")

    price_predictor = PricePredict(timesteps=timesteps,
                             model_path="models/price_model.h5",
                             scaler_path="models/price_scaler.dump")
    
    labels = trend_predictor(data)

    price_predictor(data[timesteps:], labels)