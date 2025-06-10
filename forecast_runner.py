import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import requests
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from supabase import create_client, Client
from threading import Thread
from flask import Flask
import os

# === Load Environment Variables from Render Dashboard ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
API_KEY = os.getenv("API_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
warnings.filterwarnings('ignore')

def process_forecasts():
    response = supabase.table("UserInputs").select("*").eq("processed", False).execute()
    records = response.data

    if not records:
        print("No new inputs found.")
        return

    for row in records:
        currency = row['currency']
        start = row['startdate']
        end = row['enddate']
        print(f"Processing: {currency} from {start} to {end}")

        # === Fetch Data ===
        url = f"https://marketdata.tradermade.com/api/v1/pandasDF?currency={currency}&api_key={API_KEY}&start_date={start}&end_date={end}&format=records&fields=ohlc"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch data for {currency}. Skipping...")
            continue

        data = pd.DataFrame(response.json())
        if data.empty:
            print(f"No data returned for {currency}. Skipping...")
            continue

        data['Average'] = data.select_dtypes(include='number').mean(axis=1)
        data['diff'] = data['Average'].diff()
        data = data.drop(['close', 'high', 'low', 'open'], axis=1)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)

        curr_val = data['Average'].iloc[-1]
        split_index = int(0.2 * len(data) + 1)
        train = data['diff'].iloc[split_index:].dropna()
        test = data['diff'].iloc[:split_index].dropna()

        if train.empty or test.empty:
            print(f"Insufficient data for {currency}. Skipping...")
            continue

        model = ARIMA(train, order=(1, 0, 1))
        model_fit = model.fit()
        predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1)

        future_values = curr_val + predictions.cumsum()

        # === Overwrite Forecast in Supabase ===
        supabase.table("Forecast_Results").delete().eq("currency", currency).execute()
        forecast_records = [
            {"currency": currency, "date": idx.strftime('%Y-%m-%d'), "value": float(val)}
            for idx, val in future_values.items()
        ]
        supabase.table("Forecast_Results").insert(forecast_records).execute()
        supabase.table("UserInputs").update({'processed': True}).eq("id", row["id"]).execute()
        print(f"Forecast stored and input marked as processed for {currency}.")

# === Background Loop ===
app = Flask(__name__)

@app.route("/")
def home():
    return "Forecast worker is running."

def start_flask():
    app.run(host="0.0.0.0", port=10000)

if __name__ == "__main__":
    # Start the Flask server in a separate thread
    Thread(target=start_flask).start()

    # Your looping job continues as usual
    while True:
        print("Running forecast job...")
        try:
            process_forecasts()
        except Exception as e:
            print("Error:", e)
        time.sleep(30)





