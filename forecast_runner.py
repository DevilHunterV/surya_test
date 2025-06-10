import os
import time
import pandas as pd
import numpy as np
import requests
import warnings
from flask import Flask
from threading import Thread
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from supabase import create_client, Client

# === Setup ===
print("=== Script started ===")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
API_KEY = os.getenv("API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not API_KEY:
    raise ValueError("Missing environment variables: check SUPABASE_URL, SUPABASE_KEY, or API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
warnings.filterwarnings('ignore')

# === Flask Server for Render Port Binding ===
app = Flask(__name__)

@app.route("/")
def home():
    return "Forecast service is running!"

def start_flask():
    app.run(host="0.0.0.0", port=10000)

# === Forecasting Function ===
def process_forecasts():
    print("Fetching unprocessed user inputs...")
    response = supabase.table("UserInputs").select("*").eq("processed", False).execute()
    records = response.data

    if not records:
        print("No new inputs found.")
        return

    for row in records:
        try:
            currency = row["currency"]
            start = row["startdate"]
            end = row["enddate"]
            print(f"Processing: {currency} from {start} to {end}")

            url = (
                f"https://marketdata.tradermade.com/api/v1/pandasDF?"
                f"currency={currency}&api_key={API_KEY}&start_date={start}&end_date={end}&"
                f"format=records&fields=ohlc"
            )
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Failed to fetch data for {currency}: {response.status_code}")
                continue

            data = pd.DataFrame(response.json())
            if data.empty:
                print(f"No data returned for {currency}. Skipping...")
                continue

            data['Average'] = data.select_dtypes(include='number').mean(axis=1)
            data['diff'] = data['Average'].diff()
            data = data.drop(columns=['close', 'high', 'low', 'open'], errors='ignore')
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)

            curr_val = data['Average'].iloc[-1]
            split_index = int(0.2 * len(data) + 1)
            train = data['diff'].iloc[split_index:].dropna()
            test = data['diff'].iloc[:split_index].dropna()

            if train.empty or test.empty:
                print(f"Not enough data for {currency}. Skipping...")
                continue

            model = ARIMA(train, order=(1, 0, 1))
            model_fit = model.fit()
            predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1)
            future_values = curr_val + predictions.cumsum()

            # === Overwrite Forecast in Supabase ===
            supabase.table("forecast_results").delete().eq("currency", currency).execute()
            forecast_records = [
                {"currency": currency, "date": idx.strftime('%Y-%m-%d'), "value": float(val)}
                for idx, val in future_values.items()
            ]
            supabase.table("forecast_results").insert(forecast_records).execute()
            supabase.table("UserInputs").update({'processed': True}).eq("id", row["id"]).execute()

            print(f"✅ Forecast stored for {currency}")

        except Exception as e:
            print(f"Error while processing {row.get('currency', 'unknown')}: {e}")

# === Run Loop + Flask Server ===
if __name__ == "__main__":
    # Start web server to keep Render service alive
    Thread(target=start_flask).start()

    # Continuous polling loop
    while True:
        print("Running forecast job...")
        try:
            process_forecasts()
        except Exception as e:
            print("❌ Forecasting loop failed:", e)
        print("Sleeping for 10 minutes...\n")
        time.sleep(30)  # Wait 10 minutes
