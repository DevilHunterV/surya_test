import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import requests
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from supabase import create_client, Client

# === Configuration ===
API_KEY = "T0GogrTEb62VPtatnlp7ga1xXUpEvNjq"
SUPABASE_URL = "https://dletqrcbggnevurxbstz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRsZXRxcmNiZ2duZXZ1cnhic3R6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDg4NzkzOTMsImV4cCI6MjA2NDQ1NTM5M30.HfEpJ5b7lZejR4dhYt_DWap6ia-jBBJZZVjEkLPUr8E"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
warnings.filterwarnings('ignore')

# === Fetch new unprocessed inputs ===
response = supabase.table("UserInputs").select("*").eq("processed", False).execute()
records = response.data

if not records:
    print("No new inputs found.")
else:
    for row in records:
        currency = row['currency']
        start = row['startdate']
        end = row['enddate']
        base = currency[:3]
        symbol = currency[3:]

        print(f"Processing: {currency} from {start} to {end}")

        # === New API Call Format (e.g., exchangerate.host or similar) ===
        url = f"https://api.apilayer.com/exchangerates_data/timeseries?start_date={start}&end_date={end}&base={base}&symbols={symbol}"
        headers = {"apikey": API_KEY}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch data for {currency}. Skipping...")
            continue

        raw_data = response.json()
        if 'rates' not in raw_data:
            print(f"No rate data returned. Skipping...")
            continue

        # === Convert rates to DataFrame ===
        df = pd.DataFrame.from_dict(raw_data['rates'], orient='index')
        df.columns = ['Average']
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df['diff'] = df['Average'].diff()

        curr_val = df['Average'].iloc[-1]
        split_index = int(0.2 * len(df) + 1)
        train = df['diff'].iloc[split_index:].dropna()
        test = df['diff'].iloc[:split_index].dropna()

        if train.empty or test.empty:
            print("Not enough data. Skipping...")
            continue

        # === Fit ARIMA ===
        model = ARIMA(train, order=(2, 0, 4))
        model_fit = model.fit()
        predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1)

        # === Evaluate ===
        mae = mean_absolute_error(test, predictions)
        mse = mean_squared_error(test, predictions)
        rmse = np.sqrt(mse)
        print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

        # === Forecast ===
        future = curr_val + predictions.cumsum()

        # === Plot Forecast ===
        plt.plot(df['Average'], label='Historical')
        plt.plot(future, label='Forecast')
        plt.title(f"ARIMA Forecast: {currency}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # === Overwrite Forecast in Supabase ===
        supabase.table("Forecast_Results").delete().eq("currency", currency).execute()
        forecast_records = [
            {"currency": currency, "date": idx.strftime('%Y-%m-%d'), "value": float(val)}
            for idx, val in future.items()
        ]
        supabase.table("Forecast_Results").insert(forecast_records).execute()

        # === Mark Input as Processed ===
        supabase.table("UserInputs").update({'processed': True}).eq("id", row['id']).execute()
        print(f"Forecast stored and input marked as processed.")
