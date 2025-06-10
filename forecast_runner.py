#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Python script: Forecast currency using ARIMA and store forecast in Supabase

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

# === Configuration ===
API_KEY = "bwtG8r82qn0tziJHhtRi"
SUPABASE_URL = "https://dletqrcbggnevurxbstz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRsZXRxcmNiZ2duZXZ1cnhic3R6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDg4NzkzOTMsImV4cCI6MjA2NDQ1NTM5M30.HfEpJ5b7lZejR4dhYt_DWap6ia-jBBJZZVjEkLPUr8E"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
warnings.filterwarnings('ignore')

response = supabase.table("UserInputs").select("*").eq("processed", False).execute()
records = response.data

if not records:
    print("No new inputs found.")
else:
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
            print(f"No data returned for {currency} from {start} to {end}. Skipping...")
            continue

        data['Average'] = data.select_dtypes(include='number').mean(axis=1)
        data['diff'] = data['Average'].diff()
        data = data.drop(['close', 'high', 'low', 'open'], axis=1)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        data.index = pd.to_datetime(data.index)

        curr_val = data['Average'].iloc[-1]
        split_index = int(0.2 * len(data) + 1)
        train = data['diff'].iloc[split_index:].dropna()
        test = data['diff'].iloc[:split_index].dropna()

        if train.empty or test.empty:
            print(f"Insufficient training or testing data for {currency}. Skipping...")
            continue

        # === Fit ARIMA ===
        model = ARIMA(train, order=(1, 0, 1))
        model_fit = model.fit()
        predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1)

        # === Evaluate ===
        mae = mean_absolute_error(test, predictions)
        mse = mean_squared_error(test, predictions)
        rmse = np.sqrt(mse)
        perf = pd.DataFrame([{'currency': currency, 'mae': mae, 'mse': mse, 'rmse': rmse}])

        print("Performance:")
        print(perf)

        # === Plot ===
        plt.figure(figsize=(10, 5))
        plt.plot(train, label='Train')
        plt.plot(test, label='Test')
        plt.plot(predictions, label='Predictions')
        plt.title(f"ARIMA Forecast - {currency}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # === Forecast Future Values ===
        future_values = curr_val + predictions.cumsum()
        plt.figure(figsize=(10, 5))
        plt.plot(data['Average'], label='Historical')
        plt.plot(future_values, label='Forecast')
        plt.title(f"Forecasted Values - {currency}")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # === Overwrite Forecast in Supabase ===
        # First delete existing records for this currency
        supabase.table("Forecast_Results").delete().eq("currency", currency).execute()

        # Then insert new forecast
        forecast_records = [
            {
                "currency": currency,
                "date": idx.strftime('%Y-%m-%d'),
                "value": float(val)
            } for idx, val in future_values.items()
        ]
        supabase.table("Forecast_Results").insert(forecast_records).execute()
        print(f"Forecast for {currency} overwritten in Supabase.")
        
        # === Mark Input Row as Processed ===
        supabase.table("UserInputs").update({'processed': True}).eq("id", row["id"]).execute()
        print(f"Input marked as processed for {currency}.")

