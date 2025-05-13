import requests
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import os
from flask import Flask, Response
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

# === Sabit rastgelelik ===
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds()

# === Web Servisten Veri Çekme ===
def fetch_webservice_data():
    url = "http://localhost:5000/api/WebService/CallService"
    payload = {
        "Client": "00",
        "Language": "T",
        "DBServer": "CANIAS",
        "DBName": "IAS803RDBDEV",
        "ApplicationServer": "localhost:27499",
        "Username": "BETULTEST",
        "Password": "B12345.",
        "Encrypted": False,
        "Compression": False,
        "LCheck": "",
        "VKey": "",
        "ServiceId": "WEBTESTPANDAS2",
        "Parameters": "<PARAMETERS><PARAM>param1</PARAM></PARAMETERS>",
        "Compressed": False,
        "Permanent": False,
        "ExtraVariables": "",
        "RequestId": 0
    }
    headers = {
        "Content-Type": "application/json",
        "accept": "text/plain"
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        json_response = response.json()
        xml_data = json_response["Data"]["Response"]["Value"]
        root = ET.fromstring(xml_data)
        records = []
        for element in root.findall('.//element'):
            record = {child.tag: child.text for child in element}
            records.append(record)
        return pd.DataFrame(records)
    else:
        return None

# === Dataset Oluşturucu ===
def create_dataset(dataset, look_back=12):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back, 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# === Tahmin Fonksiyonu ===
def forecast_demand(sales_data, look_back=12):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(sales_data)

    X_lstm, Y_lstm = create_dataset(scaled_data, look_back)
    X_xgb, Y_xgb = create_dataset(sales_data, look_back)

    X_lstm = X_lstm.reshape((X_lstm.shape[0], look_back, 1))

    # LSTM Eğitimi
    model_lstm = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(25),
        Dense(1)
    ])
    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(X_lstm, Y_lstm, epochs=20, batch_size=8, verbose=0)

    # XGBoost Eğitimi
    model_xgb = xgb.XGBRegressor(objective='reg:squarederror')
    model_xgb.fit(X_xgb, Y_xgb)

    r2_lstm = r2_score(Y_lstm, model_lstm.predict(X_lstm))
    r2_xgb = r2_score(Y_xgb, model_xgb.predict(X_xgb))

    predictions = []
    if r2_lstm >= r2_xgb:
        input_data = scaled_data[-look_back:]
        for _ in range(6):
            inp = input_data.reshape((1, look_back, 1))
            pred = model_lstm.predict(inp)[0, 0]
            predictions.append(pred)
            input_data = np.append(input_data[1:], [[pred]], axis=0)
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    else:
        input_data = sales_data[-look_back:].flatten()
        for _ in range(6):
            inp = input_data.reshape(1, -1)
            pred = model_xgb.predict(inp)[0]
            predictions.append(pred)
            input_data = np.append(input_data[1:], [pred])

    last_date = sales_data.index.max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='M')
    return pd.DataFrame({
        "Tarih": future_dates.strftime('%Y-%m'),
        "Tahmin": np.round(predictions).astype(int).flatten()
    })

# === DataFrame → XML ===
def dataframe_to_xml(df, root_tag='Results', row_tag='Record'):
    root = ET.Element(root_tag)
    for _, row in df.iterrows():
        record = ET.SubElement(root, row_tag)
        for col in df.columns:
            child = ET.SubElement(record, col)
            child.text = str(row[col])
    return ET.tostring(root, encoding='utf-8').decode('utf-8')

# === Flask App ===
app = Flask(__name__)

@app.route("/xml_taleptahmini", methods=["GET"])
def get_tahmin():
    try:
        df = fetch_webservice_data()
        if df is None or 'QUANTITY' not in df.columns or 'VALIDFROM' not in df.columns:
            return Response("Veri alınamadı veya uygun değil.", mimetype="text/plain")

        df['VALIDFROM'] = pd.to_datetime(df['VALIDFROM'], format='%d.%m.%Y', errors='coerce')
        df = df[['VALIDFROM', 'QUANTITY']].dropna()
        df.rename(columns={'QUANTITY': 'Satış Miktarı'}, inplace=True)
        df.sort_values('VALIDFROM', inplace=True)
        df.set_index('VALIDFROM', inplace=True)
        df['Satış Miktarı'] = pd.to_numeric(df['Satış Miktarı'], errors='coerce').fillna(method='ffill')

        prediction_df = forecast_demand(df, look_back=12)
        xml_output = dataframe_to_xml(prediction_df)
        return Response(xml_output, mimetype="application/xml")
    except Exception as e:
        return Response(f"Hata oluştu: {str(e)}", mimetype="text/plain")

# Vercel Flask handler'ı
handler = app

