import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import joblib


# Function to download data with caching to improve performance
def stiahnute_data(stock, start, end):
    try:
        data = yf.download(stock, start, end)
        if data.empty:
            st.error("Nesprávny ticker symbol: " + stock + ". Skúste to znova.")
            return None
        return data
    except Exception as e:
        st.error("Nesprávny ticker symbol alebo problém s pripojením. Skúste to znova.")
        return None

st.title("Predikcia cien akcií")

stock = st.text_input("Vložte Ticker symbol akcie ", "GOOG")
end = datetime.now()
start = datetime(end.year-10, end.month, end.day)

google_data = stiahnute_data(stock, start, end)
if google_data is None:
    st.stop()  

# Calculate moving averages
google_data['MA_pre_250_dni'] = google_data['Adj Close'].rolling(250).mean()
google_data['MA_pre_200_dni'] = google_data['Adj Close'].rolling(200).mean()
google_data['MA_pre_100_dni'] = google_data['Adj Close'].rolling(100).mean()

model = load_model('Najnovsi_model2.h5')

st.subheader("Historické dáta akcie")
st.write(google_data.drop(columns=['MA_pre_250_dni', 'MA_pre_200_dni', 'MA_pre_100_dni']))


rozdel_data = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data['Adj Close'][rozdel_data:])
skalovanie = MinMaxScaler(feature_range=(0, 1))
skalovane_data= skalovanie.fit_transform(x_test[['Adj Close']])
x_data, y_data = [], []
for i in range(100, len(skalovane_data)):
    x_data.append(skalovane_data[i-100:i])
    y_data.append(skalovane_data[i])
x_data, y_data = np.array(x_data), np.array(y_data)

predikcia = model.predict(x_data)
inv_predikcia = skalovanie.inverse_transform(predikcia)  # Ensure predictions are inverted before using

# Function to generate investment recommendations
def investicne_odporucanie(aktualna_cena, predikovana_cena):
    if aktualna_cena > predikovana_cena * 1.05:  
        return "Odporúča sa kúpiť"
    elif predikovana_cena < aktualna_cena * 0.95:  
        return "Odporúča sa predávať"
    else:
        return "Odporúča sa držať"

aktualna_cena = google_data['Adj Close'].iloc[-1]
predikovana_cena = inv_predikcia[-1][0]
odporucanie = investicne_odporucanie(aktualna_cena, predikovana_cena)
st.write("Investičné odporúčanie: ", odporucanie)

# Function to plot interactive graphs using Plotly
def plot_graf(title, series_dict):
    fig = go.Figure()
    
    for label, data in series_dict.items():
        if isinstance(data, pd.Series):

            color = 'red' if label == 'MA pre 100 dni' else None
            
            fig.add_trace(
                go.Scatter(
                    x=data.index, 
                    y=data.values, 
                    mode='lines', 
                    name=label, 
                    line=dict(color=color) ))
        
        elif isinstance(data, np.ndarray):
            # Vytvorenie časovej osi pre numpy array
            date_range = pd.date_range(
                start=google_data.index[rozdel_data + 100], 
                periods=len(data), 
                freq='D')
            
            # Pridanie dátového radu pre numpy array
            fig.add_trace(
                go.Scatter(
                    x=date_range, 
                    y=data.flatten(), 
                    mode='lines', 
                    name=label))
    
    # Nastavenie celkového vzhľadu grafu
    fig.update_layout(
        title=title, 
        xaxis_title='Dátum', 
        yaxis_title='Cena', 
        template="plotly_dark")
    
    # Zobrazenie grafu v Streamlit
    st.plotly_chart(fig, use_container_width=True)


# Plot interactive graphs for moving averages and predictions
plot_graf('Porovnanie ceny akcie a Moving Average', {
    'Uzatváracia cena': google_data['Close'],
    'MA pre 250 dni': google_data['MA_pre_250_dni'],
    'MA pre 200 dni': google_data['MA_pre_200_dni'],
    'MA pre 100 dni': google_data['MA_pre_100_dni']})


prediction_start_date = google_data.index[rozdel_data + 100]  
prediction_dates = pd.date_range(start=prediction_start_date, periods=len(inv_predikcia), freq='B') 

plot_graf('Porovnanie originálnej a predikovanej ceny', {
    'Originálna uzatváracia cena': google_data['Adj Close'][rozdel_data + 100:],  
    'Predikovaná uzatváracia cena': pd.Series(inv_predikcia.flatten(), index=prediction_dates)  
})