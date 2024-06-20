import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
import ta
import seaborn as sns
import matplotlib.pyplot as plt
import requests

# Define the neural network model architecture
def create_model(optimizer='adam'):
    model = Sequential()
    model.add(GRU(units=200, dropout=0, input_shape=(x_train.shape[1], x_train.shape[2]), activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Function to fetch data with error handling
def fetch_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not data:
            print(f"No data received from {url}")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return []

# Functions to fetch various data from CoinGecko
def fetch_coin_gecko_data(symbol='bitcoin', days='365', interval='daily'):
    url = f'https://api.coingecko.com/api/v3/coins/{symbol}/market_chart?vs_currency=usd&days={days}&interval={interval}'
    data = fetch_data(url)
    if data:
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.rename(columns={'timestamp': 'date'}, inplace=True)
        return df
    return pd.DataFrame()

# Fetch the data
df = fetch_coin_gecko_data('bitcoin', '365', 'daily')

# Check if DataFrame is empty
if df.empty:
    print("DataFrame is empty. Please check your data sources.")
    exit()

print(f"Fetched data: {df.shape[0]} rows")

# Technical indicators
df['close'] = pd.to_numeric(df['close'], errors='coerce')
df['RSI'] = ta.momentum.rsi(df['close'], fillna=True)
df['Bollinger_Band'] = ta.volatility.bollinger_hband(df["close"], fillna=True)
df['Standard_Deviation'] = df['close'].rolling(window=14).std()
df['Skewness'] = df['close'].rolling(window=14).apply(lambda x: x.skew(), raw=False)
df['Kurtosis'] = df['close'].rolling(window=14).apply(lambda x: x.kurtosis(), raw=False)
df['Z-Score'] = (df['close'] - df['close'].rolling(window=14).mean()) / df['Standard_Deviation']

# Drop rows with missing values after calculating indicators
df.dropna(inplace=True)

print(f"Data after calculating indicators and dropping NaNs: {df.shape[0]} rows")

# Use ALL the data
data = df.copy()

# Define the desired features for training
features = ['close', 'RSI', 'Bollinger_Band', 'Standard_Deviation', 'Skewness', 'Kurtosis', 'Z-Score']

# Check if all features are present in the DataFrame
missing_features = [feature for feature in features if feature not in df.columns]
if missing_features:
    print(f"Missing features: {missing_features}")
    exit()

print(f"Using features: {features}")

# Select only numeric columns for scaling
train_size = int(0.8 * len(df))
train_data = df.iloc[:train_size]

# Check if train_data is empty
if train_data.empty:
    print("Training data is empty. Please check your data.")
    exit()

print(f"Training data: {train_data.shape[0]} rows")

# Apply MinMaxScaler only on numeric features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data_train = scaler.fit_transform(train_data[features])
scaled_data = scaler.transform(df[features])

close_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close_train = close_scaler.fit_transform(train_data['close'].values.reshape(-1, 1))
scaled_close = close_scaler.transform(df['close'].values.reshape(-1, 1))

# Create a data structure with 60 timestamps and 1 output
x_train, y_train = [], []
for i in range(60, len(scaled_data)):
    x_train.append(scaled_data[i - 60:i])
    y_train.append(scaled_close[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Check if x_train and y_train are empty
if len(x_train) == 0 or len(y_train) == 0:
    print("x_train or y_train is empty. Please check your data preparation.")
    exit()

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Adjustments for predicting just tomorrow's closing price
n_future = 1  # Predicting only the next day

# Fitting the model as before
model = create_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(x_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Preparing the most recent data for prediction
x_pred_tomorrow = np.array([scaled_data[-60:]])
forecast_tomorrow = model.predict(x_pred_tomorrow)
y_pred_tomorrow = close_scaler.inverse_transform(forecast_tomorrow).flatten()[0]

# Print the predicted closing price for tomorrow
print(f"Predicted Bitcoin closing price for tomorrow: ${y_pred_tomorrow:.2f}")

# Plotting adjustments for the single prediction
forecast_date_tomorrow = pd.date_range(data['date'].iloc[-1], periods=2, freq='1d')[1]  # Getting tomorrow's date
df_forecast_tomorrow = pd.DataFrame({'date': [forecast_date_tomorrow], 'close': [y_pred_tomorrow]})

plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='close', data=df)
sns.scatterplot(x='date', y='close', data=df_forecast_tomorrow, color='red', label='Predicted for Tomorrow')
plt.title('Bitcoin Price Prediction for Tomorrow')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(['Actual', 'Predicted for Tomorrow'])
plt.show()
