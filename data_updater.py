import json
import os
import time
import schedule
import ccxt
import pandas as pd
import numpy as np
import pickle
import logging
import gc
import config
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Параметры конфигурации
MODEL_PATH = config.MODEL_PATH
SCALER_PATH = config.SCALER_PATH
DF_PATH = config.DF_PATH
LOG_PATH = config.LOG_PATH_UPD

# Настройка логирования
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

window_size = config.WINDOW
batch_size = config.BATCH_SIZE
epochs = config.EPOCHS

coins_10 = config.COINS_10
coins_20 = config.COINS_20
coins_hour = config.COINS_HOUR

exchanges = config.EXCHANGES


# Проверка и создание директорий
def ensure_directories():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    if not os.path.exists(SCALER_PATH):
        os.makedirs(SCALER_PATH)
    if not os.path.exists(DF_PATH):
        os.makedirs(DF_PATH)


# Сбор данных с нескольких бирж
def collect_data_from_multiple_exchanges(exchanges, coin, timeframe='1m'):
    combined_data = {}
    now = datetime.utcnow()
    since_time = now - timedelta(minutes=1000)
    since = int(since_time.timestamp() * 1000)

    for exchange in exchanges:
        df = get_data_from_exchange(exchange, coin, timeframe, since=since)
        if df is not None:
            combined_data[exchange] = df
    return combined_data


# Получение данных OHLCV с биржи
def get_data_from_exchange(exchange_name, coin, timeframe='1m', since=None):
    try:
        exchange_class = getattr(ccxt, exchange_name)()
        ohlcv = exchange_class.fetch_ohlcv(f'{coin}/USDT', timeframe, since=(since if timeframe != '1h' else None),
                                           limit=1000)
        df = pd.DataFrame(ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        logging.info(f'{exchange_name.capitalize()}: Получены данные для {coin}. Записей: {len(df)}')
        return df
    except Exception as e:
        logging.warning(f'{exchange_name.capitalize()}: Ошибка при получении данных для {coin}: {e}')
        return None


# Объединение данных с нескольких бирж
def combine_exchange_data(data_dict):
    combined_df = None

    for exchange, df in data_dict.items():
        df = df.rename(columns={
            'open': f'{exchange}_open',
            'high': f'{exchange}_high',
            'low': f'{exchange}_low',
            'close': f'{exchange}_close',
            'volume': f'{exchange}_volume'
        })
        df = df.set_index('date')
        if combined_df is None:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, how='outer', left_index=True, right_index=True)

    combined_df['value'] = combined_df[[col for col in combined_df.columns if col.endswith('_close')]].mean(axis=1)
    combined_df['rolling_mean'] = combined_df['value'].rolling(window=window_size).mean()
    combined_df = combined_df.dropna()
    return combined_df


# Объединение данных с Binance
def combine_binance_data(data_dict):
    binance_df = data_dict.get('binance', None)
    if binance_df is not None:
        binance_df = binance_df.set_index('date')
        binance_df['value'] = binance_df['close']
        binance_df['rolling_mean'] = binance_df['value'].rolling(window=window_size).mean()
        binance_df = binance_df.dropna()
    return binance_df


# Определение модели TCN
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        x = x[:, :, -1]
        x = self.linear(x)
        return x.squeeze()


# Функция обучения модели
def train_model(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['value'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i + window_size])
        y.append(scaled_data[i + window_size])

    X_array = np.array(X)
    y_array = np.array(y)

    X_tensor = torch.tensor(X_array, dtype=torch.float32)
    y_tensor = torch.tensor(y_array, dtype=torch.float32).squeeze()

    train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

    model = TCN(input_size=1, output_size=1, num_channels=[32, 64, 64])
    model.to(torch.device(config.DEVICE))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(torch.device(config.DEVICE)), y_batch.to(torch.device(config.DEVICE))
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    return model, scaler, X_array, y_array


# Сохранение модели, scaler и DataFrame
def save_model_and_scaler(model, scaler, df, coin, timeframe, suffix="all"):
    torch.save(model.state_dict(), f"{MODEL_PATH}tcn_model_{coin}_{timeframe}_{suffix}.pt")
    with open(f"{SCALER_PATH}scaler_{coin}_{timeframe}_{suffix}.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    df.to_pickle(f"{DF_PATH}df_{coin}_{timeframe}_{suffix}.pkl")
    target_column_data = {'target_column': 'value'}
    with open(f"{DF_PATH}target_column_{coin}_{timeframe}_{suffix}.json", 'w') as json_file:
        json.dump(target_column_data, json_file)
    logging.info(
        f"Модель, scaler, DataFrame и целевая колонка сохранены для {coin} с таймфреймом {timeframe} ({suffix})")


# Обновление данных и тренировка моделей
def update_data_and_train(coin, timeframe='1m'):
    data = collect_data_from_multiple_exchanges(exchanges, coin, timeframe)

    # Модель на данных всех бирж
    combined_df = combine_exchange_data(data)
    if combined_df is not None:
        model_all, scaler_all, X_all, y_all = train_model(combined_df)
        save_model_and_scaler(model_all, scaler_all, combined_df, coin, timeframe, "all")
        del combined_df, model_all, scaler_all, X_all, y_all
        gc.collect()

    # Модель только на данных Binance
    binance_df = combine_binance_data(data)
    if binance_df is not None:
        model_binance, scaler_binance, X_binance, y_binance = train_model(binance_df)
        save_model_and_scaler(model_binance, scaler_binance, binance_df, coin, timeframe, "binance")
        del binance_df, model_binance, scaler_binance, X_binance, y_binance
        gc.collect()


# Периодические задачи
def job_minute():
    for coin in set(coins_10 + coins_20):
        update_data_and_train(coin, '1m')


def job_hour():
    for coin in coins_hour:
        update_data_and_train(coin, '1h')


def schedule_data_updates():
    ensure_directories()
    job_minute()
    job_hour()
    schedule.every(5).minutes.do(job_minute)
    schedule.every().hour.do(job_hour)
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    schedule_data_updates()
