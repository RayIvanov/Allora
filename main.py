import json
import traceback
import threading
from datetime import datetime, timedelta
import numpy as np
import requests
import torch
import pickle
import logging
import random
import time
import config
from flask import Flask, Response, request
from data_updater import TCN

lock = threading.Lock()

# Путь к файлам модели, scaler, dataframe
MODEL_PATH = config.MODEL_PATH
SCALER_PATH = config.SCALER_PATH
DF_PATH = config.DF_PATH
LOG_PATH = config.LOG_PATH_MAIN

# Временные окна для предсказаний
window = config.WINDOW


app = Flask(__name__)

# Настройка логирования
logging.basicConfig(filename=LOG_PATH,
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

topics = config.TOPICS

# Кэш для хранения предсказаний
prediction_cache_binance = {}
prediction_cache_all = {}
# Кэш для хранения рекомендаций покупки
recommendation_cache = {}



def fetch_tradingview_data(symbol):
    url = f"https://scanner.tradingview.com/symbol?symbol=CRYPTO%3A{symbol}USD&fields=Recommend.All%7C1"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        value = data.get("Recommend.All|1")  # Извлекаем рекомендацию
        return value
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching recommendation data for {symbol}: {e}")
        return None

# Фоновая функция для обновления рекомендаций каждую минуту
def update_recommendations():
    symbols = ["BTC", "SOL", "BNB", "ARBI", "ETH"]

    # Выполняем начальное обновление рекомендаций сразу после запуска
    for symbol in symbols:
        try:
            recommendation = fetch_tradingview_data(symbol)
            if recommendation is not None:
                with lock:
                    recommendation_cache[symbol] = {
                        'recommendation': recommendation,
                        'time': datetime.now()
                    }
                logging.info(f"Initial recommendation fetched for {symbol}: {recommendation}")
            else:
                logging.warning(f"Failed to fetch initial recommendation for {symbol}")
        except Exception as e:
            logging.error(f"Error fetching initial recommendation for {symbol}: {e}")

    while True:
        now = datetime.now()
        next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
        sleep_time = (next_minute - now).total_seconds()

        logging.info(f"Sleeping for {sleep_time} seconds before updating recommendations")
        time.sleep(sleep_time)

        for symbol in symbols:
            try:
                recommendation = fetch_tradingview_data(symbol)
                if recommendation is not None:
                    with lock:
                        recommendation_cache[symbol] = {
                            'recommendation': recommendation,
                            'time': datetime.now()
                        }
                    logging.info(f"Updated recommendation for {symbol}: {recommendation}")
                else:
                    logging.warning(f"Failed to fetch recommendation for {symbol}")
            except Exception as e:
                logging.error(f"Error updating recommendation for {symbol}: {e}")


# Обновление предсказания в фоновом потоке BINANCE
def update_prediction_in_background_binance(topic, timing, coin):
    try:
        logging.info(f"Starting background update for topic from binance: {topic}")
        model, scaler, df, target_column = load_saved_model_and_scaler_binance(timing, coin)
        predicted_prices = forecast_future(model, df, scaler, timing)

        # Убедитесь, что предсказанные цены были возвращены
        if predicted_prices is not None:
            result = predicted_prices[-1]  # Берем последнее предсказание
            with lock:
                cache_prediction_binance(topic, result)
            logging.info(f"Background update completed for topic from binance {topic}: {result}")
        else:
            logging.warning(f"No predictions were made for topic from binance: {topic}")

    except Exception as e:
        logging.error(f"Error in background prediction update for {topic} from binance : {str(e)}")
        logging.error("Stack trace: %s", traceback.format_exc())



# Обновление предсказания в фоновом потоке
def update_prediction_in_background_all(topic, timing, coin):
    try:
        logging.info(f"Starting background update for topic from all: {topic}")
        model, scaler, df, target_column = load_saved_model_and_scaler_all(timing, coin)
        predicted_prices = forecast_future(model, df, scaler, timing)

        # Убедитесь, что предсказанные цены были возвращены
        if predicted_prices is not None:
            result = predicted_prices[-1]  # Берем последнее предсказание
            with lock:
                cache_prediction_all(topic, result)
            logging.info(f"Background update completed for topic from all {topic}: {result}")
        else:
            logging.warning(f"No predictions were made for topic from all: {topic}")

    except Exception as e:
        logging.error(f"Error in background prediction update for {topic} from all: {str(e)}")
        logging.error("Stack trace: %s", traceback.format_exc())


# Фоновая функция для обновления предсказаний каждую минуту
def update_predictions_binance():
    while True:
        now = datetime.now()
        next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
        sleep_time = (next_minute - now).total_seconds()

        logging.info(f"Sleeping for {sleep_time} seconds until the next minute starts")
        time.sleep(sleep_time)

        for topic in topics:
            coin, timing = topic.split('_')
            # Запуск обновления предсказания в фоновом потоке для каждого топика
            thread = threading.Thread(target=update_prediction_in_background_binance, args=(topic, timing, coin))
            thread.daemon = True
            thread.start()


def update_predictions_all():
    while True:
        now = datetime.now()
        next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
        sleep_time = (next_minute - now).total_seconds()

        logging.info(f"Sleeping for {sleep_time} seconds until the next minute starts")
        time.sleep(sleep_time)

        for topic in topics:
            coin, timing = topic.split('_')
            # Запуск обновления предсказания в фоновом потоке для каждого топика
            thread = threading.Thread(target=update_prediction_in_background_all, args=(topic, timing, coin))
            thread.daemon = True
            thread.start()


# Получение предсказания из кэша
def get_cached_prediction_binance(topic):
    cached = prediction_cache_binance.get(topic)
    if cached:
        return cached['prediction']
    return None

# Получение предсказания из кэша
def get_cached_prediction_all(topic):
    cached = prediction_cache_all.get(topic)
    if cached:
        return cached['prediction']
    return None



# Сохранение предсказания в кэш
def cache_prediction_binance(topic, prediction):
    prediction_cache_binance[topic] = {'prediction': prediction, 'time': datetime.now()}

def cache_prediction_all(topic, prediction):
    prediction_cache_all[topic] = {'prediction': prediction, 'time': datetime.now()}


# Загрузка модели и scaler
def load_saved_model_and_scaler_binance(timing, coin):
    model = TCN(input_size=1, output_size=1, num_channels=[32, 64, 64])
    model.load_state_dict(torch.load(f"{MODEL_PATH}tcn_model_{coin}_1{timing[-1]}_binance.pt", weights_only=True))
    model.eval()
    with open(f"{SCALER_PATH}scaler_{coin}_1{timing[-1]}_binance.pkl", 'rb') as f:
        scaler = pickle.load(f)
    with open(f"{DF_PATH}df_{coin}_1{timing[-1]}_binance.pkl", 'rb') as f:
        df = pickle.load(f)
    with open(f"{DF_PATH}target_column_{coin}_1{timing[-1]}_binance.json", 'r') as f:
        data = json.load(f)
    return model, scaler, df, data['target_column']




# Загрузка модели и scaler
def load_saved_model_and_scaler_all(timing, coin):
    model = TCN(input_size=1, output_size=1, num_channels=[32, 64, 64])
    model.load_state_dict(torch.load(f"{MODEL_PATH}tcn_model_{coin}_1{timing[-1]}_all.pt", weights_only=True))
    model.eval()
    with open(f"{SCALER_PATH}scaler_{coin}_1{timing[-1]}_all.pkl", 'rb') as f:
        scaler = pickle.load(f)
    with open(f"{DF_PATH}df_{coin}_1{timing[-1]}_all.pkl", 'rb') as f:
        df = pickle.load(f)
    with open(f"{DF_PATH}target_column_{coin}_1{timing[-1]}_all.json", 'r') as f:
        data = json.load(f)
    return model, scaler, df, data['target_column']



def forecast_future(model, df, scaler, timing):

    if timing == '24h':
        look_back = window
        num_predictions = 24
    elif timing == '10m' or timing == '20m':
        look_back = window
        current_time = datetime.now() - timedelta(hours=3)
        last_data_time = df.index[-1]
        time_diff_minutes = (current_time - last_data_time).total_seconds() / 60.0
        num_predictions = int(round(time_diff_minutes)) + int(timing[:2])
    else:
        raise ValueError("Unsupported timing value. Use '10m', '20m', or '24h'.")

    input_data = df.iloc[-look_back:][['value']].values  # Используйте только 'value'
    input_data_scaled = scaler.transform(input_data)

    predicted_prices = []
    for _ in range(num_predictions):
        # Убедитесь, что данные имеют правильную форму
        input_data_scaled_array = np.array([input_data_scaled]).astype(np.float32)

        # Предсказание
        with torch.no_grad():  # Отключаем градиенты для инференса
            predicted_price_scaled = model(torch.FloatTensor(input_data_scaled_array))

        # Преобразуем предсказанные значения в удобный для нас формат
        predicted_price_scaled = predicted_price_scaled.view(-1)  # Преобразуем в одномерный тензор

        # Проверка формы предсказанных значений
        if predicted_price_scaled.dim() != 1:
            logging.error(f"Unexpected output shape: {predicted_price_scaled.shape}")
            raise ValueError(f"Expected 1D output but got shape: {predicted_price_scaled.shape}")

        predicted_price_full_scaled = np.zeros((1, 1))  # Задаем форму 1x1
        predicted_price_full_scaled[0, 0] = predicted_price_scaled.numpy()[0]

        predicted_price = scaler.inverse_transform(predicted_price_full_scaled)[:, 0][0]

        predicted_prices.append(predicted_price)

        new_data_point_scaled = scaler.transform(np.array([[predicted_price]]))
        input_data_scaled = np.vstack((input_data_scaled[1:], new_data_point_scaled))

    return predicted_prices  # Возвращаем список предсказанных цен


# Обработчик запросов
@app.route("/inference/<string:timeframe>/<string:token>")
def handler(timeframe, token):
    coin = token
    topic = f"{coin}_{timeframe}"
    start_time = datetime.now()
    logging.info(f"Received request for topic: {topic} at {start_time}")

    # Получаем параметр r из строки запроса
    r = request.args.get('r', default='0')  # По умолчанию '0', если параметр не передан
    model_type = request.args.get('model_type', default='binance')  # По умолчанию 'binance'


    # Проверяем, существует ли топик
    if topic not in topics:
        error_message = "Wrong topic"
        logging.error(f"Error: {error_message}")
        return Response(json.dumps({"error": error_message}), status=500, mimetype='application/json')

    # Нормализуем название монеты для соответствия с кэшем
    cache_key = "ARBI" if coin == "ARB" else coin  # Преобразуем "ARB" в "ARBI", остальные оставляем без изменений

    with lock:
        if model_type == "binance":
            cached_prediction = get_cached_prediction_binance(topic)
        else:
            cached_prediction = get_cached_prediction_all(topic)
        if cached_prediction is not None:
            logging.info(f"Returning cached prediction for topic: {topic}")

            # Если r=1, возвращаем предсказание без рандомизации
            if r == '1':
                return str(cached_prediction)

            # Проверяем рекомендацию для монеты
            recommendation_data = recommendation_cache.get(cache_key)
            if recommendation_data:
                recommendation = recommendation_data['recommendation']
                logging.info(f"Recommendation for {cache_key}: {recommendation}")

                # Определяем знак изменения в зависимости от рекомендации
                if recommendation > 0:  # Положительная рекомендация
                    random_change_percentage = random.uniform(0.0001, 0.0005)  # Положительный процент
                else:  # Отрицательная или нейтральная рекомендация
                    random_change_percentage = random.uniform(-0.0005, -0.0001)  # Отрицательный процент
            else:
                # Если нет данных о рекомендации, используем случайный знак
                random_change_percentage = random.uniform(-0.0005, 0.0005)
                logging.warning(f"No recommendation data for {cache_key}, using random percentage")

            # Применяем рандомизацию
            return str(cached_prediction * (1 + random_change_percentage))

    logging.error(f"No cached prediction found for topic: {topic}")
    return Response(json.dumps({"error": "No prediction available"}), status=500, mimetype='application/json')


# Запуск фонового процесса
def start_background_updater():
    # Выполняем начальные расчеты
    for topic in topics:
        coin, timing = topic.split('_')

        thread_all = threading.Thread(target=update_prediction_in_background_all, args=(topic, timing, coin))
        thread_all.daemon = True
        thread_all.start()

        thread_binance = threading.Thread(target=update_prediction_in_background_binance, args=(topic, timing, coin))
        thread_binance.daemon = True
        thread_binance.start()



    # Запускаем фоновый процесс для периодического обновления предсказаний
    update_thread_binance = threading.Thread(target=update_predictions_binance)
    update_thread_binance.daemon = True
    update_thread_binance.start()

    update_thread_all = threading.Thread(target=update_predictions_all)
    update_thread_all.daemon = True
    update_thread_all.start()

    # Запускаем фоновый процесс для обновления рекомендаций каждую минуту
    recommendation_thread = threading.Thread(target=update_recommendations)
    recommendation_thread.daemon = True
    recommendation_thread.start()




if __name__ == "__main__":
    start_background_updater()
    app.run(host=config.HOST, port=config.PORT, debug=False)
