
#Пути
# MODEL_PATH = '/root/allora-model/models/'
# SCALER_PATH = '/root/allora-model/scalers/'
# DF_PATH = '/root/allora-model/df/'
# LOG_PATH_MAIN = '/root/allora-model/inference_logs.log'
# LOG_PATH_UPD = '/root/allora-model/model_training.log'

MODEL_PATH = 'models/'
SCALER_PATH = 'scalers/'
DF_PATH = 'df/'
LOG_PATH_MAIN = 'inference_logs.log'
LOG_PATH_UPD = 'model_training.log'

# Временные окна для предсказаний
WINDOW = 60



###DATA_UPDATER

BATCH_SIZE = 64
EPOCHS = 25

EXCHANGES = [
    'binance',
    'kucoin',
    'huobi',
    'bitget',
    'bitso',
]

COINS_10 = ['ETH', 'BTC', 'SOL']
COINS_20 = ['ETH', 'BNB', 'ARB']
COINS_HOUR = ['ETH', 'BTC', 'SOL']

DEVICE = 'cpu' #Видюха или проц


###MAIN

HOST = '0.0.0.0'
PORT = 8000

TOPICS = [
    'ETH_10m',
    'ETH_24h',
    'BTC_10m',
    'BTC_24h',
    'SOL_10m',
    'SOL_24h',
    'ETH_20m',
    'BNB_20m',
    'ARB_20m'
]
