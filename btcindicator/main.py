import pandas as pd
from btcindicator.params import *
from btcindicator.utils import *
from btcindicator.model import *
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential, layers



def get_data():
    """return the data from cryptodownload.com"""
    url = "https://www.cryptodatadownload.com/cdd/Bitstamp_BTCUSD_1h.csv"
    print('downloading data...')
    r = requests.get(url, allow_redirects=True, verify=False)

    print('writing data to file...')
    open('raw_data/Bitstamp_BTCUSD_1h.csv', 'wb').write(r.content)

    print("finished!")
    btc = pd.read_csv(
        "raw_data/Bitstamp_BTCUSD_1h.csv", skiprows=1)
    btc.date = pd.to_datetime(btc.date)
    btc.set_index("date", inplace=True)

    btc4h = btc.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                    'close': 'last', 'Volume BTC': 'mean', 'Volume USD': 'mean'})
    return btc4h

def split_data(selected_data):
    split = int(len(selected_data) * TRAIN_PERCENTAGE)
    data_train = selected_data[:split]
    data_test = selected_data[split:]
    return data_train, data_test

def scale_data(data_train, data_test):
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    minmax_scaler.fit(data_train)

    data_train_scaled = minmax_scaler.transform(data_train)
    data_test_scaled = minmax_scaler.transform(data_test)
    min1 = minmax_scaler.data_min_[0]
    range1 = minmax_scaler.data_range_[0]

    return data_train_scaled, data_test_scaled, range1, min1


def get_sequences(data, window_size=WINDOW_SIZE, horizon=HORIZON):
    fake_x = []
    fake_y = []
    for i in range(len(data) - window_size - horizon - 1):
        fake_x.append(data[i:i+window_size])
        fake_y.append(data[i+window_size:i+window_size+horizon])
    x = np.array(fake_x)[:, :, :]
    y = np.array(fake_y)[:, :, 0]
    return x, y

##### CLASS TRAINER #####

class Trainer(object):
    def __init__(self):
        """"This class has two options: train and predict"""

    def build_model(self):
        # Build the LSTM model
        model = Sequential()
        model.add(layers.LSTM(units=16,
                              return_sequences=True,
                              activation="tanh",
                              input_shape=X_train[0].shape))
        model.add(layers.LSTM(units=8,
                              return_sequences=False,
                              activation="relu"
                              ))
        model.add(layers.Dense(HORIZON, activation="linear"))
        model.compile(loss="mse",
                      optimizer="adam",  # rmsprop
                      metrics="mae")

        return model


    def train_model(self):
        print("building model...")
        model = build_model()

        es = EarlyStopping(patience=5,
                           restore_best_weights=True,
                           monitor="mae"
                           )
        print("training model...")
        history = model.fit(X_train, y_train,
                        validation_split=0.3,
                        batch_size=16,
                        epochs=50,
                        verbose=2,
                        callbacks=[es]
                        )
        print("training complete...")
        return model.evaluate(X_test, np.array(y_test), verbose=1)






if __name__ == '__main__':
    print("starting...")
    btc4h = get_data()
    data = feature_engineer(btc4h).dropna().copy()
    print(f"selected features are {SELECTED_FEATURES}")
    selected_data = data[SELECTED_FEATURES]
    print("splitting...")
    data_train, data_test = split_data(selected_data)
    print("scaling...")
    data_train_scaled, data_test_scaled, range1, min1 = scale_data(
        data_train, data_test)
    print("producing sequences...")
    X_train, y_train = get_sequences(data_train_scaled)
    X_test, y_test = get_sequences(data_test_scaled)
    print(f"X_train.shape {X_train.shape}, y_train.shape {y_train.shape}")
    trainer = Trainer()
    trainer.train_model()
    print("finitooooo!!")
