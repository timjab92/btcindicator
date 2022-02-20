from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping
from btcindicator.params import *

def build_model():
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
