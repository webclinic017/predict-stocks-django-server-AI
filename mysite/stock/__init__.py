import json
import sys
from WorldTradingData import WorldTradingData
import re
import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import io
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
from datetime import datetime


def predict_stock(code, days_to_predict):

    my_api_token = "BDy7iOUIGUFljXSr2yybWJ7l1pBF6Dtklb4tBH8a6i1JibHn1XCLq1AviVGj"
    wtd = WorldTradingData(my_api_token)
    data = wtd.history(code)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)

    prices = []
    prices_copy = []

    try:
        dates = list(data["history"].keys())
        days_to_predict = int(days_to_predict)
        if days_to_predict > 50:
            raise Exception
    except:
        return "error", "error", "error", "error", "error"
        
    real_dates = []

    for i in range(len(dates)):
        if int(dates[i][:4]) >= 2019:
            real_dates.append(dates[i])
            prices.append(float(data["history"][dates[i]]["close"]))
            prices_copy.append(float(data["history"][dates[i]]["close"]))

    prices = np.array(prices, dtype="float").reshape(-1, 1)
    prices_copy = np.array(prices_copy, dtype="float").reshape(-1, 1)
    scaler.fit(prices)
    prices = scaler.transform(prices)
    prices_copy = scaler.transform(prices_copy)

    prices = np.array(prices, dtype="float").reshape(1, -1)
    prices = prices[0]
    prices_copy = np.array(prices_copy, dtype="float").reshape(1, -1)
    prices_copy = prices_copy[0]

    for i in range(len(prices)):
        prices[i] = prices_copy[len(prices) - 1 - i]

    X_train= []
    y_train = []

    for i in range(len(prices)):
        try:
            if len(prices[i:i+49]) != 49:
                raise Exception
            if i <= len(prices) - 50:
                X_train.append(prices[i:i+49])
                y_train.append(prices[i+49])
            print(i)
        except:
            pass

    for i in range(len(X_train)):
        X_train[i] = [X_train[i]]

    X_train = np.array(X_train, dtype="float")
    y_train = np.array(y_train, dtype="float")

    model = keras.Sequential([
            keras.layers.LSTM(units=49, return_sequences=True, input_shape=(1,49)),
            keras.layers.LSTM(units=49),
            keras.layers.Dense(1),
    ])

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, y_train, epochs=100)

    aapl_predictions = []
    plot_data = []

    for i in range(len(X_train)):
        aapl_predictions.append(model.predict(np.array([X_train[i]])))

    for i in range(len(aapl_predictions)):
        plot_data.append(aapl_predictions[i][0][0])

    volumes = []
    volumes_copy = []

    for i in range(len(dates)):
        if int(dates[i][:4]) >= 2019:
            volumes.append(float(data["history"][dates[i]]["volume"]))
            volumes_copy.append(float(data["history"][dates[i]]["volume"]))

    volumes = np.array(volumes, dtype="float").reshape(-1, 1)
    volumes_copy = np.array(volumes_copy, dtype="float").reshape(-1, 1)

    scaler_volume = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
    scaler_volume.fit(volumes)
    volumes = scaler_volume.transform(volumes)
    volumes_copy = scaler_volume.transform(volumes_copy)

    volumes = np.array(volumes, dtype="float").reshape(1, -1)
    volumes = volumes[0]
    volumes_copy = np.array(volumes_copy, dtype="float").reshape(1, -1)
    volumes_copy = volumes_copy[0]

    for i in range(len(volumes)):
        volumes[i] = volumes_copy[len(volumes) - 1 - i]

    X_train_volume = []
    y_train_volume = []

    for i in range(len(volumes)):
        try:
            if len(volumes[i:i+49]) != 49:
                raise Exception
            if i <= len(volumes) - 50:
                X_train_volume.append(volumes[i:i+49])
                y_train_volume.append(volumes[i+49])
            print(i)
        except:
            pass

    for i in range(len(X_train_volume)):
        X_train_volume[i] = [X_train_volume[i]]

    X_train_volume = np.array(X_train_volume, dtype="float")
    y_train_volume = np.array(y_train_volume, dtype="float")

    volume_model = keras.Sequential([
            keras.layers.LSTM(units=49, return_sequences=True, input_shape=(1,49)),
            keras.layers.LSTM(units=49),
            keras.layers.Dense(1),
    ])

    volume_model.compile(loss='mean_squared_error', optimizer='adam')

    volume_model.fit(X_train_volume, y_train_volume, epochs=150)

    aapl_predictions_volume = []
    plot_data_volume = []

    for i in range(len(X_train_volume)):
        aapl_predictions_volume.append(model.predict(np.array([X_train_volume[i]])))

    for i in range(len(aapl_predictions_volume)):
        plot_data_volume.append(aapl_predictions_volume[i][0][0])


    final_set_volume = X_train_volume[len(X_train_volume) - 1]

    final_set_copy_volume = X_train_volume[len(X_train_volume) - 1]

    preds = []

    for i in range(days_to_predict):
        pred = volume_model.predict(np.array([final_set_volume], dtype="float"))
        preds.append(pred)
        final_set_volume = np.delete(final_set_volume[0], 0)
        final_set_volume = [final_set_volume]
        final_set_volume = np.append(final_set_volume, pred)
        final_set_volume = [final_set_volume]

    setted_volume = []

    for i in range(len(preds)):
        setted_volume.append(preds[i][0][0])

    def calc_obv(volume_set, price_set):
        obv = 0
        try:
            volume_set_copy = volume_set[0]
            for i in range(1, len(volume_set_copy)):
                if price_set[i] > price_set[i-1]:
                    obv += volume_set_copy[i-1]
                else:
                    obv -= volume_set_copy[i-1]
        except:
            for i in range(1, len(volume_set)):
                if price_set[i] > price_set[i-1]:
                    obv += volume_set[i-1]
                else:
                    obv -= volume_set[i-1]
        return obv

    final_set = X_train[len(X_train) - 1]

    final_set_copy = X_train[len(X_train) - 1]

    preds = []

    obv = calc_obv(final_set_copy_volume[0], final_set_copy[0])

    for i in range(days_to_predict):
        pred = model.predict(np.array([final_set], dtype="float"))
        preds.append(pred[0][0])
        pred = pred[0][0]
        if obv > 0:
            if pred > preds[i-1]:
                print()
            elif pred < preds[i-1]:
                change = pred - preds[i-1]
                preds[i] = preds[i] - change
        elif obv < 0:
            if pred < preds[i-1]:
                print()
            elif pred > preds[i-1]:
                change = pred - preds[i-1]
                preds[i] = preds[i] - change
        final_set = np.delete(final_set[0], 0)
        final_set = [final_set]
        final_set = np.append(final_set, pred)
        final_set = [final_set]
        final_set_copy_volume = np.delete(final_set_copy_volume[0], 0)
        final_set_copy_volume = [final_set_copy_volume]
        final_set_copy_volume = np.append(final_set_copy_volume, setted_volume[i])
        final_set_copy_volume = [final_set_copy_volume]
        obv = calc_obv(final_set_copy_volume, final_set[0])

    setted = []

    for i in range(len(preds)):
        setted.append(preds[i])

    prev = final_set_copy[0]

    plot_data = np.concatenate((prices[:49], np.concatenate((y_train, setted))))

    print(49, len(y_train), len(setted))

    plot_data = scaler.inverse_transform(plot_data.reshape(-1, 1))
    prices = scaler.inverse_transform(prices.reshape(-1, 1))

    prices = prices[:len(plot_data)]

    setted_volume = []

    print(final_set_volume)

    for i in range(len(final_set_volume[0])):
        setted_volume.append(final_set_volume[0][i])

    plot_data_volume = np.concatenate((volumes[:49], np.concatenate((y_train_volume, setted_volume[49-days_to_predict:49]))))

    print(49, len(y_train_volume), len(setted_volume[49-days_to_predict:49]))

    plot_data_volume = scaler_volume.inverse_transform(plot_data_volume.reshape(-1, 1))
    volumes = scaler_volume.inverse_transform(volumes.reshape(-1, 1))

    volumes = volumes[:len(plot_data_volume)]

    latest_date = datetime.today().weekday()

    if latest_date > 4:
        latest_date = 4

    last_date_in_series = real_dates[0]

    print(last_date_in_series)

    num_of_weeks = int(days_to_predict/7) + 10

    num_of_days = days_to_predict + 2*num_of_weeks

    new_dates = pd.date_range(start=last_date_in_series, periods=num_of_days)

    print(new_dates)

    more_dates = []

    for i in range(len(new_dates)):
        if latest_date < 5:
            more_dates.append(new_dates[i])
        latest_date = latest_date + 1 if latest_date < 6 else 0

    print(more_dates)

    real_dates_copy = []

    for i in range(len(real_dates)):
        real_dates[i] = pd.to_datetime(real_dates[i])
        real_dates_copy.append(pd.to_datetime(real_dates[i]))

    for i in range(len(real_dates)):
        real_dates[i] = real_dates_copy[len(real_dates) - 1 - i]

    real_dates = np.concatenate((real_dates, more_dates))
    print("----------------------")
    print(len(real_dates), len(plot_data_volume))

    return plot_data, prices, plot_data_volume, volumes, real_dates
























