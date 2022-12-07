import sys
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from os.path import join
import numpy as np
import pandas as pd
import re

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# add keras blocks
from keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense, LSTM, TimeDistributed
from keras import Sequential
from keras.callbacks import EarlyStopping

assert len(tf.config.list_logical_devices("GPU")) > 0, "NO GPU FOUND!"


def CnnModel(x_train, y_train, x_test, y_test, M, *args):
    n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=M["epochs"], batch_size=M["batch_size"], verbose=M["verbose"], callbacks=M["cbs"], 
                        validation_data=[x_test, y_test])
    y_test_predict = model.predict(x_test, verbose=M["verbose"])
    score = model.evaluate(x_test, y_test, verbose=M["verbose"])
    return model, history.history, y_test_predict, score

def LstmModel(x_train, y_train, x_test, y_test, M, *args):
    n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

    model = Sequential()
    model.add(LSTM(150,return_sequences=True, input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.5))
    model.add(LSTM(150 ,input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=M["epochs"], batch_size=M["batch_size"], verbose=M["verbose"], callbacks=M["cbs"], validation_data=[x_test, y_test])
    y_test_predict = model.predict(x_test, verbose=M["verbose"])
    score = model.evaluate(x_test, y_test, verbose=M["verbose"])
    return model, history.history, y_test_predict, score

def LstmCnnModel(x_train, y_train, x_test, y_test, M, *args):
    n_steps, n_length = 2, 5
    pool_size = 1
    n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
    xx_train = x_train.reshape((x_train.shape[0], n_steps, n_length, n_features))
    xx_test = x_test.reshape((x_test.shape[0], n_steps, n_length, n_features))

    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
    model.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(150,return_sequences=True,))
    model.add(Dropout(0.5))
    model.add(LSTM(150))
    model.add(Dropout(0.5))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(xx_train, y_train, epochs=M["epochs"], batch_size=M["batch_size"], verbose=M["verbose"], callbacks=M["cbs"], validation_data=[xx_test, y_test])
    y_test_predict = model.predict(xx_test, verbose=M["verbose"])
    score = model.evaluate(xx_test, y_test, verbose=M["verbose"])
    return model, history.history, y_test_predict, score