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

from keras_tuner import HyperParameters
import autokeras as ak

assert len(tf.config.list_logical_devices("GPU")) > 0, "NO GPU FOUND!"


hp = HyperParameters()

def ak_cnn(x_train, y_train, x_test, y_test, M, AK):
    filters = hp.Choice("filters", [8, 16, 32, 64])

    input_node = ak.ImageInput()
    output_node = ak.ConvBlock(filters=filters)(input_node)
    # output_node = ak.DenseBlock()(output_node)
    output_node = ak.ClassificationHead()(output_node)

    ak_frame = ak.AutoModel(input_node, output_node, max_trials=AK["max_trials"], overwrite=AK["overwrite"], 
                            project_name="ak_cnn", directory=f"_ak_trials/", tuner=AK["tuner"])

    ak_history = ak_frame.fit(x_train, y_train, M["batch_size"], M["epochs"], callbacks=M["cbs"], validation_data=[x_test, y_test], 
                              verbose=M["verbose"])
    ak_model = ak_frame.export_model()
    ak_opt = ak_model.optimizer.get_config

    y_test_pred = ak_model.predict(x_test, verbose=M["verbose"])
    score = ak_model.evaluate(x_test, y_test, verbose=M["verbose"])

    return ak_model, ak_history.history, y_test_pred, score


def ak_rnn(x_train, y_train, x_test, y_test, M, AK):
    input_node = ak.Input()
    output_node = ak.RNNBlock(return_sequences=True)(input_node)
    output_node = ak.ClassificationHead()(output_node)

    ak_frame = ak.AutoModel(input_node, output_node, max_trials=AK["max_trials"], overwrite=AK["overwrite"], 
                            project_name="ak_rnn", directory=f"_ak_trials/", tuner=AK["tuner"])

    ak_history = ak_frame.fit(x_train, y_train, M["batch_size"], M["epochs"], callbacks=M["cbs"], validation_data=[x_test, y_test], 
                              verbose=M["verbose"])
    ak_model = ak_frame.export_model()
    ak_opt = ak_model.optimizer.get_config

    y_test_pred = ak_model.predict(x_test, verbose=M["verbose"])
    score = ak_model.evaluate(x_test, y_test, verbose=M["verbose"])
    return ak_model, ak_history.history, y_test_pred, score
    