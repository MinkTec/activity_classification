import os
import time
from os.path import join
from shutil import rmtree

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import IPython
import keras
import keras.backend as K
import keras_tuner
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from keras.activations import sigmoid
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard)
from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv1D, Dense, Dropout, Flatten,
                          GlobalAveragePooling1D, Input, LayerNormalization,
                          MaxPool1D, Softmax, SpatialDropout1D)
from keras.models import Model, Sequential, load_model
from tensorflow_addons.optimizers import AdamW, Lookahead, RectifiedAdam

# own scripts
from utils.metrics import calculate_metrics
from utils.optim import lrfn
from utils.visualize import plot_confusion_matrix, plot_loss_acc

assert len(tf.config.list_logical_devices("GPU")) > 0, "NO GPU FOUND!"


class Classifier_INCEPTION(keras_tuner.HyperModel):
    def __init__(
        self,
        output_directory,
        input_shape,
        nb_classes,
        lbl_enc,
        callbacks=[],
        verbose=False,
        batch_size=64,
        max_trials=3,
        nb_filters=32,
        use_residual=True,
        use_bottleneck=True,
        kernel_size=41,
        epochs=100,
    ):
        self.output_directory = output_directory
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.lbl_enc = lbl_enc
        self.callbacks = callbacks
        self.verbose = verbose
        self.batch_size = batch_size
        self.max_trials = max_trials
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.kernel_size = kernel_size - 1
        self.epochs = epochs
        self.bottleneck_size = 32

    def _inception_module(self, input_tensor, stride=1, activation="linear", hp=None):
        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = Conv1D(
                filters=self.bottleneck_size,
                kernel_size=1,
                padding="same",
                activation=activation,
                use_bias=False,
            )(input_tensor)
        else:
            input_inception = input_tensor

        kernel_size_s = [self.kernel_size // (2**i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(
                Conv1D(
                    filters=self.nb_filters,
                    kernel_size=kernel_size_s[i],
                    strides=stride,
                    padding="same",
                    activation=activation,
                    use_bias=False,
                )(input_inception)
            )
            conv_list.append(SpatialDropout1D(hp.Choice("dropout_1", [0.0, 0.25, 0.5]))(input_inception))

        max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding="same")(input_tensor)

        conv_6 = Conv1D(
            filters=self.nb_filters,
            kernel_size=1,
            padding="same",
            activation=activation,
            use_bias=False,
        )(max_pool_1)

        conv_list.append(conv_6)

        x = Concatenate(axis=2)(conv_list)
        x = BatchNormalization()(x)

        x = Activation(activation="relu")(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor, hp=None):
        shortcut_y = Conv1D(
            filters=int(out_tensor.shape[-1]),
            kernel_size=1,
            padding="same",
            use_bias=False,
        )(input_tensor)
        shortcut_y = SpatialDropout1D(hp.Choice("dropout_2", [0.0, 0.25, 0.5]))(shortcut_y)
        shortcut_y = BatchNormalization()(shortcut_y)

        x = Add()([shortcut_y, out_tensor])
        x = Activation("relu")(x)
        return x

    def build(self, hp):
        input_layer = Input(self.input_shape)

        x = input_layer
        input_res = input_layer

        depth = hp.Int("depth", 4, 6, default=6)
        for d in range(depth):

            x = self._inception_module(x, hp=hp)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x, hp=hp)
                input_res = x

        gap_layer = tfa.layers.AdaptiveAveragePooling1D(1, data_format="channels_last")(x)
        gap_layer = Flatten()(gap_layer)

        gap_layer = Dropout(hp.Choice("dropout_dense", [0.0, 0.25, 0.5]))(gap_layer)

        output_layer = Dense(self.nb_classes, activation="sigmoid")(gap_layer)
        # output_layer = Dense(self.nb_classes)(gap_layer)

        model = Model(inputs=input_layer, outputs=output_layer)

        lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        wd = hp.Float("weight_decay", 1e-10, 1e-5, sampling="log")

        # opt_str = hp.Choice("opt_type", ["adamW", "Radam", "ranger"])
        # if opt_str == "adamW":
        #     opt = AdamW(learning_rate=lr, weight_decay=wd)
        # elif opt_str == "Radam":
        #     opt = RectifiedAdam(lr, weight_decay=wd)
        # elif opt_str == "ranger":
        #     opt = Lookahead(RectifiedAdam(lr, weight_decay=wd))
        opt = AdamW(learning_rate=lr, weight_decay=wd)

        model.compile(
            # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            loss=f1_loss,
            optimizer=opt,
            metrics=["accuracy", tfa.metrics.F1Score(self.nb_classes, average="macro")],
        )

        LR_SCHEDULE = [lrfn(step, num_warmup_steps=0, lr_max=lr, epochs=self.epochs, num_cycles=0.50) for step in range(self.epochs)]
        lr_scheduler_cbs = LearningRateScheduler(lambda step: LR_SCHEDULE[step], verbose=0)

        self.callbacks.append(LearningRateScheduler(lambda step: LR_SCHEDULE[step], verbose=0))
        self.callbacks.append(EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True))
        self.callbacks.append(ClearTrainingOutput())

        return model

    def build_model_default(self) -> None:
        """Build model with default hyperparameters if you want to skip the HP tuning process."""
        self.model = self.build(keras_tuner.HyperParameters())

    def build_model_best(self) -> None:
        """Build model with the best tuned hyperparameters.
        Only works after self.tune_hp() is run and self.best_hps is created to store the best hyperparameters during tuning"""
        if hasattr(self, "best_hps"):
            self.model = self.build(self.best_hps[0])
        else:
            class_str = self.__class__.__name__
            raise ValueError(f"{class_str} has no best_hps defined list. Run {class_str}.tune_hp() first!.")

    def tune_hp(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        tuner = keras_tuner.BayesianOptimization(
            self.build,
            # objective=keras_tuner.Objective("val_loss", "min"),
            objective=keras_tuner.Objective(name="val_f1_score", direction="max"),
            max_trials=self.max_trials,
            num_initial_points=None,
            alpha=1e-4,
            beta=2.6,
            executions_per_trial=1,
            overwrite=False,
            directory="_tuner_trials",
            project_name="inception_bayesian",
        )

        # tuner = keras_tuner.Hyperband(
        #     self.build,
        #     objective=keras_tuner.Objective(name="val_f1_score", direction="max"),
        #     max_epochs=self.epochs,
        #     factor=3,
        #     hyperband_iterations=1,
        #     overwrite=False,
        #     directory="_tuner_trials",
        #     project_name="inception_hyper",
        # )

        print(tuner.search_space_summary())

        tuner.search(
            x_train, y_train, epochs=self.epochs, validation_data=[x_val, y_val], batch_size=self.batch_size, callbacks=self.callbacks
        )

        self.duration = time.time() - start_time
        self.best_hps = tuner.get_best_hyperparameters(5)

        print(self.best_hps[0].values)
        df_best_hps = pd.DataFrame.from_records([hps.values for hps in self.best_hps])

        self.model_tuner = tuner.get_best_models()[0]  # best model from tuner
        self.model = self.build(self.best_hps[0])  # best parameters from tuner -> train from scratch in fit
        try:
            rmtree(join(tuner.directory, tuner.project_name))

            os.makedirs(self.output_directory, exist_ok=True)
            df_best_hps.to_csv(join(self.output_directory, "best_hyperparameters.csv"))
        except Exception as e:
            print(e)

    def fit(self, x_train, y_train, x_val, y_val, *args, **kwargs):

        start_time = time.time()
        hist = self.model.fit(
            x_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
            validation_data=(x_val, y_val),
            callbacks=self.callbacks,
            *args,
            **kwargs,
        )

        fit_duration = time.time() - start_time

        # the plot is only cosmetic. self.model and its history is not used if self.model_tuner exists.
        plot_loss_acc(hist.history, self.output_directory)
        self.duration = self.tuner_duration if hasattr(self, "tuner_duration") else fit_duration

        keras.backend.clear_session()

    def predict(self, x_val, y_val):
        inf_model = Sequential([self.model_tuner, Sigmoid()]) if hasattr(self, "model_tuner") else Sequential([self.model, Sigmoid()])
        inf_model.compile(optimizer=self.model.optimizer, loss=self.model.loss, metrics=self.model.metrics)
        y_pred = inf_model.predict(x_val, batch_size=self.batch_size, verbose=self.verbose)
        inf_model.save(join(self.output_directory, "best_model"))  # save the inference model

        # evaluate the retrained model with the best tuner parameters
        df_metrics = calculate_metrics(y_val, y_pred, self.duration, join(self.output_directory, "metrics.csv"))
        plot_confusion_matrix(y_val, y_pred, self.lbl_enc, "", self.output_directory)

        return df_metrics


class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


class Sigmoid(tf.keras.layers.Layer):
    def call(self, inputs):
        return sigmoid(inputs)


def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, "float32"), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), "float32"), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, "float32"), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), "float32"), axis=0)

    p = tp / (tp + fp + K.epsilon())
    # p = tp / (tp + fp + 1e-3)
    r = tp / (tp + fn + K.epsilon())
    # r = tp / (tp + fn + 1e-3)

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)
