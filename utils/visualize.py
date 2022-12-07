"""Aggregates all functions for different plots.
plot loss and accuracy.\
plot confusion matrix.\
plot learning rate and learning rate scheduler\
plot activity, smartphone and combined distribution
"""

import os
from os.path import join
import datetime
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pytz
import seaborn as sns
from sklearn import metrics

# own scripts
from utils.preprocess_data import parse_timeseries

# matplotlib.use("agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.sans-serif"] = "DejaVu Sans"
sns.set_theme(style="darkgrid")


def plot_loss_acc(history: dict, results_dir="results/plots") -> None:
    """Plots the models loss and accuracy into one figure and saves them at 'results_dir.

    Args:
        history (dict): Model history.history dictionary
        results_dir (str, optional): Directory to save into. Defaults to "results/plots".
    """

    history = pd.DataFrame(history)
    best_idx = np.argmin(history["val_loss"])
    scores = (history["val_loss"][best_idx], history["val_accuracy"][best_idx])

    fig = plt.figure(figsize=(9, 6))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    # ax1
    ax1.set_ylim(0, 1)
    ax1.set_title(f"Best Validation Loss: {scores[0]:.4f}")

    sns.lineplot(data=history["loss"], ax=ax1)
    sns.lineplot(data=history["val_loss"], ax=ax1)
    ax1.axvline(best_idx, color="r", linestyle="--")
    ax1.legend(["loss", "val_loss"])

    # ax2
    ax2.set_title(f"Corresponding Accuracy: {scores[1]:.4f}")

    sns.lineplot(data=history["accuracy"], ax=ax2)
    sns.lineplot(data=history["val_accuracy"], ax=ax2)
    ax2.axvline(best_idx, color="r", linestyle="--")
    ax2.legend(["accuracy", "val_accuracy"])

    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(join(results_dir, "loss_acc.png"))
    # plt.close()


def plot_confusion_matrix(y_test, y_test_pred, lbl_enc, model_suffix: str = None, results_dir: str = "results/plots"):
    """Plots Confusion matrix for activity classification

    Args:
        y_test (_type_): true activites
        y_test_pred (_type_): predicted activites
        lbl_enc (_type_): Activity Label Encoder to set axis ticks to classes
        model_suffix (str, optional): file suffix. Defaults to None.
        results_dir (str, optional): output directory. Defaults to "results/plots".
    """
    y_test = np.argmax(y_test, axis=1)
    y_test_pred = np.argmax(y_test_pred, axis=1)

    confusion_matrix = metrics.confusion_matrix(y_test, y_test_pred, normalize="true")
    # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=lbl_enc.classes_)

    plt.figure(figsize=(9, 9))
    pred_f1 = round(metrics.f1_score(y_test, y_test_pred, average="macro", zero_division=0), 4)
    pred_acc = round(sum(y_test == y_test_pred) / len(y_test), 4) * 100
    plt.title(f"Accuracy: {pred_acc:.2f} | F1 Score: {pred_f1:.4f}")

    ax = sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt=".2f", cbar=False)

    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("True Values")

    ax.xaxis.set_ticklabels(lbl_enc.classes_, rotation=90)
    ax.yaxis.set_ticklabels(lbl_enc.classes_, rotation=0)

    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(join(results_dir, f"confusion_matrix_{model_suffix}.png"))
    # plt.close()


def plot_lr(lr_scheduler, steps_per_epoch):
    """not used"""
    step = np.arange(0, 10 * steps_per_epoch)
    lr = lr_scheduler(step)
    sns.lineplot(x=step, y=lr)
    plt.title("Learning Rate over the training")
    plt.xlabel("Steps")
    plt.ylabel("Learning Rate")


def plot_lr_schedule(lr_schedule, epochs):
    """
    https://www.kaggle.com/code/akshatpattiwar/hubmap-tensorflow/notebook
    """
    fig = plt.figure(figsize=(10, 6))
    plt.plot([None] + lr_schedule + [None])
    # X Labels
    x = np.arange(1, epochs + 1)
    x_axis_labels = [i if epochs <= 40 or i % 5 == 0 or i == 1 else None for i in range(1, epochs + 1)]
    plt.xlim([1, epochs])
    plt.xticks(x, x_axis_labels)  # set tick step to 1 and let x axis start at 1

    # Increase y-limit for better readability
    plt.ylim([0, max(lr_schedule) * 1.1])

    # Title
    schedule_info = f"start: {lr_schedule[0]:.1E}, max: {max(lr_schedule):.1E}, final: {lr_schedule[-1]:.1E}"
    plt.title(f"Step Learning Rate Schedule, {schedule_info}", size=18, pad=12)

    # Plot Learning Rates
    for x, val in enumerate(lr_schedule):
        if epochs <= 40 or x % 5 == 0 or x is epochs - 1:
            if x < len(lr_schedule) - 1:
                if lr_schedule[x - 1] < val:
                    ha = "right"
                else:
                    ha = "left"
            elif x == 0:
                ha = "right"
            else:
                ha = "left"
            plt.plot(x + 1, val, "o", color="black")
            offset_y = (max(lr_schedule) - min(lr_schedule)) * 0.02
            plt.annotate(f"{val:.1E}", xy=(x + 1, val + offset_y), size=12, ha=ha)

    plt.xlabel("Epoch", size=16, labelpad=5)
    plt.ylabel("Learning Rate", size=16, labelpad=5)
    plt.grid()
    plt.show()


#########################
# visualization of whole timeseries data set

# TODO: put activity and smartphone into one function


def plot_data_distribution_single(
    input_dir: str, key_param: str = "user_model", frequency: int = 5, frame_seconds: int = 2, timestamp_begin="19.10.22 00:00"
):
    assert key_param in ["activity", "user_model"], "activity or user_model are the only valid arguments!"

    csv_fns = [csv_fn for csv_fn in os.listdir(input_dir) if csv_fn.endswith(".csv")]

    frames = frequency * frame_seconds
    hopsize = frames // 2
    frequency_str = f"{frequency}Hz"

    timestamp_begin = pytz.timezone("Europe/Berlin").localize(datetime.datetime.strptime(timestamp_begin, "%d.%m.%y %H:%M"))
    timestamp_begin = timestamp_begin.timestamp() * 1000

    data_dic = {}

    for csv_fn in csv_fns:
        data_df, activity, user_model = parse_timeseries(
            join(input_dir, csv_fn), frequency_str=frequency_str, timestamp_begin=timestamp_begin, common_only=False, n_keep=12
        )
        if data_df is False:
            continue

        if key_param == "activity":
            data_dic[activity] = data_dic.get(activity, 0) + data_df.shape[0] // hopsize
        else:
            data_dic[user_model] = data_dic.get(user_model, 0) + data_df.shape[0] // hopsize

    sorted_act = sorted(data_dic, key=lambda x: data_dic[x], reverse=True)

    data_dic = {act: data_dic[act] for act in sorted_act}

    fig = plt.figure(figsize=(20, 10))

    ax1 = plt.subplot(1, 2, 1)
    plt.barh(list(range(len(data_dic.keys()), 0, -1)), data_dic.values(), tick_label=list(data_dic.keys()))
    # plt.xlabel("Raw amount")
    plt.title(f"Grouped by {key_param} | {frequency} Hz Raw Values")

    ax2 = plt.subplot(1, 2, 2)
    plt.barh(list(range(len(data_dic.keys()), 0, -1)), data_dic.values(), tick_label=list(data_dic.keys()), log=True)
    # plt.xlabel("Log Amount")
    plt.title(f"Grouped by {key_param} | {frequency} Hz Log Values")
    plt.tight_layout()


def plot_data_distribution_combined(input_dir: str, frequency: int = 5, frame_seconds: int = 2, timestamp_begin="19.10.22 00:00"):
    csv_fns = [csv_fn for csv_fn in os.listdir(input_dir) if csv_fn.endswith(".csv")]
    hopsize = (frequency * frame_seconds) // 2
    frequency_str = f"{frequency}Hz"

    timestamp_begin = pytz.timezone("Europe/Berlin").localize(datetime.datetime.strptime(timestamp_begin, "%d.%m.%y %H:%M"))
    timestamp_begin = timestamp_begin.timestamp() * 1000

    dic = {}

    for csv_fn in csv_fns:
        data_df, activity, user_model = parse_timeseries(
            join(input_dir, csv_fn), frequency_str=frequency_str, timestamp_begin=timestamp_begin, common_only=False, n_keep=12
        )

        if data_df is False:
            continue

        dic.setdefault(user_model, {})  # needed for nested dictionaries to work
        dic[user_model][activity] = dic.get(user_model, {}).get(activity, 0) + data_df.shape[0] // hopsize

    df = pd.DataFrame.from_dict(dic).T.fillna(0).pipe(lambda df: df.loc[df.sum(1).sort_values(ascending=True).index, :])

    df.plot(kind="barh", stacked=True, figsize=(20, 10), title=frequency_str)
