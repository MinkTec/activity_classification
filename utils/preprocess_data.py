import os
from os.path import join, basename
import random
import datetime

# from functools import partial
import re
import numpy as np
import pandas as pd
import pytz

# import sklearn
from sklearn.decomposition import PCA

# from skimage.transform import resize
# import matplotlib.pyplot as plt


def load_armans_dataset(data_dir: str = "input/separated", frames: int = 10, n_keep: int = 12, trimmed_act=True, verbose=True):
    """DEPRECATED
    parse armans (student thesis) data (30 persons, ~14 activities into) into one dataset.
    Split into train and validation set

    Args:
        data_dir (str, optional): Defaults to "input/separated".
        frames (int, optional): Defaults to 10. How large the windows should be (50% overlap).
        trimmed_act (bool, optional): Default to True. If True, trims useless activities and combines similar ones
        print_shapes (bool, optional): Defaults to True.

    Returns:
        x_train, y_train, x_test, y_test : numpy arrays
        y is returned as an array of the activity strings
    """
    if frames < 1:
        raise ValueError("frames has to be an int of at least '1'")

    act_re = re.compile(r"^\d+_(\w+)$")  # e.g. 1_sleeping_right
    activities = [re.search(act_re, act_str).string for act_str in os.listdir(data_dir) if re.search(act_re, act_str)]
    activities.sort(key=lambda x: int(x.split("_")[0]))  # sort according to number in filename

    act_dic = {}  # dict with {1: sleeping_right, etc...}
    pattern_re = re.compile(r"(\d+)_(\w+)")
    for act_str in activities:
        re_res = re.search(pattern_re, act_str)
        act_dic[int(re_res.group(1))] = re_res.group(2)

    x_train_list, y_train_list, x_val_list, y_val_list = [], [], [], []

    # loop through every directory e.g 1_sleeping_right
    for (key, val) in act_dic.items():
        act_dir = join(data_dir, f"{key}_{val}")
        # loop through every participants file (1-30)
        for fn in range(1, 31):
            try:
                df_act = pd.read_csv(join(act_dir, f"{fn}.csv")).iloc[:, 1:]
            except FileNotFoundError as exception:  # some files are missing
                if verbose:
                    print(exception)
                continue

            if trimmed_act:  # remove pilot activities like bending forward etc.; combining indoor + outdoor
                if val in ["sitting"]:
                    activity = "sitting"
                elif val in ["standing"]:
                    activity = "standing"
                elif val in ["sleeping_right", "sleeping_back", "sleeping_left"]:
                    activity = "laying"
                elif val in ["walking_indoor", "walking_outdoor"]:
                    activity = "walking"
                elif val in ["jogging_outdoor"]:
                    activity = "jogging"
                elif val in ["running_outdoor"]:
                    activity = "running"
                elif val in ["stair"]:
                    activity = "walkingUpstairs"
                else:
                    continue
            else:
                activity = val

            hop_size = max(frames // 2, 1)  # stepsize: 50 % overlapping windows
            # loop through the rows of the csv files and create windows of size frames with 50% overlap
            for i in range(0, df_act.shape[0] - frames, hop_size):
                x_temp = df_act.iloc[i : i + frames, :]

                if fn in range(1, 24):  # participants for training set
                    x_train_list.append(x_temp)
                    y_train_list.append(activity)
                else:  # participants for validation set
                    x_val_list.append(x_temp)
                    y_val_list.append(activity)

    x_train, x_val = np.array(x_train_list, dtype=np.float32), np.array(x_val_list, dtype=np.float32)
    y_train, y_val = np.array(y_train_list), np.array(y_val_list)

    def cut_sensors(x, n_keep):
        """ONLY for Armans dataset (NO GYRO DATA)"""
        if n_keep is None:  # do nothing
            return x

        n_sensors = (x.shape[-1] - 3) // 2
        x_left = x[:, :, :n_keep]
        x_right = x[:, :, n_sensors : n_sensors + n_keep]
        x_acc = x[:, :, -3:]

        return np.concatenate((x_left, x_right, x_acc), axis=-1)

    x_train = cut_sensors(x_train, n_keep)
    x_val = cut_sensors(x_val, n_keep)

    if verbose:
        print("x_train shape:", x_train.shape, "\ny_train shape:", y_train.shape)
        print("x_val shape:", x_val.shape, "\ny_val shape:", y_val.shape)

    return x_train, y_train, x_val, y_val


def parse_timeseries(csv_path: str, frequency_str: str, timestamp_begin: int, common_only: bool, n_keep: int):
    timestamp, brand, model_brand, sensor_id, freq_measured, activity, *note = basename(csv_path)[:-4].split("_")
    user_model = f"{brand}_{model_brand}"  # e.g. samsung_SM-A528B

    # only return timeseries that started after timestamp_begin
    if int(timestamp) < timestamp_begin:
        return (False,) * 3

    if activity == "other":
        return (False,) * 3

    # just use the most common activities for now
    if common_only and activity not in ["cycling", "sitting", "standing", "walking"]:
        return (False,) * 3

    # only use the data with the frequency you want: 5Hz or 15Hz
    if freq_measured != frequency_str:
        return (False,) * 3

    data_df = pd.read_csv(csv_path, header=None).iloc[:, :-2]  # drop voltage and time
    data_df = cut_sensors(data_df, n_keep)  # only keep n_keep amount of sensors left and right
    data_df["move_bool"] = recognize_movement(data_df, move_tresh=100)  # add column that judges whether you are in movement or now

    return data_df, activity, user_model


# def load_timeseries_dataset(input_dir: str = "input/timeseries_data", frames: int = 10, n_keep: int = 12, frequency: int = 5):
def load_timeseries_dataset(
    input_dir: str = "input/timeseries_data",
    frequency: int = 5,
    frame_seconds: int = 2,
    n_keep: int = 12,
    timestamp_begin: str = "19.10.22 00:00",
    common_only: bool = False,
):
    """Parse raw timeseries (downloaded with aws_downloader.ipynb) into Trainings and Validation Set with Windows of size frames.

    Args:
        input_dir (str, optional): Directory that stores timeseries_data as csv files. Defaults to "input/timeseries_data".
        frequency (int, optional): Choose between 5 and 15 Hz right now
        frame_seconds (int, optional): Seconds in one datapoint/ window frame
        n_keep (int, optional): How many Sensors to keep. Defaults to 12.

    Returns:
        tuple of numpy_array: x_train, y_train, x_val, y_val
    """
    assert frequency in [5, 15], "Frequency has to set to 5 or 15 Hz!"
    frames = frequency * frame_seconds
    hopsize = max(frames // 2, 1)  # stepsize: 50 % overlapping windows
    frequency_str = f"{frequency}Hz"

    timestamp_begin = pytz.timezone("Europe/Berlin").localize(datetime.datetime.strptime(timestamp_begin, "%d.%m.%y %H:%M"))
    timestamp_begin = timestamp_begin.timestamp() * 1000

    csv_fns = [csv_path for csv_path in os.listdir(input_dir) if csv_path.endswith(".csv")]
    print(len(csv_fns), "Files read in total (ignoring different frequencies)")

    x_train_list, x_val_list, x_test_list = [], [], []
    y_train_list, y_val_list, y_test_list = [], [], []

    for csv_fn in csv_fns:
        data_df, activity, user_model = parse_timeseries(
            join(input_dir, csv_fn), frequency_str=frequency_str, timestamp_begin=timestamp_begin, common_only=common_only, n_keep=n_keep
        )

        # simplest classification
        if activity not in ["sitting", "standing", "walking"]:
            continue

        if data_df is False:
            continue

        # loop through the rows of the csv files and create windows of size frames with 50% overlap
        for i in range(0, data_df.shape[0] - frames, hopsize):
            temp_df = data_df.iloc[i : i + frames, :]
            # implement statement so it only works on moving activities (walking, cycling, etc.)
            if activity not in ["sitting", "standing", "sleeping", "driving"] and temp_df["move_bool"].all() is False:
                continue

            if random.randint(0, 1):
                # implement statement to split data into train, val and test set. also use sensor_model
                # split by different sensors for now
                x_train_list.append(temp_df.iloc[:, :-1])  # don't append the move_bool column
                y_train_list.append(activity)
            else:
                x_val_list.append(temp_df.iloc[:, :-1])
                y_val_list.append(activity)

    x_train, x_val = np.array(x_train_list), np.array(x_val_list)
    y_train, y_val = np.array(y_train_list), np.array(y_val_list)

    return x_train, y_train, x_val, y_val


def cut_sensors(x: np.ndarray, n_keep: int):
    """Remove all Sensors greater than n_keep from x. Works only on data with acceleration and gyro data"""

    if x.shape[-1] % 2 == 0:  # with gyro data and in dataframes
        n_sensors = (x.shape[-1] - 3 - 3) // 2
        x_left = x.iloc[:, :n_keep]
        x_right = x.iloc[:, n_sensors : n_sensors + n_keep]
        x_acc = x.iloc[:, -6:-3]
        x_gyro = x.iloc[:, -3:]

        return pd.concat((x_left, x_right, x_acc, x_gyro), axis=1)
    else:  # for armans dataset; without gyro data and as numpy array
        n_sensors = (x.shape[-1] - 3) // 2
        x_left = x[:, :, :n_keep]
        x_right = x[:, :, n_sensors : n_sensors + n_keep]
        x_acc = x[:, :, -3:]

        return np.concatenate((x_left, x_right, x_acc), axis=-1)


def recognize_movement(df: pd.DataFrame, move_tresh: int = 100):
    """Receives the timeseries input and decides whether FlexTail wearer is moving there or not.
    'No Movement' is then eliminated for categories like cycling while standing at a traffic light.

    Args:
        df (DataFrame): CSV timeseries_data as DataFrame
        move_tresh (int, optional): Decides True/False when movement is recognized. Defaults to 100.

    Returns:
        _type_: _description_
    """
    acc_df = df.iloc[:, -6:-3]  # only acceleration columns
    acc_df.columns = ["x", "y", "z"]
    # acc_df_windows = acc_df.rolling(10).std().iloc[10:,:]
    acc_df_windows = acc_df.rolling(10).std()

    # acc_std = acc_df_windows.apply(lambda row: row["x"] + row["y"] + row["z"], axis=1)
    # acc_std = acc_df_windows.apply(lambda row: row["x"] + row["z"], axis=1)  # y direction (left,right should make no difference)
    acc_std = acc_df_windows.apply(lambda row: row.x + row.z, axis=1)  # y direction (left,right should make no difference)
    # y_gyro hat potential
    acc_std.fillna(move_tresh, inplace=True)

    return acc_std >= move_tresh


def fix_for_missing_walking_upstairs(y):
    y -= y > np.ones(len(y)) * 11
    y -= 1

    return y


def to_onehot(y):
    assert y.ndim == 1
    n = y.shape[0]
    cat = np.zeros((n, np.max(y) + 1), dtype="float32")
    cat[np.arange(n), y] = 1
    return cat


class Standardize:
    """Creating standardizer with Z-score normalization.
        Calculate mean and std over axis passed in initialization.

    Args:
        ax (tuple(int)): Normalizes over these axis. (Defaults to (0,1))

    """

    def __init__(self, ax: tuple = (0, 1)):
        self.ax = ax
        self.mean = None
        self.std = None

    def fit_transform(self, x):
        self.mean = x.mean(self.ax, keepdims=True)
        self.std = x.std(self.ax, keepdims=True) + 1e-7

        return self._transform(x)

    def transform(self, x):
        if self.mean is None or self.std is None:
            raise NameError("Run fit_transform first!")
        return self._transform(x)

    def _transform(self, x):
        return (x - self.mean) / self.std

    def __repr__(self):
        return "standardize"


class Minmax:
    def __init__(self, rang=(0, 1)):
        self.lower_bound, self.higher_bound = rang

    def fit_transform(self, x):
        self.min = x.min((0, 1), keepdims=True)
        self.max = x.max((0, 1), keepdims=True)

        return self._transform(x)

    def transform(self, x):
        if not hasattr(self, "min") and not hasattr(self, "max"):
            raise NameError("Run fit_transform first!")
        return self._transform(x)

    def _transform(self, x):
        x = (x - self.min) / (self.max - self.min)

        return x * (self.higher_bound - self.lower_bound) + self.lower_bound

    def __repr__(self):
        return "minmax"


def scale_x(x_train, x_test, normalizer_str):
    scaler = Standardize() if normalizer_str == "standard" else (Minmax() if normalizer_str == "minmax" else None)
    if scaler:
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    return x_train, x_test


def flatten_but_batch(x):
    return x.reshape((x.shape[0], np.multiply(*x.shape[1:])))


def adjust_dimensions(x_train: np.ndarray, x_test: np.ndarray, n_components: int = 15, return_components: bool = False):
    """Reduce the features of the data into n_components with PCA/SVD

    Args:
        x_train (np.ndarray):
        x_val (np.ndarray):
        n_components (int, optional): How many dimensions to keep from PCA. Defaults to 15.
        return_components (bool, optional): Additionally, return the PCA_components matrix. Defaults to False.

    Returns:
        tuple(np.ndarray): x_train, x_test, (pca.components_.T)
    """
    pca = PCA(n_components=n_components, svd_solver="full")
    # pca = PCA(n_components=n_components, svd_solver="full", whiten=True)
    n_train, ts, feats = x_train.shape
    assert feats >= n_components, "reduce the number of n_components!"
    n_test = x_test.shape[0]

    # x_train_strain = pca.fit_transform(x_train[..., :-3].reshape((n_train * ts, feats - 3)))
    # x_test_strain = pca.transform(x_val[..., :-3].reshape((n_test * ts, feats - 3)))
    # x_train = np.concatenate((x_train_strain.reshape(n_train, ts, n_components), x_train[..., -3:]), axis=-1)
    # x_test = np.concatenate((x_test_strain.reshape(n_test, ts, n_components), x_val[..., -3:]), axis=-1)

    # PCA over all features
    x_train = pca.fit_transform(x_train.reshape((n_train * ts, feats)))
    x_test = pca.transform(x_test.reshape((n_test * ts, feats)))

    x_train = x_train.reshape(n_train, ts, n_components)
    x_test = x_test.reshape(n_test, ts, n_components)

    print(pca.explained_variance_ratio_)

    if return_components:
        return x_train, x_test, pca.components_.T
    return x_train, x_test
