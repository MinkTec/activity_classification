import time
import numpy as np
import pandas as pd
import os
from os.path import join
from datetime import datetime
from IPython.display import clear_output
from sklearn.preprocessing import LabelEncoder

from utils.visualize import plot_loss_acc, plot_confusion_matrix
from utils.preprocess_data import load_armans_dataset, scale_x, adjust_dimensions, to_onehot


def run_models(model_func_list, model_dic, ak_dic=None, run_combinations=False):
    """Runs the models in 'ak_model' and 'default_models' in turn

    Args:
        model_func_list (_type_): _description_
        model_dic (_type_): _description_
        ak_dic (_type_, optional): _description_. Defaults to None.
        run_combinations (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    date_now = datetime.now()
    results_dir = join("results", f"model_zoo_{date_now.now().strftime('%Y-%m-%d_%H:%M:%S')}")
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/scores.csv", "w") as f:
        f.write("model_name,normalizer,PCA,duration(in s),val_loss,val_acc\n")

    if run_combinations:
        normalizers = ["No", "standard", "minmax"]
        pca_bools = [True, False]
    else:
        normalizers = [model_dic["normalizer"]]
        pca_bools = [model_dic["PCA"]]

    for normalizer_str in normalizers:
        model_dic["normalizer"] = normalizer_str
        for pca_bool in pca_bools:
            model_dic["PCA"] = pca_bool

            x_train, y_train, x_test, y_test = load_armans_dataset("input/separated", trimmed_act=True, print_shapes=False)
            if model_dic["PCA"]:
                x_train, x_test = adjust_dimensions(x_train, x_test)
            x_train, x_test = scale_x(x_train, x_test, model_dic["normalizer"])

            lbl_enc = LabelEncoder()
            y_lbl = lbl_enc.fit_transform(np.concatenate((y_train, y_test)))

            y_train = y_lbl[: len(y_train)]
            y_test = y_lbl[len(y_train) :]

            y_train = to_onehot(y_train)
            y_test = to_onehot(y_test)

            for model_func in model_func_list:
                print(model_func.__name__, "is training!")
                run_model(x_train, y_train, x_test, y_test, lbl_enc, model_func, model_dic, ak_dic, results_dir)
                print(model_func.__name__, "is finished!")

    scores_df = pd.read_csv(join(results_dir, "scores.csv"))
    scores_df.sort_values(by="val_acc", inplace=True, ascending=False)
    scores_df.to_csv(join(results_dir, "scores.csv"))
    return scores_df


def run_model(x_train, y_train, x_test, y_test, lbl_enc, model_func, model_dic, ak_dic=None, results_dir=None):
    st_time = time.time()
    model_name = model_func.__name__

    if model_name.startswith("ak_"):
        model_dic["cbs"] = []
        model, history, y_test_pred, score = model_func(x_train, y_train, x_test, y_test, model_dic, ak_dic)
        clear_output()
    else:
        model, history, y_test_pred, score = model_func(x_train, y_train, x_test, y_test, model_dic)

    plot_loss_acc(history, model_name, results_dir)
    plot_confusion_matrix(y_test, y_test_pred, lbl_enc, model_name, results_dir)

    with open(f"{results_dir}/scores.csv", "a") as f:
        f.write(f"{model_name},{model_dic['normalizer']},{model_dic['PCA']},{(time.time()-st_time):.0f},{score[0]:.4f},{score[1]:.4f}\n")

    model_df = pd.DataFrame.from_dict(model_dic, orient="index").T[["batch_size", "epochs", "normalizer", "PCA"]]
    if ak_dic:
        ak_df = pd.DataFrame.from_dict(ak_dic, orient="index").T[["max_trials", "tuner"]]
        total_df = pd.merge(model_df, ak_df, left_index=True, right_index=True)
    else:
        total_df = model_df
    total_df.to_csv(join(results_dir, model_name, "params.csv"), index=False)

    def listify_dic(model_dic):
        if not isinstance(model_dic["normalizer"], list):
            model_dic["normalizer"] = [model_dic["normalizer"]]
        if not isinstance(model_dic["PCA"], list):
            model_dic["PCA"] = [model_dic["PCA"]]
        return model_dic
