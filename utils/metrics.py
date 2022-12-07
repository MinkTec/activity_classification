import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculate_metrics(y_true, y_pred, duration=0, output_path=".") -> pd.DataFrame:
    """Calculates several metrics for the model predictions and saves them as a DataFrame at 'output_path'.

    Args:
        y_true
        y_pred
        duration (int, optional): How long the tuning/fit took. Defaults to 0.
        output_path (str, optional): Output path for the DataFrame. Defaults to ".".

    Returns:
        pd.DataFrame: Holds the model metrics.
    """
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)

    res = pd.DataFrame(
        data=np.zeros((1, 5), dtype=np.float), index=[0], columns=["accuracy", "f1_score", "precision", "recall", "duration"]
    )
    res["accuracy"] = accuracy_score(y_true, y_pred)
    res["f1_score"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    res["precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    res["recall"] = recall_score(y_true, y_pred, average="macro")
    res["duration"] = int(duration)
    res.to_csv(output_path)

    return res
