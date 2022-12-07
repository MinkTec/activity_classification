import os
from os.path import join
import pathlib
from pathlib import Path
from zipfile import ZipFile
from typing import Union

import numpy as np
import tensorflow as tf


def save_model_to_lite(
    model_path: Union[str, Path],
    x_shape,
    y_shape,
    lbl_classes,
    n_keep: Union[int, None] = None,
    pca_components: Union[np.ndarray, None] = None,
    scaler=None,
) -> None:
    """Zip model as several different tflite models with extra configs for later use in App."""
    output_dir = os.path.dirname(model_path)
    zip_dir = join(output_dir, "tflite")
    if not os.path.exists(zip_dir):
        os.mkdir(zip_dir)
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    model_lite = converter.convert()

    # default tflite model
    write_to_zip_path(model_lite, zip_dir, "inception.tflite", "wb")

    # This enables quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    model_lite = converter.convert()
    write_to_zip_path(model_lite, zip_dir, "inception_quant.tflite", "wb")

    # Float16
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    model_lite = converter.convert()
    write_to_zip_path(model_lite, zip_dir, "inception_float16.tflite", "wb")

    # Integer
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    model_lite = converter.convert()
    write_to_zip_path(model_lite, zip_dir, "inception_integer.tflite", "wb")

    # write helper files into zip directory
    write_to_zip_path("\n".join(lbl_classes), zip_dir, "labels.txt")
    write_to_zip_path(f"sensor_keep,{n_keep}\n", zip_dir, "shapes.txt")
    write_to_zip_path(f"x_shape,{x_shape[1]},{x_shape[2]}\n", zip_dir, "shapes.txt", "a")
    write_to_zip_path(f"y_shape,{y_shape[1]}\n", zip_dir, "shapes.txt", "a")

    if pca_components is not None:
        np.savetxt(join(zip_dir, "pca_components.csv"), pca_components, delimiter=",")
    if scaler.__class__.__name__ == "Standardize":
        np.savetxt(join(zip_dir, "mean.csv"), scaler.mean.squeeze())
        np.savetxt(join(zip_dir, "std.csv"), scaler.std.squeeze())

    with ZipFile(join(zip_dir, "tflite.zip"), "w") as zip:
        for file in pathlib.Path(zip_dir).iterdir():
            if str(file).endswith("zip"):  # to prevent infinite loop
                continue
            zip.write(file, arcname=file.name)

    print("Model + utilities successfully saved!")


def write_to_zip_path(content: str, zip_dir: Union[str, Path], fn: Union[str, Path], write_mode="w"):
    if content:
        with open(join(zip_dir, fn), write_mode) as f:
            f.write(content)
