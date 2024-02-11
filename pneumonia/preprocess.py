"""Utils for pneumonia classification."""

import os
import random
from typing import Any, NamedTuple, Tuple

import cv2
import numpy as np
import pandas as pd
import pydicom as dcm
from tqdm.notebook import tqdm


class StandardParams(NamedTuple):
    """
    Parameters for standardization.

    Parameters
    ----------
    mean : float
        The mean value used for standardization.
    std : float
        The standard deviation value used for standardization.
    """

    mean: float
    std: float


def _create_output_dirs(out_dir: str, labels: pd.DataFrame) -> None:
    targets = labels["Target"].unique()
    for target in targets:
        os.makedirs(f"{out_dir}/train/{target}", exist_ok=True)
        os.makedirs(f"{out_dir}/val/{target}", exist_ok=True)

def preprocess_array(img: np.ndarray, shape: Tuple[int, int], pixel_format: Any):
    """
    Preprocesses an image array by normalizing pixel values and resizing it.

    Parameters:
        img (np.ndarray): The input image array.
        shape (Tuple[int, int]): The desired shape of the output image.
        pixel_format (Any): The desired pixel format of the output image.

    Returns:
        np.ndarray: The preprocessed image array.

    """
    img = img / 255.0
    return cv2.resize(img, shape).astype(pixel_format)


def preprocess(
    raw_dir: str,
    label_path: str,
    out_dir: str,
    shape: Tuple[int, int],
    val_ratio: float = 0.2,
    pixel_format: Any = np.float16,
) -> None:
    """
    Preprocesses raw DICOM images by resizing and normalizing them, and saves
    them as numpy arrays.

    Parameters
    ----------
    raw_dir : str
        The directory path containing the raw DICOM images.
    label_path : str
        The file path of the CSV file containing the labels for the images.
    out_dir : str
        The directory path where the preprocessed images will be saved.
    shape : tuple
        The desired shape (height, width) of the preprocessed images.
    labels : list, optional
        The list of labels for the images. If not provided, random labels
        will be assigned.
    val_ratio : float, optional
        The ratio of images to be used for validation. Default is 0.2.
    pixel_format : numpy.dtype, optional
        The desired pixel format for the preprocessed images. Default is
        np.float16.

    Returns
    -------
    None
        This function does not return any value.
    """
    labels = pd.read_csv(label_path)
    _create_output_dirs(out_dir, labels)

    for patient_id in tqdm(labels.patientId.unique()):
        file_path = os.path.join(raw_dir, patient_id)
        file_path = file_path + ".dcm"
        if not os.path.exists(file_path):
            continue

        img = dcm.read_file(file_path).pixel_array
        img = preprocess_array(img, shape, pixel_format)

        label = labels[labels["patientId"] == patient_id]["Target"].iloc[0]
        train_or_val = "val" if random.random() < val_ratio else "train"
        save_path = f"{out_dir}/{train_or_val}/{label}/{patient_id}"
        np.save(save_path, img)


def compute_standard_params(preproc_dir: str, shape: Tuple[int, int]
                            ) -> StandardParams:
    """
    Compute the standard parameters (pixel mean and pixel standard deviation
    for a set of images.

    Parameters
    ----------
    preproc_dir : str, optional
        Directory path where the preprocessed images are stored. Default is
        'preprocessed'.
    shape : tuple, optional
        Expected shape of the images. Default is SHAPE.

    Returns
    -------
    StandardParams
        An instance of the StandardParams class containing the computed pixel
        mean and pixel standard deviation.

    Raises
    ------
    ValueError
        If the shape of any image in the directory does not match the expected
        shape.
    """
    pixel_sum = 0
    pixel_sqrd_sum = 0
    n_img = 0

    train_dir = os.path.join(preproc_dir, "train")
    for label in os.listdir(train_dir):
        print("Processing label:", label)
        label_dir = os.path.join(train_dir, label)
        for patient_id in tqdm(os.listdir(label_dir)):
            patient_dir = os.path.join(label_dir, patient_id)
            img = np.load(patient_dir)
            if img.shape != shape:
                raise ValueError(
                    f"Expected image shape {shape}, got " f"{img.shape}")

            pixel_sum += img.sum()
            pixel_sqrd_sum += (img**2).sum()
            n_img += 1

    if n_img == 0:
        raise ValueError("No images found in the directory")

    pixel_mean = pixel_sum / (n_img * shape[0] * shape[1])
    pixel_std = np.sqrt(
        pixel_sqrd_sum / (n_img * shape[0] * shape[1]) - pixel_mean**2)
    return StandardParams(pixel_mean, pixel_std)
