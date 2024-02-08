"""
Module for evaluating a classifier model using various metrics and thresholds.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.nn.functional import sigmoid
from torchmetrics import AUROC, ROC, Accuracy, ConfusionMatrix, Precision, Recall
from tqdm import tqdm


def _preds_n_labels(model, loader, device=None):
    """
    Get the predicted probabilities and true labels from a model and data
    loader.

    Parameters
    ----------
    model : torch.nn.Module
        The classifier model.
    loader : torch.utils.data.DataLoader
        The data loader.
    device : torch.device, optional
        The device to use for computation. Defaults to None.

    Returns
    -------
    torch.Tensor
        The predicted probabilities.
    torch.Tensor
        The true labels.
    """
    preds = []
    labels = []
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for batch in tqdm(loader):
            img, label = batch
            logit = model(img.to(device))[:, 0]
            prob = sigmoid(logit)
            preds.extend(prob.cpu().numpy())
            labels.extend(label.numpy())

    return torch.tensor(preds), torch.tensor(labels)


def display_metrics(metrics, train_preds, train_labels, val_preds, val_labels,
                    thresh):
    """
    Display the evaluation metrics for the training and validation datasets.

    Parameters
    ----------
    metrics : dict
        A dictionary containing the evaluation metrics.
    train_preds : array-like
        Predictions for the training dataset.
    train_labels : array-like
        True labels for the training dataset.
    val_preds : array-like
        Predictions for the validation dataset.
    val_labels : array-like
        True labels for the validation dataset.
    thresh : float
        Threshold value for binary classification.
    """
    metric_df = pd.DataFrame(index=metrics.keys(), columns=["train", "val"])
    for dataset in ["train", "val"]:
        preds = train_preds if dataset == "train" else val_preds
        labels = train_labels if dataset == "train" else val_labels
        for metric_name, metric in metrics.items():
            metric_df.loc[metric_name, dataset] = float(metric(
                "binary", threshold=thresh)(
                preds, labels
            ))

    print(metric_df)


def plot_confusion_matrices(train_preds, train_labels, val_preds, val_labels,
                            thresh):
    """
    Plot the confusion matrices for binary classification.

    Parameters
    ----------
    train_preds : array-like
        Predictions for the training dataset.
    train_labels : array-like
        True labels for the training dataset.
    val_preds : array-like
        Predictions for the validation dataset.
    val_labels : array-like
        True labels for the validation dataset.
    thresh : float
        Threshold value for binary classification.
    """
    train_cm = ConfusionMatrix("binary", threshold=thresh)(train_preds,
                                                           train_labels)
    val_cm = ConfusionMatrix("binary", threshold=thresh)(val_preds, val_labels)
    _, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.heatmap(train_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title(f"Training CM (threshold={thresh:.2f})")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(val_cm, annot=True, fmt="d", cmap="Blues", ax=axes[1])
    axes[1].set_title(f"Validation CM (threshold={thresh:.2f})")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.show()


def plot_roc(train_preds, val_preds, train_labels, val_labels,
             ax=None):
    """
    Plot the Receiver Operating Characteristic (ROC) curve for binary
    classification.

    Parameters
    ----------
    train_preds : array-like
        Predicted probabilities for the training set.
    val_preds : array-like
        Predicted probabilities for the validation set.
    train_labels : array-like
        True labels for the training set.
    val_labels : array-like
        True labels for the validation set.
    ax : matplotlib.axes.Axes, optional
        The specific axis to plot the ROC curve. Defaults to None.

    Returns
    -------
    matplotlib.axes.Axes
        The specific axis used to plot the ROC curve.
    """
    if ax is None:
        ax = plt.gca()

    train_fpr, train_tpr, _ = ROC("binary")(train_preds, train_labels)
    val_fpr, val_tpr, _ = ROC("binary")(val_preds, val_labels)
    train_auc = AUROC("binary")(train_preds, train_labels)
    val_auc = AUROC("binary")(val_preds, val_labels)

    ax.plot(train_fpr, train_tpr,
            label=f"Training AUC: {train_auc:.2f}")
    ax.plot(val_fpr, val_tpr,
            label=f"Validation AUC: {val_auc:.2f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend()
    plt.show()

    return ax


def show_random_images(val_data, val_preds, val_labels):
    """
    Display random images from the validation dataset along with their
    predictions and true labels.

    Parameters
    ----------
    val_data : numpy.ndarray
        Array of validation images.
    val_preds : numpy.ndarray
        Array of predicted values for the validation images.
    val_labels : numpy.ndarray
        Array of true labels for the validation images.
    """
    _, axes = plt.subplots(3, 3, figsize=(9, 9))
    random_indices = np.random.choice(len(val_data), size=9, replace=False)
    for ax, i in zip(axes.flatten(), random_indices):
        ax.imshow(val_data[i][0][0], cmap='bone')
        ax.set_title(f"Prediction: {int(val_preds[i] > 0.25)}, "
                     f"True Label: {val_labels[i]}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def evaluate_classifier(
    model,
    train_loader,
    val_loader,
    metrics=None,
    thresholds=None,
    cm=True,
    roc=True,
):
    """
    Evaluate a classifier model using various metrics and thresholds.

    Parameters
    ----------
    model : torch.nn.Module
        The classifier model to evaluate.
    train_loader : torch.utils.data.DataLoader
        The data loader for the training set.
    val_loader : torch.utils.data.DataLoader
        The data loader for the validation set.
    metrics : dict, optional
        Dictionary of metrics to compute. Defaults to None.
    thresholds : list, optional
        List of thresholds for binary classification. Defaults to None.
    cm : bool, optional
        Whether to plot confusion matrices. Defaults to True.
    roc : bool, optional
        Whether to plot ROC curves. Defaults to True.

    Returns
    -------
    tuple
        A tuple containing the predicted labels and true labels for the
        training set, and the predicted labels and true labels for the
        validation set.
    """
    if metrics is None:
        metrics = {
            "acc": Accuracy,
            "precision": Precision,
            "recall": Recall,
        }

    if thresholds is None:
        thresholds = [0.25, 0.5]

    print("Computing training predictions")
    train_preds, train_labels = _preds_n_labels(model, train_loader)
    print("Computing validation predictions")
    val_preds, val_labels = _preds_n_labels(model, val_loader)

    for thresh in thresholds:
        print(f"Running evaluation with threshold: {thresh:.2f}")

        if len(metrics) > 0:
            display_metrics(
                metrics, train_preds, train_labels, val_preds, val_labels,
                thresh
            )

        if cm:
            plot_confusion_matrices(
                train_preds, train_labels, val_preds, val_labels, thresh
            )

    if roc:
        plot_roc(train_preds, val_preds, train_labels, val_labels)

    return train_preds, train_labels, val_preds, val_labels
