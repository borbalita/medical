import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset
from torchmetrics import Accuracy, Precision, Recall

from pneumonia.classifier import PneumoniaClassifier
from pneumonia.evaluate import (
    display_metrics,
    evaluate_classifier,
    plot_confusion_matrices,
    plot_roc,
)


@pytest.fixture
def random_data():
    train_preds = torch.randn(5)
    val_preds = torch.randn(5)
    train_labels = torch.randint(0, 2, (5,))
    val_labels = torch.randint(0, 2, (5,))
    return train_preds, val_preds, train_labels, val_labels

def test_plot_confusion_matrices(random_data):
    train_preds, val_preds, train_labels, val_labels = random_data
    plot_confusion_matrices(
        train_preds,
        train_labels,
        val_preds,
        val_labels,
        thresh=0.5
    )

def test_display_metrics(random_data):
    train_preds, val_preds, train_labels, val_labels = random_data
    metrics = {
        "accuracy": Accuracy,
        "precision": Precision,
        "recall": Recall
    }
    display_metrics(
        metrics,
        train_preds,
        train_labels,
        val_preds,
        val_labels,
        thresh=0.5
    )

def test_plot_roc(random_data):
    train_preds, val_preds, train_labels, val_labels = random_data
    plot_roc(
        train_preds,
        val_preds,
        train_labels,
        val_labels
    )

def test_evaluate_classifier():
    model = PneumoniaClassifier()
    train_loader = torch.utils.data.DataLoader(
        TensorDataset(torch.randn(10, 1, 224, 224), torch.randint(0, 2, (10,))),
        batch_size=2
    )
    val_loader = torch.utils.data.DataLoader(
        TensorDataset(torch.randn(10, 1, 224, 224), torch.randint(0, 2, (10,))),
        batch_size=2
    )
    evaluate_classifier(model, train_loader, val_loader)
