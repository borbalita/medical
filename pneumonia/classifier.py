"""
Module for the PneumoniaClassifier class and related functions.
"""

from typing import Any, Dict, Optional

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import cuda, sigmoid, tensor
from torch.nn import BCEWithLogitsLoss, Conv2d, Linear
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.models import resnet18


class PneumoniaClassifier(LightningModule):
    """
    Pneumonia Classifier model.

    Parameters
    ----------
    weight : float, optional
        Weight for the positive class in the loss function. Default is 1.
    metrics : dict, optional
        Dictionary of metrics to be used for evaluation. Default is None.

    Attributes
    ----------
    model : nn.Module
        ResNet18 model with modified layers for pneumonia classification.
    metrics : dict
        Dictionary of metrics used for evaluation.
    loss_fn : nn.Module
        Loss function for training the model.

    """

    def __init__(
            self, weight: float = 1, metrics: Optional[Dict[str, Any]] = None):
        super().__init__()

        self.model = resnet18(pretrained=True)
        # Freeze the weights
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.conv1 = Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                  bias=False)
        # Q: Should bias be False??? in course code its True
        self.model.fc = Linear(512, 1, bias=False)

        self.metrics = {
            "acc": Accuracy("binary")} if metrics is None else metrics
        self.loss_fn = BCEWithLogitsLoss(pos_weight=tensor(weight))

    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        return self.model(*args, **kwargs)

    def training_step(self, *args, **kwargs):
        """
        Training step of the model.

        Args:
            batch (tuple): Tuple containing input and target tensors.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss value.

        """
        img, label = kwargs["batch"]
        logit = self(img)[:, 0]
        loss = self.loss_fn(logit, label.float())

        self.log("train_loss", loss)
        for name, metric in self.metrics.items():
            prob = sigmoid(logit)
            self.log(f"train_{name}", metric(prob, label.int()), prog_bar=True)
        return loss

    def on_training_epoch_end(self):
        """
        Callback function called at the end of each training epoch.

        """
        for name, metric in self.metrics.items():
            self.log(f"batch_train_{name}", metric.compute())
            metric.reset()

    def validation_step(self, *args, **kwargs):
        """
        Validation step of the model.

        Args:
            batch (tuple): Tuple containing input and target tensors.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss value.

        """
        img, label = kwargs["batch"]
        logit = self(img)[:, 0]
        loss = self.loss_fn(logit, label.float())
        self.log("val_loss", loss)
        for name, metric in self.metrics.items():
            prob = sigmoid(logit)
            self.log(f"val_{name}", metric(prob, label.int()), prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        """
        Callback function called at the end of each validation epoch.

        """
        for name, metric in self.metrics.items():
            self.log(f"batch_val_{name}", metric.compute())
            metric.reset()

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: Optimizer object.

        """
        return Adam(self.model.parameters(), lr=1e-3)


def train_pneumonia_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    log_dir: str,
    checkpoint_dir: str,
    weight: float = 1,
    max_epochs: int = 10
) -> PneumoniaClassifier:
    """
    Train the pneumonia classifier model.

    Parameters
    ----------
    train_loader : DataLoader
        The data loader for training data.
    val_loader : DataLoader
        The data loader for validation data.
    log_dir : str
        The directory path where the logs will be saved.
    checkpoint_dir : str
        The directory path where the model checkpoints will be saved.
    weight : float, optional
        Weight for the positive class in the loss function. Default is 1.
    max_epochs : int, optional
        The maximum number of epochs for training. Default is 10.

    Returns
    -------
    PneumoniaClassifier
        The trained pneumonia classifier model.

    """
    model = PneumoniaClassifier(weight)
    logger = TensorBoardLogger(log_dir, name="./logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir, monitor="val_loss", save_top_k=1, mode="min"
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        gpus=1 if cuda.is_available() else 0,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)
    return model
