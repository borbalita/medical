import torch
from pytorch_lightning import Trainer
from torch.nn import BCEWithLogitsLoss, Conv2d, Linear
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18

from pneumonia.classifier import PneumoniaClassifier

# TODO: e2e test with quality check.

def test_classifier_initialization():
    model = PneumoniaClassifier()
    assert isinstance(model.model, type(resnet18(pretrained=True)))
    assert isinstance(model.model.conv1, Conv2d)
    assert isinstance(model.model.fc, Linear)
    assert isinstance(model.loss_fn, BCEWithLogitsLoss)
    assert "acc" in model.metrics

def test_classifier_forward_pass():
    model = PneumoniaClassifier()
    x = torch.randn(1, 1, 224, 224)
    output = model(x)
    assert output.size() == torch.Size([1, 1])

def test_classifier_training_step():
    model = PneumoniaClassifier()
    batch = (torch.randn(1, 1, 224, 224), torch.tensor([0]))
    batch_idx = 0
    output = model.training_step(batch, batch_idx)
    print(output)
    assert output.size() == torch.Size([])

def test_classifier_validation_step():
    model = PneumoniaClassifier()
    batch = (torch.randn(1, 1, 224, 224), torch.tensor([0]))
    batch_idx = 0
    output = model.validation_step(batch, batch_idx)
    assert output.size() == torch.Size([])

def test_classifier_training():
    model = PneumoniaClassifier()
    train_loader = torch.utils.data.DataLoader(
        TensorDataset(torch.randn(10, 1, 224, 224), torch.randint(0, 2, (10,))),
        batch_size=2
    )
    val_loader = torch.utils.data.DataLoader(
        TensorDataset(torch.randn(10, 1, 224, 224), torch.randint(0, 2, (10,))),
        batch_size=2
    )
    trainer = Trainer(max_epochs=2)
    trainer.fit(model, train_loader, val_loader)
    assert trainer.callback_metrics['val_loss'] is not None
    assert trainer.callback_metrics['val_acc'] is not None
    assert trainer.callback_metrics['train_loss'] is not None
    assert trainer.callback_metrics['train_acc'] is not None
