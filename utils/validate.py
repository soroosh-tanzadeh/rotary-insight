import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score
import torch.nn.functional as F


def validate_model(
    model: nn.Module,
    test_loader: DataLoader,
    model_path: str,
    device,
    labels: list[str],
):
    with torch.no_grad():
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        y_true = []
        y_pred_probs = []
        cross_entropy = nn.CrossEntropyLoss()
        validation_loss = 0
        for x, y in test_loader:
            y_true.append(y)
            x, y = x.to(device).float(), y.to(device).long()
            pred = model(x)
            y_pred_probs.append(pred.to("cpu"))
            loss = cross_entropy(pred, y)
            validation_loss += loss.item()

        y_pred_probs = torch.cat(y_pred_probs)
        y_pred = y_pred_probs.argmax(axis=1).numpy()
        y_true = torch.cat(y_true).numpy()

        report = classification_report(
            y_true, y_pred, target_names=labels, output_dict=True
        )

        y_pred_probs = F.softmax(y_pred_probs, dim=1).numpy()

        auroc = roc_auc_score(y_true, y_pred_probs, multi_class="ovr", average=None)

        metrics = {
            "classification_report": report,
            "auroc": auroc.tolist(),
        }

        return (
            y_pred,
            y_true,
            validation_loss / len(test_loader),
            metrics,
        )
