import matplotlib

matplotlib.use("Agg")

import torch
import matplotlib.pyplot as plt
from matplotlib.figure import FigureBase
import pandas as pd
from torch import nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


# metric: 'accuracy' or 'loss'
def display_history(
    history, metric, store=False, model_name="model_", store_path="./data/images/"
):
    df = pd.DataFrame(history["train_" + metric], columns=["train_" + metric])
    df["val_" + metric] = history["val_" + metric]

    plt.figure(figsize=(8, 5))
    plt.plot(df.index + 1, df["train_" + metric], label="train_" + metric)
    plt.plot(df.index + 1, df["val_" + metric], label="val_" + metric)

    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.title(f"Training and Validation {metric.capitalize()}")
    plt.legend()
    plt.grid(True)

    if store:
        plt.savefig(f"{store_path}{model_name}_{metric}.png")

    return


def save_history(history, metric, path):
    df = pd.DataFrame(history["train_" + metric], columns=["train_" + metric])
    df["val_" + metric] = history["val_" + metric]

    df.to_csv(path)

    return


def load_history_acc(history_path):
    history = pd.read_csv(history_path)
    return history


def load_history_loss(history_path):
    history = pd.read_csv(history_path)
    return history


def display_confusion_matrix_by_pred(
    y_pred,
    y_true,
    labels,
    decimas=4,
    normalize=False,
    save=False,
    model_name="model",
    store_path="./data/images/",
) -> FigureBase:
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

    if normalize:
        cm = np.round(
            cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=decimas
        )

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()

    plt = disp.figure_.get_figure()
    plt.autofmt_xdate()
    plt.set_figwidth(12)
    plt.set_figheight(10)

    if save:
        plt.savefig(f"{store_path}{model_name}_confusion_matrix.png", pad_inches=0)

    return plt


def display_confusion_matrix(
    model: nn.Module,
    x_test,
    y_test,
    labels,
    device,
    decimas=4,
    normalize=False,
    save=False,
    model_name="model",
    store_path="./data/images/",
):
    x_test = torch.from_numpy(x_test).to(device).float()
    y_pred = model(x_test).to("cpu")
    y_pred = y_pred.argmax(dim=1)

    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))

    if normalize:
        cm = np.round(
            cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=decimas
        )

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()

    plt = disp.figure_.get_figure()
    plt.autofmt_xdate()
    plt.set_figwidth(8)
    plt.set_figheight(6)

    if save:
        plt.savefig(f"{store_path}{model_name}_confusion_matrix.png", pad_inches=0)

    return
