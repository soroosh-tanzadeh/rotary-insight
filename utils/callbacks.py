# callbacks.py
import torch
from torch.optim import lr_scheduler


class ModelCheckpoint:
    def __init__(self, model, filepath):
        self.model = model
        self.filepath = filepath
        self.best_acc = 0.0

    def __call__(self, epoch, hist):
        if "val_acc" in hist and hist["val_acc"]:
            current_acc = hist["val_acc"][-1]
            if current_acc > self.best_acc:
                self.best_acc = current_acc
                torch.save(self.model.state_dict(), self.filepath)
                print(
                    "Saving model with acc {:.4f} at epoch {}".format(
                        self.best_acc, epoch
                    )
                )


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = None

    def __call__(self, epoch, hist):
        if "val_acc" in hist and hist["val_acc"]:
            current_acc = hist["val_acc"][-1]
            if self.best_acc is None:
                self.best_acc = current_acc
                print(f"Best acc: {self.best_acc}")
            elif current_acc >= self.best_acc - self.min_delta:
                self.counter = 0
                self.best_acc = current_acc
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    return "early_stop"


def create_early_stop_cb(patience=5, min_delta=0):
    return EarlyStopping(patience=patience, min_delta=min_delta)


def create_expo_lr_cb(
    opt: torch.optim.Optimizer,
    mode="min",
    min_lr=1.0e-8,
    gamma=0.9,
    warmup=10,
):
    scheduler = lr_scheduler.ExponentialLR(opt, gamma=gamma)

    def cb(epoch, hist):
        if epoch >= warmup:
            scheduler.step()
            print(f"Learning rate: {scheduler.get_last_lr()[0]}")

    return cb


def create_reduce_lr_cb(
    opt: torch.optim.Optimizer,
    mode="min",
    patience=5,
    factor=0.1,
    min_lr=1.0e-8,
    warmup=5,
):
    scheduler = lr_scheduler.ReduceLROnPlateau(
        opt, mode=mode, patience=patience, factor=factor, min_lr=min_lr
    )

    def cb(epoch, hist):
        if epoch >= warmup:
            scheduler.step(metrics=hist["val_loss"][-1])
            print(f"Learning rate: {opt.param_groups[0]['lr']}")

    return cb
