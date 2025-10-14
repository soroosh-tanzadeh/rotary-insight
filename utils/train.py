import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import mlflow


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def train(
    model: nn.Module,
    epochs: int,
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    callbacks=[],
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        train_iter = tqdm(
            dataloader_train, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False
        )
        train_loss = 0.0
        val_loss = 0.0

        train_acc = 0.0
        val_acc = 0.0
        total_train = 0
        total_val = 0

        model.train()
        correct = 0
        command = False
        for x, labels in train_iter:
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_iter.set_postfix(loss=loss.item())
        train_acc = correct / total_train

        # Log to MLFlow (if active run exists)
        if mlflow.active_run():
            mlflow.log_metric(
                "epoch_train_loss", train_loss / len(dataloader_train), step=epoch
            )
            mlflow.log_metric("epoch_train_acc", train_acc, step=epoch)

        history["train_loss"].append(train_loss / len(dataloader_train))
        history["train_acc"].append(train_acc)

        model.eval()
        correct = 0
        with torch.no_grad():
            val_iter = tqdm(
                dataloader_val, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False
            )
            for x, labels in val_iter:
                x, labels = x.to(device), labels.to(device)
                outputs = model(x)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_iter.set_postfix(loss=loss.item())

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_acc = correct / total_val
            history["val_loss"].append(val_loss / len(dataloader_val))
            history["val_acc"].append(val_acc)

            # Log to MLFlow (if active run exists)
            if mlflow.active_run():
                mlflow.log_metric(
                    "epoch_val_loss", val_loss / len(dataloader_val), step=epoch
                )
                mlflow.log_metric("epoch_val_acc", val_acc, step=epoch)

        epoch = epoch + 1
        print(
            f"Epoch {epoch}/{epochs} [Train] loss: {train_loss/len(dataloader_train):.4f}, acc: {train_acc:.4f}"
        )
        print(
            f"Epoch {epoch}/{epochs} [Val] loss: {val_loss/len(dataloader_val):.4f}, acc: {val_acc:.4f}"
        )

        for callback in callbacks:
            command = callback(epoch, history)
            if command == "early_stop":
                return history

    return history


def load_model(model, path, deivce=None):
    model.load_state_dict(torch.load(path, map_location=deivce))
    return model
