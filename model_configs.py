from torch.optim import Adam
from models import *
from utils.callbacks import create_expo_lr_cb


def get_model_config(num_classes, window_size) -> dict:
    return {
        "resnet_classifier": get_resnet_classifier_model_config(
            num_classes, window_size
        ),
        "transformer_encoder_classifier": get_transformer_encoder_classifier_model_config(
            num_classes, window_size
        ),
        "cnn_bilstm":get_CNN_BiLSTM_classifier_model_config(num_classes, window_size)
    }


def get_resnet_classifier_model_config(num_classes, window_size) -> dict:
    return {
        "model": ResNetClassfier,
        "optimizer": lambda m: Adam(m.parameters(), lr=1e-3),
        "callbacks": lambda opt: [
            create_expo_lr_cb(opt, gamma=0.9),
        ],
        "hyperparameters": {
            "num_classes": num_classes,
        },
        "epochs": 32,
    }


def get_transformer_encoder_classifier_model_config(num_classes, window_size) -> dict:
    return {
        "model": TransformerEncoderClassifier,
        "optimizer": lambda m: Adam(m.parameters(), lr=1e-3),
        "callbacks": lambda opt: [
            create_expo_lr_cb(opt, gamma=0.9),
        ],
        "hyperparameters": {
            "input_dim": window_size,
            "patch_size": 64,
            "positional_encoding": False,
            "dropout_rate": 0.2,
            "with_class_token": True,
            "num_heads": 16,
            "num_encoders": 8,
            "embedding_dim": 64,
            "feedforward_dim": 32,
            "number_of_channels": 1,
            "num_classes": num_classes,
        },
        "epochs": 32,
    }

def get_CNN_BiLSTM_classifier_model_config(num_classes, window_size) -> dict:
    return {
        "model": CNN_BiLSTM,
        "optimizer": lambda m: Adam(m.parameters(), lr=1e-3),
        "callbacks": lambda opt: [
            create_expo_lr_cb(opt, gamma=0.9),
        ],
        "hyperparameters": {
            "num_classes": num_classes,
        },
        "epochs": 32,
    }
def get_DenseNet1D_classifier_model_config(num_classes, window_size) -> dict:
    return {
        "model": DenseNet1D,
        "optimizer": lambda m: Adam(m.parameters(), lr=1e-3),
        "callbacks": lambda opt: [
            create_expo_lr_cb(opt, gamma=0.9),
        ],
        "hyperparameters": {
            "num_classes": num_classes,
        },
        "epochs": 32,
    }