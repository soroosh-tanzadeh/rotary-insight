from torch.optim import Adam
from models import TransformerEncoderClassifier
from utils.callbacks import create_expo_lr_cb


def linear_embedding_rule(value, hyperparameters):
    if value == "linear":
        return hyperparameters["embedding_hidden_dim"] == 0
    return hyperparameters["embedding_hidden_dim"] == 0


def get_model_config(num_classes, window_size) -> dict:
    return {
        "transformer_encoder_classifier": get_transformer_encoder_classifier_model_config(
            num_classes, window_size
        ),
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
            "with_class_token": True,
            "num_heads": 8,
            "num_encoders": 4,
            "embedding_dim": 64,
            "feedforward_dim": 32,
            "number_of_channels": 1,
            "num_classes": num_classes,
        },
        "epochs": 32,
    }
