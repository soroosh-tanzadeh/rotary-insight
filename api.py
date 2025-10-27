"""
API module for loading models from MLFlow and performing inference.
"""

import torch
import mlflow


# Example usage
if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")

    model = mlflow.pytorch.load_model("models:/transformer_encoder_cwru_2048/1")
    print(model)

    # Example 2: Load best model from an experiment
    # inference = ModelInference(experiment_name="my_experiment")

    # Example 3: Load model from specific run
    # inference = ModelInference(run_id="abc123")

    # Example 4: Perform inference
    # dummy_data = np.random.randn(1, 1, 2048)  # (batch_size, channels, signal_length)
    # predictions = inference.predict(dummy_data)
    # probabilities = inference.predict_proba(dummy_data)

    # Example 5: List runs from an experiment
    # runs_df = list_runs("my_experiment", max_results=10)
    # print(runs_df)
