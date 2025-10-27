"""
Example client for the Rotary Insight Inference API.

This script demonstrates how to use the API for bearing fault classification.
"""
import requests
import numpy as np
import json
from typing import List, Dict, Any


class RotaryInsightClient:
    """Client for interacting with the Rotary Insight API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API server
            api_key: API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health status of the API."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """List all available models."""
        response = requests.get(
            f"{self.base_url}/models",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def predict(
        self,
        data: np.ndarray,
        model_name: str,
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Perform inference on input data.
        
        Args:
            data: Input signal data with shape (batch_size, channels, signal_length)
            model_name: Name of the model to use
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Prediction results
        """
        # Convert numpy array to list for JSON serialization
        data_list = data.tolist()
        
        payload = {
            "data": data_list,
            "model_name": model_name,
            "return_probabilities": return_probabilities
        }
        
        response = requests.post(
            f"{self.base_url}/predict",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a specific model."""
        response = requests.post(
            f"{self.base_url}/models/{model_name}/load",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def unload_model(self, model_name: str) -> Dict[str, Any]:
        """Unload a specific model."""
        response = requests.delete(
            f"{self.base_url}/models/{model_name}/unload",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()


def main():
    """Example usage of the client."""
    
    # Configuration
    API_URL = "http://localhost:8000"
    API_KEY = "your-secret-key-1"  # Replace with your actual API key
    
    # Initialize client
    client = RotaryInsightClient(base_url=API_URL, api_key=API_KEY)
    
    print("=" * 60)
    print("Rotary Insight API Client Example")
    print("=" * 60)
    
    # 1. Health check (no authentication required)
    print("\n1. Health Check:")
    health = client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Models loaded: {health['models_loaded']}")
    print(f"   Available models: {', '.join(health['available_models'])}")
    
    # 2. List all models
    print("\n2. Available Models:")
    models_response = client.list_models()
    for name, info in models_response['models'].items():
        print(f"\n   Model: {name}")
        print(f"   - Window size: {info['window_size']}")
        print(f"   - Dataset: {info['dataset_name']}")
        print(f"   - Classes: {info['num_classes']}")
        print(f"   - Loaded: {info['loaded']}")
    
    # 3. Select a model and perform inference
    # Use the first available model
    model_name = list(models_response['models'].keys())[0]
    model_info = models_response['models'][model_name]
    window_size = model_info['window_size']
    
    print(f"\n3. Running Inference with '{model_name}':")
    
    # Generate random sample data (in practice, this would be real sensor data)
    # Shape: (batch_size=2, channels=1, signal_length=window_size)
    data = np.random.randn(2, 1, window_size).astype(np.float32)
    print(f"   Input shape: {data.shape}")
    
    # Make prediction
    result = client.predict(
        data=data,
        model_name=model_name,
        return_probabilities=True
    )
    
    print(f"   Processing time: {result['processing_time_ms']:.2f} ms")
    print(f"\n   Predictions:")
    for i, pred in enumerate(result['predictions']):
        print(f"\n   Sample {i + 1}:")
        print(f"   - Predicted class: {pred['class_name']} (index: {pred['predicted_class']})")
        print(f"   - Confidence: {pred['confidence']:.2%}")
        
        if pred['probabilities']:
            print(f"   - Top 3 classes:")
            probs = np.array(pred['probabilities'])
            top3_indices = np.argsort(probs)[::-1][:3]
            for idx in top3_indices:
                class_name = model_info['class_names'][idx]
                print(f"     {class_name}: {probs[idx]:.2%}")
    
    # 4. Load/Unload models (optional)
    print(f"\n4. Model Management:")
    print(f"   Loading model '{model_name}'...")
    load_result = client.load_model(model_name)
    print(f"   {load_result['message']}")
    
    # Optionally unload the model
    # print(f"   Unloading model '{model_name}'...")
    # unload_result = client.unload_model(model_name)
    # print(f"   {unload_result['message']}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the API server.")
        print("Make sure the server is running at http://localhost:8000")
        print("Start it with: ./start_server.bash")
    except requests.exceptions.HTTPError as e:
        print(f"\nHTTP Error: {e}")
        if e.response.status_code in [401, 403]:
            print("Check your API key in the script!")
    except Exception as e:
        print(f"\nError: {e}")

