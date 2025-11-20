import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from server.app import app
from server.inference import ModelManager, get_model_manager

# Mock data
MOCK_CONFIGS = {
    "model_1024": {
        "type": "pytorch",
        "window_size": 1024,
        "dataset_name": "CWRU",
        "task": "classification",
        "num_classes": 10
    },
    "model_2048": {
        "type": "pytorch",
        "window_size": 2048,
        "dataset_name": "CWRU",
        "task": "classification",
        "num_classes": 10
    },
    "model_512": {
        "type": "pytorch",
        "window_size": 512,
        "dataset_name": "PU",
        "task": "classification",
        "num_classes": 3
    }
}

@pytest.fixture
def mock_model_manager():
    with patch("server.routes.models.get_model_manager") as mock:
        manager = MagicMock(spec=ModelManager)
        manager.configs = MOCK_CONFIGS
        manager.models = {}
        
        # Mock list_models implementation
        def list_models_side_effect(window_size=None):
            models_info = {}
            for name, config in MOCK_CONFIGS.items():
                if window_size is not None and config["window_size"] != window_size:
                    continue
                models_info[name] = {
                    "name": name,
                    "type": config["type"],
                    "window_size": config["window_size"],
                    "dataset_name": config["dataset_name"],
                    "task": config["task"],
                    "num_classes": config["num_classes"],
                    "class_names": [],
                    "loaded": False
                }
            return models_info
            
        manager.list_models.side_effect = list_models_side_effect
        
        # Mock get_available_window_sizes implementation
        def get_window_sizes_side_effect():
            return sorted(list(set(c["window_size"] for c in MOCK_CONFIGS.values())))
            
        manager.get_available_window_sizes.side_effect = get_window_sizes_side_effect
        
        mock.return_value = manager
        yield manager

@pytest.fixture
def mock_auth():
    # Get the actual auth handler instance that was created at import time
    from server.auth import get_auth_handler
    auth_handler = get_auth_handler()
    
    # Override the dependency
    app.dependency_overrides[auth_handler.verify_api_key] = lambda: True
    yield
    # Clean up
    app.dependency_overrides = {}

def test_get_window_sizes(mock_model_manager, mock_auth):
    client = TestClient(app)
    response = client.get("/models/window-sizes")
    assert response.status_code == 200
    data = response.json()
    assert "window_sizes" in data
    assert data["window_sizes"] == [512, 1024, 2048]

def test_list_models_no_filter(mock_model_manager, mock_auth):
    client = TestClient(app)
    response = client.get("/models/")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 3
    assert "model_1024" in data["models"]
    assert "model_2048" in data["models"]
    assert "model_512" in data["models"]

def test_list_models_with_filter(mock_model_manager, mock_auth):
    client = TestClient(app)
    response = client.get("/models/?window_size=1024")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 1
    assert "model_1024" in data["models"]
    assert "model_2048" not in data["models"]
    
    response = client.get("/models/?window_size=512")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 1
    assert "model_512" in data["models"]

def test_list_models_empty_filter(mock_model_manager, mock_auth):
    client = TestClient(app)
    response = client.get("/models/?window_size=9999")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 0
    assert data["models"] == {}
