#!/usr/bin/env python3
"""
Verification script to check if the API setup is complete and correct.
Run this before starting the server to verify everything is configured properly.
"""
import os
import sys
import json
from pathlib import Path

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(text):
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}{text.center(60)}{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}\n")


def check_mark(passed):
    return f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"


def check_env_file():
    """Check if .env file exists and has required variables."""
    print(f"{BLUE}1. Checking .env file...{RESET}")

    env_path = Path(".env")
    if not env_path.exists():
        print(f"  {check_mark(False)} .env file not found")
        print(f"  {YELLOW}  Create it with: echo 'API_KEYS=your-key' > .env{RESET}")
        return False

    print(f"  {check_mark(True)} .env file exists")

    # Try to load and check for API_KEYS
    try:
        with open(env_path) as f:
            content = f.read()
            if "API_KEYS=" in content:
                print(f"  {check_mark(True)} API_KEYS variable found")
                # Get the value
                for line in content.split("\n"):
                    if line.startswith("API_KEYS="):
                        keys = line.split("=", 1)[1].strip()
                        num_keys = len([k for k in keys.split(",") if k.strip()])
                        print(f"  {YELLOW}  → {num_keys} API key(s) configured{RESET}")
                return True
            else:
                print(f"  {check_mark(False)} API_KEYS variable not found in .env")
                print(f"  {YELLOW}  Add: API_KEYS=your-secret-key{RESET}")
                return False
    except Exception as e:
        print(f"  {check_mark(False)} Error reading .env: {e}")
        return False


def check_model_config():
    """Check if model configuration file exists and is valid."""
    print(f"\n{BLUE}2. Checking model configuration...{RESET}")

    config_path = Path("model_serve_config.json")
    if not config_path.exists():
        print(f"  {check_mark(False)} model_serve_config.json not found")
        print(
            f"  {YELLOW}  Create it with: cp model_serve_config.example.json model_serve_config.json{RESET}"
        )
        return False

    print(f"  {check_mark(True)} model_serve_config.json exists")

    # Try to parse JSON
    try:
        with open(config_path) as f:
            config = json.load(f)

        if "models" not in config:
            print(f"  {check_mark(False)} 'models' key not found in config")
            return False

        num_models = len(config["models"])
        print(f"  {check_mark(True)} Valid JSON with {num_models} model(s)")

        # List models
        for name, info in config["models"].items():
            print(
                f"  {YELLOW}  → {name} (window_size: {info.get('window_size', 'N/A')}){RESET}"
            )

        return True
    except json.JSONDecodeError as e:
        print(f"  {check_mark(False)} Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"  {check_mark(False)} Error reading config: {e}")
        return False


def check_dependencies():
    """Check if required Python packages are installed."""
    print(f"\n{BLUE}3. Checking Python dependencies...{RESET}")

    required_packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic"),
        ("dotenv", "python-dotenv"),
        ("mlflow", "MLflow"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
    ]

    all_installed = True
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"  {check_mark(True)} {package_name}")
        except ImportError:
            print(
                f"  {check_mark(False)} {package_name} - install with: pip install {package_name}"
            )
            all_installed = False

    return all_installed


def check_server_files():
    """Check if server files exist."""
    print(f"\n{BLUE}4. Checking server files...{RESET}")

    server_files = [
        "server/__init__.py",
        "server/main.py",
        "server/auth.py",
        "server/models.py",
        "server/inference.py",
        "server/client_example.py",
    ]

    all_exist = True
    for file_path in server_files:
        path = Path(file_path)
        exists = path.exists()
        print(f"  {check_mark(exists)} {file_path}")
        if not exists:
            all_exist = False

    return all_exist


def check_mlflow_connection():
    """Check if MLflow server is accessible."""
    print(f"\n{BLUE}5. Checking MLflow connection...{RESET}")

    try:
        import requests

        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

        print(f"  {YELLOW}  → Testing connection to {mlflow_uri}{RESET}")

        response = requests.get(f"{mlflow_uri}/health", timeout=2)
        if response.status_code == 200:
            print(f"  {check_mark(True)} MLflow server is running")
            return True
        else:
            print(
                f"  {check_mark(False)} MLflow returned status {response.status_code}"
            )
            print(f"  {YELLOW}  Start it with: ./start_mlflow.bash{RESET}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  {check_mark(False)} MLflow server not accessible")
        print(f"  {YELLOW}  Start it with: ./start_mlflow.bash{RESET}")
        return False
    except ImportError:
        print(f"  {check_mark(False)} 'requests' package not installed")
        return False
    except Exception as e:
        print(f"  {check_mark(False)} Error: {e}")
        return False


def check_import_server():
    """Try to import the server module."""
    print(f"\n{BLUE}6. Testing server imports...{RESET}")

    try:
        # Add current directory to path
        sys.path.insert(0, str(Path.cwd()))

        # Try importing server modules
        from server import main

        print(f"  {check_mark(True)} server.main")

        from server import auth

        print(f"  {check_mark(True)} server.auth")

        from server import models

        print(f"  {check_mark(True)} server.models")

        from server import inference

        print(f"  {check_mark(True)} server.inference")

        return True
    except Exception as e:
        print(f"  {check_mark(False)} Import error: {e}")
        return False


def main():
    """Run all verification checks."""
    print_header("API Setup Verification")

    checks = [
        ("Environment file", check_env_file),
        ("Model configuration", check_model_config),
        ("Python dependencies", check_dependencies),
        ("Server files", check_server_files),
        ("MLflow connection", check_mlflow_connection),
        ("Server imports", check_import_server),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"{RED}Error in {name}: {e}{RESET}")
            results[name] = False

    # Summary
    print_header("Summary")

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  [{status}] {name}")

    print(f"\n{BLUE}Results: {passed}/{total} checks passed{RESET}")

    if passed == total:
        print(f"\n{GREEN}✓ All checks passed! Ready to start the server.{RESET}")
        print(f"{YELLOW}Run: ./start_server.bash{RESET}")
        return 0
    else:
        print(f"\n{RED}✗ Some checks failed. Please fix the issues above.{RESET}")
        print(f"{YELLOW}See API_QUICKSTART.md for setup instructions.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
