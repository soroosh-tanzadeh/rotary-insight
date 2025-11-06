# Test Quick Reference Card

## Installation

```bash
pip install -r requirements-dev.txt
# OR just pytest
pip install pytest pytest-cov
```

## Running Tests

### With Coverage

```bash
python run_tests.py --coverage
```

## Test Structure

```
tests/
├── test_pu_dataset.py     # 30+ comprehensive tests
│   ├── TestPUDataset      # Core functionality (25+ tests)
│   ├── TestSlicerFunction # Utility functions (6+ tests)
│   └── TestPUDatasetIntegration  # Integration (3+ tests)
└── test_max_per_class.py  # Legacy demo
```

## Common Tasks

### Run All Tests

```bash
pytest tests/
```

### Run Fast Tests Only

```bash
pytest tests/ -m "not slow"
```

### Run with Coverage

```bash
pytest tests/ --cov=datasets --cov-report=html
open htmlcov/index.html
```

### Run Specific Test Class

```bash
pytest tests/test_pu_dataset.py::TestPUDataset -v
```

### Debug Test Failures

```bash
pytest tests/ -v --tb=long  # Detailed traceback
pytest tests/ -x            # Stop on first failure
pytest tests/ --pdb         # Drop into debugger on failure
```

## Test Markers

```python
@pytest.mark.slow          # Long-running tests
@pytest.mark.gpu           # Requires CUDA
@pytest.mark.integration   # Integration tests
@pytest.mark.unit          # Unit tests
```

Filter by markers:

```bash
pytest tests/ -m "gpu"           # Only GPU tests
pytest tests/ -m "not slow"      # Skip slow tests
pytest tests/ -m "unit and not slow"  # Combine
```

## Writing New Tests

### Basic Template

```python
class TestMyFeature:
    @pytest.fixture
    def dataset(self):
        return PUDataset(
            rdir="./data/dataset/PU",
            max_per_class=1000,
            seed=42,
        )

    def test_something(self, dataset):
        assert dataset.something() == expected
```

### With Marker

```python
@pytest.mark.slow
def test_large_operation():
    # Long-running test
    pass
```

## Troubleshooting

| Issue             | Solution                         |
| ----------------- | -------------------------------- |
| Module not found  | Run from project root            |
| Dataset not found | Check `./data/dataset/PU` exists |
| Tests slow        | Use `max_per_class=500` in tests |
| CUDA errors       | Skip with `-m "not gpu"`         |

## Coverage Goals

- Target: >80% code coverage
- Check: `pytest --cov=datasets --cov-report=term`
- View HTML: `pytest --cov=datasets --cov-report=html`

## Key Files

| File                   | Purpose          |
| ---------------------- | ---------------- |
| `run_tests.py`         | Advanced runner  |
| `pytest.ini`           | Pytest config    |
| `requirements-dev.txt` | Dev dependencies |
| `TESTING.md`           | Full guide       |

## Quick Checks

Before committing

```bash
# Run tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=datasets

# Run all including slow
pytest tests/
```

## More Help

- Full guide: `TESTING.md`
- Test docs: `tests/README.md`
- Pytest docs: https://docs.pytest.org/
