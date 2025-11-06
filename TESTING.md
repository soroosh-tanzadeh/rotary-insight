# Testing Guide for Rotary-Insight

This document provides comprehensive information about testing the rotary-insight project.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Test Structure](#test-structure)
3. [Running Tests](#running-tests)
4. [Test Coverage](#test-coverage)
5. [Writing Tests](#writing-tests)
6. [CI/CD Integration](#cicd-integration)

## Quick Start

### Installation

Install test dependencies:

```bash
# Install development dependencies (includes pytest)
pip install -r requirements-dev.txt

# Or install only pytest
pip install pytest pytest-cov
```

### Running Tests

```bash
# Quick test run (automatically uses pytest if available)
python test.py

# Comprehensive test run with options
python run_tests.py

# Direct pytest execution
pytest tests/ -v
```

## Test Structure

### Directory Layout

```
rotary-insight/
├── tests/                      # Test directory
│   ├── __init__.py            # Test package init
│   ├── test_pu_dataset.py     # PU dataset tests (30+ tests)
│   ├── test_max_per_class.py  # Legacy demo test
│   └── README.md              # Test documentation
├── test.py                     # Quick test runner
├── run_tests.py                # Comprehensive test runner
├── pytest.ini                  # Pytest configuration
├── requirements-dev.txt        # Development dependencies
└── TESTING.md                 # This file
```

### Test Files

- **`test_pu_dataset.py`** - Main test suite with 30+ comprehensive tests
- **`test_max_per_class.py`** - Legacy demonstration script
- **`test.py`** - Simple quick test runner
- **`run_tests.py`** - Advanced test runner with options

## Running Tests

### Method 1: Quick Test (test.py)

Simplest way to run tests:

```bash
python test.py
```

- Automatically uses pytest if installed
- Falls back to manual tests if pytest unavailable
- Good for quick validation

### Method 2: Comprehensive Runner (run_tests.py)

Full-featured test execution:

```bash
# Run all tests with pytest
python run_tests.py

# Run manual tests only
python run_tests.py --mode manual

# Run both pytest and manual
python run_tests.py --mode all

# Run specific test file
python run_tests.py --test test_pu_dataset.py

# Run with coverage reporting
python run_tests.py --coverage

# Filter tests by markers
python run_tests.py --markers "not slow"
```

### Method 3: Direct Pytest

Full control over pytest:

```bash
# Run all tests
pytest tests/

# Verbose output
pytest tests/ -v

# Run specific file
pytest tests/test_pu_dataset.py

# Run specific class
pytest tests/test_pu_dataset.py::TestPUDataset

# Run specific test
pytest tests/test_pu_dataset.py::TestPUDataset::test_initialization

# Show test durations
pytest tests/ --durations=10

# Run in parallel (requires pytest-xdist)
pytest tests/ -n auto

# Stop on first failure
pytest tests/ -x

# Run last failed tests
pytest tests/ --lf

# Skip slow tests
pytest tests/ -m "not slow"
```

### Method 4: Manual Execution

Run test files directly:

```bash
# Run comprehensive tests manually
python tests/test_pu_dataset.py

# Run legacy demo
python tests/test_max_per_class.py
```

## Test Coverage

### Current Coverage

The test suite includes **30+ tests** covering:

#### PUDataset Class Tests (25+ tests)

- ✅ Initialization & configuration
- ✅ Data loading & caching
- ✅ max_per_class functionality
- ✅ Reproducibility with seeds
- ✅ Different window sizes
- ✅ Device handling (CPU/CUDA)
- ✅ Data integrity checks
- ✅ BearingDataset interface compliance
- ✅ Cache file naming conventions
- ✅ Error handling

#### Utility Function Tests (6+ tests)

- ✅ Slicer function with various configurations
- ✅ Overlapping/non-overlapping windows
- ✅ Edge case handling
- ✅ DataFrame/array output modes

#### Integration Tests (3+ tests)

- ✅ PyTorch DataLoader compatibility
- ✅ Cache persistence
- ✅ Multiple dataset instances

### Viewing Coverage Reports

```bash
# Generate coverage report
pytest tests/ --cov=datasets --cov=models --cov=preprocessings --cov=utils

# Generate HTML coverage report
pytest tests/ --cov=datasets --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Coverage Goals

- **Target**: >80% code coverage
- **Current**: Comprehensive dataset coverage
- **Future**: Add model and preprocessing tests

## Writing Tests

### Test Structure

Follow pytest conventions:

```python
import pytest
from datasets.pu_dataset import PUDataset

class TestMyFeature:
    """Test suite for my feature."""

    @pytest.fixture
    def dataset(self):
        """Fixture providing a test dataset."""
        return PUDataset(
            rdir="./data/dataset/PU",
            window_size=2048,
            max_per_class=1000,
            seed=42,
            verbose=False,
        )

    def test_basic_functionality(self, dataset):
        """Test basic functionality."""
        # Arrange
        expected = ...

        # Act
        result = dataset.some_method()

        # Assert
        assert result == expected

    def test_edge_case(self):
        """Test edge case."""
        with pytest.raises(ValueError):
            # Code that should raise ValueError
            pass
```

### Best Practices

1. **Use Descriptive Names**

   ```python
   # Good
   def test_max_per_class_respects_limit():
       pass

   # Bad
   def test_1():
       pass
   ```

2. **Use Fixtures for Setup**

   ```python
   @pytest.fixture
   def dataset():
       return PUDataset(...)
   ```

3. **Test One Thing Per Test**

   ```python
   # Good - tests one aspect
   def test_dataset_length():
       assert len(dataset) > 0

   # Bad - tests multiple things
   def test_everything():
       assert len(dataset) > 0
       assert dataset.labels() == [...]
       assert dataset.X.shape[0] > 0
   ```

4. **Use Markers for Categorization**

   ```python
   @pytest.mark.slow
   def test_full_dataset_load():
       pass

   @pytest.mark.gpu
   def test_cuda_operations():
       pass
   ```

5. **Add Docstrings**
   ```python
   def test_reproducibility():
       """Test that same seed produces identical results."""
       pass
   ```

### Common Assertions

```python
# Equality
assert value == expected

# Approximate equality (floats)
assert abs(value - expected) < 1e-6
assert pytest.approx(value) == expected

# Tensor equality
assert torch.allclose(tensor1, tensor2)
assert torch.all(tensor1 == tensor2)

# Shape checks
assert tensor.shape == (10, 2048)

# Type checks
assert isinstance(obj, MyClass)

# Exceptions
with pytest.raises(ValueError):
    function_that_raises()

# Warnings
with pytest.warns(UserWarning):
    function_that_warns()
```

## Test Markers

Available markers (defined in `pytest.ini`):

```python
@pytest.mark.slow          # Long-running tests
@pytest.mark.integration   # Integration tests
@pytest.mark.unit         # Unit tests
@pytest.mark.dataset      # Dataset-related tests
@pytest.mark.gpu          # GPU-required tests
```

Usage:

```bash
# Run only fast tests
pytest tests/ -m "not slow"

# Run only GPU tests
pytest tests/ -m gpu

# Run integration tests
pytest tests/ -m integration

# Combine markers
pytest tests/ -m "not slow and not gpu"
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run tests
        run: |
          pytest tests/ -v --cov=datasets --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v2
        with:
          file: ./coverage.xml
```

### GitLab CI Example

```yaml
test:
  image: python:3.9
  script:
    - pip install -r requirements.txt
    - pip install -r requirements-dev.txt
    - pytest tests/ -v --cov=datasets
  coverage: '/TOTAL.*\s+(\d+%)$/'
```

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'datasets'`

**Solution**: Run tests from project root:

```bash
cd /path/to/rotary-insight
python test.py
```

---

**Issue**: Dataset not found errors

**Solution**: Ensure data exists:

```bash
ls -la data/dataset/PU/
```

---

**Issue**: Tests are very slow

**Solution**: Use smaller datasets in tests:

```python
dataset = PUDataset(
    rdir="./data/dataset/PU",
    max_per_class=500,  # Small limit for tests
    verbose=False,
)
```

---

**Issue**: CUDA tests fail

**Solution**: Skip GPU tests if CUDA unavailable:

```bash
pytest tests/ -m "not gpu"
```

---

**Issue**: Cache conflicts

**Solution**: Use force_reload or different max_per_class:

```python
dataset = PUDataset(..., force_reload=True)
```

## Performance Tips

1. **Use Small Datasets in Tests**

   ```python
   # Fast tests
   dataset = PUDataset(max_per_class=500)
   ```

2. **Leverage Caching**

   ```python
   # Reuses cache
   dataset = PUDataset(force_reload=False)
   ```

3. **Run Tests in Parallel**

   ```bash
   pip install pytest-xdist
   pytest tests/ -n auto
   ```

4. **Skip Slow Tests During Development**

   ```bash
   pytest tests/ -m "not slow"
   ```

5. **Use Fixtures for Expensive Setup**
   ```python
   @pytest.fixture(scope="module")  # Setup once per module
   def dataset():
       return PUDataset(...)
   ```

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
- [Testing PyTorch Code](https://pytorch.org/docs/stable/notes/testing.html)

## Contributing Tests

When adding new features:

1. Write tests first (TDD approach)
2. Ensure >80% coverage of new code
3. Add integration tests for new workflows
4. Update this documentation
5. Run full test suite before committing:
   ```bash
   python run_tests.py --coverage
   ```

---

For more details, see `tests/README.md`.
