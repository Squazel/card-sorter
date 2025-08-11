# Testing Card Sorter

This directory contains tests for the Card Sorter project.

## Test Suite Overview

The test suite includes:

1. **Unit Tests**: Testing individual components in isolation
   - `test_card_ordering_rules.py`: Tests for card ordering and mapping functionality
   - `test_card_ordering_usage.py`: Tests for practical usage patterns of card orderings

2. **Integration Tests**: Testing components working together
   - `test_integration.py`: Tests integrating card ordering with sorting algorithms

## Development Setup

### Virtual Environment

It's recommended to use a virtual environment for development:

```bash
# Create a virtual environment in the project root
python -m venv venv

# Activate the virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.\venv\Scripts\activate.bat
# On macOS/Linux:
source venv/bin/activate

# To deactivate when you're done
deactivate
```

### Install Test Dependencies

The test suite requires additional packages that are specified in the `requirements.txt` file in this directory. To install these dependencies:

```bash
# From the tests directory (with activated virtual environment)
pip install -r requirements.txt

# Or from the project root (with activated virtual environment)
pip install -r tests/requirements.txt
```

### Test Configuration

Test configuration is defined in:
- `conftest.py`: Ensures proper module importing for tests
- `pytest.ini`: (In project root) Contains pytest configuration options

## Running Tests

### From Command Line

From the project root directory:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=.

# Run specific test file
pytest tests/test_card_ordering_rules.py

# Run specific test
pytest tests/test_card_ordering_rules.py::TestCardOrderingRules::test_bridge_mapping_conversion

# Run tests in parallel (faster on multi-core systems)
pytest -xvs
```

### From VS Code

VS Code is configured to discover and run tests automatically:

1. Open the Testing panel by clicking the beaker/flask icon in the sidebar
2. Click the Play button at the top to run all tests
3. Click the Play button next to individual tests or test files to run only those tests
4. Use the "Debug Test" option (bug icon) to debug tests with breakpoints

## Adding New Tests

To add a new test:

1. Create a Python file in this directory with a name starting with `test_`
2. Create classes with names starting with `Test`
3. Create test methods with names starting with `test_`

Example:

```python
import unittest
import card_ordering_rules as cor

class TestNewFeature(unittest.TestCase):
    
    def test_some_functionality(self):
        # Test code here
        self.assertEqual(expected_result, actual_result)
```

## Code Coverage

To generate a code coverage report:

```bash
# Generate terminal report
pytest --cov=.

# Generate HTML report
pytest --cov=. --cov-report=html
# Then open htmlcov/index.html in a browser
```
