# Tests for Postmodal

This directory contains pytest tests for the `postmodal` package.

## Test Structure

- `conftest.py`: Common fixtures used across multiple test files
- `test_types.py`: Tests for the data types and classes
- `test_complexity.py`: Tests for complexity metrics calculations
- `test_comparison.py`: Tests for MAC and mode comparison functions
- `test_manipulation.py`: Tests for modeshape manipulation functions
- `test_validation.py`: Tests for validation functions
- `test_visualization.py`: Tests for visualization functions

## Running Tests

To run all tests:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_types.py
```

To run with coverage:

```bash
pytest --cov=postmodal
```

## Test Coverage

The tests cover:

1. Data types and validation

   - `ModalData` class initialization and validation
   - `ComplexityMetrics` class initialization and validation
   - Modeshape validation

2. Complexity metrics

   - Modal Phase Collinearity (MPC)
   - Modal Amplitude Proportionality (MAP)
   - Imaginary Part Ratio (IPR)
   - Complexity Factor (CF)
   - Mean Phase Deviation (MPD)

3. Comparison functions

   - Modal Assurance Criterion (MAC)
   - MAC matrix calculation
   - Mode matching

4. Manipulation functions

   - Modeshape normalization
   - Phase alignment and normalization
   - Complex to real conversion

5. Visualization
   - MAC matrix plotting
   - Modeshape complexity plotting
