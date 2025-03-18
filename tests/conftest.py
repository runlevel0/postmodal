"""Common fixtures for pytest tests."""

import numpy as np
import pytest

from postmodal.types import ModalData


@pytest.fixture
def simple_modal_data() -> ModalData:
    """Create a simple ModalData object with real modeshapes."""
    frequencies = np.array([10.0, 20.0, 30.0])
    modeshapes = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [1.0, 4.0, 1.0, 4.0]])
    damping = np.array([0.01, 0.02, 0.03])

    return ModalData(frequencies=frequencies, modeshapes=modeshapes, damping=damping)


@pytest.fixture
def complex_modal_data() -> ModalData:
    """Create a ModalData object with complex modeshapes."""
    frequencies = np.array([10.0, 20.0, 30.0])
    modeshapes = np.array([
        [1.0 + 0.1j, 2.0 + 0.2j, 3.0 + 0.3j, 4.0 + 0.4j],
        [4.0 + 0.4j, 3.0 + 0.3j, 2.0 + 0.2j, 1.0 + 0.1j],
        [1.0 + 0.5j, 4.0 + 2.0j, 1.0 + 0.5j, 4.0 + 2.0j],
    ])
    damping = np.array([0.01, 0.02, 0.03])

    return ModalData(frequencies=frequencies, modeshapes=modeshapes, damping=damping)


@pytest.fixture
def real_modeshape() -> np.ndarray:
    """Create a simple real modeshape."""
    return np.array([1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def complex_modeshape() -> np.ndarray:
    """Create a complex modeshape."""
    return np.array([1.0 + 0.5j, 2.0 + 1.0j, 3.0 + 1.5j, 4.0 + 2.0j])


@pytest.fixture
def highly_complex_modeshape() -> np.ndarray:
    """Create a highly complex modeshape with phases all over the place."""
    return np.array([1.0 + 1.0j, -2.0 + 2.0j, -3.0 - 3.0j, 4.0 - 4.0j])


@pytest.fixture
def modeshape_batch() -> np.ndarray:
    """Create a batch of modeshapes."""
    real = np.array([1.0, 2.0, 3.0, 4.0])
    complex_simple = np.array([1.0 + 0.5j, 2.0 + 1.0j, 3.0 + 1.5j, 4.0 + 2.0j])
    complex_varied = np.array([1.0 + 1.0j, -2.0 + 2.0j, -3.0 - 3.0j, 4.0 - 4.0j])

    return np.array([real, complex_simple, complex_varied])
