import numpy as np
import pytest

from postmodal.types import ModalData
from postmodal.validation import ModalValidator, validate_modal_data


class TestModalValidator:
    def test_validate_valid_modeshape(self) -> None:
        """Test validation of valid modeshapes."""
        # Valid 1D modeshape
        modeshape_1d = np.array([1.0, 2.0, 3.0, 4.0])
        ModalValidator.validate(modeshape_1d)  # Should not raise

        # Valid 2D modeshape
        modeshape_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ModalValidator.validate(modeshape_2d)  # Should not raise

        # Valid complex modeshape
        complex_modeshape = np.array([1.0 + 0.1j, 2.0 + 0.2j, 3.0 + 0.3j])
        ModalValidator.validate(complex_modeshape)  # Should not raise

    def test_validate_invalid_modeshape(self) -> None:
        """Test validation of invalid modeshapes."""
        # Not a numpy array
        with pytest.raises(TypeError, match="Modeshape must be a numpy array"):
            ModalValidator.validate([1.0, 2.0, 3.0])  # type: ignore[arg-type]

        # 3D array
        with pytest.raises(ValueError, match="Modeshape must be 1D or 2D array"):
            ModalValidator.validate(np.zeros((2, 2, 2)))

        # Empty array
        with pytest.raises(ValueError, match="Modeshape array cannot be empty"):
            ModalValidator.validate(np.array([]))

    def test_validate_pair_valid(self) -> None:
        """Test validation of valid modeshape pairs."""
        # Valid pair of 1D modeshapes with same shape
        phi_1 = np.array([1.0, 2.0, 3.0])
        phi_2 = np.array([4.0, 5.0, 6.0])
        ModalValidator.validate_pair(phi_1, phi_2)  # Should not raise

    def test_validate_pair_invalid(self) -> None:
        """Test validation of invalid modeshape pairs."""
        # Different shapes
        phi_1 = np.array([1.0, 2.0, 3.0])
        phi_2 = np.array([4.0, 5.0])
        with pytest.raises(ValueError, match="must have the same shape"):
            ModalValidator.validate_pair(phi_1, phi_2)

        # Not numpy arrays
        with pytest.raises(TypeError, match="must be a numpy array"):
            ModalValidator.validate_pair([1.0, 2.0], np.array([3.0, 4.0]))  # type: ignore[arg-type]

    def test_validate_modal_data(self) -> None:
        """Test validation of ModalData objects."""
        # Valid ModalData
        frequencies = np.array([10.0, 20.0])
        modeshapes = np.array([[1.0, 2.0], [3.0, 4.0]])
        damping = np.array([0.01, 0.02])
        modal_data = ModalData(frequencies=frequencies, modeshapes=modeshapes, damping=damping)

        # Should not raise
        validate_modal_data(frequencies=modal_data.frequencies, modeshapes=modal_data.modeshapes)

        # Test with invalid data types
        with pytest.raises(TypeError):
            validate_modal_data(frequencies={"data": frequencies}, modeshapes=modeshapes)  # type: ignore[arg-type]
