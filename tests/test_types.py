import numpy as np
import pytest

from postmodal.types import ComplexityMetrics, ModalData


class TestModalData:
    def test_valid_initialization(self):
        """Test that ModalData can be initialized with valid data."""
        frequencies = np.array([10.0, 20.0, 30.0])
        modeshapes = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        damping = np.array([0.01, 0.02, 0.03])

        modal_data = ModalData(frequencies=frequencies, modeshapes=modeshapes, damping=damping)

        np.testing.assert_array_equal(modal_data.frequencies, frequencies)
        np.testing.assert_array_equal(modal_data.modeshapes, modeshapes)
        np.testing.assert_array_equal(modal_data.damping, damping)

    def test_default_damping(self):
        """Test that damping defaults to zeros when not provided."""
        frequencies = np.array([10.0, 20.0])
        modeshapes = np.array([[1.0, 2.0], [3.0, 4.0]])

        modal_data = ModalData(frequencies=frequencies, modeshapes=modeshapes)

        np.testing.assert_array_equal(modal_data.damping, np.zeros_like(frequencies))

    def test_invalid_frequencies(self):
        """Test that invalid frequencies raise appropriate errors."""
        # Test negative frequencies
        with pytest.raises(ValueError, match="frequencies must be positive"):
            ModalData(frequencies=np.array([-10.0, 20.0]), modeshapes=np.array([[1.0, 2.0], [3.0, 4.0]]))

        # Test complex frequencies
        with pytest.raises(ValueError, match="frequencies must be real-valued"):
            ModalData(frequencies=np.array([10.0 + 1j, 20.0]), modeshapes=np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_invalid_modeshapes(self):
        """Test that invalid modeshapes raise appropriate errors."""
        # Test mismatched shapes
        with pytest.raises(ValueError, match="Number of frequencies must match number of modes"):
            ModalData(frequencies=np.array([10.0, 20.0, 30.0]), modeshapes=np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_invalid_damping(self):
        """Test that invalid damping values raise appropriate errors."""
        frequencies = np.array([10.0, 20.0])
        modeshapes = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Test negative damping
        with pytest.raises(ValueError, match="damping ratios must be non-negative"):
            ModalData(frequencies=frequencies, modeshapes=modeshapes, damping=np.array([-0.01, 0.02]))

        # Test damping > 1
        with pytest.raises(ValueError, match="damping ratios must be less than or equal to 1"):
            ModalData(frequencies=frequencies, modeshapes=modeshapes, damping=np.array([0.01, 1.2]))


class TestComplexityMetrics:
    def test_valid_initialization(self):
        """Test that ComplexityMetrics can be initialized with valid data."""
        metrics = ComplexityMetrics(mpc=np.array([0.9, 0.8]), map=np.array([0.95, 0.85]), mpd=np.array([5.0, 10.0]))

        assert metrics.mpc.shape == (2,)
        assert metrics.map.shape == (2,)
        assert metrics.mpd.shape == (2,)

    def test_invalid_shapes(self):
        """Test that mismatched shapes raise appropriate errors."""
        with pytest.raises(ValueError, match="All metrics must have the same shape"):
            ComplexityMetrics(
                mpc=np.array([0.9, 0.8]),
                map=np.array([0.95, 0.85, 0.75]),  # Different shape
                mpd=np.array([5.0, 10.0]),
            )
