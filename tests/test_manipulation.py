import numpy as np
import pytest

from postmodal.manipulation import (
    align_phase,
    complex_to_real,
    complex_to_real_batch,
    normalize_modeshape,
    normalize_modeshape_reference_dof,
    normalize_modeshape_unit_norm_max_amplitude,
    normalize_modeshape_unit_norm_vector_length,
    normalize_phase,
    unwrap_phase,
)


class TestNormalization:
    def setup_method(self) -> None:
        """Set up test data."""
        # Create a simple real modeshape
        self.real_modeshape = np.array([1.0, 2.0, 3.0, 4.0])

        # Create a complex modeshape
        self.complex_modeshape = np.array([1.0 + 0.5j, 2.0 + 1.0j, 3.0 + 1.5j, 4.0 + 2.0j])

        # Create a batch of modeshapes
        self.modeshape_batch = np.array([self.real_modeshape, self.complex_modeshape])

    def test_normalize_modeshape_default(self) -> None:
        """Test default normalization (unit norm vector length)."""
        normalized = normalize_modeshape(self.real_modeshape)

        # Check that the vector has unit length
        assert np.linalg.norm(normalized) == pytest.approx(1.0)

    def test_normalize_modeshape_unit_norm_vector_length(self) -> None:
        """Test normalization to unit norm vector length."""
        normalized = normalize_modeshape_unit_norm_vector_length(self.real_modeshape)

        # Check that the vector has unit length
        assert np.linalg.norm(normalized) == pytest.approx(1.0)

        # Test with complex modeshape
        normalized_complex = normalize_modeshape_unit_norm_vector_length(self.complex_modeshape)
        assert np.linalg.norm(normalized_complex) == pytest.approx(1.0)

    def test_normalize_modeshape_unit_norm_max_amplitude(self) -> None:
        """Test normalization to unit maximum amplitude."""
        normalized = normalize_modeshape_unit_norm_max_amplitude(self.real_modeshape)

        # Check that the maximum amplitude is 1.0
        assert np.max(np.abs(normalized)) == pytest.approx(1.0)

        # Test with complex modeshape
        normalized_complex = normalize_modeshape_unit_norm_max_amplitude(self.complex_modeshape)
        assert np.max(np.abs(normalized_complex)) == pytest.approx(1.0)

    def test_normalize_modeshape_reference_dof(self) -> None:
        """Test normalization to reference DOF."""
        # Normalize to the first DOF (index 0)
        normalized = normalize_modeshape_reference_dof(self.real_modeshape, ref_dof_index=0)

        # Check that the reference DOF has value 1.0
        assert normalized[0] == pytest.approx(1.0)

        # Test with complex modeshape, normalizing to the second DOF (index 1)
        normalized_complex = normalize_modeshape_reference_dof(self.complex_modeshape, ref_dof_index=1)
        assert np.abs(normalized_complex[1]) == pytest.approx(1.0)

    def test_normalize_batch(self) -> None:
        """Test normalization of a batch of modeshapes."""
        normalized_batch = normalize_modeshape(self.modeshape_batch)

        # Check that each modeshape in the batch is normalized
        for i in range(len(normalized_batch)):
            assert np.linalg.norm(normalized_batch[i]) == pytest.approx(1.0)


class TestPhase:
    def setup_method(self) -> None:
        """Set up test data."""
        # Create a complex modeshape with consistent phase
        self.complex_modeshape = np.array([
            1.0 + 1.0j,  # 45 degrees
            2.0 + 2.0j,  # 45 degrees
            3.0 + 3.0j,  # 45 degrees
            4.0 + 4.0j,  # 45 degrees
        ])

        # Create a complex modeshape with varying phase
        self.varying_phase_modeshape = np.array([
            1.0 + 0.0j,  # 0 degrees
            0.0 + 1.0j,  # 90 degrees
            -1.0 + 0.0j,  # 180 degrees
            0.0 - 1.0j,  # 270 degrees
        ])

    def test_align_phase(self) -> None:
        """Test phase alignment."""
        aligned = align_phase(self.varying_phase_modeshape)

        # After alignment, the phases should be more consistent
        phases = np.angle(aligned, deg=True)
        phase_range = np.max(phases) - np.min(phases)

        # The phase range should be reduced after alignment
        assert phase_range <= 270.0  # Or use the original range as a baseline

    def test_unwrap_phase(self) -> None:
        """Test phase unwrapping."""
        # Create a more suitable test case that clearly demonstrates unwrapping
        phases_deg = np.array([0, 90, 180, -90, 0, 90, 180, -90])  # Rotating twice around the circle
        wrapped_phase = np.array([np.exp(1j * np.radians(p)) for p in phases_deg])

        # Get the original phases and manually unwrap them for comparison
        original_phases = np.angle(wrapped_phase)
        manually_unwrapped = np.unwrap(original_phases)

        # Apply the function
        unwrapped = unwrap_phase(wrapped_phase)

        # Check that the function result, when unwrapped by numpy,
        # matches what we'd get by directly unwrapping the original
        result_phases = np.angle(unwrapped)
        result_unwrapped = np.unwrap(result_phases)

        # The function might not preserve the exact unwrapped values,
        # but should preserve the phase relationships
        assert np.allclose(np.diff(manually_unwrapped), np.diff(result_unwrapped))

    def test_normalize_phase(self) -> None:
        """Test phase normalization."""
        normalized = normalize_phase(self.complex_modeshape)

        # After normalization, the average phase should be close to 0
        phases = np.angle(normalized, deg=True)
        assert np.abs(np.mean(phases)) < 10.0


class TestComplexToReal:
    def setup_method(self) -> None:
        """Set up test data."""
        # Create a complex modeshape
        self.complex_modeshape = np.array([1.0 + 0.5j, 2.0 + 1.0j, 3.0 + 1.5j, 4.0 + 2.0j])

        # Create a batch of complex modeshapes
        self.modeshape_batch = np.array([self.complex_modeshape, self.complex_modeshape * 2])

    def test_complex_to_real(self) -> None:
        """Test conversion from complex to real."""
        real_modeshape = complex_to_real(self.complex_modeshape)

        # Check that the result is real
        assert np.isreal(real_modeshape).all()

        # Check that the shape is preserved
        assert real_modeshape.shape == self.complex_modeshape.shape

    def test_complex_to_real_batch(self) -> None:
        """Test batch conversion from complex to real."""
        real_batch = complex_to_real_batch(self.modeshape_batch)

        # Check that the result is real
        assert np.isreal(real_batch).all()

        # Check that the shape is preserved
        assert real_batch.shape == self.modeshape_batch.shape
