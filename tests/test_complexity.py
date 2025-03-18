import numpy as np

from postmodal.complexity import (
    calculate_complexity_metrics,
    calculate_map,
    calculate_mpc,
    calculate_mpd,
)
from postmodal.types import ComplexityMetrics


class TestComplexityMetrics:
    def setup_method(self) -> None:
        """Set up test data."""
        # Create a perfectly real modeshape (should have MPC = 1)
        self.real_modeshape = np.array([1.0, 2.0, 3.0, 4.0])

        # Create a complex modeshape with some phase variation
        self.complex_modeshape = np.array([1.0 + 0.2j, 2.0 + 0.5j, 3.0 + 0.7j, 4.0 + 1.0j])

        # Create a highly complex modeshape (phases all over the place)
        self.highly_complex_modeshape = np.array([1.0 + 1.0j, -2.0 + 2.0j, -3.0 - 3.0j, 4.0 - 4.0j])

        # Create a batch of modeshapes
        self.modeshape_batch = np.array([self.real_modeshape, self.complex_modeshape, self.highly_complex_modeshape])

    def test_mpc_perfect_collinearity(self) -> None:
        """Test MPC calculation for perfectly collinear mode shapes."""
        # Test with zero phase (real numbers)
        real_modeshape = np.array([1.0, 2.0, 3.0, 4.0])
        mpc_real = calculate_mpc(real_modeshape)
        assert np.isclose(mpc_real, 1.0, rtol=1e-10)

        # Test with constant phase (45 degrees)
        constant_phase = np.array([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j, 4.0 + 4.0j])
        mpc_phase = calculate_mpc(constant_phase)
        assert np.isclose(mpc_phase, 1.0, rtol=1e-10)

        # Test with constant phase (30 degrees)
        constant_phase_30 = np.array([
            1.0 + 0.5773502691896258j,  # tan(30°) ≈ 0.5773502691896258
            2.0 + 1.1547005383792515j,
            3.0 + 1.7320508075688772j,
            4.0 + 2.309401076758503j,
        ])
        mpc_phase_30 = calculate_mpc(constant_phase_30)
        assert np.isclose(mpc_phase_30, 1.0, rtol=1e-10)

        # Test with constant phase (60 degrees)
        constant_phase_60 = np.array([
            1.0 + 1.7320508075688772j,  # tan(60°) ≈ 1.7320508075688772
            2.0 + 3.4641016151377544j,
            3.0 + 5.196152422706632j,
            4.0 + 6.928203230275509j,
        ])
        mpc_phase_60 = calculate_mpc(constant_phase_60)
        assert np.isclose(mpc_phase_60, 1.0, rtol=1e-10)

    def test_mpc_collinearity_with_noise(self) -> None:
        """Test MPC calculation for collinear mode shapes with small noise."""
        # Base collinear shape with small random noise
        base = np.array([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j])
        noise = np.array([0.01j, -0.005j, 0.02j])
        noisy_modeshape = base + noise

        mpc = calculate_mpc(noisy_modeshape)
        # Should still be close to 1 despite noise
        assert mpc > 0.95

        # Test with different noise levels
        noise_levels = [0.01, 0.05, 0.1, 0.2]
        for noise_level in noise_levels:
            noisy = base + noise_level * np.random.randn(*base.shape)
            mpc_noisy = calculate_mpc(noisy)
            assert mpc_noisy > 0.9  # Should still be high with small noise

    def test_mpc_perfect_non_collinearity(self) -> None:
        """Test MPC calculation for perfectly non-collinear mode shapes."""
        # Orthogonal components in complex plane
        orthogonal = np.array([1.0 + 0j, 0 + 1j])
        mpc = calculate_mpc(orthogonal)
        assert np.isclose(mpc, 0.0, atol=1e-10)

        # Random phase components
        random_phase = np.array([1.0 + 0.5j, -0.5 + 1.0j, -1.0 - 0.5j, 0.5 - 1.0j])
        mpc_random = calculate_mpc(random_phase)
        assert mpc_random < 0.5  # Should be low for non-collinear components

    def test_mpc_partial_collinearity(self) -> None:
        """Test MPC calculation for partially collinear mode shapes."""
        # Mix of collinear and non-collinear components
        mixed = np.array([1.0 + 0.5j, 2.0 + 1.0j, 0 + 1.0j, -1.0 - 0.5j])
        mpc = calculate_mpc(mixed)
        # Should be between 0 and 1
        assert 0.0 < mpc < 1.0

        # Test with varying degrees of collinearity
        base = np.array([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j])
        deviations = [0.1, 0.2, 0.3, 0.4]
        mpc_values = []
        for dev in deviations:
            non_collinear = base + dev * np.random.randn(*base.shape)
            mpc_values.append(calculate_mpc(non_collinear))

        # MPC should generally decrease as deviation increases
        assert all(0.0 <= mpc <= 1.0 for mpc in mpc_values)
        # Not strictly decreasing due to randomness, but should be lower for larger deviations
        assert np.mean(mpc_values[:2]) > np.mean(mpc_values[2:])

    def test_mpc_edge_cases(self) -> None:
        """Test MPC calculation for various edge cases."""
        # Zero magnitude components
        zero_mag = np.array([1.0 + 0j, 0 + 0j, 2.0 + 0j])
        mpc_zero = calculate_mpc(zero_mag)
        assert 0.0 <= mpc_zero <= 1.0

        # Very small magnitude components
        small_mag = np.array([1e-10 + 0j, 2e-10 + 0j, 3e-10 + 0j])
        mpc_small = calculate_mpc(small_mag)
        assert np.isclose(mpc_small, 1.0, rtol=1e-10)

        # Very large magnitude components
        large_mag = np.array([1e10 + 0j, 2e10 + 0j, 3e10 + 0j])
        mpc_large = calculate_mpc(large_mag)
        assert np.isclose(mpc_large, 1.0, rtol=1e-10)

        # Single component
        single = np.array([1.0 + 1.0j])
        mpc_single = calculate_mpc(single)
        assert np.isclose(mpc_single, 1.0, rtol=1e-10)

    def test_mpc_normalization_invariance(self) -> None:
        """Test that MPC is invariant to scaling/normalization."""
        # Base mode shape
        base = np.array([1.0 + 0.5j, 2.0 + 1.0j, 3.0 + 1.5j])
        mpc_base = calculate_mpc(base)

        # Scaled version
        scaled = base * (1.0 + 2.0j)
        mpc_scaled = calculate_mpc(scaled)

        # MPC should be the same
        assert np.isclose(mpc_base, mpc_scaled, rtol=1e-10)

        # Test with different scaling factors
        scales = [0.1, 1.0, 10.0, 100.0]
        for scale in scales:
            scaled = base * complex(scale)
            mpc = calculate_mpc(scaled)
            assert np.isclose(mpc, mpc_base, rtol=1e-10)

    def test_mpc_conjugate_invariance(self) -> None:
        """Test that MPC is invariant to complex conjugation."""
        # Base mode shape
        base = np.array([1.0 + 0.5j, 2.0 + 1.0j, 3.0 + 1.5j])
        mpc_base = calculate_mpc(base)

        # Complex conjugate
        conjugate = np.conj(base)
        mpc_conjugate = calculate_mpc(conjugate)

        # MPC should be the same
        assert np.isclose(mpc_base, mpc_conjugate, rtol=1e-10)

    def test_mpc_batch_processing(self) -> None:
        """Test MPC calculation for batches of mode shapes."""
        # Create a batch with various cases
        batch = np.array([
            [1.0 + 0j, 2.0 + 0j, 3.0 + 0j, 4.0 + 0j],  # Perfect collinearity
            [1.0 + 00.1j, 2.0 - 0.005j, 3.0 + 00.2j, 4.0 - 00.1j],  # Collinearity with noise
            [1.0 + 0j, 0 + 1j, 1.0 + 0j, 0 + 1j],  # Perfect non-collinearity
            [1.0 + 0.5j, 2.0 + 1.0j, 0 + 1.0j, -1.0 - 0.5j],  # Partial collinearity
        ])

        mpc_values = calculate_mpc(batch)

        # Check shape
        assert mpc_values.shape == (4,)

        # Check ordering (first should be highest, third should be lowest)
        assert mpc_values[0] > mpc_values[1] > mpc_values[2]
        assert 0.0 < mpc_values[3] < 1.0

    def test_map_values(self) -> None:
        """Test MAP calculation."""
        map_values = calculate_map(self.modeshape_batch)
        assert map_values.shape == (3,)
        # MAP should be between 0 and 1
        assert np.all(map_values >= 0)
        assert np.all(map_values <= 1)

    def test_mpd_values(self) -> None:
        """Test MPD calculation."""
        mpd_values = calculate_mpd(self.modeshape_batch)
        assert mpd_values.shape == (3,)
        # MPD should be non-negative (degrees)
        assert np.all(mpd_values >= 0)
        # Real modeshape should have MPD close to 0
        assert mpd_values[0] < 1.0

    def test_mpd_perfect_collinearity(self) -> None:
        """Test MPD calculation for a perfectly collinear mode shape.

        A mode shape where all components lie exactly on a straight line
        in the complex plane should have MPD = 0.
        """
        # Create a perfectly collinear mode shape
        collinear_modeshape = np.array([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j, 4.0 + 4.0j])

        # Test with both weighting schemes
        mpd_magnitude = calculate_mpd(collinear_modeshape, weights="magnitude")
        mpd_equal = calculate_mpd(collinear_modeshape, weights="equal")

        # MPD should be very close to 0 for both weighting schemes
        assert mpd_magnitude < 1e-5
        assert mpd_equal < 1e-5

    def test_mpd_varying_collinearity(self) -> None:
        """Test MPD calculation for mode shapes with varying degrees of collinearity.

        MPD should increase as the collinearity decreases.
        """
        # Create a series of mode shapes with increasing deviation from collinearity
        modeshapes = [
            np.array([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j]),  # Perfect collinearity
            np.array([1.0 + 1.1j, 2.0 + 1.9j, 3.0 + 2.8j]),  # Slight deviation
            np.array([1.0 + 2.0j, 2.0 + 0.0j, 3.0 + 4.0j]),  # Large deviation
        ]

        # Calculate MPD for each modeshape
        mpd_values = np.array([calculate_mpd(shape) for shape in modeshapes])

        # MPD should increase as collinearity decreases
        assert mpd_values[0] < mpd_values[1] < mpd_values[2]

        # All values should be non-negative and less than 180 degrees
        assert np.all(mpd_values >= 0)
        assert np.all(mpd_values < 180.0)

    def test_mpd_large_imaginary_parts(self) -> None:
        """Test MPD calculation for mode shapes with large imaginary and small real parts.

        The implementation should handle this case accurately without numerical issues.
        """
        # Create a mode shape with large imaginary and small real parts
        modeshape = np.array([0.1 + 10.0j, 0.2 + 20.0j, 0.3 + 30.0j])

        # Calculate MPD with both weighting schemes
        mpd_magnitude = calculate_mpd(modeshape, weights="magnitude")
        mpd_equal = calculate_mpd(modeshape, weights="equal")

        # Both results should be valid (non-negative, less than 180 degrees)
        assert mpd_magnitude >= 0 and mpd_magnitude < 180.0
        assert mpd_equal >= 0 and mpd_equal < 180.0

    def test_mpd_weighting_schemes(self) -> None:
        """Test MPD calculation with different weighting schemes.

        Compare magnitude-weighted vs equal-weighted MPD for a mode shape
        with varying component magnitudes.
        """
        # Create a mode shape with varying magnitudes
        modeshape = np.array([1.0 + 1.0j, 5.0 + 5.0j, 0.5 + 0.5j])

        # Calculate MPD with both weighting schemes
        mpd_magnitude = calculate_mpd(modeshape, weights="magnitude")
        mpd_equal = calculate_mpd(modeshape, weights="equal")

        # Both results should be valid
        assert mpd_magnitude >= 0 and mpd_magnitude < 180.0
        assert mpd_equal >= 0 and mpd_equal < 180.0

        # The results should be different due to weighting
        assert not np.isclose(mpd_magnitude, mpd_equal)

        # Magnitude-weighted MPD should be more influenced by the larger component
        # (5.0 + 5.0j) than the equal-weighted MPD
        assert mpd_magnitude < mpd_equal

    def test_calculate_complexity_metrics(self) -> None:
        """Test the combined complexity metrics calculation."""
        metrics = calculate_complexity_metrics(self.modeshape_batch)

        # Check that the result is a ComplexityMetrics object
        assert isinstance(metrics, ComplexityMetrics)

        # Check that all metrics have the correct shape
        assert metrics.mpc.shape == (3,)
        assert metrics.map.shape == (3,)
        assert metrics.mpd.shape == (3,)
