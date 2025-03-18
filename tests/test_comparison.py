import numpy as np
import pytest

from postmodal.comparison import calculate_mac
from postmodal.comparison.matrix import (
    calculate_mac_matrix,
    calculate_mode_matching_matrix,
    match_modes,
)


class TestMAC:
    def setup_method(self) -> None:
        """Set up test data."""
        # Create two identical modeshapes (should have MAC = 1)
        self.phi_1 = np.array([1.0, 2.0, 3.0, 4.0])
        self.phi_2 = np.array([1.0, 2.0, 3.0, 4.0])

        # Create two orthogonal modeshapes (should have MAC = 0)
        self.phi_3 = np.array([1.0, 0.0, 0.0, 0.0])
        self.phi_4 = np.array([0.0, 1.0, 0.0, 0.0])

        # Create two similar but not identical modeshapes
        self.phi_5 = np.array([1.0, 2.0, 3.0, 4.0])
        self.phi_6 = np.array([1.1, 2.1, 3.1, 4.1])

        # Create complex modeshapes
        self.phi_7 = np.array([1.0 + 0.1j, 2.0 + 0.2j, 3.0 + 0.3j, 4.0 + 0.4j])
        self.phi_8 = np.array([1.0 + 0.1j, 2.0 + 0.2j, 3.0 + 0.3j, 4.0 + 0.4j])

        # Create sets of modeshapes for matrix calculations
        self.modeset_1 = np.array([self.phi_1, self.phi_3, self.phi_5])
        self.modeset_2 = np.array([self.phi_2, self.phi_4, self.phi_6])

        self.frequencies_1 = np.array([1.0, 2.0, 3.0])
        self.frequencies_2 = np.array([1.0, 2.0, 3.0]) * 2

    def test_mac_identical(self) -> None:
        """Test MAC calculation for identical modeshapes."""
        mac = calculate_mac(self.phi_1, self.phi_2)
        assert mac == pytest.approx(1.0)

    def test_mac_orthogonal(self) -> None:
        """Test MAC calculation for orthogonal modeshapes."""
        mac = calculate_mac(self.phi_3, self.phi_4)
        assert mac == pytest.approx(0.0)

    def test_mac_similar(self) -> None:
        """Test MAC calculation for similar modeshapes."""
        mac = calculate_mac(self.phi_5, self.phi_6)
        assert mac > 0.99  # Should be close to 1 but not exactly 1
        assert mac < 1.0

    def test_mac_complex(self) -> None:
        """Test MAC calculation for complex modeshapes."""
        mac = calculate_mac(self.phi_7, self.phi_8)
        assert mac == pytest.approx(1.0)

    def test_mac_matrix(self) -> None:
        """Test MAC matrix calculation."""
        mac_matrix = calculate_mac_matrix(self.modeset_1, self.modeset_2)

        # Check matrix shape
        assert mac_matrix.shape == (3, 3)

        # Check diagonal values (should be high for matching modes)
        assert mac_matrix[0, 0] == pytest.approx(1.0)  # phi_1 vs phi_2
        assert mac_matrix[1, 1] == pytest.approx(0.0)  # phi_3 vs phi_4
        assert mac_matrix[2, 2] > 0.9  # phi_5 vs phi_6

    def test_mode_matching_matrix(self) -> None:
        """Test mode matching matrix calculation."""
        match_matrix = calculate_mode_matching_matrix(
            self.frequencies_1, self.modeset_1, self.frequencies_2, self.modeset_2
        )

        # Check matrix shape
        assert match_matrix.shape == (3, 3)

    def test_match_modes(self) -> None:
        """Test mode matching function."""
        match_matrix = calculate_mode_matching_matrix(
            self.frequencies_1, self.modeset_1, self.frequencies_2, self.modeset_2
        )

        matches = match_modes(match_matrix)

        # Should return a list of 2 tuples
        assert len(matches) == 2
        assert isinstance(matches[0], tuple)
        # assert len(matches[0]) == 2

        # # First match should be between phi_1 and phi_2 with MAC ≈ 1
        # assert matches[0][0] == 0
        # assert matches[0][1] == 0
        # assert matches[0][2] == pytest.approx(1.0)
