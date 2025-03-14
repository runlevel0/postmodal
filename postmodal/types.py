"""Custom types and data classes for modal analysis."""

from dataclasses import dataclass, field
from typing import NewType, Optional

import numpy as np

# Custom types for better type safety
Modeshape = NewType("Modeshape", np.ndarray)
Frequency = NewType("Frequency", np.ndarray)
MACValue = NewType("MACValue", float)
ComplexityValue = NewType("ComplexityValue", float)


@dataclass
class ModalData:
    """Container for modal analysis data.

    Attributes
    ----------
    frequencies : np.ndarray
        Array of natural frequencies [n_modes]. Must be positive real values.
    modeshapes : np.ndarray
        Array of mode shapes [n_modes x n_dof]
    damping : np.ndarray, optional
        Array of damping ratios [n_modes], defaults to zeros.
        Must be real-valued positive numbers in range [0, 1].
    """

    frequencies: np.ndarray
    modeshapes: np.ndarray
    damping: np.ndarray = field(default_factory=lambda: None)

    def __post_init__(self):
        """Validate the data after initialization."""
        # Validate frequencies
        if not isinstance(self.frequencies, np.ndarray):
            raise TypeError("frequencies must be a numpy array")
        if not np.isreal(self.frequencies).all():
            raise ValueError("frequencies must be real-valued")
        if np.any(self.frequencies <= 0):
            raise ValueError("frequencies must be positive")

        # Validate modeshapes
        if not isinstance(self.modeshapes, np.ndarray):
            raise TypeError("modeshapes must be a numpy array")
        if self.modeshapes.ndim not in [1, 2]:
            raise ValueError("modeshapes must be 1D or 2D array")
        if self.frequencies.shape[0] != self.modeshapes.shape[0]:
            raise ValueError("Number of frequencies must match number of modes")

        # Initialize and validate damping
        if self.damping is None:
            self.damping = np.zeros_like(self.frequencies)
        else:
            if not isinstance(self.damping, np.ndarray):
                raise TypeError("damping must be a numpy array")
            if not np.isreal(self.damping).all():
                raise ValueError("damping must be real-valued")
            if self.damping.shape != self.frequencies.shape:
                raise ValueError("damping must have same shape as frequencies")
            if np.any(self.damping < 0):
                raise ValueError("damping ratios must be non-negative")
            if np.any(self.damping > 1):
                raise ValueError("damping ratios must be less than or equal to 1")


@dataclass
class ComplexityMetrics:
    """Container for modal complexity metrics.

    Attributes
    ----------
    mpc : np.ndarray
        Modal Phase Collinearity values [n_modes]
    map : np.ndarray
        Modal Amplitude Proportionality values [n_modes]
    ipr : np.ndarray
        Imaginary Part Ratio values [n_modes]
    cf : np.ndarray
        Complexity Factor values [n_modes]
    mpd : np.ndarray
        Mean Phase Deviation values [n_modes] in degrees
    """

    mpc: np.ndarray
    map: np.ndarray
    ipr: np.ndarray
    cf: np.ndarray
    mpd: np.ndarray

    def __post_init__(self):
        """Validate the metrics after initialization."""
        shapes = [
            self.mpc.shape,
            self.map.shape,
            self.ipr.shape,
            self.cf.shape,
            self.mpd.shape,
        ]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError("All metrics must have the same shape")
