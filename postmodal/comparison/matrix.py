"""Modal comparison matrix operations."""

from typing import Optional, Sequence

import numpy as np

from ..types import MACValue, Modeshape
from ..validation import ModalValidator


def calculate_mac_matrix(phi_1: np.ndarray, phi_2: np.ndarray) -> np.ndarray:
    """Calculate the MAC matrix between two sets of modeshapes.

    Parameters
    ----------
    phi_1 : np.ndarray
        First set of modeshapes [n_modes_1 x n_dof]
    phi_2 : np.ndarray
        Second set of modeshapes [n_modes_2 x n_dof]

    Returns
    -------
    np.ndarray
        MAC matrix [n_modes_1 x n_modes_2]
    """
    ModalValidator.validate_pair(phi_1, phi_2)

    # discern between real and complex modes
    # numpy.vdot cannot be used here, since
    # "it should only be used for vectors."
    # https://numpy.org/doc/stable/reference/generated/numpy.vdot.html
    if np.iscomplexobj(phi_1) or np.iscomplexobj(phi_2):
        return np.square(
            np.abs(np.dot(phi_1, phi_2.conj().T))
            / (
                np.linalg.norm(phi_1, axis=1)[:, np.newaxis]
                * np.linalg.norm(phi_2, axis=1)
            )
        )
    else:
        return np.square(
            np.abs(np.dot(phi_1, phi_2.T))
            / (
                np.linalg.norm(phi_1, axis=1)[:, np.newaxis]
                * np.linalg.norm(phi_2, axis=1)
            )
        )


def calculate_mode_matching_matrix(
    frequencies_1: np.ndarray,
    modeshapes_1: np.ndarray,
    frequencies_2: np.ndarray,
    modeshapes_2: np.ndarray,
    modeshape_weight: float = 1.0,
) -> np.ndarray:
    """Calculate mode matching matrix considering frequency and MAC.

    Based on Simoen et al., 2014, "Dealing with uncertainty in model updating in damage assessment"

    Parameters
    ----------
    frequencies_1 : np.ndarray
        Frequencies of first set [n_modes_1]
    modeshapes_1 : np.ndarray
        Modeshapes of first set [n_modes_1 x n_dof]
    frequencies_2 : np.ndarray
        Frequencies of second set [n_modes_2]
    modeshapes_2 : np.ndarray
        Modeshapes of second set [n_modes_2 x n_dof]
    modeshape_weight : float, optional
        Weight for modeshape contribution, by default 1.0

    Returns
    -------
    np.ndarray
        Matching matrix [n_modes_1 x n_modes_2]
    """
    mac_matrix = calculate_mac_matrix(modeshapes_1, modeshapes_2)
    freq_matrix = np.abs(
        1 - frequencies_1[:, np.newaxis] / frequencies_2[np.newaxis, :]
    )

    return modeshape_weight * (1 - mac_matrix) + freq_matrix


def match_modes(
    match_matrix: np.ndarray, threshold: float = 0.6
) -> tuple[Sequence[int], Sequence[int]]:
    """Find best matching modes using matching matrix.

    Parameters
    ----------
    match_matrix : np.ndarray
        Matching matrix [n_modes_1 x n_modes_2]
    threshold : float, optional
        Threshold for acceptable matches, by default 0.6

    Returns
    -------
    tuple[Sequence[int], Sequence[int]]
        Indices of matching modes from first and second set
    """
    if not np.all(match_matrix > 0.0):
        raise ValueError("All matrix values must be positive")
    if match_matrix.ndim != 2:
        raise ValueError("Matrix must be 2D")

    n_modes_1, n_modes_2 = match_matrix.shape
    n_modes_min = min(n_modes_1, n_modes_2)

    _match_matrix = match_matrix.copy()
    matching_modes_1 = []
    matching_modes_2 = []

    for _ in range(n_modes_min):
        match = best_match(_match_matrix, threshold)
        if match is None:
            break

        matching_modes_1.append(match[0])
        matching_modes_2.append(match[1])

        # Mark matched modes
        _match_matrix[match[0], :] = threshold + 1.0
        _match_matrix[:, match[1]] = threshold + 1.0

    return tuple(matching_modes_1), tuple(matching_modes_2)


def best_match(
    match_matrix: np.ndarray, threshold: float = 0.6
) -> Optional[tuple[int, int]]:
    """Find indices of best matching mode below threshold.

    Parameters
    ----------
    match_matrix : np.ndarray
        Matching matrix [n_modes_1 x n_modes_2]
    threshold : float, optional
        Threshold for acceptable matches, by default 0.6

    Returns
    -------
    Optional[tuple[int, int]]
        Indices of best match, or None if no match below threshold
    """
    if not np.all(match_matrix > 0.0):
        raise ValueError("All matrix values must be positive")
    if match_matrix.ndim != 2:
        raise ValueError("Matrix must be 2D")

    if not np.any(match_matrix <= threshold):
        return None

    return np.unravel_index(np.argmin(match_matrix, axis=None), match_matrix.shape)
