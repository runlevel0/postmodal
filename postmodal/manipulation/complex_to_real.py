"""This module provides functions for converting complex mode shapes to real mode shapes."""

from typing import Literal, Union

import numpy as np


def find_best_fit_angle(complex_vector, method: Literal["sum", "pca"] = "pca") -> float:
    """
    Find the angle in the complex plane that best fits a set of complex numbers.
    This approach finds the angle that maximizes the sum of real parts when
    all points are rotated by that angle.

    Parameters:
    -----------
    complex_vector : numpy.ndarray
        Array of complex numbers

    Returns:
    --------
    angle : float
        Best fit angle in radians
    """
    if method not in ["sum", "pca"]:
        raise ValueError("Method must be 'sum' or 'pca'")
    if method == "pca":
        # Method 2: PCA approach - finds the direction of maximum variance
        real_parts = np.real(complex_vector)
        imag_parts = np.imag(complex_vector)
        points = np.column_stack((real_parts, imag_parts))
        u, s, vh = np.linalg.svd(points, full_matrices=False)
        direction = vh[0]
        pca_angle = np.arctan2(direction[1], direction[0])
        return pca_angle
    elif method == "sum":
        # Method 1: Use the phase of the sum of complex numbers
        # This maximizes the projection onto the real axis
        sum_vector = np.sum(complex_vector)
        angle = np.angle(sum_vector)
        return angle


def cast_modeshape_to_real_real_part(modeshape: np.ndarray) -> np.ndarray:
    """Cast a single complex modeshape or set of complex modeshapes to real modeshapes by taking only the real part.

    This method creates a real-valued mode shape approximation by simply discarding the imaginary part
    of the complex mode shape. While straightforward, this method loses phase information
    encoded in the imaginary component and the resulting real mode may not be a true eigenvector
    of any related real system. It is primarily useful for visualization or rough approximation
    when phase complexity is considered less important.

    Parameters
    ----------
    modeshape : np.ndarray
        Single complex modeshape [n_dof] or set of complex modeshapes [n_modes x n_dof].
        Must be complex-valued.

    Returns
    -------
    np.ndarray
        Real-valued modeshape(s) obtained by taking the real part.
        - Array of same shape as input `modeshape`, but with real values.

    Raises
    ------
    ValueError
        If the input `modeshape` is not a NumPy array or is not complex-valued.
    NotImplementedError
        If the input `modeshape` has dimensions other than 1 or 2.
    """
    if not isinstance(modeshape, np.ndarray) or not np.iscomplexobj(modeshape):
        raise ValueError("Input modeshape must be a complex-valued NumPy array.")

    if modeshape.ndim in [1, 2]:
        return modeshape.real
    else:
        raise NotImplementedError(f"modeshape has dimensions: {modeshape.ndim}, expecting 1 or 2.")


def _validate_ref_dof_index(ref_dof_index: Union[int, np.ndarray, Literal["max"]]) -> None:
    """Validate the reference DOF index."""
    if not (
        isinstance(ref_dof_index, int)
        or (isinstance(ref_dof_index, np.ndarray) and np.issubdtype(ref_dof_index.dtype, np.integer))
        or ref_dof_index == "max"
    ):
        raise ValueError("ref_dof_index must be an integer or a 0-dimensional numpy array of integer type.")


def _process_single_modeshape(modeshape: np.ndarray, ref_dof_index: int) -> np.ndarray:
    """Process a single modeshape with phase correction."""
    ref_dof_phase_angle = np.angle(modeshape[ref_dof_index])
    phase_correction_factor = np.exp(-1j * ref_dof_phase_angle)
    corrected_mode = modeshape * phase_correction_factor
    return corrected_mode.real


def cast_modeshape_to_real_phase_corrected(
    modeshape: np.ndarray,
    ref_dof_index: Union[int, np.ndarray, Literal["max"]] = "max",
) -> np.ndarray:
    """Cast a single complex modeshape or set of complex modeshapes to a visually 'real-mode-like' shape
    by applying phase correction based on a reference DOF.

    This method rotates the complex mode shape in the complex plane so that the component at the specified
    reference DOF becomes real (specifically, its imaginary part becomes approximately zero). This makes the
    animation of the mode shape appear more like a real mode, with motions approximately in-phase or 180°
    out-of-phase relative to the reference DOF.  It is primarily for visualization and does alter the original
    phase relationships quantitatively.

    Parameters
    ----------
    modeshape : np.ndarray
        Single complex modeshape [n_dof] or set of modeshapes [n_modes x n_dof].
        Must be complex-valued.
    ref_dof_index : int
        Index of the reference degree of freedom (0-indexed).

    Returns
    -------
    np.ndarray
        Real-valued modeshape(s) that are phase-corrected for visualization.
        - Array of same shape as input `modeshape`, but with real values.

    Raises
    ------
    ValueError
        If the input `modeshape` is not a NumPy array or is not complex-valued.
    IndexError
        If `ref_dof_index` is out of bounds.
    NotImplementedError
        If the input `modeshape` has dimensions other than 1 or 2.
    """
    if not isinstance(modeshape, np.ndarray) or not np.iscomplexobj(modeshape):
        raise ValueError("Input modeshape must be a complex-valued NumPy array.")
    
    _validate_ref_dof_index(ref_dof_index)

    if isinstance(ref_dof_index, np.ndarray):
        ref_dof_index = ref_dof_index.item()

    if modeshape.ndim == 1:
        if ref_dof_index == "max":
            ref_dof_index = np.argmax(np.abs(modeshape))
        if ref_dof_index < 0 or ref_dof_index >= modeshape.shape[0]:
            raise IndexError(
                f"ref_dof_index {ref_dof_index} is out of bounds for modeshape of length {modeshape.shape[0]}."
            )
        return _process_single_modeshape(modeshape, ref_dof_index)

    # modeshape.ndim == 2
    n_dof = modeshape.shape[1]
    n_modes = modeshape.shape[0]

    if ref_dof_index == "max":
        ref_dof_index = np.argmax(np.abs(modeshape), axis=1)
    elif isinstance(ref_dof_index, int):
        if ref_dof_index < 0 or ref_dof_index >= n_dof:
            raise IndexError(f"ref_dof_index {ref_dof_index} is out of bounds for modeshape with {n_dof} DOFs.")
        ref_dof_index = np.full(n_modes, ref_dof_index, dtype=int)

    if np.any(ref_dof_index < 0) or np.any(ref_dof_index >= n_dof):
        raise IndexError(f"Some reference indices are out of bounds for modeshape with {n_dof} DOFs.")

    return np.array([_process_single_modeshape(mode, idx) for mode, idx in zip(modeshape, ref_dof_index)])


def cast_modeshape_to_real_average_phase_rotation(modeshape: np.ndarray) -> np.ndarray:
    """Cast a single complex modeshape or set of complex modeshapes to real modeshapes
    using average phase rotation and magnitude-sign of real part.

    This method attempts to create a real-valued mode shape approximation by:
    1. Calculating the average phase angle of all components in the complex mode shape.
    2. Rotating the entire mode shape in the complex plane by subtracting this average phase angle
       from the phase of each component. This centers the phase distribution around zero.
    3. Casting to a real mode shape by taking the magnitude of each (phase-rotated) complex component
       and multiplying it by the sign of the real part of the phase-rotated component. This aims to preserve
       amplitude information and approximate in-phase/out-of-phase behavior.

    Parameters
    ----------
    modeshape : np.ndarray
        Single complex modeshape [n_dof] or set of complex modeshapes [n_modes x n_dof].
        Must be complex-valued.

    Returns
    -------
    np.ndarray
        Real-valued modeshape(s) obtained via average phase rotation and magnitude-sign method.
        - Array of same shape as input `modeshape`, but with real values.

    Raises
    ------
    ValueError
        If the input `modeshape` is not a NumPy array or is not complex-valued.
    NotImplementedError
        If the input `modeshape` has dimensions other than 1 or 2.
    """
    if not isinstance(modeshape, np.ndarray) or not np.iscomplexobj(modeshape):
        raise ValueError("Input modeshape must be a complex-valued NumPy array.")

    if modeshape.ndim == 1:
        best_fit_phase_angle = find_best_fit_angle(modeshape)
        phase_rotation_factors = np.exp(-1j * best_fit_phase_angle)
        rotated_modeshape = modeshape * phase_rotation_factors
        real_modeshape = np.abs(rotated_modeshape) * np.sign(rotated_modeshape.real)
        return real_modeshape

    elif modeshape.ndim == 2:
        real_modeshapes = []
        for mode in modeshape:
            best_fit_phase_angle = find_best_fit_angle(modeshape)
            phase_rotation_factors = np.exp(-1j * best_fit_phase_angle)
            rotated_modeshape = mode * phase_rotation_factors
            real_mode = np.abs(rotated_modeshape) * np.sign(rotated_modeshape.real)
            real_modeshapes.append(real_mode)
        return np.array(real_modeshapes)

    else:
        raise NotImplementedError(f"modeshape has dimensions: {modeshape.ndim}, expecting 1 or 2.")


def cast_modeshape_to_real(
    modeshape: np.ndarray,
    method: Literal["real_part", "phase_corrected", "average_phase_rotation"] = "average_phase_rotation",
    **kwargs,
) -> np.ndarray:
    """Wrapper function that casts a complex modeshape to real using the specified method.

    Parameters
    ----------
    modeshape : np.ndarray
        Single complex modeshape [n_dof] or set of complex modeshapes [n_modes x n_dof].
        Must be complex-valued.
    method : str, optional
        Method to use for converting complex modeshape to real:
        - "real_part": Simply takes the real part of the complex modeshape.
        - "phase_corrected": Rotates the complex modeshape to align with a reference DOF.
        - "average_phase_rotation": Uses average phase rotation and magnitude-sign approach.
        Default is "phase_corrected".
    **kwargs : dict
        Additional parameters specific to each method:
        - For "phase_corrected": 'ref_dof_index' (int or "max") - Reference DOF for phase correction.
          Default is "max" which uses the DOF with maximum amplitude.

    Returns
    -------
    np.ndarray
        Real-valued modeshape(s) of same shape as input, converted using the specified method.

    Raises
    ------
    ValueError
        If an invalid method is specified or if required parameters are missing.
    """
    if not isinstance(modeshape, np.ndarray) or not np.iscomplexobj(modeshape):
        raise ValueError("Input modeshape must be a complex-valued NumPy array.")

    if method == "real_part":
        return cast_modeshape_to_real_real_part(modeshape)

    elif method == "phase_corrected":
        ref_dof_index = kwargs.get("ref_dof_index", "max")
        return cast_modeshape_to_real_phase_corrected(modeshape, ref_dof_index)

    elif method == "average_phase_rotation":
        return cast_modeshape_to_real_average_phase_rotation(modeshape)

    else:
        raise ValueError(
            f"Invalid method: {method}. Must be one of: 'real_part', 'phase_corrected', or 'average_phase_rotation'."
        )
