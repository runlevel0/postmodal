"""Functions for calculating modal complexity metrics."""

import numpy as np

from .types import ComplexityMetrics
from .validation import ModalValidator


def calculate_mpc(modeshape: np.ndarray) -> np.ndarray:
    """Calculate Modal Phase Collinearity (MPC) for a modeshape or set of modeshapes.

    MPC quantifies the 'realness' of a mode shape by measuring the phase alignment
    of its components. A higher MPC value (≈ 1) indicates a more 'real' mode shape
    with phases close to 0° or 180°, suggesting proportionally damped or undamped behavior.
    A lower MPC value (≈ 0) suggests a more complex mode shape with scattered phases,
    indicating non-proportional damping or mode coupling.

    Math:
    -----
    .. math::
        MPC_r = \\frac{ \\left| \\sum_{j=1}^{n} Re(\\Phi_{jr}) \\right|^2 }{ \\sum_{j=1}^{n} |\\Phi_{jr}|^2 }

    Where:
    - :math:`\\Phi_{jr}`: j-th component of the r-th complex mode shape.
    - :math:`Re(\\Phi_{jr})`: Real part of :math:`\\Phi_{jr}`.
    - :math:`n`: Number of DOFs.
    - :math:`| ... |`: Magnitude (absolute value for scalar, modulus for complex).

    Parameters
    ----------
    modeshape : np.ndarray
        Single modeshape [n_dof] or set of modeshapes [n_modes x n_dof].
        Can be real or complex-valued.

    Returns
    -------
    np.ndarray
        MPC value(s).
        - Scalar if input is a single modeshape [].
        - 1D array [n_modes] if input is a set of modeshapes.

    Raises
    ------
    ValueError
        If the input `modeshape` is not a NumPy array.
    NotImplementedError
        If the input `modeshape` has dimensions other than 1 or 2.
    """
    ModalValidator.validate(modeshape)

    if modeshape.ndim == 1:
        sum_real_parts = np.sum(modeshape.real)
        numerator = np.abs(sum_real_parts) ** 2
        denominator = np.sum(np.abs(modeshape) ** 2)
        mpc_value = numerator / denominator if denominator != 0 else 0.0  # Handle potential division by zero
        return np.array(mpc_value)  # Return as 0D array for consistency

    elif modeshape.ndim == 2:
        mpc_values = []
        for mode in modeshape:
            sum_real_parts = np.sum(mode.real)
            numerator = np.abs(sum_real_parts) ** 2
            denominator = np.sum(np.abs(mode) ** 2)
            mpc_value = numerator / denominator if denominator != 0 else 0.0  # Handle potential division by zero
            mpc_values.append(mpc_value)
        return np.array(mpc_values)

    else:
        raise NotImplementedError(f"modeshape has dimensions: {modeshape.ndim}, expecting 1 or 2.")


def calculate_map(modeshape: np.ndarray) -> np.ndarray:
    """Calculate Modal Amplitude Proportionality (MAP) for a modeshape or set of modeshapes.

    MAP assesses the amplitude proportionality of a complex mode shape.
    A higher MAP value (≈ 1) indicates that the mode shape amplitudes are largely
    proportional to their real parts, suggesting a more 'real' amplitude distribution.
    A lower MAP value (< 1) indicates a deviation from real amplitude proportionality,
    suggesting a more complex amplitude distribution and significant imaginary components.

    Math:
    -----
    .. math::
        MAP_r = \\frac{ \\sum_{j=1}^{n} |Re(\\Phi_{jr})| }{ \\sum_{j=1}^{n} |\\Phi_{jr}| }

    Where:
    - :math:`\\Phi_{jr}`: j-th component of the r-th complex mode shape.
    - :math:`Re(\\Phi_{jr})`: Real part of :math:`\\Phi_{jr}`.
    - :math:`n`: Number of DOFs.
    - :math:`| ... |`: Magnitude (absolute value for scalar, modulus for complex).

    Parameters
    ----------
    modeshape : np.ndarray
        Single modeshape [n_dof] or set of modeshapes [n_modes x n_dof].
        Can be real or complex-valued.

    Returns
    -------
    np.ndarray
        MAP value(s).
        - Scalar if input is a single modeshape [].
        - 1D array [n_modes] if input is a set of modeshapes.

    Raises
    ------
    ValueError
        If the input `modeshape` is not a NumPy array.
    NotImplementedError
        If the input `modeshape` has dimensions other than 1 or 2.
    """
    ModalValidator.validate(modeshape)

    if modeshape.ndim == 1:
        numerator = np.sum(np.abs(modeshape.real))
        denominator = np.sum(np.abs(modeshape))
        map_value = numerator / denominator if denominator != 0 else 0.0  # Handle potential division by zero
        return np.array(map_value)  # Return as 0D array for consistency

    elif modeshape.ndim == 2:
        map_values = []
        for mode in modeshape:
            numerator = np.sum(np.abs(mode.real))
            denominator = np.sum(np.abs(mode))
            map_value = numerator / denominator if denominator != 0 else 0.0  # Handle potential division by zero
            map_values.append(map_value)
        return np.array(map_values)

    else:
        raise NotImplementedError(f"modeshape has dimensions: {modeshape.ndim}, expecting 1 or 2.")


def calculate_ipr(modeshape: np.ndarray) -> np.ndarray:
    """Calculate Imaginary Part Ratio (IPR) for a modeshape or set of modeshapes.

    IPR quantifies the relative magnitude of the imaginary part of a mode shape
    compared to its real part. A lower IPR value (接近 0) indicates a mode shape
    dominated by its real part, while a higher IPR signifies a more significant
    imaginary contribution, suggesting greater complexity.

    Math:
    -----
    .. math::
        IPR_r = \\frac{ || Im(\\vec{\\Phi}_r) || }{ || Re(\\vec{\\Phi}_r) || }

    Where:
    - :math:`\\vec{\\Phi}_r`: The r-th complex mode shape vector.
    - :math:`Re(\\vec{\\Phi}_r)`: Vector of real parts of :math:`\\vec{\\Phi}_r`.
    - :math:`Im(\\vec{\\Phi}_r)`: Vector of imaginary parts of :math:`\\vec{\\Phi}_r`.
    - :math:`|| ... ||`: Vector norm (Euclidean norm - 2-norm).

    Parameters
    ----------
    modeshape : np.ndarray
        Single modeshape [n_dof] or set of modeshapes [n_modes x n_dof].
        Can be real or complex-valued.

    Returns
    -------
    np.ndarray
        IPR value(s).
        - Scalar if input is a single modeshape [].
        - 1D array [n_modes] if input is a set of modeshapes.

    Raises
    ------
    TypeError
        If the input `modeshape` is not a NumPy array.
    NotImplementedError
        If the input `modeshape` has dimensions other than 1 or 2.
    """
    if not isinstance(modeshape, np.ndarray):
        raise TypeError("Input modeshape must be a NumPy array.")

    if modeshape.ndim == 1:
        real_part = modeshape.real
        imag_part = modeshape.imag
        denominator_norm = np.linalg.norm(real_part)
        numerator_norm = np.linalg.norm(imag_part)
        ipr_value = (
            numerator_norm / denominator_norm if denominator_norm != 0 else 0.0
        )  # Handle potential division by zero
        return np.array(ipr_value)  # Return as 0D array for consistency

    elif modeshape.ndim == 2:
        ipr_values = []
        for mode in modeshape:
            real_part = mode.real
            imag_part = mode.imag
            denominator_norm = np.linalg.norm(real_part)
            numerator_norm = np.linalg.norm(imag_part)
            ipr_value = (
                numerator_norm / denominator_norm if denominator_norm != 0 else 0.0
            )  # Handle potential division by zero
            ipr_values.append(ipr_value)
        return np.array(ipr_values)

    else:
        raise NotImplementedError(f"modeshape has dimensions: {modeshape.ndim}, expecting 1 or 2.")


def calculate_cf(modeshape: np.ndarray) -> np.ndarray:
    """Calculate Complexity Factor (CF) for a modeshape or set of modeshapes.

    CF is related to the Modal Phase Collinearity (MPC) and quantifies the degree of
    mode shape complexity. It's often defined as CF = 1 - MPC. A higher CF value
    (接近 1) indicates greater mode shape complexity, while a lower CF (接近 0)
    suggests a more 'real' mode shape.

    Math:
    -----
    .. math::
        CF_r = 1 - MPC_r = 1 - \\frac{ \\left| \\sum_{j=1}^{n} Re(\\Phi_{jr}) \\right|^2 }{ \\sum_{j=1}^{n} |\\Phi_{jr}|^2 }

    Where:
    - :math:`MPC_r`: Modal Phase Collinearity for mode r, calculated using `calculate_mpc`.

    Parameters
    ----------
    modeshape : np.ndarray
        Single modeshape [n_dof] or set of modeshapes [n_modes x n_dof].
        Can be real or complex-valued.

    Returns
    -------
    np.ndarray
        CF value(s).
        - Scalar if input is a single modeshape [].
        - 1D array [n_modes] if input is a set of modeshapes.

    Raises
    ------
    TypeError
        If the input `modeshape` is not a NumPy array.
    NotImplementedError
        If the input `modeshape` has dimensions other than 1 or 2.
    """
    if not isinstance(modeshape, np.ndarray):
        raise TypeError("Input modeshape must be a NumPy array.")

    mpc_values = calculate_mpc(modeshape)

    if modeshape.ndim == 1:
        return np.array(1.0 - mpc_values)
    elif modeshape.ndim == 2:
        return 1.0 - mpc_values
    else:
        raise NotImplementedError(f"modeshape has dimensions: {modeshape.ndim}, expecting 1 or 2.")


def calculate_mpd(modeshape: np.ndarray) -> np.ndarray:
    """Calculate Mean Phase Deviation (MPD) for a modeshape or set of modeshapes.

    MPD quantifies the phase scatter within a mode shape by measuring the average
    absolute deviation of each component's phase angle from the mean phase angle
    of the mode. Lower MPD indicates phases are clustered around the mean,
    suggesting more uniform phase behavior. Higher MPD indicates greater phase dispersion
    and more complex phase behavior across the mode shape.

    Math:
    -----
    .. math::
        MPD_r = \\frac{1}{n} \\sum_{j=1}^{n} |\\phi_{jr} - \\bar{\\phi}_r|

    Where:
    - :math:`\\phi_{jr}`: Phase angle (in radians) of the j-th component of the r-th complex mode shape.
    - :math:`\\bar{\\phi}_r`: Mean phase angle (in radians) for mode r: :math:`\\bar{\\phi}_r = (1/n) \\sum_{j=1}^{n} \\phi_{jr}`.
    - :math:`n`: Number of DOFs.
    - :math:`| ... |`: Absolute value.

    Parameters
    ----------
    modeshape : np.ndarray
        Single modeshape [n_dof] or set of modeshapes [n_modes x n_dof].
        Can be real or complex-valued.

    Returns
    -------
    np.ndarray
        MPD value(s) in degrees.
        - Scalar if input is a single modeshape [].
        - 1D array [n_modes] if input is a set of modeshapes.

    Raises
    ------
    TypeError
        If the input `modeshape` is not a NumPy array.
    NotImplementedError
        If the input `modeshape` has dimensions other than 1 or 2.
    """
    if not isinstance(modeshape, np.ndarray):
        raise TypeError("Input modeshape must be a NumPy array.")

    if modeshape.ndim == 1:
        phase_angles = np.angle(modeshape)  # radians
        mean_phase_angle = np.mean(phase_angles)
        deviations = np.abs(phase_angles - mean_phase_angle)
        mpd_value_radians = np.mean(deviations)
        mpd_value_degrees = np.degrees(mpd_value_radians)  # Convert to degrees for interpretability
        return np.array(mpd_value_degrees)  # Return as 0D array for consistency

    elif modeshape.ndim == 2:
        mpd_values_degrees = []
        for mode in modeshape:
            phase_angles = np.angle(mode)  # radians
            mean_phase_angle = np.mean(phase_angles)
            deviations = np.abs(phase_angles - mean_phase_angle)
            mpd_value_radians = np.mean(deviations)
            mpd_value_degrees = np.degrees(mpd_value_radians)  # Convert to degrees for interpretability
            mpd_values_degrees.append(mpd_value_degrees)
        return np.array(mpd_values_degrees)

    else:
        raise NotImplementedError(f"modeshape has dimensions: {modeshape.ndim}, expecting 1 or 2.")


def calculate_complexity_metrics(modeshape: np.ndarray) -> ComplexityMetrics:
    """Calculate all complexity metrics for a modeshape or set of modeshapes.

    This function computes all available complexity metrics:
    - Modal Phase Collinearity (MPC)
    - Modal Amplitude Proportionality (MAP)
    - Imaginary Part Ratio (IPR)
    - Complexity Factor (CF)
    - Mean Phase Deviation (MPD)

    Parameters
    ----------
    modeshape : np.ndarray
        Single modeshape [n_dof] or set of modeshapes [n_modes x n_dof].
        Can be real or complex-valued.

    Returns
    -------
    ComplexityMetrics
        Container with all computed complexity metrics.
        Each metric has shape [] for single modeshape or [n_modes] for multiple modeshapes.

    Raises
    ------
    ValueError
        If modeshape has incorrect dimensions
    """
    ModalValidator.validate(modeshape)

    # Calculate all metrics
    mpc_values = calculate_mpc(modeshape)
    map_values = calculate_map(modeshape)
    ipr_values = calculate_ipr(modeshape)
    cf_values = calculate_cf(modeshape)
    mpd_values = calculate_mpd(modeshape)

    # Return as ComplexityMetrics container
    return ComplexityMetrics(
        mpc=mpc_values,
        map=map_values,
        ipr=ipr_values,
        cf=cf_values,
        mpd=mpd_values,
    )
