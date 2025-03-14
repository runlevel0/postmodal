"""Phase manipulation utilities for modal analysis."""

from typing import Optional

import numpy as np

from ..validation import ModalValidator


def align_phase(
    modeshape: np.ndarray,
    reference_dof: Optional[int] = None,
) -> np.ndarray:
    """Align the phase of a modeshape to a reference DOF.

    This function aligns the phase of a modeshape by rotating it so that
    the phase at a reference DOF is zero. If no reference DOF is specified,
    the DOF with the largest magnitude is used.

    Parameters
    ----------
    modeshape : np.ndarray
        Complex modeshape array [n_dof]
    reference_dof : Optional[int], optional
        Index of the reference DOF, by default None

    Returns
    -------
    np.ndarray
        Phase-aligned modeshape [n_dof]

    Notes
    -----
    The phase alignment is achieved by multiplying the modeshape by
    :math:`e^{-i\\phi_{ref}}`, where :math:`\\phi_{ref}` is the phase
    angle at the reference DOF.
    """
    if reference_dof is None:
        # Use DOF with largest magnitude as reference
        reference_dof = np.argmax(np.abs(modeshape))

    # Get phase angle at reference DOF
    phase_ref = np.angle(modeshape[reference_dof])

    # Rotate modeshape to align phase
    return modeshape * np.exp(-1j * phase_ref)


def unwrap_phase(
    modeshape: np.ndarray,
    axis: Optional[int] = None,
) -> np.ndarray:
    """Unwrap the phase of a modeshape.

    This function unwraps the phase angles of a modeshape to ensure
    continuous phase variation. Phase unwrapping is useful for
    analyzing phase distributions and identifying phase jumps.

    Parameters
    ----------
    modeshape : np.ndarray
        Complex modeshape array [n_dof]
    axis : Optional[int], optional
        Axis along which to unwrap phase, by default None

    Returns
    -------
    np.ndarray
        Modeshape with unwrapped phase [n_dof]

    Notes
    -----
    Phase unwrapping is performed using numpy's unwrap function,
    which adds or subtracts 2π to ensure phase continuity.
    """
    phase = np.angle(modeshape)
    unwrapped_phase = np.unwrap(phase, axis=axis)
    return np.abs(modeshape) * np.exp(1j * unwrapped_phase)


def normalize_phase(
    modeshape: np.ndarray,
    method: str = "reference",
    reference_dof: Optional[int] = None,
) -> np.ndarray:
    """Normalize the phase of a modeshape.

    This function normalizes the phase of a modeshape using one of several methods:
    - "reference": Align phase to a reference DOF
    - "unwrap": Unwrap phase to ensure continuity
    - "both": Apply both reference alignment and unwrapping

    Parameters
    ----------
    modeshape : np.ndarray
        Complex modeshape array [n_dof]
    method : str, optional
        Phase normalization method, by default "reference"
    reference_dof : Optional[int], optional
        Index of the reference DOF for reference method, by default None

    Returns
    -------
    np.ndarray
        Phase-normalized modeshape [n_dof]

    Raises
    ------
    ValueError
        If method is not one of ["reference", "unwrap", "both"]
    """
    if method not in ["reference", "unwrap", "both"]:
        raise ValueError('method must be one of ["reference", "unwrap", "both"]')

    result = modeshape.copy()

    if method in ["reference", "both"]:
        result = align_phase(result, reference_dof)

    if method in ["unwrap", "both"]:
        result = unwrap_phase(result)

    return result


def calculate_phase_distribution(
    modeshape: np.ndarray,
    bins: int = 36,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the phase angle distribution of a modeshape.

    This function computes a histogram of phase angles in the modeshape,
    which can be useful for analyzing phase clustering and identifying
    dominant phase patterns.

    Parameters
    ----------
    modeshape : np.ndarray
        Complex modeshape array [n_dof]
    bins : int, optional
        Number of bins for the histogram, by default 36

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing (histogram, bin_edges)
        - histogram: Array of phase angle counts
        - bin_edges: Array of bin edges in radians
    """
    phase = np.angle(modeshape)
    hist, edges = np.histogram(phase, bins=bins, range=(-np.pi, np.pi))
    return hist, edges


def unwrap_modeshape_phase(modeshape: np.ndarray, reference_dof: Optional[int] = None) -> np.ndarray:
    """Unwrap the phase of a modeshape or set of modeshapes.

    This function unwraps the phase of complex modeshapes to ensure smooth phase transitions
    relative to a reference DOF. The unwrapping process removes artificial discontinuities
    in the phase angle that occur when the phase wraps from +π to -π or vice versa.

    Math:
    -----
    For a mode shape :math:`\\vec{\\Phi}_r`, the phase-unwrapped mode shape is:

    .. math::
        \\vec{\\Phi}_{r, unwrapped} = |\\vec{\\Phi}_r| e^{j\\theta_{unwrapped}}

    Where :math:`\\theta_{unwrapped}` is the unwrapped phase angle relative to the reference DOF.

    Parameters
    ----------
    modeshape : np.ndarray
        Single modeshape [n_dof] or set of modeshapes [n_modes x n_dof].
        Can be real or complex-valued.
    reference_dof : Optional[int], optional
        Index of the reference degree of freedom (0-indexed) or "max".
        If "max", the DOF with maximum amplitude will be used as reference.
        The phase at this DOF will be used as reference. By default None.

    Returns
    -------
    np.ndarray
        Phase-unwrapped modeshape(s) with the same shape as input.
        The phase will be continuous relative to the reference DOF.

    Raises
    ------
    ValueError
        If modeshape has incorrect dimensions or reference_dof is invalid
    """
    ModalValidator.validate(modeshape)

    # Handle the reference_dof parameter
    if reference_dof is None or isinstance(reference_dof, (int, np.integer)):
        use_max_amplitude = False
    elif isinstance(reference_dof, str) and reference_dof.lower() == "max":
        use_max_amplitude = True
    else:
        raise ValueError("reference_dof must be an integer or the string 'max'")

    if modeshape.ndim == 1:
        # Get magnitude and phase
        magnitude = np.abs(modeshape)
        phase = np.angle(modeshape)

        # Determine reference DOF
        if use_max_amplitude:
            reference_dof = np.argmax(magnitude)
        elif not 0 <= reference_dof < modeshape.shape[0]:
            raise ValueError(
                f"reference_dof {reference_dof} is out of bounds for modeshape of length {modeshape.shape[0]}"
            )

        # Unwrap phase relative to reference DOF
        ref_phase = phase[reference_dof]
        phase = np.unwrap(phase - ref_phase) + ref_phase

        # Return complex modeshape with unwrapped phase
        return magnitude * np.exp(1j * phase)

    # modeshape.ndim == 2
    # Process each mode
    unwrapped_modes = []
    for mode in modeshape:
        # Get magnitude and phase
        magnitude = np.abs(mode)
        phase = np.angle(mode)

        # Determine reference DOF for this mode
        if use_max_amplitude:
            mode_reference_dof = np.argmax(magnitude)
        else:
            mode_reference_dof = reference_dof
            if not 0 <= mode_reference_dof < mode.shape[0]:
                raise ValueError(
                    f"reference_dof {mode_reference_dof} is out of bounds for modeshape with {mode.shape[0]} DOFs"
                )

        # Unwrap phase relative to reference DOF
        ref_phase = phase[mode_reference_dof]
        phase = np.unwrap(phase - ref_phase) + ref_phase

        # Store unwrapped mode
        unwrapped_modes.append(magnitude * np.exp(1j * phase))

    return np.array(unwrapped_modes)


def wrap_modeshape_phase(modeshape: np.ndarray) -> np.ndarray:
    """Wrap the phase of a modeshape or set of modeshapes to [-π, π].

    This function wraps the phase angles of complex modeshapes to ensure they fall
    within the principal value range [-π, π]. This is useful when you want to
    standardize the phase representation or compare phases directly.

    Math:
    -----
    For a mode shape :math:`\\vec{\\Phi}_r`, the phase-wrapped mode shape is:

    .. math::
        \\vec{\\Phi}_{r, wrapped} = |\\vec{\\Phi}_r| e^{j\\theta_{wrapped}}

    Where :math:`\\theta_{wrapped}` is the wrapped phase angle in [-π, π].

    Parameters
    ----------
    modeshape : np.ndarray
        Single modeshape [n_dof] or set of modeshapes [n_modes x n_dof].
        Can be real or complex-valued.

    Returns
    -------
    np.ndarray
        Phase-wrapped modeshape(s) with the same shape as input.
        All phase angles will be in the range [-π, π].

    Raises
    ------
    ValueError
        If modeshape has incorrect dimensions
    """
    ModalValidator.validate(modeshape)

    if modeshape.ndim == 1:
        magnitude = np.abs(modeshape)
        phase = np.angle(modeshape)  # automatically wraps to [-π, π]
        return magnitude * np.exp(1j * phase)

    # modeshape.ndim == 2
    wrapped_modes = []
    for mode in modeshape:
        magnitude = np.abs(mode)
        phase = np.angle(mode)  # automatically wraps to [-π, π]
        wrapped_modes.append(magnitude * np.exp(1j * phase))

    return np.array(wrapped_modes)
