"""Phase manipulation utilities for modal analysis."""

from typing import Union

import numpy as np

from ..validation import ModalValidator


def unwrap_modeshape_phase(modeshape: np.ndarray, reference_dof: Union[int, str] = 0) -> np.ndarray:
    """Unwrap the phase of a modeshape or set of modeshapes.

    This function unwraps the phase of complex modeshapes to ensure smooth phase transitions
    relative to a reference DOF. The unwrapping process removes artificial discontinuities
    in the phase angle that occur when the phase wraps from +π to -π or vice versa.

    Math:
    -----
    For a mode shape :math:`\\vec{\Phi}_r`, the phase-unwrapped mode shape is:

    .. math::
        \\vec{\Phi}_{r, unwrapped} = |\\vec{\Phi}_r| e^{j\\theta_{unwrapped}}

    Where :math:`\\theta_{unwrapped}` is the unwrapped phase angle relative to the reference DOF.

    Parameters
    ----------
    modeshape : np.ndarray
        Single modeshape [n_dof] or set of modeshapes [n_modes x n_dof].
        Can be real or complex-valued.
    reference_dof : Union[int, str], optional
        Index of the reference degree of freedom (0-indexed) or "max".
        If "max", the DOF with maximum amplitude will be used as reference.
        The phase at this DOF will be used as reference. Defaults to 0.

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
    if isinstance(reference_dof, (int, np.integer)):
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
    For a mode shape :math:`\\vec{\Phi}_r`, the phase-wrapped mode shape is:

    .. math::
        \\vec{\Phi}_{r, wrapped} = |\\vec{\Phi}_r| e^{j\\theta_{wrapped}}

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
