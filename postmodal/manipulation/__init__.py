"""Manipulation submodule for modal analysis.

.. note::
   The :func:`normalize_modeshape` function is deprecated and will be removed in a future version.
   Use :func:`normalize_modeshape_unit_norm_vector_length` instead.
"""

from .normalize import (
    normalize_modeshape,  # deprecated
    normalize_modeshape_unit_norm_vector_length,
    normalize_modeshape_unit_norm_max_amplitude,
    normalize_modeshape_reference_dof,
)
from .phase import (
    unwrap_modeshape_phase,
    wrap_modeshape_phase,
)
from .complex_to_real import (
    cast_modeshape_to_real,
    cast_modeshape_to_real_average_phase_rotation,
    cast_modeshape_to_real_phase_corrected,
    cast_modeshape_to_real_real_part,
    find_best_fit_angle,
)

__all__ = [
    # Normalization
    "normalize_modeshape",  # deprecated
    "normalize_modeshape_unit_norm_vector_length",
    "normalize_modeshape_unit_norm_max_amplitude",
    "normalize_modeshape_reference_dof",
    # Phase manipulation
    "unwrap_modeshape_phase",
    "wrap_modeshape_phase",
    # Complex to real conversion
    "cast_modeshape_to_real",
    "cast_modeshape_to_real_average_phase_rotation",
    "cast_modeshape_to_real_phase_corrected",
    "cast_modeshape_to_real_real_part",
    "find_best_fit_angle",
]
