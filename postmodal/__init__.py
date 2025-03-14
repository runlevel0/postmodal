"""Modal analysis package for structural dynamics."""

from .types import (
    ModalData,
    ComplexityMetrics,
    Modeshape,
    Frequency,
    MACValue,
    ComplexityValue,
)
from .validation import ModalValidator, validate_modal_data
from .complexity import (
    calculate_mpc,
    calculate_map,
    calculate_ipr,
    calculate_cf,
    calculate_mpd,
    calculate_complexity_metrics,
)
from .comparison import calculate_mac
from .comparison.matrix import (
    calculate_mac_matrix,
    calculate_mode_matching_matrix,
    match_modes,
    best_match,
)
from .manipulation import (
    normalize_modeshape,
    normalize_modeshape_unit_norm_vector_length,
    normalize_modeshape_unit_norm_max_amplitude,
    normalize_modeshape_reference_dof,
    unwrap_modeshape_phase,
    wrap_modeshape_phase,
    cast_modeshape_to_real,
    cast_modeshape_to_real_average_phase_rotation,
    cast_modeshape_to_real_phase_corrected,
    cast_modeshape_to_real_real_part,
    find_best_fit_angle,
)
from .visualization import (
    plot_mac_matrix,
    plot_modeshape_complexity,
    plot_modeshape_complexity_grid,
)

__all__ = [
    # Types
    "ModalData",
    "ComplexityMetrics",
    "Modeshape",
    "Frequency",
    "MACValue",
    "ComplexityValue",
    # Validation
    "ModalValidator",
    "validate_modal_data",
    # Complexity metrics
    "calculate_mpc",
    "calculate_map",
    "calculate_ipr",
    "calculate_cf",
    "calculate_mpd",
    "calculate_complexity_metrics",
    # Comparison
    "calculate_mac",
    "calculate_mac_matrix",
    "calculate_mode_matching_matrix",
    "match_modes",
    "best_match",
    # Manipulation
    "normalize_modeshape",
    "normalize_modeshape_unit_norm_vector_length",
    "normalize_modeshape_unit_norm_max_amplitude",
    "normalize_modeshape_reference_dof",
    "unwrap_modeshape_phase",
    "wrap_modeshape_phase",
    "cast_modeshape_to_real",
    "cast_modeshape_to_real_average_phase_rotation",
    "cast_modeshape_to_real_phase_corrected",
    "cast_modeshape_to_real_real_part",
    "find_best_fit_angle",
    # Visualization
    "plot_mac_matrix",
    "plot_modeshape_complexity",
    "plot_modeshape_complexity_grid",
]
