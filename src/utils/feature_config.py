"""
Shared feature configuration for battery dataset.
Ensures consistency across data loading, model training, and evaluation.
"""

from typing import Dict, List
import torch

# ========== INPUT FEATURES (15) ==========
INPUT_FEATURES = [
    "SEI_kinetic_rate",
    "electrolyte_diffusivity",
    "initial_concentration",
    "separator_porosity",
    "positive_particle_radius",
    "negative_particle_radius",
    "positive_electrode_thickness",
    "negative_electrode_thickness",
    "SEI_solvent_diffusivity",
    "dead_lithium_decay",
    "lithium_plating_kinetic",
    "negative_LAM_constant",
    "negative_cracking_rate",
    "SEI_molar_volume",
    "SEI_activation_energy",
]

# ========== MICROSTRUCTURE OUTPUTS (4) ==========
MICROSTRUCTURE_FEATURES = [
    "D_eff",
    "porosity_measured",
    "tau_factor",
    "bruggeman_derived",
]

# Short names for logging
MICROSTRUCTURE_SHORT = ["D_eff", "porosity", "tau", "bruggeman"]

# ========== PERFORMANCE OUTPUTS (5) ==========
PERFORMANCE_FEATURES = [
    "projected_cycle_life",
    "capacity_fade_rate",
    "internal_resistance",
    "nominal_capacity",
    "energy_density",
]

# Short names for logging
PERFORMANCE_SHORT = ["cycle_life", "fade_rate", "DCIR", "capacity", "energy"]

# ========== DEFAULT LOSS WEIGHTS ==========
DEFAULT_PERFORMANCE_WEIGHTS = {
    "projected_cycle_life": 1.0,
    "capacity_fade_rate": 0.8,
    "internal_resistance": 0.5,
    "nominal_capacity": 0.3,
    "energy_density": 0.2,
}

DEFAULT_MICROSTRUCTURE_WEIGHTS = {
    "D_eff": 0.2,
    "porosity_measured": 0.2,
    "tau_factor": 0.3,
    "bruggeman_derived": 0.1,
}

# ========== FEATURE INDICES ==========
MICROSTRUCTURE_INDICES = {
    "D_eff": 0,
    "porosity_measured": 1,
    "tau_factor": 2,
    "bruggeman_derived": 3,
}

PERFORMANCE_INDICES = {
    "projected_cycle_life": 0,
    "capacity_fade_rate": 1,
    "internal_resistance": 2,
    "nominal_capacity": 3,
    "energy_density": 4,
}

# ========== DIMENSIONS ==========
NUM_INPUT_FEATURES = len(INPUT_FEATURES)
NUM_MICROSTRUCTURE_OUTPUTS = len(MICROSTRUCTURE_FEATURES)
NUM_PERFORMANCE_OUTPUTS = len(PERFORMANCE_FEATURES)


# ========== HELPER FUNCTIONS ==========
def get_feature_name(feature_type: str, index: int) -> str:
    """Get feature name from index"""
    if feature_type == "microstructure":
        return MICROSTRUCTURE_SHORT[index]
    elif feature_type == "performance":
        return PERFORMANCE_SHORT[index]
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")


def get_all_feature_names() -> Dict[str, List[str]]:
    """Get all feature names"""
    return {
        "input": INPUT_FEATURES,
        "microstructure": MICROSTRUCTURE_FEATURES,
        "microstructure_short": MICROSTRUCTURE_SHORT,
        "performance": PERFORMANCE_FEATURES,
        "performance_short": PERFORMANCE_SHORT,
    }


def get_loss_weight_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get loss weights as tensors in correct order.

    Returns:
        (perf_weights, micro_weights): Both as torch.Tensor
    """
    perf_weights = torch.tensor(
        [DEFAULT_PERFORMANCE_WEIGHTS[feat] for feat in PERFORMANCE_FEATURES],
        dtype=torch.float32,
    )

    micro_weights = torch.tensor(
        [DEFAULT_MICROSTRUCTURE_WEIGHTS[feat] for feat in MICROSTRUCTURE_FEATURES],
        dtype=torch.float32,
    )

    return perf_weights, micro_weights


def get_loss_weight_lists() -> tuple[list, list]:
    """
    Get loss weights as lists (for config compatibility).

    Returns:
        (perf_weights, micro_weights): Both as lists
    """
    perf_weights = [DEFAULT_PERFORMANCE_WEIGHTS[feat] for feat in PERFORMANCE_FEATURES]
    micro_weights = [
        DEFAULT_MICROSTRUCTURE_WEIGHTS[feat] for feat in MICROSTRUCTURE_FEATURES
    ]
    return perf_weights, micro_weights
