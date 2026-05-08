"""
Sweep configuration — edit the values below to change parameter ranges.

This is a regular Python module, so you can use NumPy expressions
(`np.linspace`, `np.logspace`) and arithmetic freely. `run_sweeps.py`
imports from here at startup; passing `--config other_file.py` lets you
swap in an alternate config without modifying this one.

Required names (all module-level):
    MAIN_HDF5, MAIN_CSV
    MAIN_AXES, MAIN_FIXED
    BULKRHO_MUD_VALUES, BULKRHO_SAND_VALUES
    SENS_HDF5, SENS_CSV
    BASELINE, DEFAULT_GEOM
    GEOMETRY_PERTURBATIONS
    SEAL_DEPTHS_M, SEAL_THICKNESSES, SEAL_K_M2
    SUCCESS_MUD_RECENT_YR, SUCCESS_SAND_QUIET_YR
"""

import numpy as np

# Permeability conversion: k [m²] → K [m/yr]. Imported here so users can
# write `1e-19 * PERM_FACTOR` directly when specifying baseline values.
from pockmark_model import PERM_FACTOR


# ── Output files ────────────────────────────────────────────────────────────
MAIN_HDF5 = "groundwater_sweep_profiles.h5"
MAIN_CSV  = "groundwater_sweep_summary.csv"
SENS_HDF5 = "geometry_sensitivity_profiles.h5"
SENS_CSV  = "geometry_sensitivity_summary.csv"


# ── Main 4-parameter sweep ──────────────────────────────────────────────────
# k values are stored in m². The runner multiplies by PERM_FACTOR before
# calling run_simulation (which expects m/yr internally).
MAIN_AXES = {
    "overpressure_mpa": list(np.linspace(0.7, 2.4, 9)),
    "k_mud_m2":         np.unique(np.concatenate([
                            np.logspace(-14, -18, 5),
                            np.logspace(-18, -20, 9),
                        ])).tolist(),
    "k_sand_m2":        np.logspace(-9, -13, 5).tolist(),
    "ovp_thickness_m":  list(np.linspace(1, 150, 11)),
}

# Non-swept parameters held fixed across the main sweep
MAIN_FIXED = dict(
    Lx                 = 105_000.0,
    mud_width          = 35_000.0,
    depth_to_source    = 300.0,
    ovp_lateral_extent = None,
    seal_thickness     = 0.0,
    ss                 = 1e-4,
    sy                 = 0.15,
    bulkrho_mud        = 1650.0,
    bulkrho_sand       = 2150.0,
)

# Density combinations evaluated post-hoc for the main summary CSV
BULKRHO_MUD_VALUES  = [1300, 1350, 1400, 1450, 1500, 1550, 1600,
                       1650, 1700, 1750, 1800, 1850, 1900]
BULKRHO_SAND_VALUES = [1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200]


# ── Geometry sensitivity sweep ──────────────────────────────────────────────
# Baseline parameter combination (one point in the main sweep's space)
BASELINE = dict(
    overpressure_mpa = 1.55,
    k_mud            = 1e-19 * PERM_FACTOR,
    k_sand           = 1e-11 * PERM_FACTOR,
    ovp_thickness    = 30.0,
    bulkrho_mud      = 1650.0,
    bulkrho_sand     = 2150.0,
    ss               = 1e-4,
    sy               = 0.15,
)

# Default geometry — perturbations override one key at a time
DEFAULT_GEOM = dict(
    Lx                 = 105_000.0,
    mud_width          = 35_000.0,
    depth_to_source    = 300.0,
    ovp_lateral_extent = None,
    seal_thickness     = 0.0,
    seal_depth         = 150.0,
    seal_k             = None,
)

# One-axis-at-a-time perturbations
GEOMETRY_PERTURBATIONS = {
    "mud_width":          [15_000.0, 25_000.0, 35_000.0, 45_000.0, 60_000.0],
    "depth_to_source":    [150.0, 225.0, 300.0, 400.0, 500.0],
    "ovp_lateral_extent": [10_000.0, 25_000.0, 50_000.0, 75_000.0, None],
    "Lx":                 [70_000.0, 105_000.0, 140_000.0, 175_000.0],
}

# Seal: joint depth × thickness × k grid (k values in m²)
SEAL_DEPTHS_M    = [75.0, 150.0, 225.0]
SEAL_THICKNESSES = [5.0, 10.0]
SEAL_K_M2        = [1e-19, 1e-18, 1e-17]


# ── Success criterion (used by the geometry summary CSV) ────────────────────
# A run "succeeds" if the mud zone last exceeded the heave gradient within
# the most recent SUCCESS_MUD_RECENT_YR years AND the sand zone last
# exceeded it more than SUCCESS_SAND_QUIET_YR years ago (or never).
SUCCESS_MUD_RECENT_YR = 5_000
SUCCESS_SAND_QUIET_YR = 10_000
