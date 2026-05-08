# Pockmark Groundwater Model

2D MODFLOW-6 transient flow model and parameter sweep used to test whether
residual basal overpressure can sustain late-Holocene mud-only seafloor
fluidization at the New England Mud Patch (NEMP).

## Repository layout

```
src/                                       — Python source (importable from notebooks)
├── pockmark_model.py                      — flow model (run_simulation) and project-path constants
├── sweep_utils.py                         — shared HDF5 / sweep helpers
├── sweep_config.py                        — sweep axes, baseline, seal grid
└── run_sweeps.py                          — end-to-end CLI runner for both sweeps
notebooks/                                 — Jupyter notebooks
├── groundwater_model_30k.ipynb            — primary 4-parameter sweep (notebook front-end)
├── groundwater_geometry_sensitivity.ipynb — geometry & seal sensitivity at fixed baseline
├── groundwater_figures.ipynb              — figures and statistics from the main sweep
├── offset_overpressure_seal.ipynb         — offset-source / continuous-seal scenario
└── two_seal_comparison.ipynb              — one vs. two seal comparison
data/                                      — sweep outputs (HDF5 + CSV, LFS-tracked)
├── groundwater_sweep_profiles.h5
├── groundwater_sweep_summary.csv
├── geometry_sensitivity_profiles.h5
└── geometry_sensitivity_summary.csv
figures/                                   — output PNGs from the figure notebooks
bin/
└── mf6.exe                                — MODFLOW-6 binary (LFS-tracked)
```

`pockmark_model.py` exposes the resolved layout as `PROJECT_ROOT`,
`MF6_EXE`, `DATA_DIR`, and `FIGURES_DIR`. Notebooks add `<root>/src` to
`sys.path` in their first cell, so they work regardless of the kernel's
working directory as long as they live under `<root>/notebooks/`.

## What changed vs. the original

The original notebook bundled `run_simulation` into one cell and hardcoded
several geometric parameters that were not constrained by data. This version:

- Extracts the model into `pockmark_model.py` so it can be tested and reused.
- Adds five geometric parameters that were previously fixed: `Lx`, `mud_width`,
  `depth_to_source`, `ovp_lateral_extent`, and an optional intermediate seal
  layer (`seal_depth`, `seal_thickness`, `seal_k`).
- Splits the sweep into a primary sweep (`groundwater_model_30k`) and a
  geometry sensitivity sweep (`groundwater_geometry_sensitivity`) so the
  geometric uncertainty is reported as a separate, secondary table.
- Aligns the default `ss=1e-4` with the value actually used in the sweep.
- Replaces the stale 4-layer docstring with the actual 7-layer architecture.
- Reframes `k_sand` as the *effective bulk vertical permeability* of the
  column between the mud cap and the source, since the literal subsurface
  layering is unconstrained — see the methods note below.

## Running

There are two equivalent ways to drive the sweeps. The CLI runner is the
primary entry point; the matching notebooks are kept for interactive use
and produce identical outputs.

### CLI (recommended)

```bash
python src/run_sweeps.py                 # main + geometry sweeps + summary CSVs
python src/run_sweeps.py --main-only     # 4-parameter sweep only
python src/run_sweeps.py --geometry-only # geometry/seal sensitivity only
python src/run_sweeps.py --n-jobs 4      # override parallelism
```

Sweep axes, baseline, and seal grid live in `src/sweep_config.py` — edit
there to change the parameter ranges. Both sweeps resume from any
pre-existing HDF5 checkpoint in `data/`, so re-running is safe.

### Notebook front-ends

1. `notebooks/groundwater_model_30k.ipynb` → writes
   `data/groundwater_sweep_profiles.h5` and
   `data/groundwater_sweep_summary.csv`. The 4-axis grid as configured is
   6,435 runs (~hours on 8 cores).
2. `notebooks/groundwater_geometry_sensitivity.ipynb` → writes
   `data/geometry_sensitivity_profiles.h5` and
   `data/geometry_sensitivity_summary.csv`. A few dozen runs at a single
   baseline.
3. `notebooks/groundwater_figures.ipynb` → reads `data/*.h5` and
   `data/*.csv`, displays the interactive viewer, and writes static
   figures into `figures/`.
4. `notebooks/offset_overpressure_seal.ipynb` and
   `notebooks/two_seal_comparison.ipynb` → standalone single-scenario
   notebooks; their figures also land in `figures/`.

## Methods note: effective vertical permeability

The subsurface architecture between the mud cap and the seismically-imaged
overpressured zone is unconstrained from existing data at the NEMP. For the
1D-dominant transient diffusion problem solved here, an arbitrary heterogeneous
column collapses to a single effective vertical permeability (harmonic mean
weighted by thickness). The four-orders-of-magnitude `k_sand` sweep
(10⁻¹³ to 10⁻⁹ m²) is therefore wide enough to bracket the effective bulk
vertical permeability of any plausible heterogeneous post-glacial / shelf
sediment column at this site. The optional intermediate seal sensitivity
(`groundwater_geometry_sensitivity.ipynb`) addresses the one geometry that is
*not* captured by an effective-k argument: a thin, laterally continuous
low-permeability interbed acting as a second seal.

## Limitations

1. 2D plane-flow assumption — overestimates pressure retention vs. 3D drainage.
2. Heave gradient is a permissive screen, not a pockmark formation model;
   no fracture, gas, piping, or eruption physics.
3. Monotonic decay of basal overpressure — no recharge or repeated forcing.
4. Mud-body geometry is fixed in the main sweep and varied only in the
   secondary sensitivity notebook.
5. The success window (last 5 kyr exceedance for mud, >10 kyr quiet for sand)
   is chosen to match the late-Holocene persistence of pockmarks given NEMP
   sedimentation rates of ~1–3 mm/yr in the depocentre.

## Citing the input data

Pockmark distribution and surface mud thickness from
[Andrews et al., 2019, GRL](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019GL084881);
seismic section through the NEMP from
[Goff et al., 2015, G3](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2014GC005569).
