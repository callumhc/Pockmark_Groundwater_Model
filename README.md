# Pockmark Groundwater Model — New England Mud Patch (NEMP)

**Author / maintainer:** Callum Hood-Cree
([@callumhc](https://github.com/callumhc))

Model code, input files, and sweep outputs accompanying:

> **Hood-Cree, C.** (mentored by B. Dugan). *Mud Permeability Enables Sustained
> Pockmark Formation in the New England Mud Patch: Evidence for Glaciogenic
> Overpressure from Sensitivity-Tested Flow Simulations.* Mines Undergraduate
> Research Fellowship / Geophysics Senior Design.

A 2D MODFLOW-6 transient flow model and parameter sweep that tests whether
residual basal overpressure consistent with glaciogenic emplacement ~17 ka is
physically sufficient to sustain mud-only seafloor fluidization at the NEMP up
to present day, while remaining below the critical gradient in the surrounding
sands. The primary sweep (~6,500 runs) varies mud permeability, sub-mud effective
bulk vertical permeability, basal overpressure magnitude, and overpressured zone
thickness; a secondary geometric sensitivity sweep varies mud body width, depth
to source, lateral overpressure extent, domain width, and intermediate confining
layer geometry.

Headline result: mud permeability is the dominant control on whether and when
fluidization occurs (Spearman ρ ≈ −0.96 against last-exceedance year in the
mud zone; ρ ≈ −0.99 in the sand zone). Sand permeability has negligible effect
across four orders of magnitude (|ρ| ≤ 0.02), confirming the mud cap acts as
the dominant hydraulic seal. Confining layer geometry provides a secondary
control whose influence diminishes after ~5 ka; cumulative seal thickness
governs the response, not the number of layers.

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

## Model summary

- 2D vertical cross-section, 105 km wide, 300 m deep below seafloor.
- 5,000 horizontal columns (Δx = 21 m) and 7 vertical layers (top five are
  2 m sublayers resolving the near-surface gradient; remaining layers represent
  the bulk sub-mud sediment column and basal overpressure source).
- Transient single-phase isothermal flow,
  *Sₛ ∂h/∂t = ∇·(K ∇h)*, with *Sₛ* = 1×10⁻⁴ m⁻¹ and seawater density
  *ρ*<sub>w</sub> = 1024 kg/m³.
- MODFLOW-6 via the FloPy interface, control-volume finite-difference;
  Newton-Raphson with under-relaxation for solver convergence
  (Bakker et al. references via FloPy/MODFLOW-6 documentation).
- Basal boundary initialised with elevated head consistent with glaciogenic
  emplacement at ~17 ka (0.7–2.4 MPa above hydrostatic across the sweep), at
  depths consistent with the seismic imaging of Siegel et al. (2014a). Seafloor
  held at hydrostatic; lateral boundaries no-flow.
- Total simulated time ~30 ka across four stress periods.
- Fluidization is screened via the Terzaghi heave criterion: the upward exit
  gradient between the top two sublayers is compared at each timestep against
  the critical gradient *i*<sub>c</sub> = (ρ<sub>b</sub> − ρ<sub>w</sub>)/ρ<sub>w</sub>,
  evaluated separately for mud and sand columns (Freeze & Cherry, 1979).

## Sweep design

Primary sweep (`groundwater_model_30k.ipynb` / `--main-only`):

| Parameter | Range |
|---|---|
| Mud permeability `k_mud` | 10⁻²⁰ to 10⁻¹⁴ m² (finer sampling below 10⁻¹⁸ m²) |
| Effective sub-mud permeability `k_sand` | 10⁻¹³ to 10⁻⁹ m² |
| Basal overpressure magnitude | 0.7 to 2.4 MPa |
| Overpressured zone thickness | 1 to 150 m |

Secondary geometric sensitivity sweep
(`groundwater_geometry_sensitivity.ipynb` / `--geometry-only`): mud body width,
depth to overpressure source, lateral overpressure extent, domain width, and
intermediate confining layer geometry, all at a fixed baseline drawn from the
admissible region of the primary sweep. Additional scenario notebooks examine
intermediate seal thickness (1–100 m), single vs. distributed seal
configurations (`two_seal_comparison.ipynb`), and a laterally offset
overpressure source (`offset_overpressure_seal.ipynb`).

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
overpressured zone (Siegel et al., 2014a) is unconstrained from existing data
at the NEMP. For the 1D-dominant transient diffusion problem solved here, an
arbitrary heterogeneous column collapses to a single effective vertical
permeability under harmonic averaging weighted by thickness. The
four-orders-of-magnitude `k_sand` sweep (10⁻¹³ to 10⁻⁹ m²) is therefore wide
enough to bracket the effective bulk vertical permeability of any plausible
heterogeneous post-glacial / shelf sediment column at this site. The optional
intermediate seal sensitivity (`groundwater_geometry_sensitivity.ipynb`)
addresses the one geometry that is *not* captured by an effective-*k* argument:
a thin, laterally continuous low-permeability interbed acting as a second seal.

## Limitations

1. 2D plane-flow assumption — overestimates pressure retention vs. 3D drainage.
2. The Terzaghi heave gradient is a permissive screen, not a pockmark
   formation model; no fracture, gas, piping, or eruption physics are simulated
   (cf. observed pockmark size and distribution reported by Goff, 2019).
3. Monotonic decay of an imposed initial basal overpressure — no dynamic
   glacial loading, recharge, or repeated forcing.
4. Mud-body geometry is fixed in the main sweep and varied only in the
   secondary sensitivity notebook.

## Notes on AI tooling

Portions of the code development and manuscript preparation were assisted by
AI language models (Claude Opus 4 and Claude Sonnet 4.6, Anthropic). All
scientific content, model design, parameter choices, interpretation of results,
and conclusions are the work of the author. AI tools were used under the
author's direction for code structuring and document drafting assistance only.

## References

Freeze, R. A., & Cherry, J. A. (1979). *Groundwater.* Prentice-Hall, Englewood Cliffs, NJ.

Goff, J. A. (2019). Modern and fossil pockmarks in the New England Mud Patch:
Implications for submarine groundwater discharge on the middle shelf.
*Geophysical Research Letters,* 46, 12213–12220.
<https://doi.org/10.1029/2019GL084881>

Goff, J. A., Reed, A. H., Gawarkiewicz, G., Wilson, P. S., & Knobles, D. P.
(2019). Stratigraphic analysis of a sediment pond within the New England Mud
Patch: New constraints from high-resolution chirp acoustic reflection data.
*Marine Geology,* 412, 81–94.
<https://doi.org/10.1016/j.margeo.2019.03.010>

Gustafson, C., Key, K., & Evans, R. L. (2019). Aquifer systems extending far
offshore on the U.S. Atlantic margin. *Scientific Reports,* 9, 8709.
<https://doi.org/10.1038/s41598-019-44611-7>

Person, M., Dugan, B., Swenson, J. B., Urbano, L., Stott, C., Taylor, J., &
Willett, M. (2003). Pleistocene hydrogeology of the Atlantic continental shelf,
New England. *Geological Society of America Bulletin,* 115(11), 1324–1343.
<https://doi.org/10.1130/B25285.1>

Siegel, J., Lizarralde, D., Dugan, B., Person, M., DeFoor, W., Gable, C., &
Miller, N. (2014a). Glacially generated overpressure on the New England
continental shelf: Integration of full-waveform inversion and overpressure
modeling. *Journal of Geophysical Research: Solid Earth,* 119, 3393–3409.
<https://doi.org/10.1002/2013JB010278>

Siegel, J., Person, M., Dugan, B., Cohen, D., Lizarralde, D., & Gable, C.
(2014b). Influence of late Pleistocene glaciations on the hydrogeology of the
continental shelf offshore Massachusetts, USA. *Geochemistry, Geophysics,
Geosystems,* 15, 4651–4670.
<https://doi.org/10.1002/2014GC005569>
