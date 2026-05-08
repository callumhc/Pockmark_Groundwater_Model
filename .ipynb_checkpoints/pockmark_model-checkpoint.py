"""
Pockmark groundwater model — 2D MODFLOW-6 transient flow with parameter sweep.

The model tests whether residual basal overpressure under a mud cap can sustain
upward exit gradients exceeding the Terzaghi heave criterion in the mud zone
while the adjacent sand flanks have already discharged. This is used as a
permissive screen for late-Holocene mud-only pockmark formation at sites such
as the New England Mud Patch.

Geometry (default values reflect NEMP-like setting):
  - Domain width Lx (m)
  - Mud body width mud_width (m), centred in the domain
  - 5 near-surface sublayers of 2 m each (mud or sand depending on column)
  - Bulk sand column from -10 m down to depth_to_source
  - Optional intermediate seal layer of thickness seal_thickness centred at
    seal_depth, with permeability seal_k (set seal_thickness=0 to disable)
  - Basal overpressure source layer of thickness ovp_thickness, with high
    initial head over the central ovp_lateral_extent of the domain
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory

import flopy
import numpy as np
import pandas as pd

# Physical constants and grid defaults
G = 9.81
RHO_SEAWATER = 1024.0
SUBLAYER_THICK = 2.0
N_SUBLAYERS = 5
MUD_MAX_DEPTH = N_SUBLAYERS * SUBLAYER_THICK  # 10 m: deepest the mud body can reach
MUD_FLANK_HALF_WIDTH = 2500.0  # half-width of the sloping mud-body flank section

# Permeability conversion: k [m^2] -> K [m/yr] assuming standard water properties
PERM_FACTOR = 1e7 * 365.25 * 24 * 3600


def overpressure_to_head(overpressure_mpa: float) -> float:
    """Convert basal overpressure [MPa] to equivalent head [m] in seawater."""
    return overpressure_mpa / (RHO_SEAWATER * G) * 1e6


def critical_gradient(bulk_density: float) -> float:
    """Terzaghi heave / fluidization gradient for given bulk density [kg/m^3]."""
    return (bulk_density - RHO_SEAWATER) / RHO_SEAWATER


def _build_mud_base(ncol: int, delcol: float, Lx: float, mud_width: float) -> np.ndarray:
    """Return per-column depth of the base of the mud body [m, negative].

    Columns outside the mud body have mud_base = 0 (no mud). The mud body has
    a flat bottom at -MUD_MAX_DEPTH centred in the domain, with sloping flanks
    of horizontal half-width MUD_FLANK_HALF_WIDTH on each side.
    """
    x_centres = (np.arange(ncol) + 0.5) * delcol
    centre = Lx / 2.0
    half = mud_width / 2.0
    dist_from_centre = np.abs(x_centres - centre)

    mud_base = np.zeros(ncol)
    in_flat = dist_from_centre <= (half - MUD_FLANK_HALF_WIDTH)
    in_slope = (dist_from_centre > (half - MUD_FLANK_HALF_WIDTH)) & (dist_from_centre <= half)

    mud_base[in_flat] = -MUD_MAX_DEPTH
    if in_slope.any():
        slope = -MUD_MAX_DEPTH / MUD_FLANK_HALF_WIDTH
        # zero at the outer edge (dist = half), -MUD_MAX_DEPTH at the inner edge
        mud_base[in_slope] = slope * (half - dist_from_centre[in_slope])
    return mud_base


def _harmonic_k_for_sublayer(
    cell_top: float, cell_bot: float, mud_base: np.ndarray,
    k_mud: float, k_sand: float,
) -> np.ndarray:
    """Vertical harmonic-mean conductivity for one sublayer across all columns."""
    cell_thick = cell_top - cell_bot  # > 0
    # Mud thickness within this cell at each column: max(0, cell_top - max(mud_base, cell_bot))
    mud_thick = np.maximum(0.0, cell_top - np.maximum(mud_base, cell_bot))
    sand_thick = cell_thick - mud_thick

    k_row = np.full_like(mud_base, k_sand)
    pure_mud = sand_thick <= 0.0
    mixed = (mud_thick > 0.0) & ~pure_mud
    k_row[pure_mud] = k_mud
    k_row[mixed] = cell_thick / (mud_thick[mixed] / k_mud + sand_thick[mixed] / k_sand)
    return k_row


def run_simulation(
    # ── Time discretization ──────────────────────────────────────────────
    perlen_sub: float = 1.0,
    nstp_sub: int = 10,
    perlen_init: float = 4.0,
    nstp_init: int = 4,
    perlen_early: float = 990.0,
    nstp_early: int = 198,
    perlen_late: float = 29000.0,
    nstp_late: int = 290,
    # ── Boundary / source conditions ─────────────────────────────────────
    overpressure_mpa: float = 1.5,
    ovp_thickness: float = 1.0,
    ovp_lateral_extent: float | None = None,
    # ── Hydraulic properties ─────────────────────────────────────────────
    k_mud: float = 3.156e-2,
    k_sand: float = 3.156e1,
    bulkrho_mud: float = 1650.0,
    bulkrho_sand: float = 2150.0,
    ss: float = 1e-4,
    sy: float = 0.15,
    # ── Domain geometry ──────────────────────────────────────────────────
    Lx: float = 105_000.0,
    mud_width: float = 35_000.0,
    depth_to_source: float = 300.0,
    ncol: int = 5000,
    # ── Optional intermediate seal layer ─────────────────────────────────
    seal_depth: float = 150.0,
    seal_thickness: float = 0.0,
    seal_k: float | None = None,
    # ── Misc ─────────────────────────────────────────────────────────────
    modelname: str = "2D_gw_model",
):
    """Run a 2-D transient groundwater simulation and return spatial gradients.

    Parameters
    ----------
    overpressure_mpa : float
        Excess pressure (above hydrostatic) at the basal source [MPa].
    ovp_thickness : float
        Thickness of the basal overpressure source layer [m].
    ovp_lateral_extent : float, optional
        Width [m] of the central portion of the basal layer initialised at
        elevated head. If None, the entire basal layer is overpressured.
    k_mud, k_sand : float
        Hydraulic conductivities [m/yr]. Use PERM_FACTOR to convert from m^2.
    bulkrho_mud, bulkrho_sand : float
        Bulk sediment densities [kg/m^3] — used downstream for critical
        gradient comparison; not used inside the flow model itself.
    ss, sy : float
        Specific storage [1/m] and specific yield [-].
    Lx : float
        Total domain width [m].
    mud_width : float
        Width of the mud body, centred in the domain [m].
    depth_to_source : float
        Depth from seafloor to the top of the basal overpressure source [m].
    ncol : int
        Number of horizontal cells in the model grid.
    seal_depth : float
        Depth [m] from seafloor to the centre of an optional intermediate
        low-k seal layer within the bulk sand column. Only active if
        seal_thickness > 0.
    seal_thickness : float
        Thickness of the intermediate seal [m]. Set to 0 to disable.
    seal_k : float, optional
        Hydraulic conductivity of the intermediate seal [m/yr]. Defaults to
        k_mud if not provided.

    Returns
    -------
    results : pd.DataFrame
        Per-timestep summary of maximum surface gradients in mud and sand zones.
    crit_grad_mud, crit_grad_sand : float
    grad_spatial : ndarray
        Spatial profile of surface gradient at every timestep, shape (n_t, ncol).
    x_coords : ndarray
        Cell-centre x coordinates [m].
    in_mud_zone : ndarray
        Boolean mask (length ncol) of columns where the surface 2-m cell is
        wholly mud. Slope columns (mud cap thinner than the surface sublayer)
        are classified as sand here, since their surface cell is a harmonic
        blend rather than pure mud.
    """
    # ── Derived parameters ────────────────────────────────────────────────
    overpressure_head = overpressure_to_head(overpressure_mpa)
    crit_grad_mud = critical_gradient(bulkrho_mud)
    crit_grad_sand = critical_gradient(bulkrho_sand)
    if seal_k is None:
        seal_k = k_mud

    # ── Grid geometry ─────────────────────────────────────────────────────
    nrow = 1
    delrow = Lx / 3.0  # 2D plane flow: row width is irrelevant to gradients
    delcol = Lx / ncol
    ztop = 0.0

    # ── Layering ──────────────────────────────────────────────────────────
    # Layers 0..N_SUBLAYERS-1: surface 2 m sublayers
    # Layer N_SUBLAYERS    : bulk sand above seal (or full sand if no seal)
    # Layer N_SUBLAYERS+1  : seal (only if seal_thickness > 0)
    # Layer N_SUBLAYERS+2  : bulk sand below seal (only if seal_thickness > 0)
    # Final layer          : basal overpressure source
    use_seal = seal_thickness > 0
    src_top_z = -depth_to_source
    sand_top_z = -MUD_MAX_DEPTH

    if use_seal:
        seal_top = -(seal_depth - seal_thickness / 2.0)
        seal_bot = -(seal_depth + seal_thickness / 2.0)
        if not (sand_top_z > seal_top > seal_bot > src_top_z):
            raise ValueError(
                f"Seal at depth {seal_depth} m thickness {seal_thickness} m "
                f"does not fit between {sand_top_z} m and {src_top_z} m."
            )
        nlay = N_SUBLAYERS + 4   # sublayers + bulk-sand-top + seal + bulk-sand-bot + ovp
    else:
        nlay = N_SUBLAYERS + 2   # sublayers + bulk-sand + ovp

    botm = np.zeros((nlay, nrow, ncol))
    for sl in range(N_SUBLAYERS):
        botm[sl, 0, :] = -(sl + 1) * SUBLAYER_THICK
    if use_seal:
        botm[N_SUBLAYERS,     0, :] = seal_top
        botm[N_SUBLAYERS + 1, 0, :] = seal_bot
        botm[N_SUBLAYERS + 2, 0, :] = src_top_z
        botm[N_SUBLAYERS + 3, 0, :] = src_top_z - ovp_thickness
    else:
        botm[N_SUBLAYERS,     0, :] = src_top_z
        botm[N_SUBLAYERS + 1, 0, :] = src_top_z - ovp_thickness

    # ── Material assignment ───────────────────────────────────────────────
    mud_base = _build_mud_base(ncol, delcol, Lx, mud_width)
    k_layers = []
    for sl in range(N_SUBLAYERS):
        cell_top = -sl * SUBLAYER_THICK
        cell_bot = -(sl + 1) * SUBLAYER_THICK
        k_layers.append(_harmonic_k_for_sublayer(cell_top, cell_bot, mud_base, k_mud, k_sand))
    if use_seal:
        k_layers.append(np.full(ncol, k_sand))   # bulk sand above seal
        k_layers.append(np.full(ncol, seal_k))   # intermediate seal
        k_layers.append(np.full(ncol, k_sand))   # bulk sand below seal
    else:
        k_layers.append(np.full(ncol, k_sand))   # bulk sand
    k_layers.append(np.full(ncol, k_sand))       # overpressure source

    # ── Build MODFLOW-6 simulation ────────────────────────────────────────
    temp_dir = TemporaryDirectory(delete=False)
    workspace = Path(temp_dir.name)

    sim = flopy.mf6.MFSimulation(
        sim_name=modelname, exe_name="mf6.exe", sim_ws=workspace,
    )
    flopy.mf6.ModflowTdis(
        sim, pname="tdis", time_units="YEARS", nper=4,
        perioddata=[
            (perlen_sub,   nstp_sub,   1.0),
            (perlen_init,  nstp_init,  1.0),
            (perlen_early, nstp_early, 1.0),
            (perlen_late,  nstp_late,  1.0),
        ],
    )
    flopy.mf6.ModflowIms(
        sim, pname="ims", complexity="SIMPLE", linear_acceleration="BICGSTAB",
    )
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=modelname, model_nam_file=f"{modelname}.nam",
        save_flows=False, newtonoptions="NEWTON UNDER_RELAXATION",
    )
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=nlay, nrow=nrow, ncol=ncol,
        delr=delrow, delc=delcol, top=ztop, botm=botm,
    )

    # Initial heads: zero everywhere except the basal source layer.
    strt = np.zeros((nlay, nrow, ncol))
    src_layer_idx = nlay - 1
    if ovp_lateral_extent is None or ovp_lateral_extent >= Lx:
        strt[src_layer_idx, :, :] = overpressure_head
    else:
        x_centres = (np.arange(ncol) + 0.5) * delcol
        ovp_mask = np.abs(x_centres - Lx / 2.0) <= ovp_lateral_extent / 2.0
        strt[src_layer_idx, 0, ovp_mask] = overpressure_head

    flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=strt)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=k_layers)
    flopy.mf6.ModflowGwfsto(
        gwf, ss=ss, sy=sy, transient={i: True for i in range(4)},
    )

    # CHD: layer 0 across full domain fixed at h = 0 (seafloor).
    chd_rec = [((0, 0, col), 0.0) for col in range(ncol)]
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd_rec)

    headfile = f"{modelname}.hds"
    flopy.mf6.ModflowGwfoc(
        gwf, saverecord=[("HEAD", "ALL")],
        head_filerecord=[headfile], printrecord=[("HEAD", "LAST")],
    )

    with contextlib.redirect_stdout(io.StringIO()):
        sim.write_simulation()
        success, _ = sim.run_simulation(silent=True)
    if not success:
        raise RuntimeError("MODFLOW 6 did not terminate normally.")

    # ── Post-process: surface exit gradient ───────────────────────────────
    head_file = gwf.output.head()
    times = head_file.get_times()
    kstpkpers = head_file.get_kstpkper()

    # Mask of columns whose surface (top 2 m) cell is fully mud.
    in_mud_zone = k_layers[0] == k_mud

    records = [_zero_record()]
    grad_spatial = [np.zeros(ncol)]
    for t, ksp in zip(times, kstpkpers):
        h = head_file.get_data(kstpkper=ksp)
        grad = (h[1, 0, :] - h[0, 0, :]) / SUBLAYER_THICK
        records.append(_record_from_gradient(t, ksp, grad, in_mud_zone, crit_grad_mud, crit_grad_sand))
        grad_spatial.append(grad)

    results = pd.DataFrame(records)
    head_file.file.close()
    x_coords = (np.arange(ncol) + 0.5) * delcol

    try:
        temp_dir.cleanup()
    except Exception:
        pass

    return (
        results, crit_grad_mud, crit_grad_sand,
        np.array(grad_spatial), x_coords, in_mud_zone,
    )


def _zero_record() -> dict:
    """Initial-condition record (t=0, all heads hydrostatic so all gradients zero)."""
    return {
        "time": 0.0, "kstpkper": (-1, -1),
        "max_gradient_mud":      0.0, "exceeds_critical_mud":  False, "col_index_mud":  0,
        "max_gradient_sand":     0.0, "exceeds_critical_sand": False, "col_index_sand": 0,
        "overall_max_gradient":  0.0, "overall_max_col":       0,
    }


def _record_from_gradient(
    t: float, ksp, grad: np.ndarray, in_mud_zone: np.ndarray,
    crit_grad_mud: float, crit_grad_sand: float,
) -> dict:
    grad_mud  = np.where(in_mud_zone,  grad, 0.0)
    grad_sand = np.where(~in_mud_zone, grad, 0.0)
    max_mud, max_sand, max_all = float(grad_mud.max()), float(grad_sand.max()), float(grad.max())
    return {
        "time": t, "kstpkper": ksp,
        "max_gradient_mud":      max_mud,
        "exceeds_critical_mud":  max_mud  > crit_grad_mud,
        "col_index_mud":         int(grad_mud.argmax()),
        "max_gradient_sand":     max_sand,
        "exceeds_critical_sand": max_sand > crit_grad_sand,
        "col_index_sand":        int(grad_sand.argmax()),
        "overall_max_gradient":  max_all,
        "overall_max_col":       int(grad.argmax()),
    }
