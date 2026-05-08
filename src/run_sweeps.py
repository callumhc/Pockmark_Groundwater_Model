#!/usr/bin/env python3
"""
End-to-end sweep runner.

Runs both the main 4-parameter sweep and the geometry / seal sensitivity sweep
in sequence, then writes the corresponding summary CSVs. Both sweeps resume
from any pre-existing HDF5 checkpoint, so re-running the script is safe and
will only execute runs that aren't already on disk.

Sweep parameters live in `sweep_config.py` — edit there to change axes,
baseline, or seal grid. Pass `--config other_file.py` to use a different
config without modifying the default.

Usage
-----
    python run_sweeps.py                      # run everything (default)
    python run_sweeps.py --main-only          # only the 4-parameter sweep
    python run_sweeps.py --geometry-only      # only the geometry sensitivity
    python run_sweeps.py --n-jobs 4           # override parallelism
    python run_sweeps.py --no-csv             # skip the CSV export step
    python run_sweeps.py --config alt.py      # use a different config file
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import sys
import time
import warnings
from pathlib import Path
from types import ModuleType

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pockmark_model import PERM_FACTOR, run_simulation
from sweep_utils import (
    build_sample_cols, fmt_duration, load_done_keys,
    run_key_str, write_run_to_hdf5,
)

warnings.filterwarnings("ignore")

RHO_SEAWATER = 1024.0


# ════════════════════════════════════════════════════════════════════════════
# Config loading
# ════════════════════════════════════════════════════════════════════════════

REQUIRED_CONFIG_NAMES = [
    "MAIN_HDF5", "MAIN_CSV",
    "MAIN_AXES", "MAIN_FIXED",
    "BULKRHO_MUD_VALUES", "BULKRHO_SAND_VALUES",
    "SENS_HDF5", "SENS_CSV",
    "BASELINE", "DEFAULT_GEOM",
    "GEOMETRY_PERTURBATIONS",
    "SEAL_DEPTHS_M", "SEAL_THICKNESSES", "SEAL_K_M2",
    "SUCCESS_MUD_RECENT_YR", "SUCCESS_SAND_QUIET_YR",
]

REQUIRED_MAIN_AXES = {"overpressure_mpa", "k_mud_m2", "k_sand_m2", "ovp_thickness_m"}


def load_config(path: str) -> ModuleType:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        sys.exit(f"Config file not found: {p}")
    spec = importlib.util.spec_from_file_location("sweep_config_user", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    missing = [n for n in REQUIRED_CONFIG_NAMES if not hasattr(mod, n)]
    if missing:
        sys.exit(f"Config {p} is missing required names: {', '.join(missing)}")
    missing_axes = REQUIRED_MAIN_AXES - set(mod.MAIN_AXES.keys())
    if missing_axes:
        sys.exit(f"Config MAIN_AXES is missing keys: {', '.join(sorted(missing_axes))}")
    print(f"Loaded config: {p}")
    return mod


# ════════════════════════════════════════════════════════════════════════════
# Main sweep
# ════════════════════════════════════════════════════════════════════════════

def _main_key(op: float, km_m2: float, ks_m2: float, ot: float) -> str:
    """Legacy key format: matches the names used by the original notebook so
    pre-existing HDF5 groups are still recognised on resume."""
    return f"{op:.6f}_{km_m2:.6e}_{ks_m2:.6e}_{ot:.6f}"


def _main_run_one(op, km_m2, ks_m2, ot, fixed):
    try:
        res, _cm, _cs, grad_sp, x_c, in_mud = run_simulation(
            overpressure_mpa = op,
            k_mud            = km_m2 * PERM_FACTOR,
            k_sand           = ks_m2 * PERM_FACTOR,
            ovp_thickness    = ot,
            **fixed,
        )
        return (op, km_m2, ks_m2, ot), res, grad_sp, x_c, in_mud, None
    except Exception as e:
        return (op, km_m2, ks_m2, ot), None, None, None, None, str(e)


def run_main_sweep(cfg: ModuleType, n_jobs: int) -> tuple[dict, dict]:
    runs = list(itertools.product(
        cfg.MAIN_AXES["overpressure_mpa"],
        cfg.MAIN_AXES["k_mud_m2"],
        cfg.MAIN_AXES["k_sand_m2"],
        cfg.MAIN_AXES["ovp_thickness_m"],
    ))
    total = len(runs)
    done = load_done_keys(cfg.MAIN_HDF5)
    todo = [r for r in runs if _main_key(*r) not in done]
    print(f"[main] {len(done)}/{total} already done, {len(todo)} remaining.")

    completed: dict = {}
    failed: dict = {}
    if not todo:
        return completed, failed

    sample_cols = None
    start = time.time()
    with h5py.File(cfg.MAIN_HDF5, "a") as hf:
        if "runs" not in hf:
            hf.attrs["description"]  = "Main parameter sweep — gradient profiles"
            hf.attrs["rho_seawater"] = RHO_SEAWATER
            hf.create_group("runs")
        for n, result in enumerate(
            Parallel(n_jobs=n_jobs, backend="loky", verbose=0, return_as="generator")(
                delayed(_main_run_one)(*r, cfg.MAIN_FIXED) for r in todo
            ), start=1
        ):
            (op, km, ks, ot), res, grad_sp, x_c, in_mud, err = result
            if err is not None:
                failed[(op, km, ks, ot)] = err
                continue
            if sample_cols is None:
                sample_cols = build_sample_cols(x_c)
                if "x_coords_m" not in hf:
                    hf.create_dataset("x_coords_m",  data=x_c[sample_cols].astype(np.float32), compression="gzip")
                if "in_mud_zone" not in hf:
                    hf.create_dataset("in_mud_zone", data=in_mud[sample_cols],                  compression="gzip")
            params = {
                "overpressure_mpa": op,
                "k_mud_m2":         km,
                "k_sand_m2":        ks,
                "ovp_thickness_m":  ot,
            }
            write_run_to_hdf5(
                hf, _main_key(op, km, ks, ot), params,
                res["time"].values,
                np.asarray(grad_sp)[:, sample_cols],
            )
            completed[(op, km, ks, ot)] = res
            if n % 10 == 0 or n == len(todo):
                elapsed = time.time() - start
                rate    = n / elapsed if elapsed > 0 else 0
                eta     = (len(todo) - n) / rate if rate > 0 else float("inf")
                eta_str = fmt_duration(eta) if eta < 1e8 else "…"
                print(f"  [main] {len(done) + n}/{total}  elapsed {fmt_duration(elapsed)}  ETA {eta_str}")
    print(f"[main] done. successful {len(completed)}, failed {len(failed)}.")
    return completed, failed


def build_main_summary_csv(cfg: ModuleType) -> None:
    print(f"[main] building {cfg.MAIN_CSV} from {cfg.MAIN_HDF5} …")

    def exceed_stats(grad_ts, crit, times):
        mask = grad_ts > crit
        t_exc = times[mask]
        return {
            "any_exceed":         bool(mask.any()),
            "first_exceed_yr":    float(t_exc[0])  if len(t_exc) else float("nan"),
            "last_exceed_yr":     float(t_exc[-1]) if len(t_exc) else float("nan"),
            "n_timesteps_exceed": int(mask.sum()),
        }

    rows = []
    with h5py.File(cfg.MAIN_HDF5, "r") as hf:
        in_mud = hf["in_mud_zone"][:]
        for grp_name in hf["runs"]:
            g = hf["runs"][grp_name]
            op    = float(g.attrs["overpressure_mpa"])
            km_m2 = float(g.attrs["k_mud_m2"])
            ks_m2 = float(g.attrs["k_sand_m2"])
            ot    = float(g.attrs["ovp_thickness_m"])
            times = g["time_yr"][:]
            grads = g["gradients"][:].astype(np.float32)
            g_mud  = np.where(in_mud,  grads, 0.0).max(axis=1)
            g_sand = np.where(~in_mud, grads, 0.0).max(axis=1)
            base = {
                "overpressure_mpa":      op,
                "k_mud_m2":              km_m2,
                "k_sand_m2":             ks_m2,
                "ovp_thickness_m":       ot,
                "peak_gradient_mud":     float(g_mud.max()),
                "peak_gradient_sand":    float(g_sand.max()),
                "peak_gradient_overall": float(grads.max()),
                "n_timesteps":           len(times),
                "time_start_yr":         float(times[0]),
                "time_end_yr":           float(times[-1]),
            }
            for rho_m in cfg.BULKRHO_MUD_VALUES:
                cm = (rho_m - RHO_SEAWATER) / RHO_SEAWATER
                m_stats = exceed_stats(g_mud, cm, times)
                for rho_s in cfg.BULKRHO_SAND_VALUES:
                    cs = (rho_s - RHO_SEAWATER) / RHO_SEAWATER
                    s_stats = exceed_stats(g_sand, cs, times)
                    rows.append({
                        **base,
                        "bulkrho_mud_kg_m3":  rho_m,
                        "bulkrho_sand_kg_m3": rho_s,
                        "crit_grad_mud":      cm,
                        "crit_grad_sand":     cs,
                        "any_exceed_mud":         m_stats["any_exceed"],
                        "first_exceed_yr_mud":    m_stats["first_exceed_yr"],
                        "last_exceed_yr_mud":     m_stats["last_exceed_yr"],
                        "n_timesteps_exceed_mud": m_stats["n_timesteps_exceed"],
                        "any_exceed_sand":         s_stats["any_exceed"],
                        "first_exceed_yr_sand":    s_stats["first_exceed_yr"],
                        "last_exceed_yr_sand":     s_stats["last_exceed_yr"],
                        "n_timesteps_exceed_sand": s_stats["n_timesteps_exceed"],
                    })
    df = pd.DataFrame(rows)
    df.to_csv(cfg.MAIN_CSV, index=False)
    print(f"[main] wrote {len(df):,} rows → {cfg.MAIN_CSV}")


# ════════════════════════════════════════════════════════════════════════════
# Geometry sensitivity sweep
# ════════════════════════════════════════════════════════════════════════════

def _build_geometry_runs(cfg: ModuleType) -> list:
    runs = []
    for axis, values in cfg.GEOMETRY_PERTURBATIONS.items():
        for v in values:
            runs.append((axis, {**cfg.DEFAULT_GEOM, axis: v}))
    for sd, st, sk_m2 in itertools.product(cfg.SEAL_DEPTHS_M, cfg.SEAL_THICKNESSES, cfg.SEAL_K_M2):
        runs.append(("seal", {
            **cfg.DEFAULT_GEOM,
            "seal_depth":     sd,
            "seal_thickness": st,
            "seal_k":         sk_m2 * PERM_FACTOR,
        }))
    return runs


def _geom_key(axis: str, geom: dict) -> str:
    return run_key_str(axis=axis, **geom)


def _geom_run_one(axis, geom, baseline):
    try:
        res, _cm, _cs, grad_sp, x_c, in_mud = run_simulation(**baseline, **geom)
        return axis, geom, res, grad_sp, x_c, in_mud, None
    except Exception as e:
        return axis, geom, None, None, None, None, str(e)


def run_geometry_sweep(cfg: ModuleType, n_jobs: int) -> tuple[int, dict]:
    runs = _build_geometry_runs(cfg)
    done = load_done_keys(cfg.SENS_HDF5)
    todo = [r for r in runs if _geom_key(*r) not in done]
    print(f"[geometry] {len(done)}/{len(runs)} already done, {len(todo)} remaining.")

    n_success = 0
    failed: dict = {}
    if not todo:
        return n_success, failed

    start = time.time()
    with h5py.File(cfg.SENS_HDF5, "a") as hf:
        if "runs" not in hf:
            hf.attrs["description"] = "Geometry / seal sensitivity sweep at fixed baseline"
            for k, v in cfg.BASELINE.items():
                hf.attrs[f"baseline_{k}"] = v
            hf.create_group("runs")
        for n, result in enumerate(
            Parallel(n_jobs=n_jobs, backend="loky", verbose=0, return_as="generator")(
                delayed(_geom_run_one)(*r, cfg.BASELINE) for r in todo
            ), start=1
        ):
            axis, geom, res, grad_sp, x_c, in_mud, err = result
            if err is not None:
                failed[(axis, _geom_key(axis, geom))] = err
                continue
            sample_cols = build_sample_cols(x_c)
            write_run_to_hdf5(
                hf, _geom_key(axis, geom), {"axis": axis, **geom},
                res["time"].values,
                np.asarray(grad_sp)[:, sample_cols],
                x_coords    = x_c[sample_cols],
                in_mud_zone = in_mud[sample_cols],
            )
            n_success += 1
            elapsed = time.time() - start
            print(f"  [geometry] [{n}/{len(todo)}] {axis}  elapsed {fmt_duration(elapsed)}")
    print(f"[geometry] done. successful {n_success}, failed {len(failed)}.")
    return n_success, failed


def build_geometry_summary_csv(cfg: ModuleType) -> None:
    print(f"[geometry] building {cfg.SENS_CSV} from {cfg.SENS_HDF5} …")
    crit_mud  = (cfg.BASELINE["bulkrho_mud"]  - RHO_SEAWATER) / RHO_SEAWATER
    crit_sand = (cfg.BASELINE["bulkrho_sand"] - RHO_SEAWATER) / RHO_SEAWATER
    recent = cfg.SUCCESS_MUD_RECENT_YR
    quiet  = cfg.SUCCESS_SAND_QUIET_YR

    rows = []
    with h5py.File(cfg.SENS_HDF5, "r") as hf:
        for grp_name in hf["runs"]:
            g = hf["runs"][grp_name]
            times  = g["time_yr"][:]
            grads  = g["gradients"][:].astype(np.float32)
            in_mud = g["in_mud_zone"][:]
            g_mud  = np.where(in_mud,  grads, 0.0).max(axis=1)
            g_sand = np.where(~in_mud, grads, 0.0).max(axis=1)
            em_t = times[g_mud  > crit_mud]
            es_t = times[g_sand > crit_sand]
            last_mud  = float(em_t[-1])  if len(em_t) else float("nan")
            last_sand = float(es_t[-1])  if len(es_t) else float("nan")
            t_end     = float(times[-1])
            success = (
                (not np.isnan(last_mud))
                and last_mud >= t_end - recent
                and (np.isnan(last_sand) or last_sand <= t_end - quiet)
            )
            rows.append({
                "group":               grp_name,
                "success":             success,
                "peak_grad_mud":       float(g_mud.max()),
                "peak_grad_sand":      float(g_sand.max()),
                "last_exceed_mud_yr":  last_mud,
                "last_exceed_sand_yr": last_sand,
                **{k: g.attrs[k] for k in g.attrs},
            })
    df = pd.DataFrame(rows)
    df.to_csv(cfg.SENS_CSV, index=False)
    print(f"[geometry] wrote {len(df)} rows → {cfg.SENS_CSV}")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    default_config = Path(__file__).resolve().parent / "sweep_config.py"
    parser.add_argument("--config",          default=str(default_config),
                        help=f"Path to a Python config file "
                             f"(default: sweep_config.py next to run_sweeps.py).")
    parser.add_argument("--main-only",       action="store_true",
                        help="run only the main 4-parameter sweep")
    parser.add_argument("--geometry-only",   action="store_true",
                        help="run only the geometry sensitivity sweep")
    parser.add_argument("--no-csv",          action="store_true",
                        help="skip the CSV-export step")
    parser.add_argument("--n-jobs", type=int, default=8,
                        help="parallel workers for the main sweep (default: 8)")
    parser.add_argument("--geometry-n-jobs", type=int, default=4,
                        help="parallel workers for the geometry sweep (default: 4)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    do_main     = not args.geometry_only
    do_geometry = not args.main_only

    t0 = time.time()
    if do_main:
        run_main_sweep(cfg, args.n_jobs)
        if not args.no_csv:
            build_main_summary_csv(cfg)
    if do_geometry:
        run_geometry_sweep(cfg, args.geometry_n_jobs)
        if not args.no_csv:
            build_geometry_summary_csv(cfg)

    print(f"\nTotal wall time: {fmt_duration(time.time() - t0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
