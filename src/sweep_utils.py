"""Shared utilities for parameter sweeps and HDF5 storage."""

from __future__ import annotations

import os

import h5py
import numpy as np


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def build_sample_cols(
    x_coords: np.ndarray,
    centre_buffer_m: float = 25_000.0,
    flank_stride: int = 200,
    centre_stride: int = 10,
) -> np.ndarray:
    """Build a subsampling index keeping fine resolution in the centre and
    coarser sampling on the flanks.

    The centre region is `[centre_buffer_m, Lx - centre_buffer_m]`. Inside
    it, every `centre_stride`-th column is kept; outside, every
    `flank_stride`-th column.
    """
    Lx = x_coords[-1] + (x_coords[1] - x_coords[0]) / 2
    left = np.where(x_coords <  centre_buffer_m)[0]
    right = np.where(x_coords > (Lx - centre_buffer_m))[0]
    centre = np.where(
        (x_coords >= centre_buffer_m) & (x_coords <= (Lx - centre_buffer_m))
    )[0]
    return np.concatenate([left[::flank_stride], centre[::centre_stride], right[::flank_stride]])


def run_key_str(**kwargs) -> str:
    """Build a stable HDF5 group name from the parameter dict.

    Numeric values are formatted to fixed precision so floating-point noise
    never produces two different group names for the same nominal parameter
    combination. None values are encoded as the literal string 'none'.
    """
    parts = []
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if v is None:
            parts.append(f"{k}=none")
        elif isinstance(v, float):
            if abs(v) < 1e-3 or abs(v) >= 1e6:
                parts.append(f"{k}={v:.6e}")
            else:
                parts.append(f"{k}={v:.6f}")
        else:
            parts.append(f"{k}={v}")
    return "|".join(parts)


def load_done_keys(hdf5_file: str) -> set:
    """Return the set of run group names already present in the HDF5 file."""
    if not os.path.exists(hdf5_file):
        return set()
    try:
        with h5py.File(hdf5_file, "r") as hf:
            if "runs" in hf:
                return set(hf["runs"].keys())
    except OSError:
        pass
    return set()


def write_run_to_hdf5(
    hf, group_name: str, params: dict,
    times: np.ndarray, grads_subsampled: np.ndarray,
    x_coords: np.ndarray | None = None,
    in_mud_zone: np.ndarray | None = None,
) -> None:
    """Write one completed run to the open HDF5 file.

    `params` becomes group attributes (None values are converted to the
    sentinel -1.0 since HDF5 attrs cannot be None). If `x_coords` and/or
    `in_mud_zone` are provided, they are stored as per-run datasets — useful
    when geometry varies between runs (e.g. the geometry sensitivity sweep).
    """
    if "runs" not in hf:
        hf.create_group("runs")
    grp = hf["runs"].create_group(group_name)

    for k, v in params.items():
        grp.attrs[k] = -1.0 if v is None else v

    grp.create_dataset("time_yr", data=times.astype(np.float32), compression="gzip")
    try:
        grp.create_dataset(
            "gradients", data=grads_subsampled.astype(np.float16),
            compression="gzip", compression_opts=6,
        )
    except TypeError:
        grp.create_dataset(
            "gradients", data=grads_subsampled.astype(np.float32),
            compression="gzip", compression_opts=6,
        )
        grp.attrs["dtype_fallback"] = "float32"

    if x_coords is not None:
        grp.create_dataset("x_coords_m", data=x_coords.astype(np.float32), compression="gzip")
    if in_mud_zone is not None:
        grp.create_dataset("in_mud_zone", data=in_mud_zone, compression="gzip")

    hf.flush()
