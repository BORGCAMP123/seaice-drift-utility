"""Raw-product reconstruction kernels extracted from the final experiment implementation.

The numerical functions below retain the project's published source/QC, interpolation,
Gaussian hierarchy, strict support, evaluation, and decomposition semantics.
The CLI and event manifest live outside this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]
_EASE_RADIUS_M = 6_371_228.0
_WGS84_A_M = 6_378_137.0
_WGS84_INV_F = 298.257223563
_EPSG3413_LON0_DEG = -45.0
_EPSG3413_LAT_TS_DEG = 70.0
FWHM_TO_SIGMA = 2.35482
IDENTITY_EPSILON_MULTIPLIER = 128.0


def direction(value: float) -> str:
    """Return the exact effect direction without a tuned tolerance."""
    if value > 0.0:
        return "+"
    if value < 0.0:
        return "-"
    return "0"


@dataclass(frozen=True)
class Grid:
    x_min: float
    y_min: float
    resolution_m: float
    width: int
    height: int

    @property
    def size(self) -> int:
        return self.width * self.height


def lonlat_to_epsg3408(longitude_deg: Any, latitude_deg: Any) -> tuple[FloatArray, FloatArray]:
    """Project lon/lat to the spherical NSIDC EASE-Grid North (EPSG:3408)."""
    longitude = np.asarray(longitude_deg, dtype=np.float64)
    latitude = np.asarray(latitude_deg, dtype=np.float64)
    if np.any(~np.isfinite(longitude)) or np.any(~np.isfinite(latitude)):
        raise ValueError("Longitude and latitude must be finite")
    if np.any((latitude < -90.0) | (latitude > 90.0)):
        raise ValueError("Latitude must be within [-90, 90]")
    lon = np.deg2rad(longitude)
    lat = np.deg2rad(latitude)
    rho = 2.0 * _EASE_RADIUS_M * np.sin(np.pi / 4.0 - lat / 2.0)
    return rho * np.sin(lon), -rho * np.cos(lon)


def _polar_t(latitude_rad: FloatArray, eccentricity: float) -> FloatArray:
    sine = np.sin(latitude_rad)
    ratio = (1.0 + eccentricity * sine) / (1.0 - eccentricity * sine)
    return np.tan(np.pi / 4.0 - latitude_rad / 2.0) * np.power(ratio, eccentricity / 2.0)


def lonlat_to_epsg3413(longitude_deg: Any, latitude_deg: Any) -> tuple[FloatArray, FloatArray]:
    """Project lon/lat to WGS84 NSIDC Sea Ice Polar Stereographic North."""
    longitude = np.asarray(longitude_deg, dtype=np.float64)
    latitude = np.asarray(latitude_deg, dtype=np.float64)
    if np.any(~np.isfinite(longitude)) or np.any(~np.isfinite(latitude)):
        raise ValueError("Longitude and latitude must be finite")
    if np.any((latitude < -90.0) | (latitude > 90.0)):
        raise ValueError("Latitude must be within [-90, 90]")

    flattening = 1.0 / _WGS84_INV_F
    eccentricity = np.sqrt(flattening * (2.0 - flattening))
    lat = np.deg2rad(latitude)
    lat_ts = np.deg2rad(_EPSG3413_LAT_TS_DEG)
    lon_delta = np.deg2rad(longitude - _EPSG3413_LON0_DEG)
    t = _polar_t(lat, eccentricity)
    t_c = float(_polar_t(np.asarray(lat_ts), eccentricity))
    m_c = np.cos(lat_ts) / np.sqrt(1.0 - eccentricity**2 * np.sin(lat_ts) ** 2)
    rho = _WGS84_A_M * m_c * t / t_c
    return rho * np.sin(lon_delta), -rho * np.cos(lon_delta)


def _comparison_grid(window: Mapping[str, float], resolution_m: float) -> Grid:
    samples = np.linspace(0.0, 1.0, 257)
    west, east = window["west"], window["east"]
    south, north = window["south"], window["north"]
    longitude = np.concatenate(
        (
            west + (east - west) * samples,
            west + (east - west) * samples,
            np.full(samples.shape, west),
            np.full(samples.shape, east),
        )
    )
    latitude = np.concatenate(
        (
            np.full(samples.shape, south),
            np.full(samples.shape, north),
            south + (north - south) * samples,
            south + (north - south) * samples,
        )
    )
    x, y = lonlat_to_epsg3413(longitude, latitude)
    x_min = float(np.floor(np.min(x) / resolution_m) * resolution_m)
    x_max = float(np.ceil(np.max(x) / resolution_m) * resolution_m)
    y_min = float(np.floor(np.min(y) / resolution_m) * resolution_m)
    y_max = float(np.ceil(np.max(y) / resolution_m) * resolution_m)
    return Grid(
        x_min=x_min,
        y_min=y_min,
        resolution_m=resolution_m,
        width=int(round((x_max - x_min) / resolution_m)),
        height=int(round((y_max - y_min) / resolution_m)),
    )


def longitude_window_mask(
    longitude_deg: Any, west_deg: float, east_deg: float
) -> NDArray[np.bool_]:
    """Return an inclusive longitude-window mask in a convention-independent way."""
    longitude_360 = np.mod(np.asarray(longitude_deg, dtype=np.float64), 360.0)
    if abs(float(east_deg) - float(west_deg)) >= 360.0:
        return np.ones(longitude_360.shape, dtype=np.bool_)
    west_360 = float(west_deg) % 360.0
    east_360 = float(east_deg) % 360.0
    if west_360 <= east_360:
        return (longitude_360 >= west_360) & (longitude_360 <= east_360)
    return (longitude_360 >= west_360) | (longitude_360 <= east_360)


def _read_vnp29_window(
    path: Path, window: Mapping[str, float]
) -> tuple[dict[str, IntArray | FloatArray], dict[str, int]]:
    chunks: dict[str, list[NDArray[Any]]] = {
        "class": [],
        "lon": [],
        "lat": [],
        "source_x": [],
        "source_y": [],
        "target_x": [],
        "target_y": [],
    }
    counts = {
        "window_centers": 0,
        "clear_source_candidates": 0,
        "cloud": 0,
        "land": 0,
        "inland_water": 0,
        "bowtie_trim": 0,
        "other_or_invalid": 0,
    }
    with netCDF4.Dataset(path, mode="r") as dataset:
        dataset.set_auto_maskandscale(False)
        for variable in dataset.variables.values():
            variable.set_auto_maskandscale(False)
        geo = dataset.groups["GeolocationData"]
        science = dataset.groups["SeaIceCoverData"]
        latitude_variable = geo.variables["latitude"]
        longitude_variable = geo.variables["longitude"]
        cover_variable = science.variables["SeaIceCover"]
        line_count = latitude_variable.shape[0]
        for start in range(0, line_count, 256):
            stop = min(start + 256, line_count)
            latitude = np.asarray(latitude_variable[start:stop, :], dtype=np.float64)
            longitude = np.asarray(longitude_variable[start:stop, :], dtype=np.float64)
            cover = np.asarray(cover_variable[start:stop, :], dtype=np.int64)
            geolocation_valid = (
                np.isfinite(latitude)
                & np.isfinite(longitude)
                & (latitude >= -90.0)
                & (latitude <= 90.0)
            )
            inside = (
                geolocation_valid
                & (latitude >= window["south"])
                & (latitude <= window["north"])
                & longitude_window_mask(longitude, window["west"], window["east"])
            )
            counts["window_centers"] += int(np.count_nonzero(inside))
            counts["cloud"] += int(np.count_nonzero(inside & (cover == 250)))
            counts["land"] += int(np.count_nonzero(inside & (cover == 225)))
            counts["inland_water"] += int(np.count_nonzero(inside & (cover == 237)))
            counts["bowtie_trim"] += int(np.count_nonzero(inside & (cover == 253)))
            clear = inside & np.isin(cover, [0, 1])
            counts["clear_source_candidates"] += int(np.count_nonzero(clear))
            counts["other_or_invalid"] += int(
                np.count_nonzero(inside & ~np.isin(cover, [0, 1, 225, 237, 250, 253]))
            )
            if not np.any(clear):
                continue
            selected_lon = longitude[clear]
            selected_lat = latitude[clear]
            source_x, source_y = lonlat_to_epsg3408(selected_lon, selected_lat)
            target_x, target_y = lonlat_to_epsg3413(selected_lon, selected_lat)
            chunks["class"].append(cover[clear].astype(np.int64))
            chunks["lon"].append(selected_lon)
            chunks["lat"].append(selected_lat)
            chunks["source_x"].append(source_x)
            chunks["source_y"].append(source_y)
            chunks["target_x"].append(target_x)
            chunks["target_y"].append(target_y)
    if not chunks["class"]:
        raise ValueError(f"No clear ice/water source candidates in {path.name}")
    arrays: dict[str, IntArray | FloatArray] = {}
    for name, parts in chunks.items():
        if name == "class":
            arrays[name] = np.concatenate(parts).astype(np.int64)
        else:
            arrays[name] = np.concatenate(parts).astype(np.float64)
    return arrays, counts


def _rasterize(
    classes: NDArray[np.int64], x: FloatArray, y: FloatArray, grid: Grid
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    column = np.floor((x - grid.x_min) / grid.resolution_m).astype(np.int64)
    row = np.floor((y - grid.y_min) / grid.resolution_m).astype(np.int64)
    inside = (column >= 0) & (column < grid.width) & (row >= 0) & (row < grid.height)
    index = row[inside] * grid.width + column[inside]
    counts = np.bincount(index, minlength=grid.size).astype(np.int32)
    ice = np.bincount(index, weights=classes[inside], minlength=grid.size)
    fraction = np.full(grid.size, np.nan, dtype=np.float32)
    valid = counts > 0
    fraction[valid] = (ice[valid] / counts[valid]).astype(np.float32)
    return fraction.reshape(grid.height, grid.width), counts.reshape(grid.height, grid.width)


@dataclass(frozen=True)
class AffineGrid:
    """Affine mapping from SID-CV array indices to EPSG:3413 metres."""

    x0: float
    x_i: float
    x_j: float
    y0: float
    y_i: float
    y_j: float
    rms_residual_m: float
    max_residual_m: float


@dataclass(frozen=True)
class SidcvField:
    """One valid, projected SID-CV displacement field."""

    dx_m: FloatArray
    dy_m: FloatArray
    valid: BoolArray
    affine: AffineGrid


def compute_alpha(
    viirs_source: datetime,
    viirs_target: datetime,
    sid_start: datetime,
    sid_end: datetime,
) -> float:
    """Return the explicit VIIRS-interval/SID-interval displacement scale."""
    viirs_seconds = (viirs_target - viirs_source).total_seconds()
    sid_seconds = (sid_end - sid_start).total_seconds()
    if viirs_seconds <= 0.0 or sid_seconds <= 0.0:
        raise ValueError("VIIRS and SID-CV intervals must both be positive")
    return float(viirs_seconds / sid_seconds)


def east_north_to_epsg3413(
    longitude_deg: Any,
    latitude_deg: Any,
    east_m: Any,
    north_m: Any,
    *,
    step_m: float = 100.0,
) -> tuple[FloatArray, FloatArray]:
    """Re-express local east/north displacement in the EPSG:3413 basis.

    A centered WGS84 local-tangent Jacobian is used.  The input components and
    returned components are displacements in metres; no time scaling occurs here.
    """
    lon = np.asarray(longitude_deg, dtype=np.float64)
    lat = np.asarray(latitude_deg, dtype=np.float64)
    east = np.asarray(east_m, dtype=np.float64)
    north = np.asarray(north_m, dtype=np.float64)
    lon, lat, east, north = np.broadcast_arrays(lon, lat, east, north)
    if not np.isfinite(step_m) or step_m <= 0.0:
        raise ValueError("step_m must be positive and finite")
    if np.any(np.isfinite(lat) & ((lat <= -90.0) | (lat >= 90.0))):
        raise ValueError("finite latitude must lie strictly between the poles")

    flattening = 1.0 / _WGS84_INV_F
    eccentricity_sq = flattening * (2.0 - flattening)
    latitude_rad = np.deg2rad(lat)
    sine = np.sin(latitude_rad)
    denominator = np.sqrt(1.0 - eccentricity_sq * sine**2)
    prime_vertical_radius = _WGS84_A_M / denominator
    meridional_radius = _WGS84_A_M * (1.0 - eccentricity_sq) / denominator**3
    delta_lon_deg = np.rad2deg(step_m / (prime_vertical_radius * np.cos(latitude_rad)))
    delta_lat_deg = np.rad2deg(step_m / meridional_radius)

    east_plus = lonlat_to_epsg3413(lon + delta_lon_deg, lat)
    east_minus = lonlat_to_epsg3413(lon - delta_lon_deg, lat)
    north_plus = lonlat_to_epsg3413(lon, lat + delta_lat_deg)
    north_minus = lonlat_to_epsg3413(lon, lat - delta_lat_deg)
    j11 = (east_plus[0] - east_minus[0]) / (2.0 * step_m)
    j21 = (east_plus[1] - east_minus[1]) / (2.0 * step_m)
    j12 = (north_plus[0] - north_minus[0]) / (2.0 * step_m)
    j22 = (north_plus[1] - north_minus[1]) / (2.0 * step_m)
    return (
        np.asarray(j11 * east + j12 * north, dtype=np.float64),
        np.asarray(j21 * east + j22 * north, dtype=np.float64),
    )


def fit_affine_sid_grid(x_m: Any, y_m: Any, *, maximum_samples_per_axis: int = 65) -> AffineGrid:
    """Fit and audit the regular rotated SID-CV grid in EPSG:3413."""
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or min(x.shape) < 2:
        raise ValueError("x_m and y_m must be matching two-dimensional grids")
    i_values = np.unique(
        np.linspace(0, x.shape[0] - 1, min(x.shape[0], maximum_samples_per_axis)).round()
    ).astype(np.int64)
    j_values = np.unique(
        np.linspace(0, x.shape[1] - 1, min(x.shape[1], maximum_samples_per_axis)).round()
    ).astype(np.int64)
    ii, jj = np.meshgrid(i_values, j_values, indexing="ij")
    observed_x = x[ii, jj].ravel()
    observed_y = y[ii, jj].ravel()
    design = np.column_stack((np.ones(ii.size, dtype=np.float64), ii.ravel(), jj.ravel()))
    finite = np.isfinite(observed_x) & np.isfinite(observed_y)
    if np.count_nonzero(finite) < 6:
        raise ValueError("insufficient finite SID-CV coordinates for affine fit")
    coef_x = np.linalg.lstsq(design[finite], observed_x[finite], rcond=None)[0]
    coef_y = np.linalg.lstsq(design[finite], observed_y[finite], rcond=None)[0]
    predicted_x = design[finite] @ coef_x
    predicted_y = design[finite] @ coef_y
    residual = np.hypot(predicted_x - observed_x[finite], predicted_y - observed_y[finite])
    return AffineGrid(
        x0=float(coef_x[0]),
        x_i=float(coef_x[1]),
        x_j=float(coef_x[2]),
        y0=float(coef_y[0]),
        y_i=float(coef_y[1]),
        y_j=float(coef_y[2]),
        rms_residual_m=float(np.sqrt(np.mean(residual**2))),
        max_residual_m=float(np.max(residual)),
    )


def interpolate_affine_bilinear(
    field: Any,
    field_valid: Any,
    affine: AffineGrid,
    query_x_m: Any,
    query_y_m: Any,
) -> tuple[FloatArray, BoolArray]:
    """Bilinearly interpolate a rotated regular grid with four-corner QC."""
    values = np.asarray(field, dtype=np.float64)
    valid_grid = np.asarray(field_valid, dtype=bool)
    qx = np.asarray(query_x_m, dtype=np.float64)
    qy = np.asarray(query_y_m, dtype=np.float64)
    qx, qy = np.broadcast_arrays(qx, qy)
    if values.shape != valid_grid.shape or values.ndim != 2:
        raise ValueError("field and field_valid must be matching two-dimensional grids")
    transform = np.asarray([[affine.x_i, affine.x_j], [affine.y_i, affine.y_j]], dtype=np.float64)
    determinant = float(np.linalg.det(transform))
    if not np.isfinite(determinant) or abs(determinant) < 1e-9:
        raise ValueError("SID-CV affine grid is singular")
    inverse = np.linalg.inv(transform)
    relative = np.vstack(((qx - affine.x0).ravel(), (qy - affine.y0).ravel()))
    fractional = inverse @ relative
    fi = fractional[0].reshape(qx.shape)
    fj = fractional[1].reshape(qx.shape)
    i0 = np.floor(fi).astype(np.int64)
    j0 = np.floor(fj).astype(np.int64)
    i0 = np.where(np.isclose(fi, values.shape[0] - 1, atol=1e-7), values.shape[0] - 2, i0)
    j0 = np.where(np.isclose(fj, values.shape[1] - 1, atol=1e-7), values.shape[1] - 2, j0)
    inside = (
        np.isfinite(fi)
        & np.isfinite(fj)
        & (i0 >= 0)
        & (j0 >= 0)
        & (i0 < values.shape[0] - 1)
        & (j0 < values.shape[1] - 1)
    )
    safe_i = np.clip(i0, 0, values.shape[0] - 2)
    safe_j = np.clip(j0, 0, values.shape[1] - 2)
    corners = np.stack(
        (
            values[safe_i, safe_j],
            values[safe_i + 1, safe_j],
            values[safe_i, safe_j + 1],
            values[safe_i + 1, safe_j + 1],
        )
    )
    corner_valid = np.stack(
        (
            valid_grid[safe_i, safe_j],
            valid_grid[safe_i + 1, safe_j],
            valid_grid[safe_i, safe_j + 1],
            valid_grid[safe_i + 1, safe_j + 1],
        )
    )
    valid = inside & np.all(corner_valid & np.isfinite(corners), axis=0)
    wi = fi - safe_i
    wj = fj - safe_j
    result = (
        corners[0] * (1.0 - wi) * (1.0 - wj)
        + corners[1] * wi * (1.0 - wj)
        + corners[2] * (1.0 - wi) * wj
        + corners[3] * wi * wj
    )
    return np.asarray(np.where(valid, result, np.nan)), np.asarray(valid, dtype=bool)


def load_sidcv_field(path: Path) -> tuple[SidcvField, dict[str, Any]]:
    """Read, QC and re-express one SID-CV file without time scaling."""
    required = (
        "SeaIce_U",
        "SeaIce_V",
        "lon",
        "lat",
        "time",
        "time_bnds",
        "rejection_flag",
        "qc_flag",
    )
    with netCDF4.Dataset(path, mode="r") as dataset:
        missing = [name for name in required if name not in dataset.variables]
        if missing:
            raise ValueError(f"missing SID-CV variables: {missing}")
        lon = np.asarray(np.ma.filled(dataset.variables["lon"][:], np.nan), dtype=np.float64)
        lat = np.asarray(np.ma.filled(dataset.variables["lat"][:], np.nan), dtype=np.float64)
        east = np.asarray(np.ma.filled(dataset.variables["SeaIce_U"][:], np.nan), dtype=np.float64)
        north = np.asarray(np.ma.filled(dataset.variables["SeaIce_V"][:], np.nan), dtype=np.float64)
        rejection = np.asarray(
            np.ma.filled(dataset.variables["rejection_flag"][:], 255), dtype=np.int16
        )
        qc = np.asarray(np.ma.filled(dataset.variables["qc_flag"][:], 255), dtype=np.int16)
        valid = (
            np.isfinite(lon)
            & np.isfinite(lat)
            & np.isfinite(east)
            & np.isfinite(north)
            & (rejection == 0)
            & np.isin(qc, [0, 1])
        )
        x, y = lonlat_to_epsg3413(lon, lat)
        affine = fit_affine_sid_grid(x, y)
        if affine.max_residual_m > 5.0:
            raise ValueError(
                f"SID-CV grid is not affine in EPSG:3413: {affine.max_residual_m:.3f} m"
            )
        safe_east = np.where(valid, east, 0.0)
        safe_north = np.where(valid, north, 0.0)
        dx, dy = east_north_to_epsg3413(lon, lat, safe_east, safe_north)
        dx = np.where(valid, dx, np.nan)
        dy = np.where(valid, dy, np.nan)
        metadata = {
            "time_coverage_start": str(dataset.time_coverage_start),
            "time_coverage_end": str(dataset.time_coverage_end),
            "grid_resolution_dX": str(dataset.grid_resolution_dX),
            "grid_resolution_dY": str(dataset.grid_resolution_dY),
            "valid_native_pixels": int(np.count_nonzero(valid)),
            "affine_rms_residual_m": affine.rms_residual_m,
            "affine_max_residual_m": affine.max_residual_m,
        }
    return SidcvField(dx, dy, valid, affine), metadata


def _splat_fraction(
    classes: NDArray[np.int64],
    moved_x: FloatArray,
    moved_y: FloatArray,
    valid: BoolArray,
    grid: Grid,
) -> tuple[NDArray[np.float32], NDArray[np.int32], int]:
    finite = valid & np.isfinite(moved_x) & np.isfinite(moved_y)
    column = np.full(classes.shape, -1, dtype=np.int64)
    row = np.full(classes.shape, -1, dtype=np.int64)
    column[finite] = np.floor((moved_x[finite] - grid.x_min) / grid.resolution_m).astype(np.int64)
    row[finite] = np.floor((moved_y[finite] - grid.y_min) / grid.resolution_m).astype(np.int64)
    inside = finite & (column >= 0) & (column < grid.width) & (row >= 0) & (row < grid.height)
    index = row[inside] * grid.width + column[inside]
    counts = np.bincount(index, minlength=grid.size).astype(np.int32)
    ice = np.bincount(index, weights=classes[inside], minlength=grid.size)
    fraction = np.full(grid.size, np.nan, dtype=np.float32)
    occupied = counts > 0
    fraction[occupied] = (ice[occupied] / counts[occupied]).astype(np.float32)
    return (
        fraction.reshape(grid.height, grid.width),
        counts.reshape(grid.height, grid.width),
        int(np.count_nonzero(inside)),
    )


def transport_with_sidcv(
    source: dict[str, NDArray[Any]],
    *,
    grid: Grid,
    sidcv: SidcvField,
    alpha: float,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Scale SID-CV displacement and forward hard-splat clear VIIRS classes."""
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be positive and finite")
    classes = np.asarray(source["class"], dtype=np.int64)
    source_x = np.asarray(source["target_x"], dtype=np.float64)
    source_y = np.asarray(source["target_y"], dtype=np.float64)
    dx, dx_valid = interpolate_affine_bilinear(
        sidcv.dx_m, sidcv.valid, sidcv.affine, source_x, source_y
    )
    dy, dy_valid = interpolate_affine_bilinear(
        sidcv.dy_m, sidcv.valid, sidcv.affine, source_x, source_y
    )
    valid = dx_valid & dy_valid
    moved_x = source_x + alpha * dx
    moved_y = source_y + alpha * dy
    prediction, counts, transported = _splat_fraction(classes, moved_x, moved_y, valid, grid)
    magnitude = np.hypot(alpha * dx[valid], alpha * dy[valid])
    return prediction, {
        "source_clear_pixels": int(classes.size),
        "valid_sidcv_source_pixels": int(np.count_nonzero(valid)),
        "transported_source_pixels": transported,
        "occupied_prediction_cells": int(np.count_nonzero(counts)),
        "scaled_displacement_median_m": (float(np.median(magnitude)) if magnitude.size else None),
        "scaled_displacement_p95_m": (
            float(np.percentile(magnitude, 95.0)) if magnitude.size else None
        ),
    }


def aggregate_common_support(
    fields: dict[str, NDArray[Any]], common: Any, factor: int
) -> tuple[dict[str, NDArray[np.float32]], NDArray[np.float32], NDArray[np.int32]]:
    """Area-average common 1 km support into nested coarse cells."""
    if factor < 1:
        raise ValueError("factor must be at least one")
    mask = np.asarray(common, dtype=bool)
    if mask.ndim != 2 or any(np.asarray(value).shape != mask.shape for value in fields.values()):
        raise ValueError("all fields and the common mask must share a 2-D shape")
    height, width = mask.shape
    out_h = (height + factor - 1) // factor
    out_w = (width + factor - 1) // factor
    pad_h = out_h * factor - height
    pad_w = out_w * factor - width
    padded_mask = np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=False)
    capacity = (
        np.pad(np.ones(mask.shape, dtype=np.int32), ((0, pad_h), (0, pad_w)), constant_values=0)
        .reshape(out_h, factor, out_w, factor)
        .sum(axis=(1, 3))
    )
    counts = padded_mask.reshape(out_h, factor, out_w, factor).sum(axis=(1, 3)).astype(np.int32)
    support_rate = np.divide(
        counts,
        capacity,
        out=np.zeros(counts.shape, dtype=np.float32),
        where=capacity > 0,
    )
    aggregated: dict[str, NDArray[np.float32]] = {}
    for name, value in fields.items():
        array = np.asarray(value, dtype=np.float64)
        weighted = np.where(mask, array, 0.0)
        padded = np.pad(weighted, ((0, pad_h), (0, pad_w)), constant_values=0.0)
        sums = padded.reshape(out_h, factor, out_w, factor).sum(axis=(1, 3))
        mean = np.full(counts.shape, np.nan, dtype=np.float32)
        occupied = counts > 0
        mean[occupied] = (sums[occupied] / counts[occupied]).astype(np.float32)
        aggregated[name] = mean
    return aggregated, support_rate, counts


def masked_gaussian_lowpass(
    values: Any,
    valid: Any,
    *,
    sigma_pixels: float,
    minimum_valid_weight: float,
    truncate: float,
) -> tuple[FloatArray, BoolArray, FloatArray]:
    """Gaussian low-pass with explicit missingness normalization and support QC."""
    array = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(array)
    if array.ndim != 2 or mask.shape != array.shape:
        raise ValueError("values and valid must be matching two-dimensional arrays")
    if sigma_pixels <= 0.0 or not np.isfinite(sigma_pixels):
        raise ValueError("sigma_pixels must be positive and finite")
    if not 0.0 < minimum_valid_weight <= 1.0:
        raise ValueError("minimum_valid_weight must lie in (0, 1]")
    weights = gaussian_filter(
        mask.astype(np.float64), sigma=sigma_pixels, mode="constant", cval=0.0, truncate=truncate
    )
    numerator = gaussian_filter(
        np.where(mask, array, 0.0),
        sigma=sigma_pixels,
        mode="constant",
        cval=0.0,
        truncate=truncate,
    )
    result = np.full(array.shape, np.nan, dtype=np.float64)
    local_valid = mask & (weights >= minimum_valid_weight)
    np.divide(numerator, weights, out=result, where=local_valid)
    return result, np.asarray(local_valid), np.asarray(weights)


@dataclass(frozen=True)
class GaussianSidcvHierarchy:
    """Direct-from-native Gaussian SID-CV levels on six-way common validity."""

    fields: dict[str, SidcvField]
    weighted_support: dict[str, FloatArray]
    audit: list[dict[str, Any]]
    common_native_valid: BoolArray


def gaussian_fwhm_to_sigma_pixels(fwhm_km: float, native_spacing_m: float) -> float:
    """Convert a Gaussian FWHM in kilometres to sigma on the native grid."""
    if fwhm_km <= 0.0 or not np.isfinite(fwhm_km):
        raise ValueError("fwhm_km must be positive and finite")
    if native_spacing_m <= 0.0 or not np.isfinite(native_spacing_m):
        raise ValueError("native_spacing_m must be positive and finite")
    return float(fwhm_km * 1_000.0 / native_spacing_m / FWHM_TO_SIGMA)


def gaussian_sidcv_hierarchy(
    field: SidcvField,
    *,
    fwhm_km: Sequence[float],
    minimum_valid_weight: float,
    truncate: float,
) -> GaussianSidcvHierarchy:
    """Build native plus direct Gaussian FWHM levels with harmonized validity."""
    widths = tuple(float(value) for value in fwhm_km)
    if not widths or widths != tuple(sorted(widths)) or len(set(widths)) != len(widths):
        raise ValueError("fwhm_km must contain unique increasing values")
    if truncate <= 0.0 or not np.isfinite(truncate):
        raise ValueError("truncate must be positive and finite")
    spacing_i = float(np.hypot(field.affine.x_i, field.affine.y_i))
    spacing_j = float(np.hypot(field.affine.x_j, field.affine.y_j))
    spacing = (spacing_i + spacing_j) / 2.0
    native_valid = (
        np.asarray(field.valid, dtype=bool) & np.isfinite(field.dx_m) & np.isfinite(field.dy_m)
    )
    native_dx = np.asarray(field.dx_m, dtype=np.float64)
    native_dy = np.asarray(field.dy_m, dtype=np.float64)
    raw: dict[str, tuple[FloatArray, FloatArray, BoolArray]] = {
        "native": (native_dx, native_dy, native_valid)
    }
    supports: dict[str, FloatArray] = {}
    sigma_by_label: dict[str, float] = {}
    for width in widths:
        label = f"fwhm{int(width) if width.is_integer() else f'{width:g}'}km"
        sigma_pixels = gaussian_fwhm_to_sigma_pixels(width, spacing)
        dx, dx_valid, dx_weight = masked_gaussian_lowpass(
            native_dx,
            native_valid,
            sigma_pixels=sigma_pixels,
            minimum_valid_weight=minimum_valid_weight,
            truncate=truncate,
        )
        dy, dy_valid, dy_weight = masked_gaussian_lowpass(
            native_dy,
            native_valid,
            sigma_pixels=sigma_pixels,
            minimum_valid_weight=minimum_valid_weight,
            truncate=truncate,
        )
        local_valid = native_valid & dx_valid & dy_valid & np.isfinite(dx) & np.isfinite(dy)
        raw[label] = (dx, dy, np.asarray(local_valid))
        supports[label] = np.minimum(dx_weight, dy_weight)
        sigma_by_label[label] = sigma_pixels

    common = np.logical_and.reduce([values[2] for values in raw.values()])
    fields: dict[str, SidcvField] = {}
    audit: list[dict[str, Any]] = []
    total_pixels = native_valid.size
    native_count = int(np.count_nonzero(native_valid))
    common_count = int(np.count_nonzero(common))
    for label, (dx, dy, local_valid) in raw.items():
        fields[label] = SidcvField(
            np.asarray(np.where(common, dx, np.nan)),
            np.asarray(np.where(common, dy, np.nan)),
            np.asarray(common),
            field.affine,
        )
        weight = supports.get(label)
        sampled_weight = weight[native_valid] if weight is not None else np.asarray([])
        width = (
            np.nan if label == "native" else float(label.removeprefix("fwhm").removesuffix("km"))
        )
        row: dict[str, Any] = {
            "drift_variant": label,
            "gaussian_fwhm_km": width,
            "sigma_pixels": sigma_by_label.get(label, np.nan),
            "truncate_sigma": truncate if label != "native" else np.nan,
            "minimum_valid_weight": minimum_valid_weight if label != "native" else np.nan,
            "grid_pixels": total_pixels,
            "native_valid_pixels": native_count,
            "native_valid_fraction": native_count / total_pixels,
            "variant_valid_before_intersection": int(np.count_nonzero(local_valid)),
            "valid_fraction": float(np.count_nonzero(local_valid) / total_pixels),
            "six_way_drift_common_pixels": common_count,
            "six_way_drift_common_fraction": common_count / total_pixels,
            "six_way_common_fraction_of_native_valid": common_count / native_count
            if native_count
            else np.nan,
        }
        for quantile, name in (
            (0.0, "weighted_support_min"),
            (0.05, "weighted_support_p05"),
            (0.25, "weighted_support_p25"),
            (0.50, "weighted_support_p50"),
            (0.75, "weighted_support_p75"),
            (0.95, "weighted_support_p95"),
            (1.0, "weighted_support_max"),
        ):
            row[name] = (
                float(np.quantile(sampled_weight, quantile)) if sampled_weight.size else np.nan
            )
        audit.append(row)
    return GaussianSidcvHierarchy(fields, supports, audit, np.asarray(common))


def strict_hierarchy_prediction_support(
    target: Any, persistence: Any, predictions: Mapping[str, Any]
) -> BoolArray:
    """Return the identical finite support shared by target, persistence and all predictions."""
    target_array = np.asarray(target)
    persistence_array = np.asarray(persistence)
    if target_array.shape != persistence_array.shape or any(
        np.asarray(prediction).shape != target_array.shape for prediction in predictions.values()
    ):
        raise ValueError("target, persistence and predictions must share a shape")
    if not predictions:
        raise ValueError("at least one prediction is required")
    return np.asarray(
        np.logical_and.reduce(
            [
                np.isfinite(target_array),
                np.isfinite(persistence_array),
                *(np.isfinite(np.asarray(value)) for value in predictions.values()),
            ]
        ),
        dtype=bool,
    )


def hierarchy_metric_rows(
    target: Any, persistence: Any, predictions: Mapping[str, Any]
) -> list[dict[str, float | int | str]]:
    """Compute paired hierarchy metrics and native-referenced gains on finite common support."""
    target_array = np.asarray(target, dtype=np.float64)
    persistence_array = np.asarray(persistence, dtype=np.float64)
    prediction_arrays = {
        label: np.asarray(value, dtype=np.float64) for label, value in predictions.items()
    }
    common = strict_hierarchy_prediction_support(target_array, persistence_array, prediction_arrays)
    n_eval = int(np.count_nonzero(common))
    if n_eval == 0:
        raise ValueError("evaluation support is empty")

    persistence_difference = persistence_array[common] - target_array[common]
    persistence_mae = float(np.mean(np.abs(persistence_difference)))
    persistence_rmse = float(np.sqrt(np.mean(persistence_difference**2)))
    persistence_bias = float(np.mean(persistence_difference))
    intermediate: list[dict[str, float | int | str]] = []
    for label, prediction in prediction_arrays.items():
        difference = prediction[common] - target_array[common]
        mae = float(np.mean(np.abs(difference)))
        gain = persistence_mae - mae
        intermediate.append(
            {
                "drift_variant": label,
                "n_eval": n_eval,
                "mae": mae,
                "rmse": float(np.sqrt(np.mean(difference**2))),
                "bias": float(np.mean(difference)),
                "persistence_mae": persistence_mae,
                "persistence_rmse": persistence_rmse,
                "persistence_bias": persistence_bias,
                "gain_mae": gain,
            }
        )
    native = next((row for row in intermediate if row["drift_variant"] == "native"), None)
    if native is None:
        raise ValueError("predictions must contain the native variant")
    native_mae = float(native["mae"])
    native_gain = float(native["gain_mae"])
    for row in intermediate:
        row["delta_mae_vs_native"] = float(row["mae"]) - native_mae
        row["delta_gain_vs_native"] = float(row["gain_mae"]) - native_gain
    return intermediate


def decompose_prediction_pair(
    target: Any, coarse_prediction: Any, fine_prediction: Any, valid: Any
) -> dict[str, float | int | bool | str]:
    """Decompose the MSE change caused by adding one prediction perturbation."""
    y = np.asarray(target, dtype=np.float64)
    coarse = np.asarray(coarse_prediction, dtype=np.float64)
    fine = np.asarray(fine_prediction, dtype=np.float64)
    admitted = np.asarray(valid, dtype=bool)
    if y.shape != coarse.shape or y.shape != fine.shape or y.shape != admitted.shape:
        raise ValueError("target, predictions and valid mask must share a shape")
    common = admitted & np.isfinite(y) & np.isfinite(coarse) & np.isfinite(fine)
    n = int(np.count_nonzero(common))
    if n == 0:
        raise ValueError("prediction-pair support is empty")
    y_values = y[common]
    coarse_values = coarse[common]
    fine_values = fine[common]
    perturbation = fine_values - coarse_values
    residual = y_values - coarse_values
    coarse_mse = float(np.mean(residual**2))
    fine_mse = float(np.mean((y_values - fine_values) ** 2))
    alignment = float(2.0 * np.mean(residual * perturbation))
    penalty = float(np.mean(perturbation**2))
    net = coarse_mse - fine_mse
    identity_residual = net - (alignment - penalty)
    scale = max(1.0, abs(net), abs(alignment), abs(penalty), coarse_mse, fine_mse)
    tolerance = IDENTITY_EPSILON_MULTIPLIER * np.finfo(np.float64).eps * scale
    affected_n = int(np.count_nonzero(perturbation != 0.0))
    return {
        "evaluation_cell_n": n,
        "mean_abs_prediction_perturbation": float(np.mean(np.abs(perturbation))),
        "rms_prediction_perturbation": float(np.sqrt(np.mean(perturbation**2))),
        "affected_cell_n_exact_nonzero": affected_n,
        "affected_cell_fraction_exact_nonzero": affected_n / n,
        "coarse_mse": coarse_mse,
        "fine_mse": fine_mse,
        "residual_alignment_term": alignment,
        "perturbation_penalty": penalty,
        "net_mse_gain": net,
        "net_mse_gain_direction": direction(net),
        "identity_numerical_residual": identity_residual,
        "identity_absolute_residual": abs(identity_residual),
        "identity_tolerance": tolerance,
        "identity_pass": bool(abs(identity_residual) <= tolerance),
    }
