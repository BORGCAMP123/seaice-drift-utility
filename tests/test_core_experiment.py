"""Small synthetic checks for the raw-product experiment kernels."""

from __future__ import annotations

import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seaice_reconstruction.core import (  # noqa: E402
    AffineGrid,
    Grid,
    SidcvField,
    _comparison_grid,
    _rasterize,
    _read_vnp29_window,
    aggregate_common_support,
    decompose_prediction_pair,
    gaussian_sidcv_hierarchy,
    hierarchy_metric_rows,
    load_sidcv_field,
    strict_hierarchy_prediction_support,
    transport_with_sidcv,
)

sys.path.insert(0, str(ROOT / "scripts"))
import run_core_experiment as raw_runner  # noqa: E402
from summarize_core_experiment import summarize  # noqa: E402


def test_viirs_reader_and_one_km_vote_fraction(tmp_path: Path) -> None:
    path = tmp_path / "optical.nc"
    with netCDF4.Dataset(path, "w") as ds:
        geo = ds.createGroup("GeolocationData")
        science = ds.createGroup("SeaIceCoverData")
        geo.createDimension("line", 1)
        geo.createDimension("pixel", 3)
        science.createDimension("line", 1)
        science.createDimension("pixel", 3)
        geo.createVariable("latitude", "f8", ("line", "pixel"))[:] = [[75, 75, 75]]
        geo.createVariable("longitude", "f8", ("line", "pixel"))[:] = [[10, 10, 10]]
        science.createVariable("SeaIceCover", "i2", ("line", "pixel"))[:] = [[0, 1, 250]]
    pixels, accounting = _read_vnp29_window(
        path, {"south": 68, "north": 82, "west": -15, "east": 20}
    )
    assert accounting["clear_source_candidates"] == 2
    assert accounting["cloud"] == 1
    grid = Grid(float(pixels["target_x"][0]) - 100, float(pixels["target_y"][0]) - 100, 1000, 1, 1)
    sicf, counts = _rasterize(pixels["class"], pixels["target_x"], pixels["target_y"], grid)
    assert counts[0, 0] == 2
    assert sicf[0, 0] == 0.5
    region_grid = _comparison_grid({"south": 74, "north": 76, "west": 9, "east": 11}, 1000)
    assert region_grid.width > 0 and region_grid.height > 0


def test_hierarchy_hard_warp_support_and_identity() -> None:
    shape = (101, 101)
    affine = AffineGrid(0, 1000, 0, 0, 0, 1000, 0, 0)
    field = SidcvField(np.full(shape, 1000.0), np.zeros(shape), np.ones(shape, bool), affine)
    hierarchy = gaussian_sidcv_hierarchy(
        field, fwhm_km=[1, 2, 5, 10, 25], minimum_valid_weight=0.75, truncate=4.0
    )
    assert list(hierarchy.fields) == [
        "native",
        "fwhm1km",
        "fwhm2km",
        "fwhm5km",
        "fwhm10km",
        "fwhm25km",
    ]
    assert hierarchy.common_native_valid[50, 50]
    source = {
        "class": np.array([1, 0], dtype=np.int64),
        "target_x": np.array([50500.0, 50500.0]),
        "target_y": np.array([50500.0, 50500.0]),
    }
    grid = Grid(50000, 50000, 1000, 3, 1)
    prediction, accounting = transport_with_sidcv(
        source, grid=grid, sidcv=hierarchy.fields["native"], alpha=1.0
    )
    assert accounting["valid_sidcv_source_pixels"] == 2
    assert prediction[0, 1] == 0.5
    target = np.array([[0.4, 0.6, np.nan]])
    persistence = np.array([[0.4, 0.5, np.nan]])
    predictions = {name: np.array([[0.4, 0.5, np.nan]]) for name in hierarchy.fields}
    common = strict_hierarchy_prediction_support(target, persistence, predictions)
    assert common.tolist() == [[True, True, False]]
    aggregated, _, count = aggregate_common_support(
        {"target": target, "persistence": persistence, **predictions}, common, 2
    )
    assert count.tolist() == [[2, 0]]
    metrics = hierarchy_metric_rows(
        aggregated["target"],
        aggregated["persistence"],
        {name: aggregated[name] for name in hierarchy.fields},
    )
    assert len(metrics) == 6 and all(row["n_eval"] == 1 for row in metrics)
    parts = decompose_prediction_pair(
        np.array([[0.6, 0.8]]),
        np.array([[0.4, 0.5]]),
        np.array([[0.5, 0.7]]),
        np.array([[True, True]]),
    )
    assert parts["identity_pass"]
    assert (
        abs(
            parts["net_mse_gain"]
            - (parts["residual_alignment_term"] - parts["perturbation_penalty"])
        )
        < 1e-14
    )


def test_sidcv_reader_qc_and_projection(tmp_path: Path) -> None:
    path = tmp_path / "displacement.nc"
    row, col = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("row", 5)
        ds.createDimension("col", 5)
        ds.createDimension("bounds", 2)
        ds.time_coverage_start = "2021-01-01T00:00:00Z"
        ds.time_coverage_end = "2021-01-02T00:00:00Z"
        ds.grid_resolution_dX = "400 m"
        ds.grid_resolution_dY = "400 m"
        for name, values in {
            "lon": 10.0 + 0.001 * col,
            "lat": 75.0 + 0.001 * row,
            "SeaIce_U": np.full((5, 5), 100.0),
            "SeaIce_V": np.zeros((5, 5)),
            "rejection_flag": np.zeros((5, 5)),
            "qc_flag": np.zeros((5, 5)),
        }.items():
            ds.createVariable(name, "f8", ("row", "col"))[:] = values
        ds.variables["rejection_flag"][0, 0] = 1
        ds.createVariable("time", "f8", ("bounds",))[:] = [0, 1]
        ds.createVariable("time_bnds", "f8", ("bounds",))[:] = [0, 1]
    field, metadata = load_sidcv_field(path)
    assert metadata["valid_native_pixels"] == 24
    assert not field.valid[0, 0]
    assert np.isfinite(field.dx_m[2, 2]) and np.isfinite(field.dy_m[2, 2])
    assert field.affine.max_residual_m < 5.0


def test_one_case_reconstruction_workflow(monkeypatch, tmp_path: Path) -> None:
    field = SidcvField(
        np.zeros((101, 101)),
        np.zeros((101, 101)),
        np.ones((101, 101), bool),
        AffineGrid(0, 1000, 0, 0, 0, 1000, 0, 0),
    )
    pixels = {
        "class": np.array([0, 1], dtype=np.int64),
        "target_x": np.array([50500.0, 50500.0]),
        "target_y": np.array([50500.0, 50500.0]),
    }
    monkeypatch.setattr(
        raw_runner, "_comparison_grid", lambda window, resolution: Grid(50000, 50000, 1000, 3, 1)
    )
    monkeypatch.setattr(
        raw_runner,
        "_read_vnp29_window",
        lambda path, window: (pixels, {"clear_source_candidates": 2}),
    )
    monkeypatch.setattr(
        raw_runner, "load_sidcv_field", lambda path: (field, {"valid_native_pixels": 10201})
    )
    case = pd.read_csv(ROOT / "metadata/event_inventory.csv").iloc[0]
    item = {name: str(value) for name, value in case.items()}
    config = {
        "original_region_windows": {item["region"]: {}},
        "independent_extension_region_windows": {item["region"]: {}},
        "grid_resolution_m": 1000,
        "gaussian_fwhm_km": [1, 2, 5, 10, 25],
        "gaussian_minimum_valid_weight": 0.75,
        "gaussian_truncate_sigma": 4.0,
        "evaluation_scales_km": [1, 2, 5, 10],
    }
    viirs = {
        item["source_file"]: tmp_path / item["source_file"],
        item["target_file"]: tmp_path / item["target_file"],
    }
    sidcv = {item["sidcv_file"]: tmp_path / item["sidcv_file"]}
    metrics, decomposition, metadata = raw_runner.run_case(item, config, viirs, sidcv)
    assert len(metrics) == 24 and len(decomposition) == 20
    assert all(row["identity_pass"] for row in decomposition)
    assert metadata["final_prediction_common_pixels"] == 1


def test_event_aggregation_and_discovery_rule(tmp_path: Path) -> None:
    inventory = pd.read_csv(ROOT / "metadata/event_inventory.csv")
    selected = inventory.loc[inventory.informative]
    levels = ["native", "fwhm1km", "fwhm2km", "fwhm5km", "fwhm10km", "fwhm25km"]
    values = dict(zip(levels, [0.20, 0.18, 0.10, 0.22, 0.24, 0.25], strict=True))
    metrics = []
    parts = []
    for item in selected.itertuples(index=False):
        for scale in (1, 2, 5, 10):
            for level in levels:
                metrics.append(
                    {
                        "case_id": item.case_id,
                        "primary_independence_cluster": item.primary_independence_cluster,
                        "sample_group": item.sample_group,
                        "sicf_evaluation_scale_km": scale,
                        "drift_variant": level,
                        "n_eval": 100,
                        "mae": values[level],
                        "rmse": values[level],
                        "mse": values[level] ** 2,
                        "persistence_mae": 0.3,
                        "persistence_rmse": 0.3,
                        "persistence_mse": 0.09,
                    }
                )
            for step in ("25_to_10", "10_to_5", "5_to_2", "2_to_1", "1_to_native"):
                parts.append(
                    {
                        "case_id": item.case_id,
                        "primary_independence_cluster": item.primary_independence_cluster,
                        "sicf_evaluation_scale_km": scale,
                        "hierarchy_step": step,
                        "identity_pass": True,
                        "mean_abs_prediction_perturbation": 0.1,
                        "rms_prediction_perturbation": 0.1,
                        "affected_cell_fraction_exact_nonzero": 1.0,
                        "coarse_mse": 0.04,
                        "fine_mse": 0.01,
                        "residual_alignment_term": 0.04,
                        "perturbation_penalty": 0.01,
                        "net_mse_gain": 0.03,
                        "identity_numerical_residual": 0.0,
                        "identity_absolute_residual": 0.0,
                        "identity_tolerance": 1e-12,
                    }
                )
    pd.DataFrame(metrics).to_csv(tmp_path / "case_scale_metrics.csv", index=False)
    pd.DataFrame(parts).to_csv(tmp_path / "case_prediction_error_decomposition.csv", index=False)
    summarize(tmp_path)
    response = pd.read_csv(tmp_path / "event_scale_response.csv")
    rules = pd.read_csv(tmp_path / "discovery_fixed_rule.csv")
    validation = pd.read_csv(tmp_path / "independent_extension_validation.csv")
    assert len(response) == 116
    assert len(pd.read_csv(tmp_path / "event_scale_prediction_error_decomposition.csv")) == 580
    assert rules.selected_fwhm_km.tolist() == [2, 2, 2, 2]
    assert len(validation) == 84
    assert (validation.fixed_rule_effect_mae > 0).all()
