from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

import mie_numba
import mie_worker
from dataset_sampling import grid_or_constant_params


ROOT_DIR = Path(__file__).resolve().parent.parent


OBSERVATION_KEYS = [
    "range_bins_m",
    "echo_I",
    "echo_Q",
    "echo_U",
    "echo_V",
    "echo_power",
    "echo_depol",
    "echo_event_count",
    "echo_weight_sum",
    "echo_relative_error_est",
]


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def split_name(index: int, n_samples: int, split: dict[str, float] | None) -> str:
    if not split:
        return "train"
    train_end = int(round(n_samples * float(split.get("train", 0.7))))
    val_end = train_end + int(round(n_samples * float(split.get("val", 0.15))))
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def build_quality(config: dict[str, Any], observation: dict[str, Any], backend_meta: dict[str, Any]) -> dict[str, Any]:
    counts = np.asarray(observation.get("echo_event_count", []), dtype=np.float64)
    rel_err = np.asarray(observation.get("echo_relative_error_est", []), dtype=np.float64)
    valid = counts > 0.0
    return {
        "photons": int(config.get("photons", 0)),
        "echo_event_count_sum": float(np.sum(counts)),
        "valid_bin_count": int(np.sum(valid)),
        "max_echo_relative_error_est": float(np.max(rel_err[valid])) if np.any(valid) else None,
        "code_version": git_hash(),
        "random_seed": int(config.get("seed", 0)),
        "backend_meta": backend_meta,
    }


def run_mie_sample(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = mie_worker.DEFAULT_CONFIG.copy()
    cfg.update(config)
    cfg["lidar_enabled"] = True
    cfg["field_compute_mode"] = str(cfg.get("field_compute_mode", "proxy_only"))
    cfg["grid_dim"] = int(cfg.get("grid_dim", 32))
    cfg["photons"] = int(cfg.get("photons", 10000))

    layers = mie_worker.build_mie_layers(cfg)
    temp_dir = ROOT_DIR / "temp" / "datasets" / "_single_sample"
    field = mie_worker.generate_field(cfg, temp_dir, layers)
    freq = 299792458.0 / (float(cfg["wavelength_um"]) * 1e-6) / 1e12
    range_max = float(cfg.get("range_max_m", 0.0))
    sim = mie_numba.run_advanced_simulation(
        layers_config=layers["layers_config"],
        frequency_thz=freq,
        photons=cfg["photons"],
        density_grid=field["density_norm"],
        grid_res_m=field["L"] / max(field["dim"] - 1, 1),
        source_type="planar",
        source_width_m=field["L"],
        sigma_ln=cfg["sigma_ln"],
        collect_voxel_fields=True,
        field_forward_half_angle_deg=float(cfg.get("field_forward_half_angle_deg", 90.0)),
        field_back_half_angle_deg=float(cfg.get("field_back_half_angle_deg", 90.0)),
        field_quadrature_polar=int(cfg.get("field_quadrature_polar", 2)),
        field_quadrature_azimuth=int(cfg.get("field_quadrature_azimuth", 6)),
        collect_lidar_observation=True,
        range_bin_width_m=float(cfg.get("range_bin_width_m", 1.0)),
        range_max_m=range_max if range_max > 0.0 else None,
        receiver_overlap_min=float(cfg.get("receiver_overlap_min", 1.0)),
        receiver_overlap_full_range_m=float(cfg.get("receiver_overlap_full_range_m", 0.0)),
    )
    obs = sim["arrays"]["lidar_observation"]
    return cfg, obs


def run_iitm_sample(config: dict[str, Any], project_name: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cfg = dict(config)
    cfg["lidar_enabled"] = True
    cfg["field_compute_mode"] = str(cfg.get("field_compute_mode", "proxy_only"))
    cfg["project_name"] = project_name
    cmd = [
        "pixi",
        "run",
        "-e",
        "gui",
        "python",
        "src/iitm_http_worker.py",
        "--project_name",
        project_name,
        "--config",
        json.dumps(cfg, ensure_ascii=False),
        "--cpu_limit",
        str(cfg.get("cpu_cores", cfg.get("cpu_limit", 4))),
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(cfg.get("dataset_worker_timeout_sec", 1800)),
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-4000:])
    response = None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                response = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if not response or response.get("status") != "success":
        raise RuntimeError(proc.stdout[-4000:])
    npz_path = ROOT_DIR / "outputs" / "iitm" / project_name / "density.npz"
    if not npz_path.exists():
        raise RuntimeError(f"IITM output missing: {npz_path}")
    with np.load(npz_path) as data:
        observation = {key: np.asarray(data[key], dtype=np.float64) for key in OBSERVATION_KEYS}
        if "receiver_model_json_utf8" in data.files:
            raw = bytes(np.asarray(data["receiver_model_json_utf8"], dtype=np.uint8).tolist())
            observation["receiver_model"] = json.loads(raw.decode("utf-8"))
        else:
            observation["receiver_model"] = {}
    return cfg, observation, response.get("metrics", {})


def export_sample(
    sample_dir: Path,
    backend: str,
    config: dict[str, Any],
    observation: dict[str, Any],
    backend_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    obs_arrays = {key: np.asarray(observation[key], dtype=np.float32) for key in OBSERVATION_KEYS}
    np.savez_compressed(sample_dir / "observation.npz", **obs_arrays)

    receiver = dict(observation.get("receiver_model", {}))
    truth = {
        "backend": backend,
        "medium": {
            key: config.get(key)
            for key in ["visibility_km", "r_bottom", "r_top", "sigma_ln", "m_real", "m_imag", "wavelength_um"]
        },
        "field": {
            key: config.get(key)
            for key in ["L_size", "grid_dim", "cloud_center_z", "cloud_thickness", "turbulence_scale"]
        },
    }
    quality = build_quality(config, observation, backend_meta or {})
    files = {
        "truth.json": truth,
        "receiver.json": receiver,
        "quality.json": quality,
        "run_config.json": config,
    }
    for name, payload in files.items():
        (sample_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return quality


def run_dataset(spec: dict[str, Any], output_root: Path | None = None) -> dict[str, Any]:
    backend = str(spec.get("backend", "mie")).lower()
    if backend not in {"mie", "iitm"}:
        raise NotImplementedError("dataset_runner supports backend='mie' or backend='iitm'")
    dataset_name = str(spec.get("dataset_name", "dataset"))
    n_samples = int(spec.get("n_samples", 1))
    seed = int(spec.get("seed", 0))
    output_root = output_root or (ROOT_DIR / "outputs" / "datasets")
    dataset_dir = output_root / dataset_name
    samples_dir = dataset_dir / "samples"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    base_config = dict(spec.get("base_config", {}))
    medium_samples = grid_or_constant_params(dict(spec.get("sampling", {})), n_samples, seed=seed)
    receiver_samples = grid_or_constant_params(dict(spec.get("receiver", {})), n_samples, seed=seed + 17)

    manifest_samples = []
    for i in range(n_samples):
        sample_id = f"sample_{i + 1:06d}"
        config = base_config.copy()
        config.update(medium_samples[i] if i < len(medium_samples) else {})
        config.update(receiver_samples[i] if i < len(receiver_samples) else {})
        config["seed"] = seed + i
        sample_dir = samples_dir / sample_id
        try:
            if backend == "mie":
                effective_config, observation = run_mie_sample(config)
                backend_meta = {}
            else:
                effective_config, observation, backend_meta = run_iitm_sample(config, f"{dataset_name}_{sample_id}")
            quality = export_sample(sample_dir, backend, effective_config, observation, backend_meta)
            status = "success"
            error = None
        except Exception as exc:
            sample_dir.mkdir(parents=True, exist_ok=True)
            error = str(exc)
            quality = {}
            status = "failed"
            (sample_dir / "error.json").write_text(
                json.dumps({"error": error}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        manifest_samples.append(
            {
                "id": sample_id,
                "split": split_name(i, n_samples, spec.get("split")),
                "status": status,
                "path": str(sample_dir.relative_to(dataset_dir)),
                "quality": quality,
                "error": error,
            }
        )

    manifest = {
        "dataset_name": dataset_name,
        "backend": backend,
        "n_samples": n_samples,
        "code_version": git_hash(),
        "samples": manifest_samples,
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output_root", default=None)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    manifest = run_dataset(spec, Path(args.output_root) if args.output_root else None)
    print(json.dumps({"status": "success", "manifest": manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
