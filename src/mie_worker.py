import os
import sys
from pathlib import Path
import json
import time
import logging
import shutil
import argparse
import datetime
import traceback
import numpy as np
import scipy.ndimage

import nest_asyncio
nest_asyncio.apply()

import panel as pn
try:
    pn.extension('vtk', design='material', template='plain')
except Exception:
    pass

import mie_core as phys
import mie_numba as mc_numba

AutoMieQ = phys.AutoMieQ

ROOT_DIR = Path(__file__).resolve().parent.parent

# =============================================================================
# 1. 系统配置与日志
# =============================================================================

logger = None

def setup_logging(project_name):
    """配置日志系统，输出到文件和控制台"""
    global logger
    log_dir = ROOT_DIR / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"mie_{project_name}_{timestamp}.log"
    logger = logging.getLogger("MonteCarloWorker")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(fh)
    return log_file

def log_msg(msg, level="info"):
    """同时打印到控制台和日志文件"""
    print(msg, flush=True)
    if logger:
        if level == "info":
            logger.info(msg)
        elif level == "error":
            logger.error(msg)
        elif level == "warning":
            logger.warning(msg)
        for handler in logger.handlers:
            handler.flush()

def setup_directories(project_name):
    """创建临时目录和输出目录，清理旧的临时文件"""
    temp_dir = ROOT_DIR / "temp" / "mie" / project_name
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            log_msg(f"Warning: Failed to clean temp dir: {e}", "warning")
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir = ROOT_DIR / "outputs" / "mie" / project_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir, output_dir

# =============================================================================
# 2. 物理计算层（优化 LUT）
# =============================================================================

DEFAULT_CONFIG = {
    'cpu_cores': 4,
    'L_size': 20.0, 'grid_dim': 120, 'wavelength_um': 1.55,
    'm_real': 1.311, 'm_imag': 1e-4, 'angstrom_q': 1.3,
    'r_bottom': 2.0, 'r_top': 12.0, 'sigma_ln': 0.35,
    'cloud_center_z': 10.0, 'cloud_thickness': 8.0, 'turbulence_scale': 4.0,
    'photons': 50000, 'visibility_km': 3.0, 'explode_dist': 0.7,
    'source_type': 'planar', 'source_width_m': 0.0,
    'field_compute_mode': 'proxy_only',
    'lidar_enabled': False,
    'range_bin_width_m': 1.0,
    'range_max_m': 0.0,
    'receiver_overlap_min': 1.0,
    'receiver_overlap_full_range_m': 0.0,
}


def normalize_field_compute_mode(mode):
    mode = str(mode).strip().lower()
    return mode if mode in {"proxy_only", "exact_only", "both"} else "proxy_only"


def build_field_catalog(config, exact_available=False):
    requested_mode = normalize_field_compute_mode(config.get('field_compute_mode', 'proxy_only'))
    catalog = {}
    if requested_mode != "exact_only":
        catalog["proxy"] = [
            {"name": "beta_back", "label": "后向代理场", "storage": "proxy_beta_back"},
            {"name": "beta_forward", "label": "前向代理场", "storage": "proxy_beta_forward"},
            {"name": "depol_ratio", "label": "退偏代理场", "storage": "proxy_depol_ratio"},
            {"name": "density", "label": "密度场", "storage": "density"},
        ]
    if requested_mode != "proxy_only" and exact_available:
        catalog["exact"] = [
            {"name": "beta_back", "label": "后向精确场", "storage": "exact_beta_back"},
            {"name": "beta_forward", "label": "前向精确场", "storage": "exact_beta_forward"},
            {"name": "depol_ratio", "label": "退偏精确场", "storage": "exact_depol_ratio"},
            {"name": "event_count", "label": "采样次数", "storage": "exact_event_count"},
        ]
    field_mode_note = ""
    if not catalog:
        catalog["proxy"] = [
            {"name": "beta_back", "label": "后向代理场", "storage": "proxy_beta_back"},
            {"name": "beta_forward", "label": "前向代理场", "storage": "proxy_beta_forward"},
            {"name": "depol_ratio", "label": "退偏代理场", "storage": "proxy_depol_ratio"},
            {"name": "density", "label": "密度场", "storage": "density"},
        ]
        field_mode_note = "exact field unavailable, fell back to proxy field only"
    elif requested_mode == "both" and "exact" not in catalog:
        field_mode_note = "exact field unavailable, exported proxy field only"

    effective_mode = "both" if len(catalog) >= 2 else ("exact_only" if "exact" in catalog else "proxy_only")
    meta = {
        "field_catalog": catalog,
        "available_field_families": [name for name in ("proxy", "exact") if name in catalog],
        "requested_field_compute_mode": requested_mode,
        "effective_field_compute_mode": effective_mode,
        "primary_field_family": "proxy" if "proxy" in catalog else next(iter(catalog)),
    }
    if field_mode_note:
        meta["field_mode_note"] = field_mode_note
    return meta


def build_default_proxy_catalog():
    return {
            "proxy": [
                {"name": "beta_back", "label": "后向代理场", "storage": "proxy_beta_back"},
                {"name": "beta_forward", "label": "前向代理场", "storage": "proxy_beta_forward"},
                {"name": "depol_ratio", "label": "退偏代理场", "storage": "proxy_depol_ratio"},
                {"name": "density", "label": "密度场", "storage": "density"},
            ],
    }

def _mie_layer_count(config):
    default_layers = max(6, min(16, int(np.ceil(float(config.get('grid_dim', 120)) / 8.0))))
    return max(2, int(config.get('mie_layer_count', default_layers)))


def _mie_n_radii(config):
    sigma_ln = float(config.get('sigma_ln', 0.35))
    default_n = 1 if sigma_ln <= 1e-6 else 8
    return max(1, int(config.get('mie_n_radii', default_n)))


def _sample_layer_profile(z_axis, layer_edges, layer_values):
    idx = np.searchsorted(layer_edges[1:], z_axis, side='right')
    idx = np.clip(idx, 0, len(layer_values) - 1)
    return np.asarray(layer_values, dtype=np.float64)[idx]


def build_mie_layers(config):
    """
    构造少量光学分层，供渲染场和 MC 共用。

    这样避免对每个 z 切片单独求解 Mie，又能保持垂直粒径变化的物理一致性。
    """
    L = float(config['L_size'])
    layer_count = _mie_layer_count(config)
    n_radii = _mie_n_radii(config)
    sigma_ln = float(config.get('sigma_ln', 0.35))
    size_mode = "mono" if sigma_ln <= 1e-6 else "lognormal"
    wavelength_m = float(config['wavelength_um']) * 1e-6
    wavelength_nm = float(config['wavelength_um']) * 1000.0
    m_real = float(config.get('m_real', 1.311))
    m_imag = float(config.get('m_imag', 1e-4))
    m_complex = complex(m_real, m_imag)
    visibility_km = float(config['visibility_km'])
    angstrom_q = float(config.get('angstrom_q', 1.3))
    forward_cone_deg = float(config.get('forward_cone_deg', 2.0))
    beta_ext_ref = phys.visibility_to_beta_ext_corrected(visibility_km, wavelength_nm, angstrom_q)

    edges = np.linspace(0.0, L, layer_count + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    z_norm = np.clip(centers / max(L, 1e-9), 0.0, 1.0)
    r_profile = float(config['r_bottom']) + z_norm * (float(config['r_top']) - float(config['r_bottom']))
    angles_deg = phys.generate_adaptive_angles(num_total=600)

    layers_config = []
    beta_back_profile = []
    beta_forward_profile = []
    depol_profile = []
    beta_ext_profile = []

    log_msg(f">> [Mie] Building {layer_count} optical layers | n_radii={n_radii} | size_mode={size_mode}")

    for i, radius_um in enumerate(r_profile):
        mie_res = phys.mie_effective_polarized(
            size_mode=size_mode,
            radius_um=float(radius_um),
            median_radius_um=float(radius_um),
            sigma_ln=sigma_ln,
            m_complex=m_complex,
            wavelength_m=wavelength_m,
            angles_deg=angles_deg,
            n_radii=n_radii,
        )
        scatter_obs = phys.mie_scatter_observables(mie_res, forward_cone_deg=forward_cone_deg)
        sigma_ext = max(float(mie_res.sigma_ext), 0.0)
        sigma_back = max(float(scatter_obs['sigma_back_ref']), 0.0)
        sigma_forward = max(float(scatter_obs['sigma_forward_ref']), 0.0)
        if sigma_ext > 1e-30:
            beta_back_ref = beta_ext_ref * sigma_back / sigma_ext
            beta_forward_ref = beta_ext_ref * sigma_forward / sigma_ext
        else:
            beta_back_ref = 0.0
            beta_forward_ref = 0.0

        thickness_m = float(edges[i + 1] - edges[i])
        layers_config.append({
            'thickness_m': thickness_m,
            'visibility_km': visibility_km,
            'radius_um': float(radius_um),
            'median_radius_um': float(radius_um),
            'size_mode': size_mode,
            'sigma_ln': sigma_ln,
            'n_radii': n_radii,
            'm_real': m_real,
            'm_imag': m_imag,
            '_mie_res': mie_res,
            'sigma_back_ref': sigma_back,
            'sigma_forward_ref': sigma_forward,
            'forward_back_ratio': float(scatter_obs['forward_back_ratio']),
            'depol_back': float(scatter_obs['depol_back']),
            'depol_forward': float(scatter_obs['depol_forward']),
        })
        beta_back_profile.append(beta_back_ref)
        beta_forward_profile.append(beta_forward_ref)
        depol_profile.append(float(scatter_obs['depol_back']))
        beta_ext_profile.append(beta_ext_ref)

    return {
        'layer_edges': edges,
        'layer_centers': centers,
        'layers_config': layers_config,
        'beta_back_profile': np.asarray(beta_back_profile, dtype=np.float64),
        'beta_forward_profile': np.asarray(beta_forward_profile, dtype=np.float64),
        'depol_profile': np.asarray(depol_profile, dtype=np.float64),
        'beta_ext_profile': np.asarray(beta_ext_profile, dtype=np.float64),
        'layer_count': layer_count,
        'n_radii': n_radii,
        'size_mode': size_mode,
    }


def generate_field(config, temp_dir: Path, optical_layers=None):
    """
    生成三维湍流密度场和一维光学 LUT，组合成三维后向散射系数场。

    物理模型：
        - 使用分形噪声叠加模拟云的不均匀性。
        - 垂直高斯剖面模拟云层。
        - 后向散射系数 = 密度场 × 一维 LUT（随高度变化）。
    """
    N = int(config['grid_dim'])
    log_msg(f">> [Physics] Generating Field {N}^3...")
    temp_dir.mkdir(parents=True, exist_ok=True)
    L = config['L_size']
    axis = np.linspace(0, L, N)
    rng = np.random.default_rng(101)

    def get_noise_layer(scale, strength):
        """生成指定尺度的湍流噪声层"""
        res = int(N / scale)
        if res < 2:
            res = 2
        small_grid = rng.uniform(0, 1, (res, res, res))
        zoom_factor = N / res
        return scipy.ndimage.zoom(small_grid, zoom_factor, order=1) * strength

    # 多尺度湍流叠加
    noise = (get_noise_layer(config['turbulence_scale'], 0.6) +
             get_noise_layer(config['turbulence_scale']/2, 0.3) +
             get_noise_layer(config['turbulence_scale']/4, 0.1))
    noise = (noise - noise.min()) / (noise.max() - noise.min())

    # 垂直高斯云廓线
    z_center = config['cloud_center_z']
    z_sigma = config['cloud_thickness'] / 2.355   # 高斯标准差
    vertical_profile = np.exp(-0.5 * ((axis - z_center) / z_sigma)**2).reshape(1, 1, N)

    raw_density = vertical_profile * (0.3 + 0.7 * noise)
    density_norm = np.clip(raw_density, 0, 1)
    if np.max(density_norm) < 0.01:
        density_norm = np.maximum(density_norm, 0.01)
    density_norm = np.ascontiguousarray(density_norm, dtype=np.float64)

    if optical_layers is None:
        optical_layers = build_mie_layers(config)

    lut_back = _sample_layer_profile(axis, optical_layers['layer_edges'], optical_layers['beta_back_profile'])
    lut_forward = _sample_layer_profile(axis, optical_layers['layer_edges'], optical_layers['beta_forward_profile'])
    lut_depol = _sample_layer_profile(axis, optical_layers['layer_edges'], optical_layers['depol_profile'])
    lut_ext = _sample_layer_profile(axis, optical_layers['layer_edges'], optical_layers['beta_ext_profile'])
    beta_back_vis = density_norm * lut_back.reshape(1, 1, N)
    beta_forward_vis = density_norm * lut_forward.reshape(1, 1, N)
    # Export a z-layer proxy field; this is not a voxel-history inversion.
    depol_ratio = np.broadcast_to(lut_depol.reshape(1, 1, N), density_norm.shape).copy()
    beta_ext_vis = density_norm * lut_ext.reshape(1, 1, N)

    field_file = temp_dir / "field_data.npz"
    np.savez_compressed(
        field_file,
        density=np.asfortranarray(density_norm.astype(np.float32)),
        beta_vis=np.asfortranarray(beta_back_vis.astype(np.float32)),
        beta_back=np.asfortranarray(beta_back_vis.astype(np.float32)),
        beta_forward=np.asfortranarray(beta_forward_vis.astype(np.float32)),
        depol_ratio=np.asfortranarray(depol_ratio.astype(np.float32)),
        proxy_beta_back=np.asfortranarray(beta_back_vis.astype(np.float32)),
        proxy_beta_forward=np.asfortranarray(beta_forward_vis.astype(np.float32)),
        proxy_depol_ratio=np.asfortranarray(depol_ratio.astype(np.float32)),
        axis=axis.astype(np.float32),
        lut_back=lut_back.astype(np.float32),
        lut_forward=lut_forward.astype(np.float32),
        lut_depol=lut_depol.astype(np.float32),
        proxy_lut_back=lut_back.astype(np.float32),
        proxy_lut_forward=lut_forward.astype(np.float32),
        proxy_lut_depol=lut_depol.astype(np.float32),
    )

    return {
        'L': L, 'dim': N,
        'beta_ext': beta_ext_vis,
        'beta_back': beta_back_vis,
        'beta_back_vis': beta_back_vis,
        'beta_forward': beta_forward_vis,
        'beta_forward_vis': beta_forward_vis,
        'depol_ratio': depol_ratio,
        'lut_back': lut_back,
        'lut_forward': lut_forward,
        'lut_depol': lut_depol,
        'density_norm': density_norm,
        'optical_layers': optical_layers,
    }


# =============================================================================
# 3. 可视化渲染（NPZ + 浏览器端 Plotly，与 Julia 后端同构）
# =============================================================================

def _summary_from_field(data):
    total_back = float(np.sum(data['beta_back']))
    total_forward = float(np.sum(data['beta_forward']))
    lut_depol = np.asarray(data.get('lut_depol', []), dtype=float)
    depol_back = float(np.nanmean(lut_depol)) if lut_depol.size else 0.0
    return np.asarray([
        total_back,
        total_forward,
        depol_back,
        depol_back,
        total_forward / total_back if total_back > 1e-30 else 0.0,
        0.0,
    ], dtype=np.float32)


def _summary_from_exact_fields(fields):
    total_back = float(np.sum(fields["beta_back"]))
    total_forward = float(np.sum(fields["beta_forward"]))
    depol_back = 0.0
    if total_back > 1e-20:
        pol = np.sqrt(
            float(np.sum(fields["back_Q"])) ** 2
            + float(np.sum(fields["back_U"])) ** 2
            + float(np.sum(fields["back_V"])) ** 2
        ) / total_back
        depol_back = float(np.clip(1.0 - pol, 0.0, 1.0))
    depol_forward = 0.0
    if total_forward > 1e-20:
        pol = np.sqrt(
            float(np.sum(fields["forward_Q"])) ** 2
            + float(np.sum(fields["forward_U"])) ** 2
            + float(np.sum(fields["forward_V"])) ** 2
        ) / total_forward
        depol_forward = float(np.clip(1.0 - pol, 0.0, 1.0))
    return np.asarray([
        total_back,
        total_forward,
        depol_back,
        depol_forward,
        total_forward / total_back if total_back > 1e-30 else 0.0,
        float(np.sum(fields["event_count"])),
    ], dtype=np.float32)


def attach_exact_fields(data, sim_res):
    voxel = sim_res.get("arrays", {}).get("voxel_fields")
    if not voxel:
        return False
    back_I = np.asarray(voxel["back_I"], dtype=np.float64)
    back_Q = np.asarray(voxel["back_Q"], dtype=np.float64)
    back_U = np.asarray(voxel["back_U"], dtype=np.float64)
    back_V = np.asarray(voxel["back_V"], dtype=np.float64)
    forward_I = np.asarray(voxel["forward_I"], dtype=np.float64)
    forward_Q = np.asarray(voxel["forward_Q"], dtype=np.float64)
    forward_U = np.asarray(voxel["forward_U"], dtype=np.float64)
    forward_V = np.asarray(voxel["forward_V"], dtype=np.float64)
    event_count = np.asarray(voxel["event_count"], dtype=np.float64)

    depol = np.zeros_like(back_I, dtype=np.float64)
    mask = back_I > 1e-20
    pol = np.zeros_like(back_I, dtype=np.float64)
    pol[mask] = np.sqrt(back_Q[mask] ** 2 + back_U[mask] ** 2 + back_V[mask] ** 2) / back_I[mask]
    depol[mask] = np.clip(1.0 - pol[mask], 0.0, 1.0)

    data["exact_fields"] = {
        "beta_back": back_I,
        "beta_forward": forward_I,
        "depol_ratio": depol,
        "event_count": event_count,
        "back_Q": back_Q,
        "back_U": back_U,
        "back_V": back_V,
        "forward_Q": forward_Q,
        "forward_U": forward_U,
        "forward_V": forward_V,
    }
    return True


def attach_lidar_observation(data, sim_res):
    obs = sim_res.get("arrays", {}).get("lidar_observation")
    if not obs:
        return False
    keys = [
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
    lidar = {name: np.asarray(obs[name], dtype=np.float64) for name in keys if name in obs}
    lidar["receiver_model"] = dict(obs.get("receiver_model", {}))
    data["lidar_observation"] = lidar
    return True


def save_field_npz(data, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    density = np.asfortranarray(np.asarray(data['density_norm'], dtype=np.float32))
    axis = np.asarray(np.linspace(0, data['L'], data['dim']), dtype=np.float32)
    field_meta = data.get("field_meta") or build_field_catalog({}, exact_available="exact_fields" in data)
    catalog = field_meta["field_catalog"]
    primary_family = field_meta.get("primary_field_family", "proxy")
    arrays = {
        "density": density,
        "axis": axis,
        "meta": np.asarray([float(data['L']), float(data['dim']), 0.0], dtype=np.float32),
    }

    proxy_arrays = {
        "beta_back": np.asfortranarray(np.asarray(data['beta_back'], dtype=np.float32)),
        "beta_forward": np.asfortranarray(np.asarray(data['beta_forward'], dtype=np.float32)),
        "depol_ratio": np.asfortranarray(np.asarray(data['depol_ratio'], dtype=np.float32)),
        "lut_back": np.asarray(data['lut_back'], dtype=np.float32),
        "lut_forward": np.asarray(data['lut_forward'], dtype=np.float32),
        "lut_depol": np.asarray(data['lut_depol'], dtype=np.float32),
        "summary": _summary_from_field(data),
    }

    if "proxy" in catalog:
        arrays.update({
            "proxy_beta_back": proxy_arrays["beta_back"],
            "proxy_beta_forward": proxy_arrays["beta_forward"],
            "proxy_depol_ratio": proxy_arrays["depol_ratio"],
            "proxy_lut_back": proxy_arrays["lut_back"],
            "proxy_lut_forward": proxy_arrays["lut_forward"],
            "proxy_lut_depol": proxy_arrays["lut_depol"],
            "proxy_summary": proxy_arrays["summary"],
        })

    exact_arrays = None
    if "exact" in catalog and "exact_fields" in data:
        exact = data["exact_fields"]
        exact_arrays = {
            "beta_back": np.asfortranarray(np.asarray(exact["beta_back"], dtype=np.float32)),
            "beta_forward": np.asfortranarray(np.asarray(exact["beta_forward"], dtype=np.float32)),
            "depol_ratio": np.asfortranarray(np.asarray(exact["depol_ratio"], dtype=np.float32)),
            "event_count": np.asfortranarray(np.asarray(exact["event_count"], dtype=np.float32)),
            "summary": _summary_from_exact_fields(exact),
        }
        arrays.update({
            "exact_beta_back": exact_arrays["beta_back"],
            "exact_beta_forward": exact_arrays["beta_forward"],
            "exact_depol_ratio": exact_arrays["depol_ratio"],
            "exact_event_count": exact_arrays["event_count"],
            "exact_summary": exact_arrays["summary"],
            "exact_back_Q": np.asfortranarray(np.asarray(exact["back_Q"], dtype=np.float32)),
            "exact_back_U": np.asfortranarray(np.asarray(exact["back_U"], dtype=np.float32)),
            "exact_back_V": np.asfortranarray(np.asarray(exact["back_V"], dtype=np.float32)),
            "exact_forward_Q": np.asfortranarray(np.asarray(exact["forward_Q"], dtype=np.float32)),
            "exact_forward_U": np.asfortranarray(np.asarray(exact["forward_U"], dtype=np.float32)),
            "exact_forward_V": np.asfortranarray(np.asarray(exact["forward_V"], dtype=np.float32)),
        })

    primary = exact_arrays if primary_family == "exact" and exact_arrays is not None else proxy_arrays
    arrays.update({
        "beta_back": primary["beta_back"],
        "beta_forward": primary["beta_forward"],
        "depol_ratio": primary["depol_ratio"],
        "summary": primary["summary"],
    })
    if primary_family == "proxy":
        arrays.update({
            "lut_back": proxy_arrays["lut_back"],
            "lut_forward": proxy_arrays["lut_forward"],
            "lut_depol": proxy_arrays["lut_depol"],
        })
    elif exact_arrays is not None:
        arrays["event_count"] = exact_arrays["event_count"]

    if "lidar_observation" in data:
        lidar = data["lidar_observation"]
        receiver_model_json = json.dumps(lidar.get("receiver_model", {}), ensure_ascii=False)
        arrays.update({
            "range_bins_m": np.asarray(lidar["range_bins_m"], dtype=np.float32),
            "echo_I": np.asarray(lidar["echo_I"], dtype=np.float32),
            "echo_Q": np.asarray(lidar["echo_Q"], dtype=np.float32),
            "echo_U": np.asarray(lidar["echo_U"], dtype=np.float32),
            "echo_V": np.asarray(lidar["echo_V"], dtype=np.float32),
            "echo_power": np.asarray(lidar["echo_power"], dtype=np.float32),
            "echo_depol": np.asarray(lidar["echo_depol"], dtype=np.float32),
            "echo_event_count": np.asarray(lidar["echo_event_count"], dtype=np.float32),
            "echo_weight_sum": np.asarray(lidar["echo_weight_sum"], dtype=np.float32),
            "echo_relative_error_est": np.asarray(lidar["echo_relative_error_est"], dtype=np.float32),
            "receiver_model_json": np.asarray(receiver_model_json),
        })

    npz_path = output_dir / "density.npz"
    np.savez_compressed(npz_path, **arrays)
    return npz_path


def _html_template(field_catalog_json: str, shape_info: str, view_name: str,
                   eye_x: float, eye_y: float, eye_z: float) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Mie {shape_info} - {view_name}</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
  <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    html, body {{ width: 100%; height: 100%; overflow: hidden; }}
    body {{ position: relative; background: #111; color: #ccc; font-family: monospace; overscroll-behavior: none; }}
    body.embed #toolbar {{ display: none; }}
    #plot {{ position: absolute; inset: 0; width: 100%; height: 100%; min-width: 0; min-height: 0; }}
    #toolbar {{ position: fixed; top: 8px; right: 8px; z-index: 120; display: flex; gap: 6px; flex-wrap: wrap; }}
    #toolbar button {{ border: 1px solid #333; background: rgba(20,20,20,0.85); color: #ddd; padding: 6px 10px; border-radius: 6px; cursor: pointer; font-size: 11px; }}
    #toolbar button.active {{ background: #2563eb; color: white; border-color: #3b82f6; }}
    #loading {{ position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 14px; color: #aaa; z-index: 100; }}
    #info {{ position: fixed; top: 8px; left: 8px; font-size: 11px; color: #666; pointer-events: none; }}
  </style>
</head>
<body>
  <div id="loading">正在加载数据...</div>
  <div id="info">Mie {shape_info}</div>
  <div id="toolbar"></div>
  <div id="plot" style="display:none"></div>
<script>
const CAMERA_EYE = {{x: {eye_x:.2f}, y: {eye_y:.2f}, z: {eye_z:.2f}}};
const URL_PARAMS = new URLSearchParams(window.location.search);
const EMBED_MODE = URL_PARAMS.get('embed') === '1';
const START_FAMILY = URL_PARAMS.get('family') || 'proxy';
const START_FIELD = URL_PARAMS.get('field') || 'beta_back';
const DATA_VERSION = URL_PARAMS.get('t') || String(Date.now());
const MAX_PREVIEW_GRID = Math.max(16, parseInt(URL_PARAMS.get('max_grid') || '64', 10));
const FIELD_CATALOG = {field_catalog_json};
if (EMBED_MODE) document.body.classList.add('embed');

function parseNpy(buffer) {{
  const view = new DataView(buffer);
  if (view.getUint8(0) !== 0x93) throw new Error('不是有效的 npy 文件');
  const major = view.getUint8(6);
  const headerLen = major >= 2 ? view.getUint32(8, true) : view.getUint16(8, true);
  const headerStart = major >= 2 ? 12 : 10;
  const headerStr = new TextDecoder().decode(new Uint8Array(buffer, headerStart, headerLen));
  const shapeMatch = headerStr.match(/'shape'\\s*:\\s*\\(([^)]+)\\)/);
  if (!shapeMatch) throw new Error('无法解析 shape: ' + headerStr);
  const shape = shapeMatch[1].split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
  const dtypeMatch = headerStr.match(/'descr'\\s*:\\s*'([^']+)'/);
  const dtype = dtypeMatch ? dtypeMatch[1] : '<f4';
  const dataBuffer = buffer.slice(headerStart + headerLen);
  let arr;
  if (dtype === '<f4' || dtype === '|f4') arr = new Float32Array(dataBuffer);
  else if (dtype === '<f8') arr = new Float64Array(dataBuffer);
  else if (dtype === '<i4' || dtype === '|i4') arr = new Int32Array(dataBuffer);
  else throw new Error('不支持的 dtype: ' + dtype);
  return {{data: arr, shape}};
}}

async function loadAndRender() {{
  const loading = document.getElementById('loading');
  const plotDiv = document.getElementById('plot');
  const toolbar = document.getElementById('toolbar');
  try {{
    loading.textContent = '正在下载数据文件...';
    const resp = await fetch('./density.npz?t=' + encodeURIComponent(DATA_VERSION), {{cache: 'no-store'}});
    if (!resp.ok) throw new Error('fetch 失败: ' + resp.status);
    const zip = await JSZip.loadAsync(await resp.arrayBuffer());
    async function readArray(name) {{
      const file = zip.file(name + '.npy');
      if (!file) throw new Error('找不到 ' + name + '.npy');
      return parseNpy(await file.async('arraybuffer'));
    }}
    async function readOptionalArray(name) {{
      const file = zip.file(name + '.npy');
      return file ? parseNpy(await file.async('arraybuffer')) : null;
    }}
    const axisObj = await readArray('axis');
    const summaryObj = await readArray('summary');
    const axis = axisObj.data;
    const N = axis.length;
    const stride = Math.max(1, Math.ceil(N / MAX_PREVIEW_GRID));
    const previewN = Math.ceil(N / stride);
    const total = previewN * previewN * previewN;
    const xs = new Float32Array(total), ys = new Float32Array(total), zs = new Float32Array(total);
    const previewFlatIndices = new Int32Array(total);
    let idx = 0;
    for (let iz = 0; iz < N; iz += stride) for (let iy = 0; iy < N; iy += stride) for (let ix = 0; ix < N; ix += stride) {{
      xs[idx] = axis[ix]; ys[idx] = axis[iy]; zs[idx] = axis[iz]; idx++;
      previewFlatIndices[idx - 1] = ix + N * (iy + N * iz);
    }}
    function colorscaleFor(fieldName, familyName) {{
      if (fieldName === 'beta_back') return familyName === 'exact' ? 'Turbo' : 'Hot';
      if (fieldName === 'beta_forward') return 'Viridis';
      if (fieldName === 'depol_ratio') return 'Cividis';
      if (fieldName === 'density') return 'Blues';
      if (fieldName === 'event_count') return 'Blues';
      return familyName === 'exact' ? 'Turbo' : 'Hot';
    }}
    const fieldDefs = {{}};
    const familySummaries = {{}};
    for (const [familyName, entries] of Object.entries(FIELD_CATALOG)) {{
      fieldDefs[familyName] = {{}};
      const familySummaryObj = await readOptionalArray(familyName + '_summary');
      familySummaries[familyName] = familySummaryObj ? familySummaryObj.data : summaryObj.data;
      for (const entry of entries) {{
        const storageName = entry.name === 'density' ? 'density' : (entry.storage || (familyName + '_' + entry.name));
        fieldDefs[familyName][entry.name] = {{
          label: entry.label || entry.name,
          storage: storageName,
          values: null,
          colorscale: colorscaleFor(entry.name, familyName),
          opacity: entry.name === 'depol_ratio' ? 0.18 : 0.12,
        }};
      }}
    }}
    async function loadFieldValues(def) {{
      if (def.values) return def.values;
      loading.style.display = 'block';
      loading.textContent = '正在加载字段...';
      const valuesObj = await readArray(def.storage);
      if (stride === 1) {{
        def.values = valuesObj.data;
      }} else {{
        const sampled = new Float32Array(total);
        for (let i = 0; i < total; i++) sampled[i] = valuesObj.data[previewFlatIndices[i]];
        def.values = sampled;
      }}
      return def.values;
    }}
    function fieldRange(values, fieldName, familyName) {{
      let vmin = Infinity, vmax = -Infinity;
      for (let i = 0; i < values.length; i++) {{
        const v = values[i];
        if (!Number.isFinite(v)) continue;
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
      }}
      if (!Number.isFinite(vmin) || !Number.isFinite(vmax)) return {{isomin: 0, isomax: 1}};
      if (fieldName === 'depol_ratio') return Math.abs(vmax - vmin) < Number.EPSILON ? {{isomin: Math.max(vmin - 1e-3, 0), isomax: vmin + 1e-3}} : {{isomin: Math.max(vmin, 0), isomax: vmax}};
      if (vmax <= 0) return {{isomin: 0, isomax: 1}};
      let thresholdRatio = 0.05;
      if (familyName === 'exact') {{
        thresholdRatio = fieldName === 'event_count' ? 0.00025 : 0.0005;
      }}
      return {{isomin: Math.max(vmax * thresholdRatio, Number.EPSILON), isomax: vmax}};
    }}
    function makeTrace(familyName, fieldName) {{
      const def = fieldDefs[familyName][fieldName];
      const range = fieldRange(def.values, fieldName, familyName);
      return {{
        type: 'volume', x: xs, y: ys, z: zs, value: def.values,
        isomin: range.isomin, isomax: range.isomax,
        opacity: def.opacity, surface: {{count: fieldName === 'depol_ratio' ? 8 : 6}},
        colorscale: def.colorscale,
        colorbar: {{title: {{text: def.label, font: {{color: '#aaa'}}}}, tickfont: {{color: '#aaa'}}, len: 0.6}},
        caps: {{x: {{show: false}}, y: {{show: false}}, z: {{show: false}}}},
      }};
    }}
    const layout = {{
      paper_bgcolor: '#111',
      scene: {{
        xaxis: {{title: 'X [m]', color: '#888', gridcolor: '#333', backgroundcolor: '#111'}},
        yaxis: {{title: 'Y [m]', color: '#888', gridcolor: '#333', backgroundcolor: '#111'}},
        zaxis: {{title: 'Z [m]', color: '#888', gridcolor: '#333', backgroundcolor: '#111'}},
        bgcolor: '#111', camera: {{eye: CAMERA_EYE}}, aspectmode: 'cube', uirevision: 'camera-lock',
      }},
      uirevision: 'field-switch', margin: {{l: 0, r: 0, b: 0, t: 0}},
    }};
    let currentFamily = Object.prototype.hasOwnProperty.call(fieldDefs, START_FAMILY) ? START_FAMILY : Object.keys(fieldDefs)[0];
    let currentField = Object.prototype.hasOwnProperty.call(fieldDefs[currentFamily], START_FIELD) ? START_FIELD : Object.keys(fieldDefs[currentFamily])[0];
    function updateInfo() {{
      const def = fieldDefs[currentFamily][currentField];
      const summary = familySummaries[currentFamily] || summaryObj.data;
      const ratio = Number.isFinite(summary[4]) ? summary[4].toExponential(3) : '0';
      const depolB = Number.isFinite(summary[2]) ? summary[2].toFixed(4) : '0.0000';
      document.getElementById('info').textContent = `Mie {shape_info} | ${{currentFamily}} | ${{def.label}} | F/B=${{ratio}} | depol=${{depolB}}`;
    }}
    function setActiveButton() {{
      Array.from(toolbar.querySelectorAll('button')).forEach(btn => btn.classList.toggle('active', btn.dataset.family === currentFamily && btn.dataset.field === currentField));
    }}
    async function renderField(familyName, fieldName) {{
      if (!fieldDefs[familyName] || !fieldDefs[familyName][fieldName]) return;
      currentFamily = familyName; currentField = fieldName;
      loading.style.display = 'block';
      const nextUrl = new URL(window.location.href);
      nextUrl.searchParams.set('family', currentFamily);
      nextUrl.searchParams.set('field', currentField);
      window.history.replaceState(null, '', nextUrl.toString());
      updateInfo(); setActiveButton();
      await loadFieldValues(fieldDefs[currentFamily][currentField]);
      plotDiv.style.display = 'block';
      await Plotly.react('plot', [makeTrace(familyName, fieldName)], layout, {{responsive: true, displaylogo: false, displayModeBar: !EMBED_MODE, scrollZoom: false, modeBarButtonsToRemove: ['toImage']}});
      loading.style.display = 'none';
    }}
    if (!EMBED_MODE) {{
      for (const [familyName, fields] of Object.entries(fieldDefs)) for (const [fieldName, def] of Object.entries(fields)) {{
        const btn = document.createElement('button');
        btn.textContent = familyName + ': ' + def.label;
        btn.dataset.family = familyName; btn.dataset.field = fieldName;
        btn.onclick = () => renderField(familyName, fieldName);
        toolbar.appendChild(btn);
      }}
    }}
    window.addEventListener('message', async (event) => {{
      const data = event && event.data ? event.data : null;
      if (!data || data.type !== 'iitm:set_field') return;
      await renderField(data.family || currentFamily, data.field);
      try {{ Plotly.Plots.resize(plotDiv); }} catch (_) {{}}
    }});
    await renderField(currentFamily, currentField);
  }} catch (err) {{
    loading.textContent = '渲染失败: ' + err.message;
    loading.style.color = '#f44';
    console.error(err);
  }}
}}
loadAndRender();
</script>
</body>
</html>"""


def render_headless(data, config, output_dir: Path):
    """保存 Julia 同构 NPZ 并生成浏览器端 Plotly HTML。"""
    log_msg(">> [Render] Saving NPZ + browser Plotly renderer...")
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_html in output_dir.glob("render*.html"):
        try:
            old_html.unlink()
        except Exception as exc:
            log_msg(f"Warning: Failed to remove stale HTML {old_html.name}: {exc}", "warning")
    save_field_npz(data, output_dir)

    field_meta = data.get("field_meta") or build_field_catalog(config, exact_available="exact_fields" in data)
    catalog_json = json.dumps(field_meta["field_catalog"], ensure_ascii=False)
    shape_info = "sphere"
    view_cfgs = [
        ("render_main.html", (1.5, 1.8, 1.2)),
        ("render_front.html", (0.0, 2.5, 0.5)),
        ("render_top.html", (0.0, 0.0, 2.5)),
        ("render_right.html", (2.5, 0.0, 0.5)),
    ]
    generated_files = []
    for filename, eye in view_cfgs:
        (output_dir / filename).write_text(
            _html_template(catalog_json, shape_info, filename, *eye),
            encoding="utf-8",
        )
        generated_files.append(filename)
        log_msg(f"   -> Exporting View: {filename}")
    return generated_files


# =============================================================================
# 4. 物理一致性测试套件（仅输出数据，无理论对比）
# =============================================================================

def run_consistency_tests():
    """
    运行一系列物理一致性测试，输出原始仿真结果，供手动对比理论值。
    测试包括：
        TC‑E: 无吸收能量守恒
        TC‑A: 有吸收能量守恒
        TC‑P: 线偏振入射退偏比
        TC‑R: 瑞利极限（粒径远小于波长）
        TC‑G: 几何光学极限（大粒径后向散射概率）
        TC‑L: 对数正态分布积分体积消光系数（手动积分对比）
    """
    print("\n" + "="*60)
    print(">> 物理一致性测试 (数据收集模式)")
    print(">> 仅输出仿真原始结果，请手动对比理论值")
    print("="*60)

    test_temp = ROOT_DIR / "temp" / "mie" / "test"
    test_temp.mkdir(parents=True, exist_ok=True)
    test_output = ROOT_DIR / "outputs" / "mie" / "test"
    test_output.mkdir(parents=True, exist_ok=True)

    rng_test = np.random.default_rng(20260407)

    def run_test_simulation(config_override, layer_config, photons_override=None, angstrom_q=1.3):
        config = DEFAULT_CONFIG.copy()
        config.update(config_override)
        if photons_override is not None:
            config['photons'] = photons_override
        config['grid_dim'] = int(config.get('grid_dim', 60))
        config['photons'] = int(config.get('photons', 20000))
        os.environ["NUMBA_NUM_THREADS"] = str(config.get('cpu_cores', 2))
        field = generate_field(config, test_temp)
        if np.max(field['density_norm']) < 0.1:
            field['density_norm'] = np.maximum(field['density_norm'], 0.1)
        freq = 299792458.0 / (config['wavelength_um']*1e-6) / 1e12
        grid_res = field['L'] / max(field['dim'] - 1, 1)
        sim = mc_numba.run_advanced_simulation(
            layers_config=layer_config,
            frequency_thz=freq,
            photons=config['photons'],
            density_grid=field['density_norm'],
            grid_res_m=grid_res,
            record_spatial=False,
            source_type="planar",
            source_width_m=field['L'],
            sigma_ln=config['sigma_ln'],
            angstrom_q=angstrom_q
        )
        return sim, config

    # ------------------------------------------------------------------
    # TC-E: 能量守恒（无吸收）
    # ------------------------------------------------------------------
    print("\n[TC-E] 能量守恒 (无吸收)")
    config_e = {
        'L_size': 20.0, 'grid_dim': 60,
        'cloud_center_z': 10.0, 'cloud_thickness': 18.0,
        'turbulence_scale': 1000.0,
        'r_bottom': 5.0, 'r_top': 5.0,
        'sigma_ln': 0.01,
        'visibility_km': 5.0,
        'photons': 20000,
        'cpu_cores': 2,
        'wavelength_um': 1.55
    }
    layer_e = [{'thickness_m': 20.0, 'visibility_km': 5.0, 'radius_um': 5.0, 'm_real': 1.33, 'm_imag': 0.0}]
    sim_e, _ = run_test_simulation(config_e, layer_e)
    print(f"   R_back = {sim_e['scalars']['R_back']:.6f}")
    print(f"   R_trans = {sim_e['scalars']['R_trans']:.6f}")
    print(f"   Sum = {sim_e['scalars']['R_back'] + sim_e['scalars']['R_trans']:.6f}")

    # ------------------------------------------------------------------
    # TC-A: 吸收守恒
    # ------------------------------------------------------------------
    print("\n[TC-A] 能量守恒 (有吸收)")
    config_a = config_e.copy()
    layer_a = [{'thickness_m': 20.0, 'visibility_km': 5.0, 'radius_um': 5.0, 'm_real': 1.33, 'm_imag': 0.001}]
    sim_a, _ = run_test_simulation(config_a, layer_a)
    print(f"   R_back = {sim_a['scalars']['R_back']:.6f}")
    print(f"   R_trans = {sim_a['scalars']['R_trans']:.6f}")
    print(f"   R_abs = {sim_a['scalars']['R_abs']:.6f}")
    print(f"   Sum = {sim_a['scalars']['R_back'] + sim_a['scalars']['R_trans'] + sim_a['scalars']['R_abs']:.6f}")

    # ------------------------------------------------------------------
    # TC-P: 偏振保持
    # ------------------------------------------------------------------
    print("\n[TC-P] 偏振保持 (线偏振入射)")
    config_p = {
        'L_size': 10.0, 'grid_dim': 50,
        'cloud_center_z': 5.0, 'cloud_thickness': 8.0,
        'turbulence_scale': 1000.0,
        'r_bottom': 10.0, 'r_top': 10.0,
        'sigma_ln': 0.01,
        'visibility_km': 2.0,
        'photons': 30000,
        'cpu_cores': 2,
        'wavelength_um': 1.55
    }
    layer_p = [{'thickness_m': 10.0, 'visibility_km': 2.0, 'radius_um': 10.0, 'm_real': 1.33, 'm_imag': 0.0}]
    sim_p, _ = run_test_simulation(config_p, layer_p)
    print(f"   Depolarization ratio = {sim_p['scalars']['depol']:.6f}")

    # ------------------------------------------------------------------
    # TC-R: 瑞利极限（波长比）
    # ------------------------------------------------------------------
    print("\n[TC-R] 瑞利极限 (粒径0.01um, 比较1.55um和0.55um后向回波)")
    config_r_base = {
        'L_size': 10.0, 'grid_dim': 60,
        'cloud_center_z': 5.0, 'cloud_thickness': 9.0,
        'turbulence_scale': 1000.0,
        'r_bottom': 0.01, 'r_top': 0.01,
        'sigma_ln': 0.01,
        'visibility_km': 0.1,
        'photons': 40000,
        'cpu_cores': 2,
        'wavelength_um': 1.55
    }
    layer_r = [{'thickness_m': 10.0, 'visibility_km': 0.1, 'radius_um': 0.01, 'm_real': 1.33, 'm_imag': 0.0}]
    # 1.55 um
    config_r1 = config_r_base.copy()
    config_r1['wavelength_um'] = 1.55
    sim_r1, _ = run_test_simulation(config_r1, layer_r, angstrom_q=4.0)
    back_1550 = sim_r1['scalars']['R_back']
    # 0.55 um
    config_r2 = config_r_base.copy()
    config_r2['wavelength_um'] = 0.55
    sim_r2, _ = run_test_simulation(config_r2, layer_r, angstrom_q=4.0)
    back_550 = sim_r2['scalars']['R_back']
    if back_1550 > 0:
        ratio = back_550 / back_1550
    else:
        ratio = float('inf')
    print(f"   R_back @ 1.55 um = {back_1550:.6e}")
    print(f"   R_back @ 0.55 um = {back_550:.6e}")
    print(f"   Ratio (550/1550) = {ratio:.3f}")

    # ------------------------------------------------------------------
    # TC-G: 几何光学极限（大粒径后向概率）
    # ------------------------------------------------------------------
    print("\n[TC-G] 几何光学极限 (粒径5um, 无吸收, 后向散射概率)")
    config_g = {
        'L_size': 10.0, 'grid_dim': 50,
        'cloud_center_z': 5.0, 'cloud_thickness': 8.0,
        'turbulence_scale': 1000.0,
        'r_bottom': 5.0, 'r_top': 5.0,
        'sigma_ln': 0.01,
        'visibility_km': 0.1,
        'photons': 50000,
        'cpu_cores': 2,
        'wavelength_um': 1.55
    }
    layer_g = [{'thickness_m': 10.0, 'visibility_km': 0.1, 'radius_um': 5.0, 'm_real': 1.33, 'm_imag': 0.0}]
    sim_g, _ = run_test_simulation(config_g, layer_g)
    print(f"   R_back (MC) = {sim_g['scalars']['R_back']:.6f}")

    # ------------------------------------------------------------------
    # TC-L: 对数正态分布积分（体积消光系数）
    # ------------------------------------------------------------------
    print("\n[TC-L] 对数正态分布积分 (体积消光系数 Bext)")
    try:
        r_g = 2.0
        sigma_ln = 0.35
        sigma_g = np.exp(sigma_ln)
        wl_nm = 1550
        m = 1.33 + 0.0j
        # 手动积分（作为参考）
        r_vals = np.logspace(np.log10(0.01), np.log10(100), 500)
        pdf = (1/(r_vals * sigma_ln * np.sqrt(2*np.pi))) * np.exp(-(np.log(r_vals/r_g))**2/(2*sigma_ln**2))
        pdf /= np.trapezoid(pdf, r_vals)
        Bext_manual = 0.0
        for r, w in zip(r_vals, pdf):
            q_ext = AutoMieQ(m, wl_nm, 2*r, asDict=False)[0]
            cross = q_ext * np.pi * (r*1e-6)**2
            Bext_manual += w * cross * 1e6
        print(f"   Bext (manual integration) = {Bext_manual:.6e} m^-1")
    except Exception as e:
        print(f"   TC-L 手动积分失败: {e}")

    print("\n" + "="*60)
    print(">> 数据收集完成，请手动对比理论值。")
    print("="*60)


# =============================================================================
# 5. 主逻辑
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, required=False, default=None)
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--cpu_limit", type=str, default="4")
    parser.add_argument("--test_only", action="store_true", help="仅运行物理一致性测试（数据模式）")
    args = parser.parse_args()

    if args.test_only:
        run_consistency_tests()
        return

    if args.project_name is None or args.config is None:
        print("Error: 需要提供 --project_name 和 --config，或使用 --test_only", file=sys.stderr)
        sys.exit(1)

    os.environ["NUMBA_NUM_THREADS"] = args.cpu_limit
    os.environ["OMP_NUM_THREADS"] = args.cpu_limit
    log_file = setup_logging(args.project_name)

    log_msg("="*60)
    log_msg(f">> [Init] MIE Worker Started (Offline/Embedded Mode)")
    log_msg(f">> [Info] Cores: {args.cpu_limit}")
    log_msg(f">> [Info] Log: {log_file}")
    log_msg("="*60)

    result_payload = {"status": "failed", "metrics": {}, "artifacts": []}

    try:
        user_config = json.loads(args.config)
        config = DEFAULT_CONFIG.copy()
        config.update(user_config)
        config['grid_dim'] = int(config.get('grid_dim', 120))
        config['photons'] = int(config.get('photons', 50000))
        requested_mode = normalize_field_compute_mode(config.get('field_compute_mode', 'proxy_only'))
        lidar_enabled = bool(config.get('lidar_enabled', False))
        collect_exact_fields = requested_mode != "proxy_only" or lidar_enabled
        field_meta = build_field_catalog(config, exact_available=collect_exact_fields)
        temp_dir, output_dir = setup_directories(args.project_name)

        t0 = time.time()
        optical_layers = build_mie_layers(config)
        field_data = generate_field(config, temp_dir, optical_layers)

        c = 299792458.0
        freq = (c / (config['wavelength_um'] * 1e-6)) / 1e12
        grid_res = field_data['L'] / max(field_data['dim'] - 1, 1)

        log_msg(f">> [Sim] Numba Kernel Start ({config['photons']} photons)...")
        sim_res = mc_numba.run_advanced_simulation(
            layers_config=optical_layers['layers_config'],
            frequency_thz=freq,
            photons=config['photons'],
            density_grid=field_data['density_norm'],
            grid_res_m=grid_res,
            record_spatial=False,
            record_back_hist=False,
            source_type=str(config.get('source_type', 'planar')).lower(),
            source_width_m=(
                float(config.get('source_width_m', 0.0))
                if float(config.get('source_width_m', 0.0)) > 0.0
                else field_data['L']
            ),
            sigma_ln=config['sigma_ln'],
            angstrom_q=config.get('angstrom_q', 1.3),
            collect_voxel_fields=collect_exact_fields,
            field_forward_half_angle_deg=float(config.get('field_forward_half_angle_deg', 90.0)),
            field_back_half_angle_deg=float(config.get('field_back_half_angle_deg', 90.0)),
            field_quadrature_polar=int(config.get('field_quadrature_polar', 2)),
            field_quadrature_azimuth=int(config.get('field_quadrature_azimuth', 6)),
            collect_lidar_observation=lidar_enabled,
            range_bin_width_m=float(config.get('range_bin_width_m', 1.0)),
            range_max_m=(
                float(config.get('range_max_m', 0.0))
                if float(config.get('range_max_m', 0.0)) > 0.0
                else None
            ),
            receiver_overlap_min=float(config.get('receiver_overlap_min', 1.0)),
            receiver_overlap_full_range_m=float(config.get('receiver_overlap_full_range_m', 0.0)),
        )

        exact_available = attach_exact_fields(field_data, sim_res)
        lidar_available = attach_lidar_observation(field_data, sim_res)
        field_meta = build_field_catalog(config, exact_available=exact_available)
        field_data["field_meta"] = field_meta
        if field_meta["requested_field_compute_mode"] != field_meta["effective_field_compute_mode"]:
            log_msg(
                f">> [FieldMode] requested={field_meta['requested_field_compute_mode']} "
                f"but current Mie worker exports {field_meta['effective_field_compute_mode']}"
            )

        sim_file = temp_dir / "sim_results.npy"
        np.save(sim_file, sim_res)

        html_files = render_headless(field_data, config, output_dir)
        if not html_files:
            raise RuntimeError("Rendering failed: No valid HTML files were generated.")

        duration = time.time() - t0
        log_msg(f">> [Success] All Done in {duration:.2f}s")

        result_payload["status"] = "success"
        metrics = {"duration_sec": duration}
        for k, v in sim_res.get('scalars', {}).items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                metrics[k] = float(v)
            else:
                metrics[k] = v
        result_payload["metrics"] = metrics
        result_payload["artifacts"] = html_files
        result_payload["lidar_observation_available"] = bool(lidar_available)
        result_payload.update(field_meta)

    except Exception as e:
        tb = traceback.format_exc()
        log_msg(f"Crash: {tb}", "error")
        log_msg(f">> [FATAL] {e}", "error")
        result_payload["status"] = "error"
        result_payload["error"] = str(e)

    finally:
        if logger:
            logging.shutdown()
        try:
            print(json.dumps(result_payload), flush=True)
        except:
            pass

if __name__ == "__main__":
    main()
