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
    'field_compute_mode': 'proxy_only',
}


def build_field_catalog(config):
    requested_mode = str(config.get('field_compute_mode', 'proxy_only'))
    return {
        "field_catalog": {
            "proxy": [
                {"name": "beta_back", "label": "后向代理场"},
                {"name": "beta_forward", "label": "前向代理场"},
                {"name": "depol_ratio", "label": "退偏代理场"},
            ],
        },
        "available_field_families": ["proxy"],
        "requested_field_compute_mode": requested_mode,
        "effective_field_compute_mode": "proxy_only",
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
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing='ij')
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
    vertical_profile = np.exp(-0.5 * ((Z - z_center) / z_sigma)**2)

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
        density=density_norm,
        beta_vis=beta_back_vis,
        beta_back=beta_back_vis,
        beta_forward=beta_forward_vis,
        depol_ratio=depol_ratio,
        axis=axis,
        lut_back=lut_back,
        lut_forward=lut_forward,
        lut_depol=lut_depol,
    )

    return {
        'L': L, 'dim': N, 'mesh': (X, Y, Z),
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
# 3. 可视化渲染（离线模式，使用 PyVista + Panel）
# =============================================================================

def render_headless(data, config, output_dir: Path):
    """
    离线渲染三维场和切片，生成交互式 HTML 文件（使用 VTK 后端）。
    输出多个视角的 3D 可视化。
    """
    log_msg(">> [Render] Starting Offline Renderer (Embedded Mode)...")
    try:
        import pyvista as pv
        pv.set_plot_theme("document")
    except ImportError as e:
        log_msg(f"FATAL: 缺少 PyVista: {e}", "error")
        return []

    L = data['L']
    N = int(config.get('grid_dim', 120))
    center = (L/2, L/2, L/2)
    expl_dist = L * config.get('explode_dist', 0.7)
    output_dir.mkdir(parents=True, exist_ok=True)

    field_defs = [
        ('beta_back_vis', 'Backscatter Proxy', 'viridis', ''),
        ('beta_forward_vis', 'Forward Proxy', 'plasma', '__beta_forward'),
        ('depol_ratio', 'Depolarization Proxy', 'cividis', '__depol_ratio'),
    ]
    view_configs = [
        ('render_main', [(3.0*L, 3.0*L, 2.2*L), center, (0, 0, 1)]),
        ('render_bottom', [(center[0], center[1], center[2] - 3.5*L), center, (0, 1, 0)]),
        ('render_front', [(center[0], center[1] - 3.5*L, center[2]), center, (0, 0, 1)]),
        ('render_left', [(center[0] - 3.5*L, center[1], center[2]), center, (0, 0, 1)]),
    ]
    X, Y, Z = data['mesh']
    generated_files = []

    def add_slice(plotter, scalar_name, cmap_name, x, y, z, val, vec_move):
        """添加一个切片平面并平移显示"""
        m = pv.StructuredGrid(x, y, z)
        m.point_data[scalar_name] = val.flatten(order="F")
        m.translate(np.array(vec_move) * expl_dist, inplace=True)
        plotter.add_mesh(m, scalars=scalar_name, cmap=cmap_name, opacity=0.9, show_scalar_bar=False)
        plotter.add_mesh(m.outline(), color="black", line_width=1)

    def build_plotter():
        plotter = pv.Plotter(off_screen=True, window_size=[1000, 800])
        plotter.set_background("white")
        grid = pv.ImageData()
        grid.dimensions = data['beta_ext'].shape
        grid.origin = (0, 0, 0)
        grid.spacing = (L/(N-1), L/(N-1), L/(N-1))
        grid.point_data["Extinction"] = data['beta_ext'].flatten(order="F")
        points = grid.cast_to_pointset()
        ext_threshold = max(float(np.max(data['beta_ext'])) * 0.05, 1e-12)
        valid_points = points.threshold(ext_threshold, scalars="Extinction")
        plotter.add_mesh(valid_points, scalars="Extinction", cmap="jet",
                         style="points", point_size=3.0, render_points_as_spheres=True,
                         opacity="sigmoid", show_scalar_bar=False)
        plotter.add_mesh(grid.outline(), color="grey", line_width=1)
        return plotter

    for field_key, scalar_name, cmap_name, suffix in field_defs:
        field_vis = data[field_key]
        pl = build_plotter()
        add_slice(pl, scalar_name, cmap_name, X[:, :, 0], Y[:, :, 0], Z[:, :, 0], field_vis[:, :, 0], (0, 0, -1))
        add_slice(pl, scalar_name, cmap_name, X[:, 0, :], Y[:, 0, :], Z[:, 0, :], field_vis[:, 0, :], (0, -1, 0))
        add_slice(pl, scalar_name, cmap_name, X[0, :, :], Y[0, :, :], Z[0, :, :], field_vis[0, :, :], (-1, 0, 0))

        for view_stem, cam_pos in view_configs:
            filename = f"{view_stem}{suffix}.html"
            file_path = output_dir / filename
            log_msg(f"   -> Exporting View: {filename}")
            try:
                pl.camera_position = cam_pos
                pl.render()
                vtk_pane = pn.pane.VTK(pl.ren_win, width=1000, height=800,
                                       enable_keybindings=True, orientation_widget=True)
                vtk_pane.save(str(file_path), resources='inline', embed=True, title=f"MieSim - {filename}")
                generated_files.append(filename)
            except Exception as e:
                log_msg(f"ERROR: Export failed for {filename}: {str(e)}", "error")
                traceback.print_exc()
        pl.close()

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
    # TC‑E: 能量守恒（无吸收）
    # ------------------------------------------------------------------
    print("\n[TC‑E] 能量守恒 (无吸收)")
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
    # TC‑A: 吸收守恒
    # ------------------------------------------------------------------
    print("\n[TC‑A] 能量守恒 (有吸收)")
    config_a = config_e.copy()
    layer_a = [{'thickness_m': 20.0, 'visibility_km': 5.0, 'radius_um': 5.0, 'm_real': 1.33, 'm_imag': 0.001}]
    sim_a, _ = run_test_simulation(config_a, layer_a)
    print(f"   R_back = {sim_a['scalars']['R_back']:.6f}")
    print(f"   R_trans = {sim_a['scalars']['R_trans']:.6f}")
    print(f"   R_abs = {sim_a['scalars']['R_abs']:.6f}")
    print(f"   Sum = {sim_a['scalars']['R_back'] + sim_a['scalars']['R_trans'] + sim_a['scalars']['R_abs']:.6f}")

    # ------------------------------------------------------------------
    # TC‑P: 偏振保持
    # ------------------------------------------------------------------
    print("\n[TC‑P] 偏振保持 (线偏振入射)")
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
    # TC‑R: 瑞利极限（波长比）
    # ------------------------------------------------------------------
    print("\n[TC‑R] 瑞利极限 (粒径0.01μm, 比较1.55μm和0.55μm后向回波)")
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
    # 1.55 μm
    config_r1 = config_r_base.copy()
    config_r1['wavelength_um'] = 1.55
    sim_r1, _ = run_test_simulation(config_r1, layer_r, angstrom_q=4.0)
    back_1550 = sim_r1['scalars']['R_back']
    # 0.55 μm
    config_r2 = config_r_base.copy()
    config_r2['wavelength_um'] = 0.55
    sim_r2, _ = run_test_simulation(config_r2, layer_r, angstrom_q=4.0)
    back_550 = sim_r2['scalars']['R_back']
    if back_1550 > 0:
        ratio = back_550 / back_1550
    else:
        ratio = float('inf')
    print(f"   R_back @ 1.55 μm = {back_1550:.6e}")
    print(f"   R_back @ 0.55 μm = {back_550:.6e}")
    print(f"   Ratio (550/1550) = {ratio:.3f}")

    # ------------------------------------------------------------------
    # TC‑G: 几何光学极限（大粒径后向概率）
    # ------------------------------------------------------------------
    print("\n[TC‑G] 几何光学极限 (粒径5μm, 无吸收, 后向散射概率)")
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
    # TC‑L: 对数正态分布积分（体积消光系数）
    # ------------------------------------------------------------------
    print("\n[TC‑L] 对数正态分布积分 (体积消光系数 Bext)")
    try:
        r_g = 2.0
        sigma_ln = 0.35
        sigma_g = np.exp(sigma_ln)
        wl_nm = 1550
        m = 1.33 + 0.0j
        # 手动积分（作为参考）
        r_vals = np.logspace(np.log10(0.01), np.log10(100), 500)
        pdf = (1/(r_vals * sigma_ln * np.sqrt(2*np.pi))) * np.exp(-(np.log(r_vals/r_g))**2/(2*sigma_ln**2))
        pdf /= np.trapz(pdf, r_vals)
        Bext_manual = 0.0
        for r, w in zip(r_vals, pdf):
            q_ext = AutoMieQ(m, wl_nm, 2*r, asDict=False)[0]
            cross = q_ext * np.pi * (r*1e-6)**2
            Bext_manual += w * cross * 1e6
        print(f"   Bext (manual integration) = {Bext_manual:.6e} m⁻¹")
    except Exception as e:
        print(f"   TC‑L 手动积分失败: {e}")

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
        field_meta = build_field_catalog(config)
        if field_meta["requested_field_compute_mode"] != field_meta["effective_field_compute_mode"]:
            log_msg(
                f">> [FieldMode] requested={field_meta['requested_field_compute_mode']} "
                f"but current Mie worker only exports {field_meta['effective_field_compute_mode']}"
            )
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
            source_type="planar",
            source_width_m=field_data['L'],
            sigma_ln=config['sigma_ln'],
            angstrom_q=config.get('angstrom_q', 1.3),
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
