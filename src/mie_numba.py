#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Monte Carlo Simulation (Numba Accelerated)
Updated: 2026-04-07 (Priority 1 optimizations)
Changes:
1. Support lognormal size distribution with sigma_ln.
2. Cache includes sigma_ln.
3. Fixed backscatter intensity weighting (uses actual Stokes I).
4. Added fast return when density grid is all zeros to avoid useless computation.
"""

import time
import numpy as np
from numba import jit, prange

try:
    from mie_core import (
        visibility_to_beta_ext_corrected,
        mie_effective_polarized,
        get_phase_function_cdf,
        generate_adaptive_angles,
        mie_scatter_observables,
        C_LIGHT
    )
except ImportError:
    raise ImportError("Core module 'mie_core.py' not found.")

# =============================================================================
# 辅助函数 (Numba 兼容)
# =============================================================================
@jit(nopython=True, cache=True)
def apply_mueller_numba(stokes, M11, M12, M33, M34):
    """应用 Mueller 矩阵，返回 (新Stokes, 强度缩放因子)"""
    I, Q, U, V = stokes[0], stokes[1], stokes[2], stokes[3]
    if M11 < 1e-20:
        return np.array([1.0, 0.0, 0.0, 0.0]), 1.0
    m12 = M12 / M11
    m33 = M33 / M11
    m34 = M34 / M11
    I_out_rel = 1.0 + m12 * Q
    if I_out_rel <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0]), 0.0
    scale = 1.0 / I_out_rel
    Q_new = (m12 + Q) * scale
    U_new = (m33 * U - m34 * V) * scale
    V_new = (m34 * U + m33 * V) * scale
    pol_sq = Q_new**2 + U_new**2 + V_new**2
    if pol_sq > 1.0:
        s = 1.0 / np.sqrt(pol_sq)
        Q_new *= s
        U_new *= s
        V_new *= s
    return np.array([1.0, Q_new, U_new, V_new]), I_out_rel

@jit(nopython=True, cache=True)
def get_layer_index(z, boundaries):
    """
    给定高度 z，返回所在层索引（二分查找的线性版本）。

    参数：
        z          : 高度 [m]
        boundaries : 层边界数组（升序）

    返回：
        layer_idx  : 层索引（0 基），若不在范围内返回 -1
    """
    if z < boundaries[0] or z > boundaries[-1]:
        return -1
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= z < boundaries[i+1]:
            return i
    return -1


@jit(nopython=True, cache=True)
def sample_scattering_theta_layer(r, theta_rad_grid, cdf_grid_all, layer_idx):
    """
    根据指定层的 CDF 采样散射角（线性插值，Numba 兼容）。

    参数：
        r            : 均匀随机数 [0,1)
        theta_rad_grid : 全局角度网格（弧度）
        cdf_grid_all   : 2D 数组，每层一条 CDF
        layer_idx      : 层索引

    返回：
        散射角（弧度）
    """
    cdf_grid = cdf_grid_all[layer_idx]
    idx = np.searchsorted(cdf_grid, r)
    if idx == 0:
        return theta_rad_grid[0]
    n = len(theta_rad_grid)
    if idx >= n:
        return theta_rad_grid[n-1]
    y0, y1 = cdf_grid[idx-1], cdf_grid[idx]
    x0, x1 = theta_rad_grid[idx-1], theta_rad_grid[idx]
    dy = y1 - y0
    if dy < 1e-12:
        return x0
    return x0 + (r - y0) / dy * (x1 - x0)


@jit(nopython=True, cache=True)
def rotate_stokes_numba(stokes, phi):
    """旋转 Stokes 矢量（Numba 版本，参见 mie_core.rotate_stokes）"""
    I, Q, U, V = stokes[0], stokes[1], stokes[2], stokes[3]
    cs = np.cos(2 * phi)
    sn = np.sin(2 * phi)
    return np.array([I, Q * cs + U * sn, -Q * sn + U * cs, V])


@jit(nopython=True, cache=True)
def apply_mueller(stokes, M11, M12, M33, M34):
    """
    应用 Mueller 矩阵进行单次散射。
    返回 (新 Stokes 矢量, 强度缩放因子 I_out_rel)
    """
    I, Q, U, V = stokes
    if M11 < 1e-20:
        return stokes, 1.0
    m12 = M12 / M11
    m33 = M33 / M11
    m34 = M34 / M11
    I_out_rel = 1.0 + m12 * Q
    if I_out_rel <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0]), 0.0
    scale = 1.0 / I_out_rel
    Q_new = (m12 + Q) * scale
    U_new = (m33 * U - m34 * V) * scale
    V_new = (m34 * U + m33 * V) * scale
    pol_sq = Q_new**2 + U_new**2 + V_new**2
    if pol_sq > 1.0:
        s = 1.0 / np.sqrt(pol_sq)
        Q_new *= s
        U_new *= s
        V_new *= s
    return np.array([1.0, Q_new, U_new, V_new]), I_out_rel


# =============================================================================
# 蒙特卡洛内核 (Numba 并行)
# =============================================================================

import math   # 需要在文件顶部添加

@jit(nopython=True, nogil=True)
def mc_kernel_advanced(
    n_photons,
    layer_boundaries, layer_betas, layer_omegas, layer_mie_ids, beta_max_global,
    incident_angle_rad,
    source_type, source_width_x, source_width_y,
    record_spatial, spatial_grid, spatial_origin, spatial_step,
    use_3d_grid, grid_density, grid_origin, grid_step,
    theta_rad_grid, cdf_grids_all, mie_angles_deg, mie_tables_all
):
    if beta_max_global <= 0:
        return 0, 0, 0, n_photons, 0.0, 0.0, 0.0, 0.0, np.zeros(900, dtype=np.int64)

    total_collisions = 0
    absorbed_count = 0
    back_count = 0
    trans_count = 0

    total_back_I = 0.0
    total_back_Q = 0.0
    total_back_U = 0.0
    total_back_V = 0.0

    l_angle_bins = np.zeros(900, dtype=np.int64)

    init_ux = np.sin(incident_angle_rad)
    init_uz = np.cos(incident_angle_rad)

    if record_spatial:
        s_nx, s_ny = spatial_grid.shape
        s_ox, s_oy = spatial_origin[0], spatial_origin[1]
        s_dx, s_dy = spatial_step[0], spatial_step[1]
    if use_3d_grid:
        g_ox, g_oy, g_oz = grid_origin[0], grid_origin[1], grid_origin[2]
        g_dx, g_dy, g_dz = grid_step[0], grid_step[1], grid_step[2]
        g_nx, g_ny, g_nz = grid_density.shape

    for i in prange(n_photons):
        x, y, z = 0.0, 0.0, 0.0
        if source_type == 1:
            x = (np.random.random() - 0.5) * source_width_x
            y = (np.random.random() - 0.5) * source_width_y

        ux, uy, uz = init_ux, 0.0, init_uz
        stokes = np.array([1.0, 1.0, 0.0, 0.0])
        weight = 1.0

        alive = True
        while alive:
            r_step = np.random.random()
            if r_step < 1e-12:
                r_step = 1e-12
            s_tent = -np.log(r_step) / beta_max_global

            x_new = x + s_tent * ux
            y_new = y + s_tent * uy
            z_new = z + s_tent * uz

            # 后向散射
            if z_new < layer_boundaries[0]:
                back_count += 1
                alive = False

                if record_spatial:
                    # 修复：使用 math.floor 处理负坐标
                    idx_x = int(math.floor((x_new - s_ox) / s_dx))
                    idx_y = int(math.floor((y_new - s_oy) / s_dy))
                    if 0 <= idx_x < s_nx and 0 <= idx_y < s_ny:
                        spatial_grid[idx_x, idx_y] += weight

                cos_a = -uz
                if cos_a > 1.0:
                    cos_a = 1.0
                if cos_a < 0.0:
                    cos_a = 0.0
                idx = int(np.arccos(cos_a) * 1800.0 / np.pi)
                if idx >= 900:
                    idx = 899
                l_angle_bins[idx] += 1

                total_back_I += weight
                total_back_Q += stokes[1] * weight
                total_back_U += stokes[2] * weight
                total_back_V += stokes[3] * weight
                break

            # 透射
            if z_new >= layer_boundaries[-1]:
                trans_count += 1
                alive = False
                break

            layer_idx = get_layer_index(z_new, layer_boundaries)
            if layer_idx == -1:
                continue

            beta_local = layer_betas[layer_idx]
            omega_layer = layer_omegas[layer_idx]
            mie_id = layer_mie_ids[layer_idx]

            if use_3d_grid:
                # 修复：使用 math.floor
                gx = int(math.floor((x_new - g_ox) / g_dx))
                gy = int(math.floor((y_new - g_oy) / g_dy))
                gz = int(math.floor((z_new - g_oz) / g_dz))
                if 0 <= gx < g_nx and 0 <= gy < g_ny and 0 <= gz < g_nz:
                    beta_local *= grid_density[gx, gy, gz]

            x, y, z = x_new, y_new, z_new

            if np.random.random() > (beta_local / beta_max_global):
                continue

            if np.random.random() > omega_layer:
                absorbed_count += 1
                alive = False
            else:
                total_collisions += 1
                r_sca = np.random.random()
                theta_s = sample_scattering_theta_layer(r_sca, theta_rad_grid, cdf_grids_all, mie_id)
                phi_s = np.random.random() * 2 * np.pi

                ux_old, uy_old, uz_old = ux, uy, uz

                stokes = rotate_stokes_numba(stokes, -phi_s)

                deg = theta_s * 180.0 / np.pi
                # ========== Mueller 矩阵线性插值（问题3修复） ==========
                idx1 = np.searchsorted(mie_angles_deg, deg)
                if idx1 == 0:
                    idx0 = 0
                    f = 0.0
                elif idx1 >= len(mie_angles_deg):
                    idx0 = len(mie_angles_deg) - 1
                    idx1 = idx0
                    f = 0.0
                else:
                    idx0 = idx1 - 1
                    f = (deg - mie_angles_deg[idx0]) / (mie_angles_deg[idx1] - mie_angles_deg[idx0])

                M11 = mie_tables_all[mie_id, 0, idx0] * (1 - f) + mie_tables_all[mie_id, 0, idx1] * f
                M12 = mie_tables_all[mie_id, 1, idx0] * (1 - f) + mie_tables_all[mie_id, 1, idx1] * f
                M33 = mie_tables_all[mie_id, 2, idx0] * (1 - f) + mie_tables_all[mie_id, 2, idx1] * f
                M34 = mie_tables_all[mie_id, 3, idx0] * (1 - f) + mie_tables_all[mie_id, 3, idx1] * f

                stokes, I_factor = apply_mueller_numba(stokes, M11, M12, M33, M34)
                weight *= I_factor

                st, ct = np.sin(theta_s), np.cos(theta_s)
                sp, cp = np.sin(phi_s), np.cos(phi_s)
                if np.abs(uz_old) > 0.99999:
                    ux = st * cp
                    uy = st * sp
                    uz = ct * np.sign(uz_old)
                else:
                    sqrt_part = np.sqrt(1 - uz_old * uz_old)
                    n_ux = st * (ux_old * uz_old * cp - uy_old * sp) / sqrt_part + ux_old * ct
                    n_uy = st * (uy_old * uz_old * cp + ux_old * sp) / sqrt_part + uy_old * ct
                    n_uz = -st * cp * sqrt_part + uz_old * ct
                    ux, uy, uz = n_ux, n_uy, n_uz
                norm = np.sqrt(ux*ux + uy*uy + uz*uz)
                ux /= norm
                uy /= norm
                uz /= norm

                # 子午面旋转（之前已添加）
                if np.abs(uz_old) > 0.999999:
                    N_old_x, N_old_y, N_old_z = 0.0, 1.0, 0.0
                else:
                    N_old_x = uy_old
                    N_old_y = -ux_old
                    N_old_z = 0.0
                    n_old_norm = np.sqrt(N_old_x*N_old_x + N_old_y*N_old_y)
                    if n_old_norm > 0:
                        N_old_x /= n_old_norm
                        N_old_y /= n_old_norm

                if np.abs(uz) > 0.999999:
                    N_new_x, N_new_y, N_new_z = 0.0, 1.0, 0.0
                else:
                    N_new_x = uy
                    N_new_y = -ux
                    N_new_z = 0.0
                    n_new_norm = np.sqrt(N_new_x*N_new_x + N_new_y*N_new_y)
                    if n_new_norm > 0:
                        N_new_x /= n_new_norm
                        N_new_y /= n_new_norm

                cos_i = N_old_x*N_new_x + N_old_y*N_new_y + N_old_z*N_new_z
                cross_x = N_old_y*N_new_z - N_old_z*N_new_y
                cross_y = N_old_z*N_new_x - N_old_x*N_new_z
                cross_z = N_old_x*N_new_y - N_old_y*N_new_x
                sin_i = cross_x*ux + cross_y*uy + cross_z*uz
                i = np.arctan2(sin_i, cos_i)
                stokes = rotate_stokes_numba(stokes, -i)

    return (total_collisions, absorbed_count, back_count, trans_count,
            total_back_I, total_back_Q, total_back_U, total_back_V, l_angle_bins)


# =============================================================================
# 仿真主入口 (支持多层、对数正态分布)
# =============================================================================

@jit(nopython=True, parallel=True, nogil=True)
def mc_kernel_advanced_fast(
    n_photons,
    layer_boundaries, layer_betas, layer_omegas, layer_mie_ids, beta_max_global,
    incident_angle_rad,
    source_type, source_width_x, source_width_y,
    use_3d_grid, grid_density, grid_origin, grid_step,
    theta_rad_grid, cdf_grids_all, mie_angles_deg, mie_tables_all
):
    if beta_max_global <= 0:
        return 0, 0, 0, n_photons, 0.0, 0.0, 0.0, 0.0

    total_collisions = 0
    absorbed_count = 0
    back_count = 0
    trans_count = 0

    total_back_I = 0.0
    total_back_Q = 0.0
    total_back_U = 0.0
    total_back_V = 0.0

    init_ux = np.sin(incident_angle_rad)
    init_uz = np.cos(incident_angle_rad)

    if use_3d_grid:
        g_ox, g_oy, g_oz = grid_origin[0], grid_origin[1], grid_origin[2]
        g_dx, g_dy, g_dz = grid_step[0], grid_step[1], grid_step[2]
        g_nx, g_ny, g_nz = grid_density.shape

    for _ in prange(n_photons):
        x, y, z = 0.0, 0.0, 0.0
        if source_type == 1:
            x = (np.random.random() - 0.5) * source_width_x
            y = (np.random.random() - 0.5) * source_width_y

        ux, uy, uz = init_ux, 0.0, init_uz
        stokes = np.array([1.0, 1.0, 0.0, 0.0])
        weight = 1.0

        alive = True
        while alive:
            r_step = np.random.random()
            if r_step < 1e-12:
                r_step = 1e-12
            s_tent = -np.log(r_step) / beta_max_global

            x_new = x + s_tent * ux
            y_new = y + s_tent * uy
            z_new = z + s_tent * uz

            if z_new < layer_boundaries[0]:
                back_count += 1
                total_back_I += weight
                total_back_Q += stokes[1] * weight
                total_back_U += stokes[2] * weight
                total_back_V += stokes[3] * weight
                break

            if z_new >= layer_boundaries[-1]:
                trans_count += 1
                break

            layer_idx = get_layer_index(z_new, layer_boundaries)
            if layer_idx == -1:
                continue

            beta_local = layer_betas[layer_idx]
            omega_layer = layer_omegas[layer_idx]
            mie_id = layer_mie_ids[layer_idx]

            if use_3d_grid:
                gx = int(math.floor((x_new - g_ox) / g_dx))
                gy = int(math.floor((y_new - g_oy) / g_dy))
                gz = int(math.floor((z_new - g_oz) / g_dz))
                if 0 <= gx < g_nx and 0 <= gy < g_ny and 0 <= gz < g_nz:
                    beta_local *= grid_density[gx, gy, gz]

            x, y, z = x_new, y_new, z_new

            if np.random.random() > (beta_local / beta_max_global):
                continue

            if np.random.random() > omega_layer:
                absorbed_count += 1
                break

            total_collisions += 1
            theta_s = sample_scattering_theta_layer(np.random.random(), theta_rad_grid, cdf_grids_all, mie_id)
            phi_s = np.random.random() * 2 * np.pi

            ux_old, uy_old, uz_old = ux, uy, uz
            stokes = rotate_stokes_numba(stokes, -phi_s)

            deg = theta_s * 180.0 / np.pi
            idx1 = np.searchsorted(mie_angles_deg, deg)
            if idx1 == 0:
                idx0 = 0
                f = 0.0
            elif idx1 >= len(mie_angles_deg):
                idx0 = len(mie_angles_deg) - 1
                idx1 = idx0
                f = 0.0
            else:
                idx0 = idx1 - 1
                f = (deg - mie_angles_deg[idx0]) / (mie_angles_deg[idx1] - mie_angles_deg[idx0])

            M11 = mie_tables_all[mie_id, 0, idx0] * (1 - f) + mie_tables_all[mie_id, 0, idx1] * f
            M12 = mie_tables_all[mie_id, 1, idx0] * (1 - f) + mie_tables_all[mie_id, 1, idx1] * f
            M33 = mie_tables_all[mie_id, 2, idx0] * (1 - f) + mie_tables_all[mie_id, 2, idx1] * f
            M34 = mie_tables_all[mie_id, 3, idx0] * (1 - f) + mie_tables_all[mie_id, 3, idx1] * f

            stokes, I_factor = apply_mueller_numba(stokes, M11, M12, M33, M34)
            weight *= I_factor

            st, ct = np.sin(theta_s), np.cos(theta_s)
            sp, cp = np.sin(phi_s), np.cos(phi_s)
            if np.abs(uz_old) > 0.99999:
                ux = st * cp
                uy = st * sp
                uz = ct * np.sign(uz_old)
            else:
                sqrt_part = np.sqrt(1 - uz_old * uz_old)
                n_ux = st * (ux_old * uz_old * cp - uy_old * sp) / sqrt_part + ux_old * ct
                n_uy = st * (uy_old * uz_old * cp + ux_old * sp) / sqrt_part + uy_old * ct
                n_uz = -st * cp * sqrt_part + uz_old * ct
                ux, uy, uz = n_ux, n_uy, n_uz
            norm = np.sqrt(ux*ux + uy*uy + uz*uz)
            ux /= norm
            uy /= norm
            uz /= norm

            if np.abs(uz_old) > 0.999999:
                N_old_x, N_old_y, N_old_z = 0.0, 1.0, 0.0
            else:
                N_old_x = uy_old
                N_old_y = -ux_old
                N_old_z = 0.0
                n_old_norm = np.sqrt(N_old_x*N_old_x + N_old_y*N_old_y)
                if n_old_norm > 0:
                    N_old_x /= n_old_norm
                    N_old_y /= n_old_norm

            if np.abs(uz) > 0.999999:
                N_new_x, N_new_y, N_new_z = 0.0, 1.0, 0.0
            else:
                N_new_x = uy
                N_new_y = -ux
                N_new_z = 0.0
                n_new_norm = np.sqrt(N_new_x*N_new_x + N_new_y*N_new_y)
                if n_new_norm > 0:
                    N_new_x /= n_new_norm
                    N_new_y /= n_new_norm

            cos_i = N_old_x*N_new_x + N_old_y*N_new_y + N_old_z*N_new_z
            cross_x = N_old_y*N_new_z - N_old_z*N_new_y
            cross_y = N_old_z*N_new_x - N_old_x*N_new_z
            cross_z = N_old_x*N_new_y - N_old_y*N_new_x
            sin_i = cross_x*ux + cross_y*uy + cross_z*uz
            i = np.arctan2(sin_i, cos_i)
            stokes = rotate_stokes_numba(stokes, -i)

    return (total_collisions, absorbed_count, back_count, trans_count,
            total_back_I, total_back_Q, total_back_U, total_back_V)


def run_advanced_simulation(
    layers_config, frequency_thz=300, incident_angle_deg=0.0, photons=100000,
    density_grid=None, grid_res_m=10.0,
    source_type="point", source_width_m=0.0,
    record_spatial=False, spatial_res_m=10.0, record_back_hist=False,
    m_real=1.33, m_imag=0.0, angstrom_q=1.3,
    sigma_ln=0.35
):
    """
    高级蒙特卡洛仿真的高层接口，支持：
        - 多层大气（每层独立能见度、粒径、折射率）
        - 对数正态粒径分布（每层可不同）
        - 三维密度网格（云/气溶胶不均匀结构）
        - 平面或点光源
        - 空间后向散射强度分布记录

    工作流程：
        1. 遍历 layers_config，为每一层计算 Mie 参数（缓存复用）。
        2. 根据能见度和 Mie 截面计算自洽的粒子数密度和单次反照率。
        3. 构建全局角度网格和每层的 CDF / Mueller 表。
        4. 若提供密度网格，进行 3D 调制，并更新全局最大消光系数。
        5. 分批调用 Numba 并行内核（避免内存爆炸）。
        6. 汇总结果，计算退偏比。

    参数：
        layers_config      : list of dict，每层包含：
            thickness_m, visibility_km, radius_um, m_real, m_imag,
            size_mode ("mono"/"lognormal"), median_radius_um, sigma_ln
        frequency_thz      : 频率 [THz]
        incident_angle_deg : 入射天顶角 [度]（0 为垂直向下）
        photons            : 总光子数
        density_grid       : 3D numpy array，相对密度（0~1），用于调制消光系数
        grid_res_m         : 密度网格分辨率 [m]（x,y 方向，z 方向由层厚度自动确定）
        source_type        : "point" 或 "planar"
        source_width_m     : 平面光源半宽 [m]
        record_spatial     : 是否记录后向散射空间分布
        spatial_res_m      : 空间记录分辨率 [m]
        m_real, m_imag     : 默认折射率实部/虚部（可被层配置覆盖）
        angstrom_q         : Angström 指数
        sigma_ln           : 默认对数标准差

    返回：
        dict 包含 "scalars" (R_back, R_trans, R_abs, depol) 和 "arrays"
    """
    import sys

    wavelength_m = C_LIGHT / (frequency_thz * 1e12)
    wavelength_nm = wavelength_m * 1e9

    angles_deg = generate_adaptive_angles(num_total=600)

    layer_boundaries, layer_betas, layer_omegas, layer_mie_ids = [0.0], [], [], []
    mie_cache = {}          # 缓存键 -> mie_id
    mie_data_list = []      # 存储 MiePolarizedResult 对象
    current_z = 0.0
    omega_weighted_sum = 0.0
    g_weighted_sum = 0.0
    layer_weight_sum = 0.0
    beta_back_weighted_sum = 0.0
    beta_forward_weighted_sum = 0.0
    depol_back_weighted_sum = 0.0
    depol_forward_weighted_sum = 0.0

    # ---------- 1. 逐层处理，构建光学参数 ----------
    for layer in layers_config:
        th = layer.get("thickness_m", 1000.0)
        vis = layer.get("visibility_km", 5.0)
        radius_um = layer.get("radius_um", 0.5)
        m_r = layer.get("m_real", m_real)
        m_i = layer.get("m_imag", m_imag)
        size_mode = layer.get("size_mode", "mono")
        med_radius = layer.get("median_radius_um", radius_um)
        sig_ln = layer.get("sigma_ln", sigma_ln)
        n_radii = int(layer.get("n_radii", 8 if size_mode == "lognormal" else 1))
        precomputed_mie = layer.get("_mie_res", None)

        # 缓存键：包含所有影响 Mie 结果的参数
        cache_key = (radius_um, m_r, m_i, size_mode, med_radius, sig_ln, n_radii)

        if precomputed_mie is not None:
            mie_id = len(mie_data_list)
            mie_data_list.append(precomputed_mie)
        elif cache_key in mie_cache:
            mie_id = mie_cache[cache_key]
        else:
            m_complex = complex(m_r, m_i)
            mie_res = mie_effective_polarized(
                size_mode=size_mode,
                radius_um=radius_um,
                median_radius_um=med_radius,
                sigma_ln=sig_ln,
                m_complex=m_complex,
                wavelength_m=wavelength_m,
                angles_deg=angles_deg,
                n_radii=n_radii
            )
            mie_id = len(mie_data_list)
            mie_data_list.append(mie_res)
            mie_cache[cache_key] = mie_id

        sigma_ext = mie_data_list[mie_id].sigma_ext   # 单粒子消光截面 [m²]
        sigma_sca = mie_data_list[mie_id].sigma_sca   # 单粒子散射截面 [m²]

        # 根据能见度计算体积消光系数 β_ext [m⁻¹]
        beta = visibility_to_beta_ext_corrected(vis, wavelength_nm, angstrom_q)

        # 自洽计算数密度 N = β_ext / σ_ext，以及单次反照率 ω0 = σ_sca / σ_ext
        if sigma_ext > 1e-30:
            N_number = beta / sigma_ext
            beta_sca = N_number * sigma_sca
            omega0 = beta_sca / beta
        else:
            omega0 = 0.0
        omega0 = max(0.0, min(1.0, omega0))
        scatter_obs = mie_scatter_observables(mie_data_list[mie_id])
        if sigma_ext > 1e-30:
            beta_back_ref = beta * scatter_obs["sigma_back_ref"] / sigma_ext
            beta_forward_ref = beta * scatter_obs["sigma_forward_ref"] / sigma_ext
        else:
            beta_back_ref = 0.0
            beta_forward_ref = 0.0

        current_z += th
        layer_boundaries.append(current_z)
        layer_betas.append(beta)
        layer_omegas.append(omega0)
        layer_mie_ids.append(mie_id)
        weight = beta * th
        omega_weighted_sum += omega0 * weight
        g_weighted_sum += mie_data_list[mie_id].g * weight
        beta_back_weighted_sum += beta_back_ref * th
        beta_forward_weighted_sum += beta_forward_ref * th
        depol_back_weighted_sum += scatter_obs["depol_back"] * weight
        depol_forward_weighted_sum += scatter_obs["depol_forward"] * weight
        layer_weight_sum += weight

    if not mie_data_list:
        raise ValueError("No valid layers found in layers_config")

    base_mie = mie_data_list[0]

    # 转换为 Numpy 数组（供 Numba 使用）
    nb_bounds = np.array(layer_boundaries, dtype=np.float64)
    nb_betas = np.array(layer_betas, dtype=np.float64)
    nb_omegas = np.array(layer_omegas, dtype=np.float64)
    nb_mie_ids = np.array(layer_mie_ids, dtype=np.int64)

    # ---------- 2. 构建每层的 CDF 和 Mueller 表 ----------
    theta_rad_grid, _ = get_phase_function_cdf(base_mie.angles_deg, base_mie.M11)
    cdf_all = np.zeros((len(mie_data_list), len(theta_rad_grid)), dtype=np.float64)
    mie_tabs = np.zeros((len(mie_data_list), 4, len(base_mie.angles_deg)), dtype=np.float64)

    for i, m in enumerate(mie_data_list):
        _, cdf = get_phase_function_cdf(m.angles_deg, m.M11)
        cdf_all[i, :] = cdf
        mie_tabs[i, 0, :] = m.M11
        mie_tabs[i, 1, :] = m.M12
        mie_tabs[i, 2, :] = m.M33
        mie_tabs[i, 3, :] = m.M34

    # ---------- 3. 密度网格处理（3D 不均匀性）----------
    use_grid = False
    nb_gd = np.zeros((1, 1, 1))
    nb_go = np.zeros(3)
    nb_gs = np.zeros(3)
    beta_max = np.max(nb_betas)

    if density_grid is not None:
        use_grid = True
        nb_gd = np.ascontiguousarray(density_grid, dtype=np.float64)
        nx, ny, nz = density_grid.shape
        # 假设密度网格在 z 方向与层边界对齐，步长 = 总厚度 / nz
        total_thickness = layer_boundaries[-1]
        nb_gs = np.array([grid_res_m, grid_res_m, total_thickness / nz], dtype=np.float64)
        nb_go = np.array([-grid_res_m * nx / 2, -grid_res_m * ny / 2, 0.0], dtype=np.float64)
        density_max = np.max(density_grid)
        if density_max > 0:
            beta_max *= density_max   # 全局最大消光系数需覆盖调制后的最大值

    # 快速返回：若密度网格全零且启用，介质透明
    if use_grid and np.max(nb_gd) == 0:
        print(">> [警告] 密度网格全零，介质透明，所有光子透射")
        omega_eff = omega_weighted_sum / layer_weight_sum if layer_weight_sum > 1e-30 else 0.0
        g_eff = g_weighted_sum / layer_weight_sum if layer_weight_sum > 1e-30 else 0.0
        beta_back_eff = beta_back_weighted_sum / current_z if current_z > 1e-30 else 0.0
        beta_forward_eff = beta_forward_weighted_sum / current_z if current_z > 1e-30 else 0.0
        depol_back_eff = depol_back_weighted_sum / layer_weight_sum if layer_weight_sum > 1e-30 else 0.0
        depol_forward_eff = depol_forward_weighted_sum / layer_weight_sum if layer_weight_sum > 1e-30 else 0.0
        fb_ratio_eff = beta_forward_eff / beta_back_eff if beta_back_eff > 1e-30 else 0.0
        return {
            "scalars": {
                "R_back": 0.0,
                "R_trans": 1.0,
                "R_abs": 0.0,
                "depol": 0.0,
                "depol_ratio": 0.0,
                "omega0": omega_eff,
                "g": g_eff,
                "layer_count": len(layer_betas),
                "beta_back_ref": beta_back_eff,
                "beta_forward_ref": beta_forward_eff,
                "forward_back_ratio": fb_ratio_eff,
                "depol_back": depol_back_eff,
                "depol_forward": depol_forward_eff,
            },
            "arrays": {"mc_back_dist": [0]*900, "spatial_grid": np.zeros((1,1))}
        }

    # ---------- 4. 空间记录网格（后向散射强度分布）----------
    spatial_grid = np.zeros((1, 1), dtype=np.float64)
    s_origin = np.zeros(2)
    s_step = np.zeros(2)
    if record_spatial and source_type == "planar":
        sw = source_width_m
        nx = int(sw / spatial_res_m) + 1
        spatial_grid = np.zeros((nx, nx), dtype=np.float64)
        s_step = np.array([spatial_res_m, spatial_res_m])
        s_origin = np.array([-sw / 2, -sw / 2])

    st_code = 1 if source_type == "planar" else 0

    # ---------- 5. 分批运行内核 ----------
    BATCH_SIZE = 1000
    total_photons = int(photons)

    acc_tc = 0
    acc_ac = 0
    acc_bc = 0
    acc_trc = 0
    acc_b_I = 0.0
    acc_b_Q = 0.0
    acc_b_U = 0.0
    acc_b_V = 0.0
    acc_ab = np.zeros(900, dtype=np.int64)

    t0 = time.time()
    processed = 0

    print(f">> [计算中] 开始仿真: 共 {total_photons} 光子, 分组大小: {BATCH_SIZE}", flush=True)

    use_fast_kernel = (not record_spatial) and (not record_back_hist)

    while processed < total_photons:
        current_batch = min(BATCH_SIZE, total_photons - processed)
        if use_fast_kernel:
            res = mc_kernel_advanced_fast(
                current_batch, nb_bounds, nb_betas, nb_omegas, nb_mie_ids, float(beta_max),
                np.deg2rad(incident_angle_deg),
                st_code, float(source_width_m), float(source_width_m),
                use_grid, nb_gd, nb_go, nb_gs,
                theta_rad_grid, cdf_all, base_mie.angles_deg, mie_tabs
            )
            tc, ac, bc, trc, b_I, b_Q, b_U, b_V = res
            ab = None
        else:
            res = mc_kernel_advanced(
                current_batch, nb_bounds, nb_betas, nb_omegas, nb_mie_ids, float(beta_max),
                np.deg2rad(incident_angle_deg),
                st_code, float(source_width_m), float(source_width_m),
                record_spatial, spatial_grid, s_origin, s_step,
                use_grid, nb_gd, nb_go, nb_gs,
                theta_rad_grid, cdf_all, base_mie.angles_deg, mie_tabs
            )
            tc, ac, bc, trc, b_I, b_Q, b_U, b_V, ab = res
        acc_tc += tc
        acc_ac += ac
        acc_bc += bc
        acc_trc += trc
        acc_b_I += b_I
        acc_b_Q += b_Q
        acc_b_U += b_U
        acc_b_V += b_V
        if ab is not None:
            acc_ab += ab

        processed += current_batch
        progress_pct = (processed / total_photons) * 100
        current_group = processed // BATCH_SIZE
        print(f">> [计算中] 第 {current_group} 组 | 进度: {processed}/{total_photons} ({progress_pct:.1f}%) | 累计回波: {acc_bc}", flush=True)

    dt = max(time.time() - t0, 1e-9)
    ns = total_photons if total_photons > 0 else 1

    # 计算退偏比：1 - 平均偏振度（后向散射光）
    if acc_b_I > 0:
        total_pol_mag = np.sqrt(acc_b_Q**2 + acc_b_U**2 + acc_b_V**2)
        avg_pol_degree = total_pol_mag / acc_b_I
        depol = 1.0 - avg_pol_degree
    else:
        depol = 0.0
    omega_eff = omega_weighted_sum / layer_weight_sum if layer_weight_sum > 1e-30 else 0.0
    g_eff = g_weighted_sum / layer_weight_sum if layer_weight_sum > 1e-30 else 0.0
    beta_back_eff = beta_back_weighted_sum / current_z if current_z > 1e-30 else 0.0
    beta_forward_eff = beta_forward_weighted_sum / current_z if current_z > 1e-30 else 0.0
    depol_back_eff = depol_back_weighted_sum / layer_weight_sum if layer_weight_sum > 1e-30 else 0.0
    depol_forward_eff = depol_forward_weighted_sum / layer_weight_sum if layer_weight_sum > 1e-30 else 0.0
    fb_ratio_eff = beta_forward_eff / beta_back_eff if beta_back_eff > 1e-30 else 0.0

    print(f">> [完成] 耗时: {dt:.4f}s | 速度: {total_photons/dt/1e6:.2f} M/s", flush=True)

    return {
        "scalars": {
            "R_back": acc_bc / ns,
            "R_trans": acc_trc / ns,
            "R_abs": acc_ac / ns,
            "depol": depol,
            "depol_ratio": depol,
            "omega0": omega_eff,
            "g": g_eff,
            "layer_count": len(layer_betas),
            "beta_back_ref": beta_back_eff,
            "beta_forward_ref": beta_forward_eff,
            "forward_back_ratio": fb_ratio_eff,
            "depol_back": depol_back_eff,
            "depol_forward": depol_forward_eff,
        },
        "arrays": {
            "mc_back_dist": acc_ab.tolist() if record_back_hist else [],
            "spatial_grid": spatial_grid
        }
    }
