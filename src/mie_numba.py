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


@jit(nopython=True, cache=True)
def sample_density_nearest_numba(density, origin, step, x, y, z):
    nx, ny, nz = density.shape
    fx = (x - origin[0]) / step[0]
    fy = (y - origin[1]) / step[1]
    fz = (z - origin[2]) / step[2]
    ix = int(np.round(fx))
    iy = int(np.round(fy))
    iz = int(np.round(fz))
    if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
        return 0.0
    return density[ix, iy, iz]


@jit(nopython=True, cache=True)
def voxel_index_numba(origin, step, shape_arr, x, y, z):
    fx = (x - origin[0]) / step[0]
    fy = (y - origin[1]) / step[1]
    fz = (z - origin[2]) / step[2]
    ix = int(np.round(fx))
    iy = int(np.round(fy))
    iz = int(np.round(fz))
    if ix < 0 or ix >= shape_arr[0] or iy < 0 or iy >= shape_arr[1] or iz < 0 or iz >= shape_arr[2]:
        return -1, -1, -1
    return ix, iy, iz


@jit(nopython=True, cache=True)
def slab_index_numba(z, z_edges):
    n = z_edges.shape[0] - 1
    if n <= 1:
        return 0
    if z <= z_edges[0]:
        return 0
    if z >= z_edges[n]:
        return n - 1
    idx = np.searchsorted(z_edges, z, side="right") - 1
    if idx < 0:
        return 0
    if idx >= n:
        return n - 1
    return idx


@jit(nopython=True, cache=True)
def distance_to_slab_boundary_numba(z, uz, slab_idx, z_edges):
    if uz > 1e-12:
        return (z_edges[slab_idx + 1] - z) / uz
    if uz < -1e-12:
        return (z_edges[slab_idx] - z) / uz
    return np.inf


@jit(nopython=True, cache=True)
def direction_to_scattering_angles_numba(ux, uy, uz, dxo, dyo, dzo):
    ct = ux * dxo + uy * dyo + uz * dzo
    if ct > 1.0:
        ct = 1.0
    if ct < -1.0:
        ct = -1.0
    theta = np.arccos(ct)
    if np.abs(uz) > 0.99999:
        return theta, np.arctan2(dyo, dxo)

    sq = np.sqrt(max(1.0 - uz * uz, 1e-20))
    e1x = ux * uz / sq
    e1y = uy * uz / sq
    e1z = -sq
    e2x = -uy / sq
    e2y = ux / sq
    e2z = 0.0
    px = dxo - ct * ux
    py = dyo - ct * uy
    pz = dzo - ct * uz
    a = px * e1x + py * e1y + pz * e1z
    b = px * e2x + py * e2y + pz * e2z
    return theta, np.arctan2(b, a)


@jit(nopython=True, cache=True)
def interpolate_mueller_numba(mie_angles_deg, mie_tables_all, mie_id, theta_deg):
    idx1 = np.searchsorted(mie_angles_deg, theta_deg)
    if idx1 == 0:
        idx0 = 0
        f = 0.0
    elif idx1 >= len(mie_angles_deg):
        idx0 = len(mie_angles_deg) - 1
        idx1 = idx0
        f = 0.0
    else:
        idx0 = idx1 - 1
        denom = mie_angles_deg[idx1] - mie_angles_deg[idx0]
        f = 0.0 if denom <= 1e-12 else (theta_deg - mie_angles_deg[idx0]) / denom

    return (
        mie_tables_all[mie_id, 0, idx0] * (1.0 - f) + mie_tables_all[mie_id, 0, idx1] * f,
        mie_tables_all[mie_id, 1, idx0] * (1.0 - f) + mie_tables_all[mie_id, 1, idx1] * f,
        mie_tables_all[mie_id, 2, idx0] * (1.0 - f) + mie_tables_all[mie_id, 2, idx1] * f,
        mie_tables_all[mie_id, 3, idx0] * (1.0 - f) + mie_tables_all[mie_id, 3, idx1] * f,
    )


@jit(nopython=True, cache=True)
def local_beta_numba(use_3d_grid, grid_density, grid_origin, grid_step,
                     layer_boundaries, layer_betas, x, y, z):
    layer_idx = get_layer_index(z, layer_boundaries)
    if layer_idx == -1:
        return 0.0
    beta = layer_betas[layer_idx]
    if use_3d_grid:
        beta *= sample_density_nearest_numba(grid_density, grid_origin, grid_step, x, y, z)
    return beta


@jit(nopython=True, cache=True)
def escape_transmittance_numba(use_3d_grid, grid_density, grid_origin, grid_step,
                               layer_boundaries, layer_betas, thickness,
                               x, y, z, ux, uy, uz):
    if np.abs(uz) <= 1e-12:
        return 0.0
    if uz > 0.0:
        s_exit = (thickness - z) / uz
    else:
        s_exit = -z / uz
    if s_exit <= 0.0:
        return 0.0

    min_step = grid_step[0]
    if grid_step[1] < min_step:
        min_step = grid_step[1]
    if grid_step[2] < min_step:
        min_step = grid_step[2]
    step_len = max(min_step * 0.75, 1e-3)
    n_steps = max(1, int(np.ceil(s_exit / step_len)))
    ds = s_exit / n_steps
    tau = 0.0
    for i in range(n_steps):
        s_mid = (i + 0.5) * ds
        xm = x + s_mid * ux
        ym = y + s_mid * uy
        zm = z + s_mid * uz
        tau += local_beta_numba(
            use_3d_grid, grid_density, grid_origin, grid_step,
            layer_boundaries, layer_betas, xm, ym, zm
        ) * ds
    return np.exp(-tau)


@jit(nopython=True, cache=True)
def distance_to_z_exit_numba(thickness, z, uz):
    if np.abs(uz) <= 1e-12:
        return -1.0
    if uz > 0.0:
        s_exit = (thickness - z) / uz
    else:
        s_exit = -z / uz
    return s_exit if s_exit > 0.0 else -1.0


@jit(nopython=True, cache=True)
def receiver_overlap_numba(range_m, overlap_min, overlap_full_range_m):
    if overlap_full_range_m <= 1e-9:
        return 1.0
    frac = range_m / overlap_full_range_m
    if frac < 0.0:
        frac = 0.0
    if frac > 1.0:
        frac = 1.0
    return overlap_min + (1.0 - overlap_min) * frac


@jit(nopython=True, cache=True)
def accumulate_detector_contribution_numba(
    forward_I, forward_Q, forward_U, forward_V,
    back_I, back_Q, back_U, back_V, event_count,
    ix, iy, iz, stokes, weight, ux, uy, uz, mie_id,
    use_3d_grid, grid_density, grid_origin, grid_step,
    layer_boundaries, layer_betas, thickness,
    x, y, z, mie_angles_deg, mie_tables_all,
    forward_dirs, forward_weights, back_dirs, back_weights,
    path_length, collect_lidar_observation,
    echo_I, echo_Q, echo_U, echo_V, echo_event_count, echo_weight_sum, echo_weight_sq_sum,
    range_bin_width_m, range_max_m, overlap_min, overlap_full_range_m
):
    event_count[ix, iy, iz] += 1.0

    for i in range(forward_weights.shape[0]):
        dxo = forward_dirs[i, 0]
        dyo = forward_dirs[i, 1]
        dzo = forward_dirs[i, 2]
        theta, phi = direction_to_scattering_angles_numba(ux, uy, uz, dxo, dyo, dzo)
        M11, M12, M33, M34 = interpolate_mueller_numba(
            mie_angles_deg, mie_tables_all, mie_id, theta * 180.0 / np.pi
        )
        stokes_rot = rotate_stokes_numba(stokes, -phi)
        stokes_out, i_factor = apply_mueller_numba(stokes_rot, M11, M12, M33, M34)
        if i_factor <= 0.0:
            continue
        trans = escape_transmittance_numba(
            use_3d_grid, grid_density, grid_origin, grid_step,
            layer_boundaries, layer_betas, thickness,
            x, y, z, dxo, dyo, dzo
        )
        contrib = weight * i_factor * trans * forward_weights[i]
        forward_I[ix, iy, iz] += contrib
        forward_Q[ix, iy, iz] += contrib * stokes_out[1]
        forward_U[ix, iy, iz] += contrib * stokes_out[2]
        forward_V[ix, iy, iz] += contrib * stokes_out[3]

    for i in range(back_weights.shape[0]):
        dxo = back_dirs[i, 0]
        dyo = back_dirs[i, 1]
        dzo = back_dirs[i, 2]
        theta, phi = direction_to_scattering_angles_numba(ux, uy, uz, dxo, dyo, dzo)
        M11, M12, M33, M34 = interpolate_mueller_numba(
            mie_angles_deg, mie_tables_all, mie_id, theta * 180.0 / np.pi
        )
        stokes_rot = rotate_stokes_numba(stokes, -phi)
        stokes_out, i_factor = apply_mueller_numba(stokes_rot, M11, M12, M33, M34)
        if i_factor <= 0.0:
            continue
        trans = escape_transmittance_numba(
            use_3d_grid, grid_density, grid_origin, grid_step,
            layer_boundaries, layer_betas, thickness,
            x, y, z, dxo, dyo, dzo
        )
        contrib = weight * i_factor * trans * back_weights[i]
        back_I[ix, iy, iz] += contrib
        back_Q[ix, iy, iz] += contrib * stokes_out[1]
        back_U[ix, iy, iz] += contrib * stokes_out[2]
        back_V[ix, iy, iz] += contrib * stokes_out[3]

        if collect_lidar_observation:
            s_exit = distance_to_z_exit_numba(thickness, z, dzo)
            if s_exit > 0.0:
                range_m = 0.5 * (path_length + s_exit)
                if range_m >= 0.0 and range_m <= range_max_m:
                    bin_idx = int(np.floor(range_m / range_bin_width_m))
                    if bin_idx >= 0 and bin_idx < echo_I.shape[0]:
                        overlap = receiver_overlap_numba(range_m, overlap_min, overlap_full_range_m)
                        echo_contrib = contrib * overlap
                        echo_I[bin_idx] += echo_contrib
                        echo_Q[bin_idx] += echo_contrib * stokes_out[1]
                        echo_U[bin_idx] += echo_contrib * stokes_out[2]
                        echo_V[bin_idx] += echo_contrib * stokes_out[3]
                        echo_event_count[bin_idx] += 1.0
                        echo_weight_sum[bin_idx] += echo_contrib
                        echo_weight_sq_sum[bin_idx] += echo_contrib * echo_contrib


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
                beta_local *= sample_density_nearest_numba(
                    grid_density, grid_origin, grid_step, x_new, y_new, z_new
                )

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
    z_edges, slab_betas, slab_eps,
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
            beta_majorant = beta_max_global
            s_boundary = np.inf
            if use_3d_grid:
                slab_idx = slab_index_numba(z, z_edges)
                beta_majorant = slab_betas[slab_idx]
                s_boundary = distance_to_slab_boundary_numba(z, uz, slab_idx, z_edges)
                if beta_majorant <= 1e-30:
                    if np.isfinite(s_boundary):
                        s_move = s_boundary + slab_eps
                        x_new = x + s_move * ux
                        y_new = y + s_move * uy
                        z_new = z + s_move * uz
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
                        x, y, z = x_new, y_new, z_new
                        continue
                    beta_majorant = beta_max_global

            r_step = np.random.random()
            if r_step < 1e-12:
                r_step = 1e-12
            s_tent = -np.log(r_step) / beta_majorant

            if use_3d_grid and np.isfinite(s_boundary) and s_tent >= s_boundary:
                s_move = s_boundary + slab_eps
                x_new = x + s_move * ux
                y_new = y + s_move * uy
                z_new = z + s_move * uz
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
                x, y, z = x_new, y_new, z_new
                continue

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
                beta_local *= sample_density_nearest_numba(
                    grid_density, grid_origin, grid_step, x_new, y_new, z_new
                )

            x, y, z = x_new, y_new, z_new

            if np.random.random() > (beta_local / beta_majorant):
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


@jit(nopython=True, nogil=True)
def mc_kernel_advanced_exact(
    n_photons,
    layer_boundaries, layer_betas, layer_omegas, layer_mie_ids, beta_max_global,
    incident_angle_rad,
    source_type, source_width_x, source_width_y,
    use_3d_grid, grid_density, grid_origin, grid_step,
    z_edges, slab_betas, slab_eps,
    theta_rad_grid, cdf_grids_all, mie_angles_deg, mie_tables_all,
    forward_dirs, forward_weights, back_dirs, back_weights,
    forward_I, forward_Q, forward_U, forward_V,
    back_I, back_Q, back_U, back_V, event_count,
    collect_lidar_observation,
    echo_I, echo_Q, echo_U, echo_V, echo_event_count, echo_weight_sum, echo_weight_sq_sum,
    range_bin_width_m, range_max_m, overlap_min, overlap_full_range_m
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
    g_nx, g_ny, g_nz = grid_density.shape
    shape_arr = np.array([g_nx, g_ny, g_nz], dtype=np.int64)
    thickness = layer_boundaries[-1]

    for _ in range(n_photons):
        x, y, z = 0.0, 0.0, 0.0
        if source_type == 1:
            x = (np.random.random() - 0.5) * source_width_x
            y = (np.random.random() - 0.5) * source_width_y

        ux, uy, uz = init_ux, 0.0, init_uz
        stokes = np.array([1.0, 1.0, 0.0, 0.0])
        weight = 1.0
        path_length = 0.0

        alive = True
        while alive:
            beta_majorant = beta_max_global
            s_boundary = np.inf
            if use_3d_grid:
                slab_idx = slab_index_numba(z, z_edges)
                beta_majorant = slab_betas[slab_idx]
                s_boundary = distance_to_slab_boundary_numba(z, uz, slab_idx, z_edges)
                if beta_majorant <= 1e-30:
                    if np.isfinite(s_boundary):
                        s_move = s_boundary + slab_eps
                        x_new = x + s_move * ux
                        y_new = y + s_move * uy
                        z_new = z + s_move * uz
                        path_length += s_move
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
                        x, y, z = x_new, y_new, z_new
                        continue
                    beta_majorant = beta_max_global

            r_step = np.random.random()
            if r_step < 1e-12:
                r_step = 1e-12
            s_tent = -np.log(r_step) / beta_majorant

            if use_3d_grid and np.isfinite(s_boundary) and s_tent >= s_boundary:
                s_move = s_boundary + slab_eps
                x_new = x + s_move * ux
                y_new = y + s_move * uy
                z_new = z + s_move * uz
                path_length += s_move
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
                x, y, z = x_new, y_new, z_new
                continue

            x_new = x + s_tent * ux
            y_new = y + s_tent * uy
            z_new = z + s_tent * uz
            path_length += s_tent

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
                beta_local *= sample_density_nearest_numba(
                    grid_density, grid_origin, grid_step, x_new, y_new, z_new
                )

            x, y, z = x_new, y_new, z_new

            if np.random.random() > (beta_local / beta_majorant):
                continue

            if np.random.random() > omega_layer:
                absorbed_count += 1
                break

            total_collisions += 1

            if use_3d_grid:
                vx, vy, vz = voxel_index_numba(grid_origin, grid_step, shape_arr, x, y, z)
                if vx >= 0:
                    accumulate_detector_contribution_numba(
                        forward_I, forward_Q, forward_U, forward_V,
                        back_I, back_Q, back_U, back_V, event_count,
                        vx, vy, vz, stokes, weight, ux, uy, uz, mie_id,
                        use_3d_grid, grid_density, grid_origin, grid_step,
                        layer_boundaries, layer_betas, thickness,
                        x, y, z, mie_angles_deg, mie_tables_all,
                        forward_dirs, forward_weights, back_dirs, back_weights,
                        path_length, collect_lidar_observation,
                        echo_I, echo_Q, echo_U, echo_V, echo_event_count, echo_weight_sum, echo_weight_sq_sum,
                        range_bin_width_m, range_max_m, overlap_min, overlap_full_range_m
                    )

            theta_s = sample_scattering_theta_layer(np.random.random(), theta_rad_grid, cdf_grids_all, mie_id)
            phi_s = np.random.random() * 2 * np.pi

            ux_old, uy_old, uz_old = ux, uy, uz
            stokes = rotate_stokes_numba(stokes, -phi_s)

            M11, M12, M33, M34 = interpolate_mueller_numba(
                mie_angles_deg, mie_tables_all, mie_id, theta_s * 180.0 / np.pi
            )
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
            norm = np.sqrt(ux * ux + uy * uy + uz * uz)
            ux /= norm
            uy /= norm
            uz /= norm

            if np.abs(uz_old) > 0.999999:
                N_old_x, N_old_y, N_old_z = 0.0, 1.0, 0.0
            else:
                N_old_x = uy_old
                N_old_y = -ux_old
                N_old_z = 0.0
                n_old_norm = np.sqrt(N_old_x * N_old_x + N_old_y * N_old_y)
                if n_old_norm > 0:
                    N_old_x /= n_old_norm
                    N_old_y /= n_old_norm

            if np.abs(uz) > 0.999999:
                N_new_x, N_new_y, N_new_z = 0.0, 1.0, 0.0
            else:
                N_new_x = uy
                N_new_y = -ux
                N_new_z = 0.0
                n_new_norm = np.sqrt(N_new_x * N_new_x + N_new_y * N_new_y)
                if n_new_norm > 0:
                    N_new_x /= n_new_norm
                    N_new_y /= n_new_norm

            cos_i = N_old_x * N_new_x + N_old_y * N_new_y + N_old_z * N_new_z
            cross_x = N_old_y * N_new_z - N_old_z * N_new_y
            cross_y = N_old_z * N_new_x - N_old_x * N_new_z
            cross_z = N_old_x * N_new_y - N_old_y * N_new_x
            sin_i = cross_x * ux + cross_y * uy + cross_z * uz
            i = np.arctan2(sin_i, cos_i)
            stokes = rotate_stokes_numba(stokes, -i)

    return (total_collisions, absorbed_count, back_count, trans_count,
            total_back_I, total_back_Q, total_back_U, total_back_V)


def build_detector_cone(axis="forward", half_angle_deg=90.0, n_polar=2, n_azimuth=6):
    half_angle = np.deg2rad(np.clip(float(half_angle_deg), 0.1, 90.0))
    mu_min = np.cos(half_angle)
    dirs = []
    weights = []
    n_polar = max(int(n_polar), 1)
    n_azimuth = max(int(n_azimuth), 1)
    for it in range(n_polar):
        mu_hi = 1.0 - it * (1.0 - mu_min) / n_polar
        mu_lo = 1.0 - (it + 1) * (1.0 - mu_min) / n_polar
        mu_mid = 0.5 * (mu_hi + mu_lo)
        theta = np.arccos(np.clip(mu_mid, -1.0, 1.0))
        ring_weight = 2.0 * np.pi * (mu_hi - mu_lo)
        for ip in range(n_azimuth):
            phi = 2.0 * np.pi * (ip + 0.5) / n_azimuth
            sx = np.sin(theta) * np.cos(phi)
            sy = np.sin(theta) * np.sin(phi)
            sz = np.cos(theta) if axis == "forward" else -np.cos(theta)
            dirs.append((sx, sy, sz))
            weights.append(ring_weight / n_azimuth)
    weights_arr = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights_arr))
    if total > 0:
        weights_arr /= total
    return np.asarray(dirs, dtype=np.float64), weights_arr


def build_centered_z_edges(thickness, nz):
    if nz <= 1:
        return np.asarray([0.0, float(thickness)], dtype=np.float64)
    centers = np.linspace(0.0, float(thickness), int(nz), dtype=np.float64)
    edges = np.empty(int(nz) + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[-1] = float(thickness)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    return edges


def build_z_slab_majorants(density_grid, z_edges, layer_boundaries, layer_betas):
    density = np.asarray(density_grid, dtype=np.float64)
    nz = density.shape[2]
    slab_betas = np.zeros(nz, dtype=np.float64)
    bounds = np.asarray(layer_boundaries, dtype=np.float64)
    betas = np.asarray(layer_betas, dtype=np.float64)
    for iz in range(nz):
        density_max = float(np.max(density[:, :, iz]))
        if density_max <= 0.0:
            continue
        lo = float(z_edges[iz])
        hi = float(z_edges[iz + 1])
        beta_max = 0.0
        for li, beta in enumerate(betas):
            layer_lo = float(bounds[li])
            layer_hi = float(bounds[li + 1])
            if hi >= layer_lo and lo <= layer_hi:
                beta_max = max(beta_max, float(beta))
        slab_betas[iz] = density_max * beta_max
    return slab_betas


def run_advanced_simulation(
    layers_config, frequency_thz=300, incident_angle_deg=0.0, photons=100000,
    density_grid=None, grid_res_m=10.0,
    source_type="point", source_width_m=0.0,
    record_spatial=False, spatial_res_m=10.0, record_back_hist=False,
    m_real=1.33, m_imag=0.0, angstrom_q=1.3,
    sigma_ln=0.35, collect_voxel_fields=False,
    field_forward_half_angle_deg=90.0, field_back_half_angle_deg=90.0,
    field_quadrature_polar=2, field_quadrature_azimuth=6,
    collect_lidar_observation=False,
    range_bin_width_m=1.0, range_max_m=None,
    receiver_overlap_min=1.0, receiver_overlap_full_range_m=0.0
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
    nb_z_edges = np.asarray([0.0, 1.0], dtype=np.float64)
    nb_slab_betas = np.asarray([0.0], dtype=np.float64)
    slab_eps = 0.0
    beta_max = np.max(nb_betas)

    if density_grid is not None:
        use_grid = True
        nb_gd = np.ascontiguousarray(density_grid, dtype=np.float64)
        nx, ny, nz = density_grid.shape
        total_thickness = layer_boundaries[-1]
        z_step = total_thickness / max(nz - 1, 1)
        nb_gs = np.array([grid_res_m, grid_res_m, z_step], dtype=np.float64)
        nb_go = np.array([
            -0.5 * grid_res_m * max(nx - 1, 0),
            -0.5 * grid_res_m * max(ny - 1, 0),
            0.0
        ], dtype=np.float64)
        density_max = np.max(density_grid)
        if density_max > 0:
            beta_max *= density_max   # 全局最大消光系数需覆盖调制后的最大值
        nb_z_edges = build_centered_z_edges(total_thickness, nz)
        nb_slab_betas = build_z_slab_majorants(nb_gd, nb_z_edges, nb_bounds, nb_betas)
        slab_eps = max(z_step * 1e-6, 1e-9)
        slab_beta_max = float(np.max(nb_slab_betas))
        if slab_beta_max > 0:
            beta_max = slab_beta_max

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
        arrays = {"mc_back_dist": [0]*900, "spatial_grid": np.zeros((1,1))}
        if collect_voxel_fields:
            zero_fields = np.zeros_like(nb_gd, dtype=np.float64)
            arrays["voxel_fields"] = {
                "forward_I": zero_fields.copy(),
                "forward_Q": zero_fields.copy(),
                "forward_U": zero_fields.copy(),
                "forward_V": zero_fields.copy(),
                "back_I": zero_fields.copy(),
                "back_Q": zero_fields.copy(),
                "back_U": zero_fields.copy(),
                "back_V": zero_fields.copy(),
                "event_count": zero_fields.copy(),
            }
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
            "arrays": arrays
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

    use_exact_kernel = bool(collect_voxel_fields and use_grid)
    use_fast_kernel = (not use_exact_kernel) and (not record_spatial) and (not record_back_hist)
    BATCH_SIZE = 8192 if (use_exact_kernel or use_fast_kernel) else 1000

    print(f">> [计算中] 开始仿真: 共 {total_photons} 光子, 分组大小: {BATCH_SIZE}", flush=True)
    if use_exact_kernel:
        forward_dirs, forward_weights = build_detector_cone(
            "forward", field_forward_half_angle_deg,
            field_quadrature_polar, field_quadrature_azimuth
        )
        back_dirs, back_weights = build_detector_cone(
            "back", field_back_half_angle_deg,
            field_quadrature_polar, field_quadrature_azimuth
        )
        exact_forward_I = np.zeros_like(nb_gd, dtype=np.float64)
        exact_forward_Q = np.zeros_like(nb_gd, dtype=np.float64)
        exact_forward_U = np.zeros_like(nb_gd, dtype=np.float64)
        exact_forward_V = np.zeros_like(nb_gd, dtype=np.float64)
        exact_back_I = np.zeros_like(nb_gd, dtype=np.float64)
        exact_back_Q = np.zeros_like(nb_gd, dtype=np.float64)
        exact_back_U = np.zeros_like(nb_gd, dtype=np.float64)
        exact_back_V = np.zeros_like(nb_gd, dtype=np.float64)
        exact_event_count = np.zeros_like(nb_gd, dtype=np.float64)
    else:
        forward_dirs = np.zeros((1, 3), dtype=np.float64)
        forward_weights = np.ones(1, dtype=np.float64)
        back_dirs = np.zeros((1, 3), dtype=np.float64)
        back_weights = np.ones(1, dtype=np.float64)
        exact_forward_I = exact_forward_Q = exact_forward_U = exact_forward_V = None
        exact_back_I = exact_back_Q = exact_back_U = exact_back_V = None
        exact_event_count = None

    lidar_enabled = bool(collect_lidar_observation and use_exact_kernel)
    rbw = max(float(range_bin_width_m), 1e-9)
    rmax = float(range_max_m) if range_max_m is not None else float(current_z)
    rmax = max(rmax, rbw)
    n_range_bins = max(1, int(np.ceil(rmax / rbw)))
    echo_I = np.zeros(n_range_bins, dtype=np.float64)
    echo_Q = np.zeros(n_range_bins, dtype=np.float64)
    echo_U = np.zeros(n_range_bins, dtype=np.float64)
    echo_V = np.zeros(n_range_bins, dtype=np.float64)
    echo_event_count = np.zeros(n_range_bins, dtype=np.float64)
    echo_weight_sum = np.zeros(n_range_bins, dtype=np.float64)
    echo_weight_sq_sum = np.zeros(n_range_bins, dtype=np.float64)
    overlap_min = max(0.0, min(1.0, float(receiver_overlap_min)))
    overlap_full = max(0.0, float(receiver_overlap_full_range_m))

    while processed < total_photons:
        current_batch = min(BATCH_SIZE, total_photons - processed)
        if use_exact_kernel:
            res = mc_kernel_advanced_exact(
                current_batch, nb_bounds, nb_betas, nb_omegas, nb_mie_ids, float(beta_max),
                np.deg2rad(incident_angle_deg),
                st_code, float(source_width_m), float(source_width_m),
                use_grid, nb_gd, nb_go, nb_gs,
                nb_z_edges, nb_slab_betas, float(slab_eps),
                theta_rad_grid, cdf_all, base_mie.angles_deg, mie_tabs,
                forward_dirs, forward_weights, back_dirs, back_weights,
                exact_forward_I, exact_forward_Q, exact_forward_U, exact_forward_V,
                exact_back_I, exact_back_Q, exact_back_U, exact_back_V, exact_event_count,
                lidar_enabled,
                echo_I, echo_Q, echo_U, echo_V, echo_event_count, echo_weight_sum, echo_weight_sq_sum,
                rbw, rmax, overlap_min, overlap_full
            )
            tc, ac, bc, trc, b_I, b_Q, b_U, b_V = res
            ab = None
        elif use_fast_kernel:
            res = mc_kernel_advanced_fast(
                current_batch, nb_bounds, nb_betas, nb_omegas, nb_mie_ids, float(beta_max),
                np.deg2rad(incident_angle_deg),
                st_code, float(source_width_m), float(source_width_m),
                use_grid, nb_gd, nb_go, nb_gs,
                nb_z_edges, nb_slab_betas, float(slab_eps),
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

    arrays = {
        "mc_back_dist": acc_ab.tolist() if record_back_hist else [],
        "spatial_grid": spatial_grid
    }
    if use_exact_kernel:
        inv_n = 1.0 / ns
        arrays["voxel_fields"] = {
            "forward_I": exact_forward_I * inv_n,
            "forward_Q": exact_forward_Q * inv_n,
            "forward_U": exact_forward_U * inv_n,
            "forward_V": exact_forward_V * inv_n,
            "back_I": exact_back_I * inv_n,
            "back_Q": exact_back_Q * inv_n,
            "back_U": exact_back_U * inv_n,
            "back_V": exact_back_V * inv_n,
            "event_count": exact_event_count,
        }
    if lidar_enabled:
        inv_n = 1.0 / ns
        echo_I_n = echo_I * inv_n
        echo_Q_n = echo_Q * inv_n
        echo_U_n = echo_U * inv_n
        echo_V_n = echo_V * inv_n
        echo_depol = np.zeros_like(echo_I_n)
        mask = echo_I_n > 1e-30
        echo_depol[mask] = np.clip(
            1.0 - np.sqrt(echo_Q_n[mask] ** 2 + echo_U_n[mask] ** 2 + echo_V_n[mask] ** 2) / echo_I_n[mask],
            0.0,
            1.0,
        )
        echo_power_variance = echo_weight_sq_sum * inv_n * inv_n
        echo_power_std = np.sqrt(echo_power_variance)
        echo_power_ci_low = np.maximum(echo_I_n - 1.96 * echo_power_std, 0.0)
        echo_power_ci_high = echo_I_n + 1.96 * echo_power_std
        echo_relative_error = np.zeros_like(echo_I_n)
        power_mask = echo_I_n > 1e-30
        echo_relative_error[power_mask] = echo_power_std[power_mask] / echo_I_n[power_mask]
        count_mask = (~power_mask) & (echo_event_count > 0.0)
        echo_relative_error[count_mask] = 1.0 / np.sqrt(echo_event_count[count_mask])
        arrays["lidar_observation"] = {
            "range_bins_m": (np.arange(n_range_bins, dtype=np.float64) + 0.5) * rbw,
            "echo_I": echo_I_n,
            "echo_Q": echo_Q_n,
            "echo_U": echo_U_n,
            "echo_V": echo_V_n,
            "echo_power": echo_I_n.copy(),
            "echo_depol": echo_depol,
            "echo_event_count": echo_event_count,
            "echo_weight_sum": echo_weight_sum * inv_n,
            "echo_weight_sq_sum": echo_weight_sq_sum * inv_n,
            "echo_power_variance_est": echo_power_variance,
            "echo_power_ci_low": echo_power_ci_low,
            "echo_power_ci_high": echo_power_ci_high,
            "echo_relative_error_est": echo_relative_error,
            "receiver_model": {
                "range_bin_width_m": rbw,
                "range_max_m": rmax,
                "receiver_mode": "backscatter",
                "overlap_model": "linear",
                "overlap_min": overlap_min,
                "overlap_full_range_m": overlap_full,
                "source_range_path": "event_path_plus_escape_over_two",
            },
        }

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
        "arrays": arrays
    }
