#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D Mie + Monte Carlo (Polarized & Vertical Profile Version)
Fixed & Optimized: 2026-04-07 (Priority 1 optimizations)
Changes:
1. Replaced MieQ with AutoMieQ for stability and Rayleigh limit.
2. Adaptive non-uniform angular sampling for forward peak.
3. Fixed MatrixElements fallback error.
4. Exported AutoMieQ for external test use.
"""

import time
from dataclasses import dataclass
import numpy as np
import scipy.integrate

# =============================================================================
# 兼容性修补：处理 SciPy 版本差异（trapz / trapezoid, simps / simpson）
# =============================================================================
if not hasattr(scipy.integrate, 'trapz'):
    if hasattr(scipy.integrate, 'trapezoid'):
        scipy.integrate.trapz = scipy.integrate.trapezoid
    else:
        scipy.integrate.trapz = np.trapezoid

if not hasattr(scipy.integrate, 'simps'):
    from scipy.integrate import simpson
    scipy.integrate.simps = simpson

try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:
    from scipy.integrate import cumtrapz as cumulative_trapezoid

import PyMieScatt as PMS

# 导出 AutoMieQ 供外部测试使用
AutoMieQ = PMS.AutoMieQ

C_LIGHT = 299792458.0  # 真空光速 [m/s]

# ===================== 1. 物理模型辅助函数 =====================

def visibility_to_beta_ext_corrected(visibility_km: float, wavelength_nm: float,
                                     angstrom_q: float = 1.3) -> float:
    """
    根据能见度计算大气消光系数（基于 Angström 指数修正）。

    物理原理：
        能见度 V 定义为大气对 550 nm 光的消光系数 β_550 满足：
            β_550 = 3.912 / V   (Koschmieder 公式)
        对于其他波长 λ，使用 Angström 经验关系：
            β_λ = β_550 * (550 / λ)^q
        其中 q 为 Angström 指数（典型值 1.3，表征气溶胶粒径分布）。

    参数：
        visibility_km : float  能见度 [km]
        wavelength_nm : float  波长 [nm]
        angstrom_q    : float  Angström 指数（默认 1.3）

    返回：
        beta_ext_surf : float  消光系数 [m⁻¹]
    """
    beta_550 = 3.912 / (visibility_km * 1000.0)   # 550 nm 消光系数 [m⁻¹]
    if wavelength_nm <= 0:
        return beta_550
    return beta_550 * ((550.0 / wavelength_nm) ** angstrom_q)


def lognormal_pdf(r: np.ndarray, r_med: float, sigma_ln: float) -> np.ndarray:
    """
    对数正态分布概率密度函数（用于气溶胶/云滴粒径分布）。

    数学定义：
        f(r) = 1/(r σ_ln √(2π)) * exp[ - (ln r - ln r_med)² / (2 σ_ln²) ]

    参数：
        r       : np.ndarray  半径 [μm]
        r_med   : float       中值半径 [μm]（几何平均半径）
        sigma_ln: float       对数标准差（无量纲）

    返回：
        pdf     : np.ndarray  概率密度 [μm⁻¹]
    """
    r = np.asarray(r)
    mask = r > 0
    pdf = np.zeros_like(r, dtype=float)
    if sigma_ln <= 1e-6:
        return pdf
    mu = np.log(r_med)                     # 对数均值
    pdf[mask] = (np.exp(-0.5 * ((np.log(r[mask]) - mu) / sigma_ln) ** 2)
                 / (r[mask] * sigma_ln * np.sqrt(2 * np.pi)))
    return pdf


def generate_adaptive_angles(num_total=600, forward_res=0.01, forward_max=2.0):
    """
    生成非均匀散射角度网格，在前向小角度区域加密采样。

    原理：
        Mie 散射相函数在前向具有尖锐的衍射峰，角度分辨率不足会导致
        蒙特卡洛采样误差。本函数在 [0, forward_max] 度内使用均匀高分辨率，
        之后线性递增至 180°。

    参数：
        num_total   : int    总角度点数
        forward_res : float  前向区域角度分辨率 [度]
        forward_max : float  前向区域最大角度 [度]

    返回：
        angles_deg  : np.ndarray  角度数组（度，升序，唯一）
    """
    forward_angles = np.arange(0, forward_max + forward_res, forward_res)
    if forward_angles[-1] > forward_max:
        forward_angles = forward_angles[:-1]
    remaining = np.linspace(forward_max, 180, max(2, num_total - len(forward_angles)))
    angles = np.concatenate([forward_angles, remaining[1:]])
    return np.unique(angles)


# ===================== 2. 偏振 Mie 散射核心 =====================

@dataclass
class MiePolarizedResult:
    """存储偏振 Mie 散射计算结果的数据类"""
    sigma_ext: float          # 消光截面 [m²]
    sigma_sca: float          # 散射截面 [m²]
    g: float                  # 不对称因子 g = <cosθ>
    angles_deg: np.ndarray    # 角度网格（度）
    M11: np.ndarray           # Mueller 矩阵元素 M11（相函数）
    M12: np.ndarray           # Mueller 矩阵元素 M12（偏振相关）
    M33: np.ndarray           # Mueller 矩阵元素 M33
    M34: np.ndarray           # Mueller 矩阵元素 M34



def interpolate_angular_table(angles_deg: np.ndarray, values: np.ndarray, angle_deg: float) -> float:
    """对角度表做线性插值，超出范围时钳制到端点。"""
    if len(angles_deg) == 0:
        return 0.0
    return float(np.interp(angle_deg, angles_deg, values))


def sigma_backscatter_reference(mie_res: MiePolarizedResult) -> float:
    """由归一化相函数估计 180° 后向参考散射截面。"""
    if mie_res.sigma_sca <= 0:
        return 0.0
    m11_back = interpolate_angular_table(mie_res.angles_deg, mie_res.M11, 180.0)
    return float(mie_res.sigma_sca * m11_back / (4.0 * np.pi))


def cone_average_metric(angles_deg: np.ndarray, values: np.ndarray, cone_deg: float) -> float:
    """瀵瑰皬瑙掗敟鍐呯殑鍊兼寜 solid-angle 鏉冮噸鍋氬钩鍧囥€?"""
    theta_deg = np.asarray(angles_deg, dtype=float)
    theta_rad = np.deg2rad(theta_deg)
    values = np.asarray(values, dtype=float)

    cone_deg = max(float(cone_deg), 0.01)
    mask = theta_deg <= cone_deg + 1e-12
    if np.count_nonzero(mask) < 2:
        return float(values[0]) if len(values) else 0.0

    theta_sel = theta_rad[mask]
    value_sel = values[mask]
    denom = scipy.integrate.trapz(np.sin(theta_sel), theta_sel)
    if denom <= 1e-20:
        return float(value_sel[0])
    return float(scipy.integrate.trapz(value_sel * np.sin(theta_sel), theta_sel) / denom)


def sigma_forward_reference(mie_res: MiePolarizedResult, forward_cone_deg: float = 2.0) -> float:
    """鐢卞皬鍓嶅悜閿ュ啿骞冲潎鐩稿嚱鏁颁及绠?鍓嶅悜鍙傝€冩暎灏勬埅闈€?"""
    if mie_res.sigma_sca <= 0:
        return 0.0
    m11_forward = cone_average_metric(mie_res.angles_deg, mie_res.M11, forward_cone_deg)
    return float(mie_res.sigma_sca * m11_forward / (4.0 * np.pi))


def safe_depol_ratio(m11: float, m12: float) -> float:
    """瀹夊叏璁＄畻 depol = (M11 - M12) / (M11 + M12)銆?"""
    denom = float(m11 + m12)
    if not np.isfinite(denom) or abs(denom) <= 1e-20:
        return 0.0
    ratio = float((m11 - m12) / denom)
    if not np.isfinite(ratio):
        return 0.0
    return min(max(ratio, 0.0), 1.0)


def mie_scatter_observables(mie_res: MiePolarizedResult, forward_cone_deg: float = 2.0) -> dict:
    """浠?Mie 缁撴灉涓彁鍙栧墠鍚?鍚庡悜/閫€鍋忕瓑涓婂眰瑙傛祴閲忋€?"""
    phase_m11_back = interpolate_angular_table(mie_res.angles_deg, mie_res.M11, 180.0)
    phase_m11_forward = cone_average_metric(mie_res.angles_deg, mie_res.M11, forward_cone_deg)
    phase_m12_back = interpolate_angular_table(mie_res.angles_deg, mie_res.M12, 180.0)
    phase_m12_forward = cone_average_metric(mie_res.angles_deg, mie_res.M12, forward_cone_deg)

    sigma_back_ref = sigma_backscatter_reference(mie_res)
    sigma_forward_ref = sigma_forward_reference(mie_res, forward_cone_deg=forward_cone_deg)
    forward_back_ratio = sigma_forward_ref / sigma_back_ref if sigma_back_ref > 1e-30 else 0.0

    return {
        "phase_m11_back": float(phase_m11_back),
        "phase_m11_forward": float(phase_m11_forward),
        "sigma_back_ref": float(sigma_back_ref),
        "sigma_forward_ref": float(sigma_forward_ref),
        "forward_back_ratio": float(forward_back_ratio),
        "depol_back": float(safe_depol_ratio(phase_m11_back, phase_m12_back)),
        "depol_forward": float(safe_depol_ratio(phase_m11_forward, phase_m12_forward)),
    }


def phase_halfspace_integrals(angles_deg: np.ndarray, intensity: np.ndarray,
                              split_deg: float = 90.0) -> tuple[float, float]:
    """按物理角度积分前向/后向半空间相函数。"""
    theta_deg = np.asarray(angles_deg, dtype=float)
    theta_rad = np.deg2rad(theta_deg)
    values = np.maximum(np.asarray(intensity, dtype=float), 0.0)

    fwd_mask = theta_deg <= split_deg
    back_mask = theta_deg >= split_deg
    if np.count_nonzero(fwd_mask) < 2 or np.count_nonzero(back_mask) < 2:
        return 0.0, 0.0

    fwd = scipy.integrate.simps(values[fwd_mask] * np.sin(theta_rad[fwd_mask]), theta_rad[fwd_mask])
    back = scipy.integrate.simps(values[back_mask] * np.sin(theta_rad[back_mask]), theta_rad[back_mask])
    return float(fwd), float(back)


def mie_effective_polarized(
    size_mode, radius_um, median_radius_um, sigma_ln,
    m_complex, wavelength_m, angles_deg=None, n_radii=50
):
    """
    计算给定粒径分布下的有效偏振 Mie 参数（消光/散射截面、Mueller 矩阵）。
    修复：所有积分均使用梯形法则，包括 Mueller 矩阵的粒径积分。
    """
    wavelength_nm = wavelength_m * 1e9

    if angles_deg is None:
        angles_deg = generate_adaptive_angles(num_total=600)

    # ---------- 1. 构建粒径分布 ----------
    if size_mode == "lognormal":
        if median_radius_um <= 0:
            median_radius_um = radius_um
        log_r_min = np.log(median_radius_um) - 4.0 * sigma_ln
        log_r_max = np.log(median_radius_um) + 4.0 * sigma_ln
        r_grid = np.exp(np.linspace(log_r_min, log_r_max, n_radii))
        weights = lognormal_pdf(r_grid, median_radius_um, sigma_ln)
        total_w = scipy.integrate.trapz(weights, r_grid)
        if total_w > 0:
            weights /= total_w
    else:  # mono
        r_grid = np.array([radius_um])
        weights = np.array([1.0])

    # ---------- 2. 收集各半径的贡献 ----------
    r_list = []
    w_list = []
    qext_list = []
    qsca_list = []
    g_qsca_cross_list = []
    cross_list = []
    # Mueller 矩阵列表（每个元素是一维数组，长度 = len(angles_deg)）
    m11_list = []
    m12_list = []
    m33_list = []
    m34_list = []

    mu_vals = np.cos(np.deg2rad(angles_deg))
    theta_rad = np.deg2rad(angles_deg)

    for r, w in zip(r_grid, weights):
        if w <= 0:
            continue

        diameter_um = 2.0 * r
        diameter_nm = diameter_um * 1000.0
        x = np.pi * diameter_nm / wavelength_nm

        try:
            qext, qsca, qabs, g_val, qpr, qback, qratio = AutoMieQ(
                m_complex, wavelength_nm, diameter_nm, asDict=False
            )
        except Exception:
            continue

        cross_section = np.pi * (r * 1e-6) ** 2
        r_list.append(r)
        w_list.append(w)
        qext_list.append(qext)
        qsca_list.append(qsca)
        g_qsca_cross_list.append(g_val * qsca * cross_section)
        cross_list.append(cross_section)

        # ----- Mueller 矩阵 -----
        if x < 0.01:
            # 瑞利散射矩阵
            mu = np.cos(theta_rad)
            m11_arr = (1 + mu**2)
            m12_arr = (1 - mu**2)
            m33_arr = 2 * mu
            m34_arr = np.zeros_like(mu)
        else:
            try:
                nmax = int(round(2 + x + 4 * (x ** (1/3))))
                an, bn = PMS.Mie_ab(m_complex, x)
                m11_arr = np.zeros(len(angles_deg))
                m12_arr = np.zeros(len(angles_deg))
                m33_arr = np.zeros(len(angles_deg))
                m34_arr = np.zeros(len(angles_deg))
                for i, mu in enumerate(mu_vals):
                    pin, taun = PMS.MiePiTau(mu, nmax)
                    n = np.arange(1, nmax + 1)
                    n2 = (2 * n + 1) / (n * (n + 1))
                    S1 = np.sum(n2 * (an * pin + bn * taun))
                    S2 = np.sum(n2 * (an * taun + bn * pin))
                    s1_sq = np.abs(S1) ** 2
                    s2_sq = np.abs(S2) ** 2
                    m11_arr[i] = 0.5 * (s2_sq + s1_sq)
                    m12_arr[i] = 0.5 * (s2_sq - s1_sq)
                    m33_arr[i] = 0.5 * np.real(S2 * np.conj(S1) + np.conj(S2) * S1)
                    # 修正 M34 符号
                    m34_arr[i] = -0.5 * np.imag(S1 * np.conj(S2) - S2 * np.conj(S1))
            except Exception:
                m11_arr.fill(0.0)
                m12_arr.fill(0.0)
                m33_arr.fill(0.0)
                m34_arr.fill(0.0)

        m11_list.append(m11_arr)
        m12_list.append(m12_arr)
        m33_list.append(m33_arr)
        m34_list.append(m34_arr)

    # ---------- 3. 积分得到有效截面 ----------
    if not r_list:
        return MiePolarizedResult(
            0.0, 0.0, 0.0, np.asarray(angles_deg, dtype=float),
            np.ones(len(angles_deg), dtype=float),
            np.zeros(len(angles_deg), dtype=float),
            np.zeros(len(angles_deg), dtype=float),
            np.zeros(len(angles_deg), dtype=float)
        )

    r_arr = np.array(r_list)
    weights_valid = np.array(w_list, dtype=float)
    valid_total_w = np.trapezoid(weights_valid, r_arr)
    if valid_total_w > 0:
        weights_valid /= valid_total_w
    else:
        weights_valid = np.ones_like(r_arr, dtype=float)
        weights_valid /= max(len(weights_valid), 1)
    qext_arr = np.array(qext_list)
    qsca_arr = np.array(qsca_list)
    g_qsca_cross_arr = np.array(g_qsca_cross_list)
    cross_arr = np.array(cross_list)

    if len(r_arr) == 1:
        sigma_ext_eff = float(weights_valid[0] * qext_arr[0] * cross_arr[0])
        sigma_sca_eff = float(weights_valid[0] * qsca_arr[0] * cross_arr[0])
        g_eff_sum = float(weights_valid[0] * g_qsca_cross_arr[0])
    else:
        sigma_ext_eff = np.trapezoid(weights_valid * qext_arr * cross_arr, r_arr)
        sigma_sca_eff = np.trapezoid(weights_valid * qsca_arr * cross_arr, r_arr)
        g_eff_sum = np.trapezoid(weights_valid * g_qsca_cross_arr, r_arr)
    g_final = g_eff_sum / sigma_sca_eff if sigma_sca_eff > 1e-20 else 0.0

    # ---------- 4. 积分 Mueller 矩阵（关键修复）----------
    # 将权重和半径步长考虑进去，沿粒径轴 (axis=0) 梯形积分
    # weights 形状 (n_r,)，需要广播到 (n_r, n_angles)
    weights_2d = weights_valid[:, np.newaxis]
    if len(r_arr) == 1:
        M11_sum = weights_2d[0, 0] * np.array(m11_list[0], dtype=float)
        M12_sum = weights_2d[0, 0] * np.array(m12_list[0], dtype=float)
        M33_sum = weights_2d[0, 0] * np.array(m33_list[0], dtype=float)
        M34_sum = weights_2d[0, 0] * np.array(m34_list[0], dtype=float)
    else:
        M11_sum = np.trapezoid(weights_2d * np.array(m11_list), r_arr, axis=0)
        M12_sum = np.trapezoid(weights_2d * np.array(m12_list), r_arr, axis=0)
        M33_sum = np.trapezoid(weights_2d * np.array(m33_list), r_arr, axis=0)
        M34_sum = np.trapezoid(weights_2d * np.array(m34_list), r_arr, axis=0)

    # ---------- 5. 归一化相函数 ----------
    M11_sum = np.maximum(M11_sum, 0.0)
    norm_integral = np.trapezoid(M11_sum * np.sin(theta_rad), theta_rad)
    if norm_integral > 1e-20:
        normalization_factor = 2.0 / norm_integral
        M11_norm = M11_sum * normalization_factor
        M12_norm = M12_sum * normalization_factor
        M33_norm = M33_sum * normalization_factor
        M34_norm = M34_sum * normalization_factor
    else:
        M11_norm = np.ones_like(M11_sum)
        M12_norm = np.zeros_like(M12_sum)
        M33_norm = np.zeros_like(M33_sum)
        M34_norm = np.zeros_like(M34_sum)

    return MiePolarizedResult(
        sigma_ext_eff, sigma_sca_eff, g_final, angles_deg,
        M11_norm, M12_norm, M33_norm, M34_norm
    )

# ===================== 3. 蒙特卡洛传输（偏振 + 垂直剖面） =====================

@dataclass
class MCStatsPolarized:
    """蒙特卡洛仿真统计结果"""
    avg_collisions: float          # 平均碰撞次数
    absorbed_ratio: float          # 吸收光子比例
    backscatter_ratio: float       # 后向散射（返回 z<0）比例
    transmit_ratio: float          # 透射比例
    depolarization_ratio: float    # 退偏比（1 - 平均偏振度）
    backscatter_angle_dist: np.ndarray  # 后向散射角度分布（0-90°, 0.1° 分辨率）


def rotate_stokes(stokes, phi):
    """
    Stokes 矢量绕传播方向旋转角度 φ。

    物理原理：
        当参考系旋转 φ 时，Stokes 矢量变换为：
            I' = I
            Q' = Q cos(2φ) + U sin(2φ)
            U' = -Q sin(2φ) + U cos(2φ)
            V' = V

    参数：
        stokes : [I, Q, U, V]
        phi    : 旋转角 [rad]

    返回：
        旋转后的 Stokes 矢量
    """
    I, Q, U, V = stokes
    cs = np.cos(2 * phi)
    sn = np.sin(2 * phi)
    Q_new = Q * cs + U * sn
    U_new = -Q * sn + U * cs
    return np.array([I, Q_new, U_new, V])


def apply_mueller(stokes, M11, M12, M33, M34):
    """
    应用 Mueller 矩阵进行单次散射，更新 Stokes 矢量（归一化到 I=1）。

    算法：
        1. 计算归一化 Mueller 矩阵元素 m12 = M12/M11, m33 = M33/M11, m34 = M34/M11。
        2. 出射强度 I_out ∝ 1 + m12 Q_in。
        3. 为避免发散，将结果归一化使 I_out = 1，其他分量相应缩放。
        4. 对偏振度进行裁剪（避免数值误差导致 >1）。

    参数：
        stokes : 入射 Stokes 矢量 [I, Q, U, V]（假设 I 已归一化）
        M11,M12,M33,M34 : Mueller 矩阵元素（未归一化，与相函数成比例）

    返回：
        出射 Stokes 矢量（I=1）
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
    # 数值稳定性：裁剪偏振度 > 1
    pol_sq = Q_new**2 + U_new**2 + V_new**2
    if pol_sq > 1.0:
        s = 1.0 / np.sqrt(pol_sq)
        Q_new *= s
        U_new *= s
        V_new *= s
    return np.array([1.0, Q_new, U_new, V_new]), I_out_rel


def monte_carlo_stats_polarized_profile(
    beta_ext_surf, omega0, thickness_m, scale_height_m,
    n_photons, theta_rad_grid, cdf_grid,
    mie_res: MiePolarizedResult
):
    """
    垂直指数衰减大气中的偏振蒙特卡洛传输（单层，消光系数随高度指数下降）。

    物理模型：
        消光系数 β(z) = β_surf * exp(-z / H)，其中 H 为标高。
        使用“最大横截法”（majorant）处理空间非均匀介质：
            1. 以全局最大 β_max 抽样自由路径 s_tent。
            2. 在试探步终点，以概率 β_local/β_max 接受该次碰撞。
        散射角分布由 Mie 相函数 CDF 采样。

    参数：
        beta_ext_surf  : float  地表消光系数 [m⁻¹]
        omega0         : float  单次散射反照率
        thickness_m    : float  介质厚度 [m]（z 从 0 到 thickness_m）
        scale_height_m : float  标高 [m]（指数衰减常数）
        n_photons      : int    光子数
        theta_rad_grid : np.ndarray 散射角网格（弧度）
        cdf_grid       : np.ndarray 相函数累积分布函数
        mie_res        : MiePolarizedResult  包含 Mueller 矩阵表

    返回：
        MCStatsPolarized 对象
    """
    rng = np.random.default_rng()
    total_collisions = 0
    absorbed_count = 0
    back_count = 0
    trans_count = 0
    angle_bins = np.zeros(900, dtype=int)   # 0~90°, 0.1° 分辨率

    total_back_I = 0.0
    total_back_Q = 0.0
    total_back_U = 0.0
    total_back_V = 0.0

    beta_max = beta_ext_surf
    mie_angles = mie_res.angles_deg
    M11_arr = mie_res.M11
    M12_arr = mie_res.M12
    M33_arr = mie_res.M33
    M34_arr = mie_res.M34

    for _ in range(n_photons):
        x, y, z = 0.0, 0.0, 0.0
        ux, uy, uz = 0.0, 0.0, 1.0          # 初始方向垂直向下（z 正向为向下）
        stokes = np.array([1.0, 1.0, 0.0, 0.0])  # 线偏振光（Q=1）
        weight = 1.0

        alive = True
        while alive:
            # ----- 抽样试探步长 -----
            r_step = rng.random()
            if r_step < 1e-12:
                r_step = 1e-12
            s_tent = -np.log(r_step) / beta_max

            x_new = x + s_tent * ux
            y_new = y + s_tent * uy
            z_new = z + s_tent * uz

            # ----- 边界处理：后向散射（z < 0）-----
            if z_new < 0:
                back_count += 1
                alive = False
                cos_a = -uz
                if cos_a > 1.0:
                    cos_a = 1.0
                if cos_a < 0.0:
                    cos_a = 0.0
                idx = int(np.degrees(np.arccos(cos_a)) * 10)
                if idx >= 900:
                    idx = 899
                angle_bins[idx] += 1

                total_back_I += weight
                total_back_Q += stokes[1] * weight
                total_back_U += stokes[2] * weight
                total_back_V += stokes[3] * weight
                break

            # ----- 边界处理：透射（z >= thickness）-----
            if z_new > thickness_m:
                trans_count += 1
                alive = False
                break

            x, y, z = x_new, y_new, z_new

            # ----- 局部消光系数及接受/拒绝 -----
            if scale_height_m > 0:
                beta_local = beta_ext_surf * np.exp(-z / scale_height_m)
            else:
                beta_local = beta_ext_surf

            if rng.random() > (beta_local / beta_max):
                continue   # 未发生碰撞

            # ----- 发生碰撞：吸收或散射 -----
            if rng.random() > omega0:
                absorbed_count += 1
                alive = False
            else:
                total_collisions += 1
                theta_s = sample_scattering_theta(rng, theta_rad_grid, cdf_grid)
                phi_s = rng.random() * 2 * np.pi

                # 旋转至散射参考系
                stokes = rotate_stokes(stokes, -phi_s)

                theta_deg = np.degrees(theta_s)
                idx1 = np.searchsorted(mie_angles, theta_deg)
                if idx1 == 0:
                    idx0 = 0
                    f = 0.0
                elif idx1 >= len(mie_angles):
                    idx0 = len(mie_angles) - 1
                    idx1 = idx0
                    f = 0.0
                else:
                    idx0 = idx1 - 1
                    denom = mie_angles[idx1] - mie_angles[idx0]
                    f = 0.0 if denom <= 1e-12 else (theta_deg - mie_angles[idx0]) / denom

                stokes, i_factor = apply_mueller(
                    stokes,
                    M11_arr[idx0] * (1 - f) + M11_arr[idx1] * f,
                    M12_arr[idx0] * (1 - f) + M12_arr[idx1] * f,
                    M33_arr[idx0] * (1 - f) + M33_arr[idx1] * f,
                    M34_arr[idx0] * (1 - f) + M34_arr[idx1] * f
                )
                weight *= i_factor

                # 更新方向矢量（局部坐标变换）
                sin_t, cos_t = np.sin(theta_s), np.cos(theta_s)
                sin_p, cos_p = np.sin(phi_s), np.cos(phi_s)

                if abs(uz) > 0.99999:
                    # 避免除以零：当方向几乎垂直时简化
                    ux = sin_t * cos_p
                    uy = sin_t * sin_p
                    uz = cos_t * np.sign(uz)
                else:
                    sqrt_part = np.sqrt(1 - uz * uz)
                    n_ux = sin_t * (ux * uz * cos_p - uy * sin_p) / sqrt_part + ux * cos_t
                    n_uy = sin_t * (uy * uz * cos_p + ux * sin_p) / sqrt_part + uy * cos_t
                    n_uz = -sin_t * cos_p * sqrt_part + uz * cos_t
                    ux, uy, uz = n_ux, n_uy, n_uz

                norm = np.sqrt(ux * ux + uy * uy + uz * uz)
                ux /= norm
                uy /= norm
                uz /= norm

    avg_pol_retention = (
        np.sqrt(total_back_Q**2 + total_back_U**2 + total_back_V**2) / total_back_I
    ) if total_back_I > 0 else 1.0
    depol_ratio = 1.0 - avg_pol_retention
    n_safe = n_photons if n_photons > 0 else 1

    return MCStatsPolarized(
        total_collisions / n_safe,
        absorbed_count / n_safe,
        back_count / n_safe,
        trans_count / n_safe,
        depol_ratio,
        angle_bins
    )


def sample_scattering_theta(rng, theta_rad_grid, cdf_grid):
    """
    根据累积分布函数 CDF 抽样散射角（线性插值）。

    参数：
        rng            : 随机数生成器
        theta_rad_grid : 角度网格（弧度，升序）
        cdf_grid       : 对应的 CDF（单调递增，0→1）

    返回：
        散射角（弧度）
    """
    r = rng.random()
    idx = np.searchsorted(cdf_grid, r)
    if idx == 0:
        return theta_rad_grid[0]
    if idx >= len(theta_rad_grid):
        return theta_rad_grid[-1]
    y0, y1 = cdf_grid[idx - 1], cdf_grid[idx]
    x0, x1 = theta_rad_grid[idx - 1], theta_rad_grid[idx]
    if y1 - y0 < 1e-10:
        return x0
    return x0 + (r - y0) / (y1 - y0) * (x1 - x0)


def run_simulation(
    visibility_km, frequency_thz, radius_um=0.5,
    m_real=1.33, m_imag=0.0, photons=5000,
    size_mode="mono", median_radius_um=0.5, sigma_ln=0.35,
    thickness_m=1000.0, scale_height_m=2000.0,
    angstrom_q=1.3, verbose=False
):
    """
    单层大气偏振蒙特卡洛仿真的高层封装（供外部调用）。

    流程：
        1. 计算波长，地表消光系数。
        2. 计算 Mie 偏振参数（粒径分布）。
        3. 构建散射角 CDF。
        4. 运行蒙特卡洛仿真。
        5. 返回统计量和部分数组。

    返回：
        dict 包含 "scalars" 和 "arrays"
    """
    m_complex = complex(m_real, m_imag)
    wavelength_m = C_LIGHT / (frequency_thz * 1e12)
    beta_ext_surf = visibility_to_beta_ext_corrected(visibility_km, wavelength_m * 1e9, angstrom_q)

    angles_deg = generate_adaptive_angles(num_total=600)
    mie_eff = mie_effective_polarized(
        size_mode, radius_um, median_radius_um, sigma_ln,
        m_complex, wavelength_m, angles_deg
    )

    sigma_ext = mie_eff.sigma_ext
    if sigma_ext > 1e-20:
        omega0 = mie_eff.sigma_sca / sigma_ext
    else:
        omega0 = 0.0
    omega0 = max(0.0, min(1.0, omega0))

    theta_rad_grid, cdf_grid = get_phase_function_cdf(mie_eff.angles_deg, mie_eff.M11)

    t0 = time.time()
    mc = monte_carlo_stats_polarized_profile(
        beta_ext_surf, omega0, thickness_m, scale_height_m,
        photons, theta_rad_grid, cdf_grid, mie_eff
    )
    dt = time.time() - t0

    # 计算 Mie 相函数的前向/后向积分（用于诊断）
    mie_int_fwd, mie_int_back = phase_halfspace_integrals(angles_deg, mie_eff.M11)
    scatter_obs = mie_scatter_observables(mie_eff)

    return {
        "scalars": {
            "beta_ext_surf": beta_ext_surf,
            "omega0": omega0,
            "g": mie_eff.g,
            "avg_collisions": mc.avg_collisions,
            "R_back": mc.backscatter_ratio,
            "R_trans": mc.transmit_ratio,
            "R_abs": mc.absorbed_ratio,
            "depol_ratio": mc.depolarization_ratio,
            "time": dt,
            "mie_int_back": mie_int_back,
            "mie_int_fwd": mie_int_fwd,
            "sigma_back_ref": scatter_obs["sigma_back_ref"],
            "sigma_forward_ref": scatter_obs["sigma_forward_ref"],
            "forward_back_ratio": scatter_obs["forward_back_ratio"],
            "depol_back": scatter_obs["depol_back"],
            "depol_forward": scatter_obs["depol_forward"],
            "phase_m11_back": scatter_obs["phase_m11_back"],
            "phase_m11_forward": scatter_obs["phase_m11_forward"]
        },
        "arrays": {
            "mie_M11_profile": mie_eff.M11[::10].tolist(),
            "mc_back_dist": mc.backscatter_angle_dist.tolist()
        }
    }


def get_phase_function_cdf(angles_deg, intensity):
    """
    从相函数（M11）生成散射角的累积分布函数。

    物理：
        PDF(θ) ∝ M11(θ) sinθ
        CDF(θ) = ∫₀^θ PDF(θ') dθ' / ∫₀^π PDF(θ') dθ'

    参数：
        angles_deg : 角度数组（度）
        intensity  : M11 值

    返回：
        theta_rad_grid : 弧度网格
        cdf_grid       : 累积分布函数（0→1）
    """
    theta_rad = np.deg2rad(angles_deg)
    intensity = np.maximum(intensity, 0)
    pdf = intensity * np.sin(theta_rad)
    cdf = cumulative_trapezoid(pdf, theta_rad, initial=0)
    if cdf[-1] > 0:
        cdf /= cdf[-1]
    else:
        cdf = np.linspace(0, 1, len(cdf))
    return theta_rad, cdf
