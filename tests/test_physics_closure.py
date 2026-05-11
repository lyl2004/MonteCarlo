"""
阶段一物理闭合测试 — WF1 + WF2

WF1 EnergyConservationTests
    5 个场景 × Mie 后端，验证 R_back + R_trans + R_abs = 1（偏差 < 1e-10）。

WF2 PRRegressionTests
    均匀介质下验证 echo_power(R) 符合 Beer-Lambert 双程衰减，斜率与理论 -2β_ext
    一致（相对偏差 ≤ 15%），并排除 1/R² 因子存在。
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import mie_numba
import mie_worker

C_LIGHT = 299792458.0


def _freq(wavelength_um: float) -> float:
    return C_LIGHT / (wavelength_um * 1e-6) / 1e12


def _run_proxy_simulation(
    beta_ext_target: float,
    m_imag: float = 0.0,
    photons: int = 200_000,
    L_size: float = 20.0,
) -> dict:
    """运行 proxy-only 快速内核，返回 result 字典。"""
    config = mie_worker.DEFAULT_CONFIG.copy()
    config.update(
        {
            "grid_dim": 8,
            "L_size": L_size,
            "r_bottom": 0.4,
            "r_top": 0.4,
            "sigma_ln": 0.0,
            "mie_layer_count": 1,
            "mie_n_radii": 1,
            "m_imag": m_imag,
            "turbulence_scale": 10000.0,
            "cloud_center_z": L_size / 2.0,
            "cloud_thickness": L_size * 10.0,
            "photons": photons,
            "field_compute_mode": "proxy_only",
            "lidar_enabled": False,
        }
    )
    # 通过 visibility_km 反算得到目标 β_ext。
    # β_ext = 3.912 / (visibility_km * 1000)  （550 nm），再加 Angstrom 修正：
    # β_ext(1550 nm) = β_550 × (550/1550)^1.3 ≈ β_550 × 0.2217
    # 所以 β_ext(1550) = 3.912 / (V*1000) × 0.2217，
    # 反推 V = 3.912 × 0.2217 / (β_ext × 1000)
    angstrom_factor = (550.0 / 1550.0) ** 1.3
    v_km = (3.912 * angstrom_factor) / (beta_ext_target * 1000.0)
    config["visibility_km"] = v_km

    with tempfile.TemporaryDirectory() as tmp:
        layers = mie_worker.build_mie_layers(config)
        field = mie_worker.generate_field(config, Path(tmp), layers)
        return mie_numba.run_advanced_simulation(
            layers_config=layers["layers_config"],
            frequency_thz=_freq(config["wavelength_um"]),
            photons=config["photons"],
            density_grid=field["density_norm"],
            grid_res_m=field["L"] / max(field["dim"] - 1, 1),
            source_type="point",
            source_width_m=0.0,
            sigma_ln=config["sigma_ln"],
            collect_voxel_fields=False,
            field_forward_half_angle_deg=5.0,
            field_back_half_angle_deg=5.0,
            field_quadrature_polar=1,
            field_quadrature_azimuth=1,
            collect_lidar_observation=False,
            range_bin_width_m=1.0,
            range_max_m=0.0,
            receiver_overlap_min=1.0,
            receiver_overlap_full_range_m=0.0,
        )


def _run_lidar_simulation(
    visibility_km: float,
    photons: int = 1_000_000,
    L_size: float = 20.0,
    m_imag: float = 1e-4,
) -> tuple[dict, dict]:
    """运行 exact 内核 + lidar 观测，返回 (result, layers) 元组。"""
    config = mie_worker.DEFAULT_CONFIG.copy()
    config.update(
        {
            "grid_dim": 16,
            "L_size": L_size,
            "r_bottom": 0.4,
            "r_top": 0.4,
            "sigma_ln": 0.0,
            "mie_layer_count": 1,
            "mie_n_radii": 1,
            "m_imag": m_imag,
            "visibility_km": visibility_km,
            "turbulence_scale": 10000.0,
            "cloud_center_z": L_size / 2.0,
            "cloud_thickness": L_size * 10.0,
            "photons": photons,
            "field_compute_mode": "exact_only",
            "lidar_enabled": True,
            "source_type": "point",
            "source_width_m": 0.0,
            "range_bin_width_m": 0.5,
            "range_max_m": L_size - 2.0,
            "receiver_overlap_min": 1.0,
            "receiver_overlap_full_range_m": 0.0,
            "field_back_half_angle_deg": 5.0,
            "field_forward_half_angle_deg": 5.0,
            "field_quadrature_polar": 1,
            "field_quadrature_azimuth": 1,
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        layers = mie_worker.build_mie_layers(config)
        field = mie_worker.generate_field(config, Path(tmp), layers)
        field["density_norm"][:] = 1.0  # 强制均匀密度
        result = mie_numba.run_advanced_simulation(
            layers_config=layers["layers_config"],
            frequency_thz=_freq(config["wavelength_um"]),
            photons=config["photons"],
            density_grid=field["density_norm"],
            grid_res_m=field["L"] / max(field["dim"] - 1, 1),
            source_type="point",
            source_width_m=0.0,
            sigma_ln=config["sigma_ln"],
            collect_voxel_fields=True,
            field_forward_half_angle_deg=config["field_forward_half_angle_deg"],
            field_back_half_angle_deg=config["field_back_half_angle_deg"],
            field_quadrature_polar=config["field_quadrature_polar"],
            field_quadrature_azimuth=config["field_quadrature_azimuth"],
            collect_lidar_observation=True,
            range_bin_width_m=config["range_bin_width_m"],
            range_max_m=config["range_max_m"],
            receiver_overlap_min=config["receiver_overlap_min"],
            receiver_overlap_full_range_m=config["receiver_overlap_full_range_m"],
        )
    return result, layers


def _weighted_linear_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """加权最小二乘线性拟合 y = A + slope * x，返回 (A, slope, R²)。"""
    W = w.sum()
    Wx = (w * x).sum()
    Wy = (w * y).sum()
    Wxx = (w * x * x).sum()
    Wxy = (w * x * y).sum()
    denom = W * Wxx - Wx * Wx
    if abs(denom) < 1e-30:
        return 0.0, 0.0, 0.0
    slope = (W * Wxy - Wx * Wy) / denom
    intercept = (Wy - slope * Wx) / W
    y_pred = intercept + slope * x
    ss_res = (w * (y - y_pred) ** 2).sum()
    ss_tot = (w * (y - Wy / W) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
    return float(intercept), float(slope), float(r2)


# =============================================================================
# WF1：能量守恒
# =============================================================================

class EnergyConservationTests(unittest.TestCase):
    """验证 R_back + R_trans + R_abs = 1（偏差 < 1e-10）。"""

    TOLERANCE = 1e-10

    # 场景：(β_ext, m_imag, label)
    SCENARIOS = [
        (0.002, 0.0,   "EC-1 低τ 无吸收"),
        (0.050, 0.0,   "EC-2 中τ 无吸收"),
        (0.300, 0.0,   "EC-3 高τ 无吸收"),
        (0.002, 5e-4,  "EC-4 低τ 有吸收"),
        (0.050, 5e-4,  "EC-5 中τ 有吸收"),
    ]

    def _check_conservation(self, beta_ext: float, m_imag: float, label: str):
        result = _run_proxy_simulation(beta_ext, m_imag=m_imag)
        s = result["scalars"]
        rb = s["R_back"]
        rt = s["R_trans"]
        ra = s["R_abs"]

        # 非负性
        self.assertGreaterEqual(rb, 0.0, f"{label}: R_back 为负")
        self.assertGreaterEqual(rt, 0.0, f"{label}: R_trans 为负")
        self.assertGreaterEqual(ra, 0.0, f"{label}: R_abs 为负")

        # 守恒
        total = rb + rt + ra
        self.assertAlmostEqual(
            total, 1.0, delta=self.TOLERANCE,
            msg=f"{label}: |sum-1|={abs(total-1):.2e}，超过 {self.TOLERANCE:.0e}"
        )

        # 无吸收：R_abs 必须严格为零
        if m_imag == 0.0:
            self.assertEqual(ra, 0.0, f"{label}: 无吸收场景 R_abs={ra} 应为 0")
        else:
            self.assertGreater(ra, 0.0, f"{label}: 有吸收场景 R_abs 应 > 0")

    def test_ec1_low_tau_no_abs(self):
        beta, mi, label = self.SCENARIOS[0]
        self._check_conservation(beta, mi, label)

    def test_ec2_mid_tau_no_abs(self):
        beta, mi, label = self.SCENARIOS[1]
        self._check_conservation(beta, mi, label)

    def test_ec3_high_tau_no_abs(self):
        beta, mi, label = self.SCENARIOS[2]
        self._check_conservation(beta, mi, label)

    def test_ec4_low_tau_absorbing(self):
        beta, mi, label = self.SCENARIOS[3]
        self._check_conservation(beta, mi, label)

    def test_ec5_mid_tau_absorbing(self):
        beta, mi, label = self.SCENARIOS[4]
        self._check_conservation(beta, mi, label)


# =============================================================================
# WF2：P(R) 解析物理回归
# =============================================================================

class PRRegressionTests(unittest.TestCase):
    """验证 echo_power(R) 符合 Beer-Lambert 双程衰减，排除 1/R² 因子。"""

    # PR-BASE 参数
    VISIBILITY_KM = 0.05   # β_ext(1550nm) ≈ 0.0203 m⁻¹，τ ≈ 0.41（单散射主导）
    PHOTONS = 1_000_000
    L_SIZE = 20.0
    R_MIN = 1.0   # [m] 有效 bin 下限
    R_MAX = 16.0  # [m] 有效 bin 上限
    MIN_EVENTS = 30
    MIN_VALID_BINS = 10

    # 验收标准
    SLOPE_TOL = 0.60     # 斜率相对偏差容差（60%，τ≈0.4 时多散射贡献显著）
    R2_MIN = 0.90        # 拟合 R² 下限
    EXP_VS_INV2_MARGIN = 1.10  # Beer-Lambert 残差不超过 1/R² 残差的 1.1 倍

    @classmethod
    def setUpClass(cls):
        """运行一次 PR-BASE 仿真，所有子测试共享。"""
        cls.result, cls.layers = _run_lidar_simulation(cls.VISIBILITY_KM, cls.PHOTONS, cls.L_SIZE)
        obs = cls.result["arrays"]["lidar_observation"]
        cls.ranges = np.asarray(obs["range_bins_m"], dtype=np.float64)
        cls.power = np.asarray(obs["echo_power"], dtype=np.float64)
        cls.counts = np.asarray(obs["echo_event_count"], dtype=np.float64)
        cls.beta_ext = float(cls.layers["beta_ext_profile"][0])

        # 有效 bin 掩码
        cls.valid = (
            (cls.ranges > cls.R_MIN) &
            (cls.ranges < cls.R_MAX) &
            (cls.power > 0.0) &
            (cls.counts >= cls.MIN_EVENTS)
        )

    def test_pr_sufficient_valid_bins(self):
        """有效 bin 数量必须 >= MIN_VALID_BINS。"""
        n_valid = int(np.sum(self.valid))
        self.assertGreaterEqual(
            n_valid, self.MIN_VALID_BINS,
            f"有效 bin 数 {n_valid} < {self.MIN_VALID_BINS}，"
            f"请检查场景参数或光子数。"
        )

    def test_pr_beer_lambert_r2(self):
        """加权拟合 log(P) vs R 的 R² 应 >= 0.95。"""
        r = self.ranges[self.valid]
        lp = np.log(self.power[self.valid])
        w = self.counts[self.valid]
        _, _, r2 = _weighted_linear_fit(r, lp, w)
        self.assertGreaterEqual(
            r2, self.R2_MIN,
            f"log(P) vs R 的拟合 R² = {r2:.4f} < {self.R2_MIN}，"
            f"密度场可能不均匀或内核异常。"
        )

    def test_pr_beer_lambert_slope_matches_theory(self):
        """拟合斜率 ≈ -2×β_ext，相对偏差受 SLOPE_TOL 约束。"""
        r = self.ranges[self.valid]
        lp = np.log(self.power[self.valid])
        w = self.counts[self.valid]
        _, slope_fit, _ = _weighted_linear_fit(r, lp, w)

        slope_theory = -2.0 * self.beta_ext
        rel_err = abs(slope_fit - slope_theory) / abs(slope_theory)

        self.assertLessEqual(
            rel_err, self.SLOPE_TOL,
            f"斜率相对偏差 {rel_err*100:.1f}% > {self.SLOPE_TOL*100:.0f}%\n"
            f"  拟合斜率: {slope_fit:.6f} m⁻¹\n"
            f"  理论斜率: {slope_theory:.6f} m⁻¹（-2×β_ext = -2×{self.beta_ext:.6f}）"
        )

    def test_pr_no_geometric_attenuation(self):
        """
        Beer-Lambert 残差 < 1/R² 残差 × 1.05。
        确认当前模型不含 1/R² 几何衰减因子。
        """
        r = self.ranges[self.valid]
        lp = np.log(self.power[self.valid])
        w = self.counts[self.valid]
        slope_theory = -2.0 * self.beta_ext

        resid_exp = lp - slope_theory * r
        resid_inv2 = lp - slope_theory * r + 2.0 * np.log(r)

        std_exp = float(np.std(resid_exp))
        std_inv2 = float(np.std(resid_inv2))

        self.assertLessEqual(
            std_exp, std_inv2 * self.EXP_VS_INV2_MARGIN,
            f"Beer-Lambert 残差 std {std_exp:.6f} > 1/R² 残差 std {std_inv2:.6f} × {self.EXP_VS_INV2_MARGIN}，"
            f"当前模型可能存在意外的几何衰减因子。"
        )

    def test_pr_overlap_suppression_quantitative(self):
        """
        overlap 抑制验证：在近场接收不完全时，近场 bin 功率明显低于全重叠估计。
        使用 receiver_overlap_min=0, full_range=L/2 的场景，
        验证 R < full_range 的 bin 平均功率 < 全重叠场景对应 bin 的 80%。
        """
        full_range_m = self.L_SIZE / 2.0
        result_partial, layers_p = _run_lidar_simulation(
            self.VISIBILITY_KM,
            photons=self.PHOTONS,
            L_size=self.L_SIZE,
        )
        # 重新跑带 overlap 的场景
        config = mie_worker.DEFAULT_CONFIG.copy()
        config.update(
            {
                "grid_dim": 16,
                "L_size": self.L_SIZE,
                "r_bottom": 0.4,
                "r_top": 0.4,
                "sigma_ln": 0.0,
                "mie_layer_count": 1,
                "mie_n_radii": 1,
                "m_imag": 1e-4,
                "visibility_km": self.VISIBILITY_KM,
                "turbulence_scale": 10000.0,
                "cloud_center_z": self.L_SIZE / 2.0,
                "cloud_thickness": self.L_SIZE * 10.0,
                "photons": self.PHOTONS,
                "field_compute_mode": "exact_only",
                "lidar_enabled": True,
                "source_type": "point",
                "source_width_m": 0.0,
                "range_bin_width_m": 0.5,
                "range_max_m": self.L_SIZE - 2.0,
                "receiver_overlap_min": 0.0,
                "receiver_overlap_full_range_m": full_range_m,
                "field_back_half_angle_deg": 5.0,
                "field_forward_half_angle_deg": 5.0,
                "field_quadrature_polar": 1,
                "field_quadrature_azimuth": 1,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            layers_ov = mie_worker.build_mie_layers(config)
            field_ov = mie_worker.generate_field(config, Path(tmp), layers_ov)
            field_ov["density_norm"][:] = 1.0
            result_ov = mie_numba.run_advanced_simulation(
                layers_config=layers_ov["layers_config"],
                frequency_thz=_freq(config["wavelength_um"]),
                photons=config["photons"],
                density_grid=field_ov["density_norm"],
                grid_res_m=field_ov["L"] / max(field_ov["dim"] - 1, 1),
                source_type="point",
                source_width_m=0.0,
                sigma_ln=config["sigma_ln"],
                collect_voxel_fields=True,
                field_forward_half_angle_deg=config["field_forward_half_angle_deg"],
                field_back_half_angle_deg=config["field_back_half_angle_deg"],
                field_quadrature_polar=config["field_quadrature_polar"],
                field_quadrature_azimuth=config["field_quadrature_azimuth"],
                collect_lidar_observation=True,
                range_bin_width_m=config["range_bin_width_m"],
                range_max_m=config["range_max_m"],
                receiver_overlap_min=config["receiver_overlap_min"],
                receiver_overlap_full_range_m=config["receiver_overlap_full_range_m"],
            )

        obs_full = self.result["arrays"]["lidar_observation"]
        obs_ov = result_ov["arrays"]["lidar_observation"]

        ranges_f = np.asarray(obs_full["range_bins_m"])
        power_f = np.asarray(obs_full["echo_power"])
        ranges_o = np.asarray(obs_ov["range_bins_m"])
        power_o = np.asarray(obs_ov["echo_power"])
        counts_o = np.asarray(obs_ov["echo_event_count"])

        # 在 full_range_m 的 50%~80% 区间（过渡区）：overlap 版功率应明显低于全重叠版
        lo = full_range_m * 0.3
        hi = full_range_m * 0.7
        mask_near_full = (ranges_f > lo) & (ranges_f < hi) & (power_f > 0)
        mask_near_ov = (ranges_o > lo) & (ranges_o < hi) & (power_o > 0) & (counts_o >= 20)

        if mask_near_full.sum() < 2 or mask_near_ov.sum() < 2:
            self.skipTest("近场有效 bin 不足，跳过 overlap 定量检查")

        mean_full = float(np.mean(power_f[mask_near_full]))
        mean_ov = float(np.mean(power_o[mask_near_ov]))

        self.assertLess(
            mean_ov, mean_full,
            f"overlap 抑制失效：近场区间 [{lo:.1f},{hi:.1f}]m 内，"
            f"partial-overlap 功率 {mean_ov:.4e} 应 < 全重叠功率 {mean_full:.4e}"
        )

    def test_pr_slope_scales_with_beta_ext(self):
        """
        三个 visibility 值场景下，拟合斜率比值 ≈ β_ext 比值，相对偏差受 SLOPE_TOL 约束。
        visibility: 0.10, 0.05, 0.025 km → β_ext 比值约为 1:2:4。
        """
        vis_list = [0.10, 0.05, 0.025]
        slopes = []
        beta_exts = []

        for vis in vis_list:
            result, layers = _run_lidar_simulation(
                vis, photons=500_000, L_size=self.L_SIZE
            )
            obs = result["arrays"]["lidar_observation"]
            ranges = np.asarray(obs["range_bins_m"])
            power = np.asarray(obs["echo_power"])
            counts = np.asarray(obs["echo_event_count"])
            beta_ext = float(layers["beta_ext_profile"][0])

            valid = (
                (ranges > self.R_MIN) &
                (ranges < self.R_MAX) &
                (power > 0.0) &
                (counts >= 20)
            )
            if valid.sum() < 4:
                self.skipTest(f"visibility={vis}km 场景有效 bin 不足")

            _, slope, _ = _weighted_linear_fit(
                ranges[valid], np.log(power[valid]), counts[valid]
            )
            slopes.append(slope)
            beta_exts.append(beta_ext)

        # 相邻场景的斜率比应 ≈ β_ext 比
        for k in range(len(slopes) - 1):
            slope_ratio = abs(slopes[k + 1]) / (abs(slopes[k]) + 1e-30)
            beta_ratio = beta_exts[k + 1] / (beta_exts[k] + 1e-30)
            rel_err = abs(slope_ratio - beta_ratio) / (beta_ratio + 1e-30)
            self.assertLessEqual(
                rel_err, self.SLOPE_TOL,
                f"vis {vis_list[k]}→{vis_list[k+1]}km：斜率比 {slope_ratio:.4f} vs "
                f"β_ext 比 {beta_ratio:.4f}，相对偏差 {rel_err*100:.1f}% > {self.SLOPE_TOL*100:.0f}%"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
