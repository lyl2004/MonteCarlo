"""
阶段一物理闭合测试 — WF3 光子数收敛体系

验证：随光子数 N 增加，echo_power 和 linear_depol_ratio 的统计误差以 ~1/√N 速率收敛。
标记为 slow 测试，需要环境变量 MC_RUN_SLOW_TESTS=1 启用。
"""

import os
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

RUN_SLOW = os.environ.get("MC_RUN_SLOW_TESTS", "0") == "1"


def _freq(wavelength_um: float) -> float:
    return C_LIGHT / (wavelength_um * 1e-6) / 1e12


def _run_cv_scenario(photons: int, seed: int) -> dict:
    """运行 CV-BASE 场景，返回 lidar_observation。"""
    VISIBILITY_KM = 0.05
    L_SIZE = 20.0

    config = mie_worker.DEFAULT_CONFIG.copy()
    config.update({
        "grid_dim": 16,
        "L_size": L_SIZE,
        "r_bottom": 0.4,
        "r_top": 0.4,
        "sigma_ln": 0.0,
        "mie_layer_count": 1,
        "mie_n_radii": 1,
        "m_imag": 1e-4,
        "visibility_km": VISIBILITY_KM,
        "turbulence_scale": 1000.0,
        "cloud_center_z": L_SIZE / 2.0,
        "cloud_thickness": L_SIZE * 2.0,
        "photons": photons,
        "field_compute_mode": "exact_only",
        "lidar_enabled": True,
        "source_type": "point",
        "source_width_m": 0.0,
        "range_bin_width_m": 0.5,
        "range_max_m": L_SIZE - 2.0,
        "receiver_overlap_min": 1.0,
        "receiver_overlap_full_range_m": 0.0,
        "field_back_half_angle_deg": 5.0,
        "field_forward_half_angle_deg": 5.0,
        "field_quadrature_polar": 1,
        "field_quadrature_azimuth": 1,
    })

    with tempfile.TemporaryDirectory() as tmp:
        layers = mie_worker.build_mie_layers(config)
        field = mie_worker.generate_field(config, Path(tmp), layers)
        # 不强制 density=1，允许有密度起伏
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
            field_forward_half_angle_deg=config["field_back_half_angle_deg"],
            field_back_half_angle_deg=config["field_back_half_angle_deg"],
            field_quadrature_polar=config["field_quadrature_polar"],
            field_quadrature_azimuth=config["field_quadrature_azimuth"],
            collect_lidar_observation=True,
            range_bin_width_m=config["range_bin_width_m"],
            range_max_m=config["range_max_m"],
            receiver_overlap_min=config["receiver_overlap_min"],
            receiver_overlap_full_range_m=config["receiver_overlap_full_range_m"],
        )
    return result["arrays"]["lidar_observation"]


@unittest.skipUnless(RUN_SLOW, "slow test, set MC_RUN_SLOW_TESTS=1 to enable")
class ConvergenceTests(unittest.TestCase):
    """验证光子数收敛体系。"""

    # 光子数序列（×4 等比）
    PHOTON_LEVELS = [25_000, 100_000, 400_000]
    N_SEEDS = 5  # 每级 5 个 seed
    R_MIN = 2.0
    R_MAX = 16.0
    MIN_EVENTS = 20

    # 验收标准
    RATIO_TOL_LOW = 1.6   # 收敛比下限（理论 2.0 - 40%）
    RATIO_TOL_HIGH = 2.4  # 收敛比上限（理论 2.0 + 40%）

    def test_cv_power_convergence_ratio(self):
        """
        跨 seed 的 echo_power 标准差应随光子数按 ~1/√N 收敛。
        验证 3 个收敛比均在 [1.6, 2.4]。
        """
        cross_seed_std = {}

        for N in self.PHOTON_LEVELS:
            # 每级跑 5 个 seed
            power_matrix = []  # shape: (n_seeds, n_bins)
            for seed_idx in range(self.N_SEEDS):
                obs = _run_cv_scenario(N, seed=1000 + seed_idx)
                ranges = np.asarray(obs["range_bins_m"])
                power = np.asarray(obs["echo_power"])
                counts = np.asarray(obs["echo_event_count"])

                valid = (
                    (ranges > self.R_MIN) &
                    (ranges < self.R_MAX) &
                    (power > 0.0) &
                    (counts >= self.MIN_EVENTS)
                )
                power_matrix.append(power[valid])

            # 计算跨 seed 的相对标准差（bin 平均）
            power_matrix = np.array(power_matrix)  # (n_seeds, n_valid_bins)
            if power_matrix.shape[1] < 4:
                self.skipTest(f"N={N} 有效 bin 不足")

            mean_per_bin = np.mean(power_matrix, axis=0)
            std_per_bin = np.std(power_matrix, axis=0, ddof=1)
            rel_std_per_bin = std_per_bin / (mean_per_bin + 1e-30)
            cross_seed_std[N] = float(np.mean(rel_std_per_bin))

        # 计算收敛比
        ratios = []
        for k in range(len(self.PHOTON_LEVELS) - 1):
            N_k = self.PHOTON_LEVELS[k]
            N_k1 = self.PHOTON_LEVELS[k + 1]
            ratio = cross_seed_std[N_k] / cross_seed_std[N_k1]
            ratios.append(ratio)

            self.assertGreaterEqual(
                ratio, self.RATIO_TOL_LOW,
                f"N={N_k}→{N_k1} 收敛比 {ratio:.2f} < {self.RATIO_TOL_LOW}"
            )
            self.assertLessEqual(
                ratio, self.RATIO_TOL_HIGH,
                f"N={N_k}→{N_k1} 收敛比 {ratio:.2f} > {self.RATIO_TOL_HIGH}"
            )

        print(f"\n  收敛比: {[f'{r:.2f}' for r in ratios]}")
        print(f"  跨 seed 相对标准差: {cross_seed_std}")

    def test_cv_depol_sphere_sanity(self):
        """
        球形粒子场景下，linear_depol_ratio 中位数应 < 0.20。
        注：τ≈0.4 时多散射导致偏振混合，退偏比高于单散射理论值。
        """
        obs = _run_cv_scenario(1_000_000, seed=999)
        ranges = np.asarray(obs["range_bins_m"])
        depol = np.asarray(obs["linear_depol_ratio"])
        counts = np.asarray(obs["echo_event_count"])

        valid = (
            (ranges > self.R_MIN) &
            (ranges < self.R_MAX) &
            (counts >= 50)
        )

        if valid.sum() < 4:
            self.skipTest("有效 bin 不足")

        median_depol = float(np.median(depol[valid]))
        self.assertLess(
            median_depol, 0.20,
            f"球形粒子 linear_depol_ratio 中位数 {median_depol:.4f} 应 < 0.20"
        )
        print(f"\n  球形粒子 linear_depol_ratio 中位数: {median_depol:.4f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
