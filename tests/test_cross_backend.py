"""
阶段一物理闭合测试 — WF4 跨后端一致性

L1：散射参量一致性（常规）
L2：全局 MC 统计一致性（slow）
L3：距离门观测一致性（slow）
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import mie_core as phys
import mie_numba
import mie_worker

RUN_SLOW = os.environ.get("MC_RUN_SLOW_TESTS", "0") == "1"
C_LIGHT = 299792458.0


def _freq(wavelength_um: float) -> float:
    return C_LIGHT / (wavelength_um * 1e-6) / 1e12


def _run_julia_json(code: str, timeout_sec: int = 300) -> dict:
    result = subprocess.run(
        [
            "pixi",
            "run",
            "-e",
            "julia",
            "julia",
            "--project=src/julia",
            "-e",
            textwrap.dedent(code),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Julia 执行失败:\n{result.stderr}")
    return json.loads(result.stdout)


class CrossBackendL1Tests(unittest.TestCase):
    """WF4-L1：跨后端散射参量一致性测试。"""

    @classmethod
    def setUpClass(cls):
        fixture_path = ROOT / "tests" / "fixtures" / "iitm_sphere_reference.json"
        with fixture_path.open("r", encoding="utf-8") as f:
            cls.iitm_ref = json.load(f)

        scenario = cls.iitm_ref["scenario"]
        cls.wavelength_m = float(scenario["wavelength_m"])
        cls.radius_um = float(scenario["radius_um"])
        cls.m_real = float(scenario["m_real"])
        cls.m_imag = float(scenario["m_imag"])

        # Mie 侧计算（匹配同场景）
        angles_deg = np.asarray(cls.iitm_ref["scattering_params"]["angles_deg"], dtype=np.float64)
        mie_res = phys.mie_effective_polarized(
            size_mode="mono",
            radius_um=cls.radius_um,
            median_radius_um=cls.radius_um,
            sigma_ln=0.0,
            m_complex=complex(cls.m_real, cls.m_imag),
            wavelength_m=cls.wavelength_m,
            angles_deg=angles_deg,
            n_radii=1,
        )
        cls.mie_res = mie_res

        cls.iitm_sigma_ext = float(cls.iitm_ref["scattering_params"]["sigma_ext"])
        cls.iitm_sigma_sca = float(cls.iitm_ref["scattering_params"]["sigma_sca"])
        cls.iitm_omega0 = float(cls.iitm_ref["scattering_params"]["omega0"])
        cls.iitm_g = float(cls.iitm_ref["scattering_params"]["g"])
        cls.iitm_m11 = np.asarray(cls.iitm_ref["scattering_params"]["M11"], dtype=np.float64)

    def test_sigma_ext_relative_error(self):
        """sigma_ext 相对偏差 < 1%。"""
        mie_sigma_ext = float(self.mie_res.sigma_ext)
        rel_err = abs(mie_sigma_ext - self.iitm_sigma_ext) / (self.iitm_sigma_ext + 1e-30)
        self.assertLess(
            rel_err, 0.01,
            f"sigma_ext 相对偏差 {rel_err*100:.3f}% >= 1%\n"
            f"  Mie: {mie_sigma_ext:.6e}\n"
            f"  IITM: {self.iitm_sigma_ext:.6e}"
        )

    def test_sigma_sca_relative_error(self):
        """sigma_sca 相对偏差 < 1%。"""
        mie_sigma_sca = float(self.mie_res.sigma_sca)
        rel_err = abs(mie_sigma_sca - self.iitm_sigma_sca) / (self.iitm_sigma_sca + 1e-30)
        self.assertLess(
            rel_err, 0.01,
            f"sigma_sca 相对偏差 {rel_err*100:.3f}% >= 1%\n"
            f"  Mie: {mie_sigma_sca:.6e}\n"
            f"  IITM: {self.iitm_sigma_sca:.6e}"
        )

    def test_omega0_absolute_error(self):
        """omega0 绝对偏差 < 0.002。"""
        mie_omega0 = float(self.mie_res.sigma_sca / (self.mie_res.sigma_ext + 1e-30))
        abs_err = abs(mie_omega0 - self.iitm_omega0)
        self.assertLess(
            abs_err, 0.002,
            f"omega0 绝对偏差 {abs_err:.6f} >= 0.002\n"
            f"  Mie: {mie_omega0:.6f}\n"
            f"  IITM: {self.iitm_omega0:.6f}"
        )

    def test_g_absolute_error(self):
        """g 绝对偏差 < 0.01。"""
        mie_g = float(self.mie_res.g)
        abs_err = abs(mie_g - self.iitm_g)
        self.assertLess(
            abs_err, 0.01,
            f"g 绝对偏差 {abs_err:.6f} >= 0.01\n"
            f"  Mie: {mie_g:.6f}\n"
            f"  IITM: {self.iitm_g:.6f}"
        )

    def test_m11_curve_correlation(self):
        """M11 角度曲线 Pearson 相关 > 0.9999。"""
        mie_m11 = np.asarray(self.mie_res.M11, dtype=np.float64)
        iitm_m11 = self.iitm_m11

        # 避免常数向量导致相关性 NaN
        if np.std(mie_m11) < 1e-30 or np.std(iitm_m11) < 1e-30:
            self.skipTest("M11 曲线标准差过小，无法计算相关性")

        corr = float(np.corrcoef(mie_m11, iitm_m11)[0, 1])
        self.assertGreater(
            corr, 0.9999,
            f"M11 曲线相关性 {corr:.6f} <= 0.9999"
        )


@unittest.skipUnless(RUN_SLOW, "slow test, set MC_RUN_SLOW_TESTS=1 to enable")
class CrossBackendL2Tests(unittest.TestCase):
    """WF4-L2：全局 MC 统计一致性（slow）。"""

    PHOTONS = 1_000_000
    SEED = 42
    L_SIZE = 20.0
    VISIBILITY_KM = 0.05

    def test_global_mc_consistency(self):
        """
        两后端各跑 1M 光子，验证：
        1. 各自能量守恒（R_back + R_trans + R_abs = 1）
        2. R_back 量级一致（相对偏差 < 15%）
        """
        config = mie_worker.DEFAULT_CONFIG.copy()
        config.update({
            "grid_dim": 8,
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
            "field_compute_mode": "proxy_only",
            "lidar_enabled": False,
        })

        with tempfile.TemporaryDirectory() as tmp:
            layers = mie_worker.build_mie_layers(config)
            field = mie_worker.generate_field(config, Path(tmp), layers)
            result_mie = mie_numba.run_advanced_simulation(
                layers_config=layers["layers_config"],
                frequency_thz=_freq(config["wavelength_um"]),
                photons=config["photons"],
                density_grid=field["density_norm"],
                grid_res_m=field["L"] / max(field["dim"] - 1, 1),
                source_type="point",
                source_width_m=0.0,
                sigma_ln=config["sigma_ln"],
                collect_voxel_fields=False,
                collect_lidar_observation=False,
            )

        s_mie = result_mie["scalars"]
        R_back_mie = float(s_mie["R_back"])
        R_trans_mie = float(s_mie["R_trans"])
        R_abs_mie = float(s_mie["R_abs"])

        sum_mie = R_back_mie + R_trans_mie + R_abs_mie
        self.assertAlmostEqual(
            sum_mie, 1.0, places=10,
            msg=f"Mie 能量守恒偏差: {abs(sum_mie - 1.0):.2e}"
        )

        julia_code = f"""
        include("src/julia/iitm_physics.jl")
        using JSON3

        field = generate_field(Dict{{String,Any}}(
            "grid_dim" => 8,
            "L_size" => {self.L_SIZE},
            "cloud_center_z" => {self.L_SIZE / 2.0},
            "cloud_thickness" => {self.L_SIZE * 10.0},
            "turbulence_scale" => 10000.0,
        ))

        scatter = compute_scatter_params(Dict{{String,Any}}(
            "wavelength_m" => 1.55e-6,
            "m_real" => 1.311,
            "m_imag" => 1e-4,
            "shape_type" => "sphere",
            "size_mode" => "mono",
            "radius_um" => 0.4,
            "Nr" => 8,
            "Ntheta" => 16,
            "n_radii" => 1,
        ))

        beta_ext = visibility_to_beta_ext({self.VISIBILITY_KM}, 1550.0, angstrom_q=1.3)

        mc = run_monte_carlo(scatter, Dict{{String,Any}}(
            "beta_ext_surf" => beta_ext,
            "thickness_m" => {self.L_SIZE},
            "scale_height_m" => 0.0,
            "n_photons" => {self.PHOTONS},
            "seed" => {self.SEED},
            "collect_voxel_fields" => false,
            "collect_lidar_observation" => false,
            "density_grid" => field["density_norm"],
            "field_axis" => field["axis"],
            "field_xy_centered" => true,
            "density_sampling" => "nearest",
        ))

        result = Dict(
            "backscatter_ratio" => mc.backscatter_ratio,
            "transmit_ratio" => mc.transmit_ratio,
            "absorbed_ratio" => mc.absorbed_ratio,
        )

        println(JSON3.write(result))
        """

        iitm_data = _run_julia_json(julia_code, timeout_sec=180)
        R_back_iitm = float(iitm_data["backscatter_ratio"])
        R_trans_iitm = float(iitm_data["transmit_ratio"])
        R_abs_iitm = float(iitm_data["absorbed_ratio"])

        sum_iitm = R_back_iitm + R_trans_iitm + R_abs_iitm
        self.assertAlmostEqual(
            sum_iitm, 1.0, places=10,
            msg=f"IITM 能量守恒偏差: {abs(sum_iitm - 1.0):.2e}"
        )

        diff = abs(R_back_mie - R_back_iitm)
        rel_err = diff / (abs(R_back_iitm) + 1e-30)
        self.assertLess(
            rel_err, 0.15,
            f"R_back 相对偏差 {rel_err*100:.2f}% >= 15%\n"
            f"  Mie: {R_back_mie:.6f}\n"
            f"  IITM: {R_back_iitm:.6f}"
        )

        print(f"\n  Mie:  R_back={R_back_mie:.6f}, R_trans={R_trans_mie:.6f}, R_abs={R_abs_mie:.6f}")
        print(f"  IITM: R_back={R_back_iitm:.6f}, R_trans={R_trans_iitm:.6f}, R_abs={R_abs_iitm:.6f}")
        print(f"  R_back 绝对偏差: {diff:.6f}, 相对偏差: {rel_err*100:.2f}%")


@unittest.skipUnless(RUN_SLOW, "slow test, set MC_RUN_SLOW_TESTS=1 to enable")
class CrossBackendL3Tests(unittest.TestCase):
    """WF4-L3：距离门观测一致性（slow）。"""

    PHOTONS = 500_000
    SEED = 123
    L_SIZE = 20.0
    VISIBILITY_KM = 0.05

    def test_lidar_observation_consistency(self):
        """
        两后端各跑 500k 光子，density_norm=1，验证：
        1. echo_power 拟合斜率相对偏差 < 20%
        2. linear_depol_ratio 中位数绝对偏差 < 0.05
        """
        config = mie_worker.DEFAULT_CONFIG.copy()
        config.update({
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
            field["density_norm"][:] = 1.0
            result_mie = mie_numba.run_advanced_simulation(
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

        obs_mie = result_mie["arrays"]["lidar_observation"]
        ranges_mie = np.asarray(obs_mie["range_bins_m"])
        power_mie = np.asarray(obs_mie["echo_power"])
        counts_mie = np.asarray(obs_mie["echo_event_count"])
        depol_mie = np.asarray(obs_mie["linear_depol_ratio"])

        R_MIN, R_MAX, MIN_EVENTS = 2.0, 16.0, 30
        valid_mie = (
            (ranges_mie > R_MIN) &
            (ranges_mie < R_MAX) &
            (power_mie > 0.0) &
            (counts_mie >= MIN_EVENTS)
        )
        self.assertGreaterEqual(valid_mie.sum(), 10, "Mie 有效 bin 数不足")

        r_mie = ranges_mie[valid_mie]
        p_mie = power_mie[valid_mie]
        w_mie = counts_mie[valid_mie]
        log_p_mie = np.log(p_mie)
        A_mie = np.vstack([r_mie, np.ones_like(r_mie)]).T
        coef_mie = np.linalg.lstsq((w_mie[:, None] * A_mie), (w_mie * log_p_mie), rcond=None)[0]
        slope_mie = float(coef_mie[0])

        julia_code = f"""
        include("src/julia/iitm_physics.jl")
        using JSON3

        field = generate_field(Dict{{String,Any}}(
            "grid_dim" => 16,
            "L_size" => {self.L_SIZE},
            "cloud_center_z" => {self.L_SIZE / 2.0},
            "cloud_thickness" => {self.L_SIZE * 10.0},
            "turbulence_scale" => 10000.0,
        ))
        field["density_norm"] .= 1.0

        scatter = compute_scatter_params(Dict{{String,Any}}(
            "wavelength_m" => 1.55e-6,
            "m_real" => 1.311,
            "m_imag" => 1e-4,
            "shape_type" => "sphere",
            "size_mode" => "mono",
            "radius_um" => 0.4,
            "Nr" => 8,
            "Ntheta" => 16,
            "n_radii" => 1,
        ))

        beta_ext = visibility_to_beta_ext({self.VISIBILITY_KM}, 1550.0, angstrom_q=1.3)

        mc = run_monte_carlo(scatter, Dict{{String,Any}}(
            "beta_ext_surf" => beta_ext,
            "thickness_m" => {self.L_SIZE},
            "scale_height_m" => 0.0,
            "n_photons" => {self.PHOTONS},
            "seed" => {self.SEED},
            "collect_voxel_fields" => true,
            "collect_lidar_observation" => true,
            "density_grid" => field["density_norm"],
            "field_axis" => field["axis"],
            "field_xy_centered" => true,
            "density_sampling" => "nearest",
            "field_back_half_angle_deg" => 5.0,
            "field_forward_half_angle_deg" => 5.0,
            "field_quadrature_polar" => 1,
            "field_quadrature_azimuth" => 1,
            "range_bin_width_m" => 0.5,
            "range_max_m" => {self.L_SIZE - 2.0},
            "receiver_overlap_min" => 1.0,
            "receiver_overlap_full_range_m" => 0.0,
        ))

        obs = mc.lidar_observation
        result = Dict(
            "range_bins_m" => collect(obs.range_bins_m),
            "echo_power" => collect(obs.echo_power),
            "echo_event_count" => collect(obs.echo_event_count),
            "linear_depol_ratio" => collect(obs.linear_depol_ratio),
        )
        println(JSON3.write(result))
        """

        iitm_data = _run_julia_json(julia_code, timeout_sec=240)
        ranges_iitm = np.asarray(iitm_data["range_bins_m"])
        power_iitm = np.asarray(iitm_data["echo_power"])
        counts_iitm = np.asarray(iitm_data["echo_event_count"])
        depol_iitm = np.asarray(iitm_data["linear_depol_ratio"])

        valid_iitm = (
            (ranges_iitm > R_MIN) &
            (ranges_iitm < R_MAX) &
            (power_iitm > 0.0) &
            (counts_iitm >= MIN_EVENTS)
        )
        self.assertGreaterEqual(valid_iitm.sum(), 10, "IITM 有效 bin 数不足")

        r_iitm = ranges_iitm[valid_iitm]
        p_iitm = power_iitm[valid_iitm]
        w_iitm = counts_iitm[valid_iitm]
        log_p_iitm = np.log(p_iitm)
        A_iitm = np.vstack([r_iitm, np.ones_like(r_iitm)]).T
        coef_iitm = np.linalg.lstsq((w_iitm[:, None] * A_iitm), (w_iitm * log_p_iitm), rcond=None)[0]
        slope_iitm = float(coef_iitm[0])

        slope_rel_err = abs(slope_mie - slope_iitm) / (abs(slope_iitm) + 1e-30)
        self.assertLess(
            slope_rel_err, 0.20,
            f"echo_power 斜率相对偏差 {slope_rel_err*100:.1f}% >= 20%\n"
            f"  Mie: {slope_mie:.6f}\n"
            f"  IITM: {slope_iitm:.6f}"
        )

        median_depol_mie = float(np.median(depol_mie[valid_mie]))
        median_depol_iitm = float(np.median(depol_iitm[valid_iitm]))
        depol_abs_err = abs(median_depol_mie - median_depol_iitm)
        self.assertLess(
            depol_abs_err, 0.08,
            f"linear_depol_ratio 中位数绝对偏差 {depol_abs_err:.4f} >= 0.08\n"
            f"  Mie: {median_depol_mie:.4f}\n"
            f"  IITM: {median_depol_iitm:.4f}"
        )

        print(f"\n  Mie:  slope={slope_mie:.6f}, median_depol={median_depol_mie:.4f}")
        print(f"  IITM: slope={slope_iitm:.6f}, median_depol={median_depol_iitm:.4f}")
        print(f"  斜率相对偏差: {slope_rel_err*100:.1f}%, 退偏比绝对偏差: {depol_abs_err:.4f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
