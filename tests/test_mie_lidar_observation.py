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


class MieLidarObservationTests(unittest.TestCase):
    def _run_observation(self, overlap_min=1.0, overlap_full_range_m=0.0):
        config = mie_worker.DEFAULT_CONFIG.copy()
        config.update(
            {
                "grid_dim": 5,
                "L_size": 4.0,
                "r_bottom": 0.4,
                "r_top": 0.4,
                "sigma_ln": 0.0,
                "mie_layer_count": 1,
                "mie_n_radii": 1,
                "visibility_km": 0.08,
                "cloud_center_z": 2.0,
                "cloud_thickness": 8.0,
                "turbulence_scale": 1000.0,
                "photons": 80,
                "field_back_half_angle_deg": 90.0,
                "field_forward_half_angle_deg": 90.0,
                "field_quadrature_polar": 1,
                "field_quadrature_azimuth": 2,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            layers = mie_worker.build_mie_layers(config)
            field = mie_worker.generate_field(config, Path(tmp), layers)
            freq = 299792458.0 / (config["wavelength_um"] * 1e-6) / 1e12
            return mie_numba.run_advanced_simulation(
                layers_config=layers["layers_config"],
                frequency_thz=freq,
                photons=config["photons"],
                density_grid=field["density_norm"],
                grid_res_m=field["L"] / max(field["dim"] - 1, 1),
                source_type="planar",
                source_width_m=field["L"],
                sigma_ln=config["sigma_ln"],
                collect_voxel_fields=True,
                field_forward_half_angle_deg=config["field_forward_half_angle_deg"],
                field_back_half_angle_deg=config["field_back_half_angle_deg"],
                field_quadrature_polar=config["field_quadrature_polar"],
                field_quadrature_azimuth=config["field_quadrature_azimuth"],
                collect_lidar_observation=True,
                range_bin_width_m=0.5,
                range_max_m=4.0,
                receiver_overlap_min=overlap_min,
                receiver_overlap_full_range_m=overlap_full_range_m,
            )

    def test_lidar_observation_contract(self):
        result = self._run_observation()
        obs = result["arrays"]["lidar_observation"]

        expected = {
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
            "receiver_model",
        }
        self.assertTrue(expected.issubset(obs.keys()))
        n = len(obs["range_bins_m"])
        self.assertGreater(n, 1)
        for key in expected - {"receiver_model"}:
            self.assertEqual(np.asarray(obs[key]).shape, (n,))
            self.assertTrue(np.all(np.isfinite(obs[key])))

        self.assertTrue(np.all(obs["echo_power"] >= 0.0))
        self.assertTrue(np.all(obs["echo_event_count"] >= 0.0))
        self.assertTrue(np.all(obs["echo_depol"] >= 0.0))
        self.assertTrue(np.all(obs["echo_depol"] <= 1.0))
        self.assertEqual(obs["receiver_model"]["receiver_mode"], "backscatter")

    def test_overlap_suppresses_near_range_bins(self):
        full = self._run_observation(overlap_min=1.0)
        suppressed = self._run_observation(overlap_min=0.0, overlap_full_range_m=10.0)

        full_power = np.asarray(full["arrays"]["lidar_observation"]["echo_power"])
        suppressed_power = np.asarray(suppressed["arrays"]["lidar_observation"]["echo_power"])
        near_full = float(np.sum(full_power[:3]))
        near_suppressed = float(np.sum(suppressed_power[:3]))

        self.assertLessEqual(near_suppressed, near_full + 1e-20)


if __name__ == "__main__":
    unittest.main()
