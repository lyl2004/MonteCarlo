import sys
import tempfile
import unittest
import json
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import dataset_runner


class DatasetContractTests(unittest.TestCase):
    def test_export_sample_contract(self):
        obs = {
            "range_bins_m": np.array([0.5, 1.5], dtype=np.float64),
            "echo_I": np.array([1.0, 0.5], dtype=np.float64),
            "echo_Q": np.array([0.2, 0.1], dtype=np.float64),
            "echo_U": np.zeros(2, dtype=np.float64),
            "echo_V": np.zeros(2, dtype=np.float64),
            "echo_power": np.array([1.0, 0.5], dtype=np.float64),
            "echo_depol": np.array([0.8, 0.8], dtype=np.float64),
            "echo_event_count": np.array([10.0, 4.0], dtype=np.float64),
            "echo_weight_sum": np.array([1.0, 0.5], dtype=np.float64),
            "echo_weight_sq_sum": np.array([0.12, 0.08], dtype=np.float64),
            "echo_power_variance_est": np.array([0.0012, 0.0008], dtype=np.float64),
            "echo_power_ci_low": np.array([0.9321, 0.4446], dtype=np.float64),
            "echo_power_ci_high": np.array([1.0679, 0.5554], dtype=np.float64),
            "echo_relative_error_est": np.array([0.316, 0.5], dtype=np.float64),
            "receiver_model": {"range_bin_width_m": 1.0, "receiver_mode": "backscatter"},
        }
        cfg = {
            "seed": 123,
            "photons": 100,
            "visibility_km": 3.0,
            "r_bottom": 0.5,
            "r_top": 1.0,
            "sigma_ln": 0.2,
            "m_real": 1.33,
            "m_imag": 0.0,
            "wavelength_um": 1.55,
            "L_size": 2.0,
            "grid_dim": 4,
            "cloud_center_z": 1.0,
            "cloud_thickness": 1.0,
            "turbulence_scale": 4.0,
            "field_compute_mode": "both",
            "lidar_enabled": True,
            "source_type": "point",
            "source_width_m": 0.0,
            "range_bin_width_m": 1.0,
            "range_max_m": 2.0,
            "receiver_overlap_min": 0.0,
            "receiver_overlap_full_range_m": 10.0,
            "field_back_half_angle_deg": 90.0,
            "field_forward_half_angle_deg": 90.0,
            "field_quadrature_polar": 2,
            "field_quadrature_azimuth": 6,
        }
        with tempfile.TemporaryDirectory() as tmp:
            sample_dir = Path(tmp) / "sample_000001"
            quality = dataset_runner.export_sample(sample_dir, "mie", cfg, obs)
            self.assertTrue((sample_dir / "observation.npz").exists())
            self.assertTrue((sample_dir / "truth.json").exists())
            self.assertTrue((sample_dir / "receiver.json").exists())
            self.assertTrue((sample_dir / "quality.json").exists())
            self.assertTrue((sample_dir / "run_config.json").exists())
            with np.load(sample_dir / "observation.npz") as data:
                self.assertIn("range_bins_m", data.files)
                self.assertIn("echo_power", data.files)
                self.assertIn("echo_weight_sq_sum", data.files)
                self.assertIn("echo_power_variance_est", data.files)
                self.assertIn("echo_power_ci_low", data.files)
                self.assertIn("echo_power_ci_high", data.files)
                self.assertEqual(data["echo_I"].shape, (2,))
            self.assertEqual(quality["photons"], 100)
            self.assertEqual(quality["valid_bin_count"], 2)
            self.assertEqual(quality["requested_seed"], 123)
            self.assertEqual(quality["rng_reproducibility"], "statistical_only")
            self.assertEqual(quality["field_compute_mode"], "both")
            self.assertEqual(quality["source_type"], "point")

            receiver = json.loads((sample_dir / "receiver.json").read_text(encoding="utf-8"))
            self.assertEqual(receiver["range_bin_width_m"], 1.0)
            self.assertEqual(receiver["overlap_min"], 0.0)
            self.assertEqual(receiver["field_back_half_angle_deg"], 90.0)
            self.assertEqual(receiver["field_quadrature_azimuth"], 6)
            self.assertEqual(receiver["source_type"], "point")

            truth = json.loads((sample_dir / "truth.json").read_text(encoding="utf-8"))
            self.assertEqual(truth["source"]["source_type"], "point")

    def test_mie_sample_preserves_point_source_config(self):
        field = {
            "density_norm": np.ones((2, 2, 2), dtype=np.float64),
            "L": 4.0,
            "dim": 2,
        }
        sim_result = {
            "arrays": {
                "lidar_observation": {
                    "range_bins_m": np.array([0.5], dtype=np.float64),
                    "echo_I": np.array([0.0], dtype=np.float64),
                    "echo_Q": np.array([0.0], dtype=np.float64),
                    "echo_U": np.array([0.0], dtype=np.float64),
                    "echo_V": np.array([0.0], dtype=np.float64),
                    "echo_power": np.array([0.0], dtype=np.float64),
                    "echo_depol": np.array([0.0], dtype=np.float64),
                    "echo_event_count": np.array([0.0], dtype=np.float64),
                    "echo_weight_sum": np.array([0.0], dtype=np.float64),
                    "echo_weight_sq_sum": np.array([0.0], dtype=np.float64),
                    "echo_power_variance_est": np.array([0.0], dtype=np.float64),
                    "echo_power_ci_low": np.array([0.0], dtype=np.float64),
                    "echo_power_ci_high": np.array([0.0], dtype=np.float64),
                    "echo_relative_error_est": np.array([0.0], dtype=np.float64),
                    "receiver_model": {},
                }
            }
        }
        cfg = {
            "source_type": "point",
            "source_width_m": 0.0,
            "grid_dim": 2,
            "photons": 1,
            "sigma_ln": 0.0,
            "range_bin_width_m": 1.0,
            "range_max_m": 1.0,
        }
        with mock.patch.object(dataset_runner.mie_worker, "build_mie_layers", return_value={"layers_config": []}), \
             mock.patch.object(dataset_runner.mie_worker, "generate_field", return_value=field), \
             mock.patch.object(dataset_runner.mie_numba, "run_advanced_simulation", return_value=sim_result) as run_mock:
            effective_config, _ = dataset_runner.run_mie_sample(cfg)

        kwargs = run_mock.call_args.kwargs
        self.assertEqual(kwargs["source_type"], "point")
        self.assertEqual(kwargs["source_width_m"], 0.0)
        self.assertEqual(effective_config["source_type"], "point")


if __name__ == "__main__":
    unittest.main()
