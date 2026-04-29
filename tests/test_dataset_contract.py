import sys
import tempfile
import unittest
from pathlib import Path

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
            "echo_relative_error_est": np.array([0.316, 0.5], dtype=np.float64),
            "receiver_model": {"range_bin_width_m": 1.0, "receiver_mode": "backscatter"},
        }
        cfg = {
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
                self.assertEqual(data["echo_I"].shape, (2,))
            self.assertEqual(quality["photons"], 100)
            self.assertEqual(quality["valid_bin_count"], 2)


if __name__ == "__main__":
    unittest.main()
