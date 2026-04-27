import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import mie_core
import mie_worker


class MieContractTests(unittest.TestCase):
    def test_scatter_observables_use_sigma_reference_names(self):
        angles = mie_core.generate_adaptive_angles(
            num_total=80,
            forward_res=0.05,
            forward_max=1.0,
        )
        result = mie_core.mie_effective_polarized(
            size_mode="mono",
            radius_um=0.5,
            median_radius_um=0.5,
            sigma_ln=0.0,
            m_complex=1.33 + 0.0j,
            wavelength_m=1.55e-6,
            angles_deg=angles,
            n_radii=1,
        )
        obs = mie_core.mie_scatter_observables(result)

        self.assertGreater(result.sigma_ext, 0.0)
        self.assertGreater(result.sigma_sca, 0.0)
        self.assertIn("sigma_back_ref", obs)
        self.assertIn("sigma_forward_ref", obs)
        self.assertNotIn("beta_back_ref", obs)
        self.assertNotIn("beta_forward_ref", obs)
        self.assertGreaterEqual(obs["sigma_back_ref"], 0.0)
        self.assertGreaterEqual(obs["sigma_forward_ref"], 0.0)
        self.assertGreaterEqual(obs["depol_back"], 0.0)
        self.assertGreaterEqual(obs["depol_forward"], 0.0)

    def test_mie_worker_reports_proxy_only_effective_mode(self):
        meta = mie_worker.build_field_catalog({"field_compute_mode": "both"})

        self.assertEqual(meta["requested_field_compute_mode"], "both")
        self.assertEqual(meta["effective_field_compute_mode"], "proxy_only")
        self.assertEqual(meta["available_field_families"], ["proxy"])
        self.assertEqual(
            [item["name"] for item in meta["field_catalog"]["proxy"]],
            ["beta_back", "beta_forward", "depol_ratio"],
        )

    def test_generate_field_exports_three_proxy_channels(self):
        config = mie_worker.DEFAULT_CONFIG.copy()
        config.update(
            {
                "grid_dim": 6,
                "L_size": 3.0,
                "r_bottom": 0.5,
                "r_top": 0.8,
                "sigma_ln": 0.0,
                "mie_layer_count": 2,
                "mie_n_radii": 1,
                "visibility_km": 3.0,
                "cloud_center_z": 1.5,
                "cloud_thickness": 1.2,
                "turbulence_scale": 3.0,
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp)
            layers = mie_worker.build_mie_layers(config)
            field = mie_worker.generate_field(config, temp_dir, layers)
            npz_path = temp_dir / "field_data.npz"

            self.assertTrue(npz_path.exists())
            self.assertEqual(field["beta_back"].shape, (6, 6, 6))
            self.assertEqual(field["beta_forward"].shape, (6, 6, 6))
            self.assertEqual(field["depol_ratio"].shape, (6, 6, 6))
            self.assertTrue(np.all(field["beta_back"] >= 0.0))
            self.assertTrue(np.all(field["beta_forward"] >= 0.0))

            with np.load(npz_path) as data:
                self.assertIn("beta_back", data.files)
                self.assertIn("beta_forward", data.files)
                self.assertIn("depol_ratio", data.files)
                self.assertIn("lut_back", data.files)
                self.assertIn("lut_forward", data.files)
                self.assertIn("lut_depol", data.files)


if __name__ == "__main__":
    unittest.main()
