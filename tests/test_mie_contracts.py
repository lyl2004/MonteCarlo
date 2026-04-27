import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import mie_core
import mie_numba
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
        self.assertLessEqual(obs["depol_back"], 1.0)
        self.assertLessEqual(obs["depol_forward"], 1.0)

    def test_mie_worker_field_catalog_reports_effective_mode(self):
        meta = mie_worker.build_field_catalog({"field_compute_mode": "both"})

        self.assertEqual(meta["requested_field_compute_mode"], "both")
        self.assertEqual(meta["effective_field_compute_mode"], "proxy_only")
        self.assertEqual(meta["available_field_families"], ["proxy"])
        self.assertEqual(
            [item["name"] for item in meta["field_catalog"]["proxy"]],
            ["beta_back", "beta_forward", "depol_ratio", "density"],
        )

        both_meta = mie_worker.build_field_catalog(
            {"field_compute_mode": "both"},
            exact_available=True,
        )
        self.assertEqual(both_meta["effective_field_compute_mode"], "both")
        self.assertEqual(both_meta["available_field_families"], ["proxy", "exact"])
        self.assertEqual(
            [item["name"] for item in both_meta["field_catalog"]["exact"]],
            ["beta_back", "beta_forward", "depol_ratio", "event_count"],
        )

        exact_meta = mie_worker.build_field_catalog(
            {"field_compute_mode": "exact_only"},
            exact_available=True,
        )
        self.assertEqual(exact_meta["effective_field_compute_mode"], "exact_only")
        self.assertEqual(exact_meta["available_field_families"], ["exact"])
        self.assertEqual(exact_meta["primary_field_family"], "exact")

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
                self.assertIn("proxy_beta_back", data.files)
                self.assertIn("proxy_beta_forward", data.files)
                self.assertIn("proxy_depol_ratio", data.files)
                self.assertIn("lut_back", data.files)
                self.assertIn("lut_forward", data.files)
                self.assertIn("lut_depol", data.files)

    def test_render_headless_exports_julia_style_npz_and_three_html_views(self):
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
            temp_dir = Path(tmp) / "temp"
            output_dir = Path(tmp) / "output"
            layers = mie_worker.build_mie_layers(config)
            field = mie_worker.generate_field(config, temp_dir, layers)
            files = mie_worker.render_headless(field, config, output_dir)

            self.assertEqual(files, ["render_main.html", "render_top.html", "render_front.html"])
            self.assertTrue((output_dir / "density.npz").exists())
            with np.load(output_dir / "density.npz") as data:
                self.assertIn("density", data.files)
                self.assertIn("summary", data.files)
                self.assertIn("proxy_summary", data.files)
                self.assertIn("proxy_beta_back", data.files)
                self.assertIn("proxy_beta_forward", data.files)
                self.assertIn("proxy_depol_ratio", data.files)
                self.assertIn("beta_back", data.files)
                self.assertIn("beta_forward", data.files)
                self.assertIn("depol_ratio", data.files)

    def test_render_headless_exports_exact_fields_when_available(self):
        config = mie_worker.DEFAULT_CONFIG.copy()
        config.update(
            {
                "grid_dim": 4,
                "L_size": 2.0,
                "r_bottom": 0.5,
                "r_top": 0.8,
                "sigma_ln": 0.0,
                "mie_layer_count": 2,
                "mie_n_radii": 1,
                "visibility_km": 3.0,
                "cloud_center_z": 1.0,
                "cloud_thickness": 1.0,
                "turbulence_scale": 3.0,
                "field_compute_mode": "both",
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp) / "temp"
            output_dir = Path(tmp) / "output"
            layers = mie_worker.build_mie_layers(config)
            field = mie_worker.generate_field(config, temp_dir, layers)
            zeros = np.zeros_like(field["density_norm"], dtype=np.float64)
            ones = np.ones_like(field["density_norm"], dtype=np.float64)
            field["exact_fields"] = {
                "beta_back": field["beta_back"] * 0.5,
                "beta_forward": field["beta_forward"] * 0.5,
                "depol_ratio": np.clip(field["depol_ratio"], 0.0, 1.0),
                "event_count": ones,
                "back_Q": zeros,
                "back_U": zeros,
                "back_V": zeros,
                "forward_Q": zeros,
                "forward_U": zeros,
                "forward_V": zeros,
            }
            field["field_meta"] = mie_worker.build_field_catalog(config, exact_available=True)

            mie_worker.render_headless(field, config, output_dir)

            with np.load(output_dir / "density.npz") as data:
                self.assertIn("exact_beta_back", data.files)
                self.assertIn("exact_beta_forward", data.files)
                self.assertIn("exact_depol_ratio", data.files)
                self.assertIn("exact_event_count", data.files)
                self.assertIn("exact_summary", data.files)
                self.assertIn("proxy_beta_back", data.files)
                self.assertIn("proxy_summary", data.files)

    def test_numba_exact_voxel_fields_contract(self):
        config = mie_worker.DEFAULT_CONFIG.copy()
        config.update(
            {
                "grid_dim": 4,
                "L_size": 2.0,
                "r_bottom": 0.5,
                "r_top": 0.6,
                "sigma_ln": 0.0,
                "mie_layer_count": 2,
                "mie_n_radii": 1,
                "visibility_km": 0.2,
                "cloud_center_z": 1.0,
                "cloud_thickness": 1.0,
                "turbulence_scale": 3.0,
                "photons": 6,
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            layers = mie_worker.build_mie_layers(config)
            field = mie_worker.generate_field(config, Path(tmp), layers)
            freq = 299792458.0 / (config["wavelength_um"] * 1e-6) / 1e12
            result = mie_numba.run_advanced_simulation(
                layers_config=layers["layers_config"],
                frequency_thz=freq,
                photons=config["photons"],
                density_grid=field["density_norm"],
                grid_res_m=field["L"] / max(field["dim"] - 1, 1),
                source_type="planar",
                source_width_m=field["L"],
                sigma_ln=config["sigma_ln"],
                collect_voxel_fields=True,
                field_quadrature_polar=1,
                field_quadrature_azimuth=2,
            )

        voxel = result["arrays"]["voxel_fields"]
        for name in (
            "forward_I", "forward_Q", "forward_U", "forward_V",
            "back_I", "back_Q", "back_U", "back_V", "event_count",
        ):
            self.assertEqual(voxel[name].shape, (4, 4, 4))
            self.assertTrue(np.all(np.isfinite(voxel[name])))
        self.assertTrue(np.all(voxel["event_count"] >= 0.0))


if __name__ == "__main__":
    unittest.main()
