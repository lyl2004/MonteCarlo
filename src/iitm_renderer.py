import argparse
import json
from pathlib import Path

import numpy as np


VIEWS = [
    ("render_main", "main"),
    ("render_top", "top"),
    ("render_front", "front"),
    ("render_bottom", "bottom"),
    ("render_left", "left"),
]


def _safe_title(label: str) -> str:
    return str(label).replace('"', "'")


def _view_configs(axis_min: float, axis_max: float):
    center = (
        0.5 * (axis_min + axis_max),
        0.5 * (axis_min + axis_max),
        0.5 * (axis_min + axis_max),
    )
    span = max(axis_max - axis_min, 1e-6)
    return {
        "main": [(center[0] + 2.8 * span, center[1] + 2.8 * span, center[2] + 2.1 * span), center, (0, 0, 1)],
        "top": [(center[0], center[1], center[2] + 3.4 * span), center, (0, 1, 0)],
        "front": [(center[0], center[1] - 3.4 * span, center[2]), center, (0, 0, 1)],
        "bottom": [(center[0], center[1], center[2] - 3.4 * span), center, (0, 1, 0)],
        "left": [(center[0] - 3.4 * span, center[1], center[2]), center, (0, 0, 1)],
    }


def _output_filename(view_stem: str, family: str, field_name: str) -> str:
    if family == "proxy" and field_name == "beta_back":
        return f"{view_stem}.html"
    if family == "proxy":
        return f"{view_stem}__{field_name}.html"
    return f"{view_stem}__{family}__{field_name}.html"


def _cmap_for(field_name: str, family: str) -> str:
    if field_name == "beta_forward":
        return "plasma"
    if field_name == "depol_ratio":
        return "cividis"
    if field_name == "event_count":
        return "Blues"
    if field_name == "density":
        return "viridis"
    return "turbo" if family == "exact" else "viridis"


def _finite_positive(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    return finite[finite > 0.0]


def _point_threshold(values: np.ndarray, field_name: str, family: str) -> float | None:
    positive = _finite_positive(values)
    if positive.size == 0:
        return None

    vmax = float(positive.max())
    if field_name == "density":
        return min(max(float(np.quantile(positive, 0.72)), vmax * 0.18), vmax)
    if field_name == "depol_ratio":
        nonflat = positive[np.abs(positive - float(np.nanmedian(positive))) > 1e-7]
        if nonflat.size:
            positive = nonflat
            vmax = float(positive.max())
        return min(max(float(np.quantile(positive, 0.70)), vmax * 0.10), vmax)
    if family == "exact" or positive.size <= 128:
        return float(positive.min())
    return min(max(float(np.quantile(positive, 0.86)), vmax * 0.05), vmax)


def _slice_locations(axis_min: float, axis_max: float, center: tuple[float, float, float], explode: float):
    span = max(axis_max - axis_min, 1e-6)
    offset = 0.36 * span * explode
    return [
        ("z", center[2], (0.0, 0.0, -offset)),
        ("y", center[1], (0.0, -offset, 0.0)),
        ("x", center[0], (-offset, 0.0, 0.0)),
    ]


def _add_translated_slice(plotter, grid, scalar_name: str, cmap_name: str, axis: str, origin: float,
                          translate: tuple[float, float, float], opacity: float, show_scalar_bar: bool,
                          title: str) -> None:
    if axis == "x":
        mesh = grid.slice(normal="x", origin=(origin, 0.0, 0.0))
    elif axis == "y":
        mesh = grid.slice(normal="y", origin=(0.0, origin, 0.0))
    else:
        mesh = grid.slice(normal="z", origin=(0.0, 0.0, origin))
    mesh.translate(np.asarray(translate), inplace=True)
    plotter.add_mesh(
        mesh,
        scalars=scalar_name,
        cmap=cmap_name,
        opacity=opacity,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args={"title": title} if show_scalar_bar else None,
    )
    plotter.add_mesh(mesh.outline(), color="black", line_width=1)


def _add_field_meshes(plotter, grid, values: np.ndarray, family: str, field_name: str, label: str,
                      axis_min: float, axis_max: float, center: tuple[float, float, float],
                      scalar_name: str, cmap_name: str, explode: float) -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        plotter.add_text(f"{label}: no finite data", position="upper_left", font_size=10, color="black")
        return

    vmax = float(np.nanmax(finite))
    vmin = float(np.nanmin(finite))
    if vmax <= 0.0 and vmin >= 0.0:
        plotter.add_text(f"{label}: empty / all zero", position="upper_left", font_size=10, color="black")
        return

    title = _safe_title(label)
    slice_opacity = 0.86 if family == "proxy" else 0.72
    show_bar_once = True
    for axis, origin, translate in _slice_locations(axis_min, axis_max, center, explode):
        _add_translated_slice(
            plotter,
            grid,
            scalar_name,
            cmap_name,
            axis,
            origin,
            translate,
            slice_opacity,
            show_bar_once,
            title,
        )
        show_bar_once = False

    threshold = _point_threshold(values, field_name, family)
    if threshold is None:
        return

    points = grid.cast_to_pointset()
    try:
        cloud = points.threshold((threshold, vmax), scalars=scalar_name)
    except Exception:
        cloud = points.threshold(threshold, scalars=scalar_name)

    if cloud.n_points <= 0:
        return

    if family == "exact":
        point_size = 10.0 if cloud.n_points < 128 else 6.0
        point_opacity = 1.0
    elif field_name == "density":
        point_size = 3.0
        point_opacity = 0.22
    else:
        point_size = 4.0
        point_opacity = 0.38

    plotter.add_mesh(
        cloud,
        scalars=scalar_name,
        cmap=cmap_name,
        style="points",
        point_size=point_size,
        render_points_as_spheres=True,
        opacity=point_opacity,
        show_scalar_bar=False,
    )


def _patch_panel_html_fullscreen(file_path: Path) -> None:
    html = file_path.read_text(encoding="utf-8", errors="replace")
    if "/* IITM_FULLSCREEN_PATCH */" in html:
        return
    patch = """
    <style>
      /* IITM_FULLSCREEN_PATCH */
      html, body {
        width: 100%;
        height: 100%;
        overflow: hidden;
        background: #ffffff;
      }
      body > div, .bk-root, .bk-root > div {
        width: 100% !important;
        height: 100% !important;
        max-width: none !important;
        max-height: none !important;
      }
      .bk-panel-models-layout-HTML,
      .bk-panel-models-vtk-VTK,
      .vtk-container,
      canvas {
        width: 100% !important;
        height: 100% !important;
      }
    </style>
"""
    html = html.replace("</head>", f"{patch}</head>", 1)
    file_path.write_text(html, encoding="utf-8")


def _normalise_catalog(field_catalog: dict, npz_files: set[str]) -> dict[str, list[dict[str, str]]]:
    if not isinstance(field_catalog, dict) or not field_catalog:
        field_catalog = {
            "proxy": [
                {"name": "beta_back", "label": "后向代理场", "storage": "proxy_beta_back"},
                {"name": "beta_forward", "label": "前向代理场", "storage": "proxy_beta_forward"},
                {"name": "depol_ratio", "label": "退偏代理场", "storage": "proxy_depol_ratio"},
                {"name": "density", "label": "密度场", "storage": "density"},
            ],
        }

    normalised: dict[str, list[dict[str, str]]] = {}
    for family, entries in field_catalog.items():
        if not isinstance(entries, list):
            continue
        kept: list[dict[str, str]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            if not name:
                continue
            storage = str(entry.get("storage") or f"{family}_{name}")
            if name == "density" and storage not in npz_files and "density" in npz_files:
                storage = "density"
            if storage not in npz_files:
                continue
            kept.append({
                "name": name,
                "label": str(entry.get("label", name)),
                "storage": storage,
            })
        if kept:
            normalised[str(family)] = kept
    return normalised


def render_iitm_headless(output_dir: Path, field_catalog: dict, shape_type: str, explode_dist: float = 0.7) -> list[str]:
    import nest_asyncio
    import panel as pn
    import pyvista as pv

    nest_asyncio.apply()
    try:
        pv.set_plot_theme("document")
    except Exception:
        pass
    try:
        pn.extension("vtk", design="material", template="plain")
    except Exception:
        pass

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "density.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing density.npz: {npz_path}")

    data = np.load(npz_path)
    if "axis" not in data.files or "density" not in data.files:
        raise ValueError("density.npz must contain axis and density")

    axis = np.asarray(data["axis"], dtype=np.float32)
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError("axis is invalid in density.npz")

    density = np.asarray(data["density"], dtype=np.float32)
    dims = tuple(int(v) for v in density.shape)
    if len(dims) != 3:
        raise ValueError("density must be a 3D array")

    axis_min = float(axis.min())
    axis_max = float(axis.max())
    step = float(axis[1] - axis[0]) if axis.size > 1 else 1.0
    center = (
        0.5 * (axis_min + axis_max),
        0.5 * (axis_min + axis_max),
        0.5 * (axis_min + axis_max),
    )
    camera_by_view = _view_configs(axis_min, axis_max)
    catalog = _normalise_catalog(field_catalog, set(data.files))

    generated: list[str] = []
    for family, entries in catalog.items():
        for entry in entries:
            field_name = entry["name"]
            label = entry["label"]
            storage = entry["storage"]
            values = np.asarray(data[storage], dtype=np.float32)
            if values.shape != dims:
                print(f">> [IITM Renderer] skip {family}/{field_name}: shape {values.shape} != {dims}", flush=True)
                continue

            scalar_name = f"{family}:{field_name}"
            cmap_name = _cmap_for(field_name, family)

            plotter = pv.Plotter(off_screen=True, window_size=[1000, 800])
            plotter.set_background("white")

            grid = pv.ImageData()
            grid.dimensions = dims
            grid.origin = (axis_min, axis_min, axis_min)
            grid.spacing = (step, step, step)
            grid.point_data[scalar_name] = values.flatten(order="F")

            _add_field_meshes(
                plotter,
                grid,
                values,
                family,
                field_name,
                label,
                axis_min,
                axis_max,
                center,
                scalar_name,
                cmap_name,
                float(explode_dist),
            )
            plotter.add_mesh(grid.outline(), color="grey", line_width=1)
            plotter.add_axes()
            plotter.show_grid(color="lightgrey")

            for view_stem, view_name in VIEWS:
                filename = _output_filename(view_stem, family, field_name)
                file_path = output_dir / filename
                plotter.camera_position = camera_by_view[view_name]
                plotter.render()
                vtk_pane = pn.pane.VTK(
                    plotter.ren_win,
                    min_width=1000,
                    min_height=800,
                    sizing_mode="stretch_both",
                    enable_keybindings=True,
                    orientation_widget=True,
                )
                vtk_pane.save(
                    str(file_path),
                    resources="inline",
                    embed=True,
                    title=f"IITM {shape_type} - {label} - {filename}",
                )
                _patch_panel_html_fullscreen(file_path)
                generated.append(filename)

            plotter.close()

    return list(dict.fromkeys(generated))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shape_type", required=True)
    parser.add_argument("--field_catalog", required=True)
    parser.add_argument("--explode_dist", type=float, default=0.7)
    args = parser.parse_args()

    catalog = json.loads(args.field_catalog)
    artifacts = render_iitm_headless(Path(args.output_dir), catalog, args.shape_type, args.explode_dist)
    print(json.dumps(artifacts, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
