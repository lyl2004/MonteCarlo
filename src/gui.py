import os
import json
import sys
import asyncio
import ast
import csv
import math
import platform
import struct
import subprocess
import time
import zipfile
from urllib.parse import urlencode
import httpx
from nicegui import ui, app, native

# =============================================================================
# 0. 全局路径配置
# =============================================================================

SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
LOG_DIR      = os.path.join(PROJECT_ROOT, 'log')
INPUT_BASE   = os.path.join(PROJECT_ROOT, 'inputs')
OUTPUT_BASE  = os.path.join(PROJECT_ROOT, 'outputs')

os.makedirs(LOG_DIR,     exist_ok=True)
os.makedirs(OUTPUT_BASE, exist_ok=True)

app.add_static_files('/static_outputs', OUTPUT_BASE)

# =============================================================================
# 1. 后端注册表
# =============================================================================

BACKEND_REGISTRY = {
    'mie': {
        'label':         'Mie Scattering (Numba)',
        'mode':          'subprocess',
        'pixi_env':      'mie',
        'worker':        'mie_worker.py',
        'output_sub':    'mie',
    },
    'iitm': {
        'label':         'IITM T-Matrix (Julia HTTP)',
        'mode':          'http',
        'pixi_env':      'julia',
        'worker':        'iitm_http_worker.py',
        'output_sub':    'iitm',
        'server_url':    'http://127.0.0.1:2700',
        'health_path':   '/health',
        # 使用 SSE 流式端点，实时推送分步进度
        'simulate_path': '/simulate/stream',
    },
}

# =============================================================================
# 2. 参数定义
# =============================================================================

PARAM_GROUPS = [
    "空间与网格参数", "光学参数", "粒子形态参数 (IITM)",
    "微物理参数 (粒径分布)", "场结构参数 (分形几何)", "仿真参数", "渲染参数", "其他"
]

PARAM_DEF = {
    # 空间与网格
    "L_size":         {"label": "模拟区域物理边长",       "unit": "米",   "group": "空间与网格参数"},
    "grid_dim":       {"label": "3D 网格分辨率",           "unit": "像素", "group": "空间与网格参数"},
    # 光学
    "wavelength_um":  {"label": "激光波长",                "unit": "微米", "group": "光学参数"},
    "m_real":         {"label": "折射率实部 n",            "unit": "无",   "group": "光学参数"},
    "m_imag":         {"label": "折射率虚部 κ",            "unit": "无",   "group": "光学参数"},
    # ── 粒子形态（IITM 专属）────────────────────────────────────────
    # shape_type 由专用 select 控件渲染，此处仍登记以便过滤和保存
    "shape_type":     {"label": "粒子形状",                "unit": "无",   "group": "粒子形态参数 (IITM)"},
    "r_eff":          {"label": "等效体积半径 r_eff",       "unit": "微米", "group": "粒子形态参数 (IITM)"},
    "axis_ratio":     {"label": "轴比 (cyl:h/2r  sph:c/a)","unit": "无",   "group": "粒子形态参数 (IITM)"},
    "nmax_override":  {"label": "截断阶数 nmax (0=自动)",   "unit": "无",   "group": "粒子形态参数 (IITM)"},
    "Nr":             {"label": "径向积分点数 Nr",          "unit": "无",   "group": "粒子形态参数 (IITM)"},
    "Ntheta":         {"label": "角向积分点数 Nϑ",          "unit": "无",   "group": "粒子形态参数 (IITM)"},
    # 微物理
    "r_bottom":       {"label": "云底粒子半径",            "unit": "微米", "group": "微物理参数 (粒径分布)"},
    "r_top":          {"label": "云顶粒子半径",            "unit": "微米", "group": "微物理参数 (粒径分布)"},
    "sigma_ln":       {"label": "对数正态分布宽度 σ_ln",   "unit": "无",   "group": "微物理参数 (粒径分布)"},
    "n_radii":        {"label": "粒径积分点数",            "unit": "无",   "group": "微物理参数 (粒径分布)"},
    # 场结构
    "cloud_center_z": {"label": "云层中心高度",            "unit": "米",   "group": "场结构参数 (分形几何)"},
    "cloud_thickness":{"label": "云层物理厚度",            "unit": "米",   "group": "场结构参数 (分形几何)"},
    "turbulence_scale":{"label":"湍流/分形噪声尺度",       "unit": "无",   "group": "场结构参数 (分形几何)"},
    # 仿真
    "cpu_limit":      {"label": "并行计算核心数",          "unit": "核",   "group": "仿真参数"},
    "photons":        {"label": "投入光子数量",            "unit": "个",   "group": "仿真参数"},
    "visibility_km":  {"label": "大气能见度",              "unit": "千米", "group": "仿真参数"},
    "scale_height_m": {"label": "消光标高 (指数廓线)",     "unit": "米",   "group": "仿真参数"},
    "angstrom_q":     {"label": "Angström 指数 q",         "unit": "无",   "group": "仿真参数"},
    "field_compute_mode": {"label": "场计算模式",          "unit": "无",   "group": "仿真参数"},
    "lidar_enabled": {"label": "输出距离门回波",           "unit": "开关", "group": "激光雷达观测参数"},
    "range_bin_width_m": {"label": "距离门宽度",           "unit": "米",   "group": "激光雷达观测参数"},
    "range_max_m": {"label": "最大观测距离",               "unit": "米",   "group": "激光雷达观测参数"},
    "receiver_overlap_min": {"label": "近场重叠下限",      "unit": "无",   "group": "激光雷达观测参数"},
    "receiver_overlap_full_range_m": {"label": "完全重叠距离", "unit": "米", "group": "激光雷达观测参数"},
    "tmatrix_solver": {"label": "T-Matrix 求解器",         "unit": "无",   "group": "仿真参数"},
    "forward_mode":   {"label": "前向散射取值模式",        "unit": "无",   "group": "仿真参数"},
    "forward_cone_deg": {"label": "前向散射锥角",          "unit": "度",   "group": "仿真参数"},
    "ebcm_threshold": {"label": "EBCM 收敛阈值",           "unit": "无",   "group": "仿真参数"},
    "ebcm_ndgs":      {"label": "EBCM 每阶积分点倍率",     "unit": "无",   "group": "仿真参数"},
    "ebcm_maxiter":   {"label": "EBCM 最大迭代数",         "unit": "无",   "group": "仿真参数"},
    "ebcm_loss_threshold": {"label": "EBCM 精度损失阈值",  "unit": "无",   "group": "仿真参数"},
    "iitm_nr_scale_on_fallback": {"label": "回退时 Nr 放大系数", "unit": "无", "group": "仿真参数"},
    "iitm_ntheta_scale_on_fallback": {"label": "回退时 Nϑ 放大系数", "unit": "无", "group": "仿真参数"},
    # 渲染
    "explode_dist":   {"label": "切片爆炸距离系数",        "unit": "无",   "group": "渲染参数"},
}

# IITM 专属字段集合（在 Mie 模式下整组隐藏）
IITM_ONLY_KEYS = {
    'shape_type', 'r_eff', 'axis_ratio', 'nmax_override', 'Nr', 'Ntheta', 'n_radii',
    'tmatrix_solver', 'forward_mode', 'forward_cone_deg',
    'ebcm_threshold', 'ebcm_ndgs', 'ebcm_maxiter',
    'ebcm_loss_threshold', 'iitm_nr_scale_on_fallback', 'iitm_ntheta_scale_on_fallback'
}
# 下拉选项
SHAPE_TYPE_OPTIONS = {
    'cylinder': '圆柱体 (Cylinder)',
    'spheroid': '旋转椭球 (Spheroid)',
    'sphere':   '球形 (Sphere)',
}

TMATRIX_SOLVER_OPTIONS = {
    'auto': 'AUTO: EBCM -> IITM',
    'ebcm_only': 'EBCM Only',
    'iitm_only': 'IITM Only',
}

FORWARD_MODE_OPTIONS = {
    'cone_avg': 'Cone Average',
    'point': 'Point at 0°',
}

FIELD_COMPUTE_MODE_OPTIONS = {
    'proxy_only': 'Proxy Only',
    'exact_only': 'Exact Only',
    'both': 'Proxy + Exact',
}

SELECT_FIELD_OPTIONS = {
    'shape_type': SHAPE_TYPE_OPTIONS,
    'tmatrix_solver': TMATRIX_SOLVER_OPTIONS,
    'forward_mode': FORWARD_MODE_OPTIONS,
    'field_compute_mode': FIELD_COMPUTE_MODE_OPTIONS,
}

DEFAULT_CONFIG = {
    # 空间
    "L_size": 20.0, "grid_dim": 80,
    # 光学
    "wavelength_um": 1.55, "m_real": 1.311, "m_imag": 1e-4,
    # 粒子形态（IITM）
    "shape_type": "cylinder", "r_eff": 0.0,
    "axis_ratio": 1.0, "nmax_override": 0,
    "Nr": 50, "Ntheta": 80,
    # 微物理
    "r_bottom": 2.0, "r_top": 12.0, "sigma_ln": 0.35, "n_radii": 15,
    # 场结构
    "cloud_center_z": 10.0, "cloud_thickness": 8.0, "turbulence_scale": 4.0,
    # 仿真
    "cpu_limit": 4, "photons": 10000, "visibility_km": 3.0,
    "scale_height_m": 2000.0, "angstrom_q": 1.3,
    "tmatrix_solver": "auto", "forward_mode": "cone_avg", "forward_cone_deg": 0.5,
    "ebcm_threshold": 1e-4,
    "ebcm_ndgs": 4, "ebcm_maxiter": 20, "ebcm_loss_threshold": 10.0,
    "iitm_nr_scale_on_fallback": 1.25,
    "iitm_ntheta_scale_on_fallback": 1.5,
    "field_compute_mode": "proxy_only",
    "lidar_enabled": False,
    "range_bin_width_m": 1.0,
    "range_max_m": 0.0,
    "receiver_overlap_min": 1.0,
    "receiver_overlap_full_range_m": 0.0,
    # 渲染
    "explode_dist": 0.7,
}

# =============================================================================
# 3. 应用状态
# =============================================================================

class AppState:
    def __init__(self):
        self.current_project = None
        self.simulation_backend = 'mie'
        self.config_data = DEFAULT_CONFIG.copy()
        self.current_preview_page = "3d"
        self.current_view_mode = "main"
        self.current_field_family = "proxy"
        self.current_iitm_field = "beta_back"
        self.selected_2d_charts = []
        self.echo_quality_min_events = 1.0
        self.echo_quality_max_rel_error = 0.5
        self.current_field_catalog = {}
        self.available_field_families = ["proxy"]
        self.requested_field_compute_mode = DEFAULT_CONFIG["field_compute_mode"]
        self.effective_field_compute_mode = "proxy_only"
        self.current_artifacts = []
        self.port = 8000
        self.is_running = False
        # IITM 专属：cpu_limit 是否使用 auto（True = --threads=auto）
        self.iitm_cpu_auto = True

state = AppState()

VIEW_MODE_LABELS = {
    "main": "主视",
    "front": "前视",
    "top": "顶视",
    "right": "右视",
}

VIEW_MODE_ORDER = ["main", "front", "top", "right"]

VIEW_MODE_TO_FILENAMES = {
    "main": ("render_main.html",),
    "front": ("render_front.html",),
    "top": ("render_top.html",),
    "right": ("render_right.html",),
}

LIDAR_ECHO_KEYS = [
    "range_bins_m",
    "echo_power",
    "echo_depol",
    "echo_event_count",
    "echo_relative_error_est",
]

LIDAR_DIAGNOSTIC_KEYS = [
    "range_bins_m",
    "echo_I",
    "echo_Q",
    "echo_U",
    "echo_V",
    "echo_power",
    "echo_depol",
    "echo_event_count",
    "echo_weight_sum",
    "echo_weight_sq_sum",
    "echo_power_variance_est",
    "echo_power_ci_low",
    "echo_power_ci_high",
    "echo_relative_error_est",
]

OPTIONAL_2D_CHART_OPTIONS = {
    "stokes": "Stokes I/Q/U/V",
    "normalized_stokes": "归一化 Stokes q/u/v",
    "event_count": "采样事件数",
    "relative_error": "相对误差",
    "variance": "回波方差估计",
    "confidence_interval": "回波 95% 置信区间",
    "weight_moments": "权重一阶/二阶矩",
    "range_corrected": "R² 修正诊断",
}

FIELD_FAMILY_LABELS = {
    "proxy": "代理场",
    "exact": "精确场",
}

DEFAULT_FIELD_LABELS = {
    "beta_back": "后向代理场",
    "beta_forward": "前向代理场",
    "depol_ratio": "退偏代理场",
    "density": "密度场",
    "event_count": "采样次数",
}

FIELD_LABELS_BY_FAMILY = {
    "proxy": {
        "beta_back": "后向代理场",
        "beta_forward": "前向代理场",
        "depol_ratio": "退偏代理场",
        "density": "密度场",
    },
    "exact": {
        "beta_back": "后向精确场",
        "beta_forward": "前向精确场",
        "depol_ratio": "退偏精确场",
        "event_count": "采样次数",
    },
}

MIE_FIELD_ORDER = ["beta_back", "beta_forward", "depol_ratio", "density"]
IITM_FIELD_ORDER = ["beta_back", "beta_forward", "depol_ratio", "density"]

DEFAULT_BACKEND_FIELD_CATALOG = {
    "mie": {
        "proxy": [
            {"name": name, "label": DEFAULT_FIELD_LABELS[name]}
            for name in MIE_FIELD_ORDER
        ],
    },
    "iitm": {
        "proxy": [
            {"name": name, "label": DEFAULT_FIELD_LABELS[name]}
            for name in IITM_FIELD_ORDER
        ],
    },
}

# =============================================================================
# 4. 路径工具
# =============================================================================

def get_input_dir() -> str:
    path = os.path.join(INPUT_BASE, state.simulation_backend)
    os.makedirs(path, exist_ok=True)
    return path

def get_output_dir() -> str:
    reg  = BACKEND_REGISTRY[state.simulation_backend]
    path = os.path.join(OUTPUT_BASE, reg['output_sub'])
    os.makedirs(path, exist_ok=True)
    return path

def get_formatted_label(key: str) -> str:
    if key in PARAM_DEF:
        info = PARAM_DEF[key]
        unit = info.get('unit', '无')
        return f"{info['label']} ({key}, {unit})" if unit != '无' else f"{info['label']} ({key})"
    return key

def open_log_folder():
    try:
        if platform.system() == "Windows":
            os.startfile(LOG_DIR)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", LOG_DIR])
        else:
            subprocess.Popen(["xdg-open", LOG_DIR])
        ui.notify(f"已打开日志目录: {LOG_DIR}", type='info')
    except Exception as e:
        ui.notify(f"打开失败: {e}", type='negative')

# =============================================================================
# 5. IITM 服务健康检查（异步，不阻塞 UI）
# =============================================================================

async def check_iitm_server_health() -> dict | None:
    reg = BACKEND_REGISTRY.get('iitm', {})
    url = reg.get('server_url', 'http://127.0.0.1:2700') + reg.get('health_path', '/health')
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return None

async def _update_server_status(dot_el, label_el):
    info = await check_iitm_server_health()
    if info:
        dot_el.classes(remove='bg-gray-400 bg-red-400', add='bg-green-400')
        threads = info.get('threads', '?')
        label_el.text = f"Julia 服务在线 | {threads} 线程"
        label_el.classes(remove='text-gray-500 text-red-500', add='text-green-600')
    else:
        dot_el.classes(remove='bg-gray-400 bg-green-400', add='bg-red-400')
        label_el.text = "Julia 服务未启动（运行仿真时自动启动）"
        label_el.classes(remove='text-gray-500 text-green-600', add='text-red-500')

# =============================================================================
# 6. UI 刷新逻辑
# =============================================================================

def create_value_handler(key: str):
    def handler(e):
        if e is None:
            return
        state.config_data[key] = e.value
    return handler

def load_ui_from_config():
    input_container.clear()
    with input_container:
        is_iitm = (state.simulation_backend == 'iitm')

        # ── 后端选择器 ─────────────────────────────────────────────
        ui.label("仿真内核 (Backend)").classes('text-xs font-bold text-gray-500')
        backend_options = {k: v['label'] for k, v in BACKEND_REGISTRY.items()}
        backend_select  = ui.select(
            options=backend_options,
            value=state.simulation_backend,
            label='选择仿真引擎'
        ).classes('w-full mb-2')

        def on_backend_change(e):
            state.simulation_backend = e.value
            state.current_project    = None
            state.current_artifacts  = []
            state.current_field_family = "proxy"
            state.current_iitm_field = "beta_back"
            _set_field_catalog(None, e.value)
            ui.notify(f"已切换至 {BACKEND_REGISTRY[e.value]['label']}", type='warning')
            load_ui_from_config()
            refresh_preview()

        backend_select.on_value_change(on_backend_change)

        # ── IITM 专属：Julia 服务状态指示条 ───────────────────────
        if is_iitm:
            with ui.row().classes('w-full items-center gap-2 mb-2 p-2 bg-blue-50 rounded'):
                server_dot   = ui.element('div').classes(
                    'w-2 h-2 rounded-full bg-gray-400 flex-shrink-0')
                server_label = ui.label('Julia 服务状态检查中…').classes(
                    'text-xs text-gray-500 flex-grow')
                ui.button('检查', on_click=lambda: asyncio.ensure_future(
                    _update_server_status(server_dot, server_label))
                ).props('flat dense size=xs')
            asyncio.ensure_future(_update_server_status(server_dot, server_label))

        ui.separator().classes('mb-2')

        # ── 当前工程标签 ───────────────────────────────────────────
        proj_txt = state.current_project or '未选择'
        ui.label(f"当前工程: {proj_txt}").classes('text-lg font-bold mb-4 text-blue-900')

        # ── 参数分组渲染 ───────────────────────────────────────────
        grouped_data: dict[str, dict] = {g: {} for g in PARAM_GROUPS}
        for key, value in state.config_data.items():
            if key not in PARAM_DEF:
                continue
            # Mie 模式下隐藏 IITM 专属字段
            if not is_iitm and key in IITM_ONLY_KEYS:
                continue
            group = PARAM_DEF[key].get("group", "其他")
            if group not in grouped_data:
                group = "其他"
            grouped_data[group][key] = value

        for group in PARAM_GROUPS:
            sub_data = grouped_data.get(group, {})
            if not sub_data:
                continue

            # 粒子形态组：在 Mie 模式下整体隐藏
            if group == "粒子形态参数 (IITM)" and not is_iitm:
                continue

            with ui.expansion(group, icon='settings', value=True).classes('w-full bg-blue-50'):
                with ui.column().classes('w-full p-2 gap-1'):

                    # ── 仿真参数组：cpu_limit 特殊处理 ────────────
                    if group == "仿真参数" and is_iitm:
                        _render_iitm_cpu_control()

                    for key, value in sub_data.items():
                        # cpu_limit 在 IITM 模式下已由特殊控件接管
                        if key == 'cpu_limit' and is_iitm:
                            continue

                        label_text = get_formatted_label(key)
                        handler    = create_value_handler(key)

                        if key in SELECT_FIELD_OPTIONS:
                            sel = ui.select(
                                options=SELECT_FIELD_OPTIONS[key],
                                value=value,
                                label=label_text
                            ).props('outlined dense').classes('w-full bg-white')
                            sel.on_value_change(handler)

                        elif isinstance(value, bool):
                            ui.switch(text=label_text, value=value) \
                                .classes('w-full bg-white p-2 rounded') \
                                .on_value_change(handler)

                        elif isinstance(value, (int, float)):
                            ui.number(label=label_text, value=value, format='%.4g') \
                                .props('outlined dense').classes('w-full bg-white') \
                                .on_value_change(handler)

                        else:
                            ui.input(label=label_text, value=str(value)) \
                                .props('outlined dense').classes('w-full bg-white') \
                                .on_value_change(handler)


def _render_iitm_cpu_control():
    """
    IITM 专属 CPU 控件：
      - Toggle 切换 auto / 固定数量
      - auto 时向 Julia 传 --threads=auto（榨干全部性能核）
      - 固定时 number input 可手动填写，保留 PC 余裕
    渲染位置：仿真参数 expansion 内，cpu_limit 数字框之前。
    """
    with ui.column().classes('w-full gap-1 mb-1'):
        with ui.row().classes('w-full items-center justify-between'):
            ui.label('Julia 线程数 (cpu_limit)').classes('text-xs text-gray-600')
            auto_toggle = ui.switch(
                'AUTO（榨干算力）',
                value=state.iitm_cpu_auto
            ).classes('text-xs')

        # 固定数量输入框（auto 时隐藏）
        fixed_input = ui.number(
            label='固定线程数',
            value=state.config_data.get('cpu_limit', 4),
            min=1, max=256, precision=0, format='%d'
        ).props('outlined dense').classes('w-full bg-white')
        fixed_input.set_visibility(not state.iitm_cpu_auto)

        # auto 状态提示（fixed 时隐藏）
        auto_tip = ui.label(
            '将以 --threads=auto 启动 Julia，自动匹配所有可用性能核'
        ).classes('text-xs text-blue-500 italic')
        auto_tip.set_visibility(state.iitm_cpu_auto)

        def on_auto_toggle(e):
            state.iitm_cpu_auto = e.value
            fixed_input.set_visibility(not e.value)
            auto_tip.set_visibility(e.value)
            # auto 时把 cpu_limit 写为特殊哨兵值 -1（worker 侧识别）
            if e.value:
                state.config_data['cpu_limit'] = -1
            else:
                state.config_data['cpu_limit'] = int(fixed_input.value or 4)

        def on_fixed_change(e):
            if not state.iitm_cpu_auto and e.value:
                state.config_data['cpu_limit'] = int(e.value)

        auto_toggle.on_value_change(on_auto_toggle)
        fixed_input.on_value_change(on_fixed_change)

        # 同步初始状态到 config_data
        if state.iitm_cpu_auto:
            state.config_data['cpu_limit'] = -1

# =============================================================================
# 7. 预览刷新
# =============================================================================

def _normalize_artifacts(artifacts: list[str] | None) -> list[str]:
    names: list[str] = []
    for artifact in artifacts or []:
        name = os.path.basename(str(artifact).replace('/', os.sep))
        if name.lower().endswith('.html') and name not in names:
            names.append(name)
    return names


def _artifact_to_view_mode(filename: str) -> str | None:
    for mode, candidates in VIEW_MODE_TO_FILENAMES.items():
        for candidate in candidates:
            stem = candidate[:-5] if candidate.lower().endswith('.html') else candidate
            if filename == candidate or filename.startswith(stem + "__"):
                return mode
    return None


def _default_field_catalog_for_backend(backend: str) -> dict[str, list[dict[str, str]]]:
    base = DEFAULT_BACKEND_FIELD_CATALOG.get(backend, {})
    return {
        family: [{"name": item["name"], "label": item["label"]} for item in items]
        for family, items in base.items()
    }


def _normalize_field_catalog(catalog, backend: str) -> dict[str, list[dict[str, str]]]:
    default_catalog = _default_field_catalog_for_backend(backend)
    if not isinstance(catalog, dict):
        return default_catalog

    normalized: dict[str, list[dict[str, str]]] = {}
    for family, items in catalog.items():
        if not isinstance(family, str) or not isinstance(items, list):
            continue
        fields: list[dict[str, str]] = []
        for item in items:
            if isinstance(item, dict):
                name = item.get("name")
                label = item.get("label", name)
            elif isinstance(item, str):
                name = item
                label = DEFAULT_FIELD_LABELS.get(item, item)
            else:
                continue
            if not isinstance(name, str) or not name:
                continue
            if not isinstance(label, str) or not label:
                label = DEFAULT_FIELD_LABELS.get(name, name)
            fields.append({"name": name, "label": label})
        if fields:
            normalized[family] = fields

    return normalized or default_catalog


def _field_entry(family: str, name: str) -> dict[str, str]:
    label = FIELD_LABELS_BY_FAMILY.get(family, {}).get(name, DEFAULT_FIELD_LABELS.get(name, name))
    return {"name": name, "label": label}


def _infer_field_catalog_from_artifacts(artifacts: list[str]) -> dict[str, list[dict[str, str]]]:
    proxy_order = ["beta_back", "beta_forward", "depol_ratio", "density"]
    exact_order = ["beta_back", "beta_forward", "depol_ratio", "event_count"]
    found: dict[str, set[str]] = {"proxy": set(), "exact": set()}

    for filename in artifacts:
        if not filename.lower().endswith(".html"):
            continue
        stem = filename[:-5]
        if "__exact__" in stem:
            field = stem.rsplit("__exact__", 1)[-1]
            if field:
                found["exact"].add(field)
        elif "__" in stem:
            field = stem.rsplit("__", 1)[-1]
            if field:
                found["proxy"].add(field)
        elif stem.startswith("render_"):
            found["proxy"].add("beta_back")

    catalog: dict[str, list[dict[str, str]]] = {}
    proxy_fields = [name for name in proxy_order if name in found["proxy"]]
    exact_fields = [name for name in exact_order if name in found["exact"]]
    if proxy_fields:
        catalog["proxy"] = [_field_entry("proxy", name) for name in proxy_fields]
    if exact_fields:
        catalog["exact"] = [_field_entry("exact", name) for name in exact_fields]
    return catalog


def _infer_field_catalog_from_npz() -> dict[str, list[dict[str, str]]]:
    if state.simulation_backend not in ("iitm", "mie") or not state.current_project:
        return {}
    npz_path = os.path.join(get_output_dir(), state.current_project, "density.npz")
    if not os.path.exists(npz_path):
        return {}

    try:
        with zipfile.ZipFile(npz_path) as zf:
            names = {os.path.splitext(os.path.basename(name))[0] for name in zf.namelist()}
    except Exception:
        return {}

    catalog: dict[str, list[dict[str, str]]] = {}
    proxy_fields = [
        name for name in ["beta_back", "beta_forward", "depol_ratio", "density"]
        if (name == "density" and "density" in names) or f"proxy_{name}" in names
    ]
    exact_fields = [
        name for name in ["beta_back", "beta_forward", "depol_ratio", "event_count"]
        if f"exact_{name}" in names
    ]
    if proxy_fields:
        catalog["proxy"] = [_field_entry("proxy", name) for name in proxy_fields]
    if exact_fields:
        catalog["exact"] = [_field_entry("exact", name) for name in exact_fields]
    return catalog


def _catalog_for_current_output() -> dict[str, list[dict[str, str]]] | None:
    if state.simulation_backend not in ("iitm", "mie"):
        return None
    inferred = _infer_field_catalog_from_npz()
    if inferred:
        return inferred
    artifacts = _normalize_artifacts(state.current_artifacts)
    if not artifacts:
        artifacts = _discover_output_artifacts()
        state.current_artifacts = artifacts
    inferred = _infer_field_catalog_from_artifacts(artifacts)
    return inferred or None


def _set_field_catalog(catalog=None, backend: str | None = None) -> None:
    backend = backend or state.simulation_backend
    if catalog is None and backend in ("iitm", "mie"):
        catalog = _catalog_for_current_output()
    normalized = _normalize_field_catalog(catalog, backend)
    state.current_field_catalog = normalized
    state.available_field_families = list(normalized.keys()) or ["proxy"]
    if state.current_field_family not in state.available_field_families:
        state.current_field_family = state.available_field_families[0]
    field_entries = normalized.get(state.current_field_family, [])
    field_names = [item["name"] for item in field_entries]
    if field_names and state.current_iitm_field not in field_names:
        state.current_iitm_field = field_names[0]


def _get_active_field_catalog() -> dict[str, list[dict[str, str]]]:
    if not state.current_field_catalog:
        _set_field_catalog(None, state.simulation_backend)
    return state.current_field_catalog


def _get_backend_field_entries() -> list[dict[str, str]]:
    catalog = _get_active_field_catalog()
    families = list(catalog.keys())
    if state.current_field_family not in catalog and families:
        state.current_field_family = families[0]
    return catalog.get(state.current_field_family, [])


def _get_backend_field_order() -> list[str]:
    return [item["name"] for item in _get_backend_field_entries()]


def _get_field_label(field_name: str) -> str:
    for item in _get_backend_field_entries():
        if item["name"] == field_name:
            return item["label"]
    return DEFAULT_FIELD_LABELS.get(field_name, field_name)


def _resolve_preview_filename(view_mode: str) -> str:
    base_name = view_mode if view_mode in VIEW_MODE_LABELS else "main"
    if state.simulation_backend in ("iitm", "mie"):
        return f"render_{base_name}.html"
    field_name = state.current_iitm_field
    field_order = _get_backend_field_order()
    if field_name not in field_order and field_order:
        field_name = field_order[0]
    if not field_name:
        field_name = "beta_back"
    if state.current_field_family == "proxy" and field_name == "beta_back":
        return f"render_{base_name}.html"
    if state.current_field_family != "proxy":
        return f"render_{base_name}__{state.current_field_family}__{field_name}.html"
    return f"render_{base_name}__{field_name}.html"


def _discover_output_artifacts() -> list[str]:
    if not state.current_project:
        return []
    output_dir = os.path.join(get_output_dir(), state.current_project)
    if not os.path.isdir(output_dir):
        return []

    html_files = [name for name in os.listdir(output_dir) if name.lower().endswith('.html')]
    html_files.sort(
        key=lambda name: (
            VIEW_MODE_ORDER.index(_artifact_to_view_mode(name))
            if _artifact_to_view_mode(name) in VIEW_MODE_ORDER else len(VIEW_MODE_ORDER),
            name,
        )
    )
    return html_files


def _get_available_view_files() -> dict[str, str]:
    artifacts = _normalize_artifacts(state.current_artifacts)
    if not artifacts:
        artifacts = _discover_output_artifacts()
        state.current_artifacts = artifacts

    view_files: dict[str, str] = {}
    for filename in artifacts:
        if state.simulation_backend in ("iitm", "mie"):
            if filename not in {"render_main.html", "render_front.html", "render_top.html", "render_right.html"}:
                continue
        mode = _artifact_to_view_mode(filename)
        if mode and mode not in view_files:
            view_files[mode] = filename
    return view_files


def _read_npy_header(buffer: bytes) -> tuple[dict, int]:
    if not buffer.startswith(b"\x93NUMPY"):
        raise ValueError("invalid npy header")
    major = buffer[6]
    if major == 1:
        header_len = struct.unpack_from("<H", buffer, 8)[0]
        offset = 10
    elif major in (2, 3):
        header_len = struct.unpack_from("<I", buffer, 8)[0]
        offset = 12
    else:
        raise ValueError(f"unsupported npy version: {major}")
    header = buffer[offset:offset + header_len].decode("latin1").strip()
    meta = ast.literal_eval(header)
    return meta, offset + header_len


def _read_npy_1d(buffer: bytes) -> list[float]:
    meta, data_start = _read_npy_header(buffer)
    shape = tuple(meta.get("shape", ()))
    if len(shape) != 1:
        raise ValueError("only 1D npy arrays are supported")
    count = int(shape[0])
    descr = str(meta.get("descr", ""))
    fmt_map = {
        "<f4": "f", ">f4": "f", "<f8": "d", ">f8": "d",
        "<i2": "h", ">i2": "h", "<i4": "i", ">i4": "i",
        "<i8": "q", ">i8": "q", "<u2": "H", ">u2": "H",
        "<u4": "I", ">u4": "I", "<u8": "Q", ">u8": "Q",
        "|u1": "B", "|i1": "b",
    }
    if descr not in fmt_map:
        raise ValueError(f"unsupported dtype: {descr}")
    endian = ">" if descr.startswith(">") else "<"
    fmt = endian + (fmt_map[descr] * count)
    values = struct.unpack_from(fmt, buffer, data_start)
    return [float(v) for v in values]


def _read_npy_text(buffer: bytes) -> str:
    meta, data_start = _read_npy_header(buffer)
    descr = str(meta.get("descr", ""))
    raw = buffer[data_start:]
    if descr.startswith("<U") or descr.startswith(">U"):
        encoding = "utf-32-le" if descr.startswith("<") else "utf-32-be"
        return raw.decode(encoding, errors="ignore").rstrip("\x00")
    if descr.startswith("|S") or descr.startswith("|a"):
        return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    raise ValueError(f"unsupported text dtype: {descr}")


def _load_receiver_model_from_zip(zf: zipfile.ZipFile, names: set[str]) -> dict:
    try:
        if "receiver_model_json_utf8.npy" in names:
            values = _read_npy_1d(zf.read("receiver_model_json_utf8.npy"))
            payload = bytes(max(0, min(255, int(v))) for v in values)
            return json.loads(payload.decode("utf-8"))
        if "receiver_model_json.npy" in names:
            payload = _read_npy_text(zf.read("receiver_model_json.npy"))
            return json.loads(payload)
    except Exception:
        return {}
    return {}


def _median(values: list[float]) -> float | None:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return None
    mid = len(finite) // 2
    if len(finite) % 2:
        return float(finite[mid])
    return float(0.5 * (finite[mid - 1] + finite[mid]))


def _load_lidar_echo_preview() -> dict:
    if not state.current_project or state.simulation_backend not in ("mie", "iitm"):
        return {"available": False, "reason": "no_project"}
    npz_path = os.path.join(get_output_dir(), state.current_project, "density.npz")
    if not os.path.exists(npz_path):
        return {"available": False, "reason": "missing_density_npz"}
    try:
        arrays = {}
        with zipfile.ZipFile(npz_path, "r") as zf:
            names = set(zf.namelist())
            for key in LIDAR_ECHO_KEYS:
                member = f"{key}.npy"
                if member not in names:
                    return {"available": False, "reason": "missing_echo_arrays"}
                arrays[key] = _read_npy_1d(zf.read(member))
        n = min(len(arrays[key]) for key in LIDAR_ECHO_KEYS)
        if n <= 0:
            return {"available": False, "reason": "empty_echo_arrays"}
        for key in LIDAR_ECHO_KEYS:
            arrays[key] = arrays[key][:n]
        counts = arrays["echo_event_count"]
        rel_err = arrays["echo_relative_error_est"]
        valid = [i for i, c in enumerate(counts) if c > 0.0]
        valid_rel = [rel_err[i] for i in valid if math.isfinite(rel_err[i])]
        return {
            "available": True,
            "arrays": arrays,
            "valid_bin_count": len(valid),
            "event_count_sum": sum(counts),
            "max_relative_error": max(valid_rel) if valid_rel else None,
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _load_lidar_diagnostics() -> dict:
    if not state.current_project or state.simulation_backend not in ("mie", "iitm"):
        return {"available": False, "reason": "no_project"}
    npz_path = os.path.join(get_output_dir(), state.current_project, "density.npz")
    if not os.path.exists(npz_path):
        return {"available": False, "reason": "missing_density_npz"}
    try:
        arrays = {}
        with zipfile.ZipFile(npz_path, "r") as zf:
            names = set(zf.namelist())
            if "range_bins_m.npy" not in names:
                return {"available": False, "reason": "missing_range_bins"}
            for key in LIDAR_DIAGNOSTIC_KEYS:
                member = f"{key}.npy"
                if member in names:
                    arrays[key] = _read_npy_1d(zf.read(member))
        required = {"range_bins_m", "echo_power", "echo_depol"}
        if not required.issubset(arrays):
            return {"available": False, "reason": "missing_default_echo_arrays"}
        n = min(len(values) for values in arrays.values())
        if n <= 0:
            return {"available": False, "reason": "empty_echo_arrays"}
        for key in list(arrays):
            arrays[key] = arrays[key][:n]
        counts = arrays.get("echo_event_count", [0.0] * n)
        rel_err = arrays.get("echo_relative_error_est", [0.0] * n)
        power = arrays.get("echo_power", [0.0] * n)
        ranges = arrays["range_bins_m"]
        valid = [i for i, c in enumerate(counts) if c > 0.0]
        valid_rel = [rel_err[i] for i in valid if i < len(rel_err) and math.isfinite(rel_err[i])]
        min_events = max(0.0, float(state.echo_quality_min_events))
        max_rel_error = max(0.0, float(state.echo_quality_max_rel_error))
        usable = [
            i for i in valid
            if counts[i] >= min_events
            and i < len(rel_err)
            and i < len(power)
            and power[i] > 0.0
            and math.isfinite(rel_err[i])
            and rel_err[i] <= max_rel_error
        ]
        usable_ranges = [ranges[i] for i in usable if i < len(ranges)]
        return {
            "available": True,
            "arrays": arrays,
            "receiver_model": _load_receiver_model_from_zip(zf, names),
            "valid_bin_count": len(valid),
            "usable_bin_count": len(usable),
            "usable_range_min": min(usable_ranges) if usable_ranges else None,
            "usable_range_max": max(usable_ranges) if usable_ranges else None,
            "event_count_sum": sum(counts),
            "median_event_count": _median([counts[i] for i in valid]) if valid else None,
            "median_relative_error": _median(valid_rel) if valid_rel else None,
            "max_relative_error": max(valid_rel) if valid_rel else None,
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _current_density_npz_path() -> str:
    if not state.current_project:
        return ""
    return os.path.join(get_output_dir(), state.current_project, "density.npz")


def _current_export_dir() -> str:
    if not state.current_project:
        return ""
    path = os.path.join(get_output_dir(), state.current_project, "exports")
    os.makedirs(path, exist_ok=True)
    return path


def _export_echo_observation_csv() -> None:
    diag = _load_lidar_diagnostics()
    if not diag.get("available"):
        ui.notify(f"无法导出: {diag.get('reason', 'missing data')}", type="negative")
        return
    arrays = diag["arrays"]
    keys = [key for key in LIDAR_DIAGNOSTIC_KEYS if key in arrays]
    n = min(len(arrays[key]) for key in keys)
    export_path = os.path.join(_current_export_dir(), "echo_observation.csv")
    with open(export_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n):
            writer.writerow([arrays[key][i] for key in keys])
    ui.notify(f"已导出 CSV: {export_path}", type="positive")


def _export_echo_observation_npz() -> None:
    npz_path = _current_density_npz_path()
    if not os.path.exists(npz_path):
        ui.notify("无法导出: density.npz 不存在", type="negative")
        return
    export_path = os.path.join(_current_export_dir(), "echo_observation.npz")
    with zipfile.ZipFile(npz_path, "r") as src, zipfile.ZipFile(
        export_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as dst:
        names = set(src.namelist())
        written = 0
        for key in LIDAR_DIAGNOSTIC_KEYS:
            member = f"{key}.npy"
            if member in names:
                dst.writestr(member, src.read(member))
                written += 1
        for member in ("receiver_model_json.npy", "receiver_model_json_utf8.npy"):
            if member in names:
                dst.writestr(member, src.read(member))
    if written == 0:
        ui.notify("无法导出: 未找到 echo_* 数组", type="negative")
        return
    ui.notify(f"已导出 NPZ: {export_path}", type="positive")


def _export_result_metadata_json() -> None:
    diag = _load_lidar_diagnostics()
    if not diag.get("available"):
        ui.notify(f"无法导出: {diag.get('reason', 'missing data')}", type="negative")
        return
    export_path = os.path.join(_current_export_dir(), "result_metadata.json")
    payload = {
        "project_name": state.current_project,
        "backend": state.simulation_backend,
        "exported_at_unix": time.time(),
        "output_dir": os.path.join(get_output_dir(), state.current_project),
        "density_npz": _current_density_npz_path(),
        "receiver_model": diag.get("receiver_model", {}),
        "echo_summary": {
            "valid_bin_count": diag.get("valid_bin_count"),
            "usable_bin_count": diag.get("usable_bin_count"),
            "usable_range_min_m": diag.get("usable_range_min"),
            "usable_range_max_m": diag.get("usable_range_max"),
            "event_count_sum": diag.get("event_count_sum"),
            "median_event_count": diag.get("median_event_count"),
            "median_relative_error": diag.get("median_relative_error"),
            "max_relative_error": diag.get("max_relative_error"),
            "quality_thresholds": {
                "min_event_count": state.echo_quality_min_events,
                "max_relative_error": state.echo_quality_max_rel_error,
            },
        },
        "requested_field_compute_mode": state.requested_field_compute_mode,
        "effective_field_compute_mode": state.effective_field_compute_mode,
        "available_field_families": state.available_field_families,
        "field_catalog": state.current_field_catalog,
        "artifacts": state.current_artifacts,
        "run_config": state.config_data,
    }
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    ui.notify(f"已导出元数据: {export_path}", type="positive")


def _render_export_menu(disabled: bool = False) -> None:
    button = ui.button("导出", icon="ios_share").props("dense outline")
    if disabled:
        button.disable()
    with button:
        with ui.menu():
            ui.menu_item("导出观测 CSV", on_click=_export_echo_observation_csv)
            ui.menu_item("导出观测 NPZ", on_click=_export_echo_observation_npz)
            ui.menu_item("导出元数据 JSON", on_click=_export_result_metadata_json)


def _downsample_xy(x_values: list[float], y_values: list[float], max_points: int = 320) -> list[list[float]]:
    n = min(len(x_values), len(y_values))
    if n <= 0:
        return []
    step = max(1, math.ceil(n / max_points))
    data = []
    for idx in range(0, n, step):
        x = x_values[idx]
        y = y_values[idx]
        if math.isfinite(x) and math.isfinite(y):
            data.append([x, y])
    return data


def _safe_div_series(numer: list[float], denom: list[float]) -> list[float]:
    n = min(len(numer), len(denom))
    out = []
    for i in range(n):
        d = denom[i]
        out.append(numer[i] / d if abs(d) > 1e-30 and math.isfinite(d) else 0.0)
    return out


def _series_chart(title: str, x_values: list[float], series: list[dict],
                  y_name: str = "", y_min=None, y_max=None, log_y: bool = False):
    chart_series = []
    for item in series:
        line_style = {"width": item.get("width", 2)}
        if item.get("color"):
            line_style["color"] = item["color"]
        chart_series.append({
            "name": item["name"],
            "type": "line",
            "showSymbol": False,
            "data": _downsample_xy(x_values, item["values"], max_points=640),
            "lineStyle": line_style,
        })
    y_axis = {"type": "log" if log_y else "value", "name": y_name, "scale": y_min is None and y_max is None}
    if y_min is not None:
        y_axis["min"] = y_min
    if y_max is not None:
        y_axis["max"] = y_max
    with ui.card().classes("w-full p-3 gap-2").style("border-radius:8px;"):
        ui.label(title).classes("text-sm font-bold text-gray-800")
        ui.echart({
            "animation": False,
            "grid": {"left": 58, "right": 24, "top": 24, "bottom": 42},
            "tooltip": {"trigger": "axis"},
            "legend": {"bottom": 0, "textStyle": {"fontSize": 10}},
            "xAxis": {"type": "value", "name": "R (m)", "nameGap": 20},
            "yAxis": y_axis,
            "series": chart_series,
        }).classes("w-full h-72")


def _render_preview_page_tabs() -> None:
    with ui.row().classes("absolute top-4 left-4 z-50 gap-2"):
        base = "min-width:96px; font-weight:bold; box-shadow:0 2px 4px rgba(0,0,0,0.2);"

        def style(page_name):
            active = " background:#1d4ed8; color:white;"
            normal = " background:rgba(255,255,255,0.94); color:black;"
            return base + (active if state.current_preview_page == page_name else normal)

        ui.button("3D 场展示", on_click=lambda: _switch_preview_page("3d")) \
            .style(style("3d")).props("dense size=sm")
        ui.button("2D 回波诊断", on_click=lambda: _switch_preview_page("2d")) \
            .style(style("2d")).props("dense size=sm")


def _switch_preview_page(page_name: str) -> None:
    state.current_preview_page = page_name
    refresh_preview()


def _render_2d_diagnostics_page() -> None:
    _render_preview_page_tabs()
    diag = _load_lidar_diagnostics()
    with ui.column().classes("absolute inset-0 pt-16 p-4 gap-3 bg-gray-50 overflow-auto"):
        with ui.row().classes("w-full items-center justify-between gap-3"):
            with ui.column().classes("gap-0"):
                ui.label("2D 回波诊断").classes("text-lg font-bold text-gray-900")
                if diag.get("available"):
                    max_rel = diag.get("max_relative_error")
                    med_rel = diag.get("median_relative_error")
                    rel_text = "n/a" if max_rel is None else f"{max_rel:.3g}"
                    med_rel_text = "n/a" if med_rel is None else f"{med_rel:.3g}"
                    ui.label(
                        f"valid bins={diag['valid_bin_count']}  usable bins={diag['usable_bin_count']}  "
                        f"events={diag['event_count_sum']:.0f}  median/max rel.err={med_rel_text}/{rel_text}"
                    ).classes("text-xs text-gray-600")
                else:
                    ui.label("当前项目未输出距离门观测数组").classes("text-xs text-gray-500")
            with ui.row().classes("items-center gap-2"):
                selected = ui.select(
                    OPTIONAL_2D_CHART_OPTIONS,
                    value=state.selected_2d_charts,
                    label="添加诊断图",
                    multiple=True,
                    on_change=lambda e: _update_2d_chart_selection(e.value),
                ).classes("w-80")
                selected.props("use-chips clearable dense")
                _render_export_menu(disabled=not diag.get("available"))

        if not diag.get("available"):
            with ui.card().classes("w-full p-6 items-center gap-2").style("border-radius:8px;"):
                ui.icon("show_chart", size="44px").classes("text-gray-300")
                ui.label("没有可绘制的二维回波数据").classes("text-gray-500 font-bold")
                ui.label("请启用 lidar_enabled 后重新运行仿真。").classes("text-xs text-gray-400")
            return

        arrays = diag["arrays"]
        ranges = arrays["range_bins_m"]
        power = arrays["echo_power"]
        depol = arrays["echo_depol"]
        with ui.card().classes("w-full p-3 gap-3").style("border-radius:8px;"):
            with ui.row().classes("w-full items-center gap-4"):
                usable_min = diag.get("usable_range_min")
                usable_max = diag.get("usable_range_max")
                usable_range = "n/a" if usable_min is None else f"{usable_min:.3g} - {usable_max:.3g} m"
                med_count = diag.get("median_event_count")
                med_count_text = "n/a" if med_count is None else f"{med_count:.3g}"
                ui.label(f"可用距离: {usable_range}").classes("text-sm font-bold text-gray-800")
                ui.label(f"事件数中位数: {med_count_text}").classes("text-xs text-gray-600")
                ui.label(f"质量门限: count >= {state.echo_quality_min_events:g}, rel.err <= {state.echo_quality_max_rel_error:g}") \
                    .classes("text-xs text-gray-600")
            with ui.row().classes("w-full items-center gap-3"):
                ui.number(
                    "最小事件数",
                    value=state.echo_quality_min_events,
                    min=0,
                    step=1,
                    on_change=lambda e: _update_echo_quality_min_events(e.value),
                ).classes("w-40").props("dense")
                ui.number(
                    "最大相对误差",
                    value=state.echo_quality_max_rel_error,
                    min=0,
                    step=0.05,
                    format="%.3f",
                    on_change=lambda e: _update_echo_quality_max_rel_error(e.value),
                ).classes("w-40").props("dense")
        _series_chart(
            "回波强度 P(R) - 线性坐标",
            ranges,
            [{"name": "P(R)", "values": power, "color": "#2563eb"}],
            y_name="P(R)",
        )
        _series_chart(
            "偏振态摘要",
            ranges,
            [{"name": "depol", "values": depol, "color": "#dc2626"}],
            y_name="depol",
            y_min=0,
            y_max=1,
        )

        selected_keys = [key for key in state.selected_2d_charts if key in OPTIONAL_2D_CHART_OPTIONS]
        if "stokes" in selected_keys and all(k in arrays for k in ("echo_I", "echo_Q", "echo_U", "echo_V")):
            _series_chart(
                "Stokes 通道",
                ranges,
                [
                    {"name": "I", "values": arrays["echo_I"], "color": "#2563eb"},
                    {"name": "Q", "values": arrays["echo_Q"], "color": "#dc2626"},
                    {"name": "U", "values": arrays["echo_U"], "color": "#16a34a"},
                    {"name": "V", "values": arrays["echo_V"], "color": "#9333ea"},
                ],
                y_name="Stokes",
            )
        if "normalized_stokes" in selected_keys and all(k in arrays for k in ("echo_I", "echo_Q", "echo_U", "echo_V")):
            _series_chart(
                "归一化 Stokes",
                ranges,
                [
                    {"name": "q=Q/I", "values": _safe_div_series(arrays["echo_Q"], arrays["echo_I"]), "color": "#dc2626"},
                    {"name": "u=U/I", "values": _safe_div_series(arrays["echo_U"], arrays["echo_I"]), "color": "#16a34a"},
                    {"name": "v=V/I", "values": _safe_div_series(arrays["echo_V"], arrays["echo_I"]), "color": "#9333ea"},
                ],
                y_name="q/u/v",
                y_min=-1,
                y_max=1,
            )
        if "event_count" in selected_keys and "echo_event_count" in arrays:
            _series_chart(
                "采样事件数",
                ranges,
                [{"name": "event_count", "values": arrays["echo_event_count"], "color": "#0f766e"}],
                y_name="count",
            )
        if "relative_error" in selected_keys and "echo_relative_error_est" in arrays:
            _series_chart(
                "相对误差估计",
                ranges,
                [{"name": "relative_error", "values": arrays["echo_relative_error_est"], "color": "#ea580c"}],
                y_name="rel.err",
            )
        if "variance" in selected_keys and "echo_power_variance_est" in arrays:
            _series_chart(
                "回波方差估计",
                ranges,
                [{"name": "variance", "values": arrays["echo_power_variance_est"], "color": "#7c3aed"}],
                y_name="variance",
            )
        if "confidence_interval" in selected_keys and all(k in arrays for k in ("echo_power_ci_low", "echo_power_ci_high")):
            _series_chart(
                "回波 95% 置信区间",
                ranges,
                [
                    {"name": "P(R)", "values": power, "color": "#2563eb", "width": 2},
                    {"name": "CI low", "values": arrays["echo_power_ci_low"], "color": "#93c5fd", "width": 1},
                    {"name": "CI high", "values": arrays["echo_power_ci_high"], "color": "#1d4ed8", "width": 1},
                ],
                y_name="P(R)",
            )
        if "weight_moments" in selected_keys:
            moment_series = []
            if "echo_weight_sum" in arrays:
                moment_series.append({"name": "sum_w", "values": arrays["echo_weight_sum"], "color": "#2563eb"})
            if "echo_weight_sq_sum" in arrays:
                moment_series.append({"name": "sum_w2", "values": arrays["echo_weight_sq_sum"], "color": "#dc2626"})
            if moment_series:
                _series_chart("权重矩", ranges, moment_series, y_name="weight")
        if "range_corrected" in selected_keys:
            corrected = [p * r * r for p, r in zip(power, ranges)]
            _series_chart(
                "R² 修正诊断",
                ranges,
                [{"name": "P(R) R²", "values": corrected, "color": "#0891b2"}],
                y_name="P(R) R²",
            )


def _update_2d_chart_selection(value) -> None:
    state.selected_2d_charts = list(value or [])
    refresh_preview()


def _update_echo_quality_min_events(value) -> None:
    try:
        state.echo_quality_min_events = max(0.0, float(value or 0.0))
    except Exception:
        state.echo_quality_min_events = 1.0
    refresh_preview()


def _update_echo_quality_max_rel_error(value) -> None:
    try:
        state.echo_quality_max_rel_error = max(0.0, float(value or 0.0))
    except Exception:
        state.echo_quality_max_rel_error = 0.5
    refresh_preview()


def _render_lidar_echo_panel() -> None:
    echo = _load_lidar_echo_preview()
    panel_style = (
        "width:430px; max-width:calc(100vw - 32px); "
        "background:rgba(255,255,255,0.94); color:#111827; "
        "border:1px solid rgba(15,23,42,0.18); "
        "box-shadow:0 10px 24px rgba(15,23,42,0.18); "
        "border-radius:8px; padding:10px;"
    )
    with ui.element("div").classes("absolute bottom-4 left-4 z-50").style(panel_style):
        with ui.row().classes("w-full items-center justify-between gap-2"):
            ui.label("Lidar Echo").classes("text-sm font-bold")
            if echo.get("available"):
                ui.label(f"valid bins: {echo['valid_bin_count']}").classes("text-xs text-gray-600")
        if not echo.get("available"):
            ui.label("当前项目未输出距离门观测").classes("text-xs text-gray-500")
            return

        arrays = echo["arrays"]
        range_bins = arrays["range_bins_m"]
        power_data = _downsample_xy(range_bins, arrays["echo_power"])
        depol_data = _downsample_xy(range_bins, arrays["echo_depol"])
        max_rel = echo.get("max_relative_error")
        rel_text = "n/a" if max_rel is None else f"{max_rel:.3g}"
        ui.label(
            f"events={echo['event_count_sum']:.0f}  max rel.err={rel_text}"
        ).classes("text-xs text-gray-600")
        ui.echart({
            "animation": False,
            "grid": {"left": 48, "right": 48, "top": 16, "bottom": 36},
            "tooltip": {"trigger": "axis"},
            "legend": {"bottom": 0, "textStyle": {"fontSize": 10}},
            "xAxis": {"type": "value", "name": "R (m)", "nameGap": 18},
            "yAxis": [
                {"type": "value", "name": "P(R)", "scale": True},
                {"type": "value", "name": "depol", "min": 0, "max": 1},
            ],
            "series": [
                {
                    "name": "P(R)",
                    "type": "line",
                    "showSymbol": False,
                    "data": power_data,
                    "lineStyle": {"width": 2, "color": "#2563eb"},
                },
                {
                    "name": "depol",
                    "type": "line",
                    "showSymbol": False,
                    "yAxisIndex": 1,
                    "data": depol_data,
                    "lineStyle": {"width": 2, "color": "#dc2626"},
                },
            ],
        }).classes("w-full h-56")


def _build_field_preview_query() -> str:
    max_grid = int(float(state.config_data.get("preview_max_grid", 48) or 48))
    params = {
        "t": f"{time.time()}",
        "embed": "1",
        "family": state.current_field_family,
        "field": state.current_iitm_field,
        "max_grid": str(max(16, min(max_grid, 96))),
    }
    return urlencode(params)


def _post_field_preview_update() -> None:
    if state.simulation_backend not in ("iitm", "mie"):
        refresh_preview()
        return
    payload = json.dumps({
        "type": "iitm:set_field",
        "family": state.current_field_family,
        "field": state.current_iitm_field,
    }, ensure_ascii=False)
    ui.run_javascript(f"""
    (() => {{
      const frame = document.getElementById('field-preview-frame');
      if (frame && frame.contentWindow) {{
        frame.contentWindow.postMessage({payload}, '*');
      }}
    }})();
    """)


def _switch_preview_family(family_name: str) -> None:
    state.current_field_family = family_name
    field_order = _get_backend_field_order()
    if field_order and state.current_iitm_field not in field_order:
        state.current_iitm_field = field_order[0]
    refresh_preview()


def _switch_preview_field(field_name: str) -> None:
    state.current_iitm_field = field_name
    _post_field_preview_update()


def refresh_preview(view_mode=None):
    if view_mode:
        state.current_view_mode = view_mode

    if not state.current_project:
        viewer_container.clear()
        with viewer_container:
            ui.label('请先加载或新建工程').classes('absolute-center text-gray-400')
        return

    proj    = state.current_project
    reg     = BACKEND_REGISTRY[state.simulation_backend]
    _set_field_catalog(state.current_field_catalog or None, state.simulation_backend)
    view_files = _get_available_view_files()
    family_order = state.available_field_families
    field_entries = _get_backend_field_entries()
    field_order = _get_backend_field_order()
    if state.current_iitm_field not in field_order:
        state.current_iitm_field = field_order[0]
    if view_files and state.current_view_mode not in view_files:
        state.current_view_mode = next(
            (mode for mode in VIEW_MODE_ORDER if mode in view_files),
            "main",
        )
    target_filename = _resolve_preview_filename(state.current_view_mode)
    full_output_dir = os.path.join(get_output_dir(), proj)
    file_path = os.path.join(full_output_dir, target_filename)
    if not os.path.exists(file_path):
        fallback_filename = view_files.get(state.current_view_mode, "render_main.html")
        target_filename = fallback_filename
        file_path = os.path.join(full_output_dir, target_filename)

    viewer_container.clear()
    with viewer_container:
        if state.current_preview_page == "2d":
            _render_2d_diagnostics_page()
            return

        _render_preview_page_tabs()

        if view_files:
            with ui.row().classes('absolute top-4 right-4 z-50 gap-2'):
                btn_base = 'min-width:40px; font-weight:bold; box-shadow:0 2px 4px rgba(0,0,0,0.2);'

                def btn_style(mode):
                    active = ' background:#3b82f6; color:white;'
                    normal = ' background:rgba(255,255,255,0.9); color:black;'
                    return btn_base + (active if state.current_view_mode == mode else normal)

                for mode in VIEW_MODE_ORDER:
                    if mode not in view_files:
                        continue
                    ui.button(VIEW_MODE_LABELS.get(mode, mode), on_click=lambda m=mode: refresh_preview(m)) \
                        .style(btn_style(mode)).props('dense size=sm')

        if state.simulation_backend in ('iitm', 'mie'):
            with ui.row().classes('absolute top-16 left-4 z-50 gap-2'):
                family_btn_base = 'min-width:72px; font-weight:bold; box-shadow:0 2px 4px rgba(0,0,0,0.2);'

                def family_btn_style(family_name):
                    active = ' background:#7c3aed; color:white;'
                    normal = ' background:rgba(255,255,255,0.92); color:black;'
                    return family_btn_base + (active if state.current_field_family == family_name else normal)

                for family_name in family_order:
                    ui.button(
                        FIELD_FAMILY_LABELS.get(family_name, family_name),
                        on_click=lambda fam=family_name: _switch_preview_family(fam)
                    ).style(family_btn_style(family_name)).props('dense size=sm')

            with ui.row().classes('absolute top-28 left-4 z-50 gap-2'):
                field_btn_base = 'min-width:66px; font-weight:bold; box-shadow:0 2px 4px rgba(0,0,0,0.2);'

                def field_btn_style(field_name):
                    active = ' background:#0f766e; color:white;'
                    normal = ' background:rgba(255,255,255,0.92); color:black;'
                    return field_btn_base + (active if state.current_iitm_field == field_name else normal)

                for item in field_entries:
                    field_name = item["name"]
                    ui.button(
                        _get_field_label(field_name),
                        on_click=lambda f=field_name: _switch_preview_field(f)
                    ).style(field_btn_style(field_name)).props('dense size=sm')

        if os.path.exists(file_path):
            iframe_query = _build_field_preview_query() if state.simulation_backend in ('iitm', 'mie') else f"t={time.time()}"
            iframe_src = (f"/static_outputs/{reg['output_sub']}/{proj}"
                          f"/{target_filename}?{iframe_query}")
            ui.html(
                f'<iframe id="field-preview-frame" src="{iframe_src}" '
                f'style="position:absolute;inset:0;width:100%;height:100%;border:none;display:block;" '
                f'allow="fullscreen" scrolling="no"></iframe>',
                sanitize=False
            ).classes('absolute inset-0 w-full h-full bg-white') \
             .style('position:absolute;inset:0;width:100%;height:100%;display:block;')
        else:
            with ui.column().classes('absolute-center items-center'):
                ui.icon('image_not_supported', size='48px').classes('text-gray-300')
                ui.label('视图文件未生成').classes('text-red-400 font-bold')
                ui.label('请点击"运行仿真"并等待日志显示完成').classes(
                    'text-gray-400 text-xs')

# =============================================================================
# 8. 保存工程
# =============================================================================

def save_project_file() -> bool:
    if not state.current_project:
        ui.notify("请先新建或打开工程", type='negative')
        return False
    try:
        # 保存前将 cpu_limit=-1（auto 哨兵）恢复为可读值
        save_data = state.config_data.copy()
        if save_data.get('cpu_limit') == -1:
            save_data['cpu_limit'] = -1  # 保留哨兵，worker 识别
        path = os.path.join(get_input_dir(), f"{state.current_project}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        ui.notify("配置已保存", type='positive')
        return True
    except Exception as e:
        ui.notify(f"保存失败: {e}", type='negative')
        return False

# =============================================================================
# 9. 仿真执行核心
# =============================================================================

async def run_simulation_script():
    if not state.current_project:
        return
    if state.is_running:
        ui.notify("已有任务正在运行中", type='warning')
        return
    if not save_project_file():
        return

    state.is_running = True
    proj_name        = state.current_project
    backend          = state.simulation_backend
    reg              = BACKEND_REGISTRY[backend]
    config_json_str  = json.dumps(state.config_data)

    # ── cpu_limit 字符串：IITM auto 时传 "auto"，否则传数字 ──────
    raw_cpu = state.config_data.get('cpu_limit', 4)
    if backend == 'iitm' and (state.iitm_cpu_auto or raw_cpu == -1):
        cpu_limit_str = "auto"
    else:
        cpu_limit_str = str(max(1, int(raw_cpu)))
    server_info = await check_iitm_server_health() if backend == 'iitm' else None

    # ── 日志对话框 ─────────────────────────────────────────────────
    log_dialog = ui.dialog().props('persistent')
    with log_dialog, ui.card().classes('w-[900px] h-[600px] bg-black p-0 flex flex-col'):
        with ui.row().classes('w-full bg-gray-800 p-2 items-center justify-between'):
            ui.label(f"终端控制台 — {reg['label']}").classes(
                'text-white font-mono font-bold')
            close_btn = ui.button(icon='close', on_click=log_dialog.close) \
                .props('flat round dense color=white').classes('hidden')
        log_view    = ui.log().classes(
            'w-full flex-grow font-mono text-xs text-green-400 bg-black p-4 overflow-auto')
        status_label = ui.label('正在初始化…').classes('text-gray-400 text-xs p-2')

    log_dialog.open()

    # ── 清理环境变量 ───────────────────────────────────────────────
    env = os.environ.copy()
    for k in ["PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "_OLD_VIRTUAL_PATH"]:
        env.pop(k, None)
    env["PYTHONPATH"]        = SRC_DIR
    env["PYTHONIOENCODING"]  = "utf-8"

    # ── 构建启动命令（两种后端统一入口，均为 python worker） ───────
    worker_script = os.path.join(SRC_DIR, reg['worker'])
    cmd = [
        "pixi", "run", "-e", reg['pixi_env'],
        "python", "-u", worker_script,
        "--project_name", proj_name,
        "--config",       config_json_str,
        "--cpu_limit",    cpu_limit_str,
    ]

    # ── 控制台启动摘要 ─────────────────────────────────────────────
    log_view.push(f"Working Directory : {PROJECT_ROOT}")
    log_view.push(f"Backend           : {reg['label']}")
    log_view.push(f"Mode              : {reg['mode']}")
    log_view.push(f"CPU threads       : {cpu_limit_str}")
    if backend == 'iitm':
        if server_info:
            log_view.push(f"Julia server      : {reg['server_url']}  [online: {server_info.get('threads', '?')} threads]")
            if cpu_limit_str == "auto":
                log_view.push("Thread note       : existing Julia service will be reused; AUTO will not restart it")
            elif server_info.get('threads') != int(cpu_limit_str):
                log_view.push(f"Thread note       : requested {cpu_limit_str}, but existing service will stay at {server_info.get('threads')} until restarted")
        else:
            log_view.push(f"Julia server      : {reg['server_url']}  [offline]")
        shape = state.config_data.get('shape_type', 'cylinder')
        ar    = state.config_data.get('axis_ratio', 1.0)
        reff  = state.config_data.get('r_eff', 0.0)
        nm    = state.config_data.get('nmax_override', 0)
        tm_solver = state.config_data.get('tmatrix_solver', 'auto')
        forward_mode = state.config_data.get('forward_mode', 'cone_avg')
        forward_cone = state.config_data.get('forward_cone_deg', 0.5)
        log_view.push(f"Particle shape    : {shape}  ar={ar}  r_eff={reff}  nmax={'auto' if nm==0 else nm}")
        log_view.push(f"T-Matrix solver   : {tm_solver}")
        log_view.push(f"Forward metric    : {forward_mode}  cone={forward_cone}")
    log_view.push(f"Command           : {' '.join(cmd)}")
    log_view.push("-" * 60)

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            env=env,
        )

        final_json_str = ""
        while True:
            try:
                line = await process.stdout.readline()
            except Exception:
                break
            if not line:
                break
            decoded = line.decode('utf-8', errors='replace').strip()
            if not decoded:
                continue

            # IPC 协议：最后一行为单行 JSON，含 "status" 键
            if decoded.startswith('{') and '"status"' in decoded:
                final_json_str = decoded
                log_view.push(">> 收到 IPC 数据包（任务完成）")
            else:
                log_view.push(decoded)

        if final_json_str:
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except Exception:
                pass
            try:
                res = json.loads(final_json_str)
                _handle_simulation_result(res, status_label, log_view)
            except Exception as parse_err:
                status_label.text = f"数据解析错误: {parse_err}"
                status_label.classes(remove='text-gray-400', add='text-red-500 font-bold')
        else:
            code = await process.wait()
            status_label.text = f"进程异常退出且无数据 (code {code})"
            status_label.classes(remove='text-gray-400', add='text-red-500 font-bold')

    except Exception as e:
        log_view.push(f"\n[GUI Error] 启动失败: {e}")
        status_label.text = "启动失败"
        status_label.classes(remove='text-gray-400', add='text-red-500 font-bold')
    finally:
        state.is_running = False
        close_btn.classes(remove='hidden')


def _handle_simulation_result(res: dict, status_label, log_view):
    """
    解析仿真结果，更新状态标签并刷新预览。
    兼容 mie_worker（subprocess）和 iitm_http_worker（HTTP/SSE）两种格式。

    mie_worker 返回：
      {"status":"success", "metrics":{"duration_sec":…, "R_back":…}, "artifacts":[…]}

    iitm_http_worker 透传 Julia 返回（额外含 omega0, g, depol_ratio 等）：
      {"status":"success", "metrics":{…}, "artifacts":["render_main.html", …]}
    """
    if res.get("status") == "success":
        metrics = res.get("metrics", {})
        state.current_artifacts = _normalize_artifacts(res.get("artifacts", []))
        state.requested_field_compute_mode = str(
            res.get("requested_field_compute_mode",
                    state.config_data.get("field_compute_mode", "proxy_only"))
        )
        state.effective_field_compute_mode = str(
            res.get("effective_field_compute_mode", state.requested_field_compute_mode)
        )
        _set_field_catalog(res.get("field_catalog"), state.simulation_backend)
        dur    = metrics.get("duration_sec", 0.0)
        r_back = metrics.get("R_back",       0.0)
        omega0 = metrics.get("omega0",       None)
        g_val  = metrics.get("g",            None)
        depol  = metrics.get("depol_ratio",  None)
        solver = metrics.get("solver_used", None)
        fallback_used = metrics.get("fallback_used", False)
        fallback_reason = metrics.get("fallback_reason", "")
        fb_ratio = metrics.get("forward_back_ratio", None)
        depol_back = metrics.get("depol_back", None)

        parts = [f"完成！耗时 {dur:.2f}s", f"R_back={r_back:.5f}"]
        if solver:
            parts.append(f"solver={solver}")
        if fallback_used:
            parts.append("fallback")
        if fb_ratio is not None:
            parts.append(f"F/B(proxy)={fb_ratio:.3g}")
        if omega0 is not None:
            parts.append(f"ω₀={omega0:.4f}")
        if g_val is not None:
            parts.append(f"g={g_val:.4f}")
        if depol is not None:
            parts.append(f"depol={depol:.4f}")
        elif depol_back is not None:
            parts.append(f"depol_b={depol_back:.4f}")
        if res.get("lidar_observation_available"):
            parts.append("lidar_echo=on")
        status_label.text = "  |  ".join(parts)
        status_label.classes(remove='text-gray-400 text-red-500',
                             add='text-green-500 font-bold')

        log_view.push("── 仿真指标 " + "─" * 40)
        for k, v in metrics.items():
            if k != "duration_sec" and isinstance(v, (int, float)):
                log_view.push(f"    {k:<22s} = {v:.6g}")
        log_view.push(f"    {'field_mode_req':<22s} = {state.requested_field_compute_mode}")
        log_view.push(f"    {'field_mode_eff':<22s} = {state.effective_field_compute_mode}")
        log_view.push(f"    {'field_families':<22s} = {', '.join(state.available_field_families)}")
        if solver:
            log_view.push(f"    {'solver_used':<22s} = {solver}")
        if metrics.get("solver_requested"):
            log_view.push(f"    {'solver_requested':<22s} = {metrics.get('solver_requested')}")
        if metrics.get("solver_path_summary"):
            log_view.push(f"    {'solver_path_summary':<22s} = {metrics.get('solver_path_summary')}")
        if fallback_used:
            log_view.push(f"    {'fallback_reason':<22s} = {fallback_reason}")

        ui.notify("仿真成功完成", type='positive')
        refresh_preview()
    else:
        err = res.get("error", "未知错误")
        status_label.text = f"仿真失败: {err}"
        status_label.classes(remove='text-gray-400', add='text-red-500 font-bold')
        ui.notify(f"仿真失败: {err}", type='negative')

# =============================================================================
# 10. 工程管理对话框
# =============================================================================

def open_load_action(name: str):
    try:
        path = os.path.join(get_input_dir(), f"{name}.json")
        with open(path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        merged = DEFAULT_CONFIG.copy()
        merged.update(loaded)
        state.config_data = merged
        state.requested_field_compute_mode = str(state.config_data.get('field_compute_mode', 'proxy_only'))
        state.effective_field_compute_mode = state.requested_field_compute_mode
        state.current_field_family = "proxy"
        state.current_iitm_field = "beta_back"
        # 同步 iitm_cpu_auto 状态
        state.iitm_cpu_auto = (state.config_data.get('cpu_limit', 4) == -1)
        state.current_project   = name
        state.current_artifacts = _discover_output_artifacts()
        _set_field_catalog(None, state.simulation_backend)
        state.current_view_mode = "main"
        load_ui_from_config()
        refresh_preview()
        ui.notify(f"已加载: {name}", type='positive')
    except Exception as e:
        ui.notify(f"加载失败: {e}", type='negative')

def delete_project_files(project_name: str, dialog_ref):
    import shutil
    config_file = os.path.join(get_input_dir(), f"{project_name}.json")
    output_dir  = os.path.join(get_output_dir(), project_name)
    if os.path.exists(config_file):
        os.remove(config_file)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    ui.notify(f"已删除项目 '{project_name}'", type='positive')
    if state.current_project == project_name:
        state.current_project = None
        state.current_artifacts = []
        load_ui_from_config()
        refresh_preview()
    dialog_ref.close()
    open_project_manager()

def open_project_manager():
    input_dir = get_input_dir()
    projects  = [f.replace('.json', '')
                 for f in os.listdir(input_dir) if f.endswith('.json')]
    with ui.dialog() as dialog, ui.card().classes('w-[600px] h-[500px] flex flex-col'):
        with ui.row().classes('w-full justify-between items-center'):
            label_txt = f"项目管理  [{BACKEND_REGISTRY[state.simulation_backend]['label']}]"
            ui.label(label_txt).classes('text-xl font-bold')
            ui.button(icon='close', on_click=dialog.close).props('flat round dense')
        if not projects:
            ui.label("暂无项目").classes('text-gray-400 italic')
        else:
            with ui.scroll_area().classes('w-full flex-grow'):
                for proj in projects:
                    with ui.row().classes(
                            'w-full justify-between items-center p-2 hover:bg-gray-100 rounded'):
                        ui.label(proj).classes('font-bold')
                        with ui.row():
                            ui.button(icon='folder_open',
                                      on_click=lambda p=proj: (
                                          open_load_action(p), dialog.close())
                                      ).props('flat dense')
                            ui.button(icon='delete', color='red',
                                      on_click=lambda p=proj: delete_project_files(
                                          p, dialog)
                                      ).props('flat dense')
    dialog.open()

def open_simple_load_dialog():
    input_dir = get_input_dir()
    files     = [f.replace('.json', '')
                 for f in os.listdir(input_dir) if f.endswith('.json')]
    with ui.dialog() as dialog, ui.card().classes('w-96'):
        with ui.row().classes('w-full justify-between items-center'):
            ui.label('快速打开').classes('text-xl font-bold')
            ui.button(icon='close', on_click=dialog.close).props('flat round dense')
        if not files:
            ui.label("无配置文件")
        else:
            select = ui.select(files, label='选择项目').classes('w-full')
            ui.button('加载',
                      on_click=lambda: (
                          open_load_action(select.value), dialog.close()
                      ) if select.value else None
                      ).classes('w-full mt-2')
    dialog.open()

def open_new_dialog():
    with ui.dialog() as dialog, ui.card():
        ui.label('新建工程').classes('text-xl font-bold')
        name_input = ui.input('名称 (例: test1)').classes('w-full')

        def create():
            if not name_input.value:
                return
            val = name_input.value.strip()
            state.current_project = val
            state.config_data     = DEFAULT_CONFIG.copy()
            state.requested_field_compute_mode = str(state.config_data.get('field_compute_mode', 'proxy_only'))
            state.effective_field_compute_mode = state.requested_field_compute_mode
            state.current_field_family = "proxy"
            state.current_iitm_field = "beta_back"
            _set_field_catalog(None, state.simulation_backend)
            state.current_artifacts = []
            # 新建时 IITM 默认 auto
            if state.simulation_backend == 'iitm':
                state.iitm_cpu_auto        = True
                state.config_data['cpu_limit'] = -1
            save_project_file()
            load_ui_from_config()
            refresh_preview()
            dialog.close()

        ui.button('创建', on_click=create)
    dialog.open()

# =============================================================================
# 11. 主布局
# =============================================================================

ui.query('body').style('margin:0; padding:0; overflow:hidden;')

with ui.header().classes('bg-slate-900 items-center justify-between p-2 shadow-md'):
    with ui.row().classes(
            'items-center gap-4 cursor-pointer hover:bg-slate-800 p-1 rounded transition'
    ) as logo_row:
        ui.icon('cloud_queue', size='32px').classes('text-blue-400')
        with ui.column().classes('gap-0'):
            ui.label('MonteCarlo Sim').classes(
                'text-lg font-bold text-white leading-tight')
            ui.label('Heterogeneous Compute Architecture').classes(
                'text-xs text-blue-300 leading-tight')
        with ui.menu() as main_menu:
            ui.menu_item('项目管理',     on_click=open_project_manager)
            ui.menu_item('打开日志目录', on_click=open_log_folder)
    logo_row.on('click', main_menu.open)

    with ui.row().classes('gap-2'):
        ui.button('打开', icon='folder_open',
                  on_click=open_simple_load_dialog).props('flat color=white')
        ui.button('新建', icon='add',
                  on_click=open_new_dialog).props('flat color=white')
        ui.button('保存', icon='save',
                  on_click=save_project_file).props('flat color=white')
        ui.button('运行仿真', icon='play_arrow',
                  on_click=run_simulation_script).props(
                      'unelevated color=green-600 text-color=white')

with ui.splitter(value=30).classes('w-full h-[calc(100vh-64px)]') as splitter:
    with splitter.before:
        with ui.column().classes('w-full p-4 gap-2 scroll-y h-full bg-gray-50'):
            input_container = ui.column().classes('w-full gap-2')
            with input_container:
                load_ui_from_config()

    with splitter.after:
        viewer_container = ui.element('div').classes(
            'w-full h-full min-h-0 relative overflow-hidden bg-white')
        with viewer_container:
            ui.label('等待数据…').classes('absolute-center text-gray-400')

# =============================================================================
# 12. 启动
# =============================================================================

state.port = native.find_open_port()
ui.run(
    title='MonteCarlo Studio',
    native=True,
    host='127.0.0.1',
    port=state.port,
    show=True,
    reload=False,
)
