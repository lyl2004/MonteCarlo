"""
iitm_server.jl — IITM Julia HTTP 服务  v3.0
变更（v3）：
  §3  build_configs：r_repr 安全上限策略，写入 scatter dict 供渲染层使用
  §4  step2_scatter：日志显示 nmax 估算值
      step4_render：接受 scatter 参数，调用 v3 的 compute_scatter_fields

启动：
  julia --threads auto --project=src/julia iitm_server.jl [--port 2700] [--root /path]

API：
  GET  /health
  POST /simulate
  POST /simulate/stream   （SSE，推荐）
  POST /scatter_only
"""

using HTTP
using JSON3
using Dates
using Printf

const SERVER_DIR = dirname(abspath(@__FILE__))
include(joinpath(SERVER_DIR, "iitm_physics.jl"))

# ════════════════════════════════════════════════════════════
# §1  服务配置
# ════════════════════════════════════════════════════════════

mutable struct ServerConfig
    port     :: Int
    root_dir :: String
    log_dir  :: String
end

const SCONF = ServerConfig(2700, "", "")

const DEFAULT_CONFIG = Dict{String, Any}(
    "L_size"          => 20.0,
    "grid_dim"        => 80,
    "wavelength_um"   => 1.55,
    "m_real"          => 1.311,
    "m_imag"          => 1e-4,
    "shape_type"      => "cylinder",
    "r_eff"           => 0.0,
    "axis_ratio"      => 1.0,
    "nmax_override"   => 0,
    "r_bottom"        => 0.5,
    "r_top"           => 2.0,
    "sigma_ln"        => 0.35,
    "n_radii"         => 8,
    "cloud_center_z"  => 10.0,
    "cloud_thickness" => 8.0,
    "turbulence_scale"=> 4.0,
    "photons"         => 10000,
    "visibility_km"   => 3.0,
    "scale_height_m"  => 2000.0,
    "angstrom_q"      => 1.3,
    "mc_use_3d_density" => true,
    "mc_density_sampling" => "nearest",
    "Nr"              => 40,
    "Ntheta"          => 60,
    "tmatrix_solver"  => "auto",
    "forward_mode"    => "cone_avg",
    "forward_cone_deg" => 0.5,
    "ebcm_threshold"  => 1e-4,
    "ebcm_ndgs"       => 4,
    "ebcm_maxiter"    => 20,
    "ebcm_loss_threshold" => 10.0,
    "iitm_nr_scale_on_fallback" => 1.25,
    "iitm_ntheta_scale_on_fallback" => 1.5,
    "field_compute_mode" => "proxy_only",
    "lidar_enabled" => false,
    "range_bin_width_m" => 1.0,
    "range_max_m" => 0.0,
    "receiver_overlap_min" => 1.0,
    "receiver_overlap_full_range_m" => 0.0,
    "explode_dist"    => 0.7,
)

function build_field_metadata(config::Dict, bundle::Dict)
    catalog = build_field_catalog(bundle)
    family_order = [fam for fam in ("proxy", "exact") if haskey(catalog, fam)]
    meta = Dict{String, Any}(
        "field_catalog" => catalog,
        "available_field_families" => family_order,
        "requested_field_compute_mode" => get(bundle, "requested_field_compute_mode",
                                               String(get(config, "field_compute_mode", "proxy_only"))),
        "effective_field_compute_mode" => get(bundle, "effective_field_compute_mode", "proxy_only"),
        "primary_field_family" => get(bundle, "primary_field_family", "proxy"),
        "lidar_observation_available" => haskey(bundle, "lidar_observation"),
    )
    if haskey(bundle, "field_mode_note")
        meta["field_mode_note"] = bundle["field_mode_note"]
    end
    return meta
end

# ════════════════════════════════════════════════════════════
# §2  日志（线程安全）
# ════════════════════════════════════════════════════════════

const _LOG_IO   = Ref{Union{IOStream, Nothing}}(nothing)
const _LOG_LOCK = ReentrantLock()

function slog(msg::String, level::Symbol = :info)
    ts   = Dates.format(now(), "HH:MM:SS.sss")
    line = "[$ts][$level] $msg"
    println(line); flush(stdout)
    lock(_LOG_LOCK) do
        if _LOG_IO[] !== nothing
            println(_LOG_IO[], line); flush(_LOG_IO[])
        end
    end
end

function open_run_log(proj::String)
    mkpath(SCONF.log_dir)
    ts   = Dates.format(now(), "yyyymmdd_HHMMSSsss")
    path = joinpath(SCONF.log_dir, "iitm_$(proj)_$(ts).log")
    lock(_LOG_LOCK) do
        _LOG_IO[] = open(path, "w")
    end
    return path
end

function close_run_log()
    lock(_LOG_LOCK) do
        if _LOG_IO[] !== nothing
            close(_LOG_IO[]); _LOG_IO[] = nothing
        end
    end
end

# ════════════════════════════════════════════════════════════
# §3  配置组装（v3 重构）
# ════════════════════════════════════════════════════════════

"""
构建 scatter_cfg 和 mc_cfg。

代表粒径策略（v3）：
  优先级：用户显式 r_eff  >  nmax 安全上限推导  >  几何均值
  安全 nmax 上限 nmax_safe=20 对应的最大粒径 r_safe = (nmax_safe-2)/1.5 × λ/(2π)
  保证无论用户传入多大的 r_top，计算用的 r_repr 始终让 nmax ≤ nmax_safe，
  从而将 Step2 耗时控制在可接受范围内。

r_repr 同时写入 scatter_cfg["r_repr"]，供 compute_scatter_fields v3 读取。
"""
function build_configs(config::Dict)
    wl_m  = Float64(get(config, "wavelength_um", 1.55)) * 1e-6
    wl_um = wl_m * 1e6

    r_bottom      = Float64(get(config, "r_bottom",      0.5))
    r_top         = Float64(get(config, "r_top",         2.0))
    sigma_ln      = Float64(get(config, "sigma_ln",      0.35))
    r_eff_user    = Float64(get(config, "r_eff",         0.0))
    nmax_override = Int(    get(config, "nmax_override",  0))

    # ── 安全 nmax 上限推导代表粒径 ────────────────────────
    nmax_safe = nmax_override > 0 ? nmax_override : 20
    x_safe    = Float64(nmax_safe - 2) / 1.5
    r_safe    = x_safe * wl_um / (2pi)   # [μm]

    r_geomean = sqrt(max(r_bottom, 1e-6) * max(r_top, 1e-6))

    if r_eff_user > 0.0
        r_repr = r_eff_user              # 用户显式指定，不限制
    else
        r_repr = min(r_geomean, r_safe)  # 不超过安全上限
    end

    scatter_cfg = Dict{String, Any}(
        "wavelength_m"     => wl_m,
        "m_real"           => Float64(get(config, "m_real",      1.311)),
        "m_imag"           => Float64(get(config, "m_imag",      1e-4)),
        "shape_type"       => String( get(config, "shape_type",  "cylinder")),
        "r_eff"            => r_eff_user,
        "axis_ratio"       => Float64(get(config, "axis_ratio",  1.0)),
        "nmax_override"    => nmax_override,
        "size_mode"        => "lognormal",
        "radius_um"        => r_repr,
        "median_radius_um" => r_repr,
        "sigma_ln"         => sigma_ln,
        "n_radii"          => Int(get(config, "n_radii", 8)),
        "Nr"               => Int(get(config, "Nr",      40)),
        "Ntheta"           => Int(get(config, "Ntheta",  60)),
        "tmatrix_solver"   => String(get(config, "tmatrix_solver", "auto")),
        "forward_mode"     => String(get(config, "forward_mode", "cone_avg")),
        "forward_cone_deg" => Float64(get(config, "forward_cone_deg", 0.5)),
        "ebcm_threshold"   => Float64(get(config, "ebcm_threshold", 1e-4)),
        "ebcm_ndgs"        => Int(get(config, "ebcm_ndgs", 4)),
        "ebcm_maxiter"     => Int(get(config, "ebcm_maxiter", 20)),
        "ebcm_full"        => Bool(get(config, "ebcm_full", false)),
        "ebcm_loss_guard"  => Bool(get(config, "ebcm_loss_guard", true)),
        "ebcm_loss_threshold" => Float64(get(config, "ebcm_loss_threshold", 10.0)),
        "fallback_on_anomaly" => Bool(get(config, "fallback_on_anomaly", true)),
        "iitm_nr_scale_on_fallback" =>
            Float64(get(config, "iitm_nr_scale_on_fallback", 1.25)),
        "iitm_ntheta_scale_on_fallback" =>
            Float64(get(config, "iitm_ntheta_scale_on_fallback", 1.5)),
        # ── 供 compute_scatter_fields v3 读取 ───────────────
        "r_repr"           => r_repr,
    )

    vis_km = Float64(get(config, "visibility_km", 3.0))
    aq     = Float64(get(config, "angstrom_q",    1.3))
    beta_s = visibility_to_beta_ext(vis_km, wl_m * 1e9; angstrom_q = aq)

    mc_cfg = Dict{String, Any}(
        "beta_ext_surf"  => beta_s,
        "thickness_m"    => Float64(get(config, "L_size",         20.0)),
        "scale_height_m" => Float64(get(config, "scale_height_m", 2000.0)),
        "n_photons"      => Int(    get(config, "photons",        10000)),
        "field_compute_mode" => String(get(config, "field_compute_mode", "proxy_only")),
        "use_3d_density" => Bool(   get(config, "mc_use_3d_density", true)),
        "density_sampling" => String(get(config, "mc_density_sampling", "nearest")),
        "field_forward_half_angle_deg" =>
            Float64(get(config, "field_forward_half_angle_deg", 90.0)),
        "field_back_half_angle_deg" =>
            Float64(get(config, "field_back_half_angle_deg", 90.0)),
        "field_quadrature_polar" =>
            Int(get(config, "field_quadrature_polar", 2)),
        "field_quadrature_azimuth" =>
            Int(get(config, "field_quadrature_azimuth", 6)),
        "collect_lidar_observation" =>
            Bool(get(config, "lidar_enabled", false)),
        "range_bin_width_m" =>
            Float64(get(config, "range_bin_width_m", 1.0)),
        "range_max_m" =>
            Float64(get(config, "range_max_m", get(config, "L_size", 20.0))),
        "receiver_overlap_min" =>
            Float64(get(config, "receiver_overlap_min", 1.0)),
        "receiver_overlap_full_range_m" =>
            Float64(get(config, "receiver_overlap_full_range_m", 0.0)),
    )

    return scatter_cfg, mc_cfg, beta_s
end

# ════════════════════════════════════════════════════════════
# §4  核心仿真步骤（v3 更新）
# ════════════════════════════════════════════════════════════

function step1_field(config::Dict, log_fn::Function)
    log_fn(">> [1/4] 生成 3D 密度场 (grid=$(get(config,"grid_dim",80))³)...")
    t = time()
    field = generate_field(config)
    log_fn(@sprintf(">> [1/4] 完成，耗时 %.2fs", time() - t))
    return field
end

"""
step2_scatter v3：日志中显示 r_repr 和预估 nmax，便于用户判断耗时。
"""
function step2_scatter(config::Dict, scatter_cfg::Dict, log_fn::Function)
    st     = get(scatter_cfg, "shape_type",       "cylinder")
    ar     = get(scatter_cfg, "axis_ratio",       1.0)
    nm     = get(scatter_cfg, "nmax_override",    0)
    r_rep  = get(scatter_cfg, "median_radius_um", 0.0)
    wl_um  = get(scatter_cfg, "wavelength_m",     1.55e-6) * 1e6
    x_rep  = 2pi * r_rep / wl_um
    nmax_e = nm > 0 ? nm : clamp(round(Int, x_rep + 4*x_rep^(1/3) + 2), 4, 60)
    solver = get(scatter_cfg, "tmatrix_solver", "auto")

    log_fn(@sprintf(
        ">> [2/4] T-Matrix 散射参数 [%s, solver=%s, ar=%.2f, r_repr=%.3fμm, x=%.1f, nmax_est=%d]...",
        st, solver, ar, r_rep, x_rep, nmax_e))
    t  = time()
    sc = compute_scatter_params(scatter_cfg)
    # 把 r_repr 透传进 scatter dict，供 compute_scatter_fields v3 使用
    sc["r_repr"] = r_rep
    solver_used = get(sc, "solver_used", "unknown")
    fallback_msg = Bool(get(sc, "fallback_used", false)) ?
                   " | fallback=" * String(get(sc, "fallback_reason", "")) : ""
    log_fn(@sprintf(">> [2/4] 完成 %.2fs | solver=%s | Csca=%.3e m² ω₀=%.4f g=%.4f | F/B=%.3e | depol_b=%.4f depol_f=%.4f%s",
                    time()-t, solver_used, sc["sigma_sca"], sc["omega0"], sc["g"],
                    sc["forward_back_ratio"], sc["depol_back"], sc["depol_forward"],
                    fallback_msg))
    return sc
end

function step3_mc(field::Dict, scatter::Dict, mc_cfg::Dict, log_fn::Function)
    mc_cfg_local = copy(mc_cfg)
    use_3d = Bool(get(mc_cfg_local, "use_3d_density", true))
    density_mode = String(get(mc_cfg_local, "density_sampling", "nearest"))
    requested_mode = normalize_field_compute_mode(get(mc_cfg_local, "field_compute_mode", "proxy_only"))
    if use_3d
        mc_cfg_local["density_grid"] = field["density_norm"]
        mc_cfg_local["field_axis"] = field["axis"]
        mc_cfg_local["field_xy_centered"] = true
    end
    collect_lidar = Bool(get(mc_cfg_local, "collect_lidar_observation", false))
    mc_cfg_local["collect_voxel_fields"] = requested_mode != "proxy_only" || collect_lidar
    mc_cfg_local["field_forward_half_angle_deg"] = Float64(get(mc_cfg, "field_forward_half_angle_deg", 90.0))
    mc_cfg_local["field_back_half_angle_deg"] = Float64(get(mc_cfg, "field_back_half_angle_deg", 90.0))
    mc_cfg_local["field_quadrature_polar"] = Int(get(mc_cfg, "field_quadrature_polar", 2))
    mc_cfg_local["field_quadrature_azimuth"] = Int(get(mc_cfg, "field_quadrature_azimuth", 6))
    mode_msg = use_3d ? "3D $(density_mode) + z_slab" : "1D profile"
    field_msg = requested_mode == "proxy_only" ?
        (collect_lidar ? "proxy + lidar observation" : "proxy only") :
        (collect_lidar ? "collect exact field + lidar observation" : "collect exact field")
    log_fn(@sprintf(">> [3/4] Monte Carlo (%d 光子, %s, %s)...",
                    mc_cfg["n_photons"], mode_msg, field_msg))
    t  = time()
    mc = run_monte_carlo(scatter, mc_cfg_local)
    log_fn(@sprintf(">> [3/4] 完成 %.2fs | R_back=%.5f R_trans=%.5f depol=%.4f",
                    time()-t, mc.backscatter_ratio, mc.transmit_ratio,
                    mc.depolarization_ratio))
    return mc
end

"""
step4_render v3：
  - 接受 scatter 参数（Step2 结果），传给 compute_scatter_fields v3
  - compute_scatter_fields v3 复用 scatter 中的截面数据，不重跑 IITM
  - render_to_html v3 保存 density.npz + 生成轻量 HTML 模板
"""
function step4_render(field::Dict, scatter::Dict, mc::MCStats, config::Dict,
                       output_dir::String, log_fn::Function)
    requested_mode = normalize_field_compute_mode(get(config, "field_compute_mode", "proxy_only"))
    log_fn(">> [4/4] 计算场族并保存数据文件... requested=$(requested_mode)")
    t = time()

    field_bundle = build_field_bundle(field, scatter, config, mc)
    effective_mode = String(get(field_bundle, "effective_field_compute_mode", "proxy_only"))
    log_fn(@sprintf(">> [4/4] 场族生成完成 %.2fs | effective=%s | 写入 NPZ + 生成 HTML...",
                    time() - t, effective_mode))

    files = render_to_html(field, field_bundle, scatter, config, output_dir)
    log_fn(@sprintf(">> [4/4] 完成 %.2fs | NPZ + HTML: %s",
                    time() - t, join(files, ", ")))
    return files, field_bundle
end

# ════════════════════════════════════════════════════════════
# §5  组装最终结果 Dict
# ════════════════════════════════════════════════════════════

function integrate_lobe(theta_r::Vector{Float64}, values::Vector{Float64}, side::Symbol)
    boundary = pi / 2
    split_idx = findfirst(theta -> theta >= boundary, theta_r)

    if split_idx === nothing
        return side === :front && length(theta_r) >= 2 ?
               trapz(values .* sin.(theta_r), theta_r) : 0.0
    elseif split_idx == 1
        return side === :back && length(theta_r) >= 2 ?
               trapz(values .* sin.(theta_r), theta_r) : 0.0
    end

    if isapprox(theta_r[split_idx], boundary; atol = 1e-12)
        front_theta = theta_r[1:split_idx]
        front_vals  = values[1:split_idx]
        back_theta  = theta_r[split_idx:end]
        back_vals   = values[split_idx:end]
    else
        θ0, θ1 = theta_r[split_idx - 1], theta_r[split_idx]
        v0, v1 = values[split_idx - 1], values[split_idx]
        w = (boundary - θ0) / (θ1 - θ0)
        v_boundary = v0 + w * (v1 - v0)

        front_theta = vcat(theta_r[1:split_idx - 1], boundary)
        front_vals  = vcat(values[1:split_idx - 1], v_boundary)
        back_theta  = vcat(boundary, theta_r[split_idx:end])
        back_vals   = vcat(v_boundary, values[split_idx:end])
    end

    theta_sel = side === :front ? front_theta : back_theta
    value_sel = side === :front ? front_vals  : back_vals
    return length(theta_sel) >= 2 ? trapz(value_sel .* sin.(theta_sel), theta_sel) : 0.0
end

function build_result(scatter::Dict, mc::MCStats, beta_surf::Float64,
                       html_files::Vector{String}, duration::Float64)
    M11_arr = scatter["M11"];  theta_r = deg2rad.(scatter["angles_deg"])
    iback   = integrate_lobe(theta_r, M11_arr, :back)
    ifwd    = integrate_lobe(theta_r, M11_arr, :front)

    result = Dict{String, Any}(
        "status"   => "success",
        "metrics"  => Dict{String, Any}(
            "duration_sec"   => duration,
            "beta_ext_surf"  => beta_surf,
            "omega0"         => scatter["omega0"],
            "g"              => scatter["g"],
            "R_back"         => mc.backscatter_ratio,
            "R_trans"        => mc.transmit_ratio,
            "R_abs"          => mc.absorbed_ratio,
            "depol_ratio"    => mc.depolarization_ratio,
            "avg_collisions" => mc.avg_collisions,
            "mie_int_back"   => iback,
            "mie_int_fwd"    => ifwd,
            "sigma_back_ref"  => get(scatter, "sigma_back_ref", 0.0),
            "sigma_forward_ref" => get(scatter, "sigma_forward_ref", 0.0),
            "forward_back_ratio" => get(scatter, "forward_back_ratio", 0.0),
            "depol_back"     => get(scatter, "depol_back", 0.0),
            "depol_forward"  => get(scatter, "depol_forward", 0.0),
            "forward_mode"   => get(scatter, "forward_mode", "cone_avg"),
            "forward_cone_deg" => get(scatter, "forward_cone_deg", 0.5),
            "solver_requested" => get(scatter, "solver_requested", "auto"),
            "solver_used"      => get(scatter, "solver_used", "unknown"),
            "fallback_used"    => get(scatter, "fallback_used", false),
            "fallback_count"   => get(scatter, "fallback_count", 0),
            "fallback_reason"  => get(scatter, "fallback_reason", ""),
            "solver_path_summary" => get(scatter, "solver_path_summary", ""),
            "ebcm_count"       => get(scatter, "ebcm_count", 0),
            "iitm_count"       => get(scatter, "iitm_count", 0),
            "nmax_min"         => get(scatter, "nmax_min", 0),
            "nmax_max"         => get(scatter, "nmax_max", 0),
            "ebcm_loss_estimate_max" =>
                get(scatter, "ebcm_loss_estimate_max", nothing),
            "ebcm_loss_estimate_mean" =>
                get(scatter, "ebcm_loss_estimate_mean", nothing),
            "ebcm_threshold_used" => get(scatter, "ebcm_threshold_used", nothing),
            "ebcm_ndgs"        => get(scatter, "ebcm_ndgs", 0),
            "ebcm_maxiter"     => get(scatter, "ebcm_maxiter", 0),
            "iitm_nr"          => get(scatter, "iitm_nr", 0),
            "iitm_ntheta"      => get(scatter, "iitm_ntheta", 0),
            "iitm_nr_fallback" => get(scatter, "iitm_nr_fallback", 0),
            "iitm_ntheta_fallback" => get(scatter, "iitm_ntheta_fallback", 0),
        ),
        "arrays"   => Dict{String, Any}(
            "mie_M11_profile" => M11_arr[1:10:end],
            "mc_back_dist"    => mc.backscatter_angle_dist,
        ),
        "artifacts" => html_files,
    )
    return result
end

# ════════════════════════════════════════════════════════════
# §6  HTTP 工具
# ════════════════════════════════════════════════════════════

function parse_body(req::HTTP.Request)
    body = String(req.body)
    isempty(body) && error("空请求体")
    return JSON3.read(body, Dict{String, Any})
end

function json_resp(data; status::Int = 200)
    return HTTP.Response(status,
        ["Content-Type"                => "application/json",
         "Access-Control-Allow-Origin" => "*"],
        JSON3.write(data))
end

function write_plain_response(http::HTTP.Stream, response::HTTP.Response)
    request = http.message
    request.response = response
    response.request = request
    HTTP.startwrite(http)
    write(http, response.body)
    return
end

function write_sse_event(http::HTTP.Stream, event::Symbol, data::AbstractString)
    payload = replace(data, "\r\n" => "\n")
    write(http, "event: $(String(event))\n")
    for line in split(payload, '\n'; keepempty = true)
        write(http, "data: $line\n")
    end
    write(http, "\n")
    flush(http.stream)
    return
end

# ════════════════════════════════════════════════════════════
# §7  路由处理器
# ════════════════════════════════════════════════════════════

"""GET /health"""
function handle_health(req::HTTP.Request)
    return json_resp(Dict(
        "status"  => "ok",
        "threads" => Threads.nthreads(),
        "time"    => string(now()),
        "backend" => "IITM Julia v3",
    ))
end

"""POST /simulate — 同步等待，返回完整 JSON"""
function handle_simulate(req::HTTP.Request)
    try
        body   = parse_body(req)
        proj   = get(body, "project_name", "run_$(round(Int,time()))")
        config = merge(DEFAULT_CONFIG, get(body, "config", Dict{String,Any}()))
        config["grid_dim"] = Int(get(config, "grid_dim", 80))
        config["photons"]  = Int(get(config, "photons",  10000))

        output_dir = joinpath(SCONF.root_dir, "outputs", "iitm", proj)
        mkpath(output_dir)
        open_run_log(proj)
        slog("== /simulate 开始 | 工程: $proj | 线程: $(Threads.nthreads())")

        t0 = time()
        scatter_cfg, mc_cfg, beta_s = build_configs(config)

        field = step1_field(config, slog)
        sc    = step2_scatter(config, scatter_cfg, slog)
        mc    = step3_mc(field, sc, mc_cfg, slog)
        files, field_bundle = step4_render(field, sc, mc, config, output_dir, slog)

        dur = time() - t0
        slog(@sprintf("== 完成，总耗时 %.2fs", dur))
        close_run_log()

        result = build_result(sc, mc, beta_s, files, dur)
        merge!(result, build_field_metadata(config, field_bundle))
        return json_resp(result)

    catch e
        tb = sprint(showerror, e, catch_backtrace())
        slog(">> /simulate 错误: $tb", :error)
        close_run_log()
        return json_resp(Dict("status"=>"error","error"=>string(e)); status=500)
    end
end

"""
POST /simulate/stream — SSE 推送（推荐）

事件：
  event: log    data: <进度行>
  event: result data: <最终 JSON>
  event: error  data: <错误信息>

Python 侧 iitm_http_worker.py 通过 httpx.stream 读取，
将 log 行实时 print 到 stdout，result 行作为 IPC 载荷。
"""
function handle_simulate_stream(http::HTTP.Stream)
    req = http.message
    req.body = read(http)
    HTTP.closeread(http)

    HTTP.setstatus(http, 200)
    HTTP.setheader(http, "Content-Type"                => "text/event-stream")
    HTTP.setheader(http, "Cache-Control"               => "no-cache")
    HTTP.setheader(http, "Access-Control-Allow-Origin" => "*")
    HTTP.startwrite(http)

    function log_fn(msg::String)
        write_sse_event(http, :log, msg)
        slog(msg)
    end

    try
        body   = parse_body(req)
        proj   = get(body, "project_name", "stream_$(round(Int,time()))")
        config = merge(DEFAULT_CONFIG, get(body, "config", Dict{String,Any}()))
        config["grid_dim"] = Int(get(config, "grid_dim", 80))
        config["photons"]  = Int(get(config, "photons",  10000))

        output_dir = joinpath(SCONF.root_dir, "outputs", "iitm", proj)
        mkpath(output_dir)
        open_run_log(proj)
        log_fn("== 流式仿真开始 | 工程: $proj | 线程: $(Threads.nthreads())")

        t0 = time()
        scatter_cfg, mc_cfg, beta_s = build_configs(config)

        field = step1_field(config, log_fn)
        sc    = step2_scatter(config, scatter_cfg, log_fn)
        mc    = step3_mc(field, sc, mc_cfg, log_fn)
        files, field_bundle = step4_render(field, sc, mc, config, output_dir, log_fn)

        dur = time() - t0
        log_fn(@sprintf("== 完成，总耗时 %.2fs ==", dur))
        close_run_log()

        result = build_result(sc, mc, beta_s, files, dur)
        merge!(result, build_field_metadata(config, field_bundle))
        write_sse_event(http, :result, JSON3.write(result))
    catch e
        tb = sprint(showerror, e, catch_backtrace())
        slog(">> stream 错误: $tb", :error)
        close_run_log()
        write_sse_event(http, :error, string(e))
    finally
        HTTP.closewrite(http)
    end
    return
end

"""POST /scatter_only — 仅计算散射参数（调试）"""
function handle_scatter_only(req::HTTP.Request)
    try
        body   = parse_body(req)
        config = merge(DEFAULT_CONFIG, get(body, "config", Dict{String,Any}()))
        sc_cfg, _, _ = build_configs(config)
        sc = compute_scatter_params(sc_cfg)
        r_rep = get(sc_cfg, "median_radius_um", 0.0)
        wl_um = get(sc_cfg, "wavelength_m", 1.55e-6) * 1e6
        x_rep = 2pi * r_rep / wl_um
        nm    = Int(get(sc_cfg, "nmax_override", 0))
        return json_resp(Dict(
            "status"     => "ok",
            "sigma_ext"  => sc["sigma_ext"],
            "sigma_sca"  => sc["sigma_sca"],
            "omega0"     => sc["omega0"],
            "g"          => sc["g"],
            "shape_type" => get(sc_cfg, "shape_type", "?"),
            "axis_ratio" => get(sc_cfg, "axis_ratio",  1.0),
            "r_repr"     => r_rep,
            "x_repr"     => x_rep,
            "nmax_est"   => nm > 0 ? nm :
                            clamp(round(Int, x_rep + 4*x_rep^(1/3) + 2), 4, 60),
        ))
    catch e
        return json_resp(Dict("status"=>"error","error"=>string(e)); status=500)
    end
end

"""OPTIONS 预检"""
function handle_options(req::HTTP.Request)
    return HTTP.Response(204, [
        "Access-Control-Allow-Origin"  => "*",
        "Access-Control-Allow-Methods" => "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers" => "Content-Type",
    ])
end

# ════════════════════════════════════════════════════════════
# §8  路由 & 启动
# ════════════════════════════════════════════════════════════

const ROUTER = HTTP.Router()
HTTP.register!(ROUTER, "GET",     "/health",          handle_health)
HTTP.register!(ROUTER, "POST",    "/simulate",        handle_simulate)
HTTP.register!(ROUTER, "POST",    "/simulate/stream", handle_simulate_stream)
HTTP.register!(ROUTER, "POST",    "/scatter_only",    handle_scatter_only)
HTTP.register!(ROUTER, "OPTIONS", "/*",               handle_options)

function parse_server_args()
    port     = 2700
    root_dir = dirname(dirname(dirname(abspath(@__FILE__))))
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--port" && i < length(ARGS)
            port = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--root" && i < length(ARGS)
            root_dir = ARGS[i+1]; i += 2
        else
            i += 1
        end
    end
    return port, root_dir
end

function main()
    port, root_dir = parse_server_args()
    SCONF.port     = port
    SCONF.root_dir = root_dir
    SCONF.log_dir  = joinpath(root_dir, "log")
    mkpath(SCONF.log_dir)

    slog("=" ^ 60)
    slog("IITM Julia HTTP Server  v3.0")
    slog("  Port    : $port")
    slog("  Root    : $root_dir")
    slog("  Threads : $(Threads.nthreads())")
    slog("  Julia   : $(VERSION)")
    slog("  渲染模式 : NPZ + 浏览器端 Plotly volume")
    slog("  支持形状 : cylinder | spheroid | sphere")
    slog("=" ^ 60)
    slog("服务就绪，等待请求...")

    HTTP.serve("127.0.0.1", port; verbose = false, stream = true) do http::HTTP.Stream
        req = http.message
        path = split(String(req.target), '?'; limit = 2)[1]

        if req.method == "POST" && path == "/simulate/stream"
            handle_simulate_stream(http)
        else
            req.body = read(http)
            HTTP.closeread(http)
            write_plain_response(http, ROUTER(req))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
