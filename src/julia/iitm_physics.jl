"""
iitm_physics.jl — IITM 独立物理计算核心  v3.0
变更（v3）：
  §7  compute_scatter_fields 重构：接受 scatter Dict，复用 Step2 结果，不重跑 IITM
  §8  渲染层重构：改为方案 D（NPZ 数据文件 + 浏览器端 JS 渲染）
      save_field_npz()   — 保存二进制数据文件（<5MB，替代 50MB JSON HTML）
      render_to_html()   — 生成轻量 HTML 模板，浏览器端 fetch npz 后渲染

直接运行测试：
  julia --threads auto --project=src/julia iitm_physics.jl
"""

using TransitionMatrices
using LinearAlgebra
using Random
using Statistics
using NPZ
using JSON3

# ═══════════════════════════════════════════════════════════
# §0  常量
# ═══════════════════════════════════════════════════════════

const C_LIGHT   = 299_792_458.0
const UM2_TO_M2 = 1e-12   # μm² → m²
const _EBCM_LOSS_MODULE = Ref{Any}(nothing)
const _EBCM_LOSS_MODULE_ATTEMPTED = Ref(false)

# ═══════════════════════════════════════════════════════════
# §1  基础数学工具
# ═══════════════════════════════════════════════════════════

"""Koschmieder + Angström 修正：能见度 → 消光系数 [m⁻¹]"""
function visibility_to_beta_ext(visibility_km::Float64, wavelength_nm::Float64;
                                 angstrom_q::Float64 = 1.3)
    beta550 = 3.912 / (visibility_km * 1000.0)
    wavelength_nm <= 0.0 && return beta550
    return beta550 * (550.0 / wavelength_nm)^angstrom_q
end

"""梯形积分"""
function trapz(y::AbstractVector{T}, x::AbstractVector{T}) where {T<:Real}
    length(y) == length(x) || error("trapz: 长度不匹配")
    s = zero(T)
    @inbounds for i in 2:length(x)
        s += (y[i] + y[i-1]) * (x[i] - x[i-1])
    end
    return s / 2
end

"""累积梯形积分（首元素为 0）"""
function cumtrapz(y::AbstractVector{T}, x::AbstractVector{T}) where {T<:Real}
    out = zeros(T, length(y))
    @inbounds for i in 2:length(y)
        out[i] = out[i-1] + (y[i] + y[i-1]) * (x[i] - x[i-1]) / 2
    end
    return out
end

"""对数正态 PDF"""
function lognormal_pdf(r::AbstractVector{T}, r_med::T, sigma_ln::T) where {T<:AbstractFloat}
    pdf = zeros(T, length(r))
    sigma_ln <= 1e-6 && return pdf
    mu = log(r_med)
    @inbounds for i in eachindex(r)
        r[i] > 0 || continue
        pdf[i] = exp(-0.5 * ((log(r[i]) - mu) / sigma_ln)^2) /
                 (r[i] * sigma_ln * sqrt(2pi))
    end
    return pdf
end

# ═══════════════════════════════════════════════════════════
# §2  角度网格 & 相函数 CDF
# ═══════════════════════════════════════════════════════════

"""前向加密散射角网格 [度]"""
function generate_adaptive_angles(; num_total::Int = 600,
                                    forward_res::Float64 = 0.01,
                                    forward_max::Float64 = 2.0)
    fwd  = collect(0.0:forward_res:forward_max)
    !isempty(fwd) && fwd[end] > forward_max && pop!(fwd)
    rest = collect(range(forward_max, 180.0; length = max(2, num_total - length(fwd))))
    return unique(sort(vcat(fwd, rest[2:end])))
end

"""M11 → 散射角 CDF，返回 (theta_rad_grid, cdf)"""
function build_phase_cdf(angles_deg::Vector{Float64}, M11::Vector{Float64})
    theta_rad = deg2rad.(angles_deg)
    pdf       = max.(M11, 0.0) .* sin.(theta_rad)
    cdf       = cumtrapz(pdf, theta_rad)
    cdf[end] > 0.0 && (cdf ./= cdf[end])
    return theta_rad, cdf
end

"""逆 CDF 采样单散射角 [rad]"""
@inline function sample_scattering_theta(rng::AbstractRNG,
                                          theta_grid::Vector{Float64},
                                          cdf::Vector{Float64})
    u   = rand(rng)
    idx = searchsortedfirst(cdf, u)
    idx == 1 && return theta_grid[1]
    idx > length(theta_grid) && return theta_grid[end]
    dy  = cdf[idx] - cdf[idx-1]
    dy < 1e-10 && return theta_grid[idx-1]
    return theta_grid[idx-1] + (u - cdf[idx-1]) / dy * (theta_grid[idx] - theta_grid[idx-1])
end

# ═══════════════════════════════════════════════════════════
# §3  粒子形状工厂 & nmax 估算
# ═══════════════════════════════════════════════════════════

"""
Wiscombe 准则估算截断阶数。
nmax_override > 0 时直接使用手动值。
"""
function resolve_nmax(r_um::Float64, wavelength_um::Float64;
                      nmax_override::Int = 0, nmax_cap::Int = 60)
    nmax_override > 0 && return clamp(nmax_override, 1, nmax_cap)
    x    = 2pi * r_um / wavelength_um
    auto = round(Int, x + 4 * x^(1/3) + 2)
    return clamp(max(4, auto), 1, nmax_cap)
end

"""
粒子形状工厂。
shape_type : "cylinder" | "spheroid" | "sphere"
axis_ratio : cylinder→h/(2r)；spheroid→c/a；sphere→忽略
"""
function make_particle(shape_type::String,
                        radius_um::Float64,
                        m::ComplexF64;
                        r_eff::Float64      = 0.0,
                        axis_ratio::Float64 = 1.0)
    r  = r_eff > 0.0 ? r_eff : radius_um
    st = lowercase(strip(shape_type))
    if st == "cylinder"
        return Cylinder(r, 2.0 * r * axis_ratio, m)
    elseif st == "spheroid"
        return Spheroid(r, r * axis_ratio, m)
    elseif st == "sphere"
        return Spheroid(r, r, m)
    else
        @warn "未知形状 '$shape_type'，回落到 cylinder"
        return Cylinder(r, 2.0 * r * axis_ratio, m)
    end
end

function normalize_solver_choice(solver_choice)::String
    solver = lowercase(strip(String(solver_choice)))
    solver in ("auto", "ebcm_only", "iitm_only") ? solver : "auto"
end

function get_ebcm_loss_module()
    _EBCM_LOSS_MODULE_ATTEMPTED[] && return _EBCM_LOSS_MODULE[]
    _EBCM_LOSS_MODULE_ATTEMPTED[] = true

    try
        tm_root = normpath(joinpath(dirname(pathof(TransitionMatrices)), ".."))
        mod_path = joinpath(tm_root, "packages", "EBCMPrecisionLossEstimators", "src",
                            "EBCMPrecisionLossEstimators.jl")
        if !isfile(mod_path)
            return nothing
        end

        host = Module(:LocalEBCMPrecisionLossEstimatorsHost)
        Core.eval(host, :(using TransitionMatrices))
        Base.include(host, mod_path)
        _EBCM_LOSS_MODULE[] = getfield(host, :EBCMPrecisionLossEstimators)
    catch
        _EBCM_LOSS_MODULE[] = nothing
    end

    return _EBCM_LOSS_MODULE[]
end

function estimate_ebcm_total_loss_safe(shape, nmax::Int)
    mod = get_ebcm_loss_module()
    mod === nothing && return nothing
    try
        return Float64(mod.estimate_total_loss(shape, nmax))
    catch
        return nothing
    end
end

function build_solver_diagnostics(; solver_requested::String,
                                  solver_used::String,
                                  attempted_solvers::Vector{String},
                                  fallback_used::Bool,
                                  fallback_reason::String = "",
                                  nmax::Int,
                                  ebcm_threshold::Float64 = 1e-4,
                                  ebcm_ndgs::Int = 4,
                                  ebcm_maxiter::Int = 20,
                                  ebcm_loss_estimate = nothing,
                                  ebcm_loss_threshold::Float64 = 10.0,
                                  Nr::Int = 0,
                                  Ntheta::Int = 0,
                                  Nr_fallback::Int = 0,
                                  Ntheta_fallback::Int = 0)
    return Dict{String, Any}(
        "solver_requested"      => solver_requested,
        "solver_used"           => solver_used,
        "solver_path"           => join(attempted_solvers, " -> "),
        "fallback_used"         => fallback_used,
        "fallback_reason"       => fallback_reason,
        "nmax"                  => nmax,
        "ebcm_threshold"        => ebcm_threshold,
        "ebcm_ndgs"             => ebcm_ndgs,
        "ebcm_maxiter"          => ebcm_maxiter,
        "ebcm_loss_estimate"    => ebcm_loss_estimate,
        "ebcm_loss_threshold"   => ebcm_loss_threshold,
        "Nr"                    => Nr,
        "Ntheta"                => Ntheta,
        "Nr_fallback"           => Nr_fallback,
        "Ntheta_fallback"       => Ntheta_fallback,
    )
end

function validate_tm_result(sigma_ext::Float64, sigma_sca::Float64, g_val::Float64,
                            angles_deg::Vector{Float64}, M11::Vector{Float64},
                            M12::Vector{Float64}, M33::Vector{Float64},
                            M34::Vector{Float64})
    reasons = String[]

    if !isfinite(sigma_ext) || sigma_ext <= 0
        push!(reasons, "sigma_ext_invalid")
    end
    if !isfinite(sigma_sca) || sigma_sca < 0
        push!(reasons, "sigma_sca_invalid")
    end
    omega0 = sigma_ext > 0 ? sigma_sca / sigma_ext : NaN
    if !isfinite(omega0) || omega0 < -1e-8 || omega0 > 1.0 + 1e-6
        push!(reasons, "omega0_out_of_range")
    end
    if !isfinite(g_val) || abs(g_val) > 1.0 + 1e-6
        push!(reasons, "g_out_of_range")
    end

    for (name, arr) in [("M11", M11), ("M12", M12), ("M33", M33), ("M34", M34)]
        any(x -> !isfinite(x), arr) && push!(reasons, "$(name)_nonfinite")
    end

    if isempty(M11) || maximum(abs.(M11)) <= 1e-14
        push!(reasons, "M11_degenerate")
    else
        min_ratio = minimum(M11) / max(maximum(abs.(M11)), 1e-12)
        min_ratio < -1e-3 && push!(reasons, "M11_negative")
    end

    theta_r  = deg2rad.(angles_deg)
    norm_val = trapz(max.(M11, 0.0) .* sin.(theta_r), theta_r)
    if !isfinite(norm_val) || norm_val <= 1e-12
        push!(reasons, "phase_normalization_failed")
    end

    return isempty(reasons), join(unique(reasons), "; ")
end

function extract_scatter_fields(T_matrix, lambda_um::Float64, angles_deg::Vector{Float64})
    F = scattering_matrix(T_matrix, lambda_um, angles_deg)
    M11 = Float64.(real.(F[:, 1]))
    M12 = Float64.(real.(F[:, 2]))
    M33 = Float64.(real.(F[:, 4]))
    M34 = Float64.(real.(F[:, 5]))
    return M11, M12, M33, M34
end

function cone_average_metric(angles_deg::Vector{Float64}, values::Vector{Float64},
                             cone_deg::Float64)
    theta = deg2rad.(angles_deg)
    cone = deg2rad(max(cone_deg, 0.01))
    idx = findall(t -> t <= cone + 1e-12, theta)
    if length(idx) < 2
        return values[1]
    end
    theta_sel = theta[idx]
    value_sel = values[idx]
    denom = trapz(sin.(theta_sel), theta_sel)
    denom <= 1e-20 && return value_sel[1]
    return trapz(value_sel .* sin.(theta_sel), theta_sel) / denom
end

function safe_depol_ratio(M11::Float64, M12::Float64)
    denom = M11 + M12
    if !isfinite(denom) || abs(denom) <= 1e-20
        return 0.0
    end
    ratio = (M11 - M12) / denom
    if !isfinite(ratio)
        return 0.0
    end
    return clamp(ratio, 0.0, 1.0)
end

function finalize_single_particle_result(T_matrix, lambda_um::Float64, shape_type::String,
                                         solver_requested::String, solver_used::String,
                                         diagnostics::Dict{String, Any},
                                         angles_deg::Vector{Float64})
    Csca_um2 = scattering_cross_section(T_matrix, lambda_um)
    Cext_um2 = extinction_cross_section(T_matrix, lambda_um)
    g_val    = asymmetry_parameter(T_matrix, lambda_um)

    sigma_sca = Float64(real(Csca_um2)) * UM2_TO_M2
    sigma_ext = Float64(real(Cext_um2)) * UM2_TO_M2
    g_float   = Float64(real(g_val))

    M11, M12, M33, M34 = extract_scatter_fields(T_matrix, lambda_um, angles_deg)
    ok, reason = validate_tm_result(sigma_ext, sigma_sca, g_float, angles_deg, M11, M12,
                                    M33, M34)
    if !ok
        error("$(solver_used) result invalid: $reason")
    end

    return (sigma_ext  = sigma_ext,
            sigma_sca  = sigma_sca,
            g          = g_float,
            T_matrix   = T_matrix,
            lambda_um  = lambda_um,
            shape_name = shape_type,
            solver_used = solver_used,
            diagnostics = diagnostics,
            M11        = M11,
            M12        = M12,
            M33        = M33,
            M34        = M34)
end

# ═══════════════════════════════════════════════════════════
# §4  单粒子 T-Matrix 求解器
# ═══════════════════════════════════════════════════════════

function single_particle_iitm(radius_um::Float64,
                              m::ComplexF64,
                              wavelength_m::Float64;
                              shape_type::String    = "cylinder",
                              r_eff::Float64        = 0.0,
                              axis_ratio::Float64   = 1.0,
                              Nr::Int               = 50,
                              Ntheta::Int           = 80,
                              nmax_override::Int    = 0,
                              angles_deg::Vector{Float64} = generate_adaptive_angles(),
                              solver_requested::String = "iitm_only",
                              fallback_used::Bool   = false,
                              fallback_reason::String = "",
                              attempted_solvers::Vector{String} = ["iitm"],
                              Nr_fallback::Int      = 0,
                              Ntheta_fallback::Int  = 0)
    lambda_um = wavelength_m * 1e6
    r_calc    = r_eff > 0.0 ? r_eff : radius_um

    nmax     = resolve_nmax(r_calc, lambda_um; nmax_override = nmax_override)
    particle = make_particle(shape_type, radius_um, m;
                              r_eff = r_eff, axis_ratio = axis_ratio)

    T = transition_matrix_iitm(particle, lambda_um, nmax, Nr, Ntheta)
    diagnostics = build_solver_diagnostics(
        solver_requested = solver_requested,
        solver_used = "iitm",
        attempted_solvers = attempted_solvers,
        fallback_used = fallback_used,
        fallback_reason = fallback_reason,
        nmax = nmax,
        Nr = Nr,
        Ntheta = Ntheta,
        Nr_fallback = Nr_fallback,
        Ntheta_fallback = Ntheta_fallback,
    )

    return finalize_single_particle_result(T, lambda_um, shape_type, solver_requested, "iitm",
                                           diagnostics, angles_deg)
end

function single_particle_ebcm(radius_um::Float64,
                              m::ComplexF64,
                              wavelength_m::Float64;
                              shape_type::String    = "cylinder",
                              r_eff::Float64        = 0.0,
                              axis_ratio::Float64   = 1.0,
                              nmax_override::Int    = 0,
                              angles_deg::Vector{Float64} = generate_adaptive_angles(),
                              solver_requested::String = "ebcm_only",
                              ebcm_threshold::Float64 = 1e-4,
                              ebcm_ndgs::Int        = 4,
                              ebcm_maxiter::Int     = 20,
                              ebcm_full::Bool       = false,
                              ebcm_loss_estimate = nothing,
                              ebcm_loss_threshold::Float64 = 10.0)
    lambda_um = wavelength_m * 1e6
    r_calc    = r_eff > 0.0 ? r_eff : radius_um
    nmax      = resolve_nmax(r_calc, lambda_um; nmax_override = nmax_override)
    particle  = make_particle(shape_type, radius_um, m;
                              r_eff = r_eff, axis_ratio = axis_ratio)

    T = transition_matrix(
        particle,
        lambda_um;
        threshold = ebcm_threshold,
        ndgs = ebcm_ndgs,
        nₛₜₐᵣₜ = nmax,
        Ngₛₜₐᵣₜ = max(8, nmax * ebcm_ndgs),
        maxiter = ebcm_maxiter,
        full = ebcm_full,
    )

    diagnostics = build_solver_diagnostics(
        solver_requested = solver_requested,
        solver_used = "ebcm",
        attempted_solvers = ["ebcm"],
        fallback_used = false,
        nmax = length(T.𝐓) - 1,
        ebcm_threshold = ebcm_threshold,
        ebcm_ndgs = ebcm_ndgs,
        ebcm_maxiter = ebcm_maxiter,
        ebcm_loss_estimate = ebcm_loss_estimate,
        ebcm_loss_threshold = ebcm_loss_threshold,
    )

    return finalize_single_particle_result(T, lambda_um, shape_type, solver_requested, "ebcm",
                                           diagnostics, angles_deg)
end

function single_particle_tm_auto(radius_um::Float64,
                                 m::ComplexF64,
                                 wavelength_m::Float64;
                                 shape_type::String    = "cylinder",
                                 r_eff::Float64        = 0.0,
                                 axis_ratio::Float64   = 1.0,
                                 Nr::Int               = 50,
                                 Ntheta::Int           = 80,
                                 nmax_override::Int    = 0,
                                 angles_deg::Vector{Float64} = generate_adaptive_angles(),
                                 solver_requested::String = "auto",
                                 ebcm_threshold::Float64 = 1e-4,
                                 ebcm_ndgs::Int        = 4,
                                 ebcm_maxiter::Int     = 20,
                                 ebcm_full::Bool       = false,
                                 ebcm_loss_guard::Bool = true,
                                 ebcm_loss_threshold::Float64 = 10.0,
                                 fallback_on_anomaly::Bool = true,
                                 iitm_nr_scale_on_fallback::Float64 = 1.25,
                                 iitm_ntheta_scale_on_fallback::Float64 = 1.5)
    solver_requested = normalize_solver_choice(solver_requested)
    particle = make_particle(shape_type, radius_um, m;
                             r_eff = r_eff, axis_ratio = axis_ratio)
    lambda_um = wavelength_m * 1e6
    r_calc    = r_eff > 0.0 ? r_eff : radius_um
    nmax      = resolve_nmax(r_calc, lambda_um; nmax_override = nmax_override)
    loss_est  = estimate_ebcm_total_loss_safe(particle, nmax)
    path      = String[]

    if solver_requested == "iitm_only"
        return single_particle_iitm(
            radius_um, m, wavelength_m;
            shape_type = shape_type,
            r_eff = r_eff,
            axis_ratio = axis_ratio,
            Nr = Nr,
            Ntheta = Ntheta,
            nmax_override = nmax_override,
            angles_deg = angles_deg,
            solver_requested = solver_requested,
            attempted_solvers = ["iitm"],
        )
    end

    if solver_requested == "auto" && ebcm_loss_guard && !isnothing(loss_est) &&
       loss_est > ebcm_loss_threshold
        push!(path, "ebcm_guard")
        if fallback_on_anomaly
            Nr_fb = max(Nr + 2, ceil(Int, Nr * iitm_nr_scale_on_fallback))
            Ntheta_fb = max(Ntheta + 4, ceil(Int, Ntheta * iitm_ntheta_scale_on_fallback))
            return single_particle_iitm(
                radius_um, m, wavelength_m;
                shape_type = shape_type,
                r_eff = r_eff,
                axis_ratio = axis_ratio,
                Nr = Nr_fb,
                Ntheta = Ntheta_fb,
                nmax_override = nmax_override,
                angles_deg = angles_deg,
                solver_requested = solver_requested,
                fallback_used = true,
                fallback_reason = "predicted_ebcm_precision_loss",
                attempted_solvers = vcat(path, ["iitm"]),
                Nr_fallback = Nr_fb,
                Ntheta_fallback = Ntheta_fb,
            )
        end
        error("EBCM skipped by precision-loss guard")
    end

    try
        push!(path, "ebcm")
        return single_particle_ebcm(
            radius_um, m, wavelength_m;
            shape_type = shape_type,
            r_eff = r_eff,
            axis_ratio = axis_ratio,
            nmax_override = nmax_override,
            angles_deg = angles_deg,
            solver_requested = solver_requested,
            ebcm_threshold = ebcm_threshold,
            ebcm_ndgs = ebcm_ndgs,
            ebcm_maxiter = ebcm_maxiter,
            ebcm_full = ebcm_full,
            ebcm_loss_estimate = loss_est,
            ebcm_loss_threshold = ebcm_loss_threshold,
        )
    catch e
        if solver_requested == "ebcm_only" || !fallback_on_anomaly
            rethrow(e)
        end

        Nr_fb = max(Nr + 2, ceil(Int, Nr * iitm_nr_scale_on_fallback))
        Ntheta_fb = max(Ntheta + 4, ceil(Int, Ntheta * iitm_ntheta_scale_on_fallback))
        return single_particle_iitm(
            radius_um, m, wavelength_m;
            shape_type = shape_type,
            r_eff = r_eff,
            axis_ratio = axis_ratio,
            Nr = Nr_fb,
            Ntheta = Ntheta_fb,
            nmax_override = nmax_override,
            angles_deg = angles_deg,
            solver_requested = solver_requested,
            fallback_used = true,
            fallback_reason = sprint(showerror, e),
            attempted_solvers = vcat(path, ["iitm"]),
            Nr_fallback = Nr_fb,
            Ntheta_fallback = Ntheta_fb,
        )
    end
end

# ═══════════════════════════════════════════════════════════
# §5  有效散射参数（粒径分布加权积分）
# ═══════════════════════════════════════════════════════════

function compute_scatter_params(config::Dict)
    wavelength_m  = Float64(get(config, "wavelength_m",      1.55e-6))
    m_real        = Float64(get(config, "m_real",             1.311))
    m_imag        = Float64(get(config, "m_imag",             1e-4))
    shape_type    = String( get(config, "shape_type",         "cylinder"))
    size_mode     = String( get(config, "size_mode",          "mono"))
    radius_um     = Float64(get(config, "radius_um",          0.5))
    r_eff         = Float64(get(config, "r_eff",              0.0))
    median_r      = Float64(get(config, "median_radius_um",   radius_um))
    sigma_ln      = Float64(get(config, "sigma_ln",           0.35))
    n_radii       = Int(    get(config, "n_radii",            20))
    axis_ratio    = Float64(get(config, "axis_ratio",         1.0))
    Nr            = Int(    get(config, "Nr",                 50))
    Ntheta        = Int(    get(config, "Ntheta",             80))
    nmax_override = Int(    get(config, "nmax_override",      0))
    solver_requested = normalize_solver_choice(get(config, "tmatrix_solver", "auto"))
    forward_mode     = String(get(config, "forward_mode", "cone_avg"))
    forward_cone_deg = Float64(get(config, "forward_cone_deg", 0.5))
    ebcm_threshold   = Float64(get(config, "ebcm_threshold",  1e-4))
    ebcm_ndgs        = Int(    get(config, "ebcm_ndgs",       4))
    ebcm_maxiter     = Int(    get(config, "ebcm_maxiter",    20))
    ebcm_full        = Bool(   get(config, "ebcm_full",       false))
    ebcm_loss_guard  = Bool(   get(config, "ebcm_loss_guard", true))
    ebcm_loss_threshold = Float64(get(config, "ebcm_loss_threshold", 10.0))
    fallback_on_anomaly = Bool(get(config, "fallback_on_anomaly", true))
    iitm_nr_scale_on_fallback = Float64(get(config, "iitm_nr_scale_on_fallback", 1.25))
    iitm_ntheta_scale_on_fallback = Float64(get(config, "iitm_ntheta_scale_on_fallback",
                                               1.5))

    m_complex  = complex(m_real, m_imag)
    angles_deg = generate_adaptive_angles()
    Nang       = length(angles_deg)

    if size_mode == "lognormal"
        median_r <= 0.0 && (median_r = radius_um)
        log_r_min = log(median_r) - 4.0 * sigma_ln
        log_r_max = log(median_r) + 4.0 * sigma_ln
        r_grid    = exp.(range(log_r_min, log_r_max; length = n_radii))
        weights   = lognormal_pdf(r_grid, median_r, sigma_ln)
        tw = trapz(weights, r_grid)
        tw > 0 && (weights ./= tw)
    else
        r_mono = r_eff > 0.0 ? r_eff : radius_um
        r_grid  = [r_mono]
        weights = [1.0]
    end

    sigma_ext_eff = 0.0
    sigma_sca_eff = 0.0
    g_sum         = 0.0
    M11_sum = zeros(Float64, Nang)
    M12_sum = zeros(Float64, Nang)
    M33_sum = zeros(Float64, Nang)
    M34_sum = zeros(Float64, Nang)
    valid_radii = Float64[]
    valid_weights = Float64[]
    sigma_ext_vals = Float64[]
    sigma_sca_vals = Float64[]
    g_sigma_sca_vals = Float64[]
    m11_vals = Vector{Vector{Float64}}()
    m12_vals = Vector{Vector{Float64}}()
    m33_vals = Vector{Vector{Float64}}()
    m34_vals = Vector{Vector{Float64}}()
    valid_particles = 0
    solver_used_per_radius = String[]
    solver_paths = String[]
    fallback_reasons = String[]
    nmax_values = Int[]
    ebcm_loss_estimates = Float64[]
    nr_fallback_values = Int[]
    ntheta_fallback_values = Int[]

    for (r, w) in zip(r_grid, weights)
        w <= 0.0 && continue
        local res
        try
            res = single_particle_tm_auto(
                r, m_complex, wavelength_m;
                shape_type = shape_type,
                r_eff = r_eff,
                axis_ratio = axis_ratio,
                Nr = Nr,
                Ntheta = Ntheta,
                nmax_override = nmax_override,
                angles_deg = angles_deg,
                solver_requested = solver_requested,
                ebcm_threshold = ebcm_threshold,
                ebcm_ndgs = ebcm_ndgs,
                ebcm_maxiter = ebcm_maxiter,
                ebcm_full = ebcm_full,
                ebcm_loss_guard = ebcm_loss_guard,
                ebcm_loss_threshold = ebcm_loss_threshold,
                fallback_on_anomaly = fallback_on_anomaly,
                iitm_nr_scale_on_fallback = iitm_nr_scale_on_fallback,
                iitm_ntheta_scale_on_fallback = iitm_ntheta_scale_on_fallback,
            )
        catch e
            @warn "跳过 r=$(round(r,digits=3))μm [$(shape_type)]: $(sprint(showerror, e))"
            continue
        end

        valid_particles += 1
        push!(solver_used_per_radius, res.solver_used)
        push!(solver_paths, get(res.diagnostics, "solver_path", res.solver_used))
        push!(nmax_values, Int(get(res.diagnostics, "nmax", 0)))
        loss_est = get(res.diagnostics, "ebcm_loss_estimate", nothing)
        if loss_est isa Number && isfinite(loss_est)
            push!(ebcm_loss_estimates, Float64(loss_est))
        end
        nr_fb = Int(get(res.diagnostics, "Nr_fallback", 0))
        nth_fb = Int(get(res.diagnostics, "Ntheta_fallback", 0))
        nr_fb > 0 && push!(nr_fallback_values, nr_fb)
        nth_fb > 0 && push!(ntheta_fallback_values, nth_fb)
        if Bool(get(res.diagnostics, "fallback_used", false))
            push!(fallback_reasons, String(get(res.diagnostics, "fallback_reason", "unknown")))
        end

        push!(valid_radii, r)
        push!(valid_weights, w)
        push!(sigma_ext_vals, res.sigma_ext)
        push!(sigma_sca_vals, res.sigma_sca)
        push!(g_sigma_sca_vals, res.g * res.sigma_sca)
        push!(m11_vals, res.M11)
        push!(m12_vals, res.M12)
        push!(m33_vals, res.M33)
        push!(m34_vals, res.M34)
    end

    valid_particles > 0 || error("所有粒径点的 T-Matrix 求解均失败")

    r_valid = copy(valid_radii)
    w_valid = copy(valid_weights)
    if length(r_valid) == 1
        w_valid[1] = 1.0
        sigma_ext_eff = w_valid[1] * sigma_ext_vals[1]
        sigma_sca_eff = w_valid[1] * sigma_sca_vals[1]
        g_sum         = w_valid[1] * g_sigma_sca_vals[1]
        M11_sum .= w_valid[1] * sigma_sca_vals[1] .* m11_vals[1]
        M12_sum .= w_valid[1] * sigma_sca_vals[1] .* m12_vals[1]
        M33_sum .= w_valid[1] * sigma_sca_vals[1] .* m33_vals[1]
        M34_sum .= w_valid[1] * sigma_sca_vals[1] .* m34_vals[1]
    else
        tw_valid = trapz(w_valid, r_valid)
        if tw_valid > 0.0
            w_valid ./= tw_valid
        else
            fill!(w_valid, 1.0 / length(w_valid))
        end
        sigma_ext_eff = trapz(w_valid .* sigma_ext_vals, r_valid)
        sigma_sca_eff = trapz(w_valid .* sigma_sca_vals, r_valid)
        g_sum         = trapz(w_valid .* g_sigma_sca_vals, r_valid)
        for j in eachindex(M11_sum)
            M11_sum[j] = trapz([w_valid[i] * sigma_sca_vals[i] * m11_vals[i][j] for i in eachindex(r_valid)], r_valid)
            M12_sum[j] = trapz([w_valid[i] * sigma_sca_vals[i] * m12_vals[i][j] for i in eachindex(r_valid)], r_valid)
            M33_sum[j] = trapz([w_valid[i] * sigma_sca_vals[i] * m33_vals[i][j] for i in eachindex(r_valid)], r_valid)
            M34_sum[j] = trapz([w_valid[i] * sigma_sca_vals[i] * m34_vals[i][j] for i in eachindex(r_valid)], r_valid)
        end
    end

    if sigma_sca_eff > 1e-40
        inv_s = 1.0 / sigma_sca_eff
        M11_sum .*= inv_s;  M12_sum .*= inv_s
        M33_sum .*= inv_s;  M34_sum .*= inv_s
        g_eff = g_sum * inv_s
    else
        g_eff = 0.0
    end

    theta_r  = deg2rad.(angles_deg)
    norm_val = trapz(max.(M11_sum, 0.0) .* sin.(theta_r), theta_r)
    if norm_val > 1e-20
        f = 2.0 / norm_val
        M11_sum .*= f;  M12_sum .*= f
        M33_sum .*= f;  M34_sum .*= f
    else
        fill!(M11_sum, 1.0); fill!(M12_sum, 0.0)
        fill!(M33_sum, 0.0); fill!(M34_sum, 0.0)
    end

    omega0 = sigma_ext_eff > 1e-40 ?
             clamp(sigma_sca_eff / sigma_ext_eff, 0.0, 1.0) : 0.0

    theta_grid, cdf = build_phase_cdf(angles_deg, M11_sum)
    phase_m11_back = M11_sum[end]
    phase_m11_forward = forward_mode == "cone_avg" ?
                        cone_average_metric(angles_deg, M11_sum, forward_cone_deg) :
                        M11_sum[1]
    depol_back = safe_depol_ratio(M11_sum[end], M12_sum[end])
    depol_forward = forward_mode == "cone_avg" ?
                    safe_depol_ratio(
                        cone_average_metric(angles_deg, M11_sum, forward_cone_deg),
                        cone_average_metric(angles_deg, M12_sum, forward_cone_deg),
                    ) :
                    safe_depol_ratio(M11_sum[1], M12_sum[1])

    sigma_back_ref = sigma_sca_eff / (4pi) * phase_m11_back
    sigma_forward_ref = sigma_sca_eff / (4pi) * phase_m11_forward
    forward_back_ratio = sigma_back_ref > 1e-30 ? sigma_forward_ref / sigma_back_ref : 0.0
    unique_solvers = unique(solver_used_per_radius)
    solver_used = length(unique_solvers) == 1 ? unique_solvers[1] : "mixed"
    fallback_used = !isempty(fallback_reasons)
    fallback_summary = isempty(fallback_reasons) ? "" : join(unique(fallback_reasons), " | ")

    path_counts = Dict{String, Int}()
    for path in solver_paths
        path_counts[path] = get(path_counts, path, 0) + 1
    end
    solver_path_summary = isempty(path_counts) ? "" :
        join(["$(k) × $(v)" for (k, v) in sort(collect(path_counts); by = first)], ", ")

    return Dict{String, Any}(
        "sigma_ext"      => sigma_ext_eff,
        "sigma_sca"      => sigma_sca_eff,
        "omega0"         => omega0,
        "g"              => g_eff,
        "angles_deg"     => angles_deg,
        "M11"            => M11_sum,
        "M12"            => M12_sum,
        "M33"            => M33_sum,
        "M34"            => M34_sum,
        "theta_rad_grid" => theta_grid,
        "cdf_grid"       => cdf,
        "forward_mode"     => forward_mode,
        "forward_cone_deg" => forward_cone_deg,
        "phase_m11_back"   => phase_m11_back,
        "phase_m11_forward" => phase_m11_forward,
        "sigma_back_ref"   => sigma_back_ref,
        "sigma_forward_ref" => sigma_forward_ref,
        "forward_back_ratio" => forward_back_ratio,
        "depol_back"       => depol_back,
        "depol_forward"    => depol_forward,
        "solver_requested" => solver_requested,
        "solver_used"      => solver_used,
        "fallback_used"    => fallback_used,
        "fallback_count"   => length(fallback_reasons),
        "fallback_reason"  => fallback_summary,
        "solver_path_summary" => solver_path_summary,
        "ebcm_count"       => count(==("ebcm"), solver_used_per_radius),
        "iitm_count"       => count(==("iitm"), solver_used_per_radius),
        "nmax_min"         => minimum(nmax_values),
        "nmax_max"         => maximum(nmax_values),
        "ebcm_loss_estimate_max" =>
            (isempty(ebcm_loss_estimates) ? nothing : maximum(ebcm_loss_estimates)),
        "ebcm_loss_estimate_mean" =>
            (isempty(ebcm_loss_estimates) ? nothing : mean(ebcm_loss_estimates)),
        "ebcm_threshold_used" => ebcm_threshold,
        "ebcm_ndgs"          => ebcm_ndgs,
        "ebcm_maxiter"       => ebcm_maxiter,
        "iitm_nr"            => Nr,
        "iitm_ntheta"        => Ntheta,
        "iitm_nr_fallback"   => (isempty(nr_fallback_values) ? 0 : maximum(nr_fallback_values)),
        "iitm_ntheta_fallback" =>
            (isempty(ntheta_fallback_values) ? 0 : maximum(ntheta_fallback_values)),
    )
end

# ═══════════════════════════════════════════════════════════
# §6  偏振蒙特卡洛传输
# ═══════════════════════════════════════════════════════════

@inline function rotate_stokes(S::NTuple{4,Float64}, phi::Float64)
    I, Q, U, V = S
    c2 = cos(2phi); s2 = sin(2phi)
    return (I, Q*c2 + U*s2, -Q*s2 + U*c2, V)
end

@inline function apply_mueller(S::NTuple{4,Float64},
                                M11::Float64, M12::Float64,
                                M33::Float64, M34::Float64)
    M11 < 1e-20 && return S, 1.0
    I, Q, U, V = S
    m12   = M12 / M11;  m33 = M33 / M11;  m34 = M34 / M11
    I_out = 1.0 + m12 * Q
    I_out <= 1e-12 && return (1.0, 0.0, 0.0, 0.0), 0.0
    inv_I = 1.0 / I_out
    Q_new = (m12 + Q)       * inv_I
    U_new = (m33*U - m34*V) * inv_I
    V_new = (m34*U + m33*V) * inv_I
    pol_sq = Q_new^2 + U_new^2 + V_new^2
    if pol_sq > 1.0
        s = 1.0 / sqrt(pol_sq)
        Q_new *= s; U_new *= s; V_new *= s
    end
    return (1.0, Q_new, U_new, V_new), I_out
end

@inline function scatter_direction(ux::Float64, uy::Float64, uz::Float64,
                                    theta_s::Float64, phi_s::Float64)
    st = sin(theta_s); ct = cos(theta_s)
    sp = sin(phi_s);   cp = cos(phi_s)
    if abs(uz) > 0.99999
        sgn = sign(uz)
        return (st * cp, st * sp, ct * sgn)
    end
    sq = sqrt(1.0 - uz*uz)
    nx = st * (ux*uz*cp - uy*sp) / sq + ux*ct
    ny = st * (uy*uz*cp + ux*sp) / sq + uy*ct
    nz = -st * cp * sq                 + uz*ct
    n  = sqrt(nx*nx + ny*ny + nz*nz)
    return (nx/n, ny/n, nz/n)
end

struct MCStats
    avg_collisions       :: Float64
    absorbed_ratio       :: Float64
    backscatter_ratio    :: Float64
    transmit_ratio       :: Float64
    depolarization_ratio :: Float64
    backscatter_angle_dist :: Vector{Int}
    voxel_fields         :: Union{Nothing, Any}
end

struct MCVoxelObservables
    forward_I  :: Array{Float64,3}
    forward_Q  :: Array{Float64,3}
    forward_U  :: Array{Float64,3}
    forward_V  :: Array{Float64,3}
    back_I     :: Array{Float64,3}
    back_Q     :: Array{Float64,3}
    back_U     :: Array{Float64,3}
    back_V     :: Array{Float64,3}
    event_count :: Array{Float64,3}
end

mutable struct MCVoxelSparseEntry
    forward_I  :: Float64
    forward_Q  :: Float64
    forward_U  :: Float64
    forward_V  :: Float64
    back_I     :: Float64
    back_Q     :: Float64
    back_U     :: Float64
    back_V     :: Float64
    event_count :: Float64
end

struct MCChunkResult
    total_collisions :: Int
    absorbed_count   :: Int
    back_count       :: Int
    trans_count      :: Int
    angle_bins       :: Vector{Int}
    total_back_I     :: Float64
    total_back_Q     :: Float64
    total_back_U     :: Float64
    total_back_V     :: Float64
    voxel_map        :: Union{Nothing, Dict{Int,MCVoxelSparseEntry}}
end

struct DetectorCone
    dirs    :: Vector{NTuple{3,Float64}}
    weights :: Vector{Float64}
end

@inline function new_voxel_sparse_entry()
    return MCVoxelSparseEntry(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

@inline function voxel_linear_index(ix::Int, iy::Int, iz::Int, nx::Int, ny::Int)
    return ix + (iy - 1) * nx + (iz - 1) * nx * ny
end

@inline function stable_chunk_seed(base_seed::Int, chunk_idx::Int)
    seed_u = reinterpret(UInt64, Int64(base_seed))
    x = seed_u + UInt64(chunk_idx) * 0x9e3779b97f4a7c15
    x = xor(x, x >> 30) * 0xbf58476d1ce4e5b9
    x = xor(x, x >> 27) * 0x94d049bb133111eb
    x = xor(x, x >> 31)
    return Int(x % UInt64(typemax(Int) - 1)) + 1
end

function merge_sparse_voxel_fields!(voxel_fields::MCVoxelObservables,
                                    voxel_map::Dict{Int,MCVoxelSparseEntry})
    @inbounds for (lin, src) in voxel_map
        voxel_fields.forward_I[lin] += src.forward_I
        voxel_fields.forward_Q[lin] += src.forward_Q
        voxel_fields.forward_U[lin] += src.forward_U
        voxel_fields.forward_V[lin] += src.forward_V
        voxel_fields.back_I[lin]    += src.back_I
        voxel_fields.back_Q[lin]    += src.back_Q
        voxel_fields.back_U[lin]    += src.back_U
        voxel_fields.back_V[lin]    += src.back_V
        voxel_fields.event_count[lin] += src.event_count
    end
end

@inline function normalize_density_sampling_mode(mode)::Symbol
    ms = lowercase(strip(String(mode)))
    return ms == "trilinear" ? :trilinear : :nearest
end

@inline function build_centered_edges(origin::Float64, step::Float64, n::Int,
                                      lower::Float64, upper::Float64)
    edges = zeros(Float64, n + 1)
    edges[1] = lower
    edges[end] = upper
    if n > 1
        @inbounds for i in 1:(n - 1)
            edges[i + 1] = origin + (i - 0.5) * step
        end
    end
    return edges
end

@inline function slab_index(z::Float64, edges::Vector{Float64})
    return clamp(searchsortedlast(edges, z), 1, length(edges) - 1)
end

@inline function distance_to_slab_boundary(z::Float64, uz::Float64, idx::Int,
                                           edges::Vector{Float64})
    abs(uz) <= 1e-12 && return Inf
    z_edge = uz > 0.0 ? edges[idx + 1] : edges[idx]
    s = (z_edge - z) / uz
    return s > 0.0 ? s : 0.0
end

@inline function sample_density_nearest(density::Array{Float64,3},
                                        x0::Float64, y0::Float64, z0::Float64,
                                        dx::Float64, dy::Float64, dz::Float64,
                                        x::Float64, y::Float64, z::Float64)
    nx, ny, nz = size(density)
    fx = (x - x0) / dx + 1.0
    fy = (y - y0) / dy + 1.0
    fz = (z - z0) / dz + 1.0
    if fx < 1.0 || fx > nx || fy < 1.0 || fy > ny || fz < 1.0 || fz > nz
        return 0.0
    end
    ix = clamp(round(Int, fx), 1, nx)
    iy = clamp(round(Int, fy), 1, ny)
    iz = clamp(round(Int, fz), 1, nz)
    return density[ix, iy, iz]
end

@inline function sample_density_trilinear(density::Array{Float64,3},
                                          x0::Float64, y0::Float64, z0::Float64,
                                          dx::Float64, dy::Float64, dz::Float64,
                                          x::Float64, y::Float64, z::Float64)
    nx, ny, nz = size(density)
    fx = (x - x0) / dx + 1.0
    fy = (y - y0) / dy + 1.0
    fz = (z - z0) / dz + 1.0
    if fx < 1.0 || fx > nx || fy < 1.0 || fy > ny || fz < 1.0 || fz > nz
        return 0.0
    end

    ix0 = clamp(floor(Int, fx), 1, nx)
    iy0 = clamp(floor(Int, fy), 1, ny)
    iz0 = clamp(floor(Int, fz), 1, nz)
    ix1 = min(ix0 + 1, nx)
    iy1 = min(iy0 + 1, ny)
    iz1 = min(iz0 + 1, nz)

    tx = ix0 == nx ? 0.0 : clamp(fx - ix0, 0.0, 1.0)
    ty = iy0 == ny ? 0.0 : clamp(fy - iy0, 0.0, 1.0)
    tz = iz0 == nz ? 0.0 : clamp(fz - iz0, 0.0, 1.0)

    c000 = density[ix0, iy0, iz0]
    c100 = density[ix1, iy0, iz0]
    c010 = density[ix0, iy1, iz0]
    c110 = density[ix1, iy1, iz0]
    c001 = density[ix0, iy0, iz1]
    c101 = density[ix1, iy0, iz1]
    c011 = density[ix0, iy1, iz1]
    c111 = density[ix1, iy1, iz1]

    c00 = c000 * (1 - tx) + c100 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c11 = c011 * (1 - tx) + c111 * tx
    c0  = c00 * (1 - ty) + c10 * ty
    c1  = c01 * (1 - ty) + c11 * ty
    return c0 * (1 - tz) + c1 * tz
end

@inline function sample_density_local(density::Array{Float64,3}, sampling_mode::Symbol,
                                      x0::Float64, y0::Float64, z0::Float64,
                                      dx::Float64, dy::Float64, dz::Float64,
                                      x::Float64, y::Float64, z::Float64)
    if sampling_mode == :trilinear
        return sample_density_trilinear(density, x0, y0, z0, dx, dy, dz, x, y, z)
    end
    return sample_density_nearest(density, x0, y0, z0, dx, dy, dz, x, y, z)
end

function build_slab_majorants(beta_surf::Float64, density::Array{Float64,3},
                              sampling_mode::Symbol)
    nz = size(density, 3)
    slice_max = zeros(Float64, nz)
    @inbounds for iz in 1:nz
        slice_max[iz] = maximum(@view density[:, :, iz])
    end

    slab_beta = similar(slice_max)
    if sampling_mode == :trilinear
        @inbounds for iz in 1:nz
            lo = max(1, iz - 1)
            hi = min(nz, iz + 1)
            slab_beta[iz] = beta_surf * maximum(@view slice_max[lo:hi])
        end
    else
        @inbounds for iz in 1:nz
            slab_beta[iz] = beta_surf * slice_max[iz]
        end
    end
    return slab_beta
end

@inline function interpolate_mueller(a_deg::Vector{Float64},
                                     M11::Vector{Float64},
                                     M12::Vector{Float64},
                                     M33::Vector{Float64},
                                     M34::Vector{Float64},
                                     theta_deg::Float64)
    idx1 = clamp(searchsortedfirst(a_deg, theta_deg), 1, length(M11))
    if idx1 <= 1
        idx0 = 1
        frac = 0.0
    elseif idx1 > length(M11)
        idx0 = length(M11)
        idx1 = idx0
        frac = 0.0
    else
        idx0 = idx1 - 1
        denom = a_deg[idx1] - a_deg[idx0]
        frac = denom > 1e-12 ? (theta_deg - a_deg[idx0]) / denom : 0.0
    end
    inv_frac = 1.0 - frac
    return (
        M11[idx0] * inv_frac + M11[idx1] * frac,
        M12[idx0] * inv_frac + M12[idx1] * frac,
        M33[idx0] * inv_frac + M33[idx1] * frac,
        M34[idx0] * inv_frac + M34[idx1] * frac,
    )
end

@inline function local_beta_value(use_density_grid::Bool,
                                  density_grid::Array{Float64,3},
                                  sampling_mode::Symbol,
                                  beta_surf::Float64,
                                  H::Float64,
                                  x0::Float64, y0::Float64, z0::Float64,
                                  dx::Float64, dy::Float64, dz::Float64,
                                  x::Float64, y::Float64, z::Float64)
    if use_density_grid
        return beta_surf * sample_density_local(density_grid, sampling_mode,
                                                x0, y0, z0, dx, dy, dz, x, y, z)
    end
    return H > 0.0 ? beta_surf * exp(-z / H) : beta_surf
end

function build_detector_cone(axis::Symbol, half_angle_deg::Float64;
                             n_polar::Int = 2, n_azimuth::Int = 6)
    half_angle = deg2rad(clamp(half_angle_deg, 0.1, 90.0))
    μ_min = cos(half_angle)
    dirs = NTuple{3,Float64}[]
    weights = Float64[]
    for it in 1:max(n_polar, 1)
        μ_hi = 1.0 - (it - 1) * (1.0 - μ_min) / max(n_polar, 1)
        μ_lo = 1.0 - it * (1.0 - μ_min) / max(n_polar, 1)
        μ_mid = 0.5 * (μ_hi + μ_lo)
        θ = acos(clamp(μ_mid, -1.0, 1.0))
        ring_weight = 2pi * (μ_hi - μ_lo)
        for ip in 1:max(n_azimuth, 1)
            ϕ = 2pi * (ip - 0.5) / max(n_azimuth, 1)
            sx = sin(θ) * cos(ϕ)
            sy = sin(θ) * sin(ϕ)
            sz = axis === :forward ? cos(θ) : -cos(θ)
            push!(dirs, (sx, sy, sz))
            push!(weights, ring_weight / max(n_azimuth, 1))
        end
    end
    total_weight = sum(weights)
    total_weight > 0.0 && (weights ./= total_weight)
    return DetectorCone(dirs, weights)
end

@inline function voxel_index(x::Float64, y::Float64, z::Float64,
                             x0::Float64, y0::Float64, z0::Float64,
                             dx::Float64, dy::Float64, dz::Float64,
                             nx::Int, ny::Int, nz::Int)
    fx = (x - x0) / dx + 1.0
    fy = (y - y0) / dy + 1.0
    fz = (z - z0) / dz + 1.0
    if fx < 1.0 || fx > nx || fy < 1.0 || fy > ny || fz < 1.0 || fz > nz
        return nothing
    end
    return (
        clamp(round(Int, fx), 1, nx),
        clamp(round(Int, fy), 1, ny),
        clamp(round(Int, fz), 1, nz),
    )
end

@inline function direction_to_scattering_angles(ux::Float64, uy::Float64, uz::Float64,
                                                dxo::Float64, dyo::Float64, dzo::Float64)
    ct = clamp(ux * dxo + uy * dyo + uz * dzo, -1.0, 1.0)
    θ = acos(ct)
    if abs(uz) > 0.99999
        ϕ = atan(dyo, dxo)
        return θ, ϕ
    end
    sq = sqrt(max(1.0 - uz * uz, 1e-20))
    e1x = ux * uz / sq
    e1y = uy * uz / sq
    e1z = -sq
    e2x = -uy / sq
    e2y =  ux / sq
    e2z =  0.0
    px = dxo - ct * ux
    py = dyo - ct * uy
    pz = dzo - ct * uz
    a = px * e1x + py * e1y + pz * e1z
    b = px * e2x + py * e2y + pz * e2z
    return θ, atan(b, a)
end

function escape_transmittance(use_density_grid::Bool,
                              density_grid::Array{Float64,3},
                              sampling_mode::Symbol,
                              beta_surf::Float64,
                              H::Float64,
                              x0::Float64, y0::Float64, z0::Float64,
                              dx::Float64, dy::Float64, dz::Float64,
                              thickness::Float64,
                              x::Float64, y::Float64, z::Float64,
                              ux::Float64, uy::Float64, uz::Float64)
    abs(uz) <= 1e-12 && return 0.0
    s_exit = uz > 0.0 ? (thickness - z) / uz : -z / uz
    s_exit <= 0.0 && return 0.0
    step_len = max(min(dx, dy, dz) * 0.75, 1e-3)
    n_steps = max(1, ceil(Int, s_exit / step_len))
    ds = s_exit / n_steps
    τ = 0.0
    @inbounds for i in 1:n_steps
        s_mid = (i - 0.5) * ds
        xm = x + s_mid * ux
        ym = y + s_mid * uy
        zm = z + s_mid * uz
        β = local_beta_value(use_density_grid, density_grid, sampling_mode, beta_surf, H,
                             x0, y0, z0, dx, dy, dz, xm, ym, zm)
        τ += β * ds
    end
    return exp(-τ)
end

function accumulate_detector_contribution!(voxel_fields::MCVoxelObservables,
                                          ix::Int, iy::Int, iz::Int,
                                          S_in::NTuple{4,Float64}, weight::Float64,
                                          ux::Float64, uy::Float64, uz::Float64,
                                          use_density_grid::Bool,
                                          density_grid::Array{Float64,3},
                                          sampling_mode::Symbol,
                                          beta_surf::Float64,
                                          H::Float64,
                                          x0::Float64, y0::Float64, z0::Float64,
                                          dx::Float64, dy::Float64, dz::Float64,
                                          thickness::Float64,
                                          x::Float64, y::Float64, z::Float64,
                                          a_deg::Vector{Float64},
                                          M11::Vector{Float64},
                                          M12::Vector{Float64},
                                          M33::Vector{Float64},
                                          M34::Vector{Float64},
                                          forward_cone::DetectorCone,
                                          back_cone::DetectorCone)
    voxel_fields.event_count[ix, iy, iz] += 1.0
    for (dir, w_cone) in zip(forward_cone.dirs, forward_cone.weights)
        dxo, dyo, dzo = dir
        θ, ϕ = direction_to_scattering_angles(ux, uy, uz, dxo, dyo, dzo)
        m11i, m12i, m33i, m34i = interpolate_mueller(a_deg, M11, M12, M33, M34, rad2deg(θ))
        S_rot = rotate_stokes(S_in, -ϕ)
        S_out, i_factor = apply_mueller(S_rot, m11i, m12i, m33i, m34i)
        i_factor <= 0.0 && continue
        T = escape_transmittance(use_density_grid, density_grid, sampling_mode, beta_surf, H,
                                 x0, y0, z0, dx, dy, dz, thickness, x, y, z, dxo, dyo, dzo)
        contrib = weight * i_factor * T * w_cone
        voxel_fields.forward_I[ix, iy, iz] += contrib
        voxel_fields.forward_Q[ix, iy, iz] += contrib * S_out[2]
        voxel_fields.forward_U[ix, iy, iz] += contrib * S_out[3]
        voxel_fields.forward_V[ix, iy, iz] += contrib * S_out[4]
    end
    for (dir, w_cone) in zip(back_cone.dirs, back_cone.weights)
        dxo, dyo, dzo = dir
        θ, ϕ = direction_to_scattering_angles(ux, uy, uz, dxo, dyo, dzo)
        m11i, m12i, m33i, m34i = interpolate_mueller(a_deg, M11, M12, M33, M34, rad2deg(θ))
        S_rot = rotate_stokes(S_in, -ϕ)
        S_out, i_factor = apply_mueller(S_rot, m11i, m12i, m33i, m34i)
        i_factor <= 0.0 && continue
        T = escape_transmittance(use_density_grid, density_grid, sampling_mode, beta_surf, H,
                                 x0, y0, z0, dx, dy, dz, thickness, x, y, z, dxo, dyo, dzo)
        contrib = weight * i_factor * T * w_cone
        voxel_fields.back_I[ix, iy, iz] += contrib
        voxel_fields.back_Q[ix, iy, iz] += contrib * S_out[2]
        voxel_fields.back_U[ix, iy, iz] += contrib * S_out[3]
        voxel_fields.back_V[ix, iy, iz] += contrib * S_out[4]
    end
end

function accumulate_detector_contribution!(voxel_map::Dict{Int,MCVoxelSparseEntry},
                                          ix::Int, iy::Int, iz::Int,
                                          nx::Int, ny::Int,
                                          S_in::NTuple{4,Float64}, weight::Float64,
                                          ux::Float64, uy::Float64, uz::Float64,
                                          use_density_grid::Bool,
                                          density_grid::Array{Float64,3},
                                          sampling_mode::Symbol,
                                          beta_surf::Float64,
                                          H::Float64,
                                          x0::Float64, y0::Float64, z0::Float64,
                                          dx::Float64, dy::Float64, dz::Float64,
                                          thickness::Float64,
                                          x::Float64, y::Float64, z::Float64,
                                          a_deg::Vector{Float64},
                                          M11::Vector{Float64},
                                          M12::Vector{Float64},
                                          M33::Vector{Float64},
                                          M34::Vector{Float64},
                                          forward_cone::DetectorCone,
                                          back_cone::DetectorCone)
    lin = voxel_linear_index(ix, iy, iz, nx, ny)
    entry = get!(voxel_map, lin) do
        new_voxel_sparse_entry()
    end
    entry.event_count += 1.0
    for (dir, w_cone) in zip(forward_cone.dirs, forward_cone.weights)
        dxo, dyo, dzo = dir
        θ, ϕ = direction_to_scattering_angles(ux, uy, uz, dxo, dyo, dzo)
        m11i, m12i, m33i, m34i = interpolate_mueller(a_deg, M11, M12, M33, M34, rad2deg(θ))
        S_rot = rotate_stokes(S_in, -ϕ)
        S_out, i_factor = apply_mueller(S_rot, m11i, m12i, m33i, m34i)
        i_factor <= 0.0 && continue
        T = escape_transmittance(use_density_grid, density_grid, sampling_mode, beta_surf, H,
                                 x0, y0, z0, dx, dy, dz, thickness, x, y, z, dxo, dyo, dzo)
        contrib = weight * i_factor * T * w_cone
        entry.forward_I += contrib
        entry.forward_Q += contrib * S_out[2]
        entry.forward_U += contrib * S_out[3]
        entry.forward_V += contrib * S_out[4]
    end
    for (dir, w_cone) in zip(back_cone.dirs, back_cone.weights)
        dxo, dyo, dzo = dir
        θ, ϕ = direction_to_scattering_angles(ux, uy, uz, dxo, dyo, dzo)
        m11i, m12i, m33i, m34i = interpolate_mueller(a_deg, M11, M12, M33, M34, rad2deg(θ))
        S_rot = rotate_stokes(S_in, -ϕ)
        S_out, i_factor = apply_mueller(S_rot, m11i, m12i, m33i, m34i)
        i_factor <= 0.0 && continue
        T = escape_transmittance(use_density_grid, density_grid, sampling_mode, beta_surf, H,
                                 x0, y0, z0, dx, dy, dz, thickness, x, y, z, dxo, dyo, dzo)
        contrib = weight * i_factor * T * w_cone
        entry.back_I += contrib
        entry.back_Q += contrib * S_out[2]
        entry.back_U += contrib * S_out[3]
        entry.back_V += contrib * S_out[4]
    end
end

function run_monte_carlo_chunk(photon_range::UnitRange{Int},
                               chunk_seed::Int,
                               theta_grid::Vector{Float64},
                               cdf::Vector{Float64},
                               M11::Vector{Float64},
                               M12::Vector{Float64},
                               M33::Vector{Float64},
                               M34::Vector{Float64},
                               a_deg::Vector{Float64},
                               use_density_grid::Bool,
                               density_grid::Array{Float64,3},
                               sampling_mode::Symbol,
                               beta_surf::Float64,
                               omega0::Float64,
                               thickness::Float64,
                               H::Float64,
                               x0::Float64, y0::Float64, z0::Float64,
                               dx::Float64, dy::Float64, dz::Float64,
                               z_edges::Vector{Float64},
                               slab_beta::Vector{Float64},
                               global_beta_max::Float64,
                               slab_eps::Float64,
                               collect_voxel_fields::Bool,
                               forward_cone::DetectorCone,
                               back_cone::DetectorCone)
    nx, ny, nz = size(density_grid)
    rng = MersenneTwister(chunk_seed)
    total_collisions = 0
    absorbed_count   = 0
    back_count       = 0
    trans_count      = 0
    angle_bins       = zeros(Int, 900)
    total_back_I     = 0.0
    total_back_Q     = 0.0
    total_back_U     = 0.0
    total_back_V     = 0.0
    voxel_map = collect_voxel_fields ? Dict{Int,MCVoxelSparseEntry}() : nothing

    for _ in photon_range
        x, y, z    = 0.0, 0.0, 0.0
        ux, uy, uz = 0.0, 0.0, 1.0
        S          = (1.0, 1.0, 0.0, 0.0)
        weight     = 1.0
        alive      = true

        while alive
            beta_majorant = global_beta_max
            s_boundary = Inf
            if use_density_grid
                idx_slab = slab_index(z, z_edges)
                beta_majorant = slab_beta[idx_slab]
                s_boundary = distance_to_slab_boundary(z, uz, idx_slab, z_edges)
                if beta_majorant <= 1e-30
                    if isfinite(s_boundary)
                        s_move = s_boundary + slab_eps
                        x_new  = x + s_move * ux
                        y_new  = y + s_move * uy
                        z_new  = z + s_move * uz
                        if z_new < 0.0
                            back_count += 1
                            alive       = false
                            cos_a = clamp(-uz, 0.0, 1.0)
                            bin   = min(round(Int, rad2deg(acos(cos_a)) * 10) + 1, 900)
                            angle_bins[bin] += 1
                            _, Q, U, V = S
                            total_back_I += weight
                            total_back_Q += Q * weight
                            total_back_U += U * weight
                            total_back_V += V * weight
                            break
                        end
                        if z_new >= thickness
                            trans_count += 1
                            alive         = false
                            break
                        end
                        x, y, z = x_new, y_new, z_new
                        continue
                    end
                    beta_majorant = global_beta_max
                end
            end

            r_step = max(rand(rng), 1e-12)
            s_tent = -log(r_step) / beta_majorant
            if use_density_grid && isfinite(s_boundary) && s_tent >= s_boundary
                s_move = s_boundary + slab_eps
                x_new  = x + s_move * ux
                y_new  = y + s_move * uy
                z_new  = z + s_move * uz
                if z_new < 0.0
                    back_count += 1
                    alive       = false
                    cos_a = clamp(-uz, 0.0, 1.0)
                    bin   = min(round(Int, rad2deg(acos(cos_a)) * 10) + 1, 900)
                    angle_bins[bin] += 1
                    _, Q, U, V = S
                    total_back_I += weight
                    total_back_Q += Q * weight
                    total_back_U += U * weight
                    total_back_V += V * weight
                    break
                end
                if z_new >= thickness
                    trans_count += 1
                    alive         = false
                    break
                end
                x, y, z = x_new, y_new, z_new
                continue
            end

            x_new  = x + s_tent * ux
            y_new  = y + s_tent * uy
            z_new  = z + s_tent * uz

            if z_new < 0.0
                back_count += 1
                alive       = false
                cos_a = clamp(-uz, 0.0, 1.0)
                bin   = min(round(Int, rad2deg(acos(cos_a)) * 10) + 1, 900)
                angle_bins[bin] += 1
                _, Q, U, V = S
                total_back_I += weight
                total_back_Q += Q * weight
                total_back_U += U * weight
                total_back_V += V * weight
                break
            end

            if z_new >= thickness
                trans_count += 1
                alive         = false
                break
            end

            x, y, z = x_new, y_new, z_new

            beta_local = local_beta_value(use_density_grid, density_grid, sampling_mode,
                                          beta_surf, H, x0, y0, z0, dx, dy, dz, x, y, z)
            rand(rng) > (beta_local / beta_majorant) && continue

            if rand(rng) > omega0
                absorbed_count += 1
                alive           = false
                continue
            end

            total_collisions += 1

            if voxel_map !== nothing
                vidx = voxel_index(x, y, z, x0, y0, z0, dx, dy, dz, nx, ny, nz)
                if vidx !== nothing
                    ix, iy, iz = vidx
                    accumulate_detector_contribution!(voxel_map, ix, iy, iz, nx, ny, S, weight,
                                                      ux, uy, uz, use_density_grid, density_grid,
                                                      sampling_mode, beta_surf, H, x0, y0, z0,
                                                      dx, dy, dz, thickness, x, y, z,
                                                      a_deg, M11, M12, M33, M34,
                                                      forward_cone, back_cone)
                end
            end

            theta_s = sample_scattering_theta(rng, theta_grid, cdf)
            phi_s   = rand(rng) * 2pi
            S = rotate_stokes(S, -phi_s)

            theta_d = rad2deg(theta_s)
            idx1    = clamp(searchsortedfirst(a_deg, theta_d), 1, length(M11))
            if idx1 <= 1
                idx0 = 1
                fθ   = 0.0
            elseif idx1 > length(M11)
                idx0 = length(M11)
                idx1 = idx0
                fθ   = 0.0
            else
                idx0 = idx1 - 1
                denom = a_deg[idx1] - a_deg[idx0]
                fθ = denom > 1e-12 ? (theta_d - a_deg[idx0]) / denom : 0.0
            end
            M11i = M11[idx0] * (1.0 - fθ) + M11[idx1] * fθ
            M12i = M12[idx0] * (1.0 - fθ) + M12[idx1] * fθ
            M33i = M33[idx0] * (1.0 - fθ) + M33[idx1] * fθ
            M34i = M34[idx0] * (1.0 - fθ) + M34[idx1] * fθ
            S, i_factor = apply_mueller(S, M11i, M12i, M33i, M34i)
            weight *= i_factor

            ux, uy, uz = scatter_direction(ux, uy, uz, theta_s, phi_s)
        end
    end

    return MCChunkResult(
        total_collisions,
        absorbed_count,
        back_count,
        trans_count,
        angle_bins,
        total_back_I,
        total_back_Q,
        total_back_U,
        total_back_V,
        voxel_map,
    )
end

function run_monte_carlo(scatter::Dict, mc_cfg::Dict)
    beta_surf = Float64(mc_cfg["beta_ext_surf"])
    omega0    = Float64(get(mc_cfg, "omega0", scatter["omega0"]))
    thickness = Float64(mc_cfg["thickness_m"])
    H         = Float64(get(mc_cfg, "scale_height_m", 0.0))
    n_photons = Int(    mc_cfg["n_photons"])
    seed      = Int(    get(mc_cfg, "seed", 0))
    collect_voxel_fields = Bool(get(mc_cfg, "collect_voxel_fields", false))
    forward_half_angle_deg = Float64(get(mc_cfg, "field_forward_half_angle_deg", 90.0))
    back_half_angle_deg = Float64(get(mc_cfg, "field_back_half_angle_deg", 90.0))
    quad_polar = Int(get(mc_cfg, "field_quadrature_polar", 2))
    quad_azimuth = Int(get(mc_cfg, "field_quadrature_azimuth", 6))

    theta_grid = scatter["theta_rad_grid"] :: Vector{Float64}
    cdf        = scatter["cdf_grid"]       :: Vector{Float64}
    M11        = scatter["M11"]            :: Vector{Float64}
    M12        = scatter["M12"]            :: Vector{Float64}
    M33        = scatter["M33"]            :: Vector{Float64}
    M34        = scatter["M34"]            :: Vector{Float64}
    a_deg      = scatter["angles_deg"]     :: Vector{Float64}

    density_grid_raw = get(mc_cfg, "density_grid", nothing)
    use_density_grid = density_grid_raw isa Array{Float64,3}
    sampling_mode = normalize_density_sampling_mode(get(mc_cfg, "density_sampling", "nearest"))
    field_xy_centered = Bool(get(mc_cfg, "field_xy_centered", true))

    density_grid = zeros(Float64, 1, 1, 1)
    z_edges = Float64[]
    x0 = 0.0
    y0 = 0.0
    z0 = 0.0
    dx = 1.0
    dy = 1.0
    dz = thickness > 0.0 ? thickness : 1.0
    slab_beta = Float64[]
    global_beta_max = beta_surf
    voxel_fields = nothing
    forward_cone = DetectorCone(NTuple{3,Float64}[], Float64[])
    back_cone = DetectorCone(NTuple{3,Float64}[], Float64[])
    if use_density_grid
        density_grid = density_grid_raw :: Array{Float64,3}
        nx, ny, nz = size(density_grid)
        axis_raw = get(mc_cfg, "field_axis", nothing)
        if axis_raw isa Vector{Float64} && length(axis_raw) == nz
            axis = axis_raw :: Vector{Float64}
            z0 = axis[1]
            dz = nz > 1 ? max(axis[2] - axis[1], 1e-9) : max(thickness, 1e-9)
        else
            z0 = 0.0
            dz = nz > 1 ? max(thickness / (nz - 1), 1e-9) : max(thickness, 1e-9)
        end
        dx = nx > 1 ? max(thickness / (nx - 1), 1e-9) : max(thickness, 1e-9)
        dy = ny > 1 ? max(thickness / (ny - 1), 1e-9) : max(thickness, 1e-9)
        x0 = field_xy_centered ? -0.5 * (nx - 1) * dx : 0.0
        y0 = field_xy_centered ? -0.5 * (ny - 1) * dy : 0.0
        z_edges = build_centered_edges(z0, dz, nz, 0.0, thickness)
        slab_beta = build_slab_majorants(beta_surf, density_grid, sampling_mode)
        global_beta_max = maximum(slab_beta)
        if global_beta_max <= 1e-30
            return MCStats(0.0, 0.0, 0.0, 1.0, 0.0, zeros(Int, 900), nothing)
        end
        if collect_voxel_fields
            voxel_fields = MCVoxelObservables(
                zeros(Float64, nx, ny, nz),
                zeros(Float64, nx, ny, nz),
                zeros(Float64, nx, ny, nz),
                zeros(Float64, nx, ny, nz),
                zeros(Float64, nx, ny, nz),
                zeros(Float64, nx, ny, nz),
                zeros(Float64, nx, ny, nz),
                zeros(Float64, nx, ny, nz),
                zeros(Float64, nx, ny, nz),
            )
            forward_cone = build_detector_cone(:forward, forward_half_angle_deg;
                                               n_polar = quad_polar,
                                               n_azimuth = quad_azimuth)
            back_cone = build_detector_cone(:back, back_half_angle_deg;
                                            n_polar = quad_polar,
                                            n_azimuth = quad_azimuth)
        end
    end

    slab_eps         = use_density_grid ? max(dz * 1e-6, 1e-9) : 0.0
    nx, ny, nz = size(density_grid)
    chunk_size = 16_384
    n_chunks = max(1, cld(n_photons, chunk_size))
    chunk_ranges = Vector{UnitRange{Int}}(undef, n_chunks)
    for chunk_idx in 1:n_chunks
        start_idx = (chunk_idx - 1) * chunk_size + 1
        stop_idx = min(chunk_idx * chunk_size, n_photons)
        chunk_ranges[chunk_idx] = start_idx:stop_idx
    end

    chunk_seeds = Vector{Int}(undef, n_chunks)
    if seed == 0
        seed_rng = Random.default_rng()
        @inbounds for chunk_idx in 1:n_chunks
            chunk_seeds[chunk_idx] = rand(seed_rng, 1:(typemax(Int) - 1))
        end
    else
        @inbounds for chunk_idx in 1:n_chunks
            chunk_seeds[chunk_idx] = stable_chunk_seed(seed, chunk_idx)
        end
    end

    chunk_results = Vector{MCChunkResult}(undef, n_chunks)
    Threads.@threads for chunk_idx in 1:n_chunks
        chunk_results[chunk_idx] = run_monte_carlo_chunk(
            chunk_ranges[chunk_idx],
            chunk_seeds[chunk_idx],
            theta_grid,
            cdf,
            M11,
            M12,
            M33,
            M34,
            a_deg,
            use_density_grid,
            density_grid,
            sampling_mode,
            beta_surf,
            omega0,
            thickness,
            H,
            x0, y0, z0,
            dx, dy, dz,
            z_edges,
            slab_beta,
            global_beta_max,
            slab_eps,
            voxel_fields !== nothing,
            forward_cone,
            back_cone,
        )
    end

    total_collisions = 0
    absorbed_count   = 0
    back_count       = 0
    trans_count      = 0
    angle_bins       = zeros(Int, 900)
    total_back_I     = 0.0
    total_back_Q     = 0.0
    total_back_U     = 0.0
    total_back_V     = 0.0
    for chunk_result in chunk_results
        total_collisions += chunk_result.total_collisions
        absorbed_count   += chunk_result.absorbed_count
        back_count       += chunk_result.back_count
        trans_count      += chunk_result.trans_count
        total_back_I     += chunk_result.total_back_I
        total_back_Q     += chunk_result.total_back_Q
        total_back_U     += chunk_result.total_back_U
        total_back_V     += chunk_result.total_back_V
        @inbounds for i in eachindex(angle_bins)
            angle_bins[i] += chunk_result.angle_bins[i]
        end
        if voxel_fields !== nothing && chunk_result.voxel_map !== nothing
            merge_sparse_voxel_fields!(voxel_fields, chunk_result.voxel_map)
        end
    end

    avg_pol = total_back_I > 0.0 ?
              sqrt(total_back_Q^2 + total_back_U^2 + total_back_V^2) / total_back_I : 1.0
    n_safe  = max(n_photons, 1)
    if voxel_fields !== nothing
        inv_n = 1.0 / n_safe
        voxel_fields.forward_I   .*= inv_n
        voxel_fields.forward_Q   .*= inv_n
        voxel_fields.forward_U   .*= inv_n
        voxel_fields.forward_V   .*= inv_n
        voxel_fields.back_I      .*= inv_n
        voxel_fields.back_Q      .*= inv_n
        voxel_fields.back_U      .*= inv_n
        voxel_fields.back_V      .*= inv_n
    end

    return MCStats(
        total_collisions / n_safe,
        absorbed_count   / n_safe,
        back_count       / n_safe,
        trans_count      / n_safe,
        1.0 - avg_pol,
        angle_bins,
        voxel_fields,
    )
end

# ═══════════════════════════════════════════════════════════
# §7  3D 密度场 & 多场散射场（v3 重构）
# ═══════════════════════════════════════════════════════════

function generate_field(config::Dict)
    N   = Int(   get(config, "grid_dim",          80))
    L   = Float64(get(config, "L_size",           20.0))
    z_c = Float64(get(config, "cloud_center_z",   10.0))
    z_s = Float64(get(config, "cloud_thickness",   8.0)) / 2.355
    ts  = Float64(get(config, "turbulence_scale",  4.0))

    axis = collect(range(0.0, L; length = N))
    rng  = MersenneTwister(101)

    function noise_layer(scale::Float64, strength::Float64)
        res = max(2, round(Int, N / scale))
        raw = rand(rng, Float64, res, res, res)
        out = zeros(Float64, N, N, N)
        denom = max(N - 1, 1)
        @inbounds for iz in 1:N, iy in 1:N, ix in 1:N
            si = clamp(round(Int, (ix-1)*(res-1)/denom) + 1, 1, res)
            sj = clamp(round(Int, (iy-1)*(res-1)/denom) + 1, 1, res)
            sk = clamp(round(Int, (iz-1)*(res-1)/denom) + 1, 1, res)
            out[ix,iy,iz] = raw[si,sj,sk] * strength
        end
        return out
    end

    noise = noise_layer(ts, 0.6) .+ noise_layer(ts/2, 0.3) .+ noise_layer(ts/4, 0.1)
    mn, mx = extrema(noise)
    noise .= (noise .- mn) ./ max(mx - mn, 1e-12)

    density = zeros(Float64, N, N, N)
    @inbounds for iz in 1:N
        vp = exp(-0.5 * ((axis[iz] - z_c) / z_s)^2)
        @views density[:, :, iz] .= vp .* (0.3 .+ 0.7 .* noise[:, :, iz])
    end
    density .= clamp.(density, 0.0, 1.0)
    maximum(density) < 0.01 && (density .= max.(density, 0.01))

    return Dict{String, Any}(
        "density_norm" => density,
        "axis"         => axis,
        "L"            => L,
        "dim"          => N,
    )
end

"""
从 Step2 的散射参数推导三维代理散射场。

当前返回：
  - beta_back     : 后向散射强度场
  - beta_forward  : 前向散射强度场
  - depol_ratio   : 退偏比场（当前使用后向退偏比代理）
  - lut_back      : 后向一维 LUT
  - lut_forward   : 前向一维 LUT
  - lut_depol     : 退偏一维 LUT
"""
function compute_scatter_fields(field::Dict, scatter::Dict, config::Dict, mc::MCStats)
    N = field["dim"] :: Int
    voxel = mc.voxel_fields
    voxel === nothing && return compute_scatter_fields(field, scatter, config)

    beta_back = copy(voxel.back_I)
    beta_forward = copy(voxel.forward_I)
    forward_Q = copy(voxel.forward_Q)
    forward_U = copy(voxel.forward_U)
    forward_V = copy(voxel.forward_V)
    back_Q = copy(voxel.back_Q)
    back_U = copy(voxel.back_U)
    back_V = copy(voxel.back_V)
    event_count = copy(voxel.event_count)

    depol_ratio = zeros(Float64, N, N, N)
    @inbounds for iz in 1:N, iy in 1:N, ix in 1:N
        I = beta_back[ix, iy, iz]
        if I > 1e-20
            pol = sqrt(back_Q[ix, iy, iz]^2 + back_U[ix, iy, iz]^2 + back_V[ix, iy, iz]^2) / I
            depol_ratio[ix, iy, iz] = clamp(1.0 - pol, 0.0, 1.0)
        end
    end

    lut_back = zeros(Float64, N)
    lut_forward = zeros(Float64, N)
    lut_depol = zeros(Float64, N)
    @inbounds for iz in 1:N
        back_slice = @view beta_back[:, :, iz]
        forward_slice = @view beta_forward[:, :, iz]
        q_slice = @view back_Q[:, :, iz]
        u_slice = @view back_U[:, :, iz]
        v_slice = @view back_V[:, :, iz]
        lut_back[iz] = mean(back_slice)
        lut_forward[iz] = mean(forward_slice)
        I_sum = sum(back_slice)
        if I_sum > 1e-20
            pol = sqrt(sum(q_slice)^2 + sum(u_slice)^2 + sum(v_slice)^2) / I_sum
            lut_depol[iz] = clamp(1.0 - pol, 0.0, 1.0)
        end
    end

    total_back = sum(beta_back)
    total_forward = sum(beta_forward)
    depol_back = 0.0
    if total_back > 1e-20
        pol = sqrt(sum(back_Q)^2 + sum(back_U)^2 + sum(back_V)^2) / total_back
        depol_back = clamp(1.0 - pol, 0.0, 1.0)
    end
    depol_forward = 0.0
    if total_forward > 1e-20
        pol = sqrt(sum(forward_Q)^2 + sum(forward_U)^2 + sum(forward_V)^2) / total_forward
        depol_forward = clamp(1.0 - pol, 0.0, 1.0)
    end
    forward_back_ratio = total_back > 1e-30 ? total_forward / total_back : 0.0

    return Dict{String, Any}(
        "beta_back"     => beta_back,
        "beta_forward"  => beta_forward,
        "depol_ratio"   => depol_ratio,
        "forward_Q"     => forward_Q,
        "forward_U"     => forward_U,
        "forward_V"     => forward_V,
        "back_Q"        => back_Q,
        "back_U"        => back_U,
        "back_V"        => back_V,
        "event_count"   => event_count,
        "lut_back"      => lut_back,
        "lut_forward"   => lut_forward,
        "lut_depol"     => lut_depol,
        "depol_back"    => depol_back,
        "depol_forward" => depol_forward,
        "forward_back_ratio" => forward_back_ratio,
        "sum_back"      => total_back,
        "sum_forward"   => total_forward,
        "sample_count"  => sum(event_count),
    )
end

function compute_scatter_fields(field::Dict, scatter::Dict, config::Dict)
    axis    = field["axis"]         :: Vector{Float64}
    density = field["density_norm"] :: Array{Float64,3}
    N       = field["dim"]          :: Int
    L       = field["L"]            :: Float64

    r_bottom = Float64(get(config, "r_bottom", 0.5))
    r_top    = Float64(get(config, "r_top",    2.0))

    sigma_back_ref = Float64(get(scatter, "sigma_back_ref", 0.0))
    sigma_forward_ref = Float64(get(scatter, "sigma_forward_ref", sigma_back_ref))
    depol_back = Float64(get(scatter, "depol_back", 0.0))
    depol_forward = Float64(get(scatter, "depol_forward", depol_back))

    # 代表粒径（build_configs 写入）；回落到几何均值
    r_repr = Float64(get(scatter, "r_repr", sqrt(max(r_bottom, 1e-6) * max(r_top, 1e-6))))
    geo_ref = pi * (r_repr * 1e-6)^2   # [m²]

    z_norm = clamp.(axis ./ L, 0.0, 1.0)
    r_prof = r_bottom .+ z_norm .* (r_top - r_bottom)

    lut_back = zeros(Float64, N)
    lut_forward = zeros(Float64, N)
    # Export a z-layer proxy field; this is not a voxel-history inversion.
    lut_depol = fill(depol_back, N)
    @inbounds for i in 1:N
        geo_i  = pi * (r_prof[i] * 1e-6)^2
        scale  = geo_ref > 0.0 ? geo_i / geo_ref : 1.0
        lut_back[i] = sigma_back_ref * scale * 1e-6
        lut_forward[i] = sigma_forward_ref * scale * 1e-6
    end

    beta_back = zeros(Float64, N, N, N)
    beta_forward = zeros(Float64, N, N, N)
    depol_ratio = zeros(Float64, N, N, N)
    @inbounds for iz in 1:N
        @views beta_back[:, :, iz] .= density[:, :, iz] .* lut_back[iz]
        @views beta_forward[:, :, iz] .= density[:, :, iz] .* lut_forward[iz]
        @views depol_ratio[:, :, iz] .= lut_depol[iz]
    end
    total_back = sum(beta_back)
    total_forward = sum(beta_forward)
    forward_back_ratio = total_back > 1e-30 ? total_forward / total_back : 0.0
    return Dict{String, Any}(
        "beta_back"    => beta_back,
        "beta_forward" => beta_forward,
        "depol_ratio"  => depol_ratio,
        "lut_back"     => lut_back,
        "lut_forward"  => lut_forward,
        "lut_depol"    => lut_depol,
        "depol_back"   => depol_back,
        "depol_forward" => depol_forward,
        "forward_back_ratio" => forward_back_ratio,
        "sum_back" => total_back,
        "sum_forward" => total_forward,
    )
end

function normalize_field_compute_mode(mode)::String
    mode_str = lowercase(strip(String(mode)))
    return mode_str in ("proxy_only", "exact_only", "both") ? mode_str : "proxy_only"
end

function build_field_bundle(field::Dict, scatter::Dict, config::Dict, mc::MCStats)
    requested_mode = normalize_field_compute_mode(get(config, "field_compute_mode", "proxy_only"))
    families = Dict{String, Any}()
    mode_note = nothing

    if requested_mode != "exact_only"
        families["proxy"] = compute_scatter_fields(field, scatter, config)
    end

    if requested_mode != "proxy_only" && mc.voxel_fields !== nothing
        families["exact"] = compute_scatter_fields(field, scatter, config, mc)
    end

    if isempty(families)
        families["proxy"] = compute_scatter_fields(field, scatter, config)
        mode_note = "exact field unavailable, fell back to proxy field only"
    elseif requested_mode == "both" && !haskey(families, "exact")
        mode_note = "exact field unavailable, exported proxy field only"
    end

    effective_mode = length(families) >= 2 ? "both" :
                     (haskey(families, "exact") ? "exact_only" : "proxy_only")
    primary_family = haskey(families, "proxy") ? "proxy" : first(keys(families))

    bundle = Dict{String, Any}(
        "families" => families,
        "requested_field_compute_mode" => requested_mode,
        "effective_field_compute_mode" => effective_mode,
        "primary_field_family" => primary_family,
    )
    mode_note === nothing || (bundle["field_mode_note"] = mode_note)
    return bundle
end

function build_field_catalog(bundle::Dict)
    families = bundle["families"] :: Dict{String, Any}
    catalog = Dict{String, Any}()

    if haskey(families, "proxy")
        catalog["proxy"] = Any[
            Dict("name" => "beta_back", "label" => "后向代理场", "storage" => "proxy_beta_back"),
            Dict("name" => "beta_forward", "label" => "前向代理场", "storage" => "proxy_beta_forward"),
            Dict("name" => "depol_ratio", "label" => "退偏代理场", "storage" => "proxy_depol_ratio"),
            Dict("name" => "density", "label" => "密度场", "storage" => "density"),
        ]
    end

    if haskey(families, "exact")
        exact_entries = Any[
            Dict("name" => "beta_back", "label" => "后向精确场", "storage" => "exact_beta_back"),
            Dict("name" => "beta_forward", "label" => "前向精确场", "storage" => "exact_beta_forward"),
            Dict("name" => "depol_ratio", "label" => "退偏精确场", "storage" => "exact_depol_ratio"),
        ]
        if haskey(families["exact"], "event_count")
            push!(exact_entries, Dict("name" => "event_count", "label" => "采样计数", "storage" => "exact_event_count"))
        end
        catalog["exact"] = exact_entries
    end

    return catalog
end

function build_field_catalog(bundle::Dict, ::Val{:stable_labels})
    families = bundle["families"] :: Dict{String, Any}
    catalog = Dict{String, Any}()

    if haskey(families, "proxy")
        catalog["proxy"] = Any[
            Dict("name" => "beta_back", "label" => "后向代理场", "storage" => "proxy_beta_back"),
            Dict("name" => "beta_forward", "label" => "前向代理场", "storage" => "proxy_beta_forward"),
            Dict("name" => "depol_ratio", "label" => "退偏代理场", "storage" => "proxy_depol_ratio"),
            Dict("name" => "density", "label" => "密度场", "storage" => "density"),
        ]
    end

    if haskey(families, "exact")
        exact_entries = Any[
            Dict("name" => "beta_back", "label" => "后向精确场", "storage" => "exact_beta_back"),
            Dict("name" => "beta_forward", "label" => "前向精确场", "storage" => "exact_beta_forward"),
            Dict("name" => "depol_ratio", "label" => "退偏精确场", "storage" => "exact_depol_ratio"),
        ]
        if haskey(families["exact"], "event_count")
            push!(exact_entries, Dict("name" => "event_count", "label" => "采样次数", "storage" => "exact_event_count"))
        end
        catalog["exact"] = exact_entries
    end

    return catalog
end

build_field_catalog(bundle::Dict) = build_field_catalog(bundle, Val(:stable_labels))

function summarize_field_family(fields::Dict, scatter::Dict)
    return Float32[
        Float64(get(fields, "sum_back", get(scatter, "sigma_back_ref", 0.0))),
        Float64(get(fields, "sum_forward", get(scatter, "sigma_forward_ref", 0.0))),
        Float64(get(fields, "depol_back", get(scatter, "depol_back", 0.0))),
        Float64(get(fields, "depol_forward", get(scatter, "depol_forward", 0.0))),
        Float64(get(fields, "forward_back_ratio", get(scatter, "forward_back_ratio", 0.0))),
        Float64(get(fields, "sample_count", 0.0)),
    ]
end

function ensure_field_bundle(fields_or_bundle::Dict, scatter::Dict)
    if haskey(fields_or_bundle, "families")
        return fields_or_bundle
    end
    return Dict{String, Any}(
        "families" => Dict{String, Any}("proxy" => fields_or_bundle),
        "requested_field_compute_mode" => "proxy_only",
        "effective_field_compute_mode" => "proxy_only",
        "primary_field_family" => "proxy",
    )
end

# ═══════════════════════════════════════════════════════════
# §8  可视化渲染 — 方案 D：NPZ 数据文件 + 浏览器端 JS 渲染
#
# 架构说明：
#   Julia 只负责保存二进制数据（density.npz，<5MB），
#   浏览器加载轻量 HTML 后 fetch npz，用 JSZip 解压、
#   手写 npy 解析器还原 Float32Array，交给 Plotly 渲染。
#
#   相比旧方案（JSON 内联到 HTML）：
#     文件体积  50MB+ → <5MB
#     Julia 耗时  800s → <1s
#     HTML 加载  卡顿 → 流畅
#     空白 bug    存在 → 消除（坐标由 JS 正确生成）
# ═══════════════════════════════════════════════════════════

"""
保存场数据到 density.npz。
数组均以 Float32 存储（对可视化精度足够，体积减半）。

字段：
  density       Float32[N,N,N]  归一化密度场
  beta_back     Float32[N,N,N]  后向散射场
  beta_forward  Float32[N,N,N]  前向散射场
  depol_ratio   Float32[N,N,N]  退偏场
  axis          Float32[N]      坐标轴 [L 单位，m]
  lut_back/lut_forward/lut_depol Float32[N]
  angles_deg / M11 / M12 / M33 / M34 Float32[*]
  summary       Float32[6]      [sigma_back_ref, sigma_forward_ref, depol_back, depol_forward, forward_back_ratio(proxy), 0]
"""
function save_field_npz(field::Dict, fields::Dict, scatter::Dict, output_dir::String)
    mkpath(output_dir)
    L    = field["L"]   :: Float64
    axis = field["axis"] :: Vector{Float64}
    dens = field["density_norm"] :: Array{Float64,3}
    bundle = ensure_field_bundle(fields, scatter)
    families = bundle["families"] :: Dict{String, Any}
    primary_family = String(get(bundle, "primary_field_family", first(keys(families))))
    primary_fields = families[primary_family] :: Dict

    arrays = Dict{String, Any}(
        "density"       => Float32.(dens),
        "axis"          => Float32.(axis),
        "angles_deg"    => Float32.(scatter["angles_deg"]),
        "M11"           => Float32.(scatter["M11"]),
        "M12"           => Float32.(scatter["M12"]),
        "M33"           => Float32.(scatter["M33"]),
        "M34"           => Float32.(scatter["M34"]),
        "meta"          => Float32[L, field["dim"], 0.0],
    )

    for (family, fam_any) in families
        fam = fam_any :: Dict
        prefix = "$(family)_"
        arrays[prefix * "beta_back"] = Float32.(fam["beta_back"])
        arrays[prefix * "beta_forward"] = Float32.(fam["beta_forward"])
        arrays[prefix * "depol_ratio"] = Float32.(fam["depol_ratio"])
        arrays[prefix * "lut_back"] = Float32.(get(fam, "lut_back", zeros(Float64, length(axis))))
        arrays[prefix * "lut_forward"] = Float32.(get(fam, "lut_forward", zeros(Float64, length(axis))))
        arrays[prefix * "lut_depol"] = Float32.(get(fam, "lut_depol", zeros(Float64, length(axis))))
        arrays[prefix * "summary"] = summarize_field_family(fam, scatter)

        if haskey(fam, "back_Q")
            arrays[prefix * "back_Q"] = Float32.(fam["back_Q"])
        end
        if haskey(fam, "back_U")
            arrays[prefix * "back_U"] = Float32.(fam["back_U"])
        end
        if haskey(fam, "back_V")
            arrays[prefix * "back_V"] = Float32.(fam["back_V"])
        end
        if haskey(fam, "forward_Q")
            arrays[prefix * "forward_Q"] = Float32.(fam["forward_Q"])
        end
        if haskey(fam, "forward_U")
            arrays[prefix * "forward_U"] = Float32.(fam["forward_U"])
        end
        if haskey(fam, "forward_V")
            arrays[prefix * "forward_V"] = Float32.(fam["forward_V"])
        end
        if haskey(fam, "event_count")
            arrays[prefix * "event_count"] = Float32.(fam["event_count"])
        end
    end

    # Legacy aliases keep existing tests and older HTML files readable.
    arrays["beta_back"] = Float32.(primary_fields["beta_back"])
    arrays["beta_forward"] = Float32.(primary_fields["beta_forward"])
    arrays["depol_ratio"] = Float32.(primary_fields["depol_ratio"])
    arrays["back_Q"] = Float32.(get(primary_fields, "back_Q", zeros(Float64, size(dens)...)))
    arrays["back_U"] = Float32.(get(primary_fields, "back_U", zeros(Float64, size(dens)...)))
    arrays["back_V"] = Float32.(get(primary_fields, "back_V", zeros(Float64, size(dens)...)))
    arrays["forward_Q"] = Float32.(get(primary_fields, "forward_Q", zeros(Float64, size(dens)...)))
    arrays["forward_U"] = Float32.(get(primary_fields, "forward_U", zeros(Float64, size(dens)...)))
    arrays["forward_V"] = Float32.(get(primary_fields, "forward_V", zeros(Float64, size(dens)...)))
    arrays["event_count"] = Float32.(get(primary_fields, "event_count", zeros(Float64, size(dens)...)))
    arrays["lut_back"] = Float32.(get(primary_fields, "lut_back", zeros(Float64, length(axis))))
    arrays["lut_forward"] = Float32.(get(primary_fields, "lut_forward", zeros(Float64, length(axis))))
    arrays["lut_depol"] = Float32.(get(primary_fields, "lut_depol", zeros(Float64, length(axis))))
    arrays["summary"] = summarize_field_family(primary_fields, scatter)

    npz_path = joinpath(output_dir, "density.npz")
    npzwrite(npz_path, arrays)
    return npz_path
end

"""
生成三个视角 HTML 文件。

每个 HTML 约 10KB（不含数据），通过 fetch('./density.npz') 加载数据。
浏览器端完成：JSZip 解压 → npy 解析 → 笛卡尔积坐标生成 → Plotly volume 渲染。

npy 格式解析在 JS 中手写（30行），无需额外 CDN 依赖。
JSZip 从 cdnjs 加载（CSP 白名单内）。

返回生成的文件名列表（相对路径）。
"""
function render_to_html(field::Dict, fields::Dict, scatter::Dict,
                         config::Dict, output_dir::String)
    mkpath(output_dir)

    # 保存数据文件
    save_field_npz(field, fields, scatter, output_dir)

    shape_info = get(config, "shape_type", "cylinder")
    bundle = ensure_field_bundle(fields, scatter)
    field_catalog_json = String(JSON3.write(build_field_catalog(bundle)))

    # 各视角的初始相机方向（归一化到 [-2,2] 范围，Plotly 约定）
    view_cfgs = [
        ("render_main.html",  ( 1.5,  1.8,  1.2)),
        ("render_top.html",   ( 0.0,  0.0,  2.5)),
        ("render_front.html", ( 0.0,  2.5,  0.5)),
    ]

    # ── HTML 模板（数据通过 fetch 加载，不内联）──────────────────
    html_template = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>IITM [SHAPE_INFO] — VIEW_NAME</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
  <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body { width: 100%; height: 100%; overflow: hidden; }
    body { position: relative; background: #111; color: #ccc; font-family: monospace; overscroll-behavior: none; }
    body.embed #toolbar { display: none; }
    #plot {
      position: absolute; inset: 0;
      width: 100%; height: 100%;
      min-width: 0; min-height: 0;
    }
    #toolbar {
      position: fixed; top: 8px; right: 8px; z-index: 120;
      display: flex; gap: 6px; flex-wrap: wrap;
    }
    #toolbar button {
      border: 1px solid #333; background: rgba(20,20,20,0.85); color: #ddd;
      padding: 6px 10px; border-radius: 6px; cursor: pointer; font-size: 11px;
    }
    #toolbar button.active { background: #2563eb; color: white; border-color: #3b82f6; }
    #loading {
      position: fixed; top: 50%; left: 50%;
      transform: translate(-50%, -50%);
      font-size: 14px; color: #aaa; z-index: 100;
    }
    #info {
      position: fixed; top: 8px; left: 8px;
      font-size: 11px; color: #666; pointer-events: none;
    }
  </style>
</head>
<body>
  <div id="loading">正在加载数据...</div>
  <div id="info">IITM [SHAPE_INFO]</div>
  <div id="toolbar"></div>
  <div id="plot" style="display:none"></div>

<script>
// ── 相机初始方向（由 Julia 参数化注入）──
const CAMERA_EYE = {x: EYE_X, y: EYE_Y, z: EYE_Z};
const URL_PARAMS = new URLSearchParams(window.location.search);
const EMBED_MODE = URL_PARAMS.get('embed') === '1';
const START_FAMILY = URL_PARAMS.get('family') || 'proxy';
const START_FIELD = URL_PARAMS.get('field') || 'beta_back';
const DATA_VERSION = URL_PARAMS.get('t') || String(Date.now());
const FIELD_CATALOG = FIELD_CATALOG_JSON;
if (EMBED_MODE) document.body.classList.add('embed');

// ── npy 格式解析器 ──────────────────────────────────────────
// npy 格式：魔数(6B) + 版本(2B) + header长度(2B LE) + header(ASCII) + 二进制数据
function parseNpy(buffer) {
  const view = new DataView(buffer);
  // 验证魔数 \\x93NUMPY
  if (view.getUint8(0) !== 0x93) throw new Error('不是有效的 npy 文件');
  const headerLen = view.getUint16(8, true);  // 小端
  const headerStr = new TextDecoder().decode(new Uint8Array(buffer, 10, headerLen));

  // 解析 shape，例如 "'shape': (80, 80, 80)"
  const shapeMatch = headerStr.match(/'shape'\\s*:\\s*\\(([^)]+)\\)/);
  if (!shapeMatch) throw new Error('无法解析 shape: ' + headerStr);
  const shape = shapeMatch[1].split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));

  // 解析 dtype，目前只处理 float32 和 int32
  const dtypeMatch = headerStr.match(/'descr'\\s*:\\s*'([^']+)'/);
  const dtype = dtypeMatch ? dtypeMatch[1] : '<f4';

  const dataOffset = 10 + headerLen;
  const dataBuffer = buffer.slice(dataOffset);

  let arr;
  if (dtype === '<f4' || dtype === '|f4') {
    arr = new Float32Array(dataBuffer);
  } else if (dtype === '<i4' || dtype === '|i4') {
    arr = new Int32Array(dataBuffer);
  } else if (dtype === '<f8') {
    arr = new Float64Array(dataBuffer);
  } else {
    throw new Error('不支持的 dtype: ' + dtype);
  }

  return { data: arr, shape: shape };
}

// ── 主加载与渲染逻辑 ────────────────────────────────────────
async function loadAndRender() {
  const loading = document.getElementById('loading');
  const plotDiv = document.getElementById('plot');
  const toolbar = document.getElementById('toolbar');

  try {
    // 1. fetch density.npz（相对路径，与 HTML 同目录）
    loading.textContent = '正在下载数据文件...';
    const resp = await fetch('./density.npz?t=' + encodeURIComponent(DATA_VERSION), { cache: 'no-store' });
    if (!resp.ok) throw new Error('fetch 失败: ' + resp.status);
    const npzBuf = await resp.arrayBuffer();

    // 2. JSZip 解压 npz（npz = ZIP 容器，每个数组是一个 .npy 文件）
    loading.textContent = '正在解压数据...';
    const zip = await JSZip.loadAsync(npzBuf);

    async function readArray(name) {
      const file = zip.file(name + '.npy');
      if (!file) throw new Error('找不到 ' + name + '.npy');
      const buf = await file.async('arraybuffer');
      return parseNpy(buf);
    }

    async function readOptionalArray(name) {
      const file = zip.file(name + '.npy');
      if (!file) return null;
      const buf = await file.async('arraybuffer');
      return parseNpy(buf);
    }

    const densObj = await readArray('density');
    const summaryObj = await readArray('summary');

    const density = densObj.data;   // Float32Array，长度 N³
    const defaultSummary = summaryObj.data;
    const shape   = densObj.shape;  // [N, N, N]
    const N       = shape[0];

    // 3. 生成坐标轴
    const axisObj = await readArray('axis');
    const axis    = axisObj.data;   // Float32Array，长度 N

    // 4. 展开完整笛卡尔积坐标（Plotly volume 要求所有 N³ 点都有坐标）
    loading.textContent = '正在生成坐标网格...';
    const total = N * N * N;
    const xs = new Float32Array(total);
    const ys = new Float32Array(total);
    const zs = new Float32Array(total);

    // Julia 存储顺序：density[ix, iy, iz]，列主序（Fortran order）
    // 展开时需要与 Julia 的内存布局一致
    let idx = 0;
    for (let iz = 0; iz < N; iz++)
      for (let iy = 0; iy < N; iy++)
        for (let ix = 0; ix < N; ix++) {
          xs[idx] = axis[ix];
          ys[idx] = axis[iy];
          zs[idx] = axis[iz];
          idx++;
        }

    // 5. Plotly volume 渲染
    loading.textContent = '正在渲染 3D 体积...';
    plotDiv.style.display = 'block';

    function colorscaleFor(fieldName, familyName) {
      if (fieldName === 'beta_forward') return 'Viridis';
      if (fieldName === 'depol_ratio') return 'Cividis';
      if (fieldName === 'event_count') return 'Blues';
      return familyName === 'exact' ? 'Turbo' : 'Hot';
    }

    const fieldDefs = {};
    const familySummaries = {};
    for (const [familyName, entries] of Object.entries(FIELD_CATALOG)) {
      fieldDefs[familyName] = {};
      const familySummaryObj = await readOptionalArray(familyName + '_summary');
      familySummaries[familyName] = familySummaryObj ? familySummaryObj.data : defaultSummary;
      for (const entry of entries) {
        const fieldName = entry.name;
        const storageName = entry.storage || (familyName + '_' + fieldName);
        const valuesObj = fieldName === 'density' ? densObj : await readArray(storageName);
        fieldDefs[familyName][fieldName] = {
          label: entry.label || fieldName,
          values: valuesObj.data,
          colorscale: colorscaleFor(fieldName, familyName),
          opacity: fieldName === 'depol_ratio' ? 0.18 : 0.12,
        };
      }
    }

    function fieldRange(values, fieldName, familyName) {
      let vmin = Infinity;
      let vmax = -Infinity;
      for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (!Number.isFinite(v)) continue;
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
      }
      if (!Number.isFinite(vmin) || !Number.isFinite(vmax)) return {isomin: 0, isomax: 1};
      if (fieldName === 'depol_ratio') {
        if (Math.abs(vmax - vmin) < Number.EPSILON) return {isomin: Math.max(vmin - 1e-3, 0), isomax: vmin + 1e-3};
        return {isomin: Math.max(vmin, 0), isomax: vmax};
      }
      if (vmax <= 0) return {isomin: 0, isomax: 1};
      let thresholdRatio = 0.05;
      // Exact fields are far more center-dominated than proxy fields; use a much
      // lower cutoff so off-axis weak scattering remains visible in the volume.
      if (familyName === 'exact') {
        thresholdRatio = fieldName === 'event_count' ? 0.00025 : 0.0005;
      }
      return {isomin: Math.max(vmax * thresholdRatio, Number.EPSILON), isomax: vmax};
    }

    function makeTrace(familyName, fieldName) {
      const def = fieldDefs[familyName][fieldName];
      const range = fieldRange(def.values, fieldName, familyName);
      return {
        type: 'volume',
        x: xs,
        y: ys,
        z: zs,
        value: def.values,
        isomin: range.isomin,
        isomax: range.isomax,
        opacity: def.opacity,
        surface: { count: fieldName === 'depol_ratio' ? 8 : 6 },
        colorscale: def.colorscale,
        colorbar: {
          title: { text: def.label, font: { color: '#aaa' } },
          tickfont: { color: '#aaa' },
          len: 0.6
        },
        caps: { x: { show: false }, y: { show: false }, z: { show: false } }
      };
    }

    const layout = {
      paper_bgcolor: '#111',
      scene: {
        xaxis: { title: 'X [m]', color: '#888', gridcolor: '#333', backgroundcolor: '#111' },
        yaxis: { title: 'Y [m]', color: '#888', gridcolor: '#333', backgroundcolor: '#111' },
        zaxis: { title: 'Z [m]', color: '#888', gridcolor: '#333', backgroundcolor: '#111' },
        bgcolor: '#111',
        camera: { eye: CAMERA_EYE },
        aspectmode: 'cube',
        uirevision: 'camera-lock'
      },
      uirevision: 'field-switch',
      margin: { l: 0, r: 0, b: 0, t: 0 }
    };

    let currentFamily = Object.prototype.hasOwnProperty.call(fieldDefs, START_FAMILY)
      ? START_FAMILY
      : Object.keys(fieldDefs)[0];
    let currentField = Object.prototype.hasOwnProperty.call(fieldDefs[currentFamily], START_FIELD)
      ? START_FIELD
      : Object.keys(fieldDefs[currentFamily])[0];

    function updateInfo() {
      const def = fieldDefs[currentFamily][currentField];
      const summary = familySummaries[currentFamily] || defaultSummary;
      const ratio = Number.isFinite(summary[4]) ? summary[4].toExponential(3) : '0';
      const depolB = Number.isFinite(summary[2]) ? summary[2].toFixed(4) : '0.0000';
      const depolF = Number.isFinite(summary[3]) ? summary[3].toFixed(4) : '0.0000';
      const sampleCount = Number.isFinite(summary[5]) ? summary[5].toFixed(0) : '0';
      document.getElementById('info').textContent =
        `IITM [SHAPE_INFO] | \${currentFamily} | \${def.label} | F/B=\${ratio} | depol_back=\${depolB} | depol_forward=\${depolF} | samples=\${sampleCount}`;
    }

    function setActiveButton() {
      Array.from(toolbar.querySelectorAll('button')).forEach(btn => {
        btn.classList.toggle('active', btn.dataset.family === currentFamily && btn.dataset.field === currentField);
      });
    }

    async function renderField(familyName, fieldName) {
      if (!Object.prototype.hasOwnProperty.call(fieldDefs, familyName)) return;
      if (!Object.prototype.hasOwnProperty.call(fieldDefs[familyName], fieldName)) return;
      currentFamily = familyName;
      currentField = fieldName;
      const nextUrl = new URL(window.location.href);
      nextUrl.searchParams.set('family', currentFamily);
      nextUrl.searchParams.set('field', currentField);
      window.history.replaceState(null, '', nextUrl.toString());
      updateInfo();
      setActiveButton();
      await Plotly.react('plot', [makeTrace(familyName, fieldName)], layout, {
        responsive: true,
        displaylogo: false,
        displayModeBar: !EMBED_MODE,
        scrollZoom: false,
        modeBarButtonsToRemove: ['toImage']
      });
    }

    if (!EMBED_MODE) {
      for (const [familyName, fields] of Object.entries(fieldDefs)) {
        for (const [fieldName, def] of Object.entries(fields)) {
          const btn = document.createElement('button');
          btn.textContent = familyName + ': ' + def.label;
          btn.dataset.family = familyName;
          btn.dataset.field = fieldName;
          btn.onclick = () => renderField(familyName, fieldName);
          toolbar.appendChild(btn);
        }
      }
    }

    let resizeTimer = null;
    function scheduleResize() {
      if (!plotDiv || plotDiv.style.display === 'none') return;
      if (resizeTimer) clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        try { Plotly.Plots.resize(plotDiv); } catch (_) {}
      }, 30);
    }

    window.addEventListener('resize', scheduleResize);
    if (typeof ResizeObserver !== 'undefined') {
      const ro = new ResizeObserver(() => scheduleResize());
      ro.observe(document.body);
      ro.observe(plotDiv);
    }

    window.addEventListener('message', async (event) => {
      const data = event && event.data ? event.data : null;
      if (!data || data.type !== 'iitm:set_field') return;
      const nextFamily = data.family || currentFamily;
      const nextField = data.field;
      await renderField(nextFamily, nextField);
      scheduleResize();
    });

    await renderField(currentFamily, currentField);
    requestAnimationFrame(() => scheduleResize());
    setTimeout(() => scheduleResize(), 120);

    loading.style.display = 'none';

  } catch (err) {
    loading.textContent = '渲染失败: ' + err.message;
    loading.style.color = '#f44';
    console.error(err);
  }
}

// 页面加载完成后立即开始
loadAndRender();
</script>
</body>
</html>"""

    generated = String[]
    for (fname, (ex, ey, ez)) in view_cfgs
        html = html_template
        html = replace(html, "SHAPE_INFO" => shape_info)
        html = replace(html, "VIEW_NAME"  => fname)
        html = replace(html, "EYE_X"      => string(round(ex; digits=2)))
        html = replace(html, "EYE_Y"      => string(round(ey; digits=2)))
        html = replace(html, "EYE_Z"      => string(round(ez; digits=2)))
        html = replace(html, "FIELD_CATALOG_JSON" => field_catalog_json)

        fpath = joinpath(output_dir, fname)
        open(fpath, "w") do io; write(io, html); end
        push!(generated, fname)
    end

    return generated
end

# ═══════════════════════════════════════════════════════════
# §9  嵌入式测试套件
#     运行：julia --threads auto --project=src/julia iitm_physics.jl
# ═══════════════════════════════════════════════════════════

function mc_parallel_regression_metrics(; n_photons::Int = 17_000, seed::Int = 17)
    sc = compute_scatter_params(Dict{String,Any}(
        "wavelength_m" => 1.55e-6, "m_real" => 1.311, "m_imag" => 1e-4,
        "shape_type"   => "sphere", "size_mode" => "mono",
        "radius_um"    => 0.8, "Nr" => 12, "Ntheta" => 24,
    ))
    mc = run_monte_carlo(sc, Dict{String,Any}(
        "beta_ext_surf" => 5e-4, "thickness_m" => 6.0, "scale_height_m" => 0.0,
        "n_photons" => n_photons, "seed" => seed,
    ))
    total_ratio = mc.backscatter_ratio + mc.transmit_ratio + mc.absorbed_ratio
    return Dict{String,Any}(
        "threads" => Threads.nthreads(),
        "n_photons" => n_photons,
        "seed" => seed,
        "total_ratio" => total_ratio,
        "backscatter_ratio" => mc.backscatter_ratio,
        "transmit_ratio" => mc.transmit_ratio,
        "absorbed_ratio" => mc.absorbed_ratio,
        "avg_collisions" => mc.avg_collisions,
    )
end

function mc_parallel_regression_metrics_json(; n_photons::Int = 17_000, seed::Int = 17)
    return JSON3.write(mc_parallel_regression_metrics(; n_photons = n_photons, seed = seed))
end

function read_mc_parallel_regression_metrics_subprocess(nthreads::Int;
                                                        n_photons::Int = 17_000,
                                                        seed::Int = 17)
    julia_bin = joinpath(Sys.BINDIR, Base.julia_exename())
    script_path = replace(abspath(@__FILE__), "\\" => "\\\\")
    expr = "include(raw\"$script_path\"); " *
           "print(mc_parallel_regression_metrics_json(n_photons=$n_photons, seed=$seed))"
    output = read(`$(julia_bin) --threads=$(nthreads) --project=$(@__DIR__) -e $expr`, String)
    return JSON3.read(output)
end

function run_tests(; verbose::Bool = true)
    n_pass = Ref(0); n_fail = Ref(0)

    function hdr(name); println("\n" * "─"^56); println("  TEST: $name"); println("─"^56); end

    function chk(cond::Bool, msg::String)
        if cond
            verbose && println("  ✓  $msg")
            n_pass[] += 1
        else
            println("  ✗  FAIL: $msg")
            n_fail[] += 1
        end
    end

    # ─── T1 物理工具 ─────────────────────────────────────
    hdr("T1  物理工具函数")
    b = visibility_to_beta_ext(1.0, 550.0)
    chk(isapprox(b, 3.912e-3; rtol=1e-4), "beta_ext(1km,550nm) ≈ 3.912e-3")
    chk(visibility_to_beta_ext(1.0,1550.0) < b, "Angström: 1550nm < 550nm")
    chk(isapprox(trapz([0.0,1.0,0.0],[0.0,1.0,2.0]), 1.0; atol=1e-10), "trapz 面积=1")
    cum = cumtrapz([0.0,1.0,0.0],[0.0,1.0,2.0])
    chk(cum[1]==0.0 && isapprox(cum[end],1.0;atol=1e-10), "cumtrapz 首零末一")
    pdf = lognormal_pdf([0.5,1.0,2.0], 1.0, 0.3)
    chk(all(pdf .>= 0) && pdf[2] > pdf[1], "lognormal_pdf 正值且峰在中值")

    # ─── T2 粒子形状工厂 ─────────────────────────────────
    hdr("T2  粒子形状工厂（三种形状）")
    m_t = 1.5 + 0.01im
    for (st, ar) in [("cylinder",1.2),("spheroid",0.7),("sphere",1.0)]
        try
            make_particle(st, 0.5, m_t; axis_ratio=ar)
            chk(true, "make_particle($st, ar=$ar) 构造成功")
        catch e
            chk(false, "make_particle($st) 抛出: $e")
        end
    end

    # ─── T3 nmax_override ────────────────────────────────
    hdr("T3  nmax_override 与自动估算")
    auto_n = resolve_nmax(0.3, 1.55)
    over_n = resolve_nmax(0.3, 1.55; nmax_override=6)
    chk(over_n == 6,  "nmax_override=6 → resolve_nmax=6")
    chk(auto_n >= 4,  "自动 nmax ≥ 4 (值=$auto_n)")

    # ─── T4 单粒子截面 (sphere) ──────────────────────────
    hdr("T4  单粒子截面合理性 (sphere)")
    wl_m = 1.55e-6
    try
        res = single_particle_iitm(0.5, 1.311+1e-4im, wl_m;
                                    shape_type="sphere", Nr=25, Ntheta=50)
        geo = pi * (0.5e-6)^2
        chk(res.sigma_ext > 0,              "sigma_ext > 0")
        chk(res.sigma_sca <= res.sigma_ext * 1.001, "sigma_sca ≤ sigma_ext")
        chk(-1.0 <= res.g <= 1.0,           "g ∈ [-1,1]")
        chk(0.0 < res.sigma_ext/geo < 8.0,  "Qext ∈ (0,8)")
        verbose && println("    sigma_ext=$(round(res.sigma_ext*1e12,digits=4)) μm²  g=$(round(res.g,digits=4))")
    catch e; chk(false, "单粒子异常: $e"); end

    # ─── T5 cylinder vs spheroid ─────────────────────────
    hdr("T5  cylinder vs spheroid 截面量级一致")
    try
        rc = single_particle_iitm(0.5, 1.311+1e-4im, wl_m;
                                   shape_type="cylinder", Nr=20, Ntheta=40)
        rs = single_particle_iitm(0.5, 1.311+1e-4im, wl_m;
                                   shape_type="spheroid", Nr=20, Ntheta=40)
        ratio = rc.sigma_ext / max(rs.sigma_ext, 1e-40)
        chk(0.1 < ratio < 10.0, "cylinder/spheroid 截面比 ∈ (0.1,10)  ratio=$(round(ratio,digits=2))")
    catch e; chk(false, "形状对比失败: $e"); end

    # ─── T6 粒径分布归一化 ───────────────────────────────
    hdr("T6  对数正态粒径分布归一化")
    r_arr = exp.(range(log(0.1), log(10.0); length=80))
    pdf2  = lognormal_pdf(r_arr, 1.0, 0.4)
    chk(isapprox(trapz(pdf2, r_arr), 1.0; atol=0.01), "∫pdf dr ≈ 1.0")

    # ─── T7 相函数归一化 & CDF ───────────────────────────
    hdr("T7  相函数归一化 & CDF 性质")
    try
        sc = compute_scatter_params(Dict{String,Any}(
            "wavelength_m" => 1.55e-6, "m_real" => 1.311, "m_imag" => 1e-4,
            "shape_type"   => "sphere", "size_mode" => "mono",
            "radius_um"    => 0.5, "Nr" => 20, "Ntheta" => 40,
        ))
        M11 = sc["M11"]; tr = deg2rad.(sc["angles_deg"])
        nrm = trapz(M11 .* sin.(tr), tr)
        chk(isapprox(nrm, 2.0; rtol=0.02), "∫M11 sinθ dθ ≈ 2.0")
        cdf2 = sc["cdf_grid"]
        chk(cdf2[1] >= 0.0,                      "CDF[1] ≥ 0")
        chk(isapprox(cdf2[end], 1.0; atol=1e-6), "CDF[end] ≈ 1")
        chk(all(diff(cdf2) .>= -1e-10),           "CDF 单调不减")
        chk(0.0 < sc["omega0"] <= 1.0,            "omega0 ∈ (0,1]")
    catch e; chk(false, "相函数测试失败: $e"); end

    # ─── T8 MC 能量守恒（无吸收）────────────────────────
    hdr("T8  MC 能量守恒（无吸收）")
    try
        sc = compute_scatter_params(Dict{String,Any}(
            "wavelength_m" => 1.55e-6, "m_real" => 1.311, "m_imag" => 0.0,
            "shape_type"   => "sphere", "size_mode" => "mono",
            "radius_um"    => 2.0, "Nr" => 20, "Ntheta" => 40,
        ))
        mc = run_monte_carlo(sc, Dict{String,Any}(
            "beta_ext_surf" => 1e-3, "thickness_m" => 20.0,
            "scale_height_m" => 0.0, "n_photons" => 8000, "seed" => 42,
        ))
        tot = mc.backscatter_ratio + mc.transmit_ratio + mc.absorbed_ratio
        chk(isapprox(tot, 1.0; atol=0.02), "R_back+R_trans+R_abs=$(round(tot,digits=4)) ≈ 1")
        chk(mc.absorbed_ratio < 0.02, "无吸收时 R_abs < 2%")
    catch e; chk(false, "MC 无吸收测试失败: $e"); end

    # ─── T9 MC 能量守恒（有吸收）────────────────────────
    hdr("T9  MC 能量守恒（有吸收）")
    try
        sc = compute_scatter_params(Dict{String,Any}(
            "wavelength_m" => 1.55e-6, "m_real" => 1.311, "m_imag" => 0.01,
            "shape_type"   => "sphere", "size_mode" => "mono",
            "radius_um"    => 2.0, "Nr" => 20, "Ntheta" => 40,
        ))
        mc = run_monte_carlo(sc, Dict{String,Any}(
            "beta_ext_surf" => 1e-3, "thickness_m" => 20.0,
            "scale_height_m" => 0.0, "n_photons" => 6000, "seed" => 99,
        ))
        tot = mc.backscatter_ratio + mc.transmit_ratio + mc.absorbed_ratio
        chk(isapprox(tot, 1.0; atol=0.02), "有吸收能量守恒=$(round(tot,digits=4))")
        chk(mc.absorbed_ratio > 0.0, "有吸收时 R_abs > 0")
    catch e; chk(false, "MC 有吸收测试失败: $e"); end

    # ─── T10 密度场生成 ──────────────────────────────────
    hdr("T10 密度场值域与尺寸")
    fld = generate_field(Dict{String,Any}(
        "grid_dim"=>16,"L_size"=>8.0,
        "cloud_center_z"=>4.0,"cloud_thickness"=>3.0,"turbulence_scale"=>4.0,
    ))
    dn = fld["density_norm"]
    chk(minimum(dn) >= 0.0,        "density ≥ 0")
    chk(maximum(dn) <= 1.0+1e-9,   "density ≤ 1")
    chk(size(dn) == (16,16,16),     "shape = (16,16,16)")

    # ─── T11 compute_scatter_fields（多场接口）───────────
    hdr("T11 compute_scatter_fields（后向/前向/退偏）")
    try
        sc11 = compute_scatter_params(Dict{String,Any}(
            "wavelength_m" => 1.55e-6, "m_real" => 1.311, "m_imag" => 1e-4,
            "shape_type"   => "sphere", "size_mode" => "mono",
            "radius_um"    => 1.0, "Nr" => 15, "Ntheta" => 30,
        ))
        cfg11 = Dict{String,Any}("r_bottom"=>0.5,"r_top"=>2.0)
        mc11 = run_monte_carlo(sc11, Dict{String,Any}(
            "beta_ext_surf" => 5e-4, "thickness_m" => 8.0, "scale_height_m" => 0.0,
            "n_photons" => 2500, "seed" => 11, "collect_voxel_fields" => true,
            "density_grid" => fld["density_norm"], "field_axis" => fld["axis"],
            "field_xy_centered" => true, "density_sampling" => "nearest",
            "field_forward_half_angle_deg" => 90.0, "field_back_half_angle_deg" => 90.0,
        ))
        fields11 = compute_scatter_fields(fld, sc11, cfg11, mc11)
        chk(size(fields11["beta_back"]) == (16,16,16), "beta_back shape 正确")
        chk(size(fields11["beta_forward"]) == (16,16,16), "beta_forward shape 正确")
        chk(size(fields11["depol_ratio"]) == (16,16,16), "depol_ratio shape 正确")
        chk(minimum(fields11["beta_back"]) >= 0.0, "beta_back ≥ 0")
        chk(minimum(fields11["beta_forward"]) >= 0.0, "beta_forward ≥ 0")
        chk(length(fields11["lut_back"]) == 16, "lut_back 长度 = 16")
        chk(length(fields11["lut_forward"]) == 16, "lut_forward 长度 = 16")
        chk(length(fields11["lut_depol"]) == 16, "lut_depol 长度 = 16")
    catch e; chk(false, "compute_scatter_fields 失败: $e"); end

    # ─── T12 save_field_npz ──────────────────────────────
    hdr("T12 NPZ 文件读写（save_field_npz）")
    try
        tmpdir = mktempdir()
        fld12  = generate_field(Dict{String,Any}(
            "grid_dim"=>8, "L_size"=>4.0,
            "cloud_center_z"=>2.0,"cloud_thickness"=>2.0,"turbulence_scale"=>4.0,
        ))
        sc12 = compute_scatter_params(Dict{String,Any}(
            "wavelength_m" => 1.55e-6, "m_real" => 1.311, "m_imag" => 1e-4,
            "shape_type"   => "sphere", "size_mode" => "mono",
            "radius_um"    => 0.8, "Nr" => 12, "Ntheta" => 24,
        ))
        mc12 = run_monte_carlo(sc12, Dict{String,Any}(
            "beta_ext_surf" => 5e-4, "thickness_m" => 4.0, "scale_height_m" => 0.0,
            "n_photons" => 1800, "seed" => 12, "collect_voxel_fields" => true,
            "density_grid" => fld12["density_norm"], "field_axis" => fld12["axis"],
            "field_xy_centered" => true, "density_sampling" => "nearest",
            "field_forward_half_angle_deg" => 90.0, "field_back_half_angle_deg" => 90.0,
        ))
        fields12 = compute_scatter_fields(fld12, sc12, Dict{String,Any}("r_bottom"=>0.5,"r_top"=>2.0), mc12)
        npz_p  = save_field_npz(fld12, fields12, sc12, tmpdir)
        chk(isfile(npz_p), "density.npz 文件已生成")
        data12 = npzread(npz_p)
        chk(haskey(data12, "density"),  "npz 包含 density 字段")
        chk(haskey(data12, "beta_back"), "npz 包含 beta_back 字段")
        chk(haskey(data12, "beta_forward"), "npz 包含 beta_forward 字段")
        chk(haskey(data12, "depol_ratio"), "npz 包含 depol_ratio 字段")
        chk(haskey(data12, "axis"),     "npz 包含 axis 字段")
        chk(haskey(data12, "lut_back"), "npz 包含 lut_back 字段")
        chk(haskey(data12, "M11"), "npz 包含 M11 字段")
        chk(haskey(data12, "meta"),     "npz 包含 meta 字段")
        chk(size(data12["density"]) == (8,8,8), "density shape = (8,8,8)")
        chk(data12["meta"][1] ≈ 4.0f0, "meta[1] = L = 4.0")
        rm(tmpdir; recursive=true)
    catch e; chk(false, "NPZ 读写失败: $e"); end

    # ─── T13 render_to_html（生成文件检查）──────────────
    hdr("T13 render_to_html（HTML 文件生成）")
    try
        tmpdir2 = mktempdir()
        fld13   = generate_field(Dict{String,Any}(
            "grid_dim"=>8,"L_size"=>4.0,
            "cloud_center_z"=>2.0,"cloud_thickness"=>2.0,"turbulence_scale"=>4.0,
        ))
        sc13 = compute_scatter_params(Dict{String,Any}(
            "wavelength_m" => 1.55e-6, "m_real" => 1.311, "m_imag" => 1e-4,
            "shape_type"   => "sphere", "size_mode" => "mono",
            "radius_um"    => 0.8, "Nr" => 12, "Ntheta" => 24,
        ))
        mc13 = run_monte_carlo(sc13, Dict{String,Any}(
            "beta_ext_surf" => 5e-4, "thickness_m" => 4.0, "scale_height_m" => 0.0,
            "n_photons" => 1800, "seed" => 13, "collect_voxel_fields" => true,
            "density_grid" => fld13["density_norm"], "field_axis" => fld13["axis"],
            "field_xy_centered" => true, "density_sampling" => "nearest",
            "field_forward_half_angle_deg" => 90.0, "field_back_half_angle_deg" => 90.0,
        ))
        fields13 = compute_scatter_fields(fld13, sc13, Dict{String,Any}("r_bottom"=>0.5,"r_top"=>2.0), mc13)
        cfg13   = Dict{String,Any}("shape_type"=>"sphere")
        files13 = render_to_html(fld13, fields13, sc13, cfg13, tmpdir2)
        chk(length(files13) == 3,                      "生成 3 个 HTML 文件")
        chk(isfile(joinpath(tmpdir2,"density.npz")),   "density.npz 存在")
        chk(isfile(joinpath(tmpdir2,"render_main.html")), "render_main.html 存在")
        # 检查 HTML 包含关键 JS 字符串
        html_content = read(joinpath(tmpdir2,"render_main.html"), String)
        chk(occursin("JSZip",      html_content), "HTML 引用 JSZip")
        chk(occursin("parseNpy",   html_content), "HTML 包含 npy 解析器")
        chk(occursin("density.npz",html_content), "HTML 引用 density.npz")
        chk(occursin("beta_forward", html_content), "HTML 支持前向场切换")
        chk(occursin("depol_ratio", html_content), "HTML 支持退偏场切换")
        chk(occursin("volume",     html_content), "HTML 使用 Plotly volume")
        rm(tmpdir2; recursive=true)
    catch e; chk(false, "render_to_html 测试失败: $e"); end

    # ─── T14 完整流程冒烟 ────────────────────────────────
    hdr("T14 完整流程冒烟 (cylinder lognormal)")
    try
        sc14 = compute_scatter_params(Dict{String,Any}(
            "wavelength_m"     => 1.55e-6,
            "m_real"           => 1.311,
            "m_imag"           => 1e-4,
            "shape_type"       => "cylinder",
            "size_mode"        => "lognormal",
            "radius_um"        => 1.0,
            "median_radius_um" => 1.0,
            "sigma_ln"         => 0.3,
            "n_radii"          => 4,
            "axis_ratio"       => 1.0,
            "Nr"               => 18,
            "Ntheta"           => 36,
        ))
        chk(sc14["sigma_ext"] > 0, "smoke: sigma_ext > 0")
        chk(sc14["omega0"]    > 0, "smoke: omega0 > 0")

        mc14 = run_monte_carlo(sc14, Dict{String,Any}(
            "beta_ext_surf"  => 5e-4, "thickness_m" => 10.0,
            "scale_height_m" => 0.0,  "n_photons"   => 3000, "seed" => 7,
        ))
        tot14 = mc14.backscatter_ratio + mc14.transmit_ratio + mc14.absorbed_ratio
        chk(isapprox(tot14,1.0;atol=0.03), "smoke: 能量守恒 ≈ 1")
        verbose && println("    sigma_ext=$(round(sc14["sigma_ext"]*1e12,digits=3)) μm²  " *
                           "R_back=$(round(mc14.backscatter_ratio,digits=4))")
    catch e; chk(false, "完整流程异常: $e"); end

    # ─── 汇总 ────────────────────────────────────────────
    hdr("T15 3D density-aware Monte Carlo (nearest / trilinear)")
    try
        fld15 = generate_field(Dict{String,Any}(
            "grid_dim"=>12, "L_size"=>6.0,
            "cloud_center_z"=>3.0, "cloud_thickness"=>2.5, "turbulence_scale"=>3.0,
        ))
        sc15 = compute_scatter_params(Dict{String,Any}(
            "wavelength_m" => 1.55e-6, "m_real" => 1.311, "m_imag" => 1e-4,
            "shape_type"   => "sphere", "size_mode" => "mono",
            "radius_um"    => 0.8, "Nr" => 12, "Ntheta" => 24,
        ))
        mc15n = run_monte_carlo(sc15, Dict{String,Any}(
            "beta_ext_surf" => 5e-4, "thickness_m" => 6.0, "scale_height_m" => 0.0,
            "n_photons" => 2500, "seed" => 17,
            "density_grid" => fld15["density_norm"], "field_axis" => fld15["axis"],
            "field_xy_centered" => true, "density_sampling" => "nearest",
        ))
        tot15n = mc15n.backscatter_ratio + mc15n.transmit_ratio + mc15n.absorbed_ratio
        chk(isapprox(tot15n,1.0;atol=0.04), "3D nearest: energy closes")

        mc15t = run_monte_carlo(sc15, Dict{String,Any}(
            "beta_ext_surf" => 5e-4, "thickness_m" => 6.0, "scale_height_m" => 0.0,
            "n_photons" => 2500, "seed" => 17,
            "density_grid" => fld15["density_norm"], "field_axis" => fld15["axis"],
            "field_xy_centered" => true, "density_sampling" => "trilinear",
        ))
        tot15t = mc15t.backscatter_ratio + mc15t.transmit_ratio + mc15t.absorbed_ratio
        chk(isapprox(tot15t,1.0;atol=0.04), "3D trilinear: energy closes")
        chk(mc15n.transmit_ratio >= 0.0, "3D nearest: valid result")
        chk(mc15t.transmit_ratio >= 0.0, "3D trilinear: valid result")
    catch e; chk(false, "3D density MC failed: $e"); end

    println("\n" * "═"^56)
    hdr("T16 dual-field bundle / metadata consistency")
    try
        fld16 = generate_field(Dict{String,Any}(
            "grid_dim"=>10, "L_size"=>5.0,
            "cloud_center_z"=>2.5, "cloud_thickness"=>2.0, "turbulence_scale"=>3.0,
        ))
        sc16 = compute_scatter_params(Dict{String,Any}(
            "wavelength_m" => 1.55e-6, "m_real" => 1.311, "m_imag" => 1e-4,
            "shape_type"   => "sphere", "size_mode" => "mono",
            "radius_um"    => 0.9, "Nr" => 12, "Ntheta" => 24,
        ))
        mc16 = run_monte_carlo(sc16, Dict{String,Any}(
            "beta_ext_surf" => 5e-2, "thickness_m" => 5.0, "scale_height_m" => 0.0,
            "n_photons" => 2200, "seed" => 23, "collect_voxel_fields" => true,
            "density_grid" => fld16["density_norm"], "field_axis" => fld16["axis"],
            "field_xy_centered" => true, "density_sampling" => "nearest",
            "field_forward_half_angle_deg" => 90.0, "field_back_half_angle_deg" => 90.0,
        ))
        bundle16 = build_field_bundle(fld16, sc16, Dict{String,Any}(
            "field_compute_mode" => "both", "r_bottom" => 0.5, "r_top" => 2.0,
        ), mc16)
        chk(bundle16["effective_field_compute_mode"] == "both", "bundle effective mode = both")
        chk(haskey(bundle16["families"], "proxy"), "bundle contains proxy family")
        chk(haskey(bundle16["families"], "exact"), "bundle contains exact family")
        cat16 = build_field_catalog(bundle16)
        chk(haskey(cat16, "proxy") && haskey(cat16, "exact"), "catalog contains proxy/exact")
        tmpdir16 = mktempdir()
        npz16 = save_field_npz(fld16, bundle16, sc16, tmpdir16)
        data16 = npzread(npz16)
        chk(haskey(data16, "proxy_beta_back"), "npz contains proxy_beta_back")
        chk(haskey(data16, "exact_beta_back"), "npz contains exact_beta_back")
        chk(haskey(data16, "exact_event_count"), "npz contains exact_event_count")
        chk(sum(Float64.(data16["exact_event_count"])) > 0.0, "exact_event_count has samples")
        rm(tmpdir16; recursive=true)
    catch e; chk(false, "dual-field bundle failed: $e"); end

    println("\n" * "═"^56)
    hdr("T17 large-photon parallel MC regression")
    try
        metrics17a = mc_parallel_regression_metrics(; n_photons = 17_000, seed = 17)
        metrics17b = mc_parallel_regression_metrics(; n_photons = 17_000, seed = 17)
        total17 = Float64(metrics17a["total_ratio"])
        chk(isapprox(total17, 1.0; atol = 1e-12),
            "large MC energy closes: total=$(round(total17, digits=6))")
        chk(metrics17a["n_photons"] > 16_384,
            "large MC crosses fixed chunk threshold")
        chk(isapprox(Float64(metrics17a["transmit_ratio"]),
                     Float64(metrics17b["transmit_ratio"]); atol = 0.0),
            "same-thread repeated run is deterministic")
        chk(isapprox(Float64(metrics17a["avg_collisions"]),
                     Float64(metrics17b["avg_collisions"]); atol = 0.0),
            "same-thread collision statistic is deterministic")
        if Threads.nthreads() > 1
            metrics17s = read_mc_parallel_regression_metrics_subprocess(1;
                n_photons = 17_000, seed = 17)
            chk(isapprox(Float64(metrics17a["total_ratio"]),
                         Float64(metrics17s["total_ratio"]); atol = 0.0),
                "multi-thread total equals 1-thread baseline")
            chk(isapprox(Float64(metrics17a["transmit_ratio"]),
                         Float64(metrics17s["transmit_ratio"]); atol = 0.0),
                "multi-thread transmit equals 1-thread baseline")
            chk(isapprox(Float64(metrics17a["backscatter_ratio"]),
                         Float64(metrics17s["backscatter_ratio"]); atol = 0.0),
                "multi-thread backscatter equals 1-thread baseline")
            chk(isapprox(Float64(metrics17a["avg_collisions"]),
                         Float64(metrics17s["avg_collisions"]); atol = 0.0),
                "multi-thread collisions equal 1-thread baseline")
        else
            chk(true, "single-thread run skips subprocess thread baseline")
        end
    catch e; chk(false, "large-photon parallel MC regression failed: $e"); end

    total = n_pass[] + n_fail[]
    println("  结果：$(n_pass[]) / $total 通过" *
            (n_fail[] > 0 ? "  [$(n_fail[]) 失败]" : "  [全部通过 ✓]"))
    println("═"^56)
    return n_fail[] == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("iitm_physics.jl v3 — 独立测试模式")
    println("Julia $(VERSION)  线程数: $(Threads.nthreads())")
    ok = run_tests(verbose = true)
    exit(ok ? 0 : 1)
end
