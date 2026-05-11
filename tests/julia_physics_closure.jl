#!/usr/bin/env julia
"""
阶段一物理闭合测试 — Julia/IITM 端

WF1: 能量守恒验证（5 场景）
WF2: P(R) 解析回归（简化版，验证趋势）
"""

using Printf

root = dirname(dirname(abspath(@__FILE__)))
include(joinpath(root, "src", "julia", "iitm_physics.jl"))

# =============================================================================
# WF1: 能量守恒测试
# =============================================================================

println("=== WF1: Energy Conservation Tests ===\n")

# 场景参数：(beta_ext_target, m_imag, label)
scenarios = [
    (0.002, 0.0,   "EC-1 低τ 无吸收"),
    (0.050, 0.0,   "EC-2 中τ 无吸收"),
    (0.300, 0.0,   "EC-3 高τ 无吸收"),
    (0.002, 5e-4,  "EC-4 低τ 有吸收"),
    (0.050, 5e-4,  "EC-5 中τ 有吸收"),
]

TOLERANCE = 1e-10
L_SIZE = 20.0
N_PHOTONS = 200_000

function run_energy_conservation_test(beta_ext_target::Float64, m_imag::Float64, label::String)
    # 通过 visibility 反算 beta_ext
    # beta_ext(1550nm) = visibility_to_beta_ext(V, 1550, q=1.3)
    # 反推：V = 3.912 * (550/1550)^1.3 / (beta_ext * 1000)
    angstrom_factor = (550.0 / 1550.0)^1.3
    visibility_km = (3.912 * angstrom_factor) / (beta_ext_target * 1000.0)

    beta_ext_actual = visibility_to_beta_ext(visibility_km, 1550.0, angstrom_q=1.3)

    # 生成均匀密度场（turbulence_scale 极大）
    field = generate_field(Dict{String,Any}(
        "grid_dim" => 8,
        "L_size" => L_SIZE,
        "cloud_center_z" => L_SIZE / 2.0,
        "cloud_thickness" => L_SIZE * 10.0,
        "turbulence_scale" => 10000.0,
    ))

    # 单分散球形粒子
    scatter = compute_scatter_params(Dict{String,Any}(
        "wavelength_m" => 1.55e-6,
        "m_real" => 1.311,
        "m_imag" => m_imag,
        "shape_type" => "sphere",
        "size_mode" => "mono",
        "radius_um" => 0.4,
        "Nr" => 8,
        "Ntheta" => 16,
        "n_radii" => 1,
    ))

    # Monte Carlo（proxy_only 快速路径）
    mc = run_monte_carlo(scatter, Dict{String,Any}(
        "beta_ext_surf" => beta_ext_actual,
        "thickness_m" => L_SIZE,
        "scale_height_m" => 0.0,
        "n_photons" => N_PHOTONS,
        "seed" => 42,
        "collect_voxel_fields" => false,
        "collect_lidar_observation" => false,
        "density_grid" => field["density_norm"],
        "field_axis" => field["axis"],
        "field_xy_centered" => true,
        "density_sampling" => "nearest",
    ))

    R_back = mc.backscatter_ratio
    R_trans = mc.transmit_ratio
    R_abs = mc.absorbed_ratio

    # 验证非负性
    @assert R_back >= 0.0 "$label: R_back < 0"
    @assert R_trans >= 0.0 "$label: R_trans < 0"
    @assert R_abs >= 0.0 "$label: R_abs < 0"

    # 验证守恒
    total = R_back + R_trans + R_abs
    deviation = abs(total - 1.0)
    @assert deviation < TOLERANCE "$label: |sum-1| = $(deviation) >= $(TOLERANCE)"

    # 无吸收场景：R_abs 必须为零
    if m_imag == 0.0
        @assert R_abs == 0.0 "$label: 无吸收场景 R_abs=$(R_abs) 应为 0"
    else
        @assert R_abs > 0.0 "$label: 有吸收场景 R_abs 应 > 0"
    end

    @printf("  ✓ %s: R_back=%.6f, R_trans=%.6f, R_abs=%.6f, sum=%.10f, |Δ|=%.2e\n",
            label, R_back, R_trans, R_abs, total, deviation)
end

for (beta_ext, m_imag, label) in scenarios
    run_energy_conservation_test(beta_ext, m_imag, label)
end

println("\n✓ WF1 通过：5/5 场景能量守恒偏差 < 1e-10\n")

# =============================================================================
# WF2: P(R) 解析回归（简化版）
# =============================================================================

println("=== WF2: P(R) Analytical Regression (Simplified) ===\n")

# PR-BASE 场景
VISIBILITY_KM = 0.05  # β_ext ≈ 0.0203 m⁻¹，τ ≈ 0.41
PHOTONS_PR = 1_000_000

beta_ext_pr = visibility_to_beta_ext(VISIBILITY_KM, 1550.0, angstrom_q=1.3)
@printf("  beta_ext = %.6f m^-1, tau = %.4f\n", beta_ext_pr, beta_ext_pr * L_SIZE)

field_pr = generate_field(Dict{String,Any}(
    "grid_dim" => 16,
    "L_size" => L_SIZE,
    "cloud_center_z" => L_SIZE / 2.0,
    "cloud_thickness" => L_SIZE * 10.0,
    "turbulence_scale" => 10000.0,
))

# 强制均匀密度
field_pr["density_norm"] .= 1.0

scatter_pr = compute_scatter_params(Dict{String,Any}(
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

mc_pr = run_monte_carlo(scatter_pr, Dict{String,Any}(
    "beta_ext_surf" => beta_ext_pr,
    "thickness_m" => L_SIZE,
    "scale_height_m" => 0.0,
    "n_photons" => PHOTONS_PR,
    "seed" => 123,
    "collect_voxel_fields" => true,
    "collect_lidar_observation" => true,
    "density_grid" => field_pr["density_norm"],
    "field_axis" => field_pr["axis"],
    "field_xy_centered" => true,
    "density_sampling" => "nearest",
    "field_back_half_angle_deg" => 5.0,
    "field_forward_half_angle_deg" => 5.0,
    "field_quadrature_polar" => 1,
    "field_quadrature_azimuth" => 1,
    "range_bin_width_m" => 0.5,
    "range_max_m" => L_SIZE - 2.0,
    "receiver_overlap_min" => 1.0,
    "receiver_overlap_full_range_m" => 0.0,
))

@assert mc_pr.lidar_observation !== nothing "lidar_observation 为空"
obs = mc_pr.lidar_observation

ranges = obs.range_bins_m
power = obs.echo_power
counts = obs.echo_event_count

# 有效 bin 掩码
R_MIN = 1.0
R_MAX = 16.0
MIN_EVENTS = 30
valid_mask = (ranges .> R_MIN) .& (ranges .< R_MAX) .& (power .> 0.0) .& (counts .>= MIN_EVENTS)
n_valid = sum(valid_mask)

@assert n_valid >= 10 "有效 bin 数 $(n_valid) < 10"
@printf("  有效 bin 数: %d\n", n_valid)

# 简化验证：检查 echo_power 随距离递减
r_valid = ranges[valid_mask]
p_valid = power[valid_mask]

# 前半段与后半段功率比应 > 1（递减趋势）
mid_idx = div(length(r_valid), 2)
if mid_idx >= 2
    mean_first_half = sum(p_valid[1:mid_idx]) / mid_idx
    mean_second_half = sum(p_valid[mid_idx+1:end]) / (length(p_valid) - mid_idx)
    ratio = mean_first_half / mean_second_half
    @assert ratio > 1.0 "前半段/后半段功率比 $(ratio) 应 > 1（递减趋势）"
    @printf("  前半段/后半段功率比: %.2f (递减趋势确认)\n", ratio)
end

# 检查 linear_depol_ratio 定义正确性
parallel = obs.echo_parallel_power
perpendicular = obs.echo_perpendicular_power
depol = obs.linear_depol_ratio

for i in eachindex(depol)
    if parallel[i] > 1e-30
        expected = perpendicular[i] / parallel[i]
        @assert isapprox(depol[i], expected, rtol=1e-10) "linear_depol_ratio 定义错误"
    else
        @assert depol[i] == 0.0 "零平行功率时 depol 应为 0"
    end
end

println("  ✓ linear_depol_ratio 定义正确")
println("\n✓ WF2 通过：P(R) 递减趋势确认，偏振字段定义正确\n")

# =============================================================================
# 总结
# =============================================================================

println("=" ^ 60)
println("Julia 物理闭合测试全部通过")
println("  WF1: 5/5 能量守恒场景")
println("  WF2: P(R) 趋势验证 + 偏振字段合约")
println("=" ^ 60)
