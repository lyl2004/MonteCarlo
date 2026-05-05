#!/usr/bin/env julia

using NPZ

root = dirname(dirname(abspath(@__FILE__)))
include(joinpath(root, "src", "julia", "iitm_physics.jl"))

field = generate_field(Dict{String,Any}(
    "grid_dim" => 5,
    "L_size" => 4.0,
    "cloud_center_z" => 2.0,
    "cloud_thickness" => 8.0,
    "turbulence_scale" => 1000.0,
))

scatter = compute_scatter_params(Dict{String,Any}(
    "wavelength_m" => 1.55e-6,
    "m_real" => 1.311,
    "m_imag" => 1e-4,
    "shape_type" => "sphere",
    "size_mode" => "mono",
    "radius_um" => 0.5,
    "Nr" => 8,
    "Ntheta" => 16,
    "n_radii" => 1,
))

mc = run_monte_carlo(scatter, Dict{String,Any}(
    "beta_ext_surf" => 0.5,
    "thickness_m" => 4.0,
    "scale_height_m" => 0.0,
    "n_photons" => 80,
    "seed" => 20260429,
    "collect_voxel_fields" => true,
    "collect_lidar_observation" => true,
    "density_grid" => field["density_norm"],
    "field_axis" => field["axis"],
    "field_xy_centered" => true,
    "density_sampling" => "nearest",
    "field_forward_half_angle_deg" => 90.0,
    "field_back_half_angle_deg" => 90.0,
    "field_quadrature_polar" => 1,
    "field_quadrature_azimuth" => 2,
    "range_bin_width_m" => 0.5,
    "range_max_m" => 4.0,
    "receiver_overlap_min" => 1.0,
    "receiver_overlap_full_range_m" => 0.0,
))

@assert mc.lidar_observation !== nothing
obs = mc.lidar_observation
@assert length(obs.range_bins_m) == 8
@assert length(obs.echo_I) == 8
@assert all(isfinite, obs.echo_I)
@assert all(x -> x >= 0.0, obs.echo_power)
@assert all(x -> x >= 0.0, obs.echo_weight_sq_sum)
@assert all(x -> x >= 0.0, obs.echo_power_variance_est)
@assert all(x -> x >= 0.0, obs.echo_power_ci_low)
@assert all(obs.echo_power_ci_high .>= obs.echo_power_ci_low)
@assert all(x -> 0.0 <= x <= 1.0, obs.echo_depol)
@assert obs.receiver_model["receiver_mode"] == "backscatter"

bundle = build_field_bundle(field, scatter, Dict{String,Any}(
    "field_compute_mode" => "both",
), mc)
@assert haskey(bundle, "lidar_observation")

tmpdir = mktempdir()
try
    npz_path = save_field_npz(field, bundle, scatter, tmpdir)
    data = npzread(npz_path)
    @assert haskey(data, "range_bins_m")
    @assert haskey(data, "echo_I")
    @assert haskey(data, "echo_depol")
    @assert haskey(data, "echo_weight_sq_sum")
    @assert haskey(data, "echo_power_variance_est")
    @assert haskey(data, "echo_power_ci_low")
    @assert haskey(data, "echo_power_ci_high")
    @assert haskey(data, "receiver_model_json_utf8")
finally
    rm(tmpdir; recursive=true, force=true)
end

println("julia_lidar_contract_ok")
