#!/usr/bin/env julia

root = dirname(dirname(abspath(@__FILE__)))
include(joinpath(root, "src", "julia", "iitm_server.jl"))

field = generate_field(Dict{String,Any}(
    "grid_dim" => 6,
    "L_size" => 3.0,
    "cloud_center_z" => 1.5,
    "cloud_thickness" => 1.2,
    "turbulence_scale" => 3.0,
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

base_mc_cfg = Dict{String,Any}(
    "beta_ext_surf" => 0.2,
    "thickness_m" => 3.0,
    "scale_height_m" => 0.0,
    "n_photons" => 40,
    "use_3d_density" => true,
    "density_sampling" => "nearest",
    "field_forward_half_angle_deg" => 90.0,
    "field_back_half_angle_deg" => 90.0,
    "field_quadrature_polar" => 1,
    "field_quadrature_azimuth" => 4,
)

logs = String[]
log_fn(msg::String) = push!(logs, msg)

proxy_cfg = copy(base_mc_cfg)
proxy_cfg["field_compute_mode"] = "proxy_only"
mc_proxy = step3_mc(field, scatter, proxy_cfg, log_fn)
@assert mc_proxy.voxel_fields === nothing

both_cfg = copy(base_mc_cfg)
both_cfg["field_compute_mode"] = "both"
mc_both = step3_mc(field, scatter, both_cfg, log_fn)
@assert mc_both.voxel_fields !== nothing

println("julia_server_contract_ok")
