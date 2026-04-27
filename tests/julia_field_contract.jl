#!/usr/bin/env julia

using NPZ

root = dirname(dirname(abspath(@__FILE__)))
include(joinpath(root, "src", "julia", "iitm_physics.jl"))

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

mc = run_monte_carlo(scatter, Dict{String,Any}(
    "beta_ext_surf" => 0.2,
    "thickness_m" => 3.0,
    "scale_height_m" => 0.0,
    "n_photons" => 160,
    "seed" => 20260427,
    "collect_voxel_fields" => true,
    "density_grid" => field["density_norm"],
    "field_axis" => field["axis"],
    "field_xy_centered" => true,
    "density_sampling" => "nearest",
    "field_forward_half_angle_deg" => 90.0,
    "field_back_half_angle_deg" => 90.0,
))

bundle = build_field_bundle(field, scatter, Dict{String,Any}(
    "field_compute_mode" => "both",
    "r_bottom" => 0.5,
    "r_top" => 1.0,
), mc)

@assert bundle["requested_field_compute_mode"] == "both"
@assert bundle["effective_field_compute_mode"] == "both"
@assert haskey(bundle["families"], "proxy")
@assert haskey(bundle["families"], "exact")

catalog = build_field_catalog(bundle)
@assert haskey(catalog, "proxy")
@assert haskey(catalog, "exact")
@assert any(entry -> entry["name"] == "density", catalog["proxy"])
@assert any(entry -> entry["name"] == "event_count", catalog["exact"])

tmpdir = mktempdir()
try
    npz_path = save_field_npz(field, bundle, scatter, tmpdir)
    data = npzread(npz_path)
    @assert haskey(data, "proxy_beta_back")
    @assert haskey(data, "proxy_beta_forward")
    @assert haskey(data, "proxy_depol_ratio")
    @assert haskey(data, "exact_beta_back")
    @assert haskey(data, "exact_beta_forward")
    @assert haskey(data, "exact_depol_ratio")
    @assert haskey(data, "exact_event_count")
    @assert size(data["density"]) == (6, 6, 6)
finally
    rm(tmpdir; recursive=true, force=true)
end

println("julia_field_contract_ok")
