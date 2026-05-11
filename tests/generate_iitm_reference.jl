#!/usr/bin/env julia
"""
生成 IITM 后端的散射参量参考值（WF4-L1）。

场景：单一球形粒子
- r = 1.0 μm
- λ = 1550 nm
- m = 1.5 + 0i（无吸收）
- 数密度 = 1e6 m⁻³
"""

using JSON3

include("../src/julia/iitm_physics.jl")

function generate_reference()
    # 场景参数（与 Python 侧对比保持一致）
    config = Dict{String, Any}(
        "wavelength_m" => 1.55e-6,
        "m_real" => 1.5,
        "m_imag" => 0.0,
        "shape_type" => "sphere",
        "size_mode" => "mono",
        "radius_um" => 1.0,
        "n_radii" => 1,
        "Nr" => 32,
        "Ntheta" => 64,
    )

    result = compute_scatter_params(config)

    reference = Dict{String, Any}(
        "scenario" => Dict(
            "wavelength_m" => config["wavelength_m"],
            "m_real" => config["m_real"],
            "m_imag" => config["m_imag"],
            "shape_type" => config["shape_type"],
            "size_mode" => config["size_mode"],
            "radius_um" => config["radius_um"],
            "n_radii" => config["n_radii"],
            "Nr" => config["Nr"],
            "Ntheta" => config["Ntheta"],
        ),
        "scattering_params" => Dict(
            "sigma_ext" => result["sigma_ext"],
            "sigma_sca" => result["sigma_sca"],
            "omega0" => result["omega0"],
            "g" => result["g"],
            "angles_deg" => collect(result["angles_deg"]),
            "M11" => collect(result["M11"]),
            "M12" => collect(result["M12"]),
        ),
        "backend" => "IITM",
        "generated_by" => "tests/generate_iitm_reference.jl"
    )

    return reference
end

function main()
    ref = generate_reference()

    output_path = joinpath(@__DIR__, "fixtures", "iitm_sphere_reference.json")
    mkpath(dirname(output_path))

    open(output_path, "w") do io
        write(io, String(JSON3.write(ref)))
    end

    println("✓ IITM 参考值已保存到: $output_path")
    println("\n散射参量:")
    println("  sigma_sca = $(ref["scattering_params"]["sigma_sca"]) m²")
    println("  sigma_ext = $(ref["scattering_params"]["sigma_ext"]) m²")
    println("  omega0    = $(ref["scattering_params"]["omega0"])")
    println("  g         = $(ref["scattering_params"]["g"])")
end

main()
