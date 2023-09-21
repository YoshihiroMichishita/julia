using LinearAlgebra
using Plots

struct Parm
    t_step::Int
    HS_size::Int
    Ω::Float32
    ξ::Float32
    Jz::Float32
    Jx::Float32
    hz::Float32
    H_0::Hermitian{ComplexF32, Matrix{ComplexF32}}
    V_t::Hermitian{ComplexF32, Matrix{ComplexF32}}
    dt::Float32
end

function init_parm(arg::Vector{String})
    t_step = parse(Int, arg[1])
    HS_size = parse(Int, arg[2])
    Ω = parse(Float32, arg[3])
    ξ = parse(Float32, arg[4])
    Jz = parse(Float32, arg[5])
    Jx = parse(Float32, arg[6])
    hz = parse(Float32, arg[7])
    H_0 = Hermitian([ -Jz-2hz 0 0 -Jx; 0 Jz -Jx 0; 0 -Jx Jz 0; -Jx 0 0 -Jz+2hz])
    V_t = Hermitian([ 0 -ξ -ξ 0; -ξ 0 0 -ξ; -ξ 0 0 -ξ; 0 -ξ -ξ 0])
    dt = 2pi/Ω/t_step

    return Parm(t_step, HS_size, Ω, ξ, Jz, Jx, hz, H_0, V_t, dt)
end

function calc_exact_hr(parm::Parm)
    