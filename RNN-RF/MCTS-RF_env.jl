using LinearAlgebra


struct Env
    max_turn::Int
    act_ind::Int
    input_dim::Int
    output::Int

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

    Cb::Int
    Ci::Float32
    C::Float32 #L2 norm weight
end

function init_Env(args::Vector{String})
    max_turn = parse(Int, args[1])
    act_ind = parse(Int, args[2])
    input_dim = parse(Int, args[3])
    output =  parse(Int, args[4])
    t_step = parse(Int, args[5])
    HS_size = parse(Int, args[6])
    Ω = parse(Float32, args[7])
    ξ = parse(Float32, args[8])
    Jz = parse(Float32, args[9])
    Jx = parse(Float32, args[10])
    hz = parse(Float32, args[11])
    H_0 = Hermitian([ -Jz-2hz 0 0 -Jx; 0 Jz -Jx 0; 0 -Jx Jz 0; -Jx 0 0 -Jz+2hz])
    V_t = Hermitian([ 0 -ξ -ξ 0; -ξ 0 0 -ξ; -ξ 0 0 -ξ; 0 -ξ -ξ 0])
    dt = Float32(2pi/t_step/Ω)

    Cb = parse(Int, args[12])
    Ci = parse(Float32, args[13])
    C = parse(Float32, args[14])

    return max_turn, act_ind, input_dim, output, t_step, HS_size, Ω, ξ, Jz, Jx, hz, H_0, V_t, dt, Cb, Ci, C
end

function calc_Kt(act_vec::Vector{Int}, env::Env)
    Kt::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}} = []
    for i in 1:env.t_step
        Kt0 = sin(act_vec[i])*env.V_t + env.H_0
        push!(Kt, Kt0)
    end
    return Kt
end

function calc_Hr(Kt::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}}, env::Env)
    Hr::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}} = []
    for i in 1:env.t_step
        ip = i+1
        if(ip>env.t_step)
            ip = 1
        end
        imm = i-1
        if(imm<1)
            imm = env.t_step
        end
        Hr0 = expm(1im*Kt[i]*env.dt)*(env.H_0 + env.V_t*sin(env.Ω*i*env.dt)) * expm(-1im*Kt[i]*env.dt) - 1im* expm(1im*Kt[i]*env.dt)*(expm(-1im*Kt[ip]*env.dt)-expm(-1im*Kt[imm]*env.dt))/2env.dt
        push!(Hr, Hr0)
    end
    return Hr
end

function calc_score(Hr::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}}, env::Env)
    score::Float32 = 0.0
    for i in 1:env.t_step
        if(i==1)
            continue
        end
        score -= real(tr((Hr[i]-Hr[i-1])^2))
    end
    return score/env.t_step + Float32(1.0)
end

function calc_score(act_vec::Vector{Int}, env::Env)
    Kt = calc_Kt(act_vec, env)
    Hr = calc_Hr(Kt, env)
    score = calc_score(Hr, env)
    return score
end

