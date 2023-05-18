using LinearAlgebra
using Flux
using ParameterSchedulers
using SymPy


struct Env
    max_turn::Int
    num_player::Int
    val_num::Int
    br_num::Int
    fn_num::Int
    act_ind::Int
    input_dim::Int
    middle_dim::Int
    output::Int

    #training parameter
    training_step::Int
    checkpoint_interval::Int
    batch_size::Int
    η::Float32
    momentum::Float32
    scheduler

    num_simulation::Int
    α::Float32
    frac::Float32

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

#max_turn, num_player, num_simulation, a, frac, t_step, HS_size, Ω, ξ, Jz, Jx, hz, Cb, Ci, C
function init_Env(args::Vector{String})
    max_turn = parse(Int, args[1])
    num_player = parse(Int, args[2])
    val_num::Int = 2
    br_num::Int = 3
    fn_num::Int = 1
    act_ind = val_num+br_num+fn_num
    input_dim = act_ind*max_turn
    middle_dim = 128
    output =  act_ind + 1

    #training parameter
    training_step = 100000
    checkpoint_interval = 1000
    batch_size = 128
    η = 1f-4
    momentum = 0.9
    scheduler = Step(2f-1, Float32(0.1), 20000)


    num_simulation = parse(Int, args[3])
    α = parse(Float32, args[4])
    frac = parse(Float32, args[5])

    t_step = parse(Int, args[6])
    HS_size = parse(Int, args[7])
    Ω = parse(Float32, args[8])
    ξ = parse(Float32, args[9])
    Jz = parse(Float32, args[10])
    Jx = parse(Float32, args[11])
    hz = parse(Float32, args[12])
    H_0 = Hermitian([ -Jz-2hz 0 0 -Jx; 0 Jz -Jx 0; 0 -Jx Jz 0; -Jx 0 0 -Jz+2hz])
    V_t = Hermitian([ 0 -ξ -ξ 0; -ξ 0 0 -ξ; -ξ 0 0 -ξ; 0 -ξ -ξ 0])
    dt = 2pi/t_step/Ω

    Cb = parse(Int, args[13])
    Ci = parse(Float32, args[14])
    C = parse(Float32, args[15])

    return Env(max_turn, num_player, val_num, br_num, fn_num, act_ind, input_dim, middle_dim, output, training_step, checkpoint_interval, batch_size, η, momentum, scheduler, num_simulation, α, frac, t_step, HS_size, Ω, ξ, Jz, Jx, hz, H_0, V_t, dt, Cb, Ci, C)
end

x = symbols("x")
sx = sin(x)

function calc_Kt(history::Vector{Int}, env::Env)
    MV = []
    his = copy(history)
    #println(length(his))
    for it in 1:length(his)
        sw = pop!(his)
        if(sw==1)
            push!(MV, env.H_0)
        elseif(sw==2)
            push!(MV, env.V_t*sx)
        elseif(sw==3)
            A = pop!(MV)
            B = pop!(MV)
            C = A + B
            push!(MV, C)
        elseif(sw==4)
            A = pop!(MV)
            B = pop!(MV)
            C = -1.0im*(A*B - B*A)
            push!(MV, C)
        elseif(sw==5)
            A = pop!(MV)
            B = pop!(MV)
            C = (A*B + B*A)/2
            push!(MV, C)
        elseif(sw==6)
            A = pop!(MV)
            B = A.integrate((x, 0, x))
            push!(MV, B)
        end
        #@show MV
    end
    t = collect(0:env.dt:2pi)

    Ks = MV[end]
    #println(Ks)
    Kt::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}} = [Hermitian(N(Ks.subs(x,t[i]))) for i in 1:env.t_step]
    #Kt = [Hermitian(N(Ks.subs(x,t[i]))) for i in 1:env.t_step]
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

function calc_loss(Hr::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}}, env::Env)
    score::Float32 = 0.0
    for i in 1:env.t_step
        if(i==1)
            continue
        end
        score -= real(tr((Hr[i]-Hr[i-1])^2))
    end
    return score/env.t_step + Float32(1.0)
end

function calc_score(history::Vector{Int}, env::Env)
    Kt = calc_Kt(history, env)
    Hr = calc_Hr(Kt, env)
    score = calc_loss(Hr, env)
    return score
end
#=
function test()
    env = init_Env(ARGS)
    history = [3, 6, 2, 1]
    Kt_test = calc_Kt(history, env)
    @show Kt_test
    #score = calc_score(history, env)
    #println(score)
end

test()=#

