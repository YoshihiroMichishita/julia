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
    depth::Int

    #training parameter
    training_step::Int
    checkpoint_interval::Int
    batch_size::Int
    batch_num::Int
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
    middle_dim = parse(Int, args[3])
    output =  act_ind + 1
    depth = parse(Int, args[4])

    #training parameter
    training_step = parse(Int, args[5])
    checkpoint_interval = 500
    batch_size = parse(Int, args[6])
    batch_num = parse(Int, args[7])
    η = 1f-4
    momentum = 0.9
    scheduler = Step(2f-1, Float32(0.1), 500)


    num_simulation = parse(Int, args[8])
    α = parse(Float32, args[9])
    frac = parse(Float32, args[10])

    t_step = parse(Int, args[11])
    HS_size = parse(Int, args[12])
    Ω = parse(Float32, args[13])
    ξ = parse(Float32, args[14])
    Jz = parse(Float32, args[15])
    Jx = parse(Float32, args[16])
    hz = parse(Float32, args[17])
    H_0 = Hermitian([ -Jz-2hz 0 0 -Jx; 0 Jz -Jx 0; 0 -Jx Jz 0; -Jx 0 0 -Jz+2hz])
    V_t = Hermitian([ 0 -ξ -ξ 0; -ξ 0 0 -ξ; -ξ 0 0 -ξ; 0 -ξ -ξ 0])
    dt = 2pi/t_step/Ω

    Cb = parse(Int, args[18])
    Ci = parse(Float32, args[19])
    C = parse(Float32, args[20])

    return Env(max_turn, num_player, val_num, br_num, fn_num, act_ind, input_dim, middle_dim, output, training_step, checkpoint_interval, batch_size, batch_num, η, momentum, scheduler, num_simulation, α, frac, t_step, HS_size, Ω, ξ, Jz, Jx, hz, H_0, V_t, dt, Cb, Ci, C)
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
            C = -1im*(A*B - B*A)
            push!(MV, C)
        elseif(sw==5)
            A = pop!(MV)
            B = pop!(MV)
            C = (A*B + B*A)/2
            push!(MV, C)
        elseif(sw==6)
            A = pop!(MV)
            try
                #B = A.integrate((x, 0, x))/env.Ω
                B = A.integrate(x)/env.Ω
            catch
                B = A
            end
            push!(MV, B)
        end
        #@show MV
    end
    t = collect(0:env.Ω*env.dt:2pi)

    Ks = MV[end]
    
    #println(Ks)
    if(typeof(Ks)==Matrix{Sym})
        K0 = N(Ks.subs(x,t[1]))
        #Kt::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}} = [Hermitian(N(Ks.subs(x,t[i]))) for i in 1:env.t_step]
        Kt::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}} = [Hermitian(N(Ks.subs(x,t[i]))-K0) for i in 1:env.t_step]
        return Kt
    else
        Kh::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}} = [Hermitian(Ks) for i in 1:env.t_step]
        return Kh
    end
    #Kt = [Hermitian(N(Ks.subs(x,t[i]))) for i in 1:env.t_step]
    
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
        U = exp(1im*Kt[i])
        Hr0 = Hermitian(U*(env.H_0 + env.V_t*sin(env.Ω*i*env.dt)) * U' - 1im* U*(exp(-1im*Kt[ip])-exp(-1im*Kt[imm]))/2env.dt)
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
        score += real(tr((Hr[i]-Hr[i-1])^2))
    end
    return -log(score/env.t_step+1f-10)
end
#=
function calc_loss(Hr::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}}, env::Env)
    score::Float32 = 0.0
    for i in 1:env.t_step
        for j in 1:env.t_step
            score += real(tr((Hr[i]-Hr[j])^2))/env.t_step
        end
    end
    return -score+Float32(1.0)
end=#

function calc_score(history::Vector{Int}, env::Env)
    Kt = calc_Kt(history, env)
    Hr = calc_Hr(Kt, env)
    score = calc_loss(Hr, env)
    return score
end

function score_test()
    env = init_Env(ARGS)
    history = [6, 2]
    println(calc_score(history, env))
    history = [6, 3, 2, 4, 1, 6, 2]
    println(calc_score(history, env))
    history = [6, 3, 2, 4, 6, 2, 1]
    println(calc_score(history, env))
end

#score_test()

