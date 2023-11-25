using LinearAlgebra
using Flux
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

    num_simulation::Int
    α::Float32
    frac::Float32
    ratio::Float32
    ratio_r::Float32

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

#max_turn, num_player, middle=dim, depth, training_step, batch_size, batch_num, num_simulation, a, frac, t_step, HS_size, Ω, ξ, Jz, Jx, hz, Cb, Ci, C
function init_Env(args::Vector{String})
    max_turn = parse(Int, args[1])
    println("max_turn:  $(max_turn)")
    num_player = parse(Int, args[2])
    println("num_player:  $(num_player)")
    val_num::Int = 2
    br_num::Int = 3
    fn_num::Int = 1
    act_ind = val_num+br_num+fn_num
    input_dim = act_ind*max_turn
    middle_dim = parse(Int, args[3])
    println("middle_dim:  $(middle_dim)")
    output =  act_ind + 1
    depth = parse(Int, args[4])
    println("depth:  $(depth)")

    #training parameter
    training_step = parse(Int, args[5])
    println("training_step:  $(training_step)")
    checkpoint_interval = 200
    batch_size = parse(Int, args[6])
    println("batch_size:  $(batch_size)")
    batch_num = parse(Int, args[7])
    println("batch_num:  $(batch_num)")
    η = 1f-5
    momentum = 0.9


    num_simulation = parse(Int, args[8])
    println("num_simulation:  $(num_simulation)")
    α = parse(Float32, args[9])
    println("α:  $(α)")
    frac = parse(Float32, args[10])
    println("frac:  $(frac)")
    ratio = parse(Float32, args[21])
    println("ratio:  $(ratio)")
    ratio_r = parse(Float32, args[22])
    println("ratio_r:  $(ratio_r)")


    t_step = parse(Int, args[11])
    HS_size = parse(Int, args[12])
    Ω = parse(Float32, args[13])
    println("Ω:  $(Ω)")
    ξ = parse(Float32, args[14])
    println("ξ:  $(ξ)")
    Jz = parse(Float32, args[15])
    Jx = parse(Float32, args[16])
    hz = parse(Float32, args[17])
    H_0 = Hermitian([ -Jz-2hz 0 0 -Jx; 0 Jz -Jx 0; 0 -Jx Jz 0; -Jx 0 0 -Jz+2hz])
    V_t = Hermitian([ 0 -ξ -ξ 0; -ξ 0 0 -ξ; -ξ 0 0 -ξ; 0 -ξ -ξ 0])
    dt = 2pi/t_step/Ω

    Cb = parse(Int, args[18])
    Ci = parse(Float32, args[19])
    C = parse(Float32, args[20])

    return Env(max_turn, num_player, val_num, br_num, fn_num, act_ind, input_dim, middle_dim, output, depth, training_step, checkpoint_interval, batch_size, batch_num, η, momentum, num_simulation, α, frac, ratio, ratio_r, t_step, HS_size, Ω, ξ, Jz, Jx, hz, H_0, V_t, dt, Cb, Ci, C)
end

function init_Env_forcheck(args::Vector{String})
    max_turn = parse(Int, args[1])
    println("max_turn:  $(max_turn)")
    num_player = parse(Int, args[2])
    println("num_player:  $(num_player)")
    val_num::Int = 2
    br_num::Int = 3
    fn_num::Int = 1
    act_ind = val_num+br_num+fn_num
    input_dim = act_ind*max_turn
    middle_dim = parse(Int, args[3])
    println("middle_dim:  $(middle_dim)")
    output =  act_ind + 1
    depth = parse(Int, args[4])
    println("depth:  $(depth)")

    #training parameter
    training_step = parse(Int, args[5])
    println("training_step:  $(training_step)")
    checkpoint_interval = 200
    batch_size = parse(Int, args[6])
    println("batch_size:  $(batch_size)")
    batch_num = parse(Int, args[7])
    println("batch_num:  $(batch_num)")
    η = 1f-5
    momentum = 0.9


    num_simulation = parse(Int, args[8])
    println("num_simulation:  $(num_simulation)")
    α = parse(Float32, args[9])
    println("α:  $(α)")
    frac = parse(Float32, args[10])
    println("frac:  $(frac)")
    ratio = parse(Float32, args[21])
    println("ratio:  $(ratio)")
    ratio_r = parse(Float32, args[22])
    println("ratio_r:  $(ratio_r)")

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

    return Env(max_turn, num_player, val_num, br_num, fn_num, act_ind, input_dim, middle_dim, output, depth, training_step, checkpoint_interval, batch_size, batch_num, η, momentum, num_simulation, α, frac, ratio, ratio_r, t_step, HS_size, Ω, ξ, Jz, Jx, hz, H_0, V_t, dt, Cb, Ci, C)
end

#max_turn, middle_dim, depth, α, frac, Cb, Ci
function init_Env_quick(args::Vector{String}, hyperparams::Vector{Any})
    max_turn = parse(Int, args[1])
    num_player = 200
    #parse(Int, args[2])
    val_num::Int = 2
    br_num::Int = 3
    fn_num::Int = 1
    act_ind = val_num+br_num+fn_num
    input_dim = act_ind*max_turn
    middle_dim = parse(Int, args[2])
    output =  act_ind + 1
    depth = parse(Int, args[3])

    #training parameter
    training_step = 1000
    #parse(Int, args[5])
    checkpoint_interval = 200
    batch_size = 500
    #parse(Int, args[6])
    batch_num = 1
    η = 1f-5
    momentum = 0.9


    num_simulation = 512
    #parse(Int, args[8])
    α = Float32(hyperparams[1])
    frac = parse(Float32, args[4])
    ratio = 0.1f0
    ratio_r = 1.0f0

    t_step = 100
    HS_size = 4
    Ω = 10.0f0
    ξ = 0.4f0
    Jz = 1.0f0
    Jx = 0.7f0
    hz = 0.5f0
    H_0 = Hermitian([ -Jz-2hz 0 0 -Jx; 0 Jz -Jx 0; 0 -Jx Jz 0; -Jx 0 0 -Jz+2hz])
    V_t = Hermitian([ 0 -ξ -ξ 0; -ξ 0 0 -ξ; -ξ 0 0 -ξ; 0 -ξ -ξ 0])
    dt = 2pi/t_step/Ω

    Cb = Int(hyperparams[2])
    Ci = Float32(hyperparams[3])
    C = 1f-6

    return Env(max_turn, num_player, val_num, br_num, fn_num, act_ind, input_dim, middle_dim, output, depth, training_step, checkpoint_interval, batch_size, batch_num, η, momentum, num_simulation, α, frac, ratio, ratio_r, t_step, HS_size, Ω, ξ, Jz, Jx, hz, H_0, V_t, dt, Cb, Ci, C)
end

x = symbols("x", real=true)
sx = sin(x)

function calc_Kt_sym(history::Vector{Int}, env::Env)
    MV = []
    his = copy(history)
    t = collect(0:env.Ω*env.dt:2pi)
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
            a1 = sympy.re.(A)
            a2 = sympy.im.(A)
            B = (a1.integrate(x) + 1im*a2.integrate(x))/env.Ω
            #=
            try
                S = A.subs(x, t[1])-A.subs(x, t[div(env.t_step,10)])
                if(S==zeros(env.HS_size, env.HS_size))
                    B = A
                else
                    a1 = sympy.re.(A)
                    a2 = sympy.im.(A)
                    #println("it=$(it): Integral!")
                    B = (a1.integrate(x) + 1im*a2.integrate(x))/env.Ω
                end
                #println("try!$(A)")
            catch
                B = A
            end=#
            push!(MV, B)
        end
        #println("it=$(it): $(MV[end])")
    end
    return MV[end], t
end

function calc_Kt(history::Vector{Int}, env::Env)
    #=
    MV = []
    his = copy(history)
    t = collect(0:env.Ω*env.dt:2pi)
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
                S = A.subs(x, t[1])-A.subs(x, t[div(env.t_step,4)])
                #println(S)
                if(S==zeros(env.HS_size, env.HS_size))
                    B = A
                else
                    B = A.integrate(x)/env.Ω
                end
            catch
                B = A
            end
            push!(MV, B)
        end
        #@show MV
    end
    #t = collect(0:env.Ω*env.dt:2pi)

    Ks = MV[end]=#
    Ks, t = calc_Kt_sym(history, env)
    
    #println(Ks)
    if(typeof(Ks)==Matrix{Sym})
        K0 = convert(Matrix{ComplexF32}, Ks.subs(x,t[1]))
        Kt::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}} = [Hermitian(convert(Matrix{ComplexF32}, Ks.subs(x,t[i]))-K0) for i in 1:env.t_step]
        return Kt
    else
        Kh::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}} = [Hermitian(convert(Matrix{ComplexF32}, Ks)) for i in 1:env.t_step]
        return Kh
    end
end

dict = Dict(1=>"H_0 ", 2=>"V(t) ", 3=>"+ ", 4=>"-i[,] ", 5=>"{,}/2 ", 6=>"∱dt ")

function hist2eq(history::Vector{Int})
    hist = copy(history)
    S = ""
    for i in hist
        S *= dict[i]
    end
    return S
end

function legal_action(env::Env, history::Vector{Int}, branch_left::Vector{Int})
    if(isempty(history))
        return [i for i in 1:env.act_ind]
    #elseif(env.max_turn-length(history)<=length(branch_left)+1)
    elseif(env.max_turn-length(history)<=length(branch_left)+2)
        return [i for i in 1:env.val_num]
    elseif(history[end]>env.val_num && history[end]<=env.val_num+env.br_num)
        return [i for i in 1:env.act_ind if(i!=history[end])]
    elseif(history[end]==6)
        return [i for i in 2:env.act_ind-1]
    else
        return [i for i in 1:env.act_ind]
    end
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
            score += real(tr((Hr[i]-Hr[end])^2))
        else
            score += real(tr((Hr[i]-Hr[i-1])^2))
        end 
    end
    return -log(score/env.t_step+1f-15)
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
    #println("history: $(history)")
    Kt = calc_Kt(history, env)
    Hr = calc_Hr(Kt, env)
    score = calc_loss(Hr, env)
    if(isnan(score))
        println("score: nan")
        println(history)
        score = Float32(-10.0)
    end
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

