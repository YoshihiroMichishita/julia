###########################
#You can utilyze Alpha Zero for Physics just by rewriting this env code for your problems. What you have to consider are just setting of the scores and defining of the function, branch, and variable nodes.  
###########################

using LinearAlgebra
using Flux
#using SymPy #DO NOT USE VER:2.0.1!!!!! THIS CODE DOES NOT WORK!!!!!!!!! PLEASE USE VER:1.2.1. Pkg> add SymPy@1.2.1

#ENV["PYTHON"] = "/Users/johnbrother/miniforge3/envs/p310/bin/python"


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

    #Physics Model Parameter
    τ::Float32
    s_dim::Int
    e_dim::Int
    tot_dim::Int

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
    val_num::Int = 2 # ρ, Λσ
    br_num::Int = 1 # UρUd, 
    fn_num::Int = 4 # nroot_m(2, -2), Λ, Λd
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
    ratio = parse(Float32, args[11])
    println("ratio:  $(ratio)")
    ratio_r = parse(Float32, args[12])
    println("ratio_r:  $(ratio_r)")


    τ::Float32 = parse(Float32, args[13])
    s_dim::Int = parse(Int, args[14])
    e_dim::Int = parse(Int, args[15])
    tot_dim = s_dim * e_dim

    Cb = parse(Int, args[16])
    Ci = parse(Float32, args[17])
    C = parse(Float32, args[18])

    return Env(max_turn, num_player, val_num, br_num, fn_num, act_ind, input_dim, middle_dim, output, depth, training_step, checkpoint_interval, batch_size, batch_num, η, momentum, num_simulation, α, frac, ratio, ratio_r, τ, s_dim, e_dim, tot_dim, Cb, Ci, C)
end

function init_Env_forcheck(args::Vector{String})
    max_turn = parse(Int, args[1])
    println("max_turn:  $(max_turn)")
    num_player = parse(Int, args[2])
    println("num_player:  $(num_player)")
    val_num::Int = 2
    br_num::Int = 1
    fn_num::Int = 4
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
    ratio = parse(Float32, args[11])
    println("ratio:  $(ratio)")
    ratio_r = parse(Float32, args[12])
    println("ratio_r:  $(ratio_r)")

    τ::Float32 = parse(Float32, args[13])
    s_dim::Int = parse(Int, args[14])
    e_dim::Int = parse(Int, args[15])
    tot_dim = s_dim * e_dim

    Cb = parse(Int, args[16])
    Ci = parse(Float32, args[17])
    C = parse(Float32, args[18])

    return Env(max_turn, num_player, val_num, br_num, fn_num, act_ind, input_dim, middle_dim, output, depth, training_step, checkpoint_interval, batch_size, batch_num, η, momentum, num_simulation, α, frac, ratio, ratio_r, τ, s_dim, e_dim, tot_dim, Cb, Ci, C)
end

#max_turn, middle_dim, depth, α, frac, Cb, Ci
function init_Env_quick(args::Vector{String})
    max_turn = parse(Int, args[1])
    num_player = 200
    #parse(Int, args[2])
    val_num::Int = 2
    br_num::Int = 1
    fn_num::Int = 4
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
    α = Float32(0.3)
    frac = parse(Float32, args[4])
    ratio = 0.1f0
    ratio_r = 1.0f0

    τ::Float32 = parse(Float32, args[5])
    s_dim::Int = 2
    e_dim::Int = 2
    tot_dim = s_dim * e_dim

    Cb = parse(Int, args[6])
    Ci = parse(Float32, args[7])
    C = parse(Float32, args[8])

    return Env(max_turn, num_player, val_num, br_num, fn_num, act_ind, input_dim, middle_dim, output, depth, training_step, checkpoint_interval, batch_size, batch_num, η, momentum, num_simulation, α, frac, ratio, ratio_r, τ, s_dim, e_dim, tot_dim, Cb, Ci, C)
end
const fi = ComplexF32(1.0im)

struct DMs
    s_dim::Int
    e_dim::Int
    tot_dim::Int
    s_dm::Hermitian{ComplexF32, Matrix{ComplexF32}}
    e_dm::Hermitian{ComplexF32, Matrix{ComplexF32}}
    s_evs::Matrix{ComplexF32}
    s_es::Vector{Float32}

    U::Matrix{ComplexF32}
    Ms::Vector{Matrix{ComplexF32}}
end

function vec2hermite(v::Vector{Float32})
    N = round(Int, sqrt(length(v)))
    H = zeros(ComplexF32, N, N)
    for i in 1:N
        for j in i:N
            l = N*(i-1) + 2j - i^2
            if(i == j)
                H[i,j] = v[l]
            else
                H[i,j] = v[l-1] + fi*v[l]
            end 
        end
    end
    H = Hermitian(H)
    return H
end

function vec2unitary(v::Vector{Float32}, τ::Float32)
    H = vec2hermite(v)
    U = exp(fi*(τ*H))
    return U
end

function make_unitary(N::Int, τ::Float32)
    v = rand(Float32, N^2)
    U = vec2unitary(v, τ)
    return U
end

function norm!(m::Hermitian{ComplexF32, Matrix{ComplexF32}})
    T = real(tr(m))
    m = m./T
end

function make_rand_dm(dim::Int)
    ρ_vec = rand(Float32, dim^2)
    rt_ρ = vec2hermite(ρ_vec)
    ρ = Hermitian(norm!(Hermitian(rt_ρ*rt_ρ')))
    return ρ
end

function ehot(vs::Vector{ComplexF32}, i::Int, s_dim::Int, e_dim::Int)
    ve = zeros(ComplexF32, e_dim*s_dim)
    ve[(s_dim*(i-1)+1):(s_dim*i)] = vs
    return ve
end

function make_ev(s_ev::Matrix{ComplexF32}, s_dim::Int, e_dim::Int)
    s_vec::Vector{Matrix{ComplexF32}} = []
    tot_dim = s_dim * e_dim
    for i in 1:s_dim
        sm = zeros(ComplexF32, tot_dim, e_dim)
        for j in 1:e_dim
            sm[:,j] = ehot(s_ev[:,i], j, s_dim, e_dim)
        end
        push!(s_vec, sm)
    end
    return s_vec
end

function make_Mk(U::Matrix{ComplexF32}, s_vec::Vector{Matrix{ComplexF32}}, s_dim::Int, e_dim::Int)
    L = s_dim * e_dim
    #L = size(U)[1]
    #e_dim = length(s_vec)
    #s_dim = div(L,e_dim)
    Ms::Vector{Matrix{ComplexF32}} = []
    for j in 1:e_dim
        for k in 1:s_dim
            push!(Ms, (s_vec[k]'*U*s_vec[j]))
        end
    end
    return Ms
end

function init_dms(en::Env)
    s_dim = en.s_dim
    e_dim = en.e_dim
    tot_dim = s_dim * e_dim
    s_dm = make_rand_dm(s_dim)
    e_dm = make_rand_dm(e_dim)
    s_es, s_evs = eigen(s_dm)
    U = make_unitary(tot_dim, τ)
    s_evsa = make_ev(s_evs, s_dim, e_dim)
    Ms = make_Mk(U, s_evsa, s_dim, e_dim)
    return DMs(s_dim, e_dim, tot_dim, s_dm, e_dm, s_evs, s_es, U, Ms)
end



function UρUd(U, ρ::Hermitian{ComplexF32, Matrix{ComplexF32}})
    return Hermitian(U*ρ*U')
end

function nroot_m(ρ::Hermitian{ComplexF32, Matrix{ComplexF32}}, n::Int)
    #ρ_vec = zeros(Float32, tot_dim^2)
    e, v = eigen(ρ)
    en = e.^(1.0f0/n)
    ρ_n = v*Diagonal(en)*v'
    return Hermitian(ρ_n)
end

function Λρ(ρ::Hermitian{ComplexF32, Matrix{ComplexF32}}, dms::DMs)
    Lρ = zeros(ComplexF32, dms.s_dim, dms.s_dim)
    for i in 1:dms.tot_dim
        Lρ += dms.s_es[div(i-1,ds.e_dim)+1]*dms.Ms[i]'*ρ*dms.Ms[i]
    end
    return Hermitian(Lρ)
end

function Λρd(ρ::Hermitian{ComplexF32, Matrix{ComplexF32}}, dms::DMs)
    Lρ = zeros(ComplexF32, dms.s_dim, dms.s_dim)
    for i in 1:dms.tot_dim
        Lρ += dms.s_es[div(i-1,ds.e_dim)+1]*dms.Ms[i]'*ρ*dms.Ms[i]
    end
    return Hermitian(Lρ)
end

#inds:symbols| 1:Λσ, 2:ρ, 3:UρUd, 4:nroot_m(2, -2), 5:Λ, 6:Λd 
function calc_RecoveryMap(history::Vector{Int}, dms::DMs, Λσ::Hermitian{ComplexF32, Matrix{ComplexF32}})
    MV = []
    his = copy(history)

    for it in 1:length(his)
        sw = pop!(his)
        if(sw==1)
            push!(MV, Λσ)
        elseif(sw==2)
            push!(MV, dms.s_dm)
        elseif(sw==3)
            A = pop!(MV)
            B = pop!(MV)
            C = UρUd(A, B)
            push!(MV, C)
        elseif(sw==4)
            A = pop!(MV)
            B = nroot_m(A, 2)
            push!(MV, B)
        elseif(sw==5)
            A = pop!(MV)
            B = nroot_m(A, -2)
            push!(MV, B)
        elseif(sw==6)
            A = pop!(MV)
            B = Λρ(A, dms)
            push!(MV, B)
        elseif(sw==7)
            A = pop!(MV)
            B = Λdρ(A, dms)
            push!(MV, B)
        end
    end
    return MV[end]
end
#=
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
end=#

dict = Dict(1=>"Λσ ", 2=>"ρ ", 3=>"UρUd ", 4=>"M^{1/2} ", 5=>"M^{-1/2} ", 6=>"Λ ", 7=>"Λd")

function hist2eq(history::Vector{Int})
    hist = copy(history)
    S = ""
    for i in hist
        S *= dict[i]
    end
    return S
end

#Rule of AZfP
function legal_action(env::Env, history::Vector{Int}, branch_left::Vector{Int})
    if(isempty(history))
        return [i for i in 1:env.act_ind]
    #elseif(env.max_turn-length(history)<=length(branch_left)+1)
    elseif(env.max_turn-length(history)<=length(branch_left)+2)
        return [i for i in 1:env.val_num]
    else
        return [i for i in 1:env.act_ind]
    end
end

function KL_divergence(ρ::Hermitian{ComplexF32, Matrix{ComplexF32}}, σ::Hermitian{ComplexF32, Matrix{ComplexF32}})
    return real(tr(ρ*(log(ρ)-log(σ))))
end

function calc_score_σ(history::Vector{Int}, dms::DMs, σ::Hermitian{ComplexF32, Matrix{ComplexF32}})
    Λσ = Λρ(σ, dms)
    σ_recov = calc_RecoveryMap(history, dms, Λσ)
    norm!(σ_recov)
    return -KL_divergence(σ, σ_recov)+1.0f0
end

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

