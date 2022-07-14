using Flux
include("TwoSpin_env.jl")


mutable struct agtQ
    in_size::Int
    out_size::Int
    n_dense::Int
    ϵ::Float64
    γ::Float64
    HF_TL::Matrix{Float64}
    K_TL::Matrix{Float64}
end

function init_nQ(en::TS_env, n::Int=32, γ0::Float64=0.9)
    #H_0,V_tのパラメータの数＋K_tの行列＋H_F^a(t)の行列
    in_size::Int = en.num_parm + 2*en.HS_size^2 

    #K'(t)の行列を出力
    out_size::Int = en.HS_size^2

    #中間層のニューロンの数
    n_dense::Int = n

    #乱数発生用のパラメータ
    ϵ::Float64 = 0.1

    #割引率
    γ::Float64 = γ0

    HF_TL = zeros(Float64, en.t_size, en.HS_size^2)
    K_TL = zeros(Float64, en.t_size, en.HS_size^2)

    return in_size, out_size, n_dense, ϵ, γ, HF_TL, K_TL
end

#=
mutable struct models
    model
    opt
    loss
end
=#

#using FFTW

#K'(t)からK(t),H_F^a(t)を計算する関数
function micro_motion(Kp_t, K_t, en::TS_env, t::Int)
    Kp = vec_to_matrix(Kp_t)
    K_t_new = K_t + (2pi/en.t_size/en.Ω) * Kp_t 
    Kt = vec_to_matrix(K_t_new)
    HF_m = exp(1.0im*Kt)*(en.H_0 + en.V_t*sin(2pi*t/en.t_size) - Kp)*exp(-1.0im*Kt)
    HF = matrix_to_vec(HF_m)
    return K_t_new, HF
end


function diff_norm(V::Vector{Float64})
    M = vec_to_matrix(V)
    e, v = eigen(M)
    n::Float64 = e' * e
    return n
end


#lossの関数
function loss_F(en::TS_env, ag::agtQ, t::Int, sw::Int)
    l::Float64 = 0.0
    for n in 1:en.t_size-1 
        if(n<t)
            lt = t-n
        elseif(n==t)
            lt = en.t_size
        else
            lt = t-n+en.t_size
            if(sw==1) 
                break
            end
        end
        l -= ag.γ^(n-1) * diff_norm(ag.HF_TL[t,:]-ag.HF_TL[lt,:])
    end
    return l
end

function loss_t!(model0, en::TS_env, ag::agtQ, t::Int, it::Int)
    if(t==1)
        tt=en.t_size
    else
        tt=t-1
    end
    p = [en.Ω, en.ξ*sin(2pi*t/en.t_size), en.Jz, en.Jx, en.hz]
    x = [p ag.K_TL[tt,:] ag.HF_TL[tt,:]]
    Kp = model0(x)
    ag.K_TL[t,:], ag.HF_TL[t,:] = micro_motion(Kp, ag.K_TL[tt,:],en,t)
    l = loss_F(en, ag, t, it)
    return l 
end
 

#NNの初期化
#=
function build_model(nq::agtQ)
    #model = Chain(Flux.flatten', Dense(nq.input_size, nq.n_dense, relu), Dense(nq.n_dense, nq.n_act))
    model = Chain(Dense(nq.input_size, nq.n_dense, relu), Dense(nq.n_dense, nq.n_dense, relu), Dense(nq.n_dense, nq.out_size))
    opt = ADAM()
    #loss(x,y) = Flux.mse(model(x),y)

    return model, opt, loss
end
=#

#U: NNから出力されるKick Operator(Hermite)の微分をベクトル表示したものを出力
#=
function get_U(m::models , obs)
    U = m.model(Flux.flatten(obs)')
    return U
end
=#

#using RandomMatrices

#ある確率でランダムなKick Operatorを出力、そうでないならNNから出力
#=
function decide_action(nq::agtQ, m::models, obs)

    if(rand()< nq.ϵ)
        her = GaussianHermite(2)
        U = rand(her,4)
        act = matrix_to_vec(U)
    else
        act = get_U(m, obs)
    end

    return act
end
=#
#=
function learn(nq::agtQ, m::models, obs, act, rwd, done, next_obs)
    if(isnothing(rwd))
        return
    end

    y = get_U(obs)
    target = copy(y)

    if(!done)
        next_y = get_Q(next_obs)
        target_act = rwd + nq.γ*maximum(next_y)
    else
        target_act = rwd
    end

    target[act] = target_act

    Flux.train!(m.loss,Flux.params(m.model),obs, m.opt)
end
=#

function main(arg::Array{String,1})

    en = TS_env(init_env(parse(Int,arg[1]), parse(Float64,arg[2]), parse(Float64,arg[3]), parse(Float64,arg[4]), parse(Float64,arg[5]), parse(Float64,arg[6]))...)

    ag = agtQ(init_nQ(en,parse(Int,arg[7]),parse(Float64,arg[8]))...)

    ag.HF_TL[en.t_size,:] = matrix_to_vec(en.H_0)
    ag.K_TL[en.t_size,:] = -matrix_to_vec(en.V_t)/en.Ω

    model = Chain(Dense(ag.input_size, ag.n_dense, relu), Dense(ag.n_dense, ag.n_dense, relu), Dense(ag.n_dense, ag.out_size))
    opt = ADAM()


    it_MAX = parse(Int,arg[9])
    ll_it = zeros(Float64, it_MAX)
    for it in 1:it_MAX
        for t_step in 1:en.t_size
            grads = Flux.gradient(Flux.params(model)) do
                loss_t!(model,en, ag, t_step, it)
            end
            Flux.Optimise.update!(opt, Flux.params(model), grads)
            ll_it[it] += loss_t!(model,en, ag, t_step, it)
        end
        println(ll_it[it])
        if(it%10 == 0)
            ee = zeros(Float64, en.t_size, en.HS_size)
            for t_step in 1:en.t_size
                ee[t_step], v = eigen(vec_to_matrix(ag.HF_TL[t_step,:]))
            end
            save_data1 = DataFrame(transpose(ee))
            CSV.write("./HFt_it="*"$it" *".csv", save_data1)

        end
        if(it == it_MAX)
            save_data2 = DataFrame(transpose(ag.K_TL))
            CSV.write("./Kt_Ω.csv", save_data2)
        end
        
    end
end