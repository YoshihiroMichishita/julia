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
    Kp_TL::Matrix{Float64}
end

function init_nQ(en::TS_env, n::Int=32, γ0::Float64=0.9, ϵ0::Float64=1.0)
    #H_0,V_tのパラメータの数＋K_tの行列＋H_F^a(t)の行列
    in_size::Int = en.num_parm + en.HS_size^2 

    #K'(t)の行列を出力
    out_size::Int = en.HS_size^2

    #中間層のニューロンの数
    n_dense::Int = n

    #乱数発生用のパラメータ
    ϵ::Float64 = ϵ0

    #割引率
    γ::Float64 = γ0

    HF_TL = zeros(Float64, en.t_size, en.HS_size^2)
    K_TL = zeros(Float64, en.t_size, en.HS_size^2)
    Kp_TL = zeros(Float64, en.t_size, en.HS_size^2)

    return in_size, out_size, n_dense, ϵ, γ, HF_TL, K_TL, Kp_TL
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
function micro_motion(Kp_t::Vector{Float64}, K_t::Vector{Float64}, en::TS_env, t::Int)
    Kp = VtoM(Kp_t,en)
    K_t_new = K_t + (2pi/en.t_size/en.Ω) * Kp_t 
    Kt = VtoM(K_t_new,en)
    U = exp(1.0im*Kt)
    HF_m = Hermitian(U*(en.H_0 + en.V_t*sin(2pi*t/en.t_size) - Kp)*U')
    HF = MtoV(HF_m, en)
    return K_t_new, HF
end

function micro_motion2(Kp_t::Vector{Float64}, K_t::Vector{Float64}, en::TS_env, t::Int)
    Kp = VtoM(Kp_t,en)
    K_t_new = K_t + (2pi/en.t_size/en.Ω) * Kp_t 
    Kt = VtoM(K_t_new,en)
    U = exp(1.0im*Kt)
    HF_m = Hermitian(U*(en.H_0 + en.V_t*sin(2pi*t/en.t_size) - Kp)*U')
    HF = MtoV(HF_m,en)
    return HF
end


function diff_norm(V::Vector{Float64}, en::TS_env)
    M = VtoM(V,en)
    e, v = eigen(M)
    #n::Float64 = V' * V
    n::Float64 = e' * e
    #n = sum(e[n]^2 for n in 1:size(e))
    return n
end


#lossの関数
function loss_fn(en::TS_env, ag::agtQ, t::Int, sw::Int)
    l::Float64 = 0.0
    for n in 1:(en.t_size-1) 
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
        l -= (ag.γ^(n-1)) * diff_norm(ag.HF_TL[t,:]-ag.HF_TL[lt,:],en)
    end
    return l
end

function loss_fn_given(en::TS_env, ag::agtQ,H_t::Vector{Float64}, t::Int, sw::Int)
    l::Float64 = 0.0
    for n in 1:(en.t_size-1) 
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
        l -= (ag.γ^(n-1)) * diff_norm((H_t-ag.HF_TL[lt,:]),en)
    end
    return l
end

function loss_fn_simple(en::TS_env, HF_given::Vector{Float64}, HF_calc::Vector{Float64})
    l = -diff_norm(HF_given - HF_calc,en)
    return l
end

function loss_fn_hybrid(en::TS_env, ag::agtQ, HF_given::Vector{Float64}, HF_calc::Vector{Float64}, t::Int)
    l::Float64 = 0.0
    for n in 1:(en.t_size-1) 
        if(n<t)
            lt = t-n
        elseif(n==t)
            lt = en.t_size
        else
            break
            #lt = t-n+en.t_size
        end
        l += ag.ϵ*(ag.γ^(n-1)) * diff_norm((HF_calc-ag.HF_TL[lt,:]),en)/en.t_size
    end
    #l += diff_norm(HF_given - HF_calc,en)/en.t_size
    return l
end

function loss_calc0(model0, en::TS_env, ag::agtQ, t::Int, HF_given::Vector{Float64})
    if(t==1)
        tt=en.t_size
    else
        tt=t-1
    end
    p = [en.Ω, en.ξ*sin(2pi*t/en.t_size), en.Jz, en.Jx, en.hz]
    x = vcat([p, ag.K_TL[tt,:], ag.HF_TL[tt,:]]...)
    Kp = model0(x)
    
    #ag.K_TL[t,:] += Kp
    HF_calc = micro_motion2(Kp, ag.K_TL[tt,:],en,t)
    l = -loss_fn_simple(en, HF_given, HF_calc)
    #l = Kp' * Kp
    return l 
end

function loss_calc(model0, en::TS_env, ag::agtQ, t::Int, it::Int)
    if(t==1)
        tt=en.t_size
    else
        tt=t-1
    end
    p = [en.Ω, en.ξ*sin(2pi*t/en.t_size), en.Jz, en.Jx, en.hz]
    x = vcat([p, ag.K_TL[tt,:], ag.HF_TL[tt,:]]...)
    Kp = model0(x)
    
    #ag.K_TL[t,:] += Kp
    HF_t = micro_motion2(Kp, ag.K_TL[tt,:],en,t)
    l = -loss_fn_given(en, ag, HF_t, t, it)
    #l = Kp' * Kp
    return l 
end

function loss_calc_hyb(model0, en::TS_env, ag::agtQ, HF_given::Vector{Float64})
    l::Float64 = 0.0
    kp_sum = zeros(Float64, en.HS_size^2)
    for t in 1:en.t_size
        if(t==1)
            tt=en.t_size
        else
            tt=t-1
        end
        p = [en.Ω, en.ξ*sin(2pi*t/en.t_size), en.Jz, en.Jx, en.hz]
        #x = vcat([p, ag.K_TL[tt,:], ag.Kp_TL[tt,:]]...)
        x = vcat([p, ag.K_TL[tt,:]]...)
        Kp = model0(x)
        kp_sum += Kp
        #ag.K_TL[t,:] += Kp
        HF_calc = micro_motion2(Kp, ag.K_TL[tt,:],en,t)
        l += loss_fn_hybrid(en,ag, HF_given, HF_calc,t)
        #l += ag.ϵ^2*diff_norm(kp_sum,en)/en.t_size
        #l += diff_norm(kp_sum,en)/en.t_size
        #l += ag.γ^(5*(en.t_size/2 - abs(en.t_size/2-t))) * diff_norm(ag.K_TL[t,:],en)
        if(t==en.t_size)
            l += diff_norm(HF_calc-ag.HF_TL[1,:],en)
        end
    end
    l += diff_norm(kp_sum,en)/en.t_size
    return l 
end

function loss_calc_hyb!(model0, en::TS_env, ag::agtQ, HF_given::Vector{Float64})
    l::Float64 = 0.0
    kp_sum = zeros(Float64, en.HS_size^2)
    for t in 1:en.t_size
        if(t==1)
            tt=en.t_size
        else
            tt=t-1
        end
        p = [en.Ω, en.ξ*sin(2pi*t/en.t_size), en.Jz, en.Jx, en.hz]
        #x = vcat([p, ag.K_TL[tt,:], ag.Kp_TL[tt,:]]...)
        x = vcat([p, ag.K_TL[tt,:]]...)
        #Kp = model0(x)
        ag.Kp_TL[t,:] = model0(x)
        kp_sum += ag.Kp_TL[t,:]
        #ag.K_TL[t,:] += Kp
        ag.K_TL[t,:], ag.HF_TL[t,:] = micro_motion(ag.Kp_TL[t,:], ag.K_TL[tt,:],en,t)
        l += loss_fn_hybrid(en,ag, HF_given, ag.HF_TL[t,:],t)
        #l += diff_norm(kp_sum,en)/en.t_size
        #l += ag.γ^(5*(en.t_size/2 - abs(en.t_size/2-t))) * diff_norm(ag.K_TL[t,:],en)
        #l += ag.γ^(5*(en.t_size - t)) * diff_norm(ag.K_TL[t,:],en)
        if(t==en.t_size)
            l += diff_norm(ag.HF_TL[t,:]-ag.HF_TL[1,:],en)
        end
    end
    l += diff_norm(kp_sum,en)
    #=
    if((t+en.t_size/10)>=en.t_size)
        v = ag.K_TL[t,:]
        l += ag.γ^(en.t_size-t) * (v' * v)
    elseif(t <= en.t_size/10)
        v = ag.K_TL[t,:]
        l += ag.γ^(t) * (v' * v)
    end=#
    #l = Kp' * Kp
    return l, kp_sum/en.t_size
end

function loss_calc!(model0, en::TS_env, ag::agtQ, t::Int, HF_given::Vector{Float64})
    if(t==1)
        tt=en.t_size
    else
        tt=t-1
    end
    p = [en.Ω, en.ξ*sin(2pi*t/en.t_size), en.Jz, en.Jx, en.hz]
    x = vcat([p, ag.K_TL[tt,:]]...)
    Kp = model0(x)
    
    #ag.K_TL[t,:] += Kp
    ag.K_TL[t,:], ag.HF_TL[t,:] = micro_motion(Kp, ag.K_TL[tt,:],en,t)
    l = -loss_fn_simple(en, HF_given, ag.HF_TL[t,:])
    #l = -loss_fn(en, ag, t, it)
    #l = Kp' * Kp
    return l 
end

function updata_KpK!(en::TS_env, ag::agtQ, Kp_av::Vector{Float64})
    for t in 1:en.t_size
        ag.Kp_TL[t,:] = ag.Kp_TL[t,:] - Kp_av
        ag.K_TL[t,:] = ag.K_TL[t,:] - t*Kp_av
        ag.HF_TL[t,:] = micro_motion2(ag.Kp_TL[t,:], ag.K_TL[t,:], en, t)
    end
end


function init_HF(en::TS_env)
    jp = en.Jz + en.hz
    jm = en.Jz - en.hz
    VHmHV::Vector{Float64} = 4*en.ξ*[0.0, 0.0, jp, 0.0, jp, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -jm, 0.0, 0.0, -jm, 0.0]
    init = MtoV(en.H_0, en) + VHmHV
    return init
end

using DataFrames
using CSV
using BSON: @save
using BSON: @load
using Plots
ENV["GKSwstype"]="nul"

function main(arg::Array{String,1})

    en = TS_env(init_env(parse(Int,arg[1]), parse(Float64,arg[2]), parse(Float64,arg[3]), parse(Float64,arg[4]), parse(Float64,arg[5]), parse(Float64,arg[6]))...)

    ag = agtQ(init_nQ(en,parse(Int,arg[7]),parse(Float64,arg[8]),parse(Float64,arg[9]))...)

    #二次の高周波展開で初期値を代入
    ag.HF_TL[en.t_size,:] = init_HF(en)
    
    #-MtoV(en.V_t, en)/en.Ω

    #model = Chain(Dense(ag.in_size, ag.n_dense, tanh), Dense(ag.n_dense, ag.n_dense, tanh), Dense(ag.n_dense, ag.n_dense, tanh), Dense(ag.n_dense, ag.out_size))
    if(arg[11]=="init")
        model = Chain(Dense(ag.in_size, ag.n_dense, tanh), Dense(ag.n_dense, ag.n_dense, tanh), Dense(ag.n_dense, ag.n_dense, tanh), Dense(ag.n_dense, ag.out_size))
        ag.K_TL[en.t_size,:] = zeros(Float64, en.HS_size^2)
    else
        @load arg[11] model
        ag.K_TL = Matrix(CSV.read(arg[12], DataFrame))
    end
    #model = Chain(Dense(zeros(Float64, ag.n_dense, ag.in_size), zeros(Float64, ag.n_dense), tanh), Dense(zeros(Float64, ag.n_dense, ag.n_dense), zeros(Float64, ag.n_dense), tanh), Dense(zeros(Float64, ag.out_size, ag.n_dense), zeros(Float64, ag.out_size)))
    opt = Flux.Optimise.Descent(0.02)
    #Optimiser(ClipValue(1e-3), Adam(1e-3))


    it_MAX = parse(Int,arg[10])
    ll_it = zeros(Float64, it_MAX)
    println("start!")
    #ll_it[1] = loss_t!(model,en, ag, 2, 1)
    #println(ll_it[1])
    
    for it in 1:it_MAX
        HF_it = zeros(Float64, en.HS_size^2) 
        if(it==1)
            for t_step in 1:en.t_size
                if(t_step==1)
                    tt=en.t_size
                else
                    tt=t_step-1
                end
                p = [en.Ω, en.ξ*sin(2pi*t_step/en.t_size), en.Jz, en.Jx, en.hz]
                #x = vcat([p, ag.K_TL[tt,:], ag.Kp_TL[tt,:]]...)
                x = vcat([p, ag.K_TL[tt,:]]...)
                ag.Kp_TL[t_step,:] = model(x)
                ag.K_TL[t_step,:], ag.HF_TL[t_step,:] = micro_motion(ag.Kp_TL[t_step,:], ag.K_TL[tt,:],en,t_step)
                #ag.K_TL[t,:] += Kp
                HF_it += ag.HF_TL[t_step,:]/en.t_size
            end
            println("HF_calc Finish!")
        end
        
        if(it_MAX>1)
            grads = Flux.gradient(Flux.params(model)) do
                loss_calc_hyb(model, en, ag, HF_it)
                #ll_it[it] = loss_calc_hyb_test(model, en, ag, HF_it)
                #loss_calc0(model, en, ag, t_step, HF_it)
                #loss_t(model, en, ag, t_step, it)
                #loss_t!(model, en, ag, t_step, it)
            end
            Flux.Optimise.update!(opt, Flux.params(model), grads)

            if(it==1) 
                println("First Learning Finish!")
            end

            ag.K_TL[en.t_size,:] = zeros(Float64, en.HS_size^2)
            ll_it[it], Kp_av = loss_calc_hyb!(model,en, ag, HF_it)
        end
        
        #=
        if(it%10 == 0)
            if(it!=it_MAX)
                updata_KpK!(en, ag, Kp_av)
            end
            println("it:"*string(it))
            println(ll_it[it])
        end=#
        #if(ll_it[it]>100.0)
        #    break
        #end
        #=
        if(it%10 == 0)
            ee = zeros(Float64, en.t_size, en.HS_size)
            for t_step in 1:en.t_size
                ee[t_step], v = eigen(vec_to_matrix(ag.HF_TL[t_step,:]))
            end
            save_data1 = DataFrame(transpose(ee))
            CSV.write("./HFt_it="*"$it" *".csv", save_data1)

        end=#
        if(it%500 == 0 && it!=0)
            E = zeros(Float64, en.t_size, en.HS_size)
            for t_step in 1:en.t_size
                E[t_step,:], v = eigen(VtoM(ag.HF_TL[t_step,:],en))
            end

            p1 = plot(E[:,1].-E[1,1], xlabel="t_step", ylabel="E of HF_t", width=3.0)
            p1 = plot!(E[:,2].-E[1,2], width=3.0)
            p1 = plot!(E[:,3].-E[1,3], width=3.0)
            p1 = plot!(E[:,4].-E[1,4], width=3.0)
            savefig(p1,"./HF_t$(it).png")
            println("Drawing Finish!")
            #println(E[:,4])
            p2 = plot(ag.K_TL[:,1], xlabel="t_step", ylabel="E of K_t", width=2.0)
            for i in 2:en.HS_size^2
                p2 = plot!(ag.K_TL[:,i], width=2.0)
            end
            save_data1 = DataFrame(ag.K_TL, :auto)
            CSV.write("./K_TL$(it).csv", save_data1)
            savefig(p2,"./K_t$(it).png")
            p4 = plot(ag.Kp_TL[:,1], xlabel="t_step", ylabel="E of Kp_t", width=2.0)
            for i in 2:en.HS_size^2
                p4 = plot!(ag.Kp_TL[:,i], width=2.0)
            end
            savefig(p4,"./Kp_t$(it).png")
        end

        if(it_MAX==1)
            E = zeros(Float64, en.t_size, en.HS_size)
            for t_step in 1:en.t_size
                E[t_step,:], v = eigen(VtoM(ag.HF_TL[t_step,:],en))
            end

            p1 = plot(E[:,1].-E[1,1], xlabel="t_step", ylabel="E of HF_t", width=3.0)
            p1 = plot!(E[:,2].-E[1,2], width=3.0)
            p1 = plot!(E[:,3].-E[1,3], width=3.0)
            p1 = plot!(E[:,4].-E[1,4], width=3.0)
            savefig(p1,"./HF_t_check_gene.png")
            println("Drawing Finish!")
            #println(E[:,4])
            p2 = plot(ag.K_TL[:,1], xlabel="t_step", ylabel="E of K_t", width=2.0)
            for i in 2:en.HS_size^2
                p2 = plot!(ag.K_TL[:,i], width=2.0)
            end
            save_data1 = DataFrame(ag.K_TL, :auto)
            CSV.write("./K_TL_check_gene.csv", save_data1)
            savefig(p2,"./K_t_check_gene.png")
            p4 = plot(ag.Kp_TL[:,1], xlabel="t_step", ylabel="E of Kp_t", width=2.0)
            for i in 2:en.HS_size^2
                p4 = plot!(ag.Kp_TL[:,i], width=2.0)
            end
            savefig(p4,"./Kp_t_check_gene.png")
            @save "mymodel_check_gene.bson" model
        end
    end
    println("Learning Finish!")
    #=
    E = zeros(Float64, en.t_size, en.HS_size)
    for t_step in 1:en.t_size
        E[t_step,:], v = eigen(VtoM(ag.HF_TL[t_step,:],en))
    end

    println("Eval Finish! Using Plots")

    p1 = plot(E[:,1].-E[1,1], xlabel="t_step", ylabel="E of HF_t", width=3.0)
    p1 = plot!(E[:,2].-E[1,2], width=3.0)
    p1 = plot!(E[:,3].-E[1,3], width=3.0)
    p1 = plot!(E[:,4].-E[1,4], width=3.0)
    savefig(p1,"./HF_t.png")
    println("Drawing Finish!")
    #println(E[:,4])
    p2 = plot(ag.K_TL[:,1], xlabel="t_step", ylabel="E of K_t", width=2.0)
    for i in 2:en.HS_size^2
        p2 = plot!(ag.K_TL[:,i], width=2.0)
    end
    save_data1 = DataFrame(ag.K_TL, :auto)
    CSV.write("./K_TL.csv", save_data1)
    savefig(p2,"./K_t.png")
    p4 = plot(ag.Kp_TL[:,1], xlabel="t_step", ylabel="E of Kp_t", width=2.0)
    for i in 2:en.HS_size^2
        p4 = plot!(ag.Kp_TL[:,i], width=2.0)
    end
    savefig(p4,"./Kp_t.png")=#

    p3 = plot(ll_it, xlabel="it_step", ylabel="loss", yaxis=:log, width=3.0)
    savefig(p3,"./loss_iterate.png")
    println("Drawing Finish!")

    @save "mymodel.bson" model
    
    
end

@time main(ARGS)