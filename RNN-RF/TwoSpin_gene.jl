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
    in_size::Int = en.num_parm + en.HS_size^2 -1

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


function micro_motion(Kp_t::Vector{Float64}, K_t::Vector{Float64}, en::TS_env, t::Int)
    Kt = VtoM(K_t, en)
    U = exp(1.0im*Kt)
    K_t_new = K_t + en.dt * Kp_t
    Kt_new = VtoM(K_t_new,en)
    U_new = exp(1.0im*Kt_new)
    dU = (U_new - U)/en.dt
    HF_m = Hermitian(U*(en.H_0 + en.V_t*sin(2pi*t/en.t_size))*U' -1.0im*(U*dU'-dU*U')/2)
    HF = MtoV(HF_m, en)

    return K_t_new, HF
end

#calculate H_r(t) from K'(t) & K(t)

function micro_motion2(Kp_t::Vector{Float64}, K_t::Vector{Float64}, en::TS_env, t::Int)
    #dt = (2pi/en.t_size/en.Ω)
    Kt = VtoM(K_t, en)
    U = exp(1.0im*Kt)
    K_t_new = K_t + en.dt * Kp_t
    Kt_new = VtoM(K_t_new,en)
    U_new = exp(1.0im*Kt_new)
    dU = (U_new - U)/en.dt
    HF_m = Hermitian(U*(en.H_0 + en.V_t*sin(2pi*t/en.t_size))*U' -1.0im*(dU*dU'-dU*U')/2)
    #HF_m = Hermitian(U*(en.H_0 + en.V_t*sin(2pi*t/en.t_size) - Kp)*U')
    HF = MtoV(HF_m,en)
    return HF
end


function diff_norm(V::Vector{Float64}, en::TS_env)
    M = VtoM(V,en)
    n = tr(M*M)
    #e, v = eigen(M)
    #n::Float64 = V' * V
    #n::Float64 = e' * e
    #n = sum(e[n]^2 for n in 1:size(e))
    return n
end

function diff_L1norm(V::Vector{Float64}, en::TS_env)
    n::Float64 = sum(abs.(V))
    return n
end

#lossの関数
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

#gradient内で変数をHF,Ktを更新する事が出来ないので更新しないversion
function loss_calc_hyb(model0, en::TS_env, ag::agtQ, HF_given::Vector{Float64})
    l::Float64 = 0.0
    kp_sum = zeros(Float64, en.HS_size^2)
    for t in 1:en.t_size
        if(t==1)
            tt=en.t_size
        else
            tt=t-1
        end
        p = [en.ξ*sin(2pi*t/en.t_size), en.Jz, en.Jx, en.hz]
        #x = vcat([p, ag.K_TL[tt,:], ag.Kp_TL[tt,:]]...)
        x = vcat([p, ag.K_TL[tt,:]]...)
        Kp = model0(x)
        kp_sum += Kp

        HF_calc = micro_motion2(Kp, ag.K_TL[tt,:],en,t)
        l += loss_fn_hybrid(en,ag, HF_given, HF_calc,t)

        #=
        if(t==en.t_size)
            l += diff_norm(HF_calc-ag.HF_TL[1,:],en)
        end
        =#
    end
    l += diff_norm(kp_sum,en)/en.t_size
    return l 
end

#更新するversion
function loss_calc_hyb!(model0, en::TS_env, ag::agtQ, HF_given::Vector{Float64})
    l::Float64 = 0.0
    kp_sum = zeros(Float64, en.HS_size^2)
    for t in 1:en.t_size
        if(t==1)
            tt=en.t_size
        else
            tt=t-1
        end
        p = [en.ξ*sin(2pi*t/en.t_size), en.Jz, en.Jx, en.hz]
        #x = vcat([p, ag.K_TL[tt,:], ag.Kp_TL[tt,:]]...)
        x = vcat([p, ag.K_TL[tt,:]]...)

        ag.Kp_TL[t,:] = model0(x)

        kp_sum += ag.Kp_TL[t,:]

        ag.K_TL[t,:], ag.HF_TL[t,:] = micro_motion(ag.Kp_TL[t,:], ag.K_TL[tt,:],en,t)
        l += loss_fn_hybrid(en,ag, HF_given, ag.HF_TL[t,:],t)
        #=
        if(t==en.t_size)
            l += diff_norm(ag.HF_TL[t,:]-ag.HF_TL[1,:],en)
        end
        =#
    end
    l += diff_norm(kp_sum,en)/en.t_size
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

function micro_motion2(Kp_t::Vector{Float64}, ag::agtQ, en::TS_env, t::Int)
    if(t==1)
        #Kt = VtoM(K_t, en)
        KtM = zeros(Float64, en.HS_size, en.HS_size)
        KtpM = VtoM(Kp_t*en.dt ,en)
        KtmM = VtoM(-Kp_t*en.dt,en)
        Ud = exp(-1.0im*KtM)
        U = Ud'
        Ud_p = (exp(-1.0im*KtpM)-exp(-1.0im*KtmM))/(2en.dt)
        HF_m = U*(en.H_0 + sin(2pi*(t-1)/en.t_size)*en.V_t)*Ud -1.0im*U*Ud_p
        HF_m = Hermitian(HF_m)
    elseif(t==2)
        KtM = VtoM(ag.K_TL[t-1,:], en)
        KtpM = VtoM(ag.K_TL[t-1,:]+Kp_t*en.dt ,en)
        KtmM = zeros(Float64, en.HS_size, en.HS_size)
        Ud = exp(-1.0im*KtM)
        U = Ud'
        Ud_p = (exp(-1.0im*KtpM)-exp(-1.0im*KtmM))/(2en.dt)
        HF_m = U*(en.H_0 + sin(2pi*(t-1)/en.t_size)*en.V_t)*Ud -1.0im*U*Ud_p
        HF_m = Hermitian(HF_m)
    else
        #Kt = VtoM(K_t, en)
        KtM = VtoM(ag.K_TL[t-1,:], en)
        KtpM = VtoM(ag.K_TL[t-1,:]+Kp_t*en.dt ,en)
        KtmM = VtoM(ag.K_TL[t-2,:],en)
        Ud = exp(-1.0im*KtM)
        U = Ud'
        Ud_p = (exp(-1.0im*KtpM)-exp(-1.0im*KtmM))/(2en.dt)
        HF_m = U*(en.H_0 + sin(2pi*(t-1)/en.t_size)*en.V_t)*Ud -1.0im*U*Ud_p
        HF_m = Hermitian(HF_m)
    end
    #HF_m = Hermitian(U*(en.H_0 + en.V_t*sin(2pi*t/en.t_size) - Kp)*U')
    HF = MtoV(HF_m,en)
    return HF
end
function loss_fn_hybrid2(en::TS_env, ag::agtQ, HF_calc::Vector{Float64}, t::Int)
    l::Float64 = 0.0
    if(t>2)
        for i in 1:t-2
            l += ag.γ^(i) * diff_norm((HF_calc-ag.HF_TL[t-1-i,:]),en)*en.dt
        end
    end
    #l += diff_norm(HF_given - HF_calc,en)/en.t_size
    return l
end

function loss_calc_hyb2(model0, en::TS_env, ag::agtQ)
    l::Float64 = 0.0
    kp_sum = zeros(Float64, en.HS_size^2)
    for t in 1:en.t_size
        if(t==1)
            tt=en.t_size
        else
            tt=t-1
        end
        p = [en.ξ*sin(2pi*t/en.t_size), en.Jz, en.Jx, en.hz]
        x = vcat([p, ag.K_TL[tt,:]]...)
        Kp = model0(x)
        kp_sum += Kp*en.dt

        HF_calc = micro_motion2(Kp, ag,en,t)
        l += loss_fn_hybrid2(en,ag, HF_calc,t)
    end
    l += ag.ϵ*diff_norm(kp_sum,en)
    return l 
end

function loss_calc_hyb2!(model0, en::TS_env, ag::agtQ)
    l::Float64 = 0.0
    kp_sum = zeros(Float64, en.HS_size^2)
    for t in 1:en.t_size
        if(t==1)
            tt=en.t_size
        else
            tt=t-1
        end
        p = [en.ξ*sin(2pi*t/en.t_size), en.Jz, en.Jx, en.hz]
        x = vcat([p, ag.K_TL[tt,:]]...)
        ag.Kp_TL[t,:] = model0(x)
        kp_sum += ag.Kp_TL[t,:]*en.dt
        ag.K_TL[t,:] = kp_sum 

        ag.HF_TL[tt,:] = micro_motion2(ag.Kp_TL[t,:], ag,en,t)
        l += loss_fn_hybrid2(en,ag, ag.HF_TL[tt,:],t)
    end
    l += ag.ϵ*diff_norm(kp_sum,en)
    return l 
end

function loss_fn_hybrid3(en::TS_env, ag::agtQ, HF_calc::Vector{Float64}, HF_sum::Vector{Float64}, t::Int)
    l::Float64 = 0.0
    if(t>1)
        l += ag.ϵ * diff_norm((HF_calc-HF_sum),en)/en.t_size
    end
    #l += diff_norm(HF_given - HF_calc,en)/en.t_size
    return l
end

function loss_calc_hyb3(model0, en::TS_env, ag::agtQ)
    l::Float64 = 0.0
    kp_sum = zeros(Float64, en.HS_size^2)
    HF_sum = zeros(Float64, en.HS_size^2)
    aa::Float64 = 1.0
    for t in 1:en.t_size
        if(t==1)
            tt=en.t_size
        else
            tt=t-1
        end
        p = [en.ξ*sin(2pi*t/en.t_size), en.Jz, en.Jx, en.hz]
        x = vcat([p, ag.K_TL[tt,:]]...)
        Kp = model0(x)
        kp_sum += Kp*en.dt

        HF_calc = micro_motion2(Kp, ag,en,t)
        l += loss_fn_hybrid3(en,ag, HF_calc, (HF_sum/aa),t)
        HF_sum += HF_calc
        HF_sum *= ag.γ
        aa += ag.γ^t
    end
    l += diff_norm(kp_sum,en)
    return l 
end

function loss_calc_hyb3!(model0, en::TS_env, ag::agtQ)
    l::Float64 = 0.0
    kp_sum = zeros(Float64, en.HS_size^2)
    HF_sum = zeros(Float64, en.HS_size^2)
    aa::Float64 = 1.0
    for t in 1:en.t_size
        if(t==1)
            tt=en.t_size
        else
            tt=t-1
        end
        p = [en.ξ*sin(2pi*t/en.t_size), en.Jz, en.Jx, en.hz]
        x = vcat([p, ag.K_TL[tt,:]]...)
        ag.Kp_TL[t,:] = model0(x)
        kp_sum += ag.Kp_TL[t,:]*en.dt
        ag.K_TL[t,:] = kp_sum 

        ag.HF_TL[tt,:] = micro_motion2(ag.Kp_TL[t,:], ag,en,t)
        l += loss_fn_hybrid3(en,ag, ag.HF_TL[tt,:], (HF_sum/aa),t)
        HF_sum += ag.HF_TL[tt,:]
        HF_sum *= ag.γ
        aa += ag.γ^t
    end
    l += diff_norm(kp_sum,en)
    return l 
end

function micro_motion_new(Kp::Vector{Float64}, kp_sum::Vector{Float64}, kt_old::Vector{Float64},en::TS_env,t::Int)
    KtM = VtoM(kp_sum,en)
    KtpM = VtoM(kp_sum + Kp*en.dt ,en)
    KtmM = VtoM(kt_old,en)
    Ud = exp(-1.0im*KtM)
    U = Ud'
    Ud_p = (exp(-1.0im*KtpM)-exp(-1.0im*KtmM))/(2en.dt)
    HfM = U*(en.H_0 + sin(2pi*(t-1)/en.t_size)*en.V_t)*Ud -1.0im*U*Ud_p
    HfM = Hermitian(HfM)
    return HfM
end

function loss_calc_new(model0, en::TS_env, ag::agtQ)
    l::Float64 = 0.0
    kp_sum = zeros(Float64, en.HS_size^2)
    kt_old = zeros(Float64, en.HS_size^2)
    HF_old = zeros(Float64, en.HS_size, en.HS_size)
    for t in 1:en.t_size
        p = [en.ξ*sin(2pi*t/en.t_size), en.Jz, en.Jx, en.hz]
        #x = vcat([p, ag.K_TL[tt,:], ag.Kp_TL[tt,:]]...)
        x = vcat([p, kp_sum]...)
        Kp = model0(x)
        if(t>1)
            HF_calc = micro_motion_new(Kp, kp_sum, kt_old,en,t)
            del = HF_calc-HF_old
            l += ag.ϵ*tr(del*del)
            HF_old = HF_calc
        end
        kt_old = kp_sum
        kp_sum += Kp * en.dt
    end
    l += diff_norm(kp_sum,en)
    return l 
end

function loss_calc_new!(model0, en::TS_env, ag::agtQ)
    l::Float64 = 0.0
    kp_sum = zeros(Float64, en.HS_size^2)
    kt_old = zeros(Float64, en.HS_size^2)
    HF_old = zeros(Float64, en.HS_size, en.HS_size)
    for t in 1:en.t_size
        p = [en.ξ*sin(2pi*t/en.t_size), en.Jz, en.Jx, en.hz]
        #x = vcat([p, ag.K_TL[tt,:], ag.Kp_TL[tt,:]]...)
        x = vcat([p, kp_sum]...)
        ag.Kp_TL[t,:] = model0(x)
        if(t>1)
            HF_calc = micro_motion_new(ag.Kp_TL[t,:], kp_sum, kt_old,en,t)
            del = HF_calc-HF_old
            l += ag.ϵ*tr(del*del)
            HF_old = HF_calc
            ag.HF_TL[t-1,:] = MtoV(HF_calc, en)
        end
        kt_old = kp_sum
        kp_sum += ag.Kp_TL[t,:] * en.dt
        ag.K_TL[t,:] = kp_sum
    end
    l += diff_norm(kp_sum,en)
    return l 
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

    if(arg[1]=="clip1")
        opt = Flux.Optimise.Optimiser(ClipValue(1e-1),Adam(1e-1))
    elseif(arg[1]=="clip2")
        opt = Flux.Optimise.Optimiser(ClipValue(1e-2),Adam(1e-2))
    elseif(arg[1]=="clip3")
        opt = Flux.Optimise.Optimiser(ClipValue(1e-3),Adam(1e-3))
    elseif(arg[1]=="rms")
        opt = RMSProp()
    elseif(arg[1]=="gd")
        opt = Descent()
    else
        opt = ADAM()
    end

    it_MAX = parse(Int,arg[2])
    b_MAX = parse(Int,arg[3])
    ll_it = zeros(Float64, it_MAX)
    println("start!")
    t_size = parse(Int,arg[4])
    H_size = parse(Int,arg[5])
    width = parse(Int,arg[6])
    Ω = parse(Float64,arg[7])
    γ = parse(Float64,arg[8])
    ϵ = parse(Float64,arg[9])
    model = Chain(Dense(20, width, tanh), Dense(width, width, tanh), Dense(width, H_size^2))
    for b in 1:b_MAX
        xi = rand()
        Jz = rand()
        Jx = rand()
        hz = rand()
        println("xi=$(xi), Jz=$(Jz), Jx=$(Jx), hz=$(hz)")
        en = TS_env(init_env(t_size, Ω, xi, Jz, Jx, hz)...)
        ag = agtQ(init_nQ(en,width,γ,ϵ)...)
        ag.K_TL[en.t_size,:] = zeros(Float64, en.HS_size^2)
        #model = Chain(Dense(zeros(Float64, ag.n_dense, ag.in_size), zeros(Float64, ag.n_dense), tanh), Dense(zeros(Float64, ag.n_dense, ag.n_dense), zeros(Float64, ag.n_dense), tanh), Dense(zeros(Float64, ag.out_size, ag.n_dense), zeros(Float64, ag.out_size)))

        for it in 1:it_MAX
            
            if(it==1)
                k_sum = zeros(Float64, en.HS_size^2)
                for t_step in 1:en.t_size
                    if(t_step==1)
                        tt=en.t_size
                    else
                        tt=t_step-1
                    end
                    p = [en.ξ*sin(2pi*t_step/en.t_size), en.Jz, en.Jx, en.hz]
                    x = vcat([p, ag.K_TL[tt,:]]...)

                    ag.Kp_TL[t_step,:] = model(x)
                    k_sum += ag.Kp_TL[t_step,:] * en.dt
                    ag.K_TL[t_step,:] = k_sum
                    ag.HF_TL[t_step,:] = micro_motion2(ag.Kp_TL[t_step,:], ag, en, t_step)
                end
                println("HF_calc Finish!")
            end

            grads = Flux.gradient(Flux.params(model)) do
                #loss_calc_new(model, en, ag)
                loss_calc_hyb2(model, en, ag)
                #loss_calc_hyb3(model, en, ag)
            end
            Flux.Optimise.update!(opt, Flux.params(model), grads)

            if(it==1) 
                println("First Learning Finish!")
            end

            ag.K_TL[en.t_size,:] = zeros(Float64, en.HS_size^2)
            ll_it[(b-1)*it_MAX + it] = loss_calc_hyb2!(model,en, ag)

            if(it == it_MAX)
                @save "mymodel$(b).bson" model
            end
        end
        

    end
    println("Training Finish!")
    println("=====================")

    xi = parse(Float64,arg[10])
    Jz = parse(Float64,arg[11])
    Jx = parse(Float64,arg[12])
    hz = parse(Float64,arg[13])
    println("test: xi=$(xi), Jz=$(Jz), Jx=$(Jx), hz=$(hz)")
    en = TS_env(init_env(t_size, Ω, xi, Jz, Jx, hz)...)
    ag = agtQ(init_nQ(en,width,γ,ϵ)...)
    ag.K_TL[en.t_size,:] = zeros(Float64, en.HS_size^2)
    ll_test = loss_calc_hyb2!(model,en, ag)
    println("test Finish!")
    println("=====================")

    save_data2 = DataFrame(ag.HF_TL, :auto)
    CSV.write("./HF_TL.csv", save_data2)

    p2 = plot(ag.K_TL[:,1], xlabel="t_step", ylabel="E of K_t", width=2.0)
    for i in 2:en.HS_size^2
        p2 = plot!(ag.K_TL[:,i], width=2.0)
    end
    savefig(p2,"./Kt_test.png")
    

    save_data0 = DataFrame(ag.K_TL, :auto)
    CSV.write("./K_TL_test.csv", save_data0)

    

    p3 = plot(ll_it, xlabel="it_step", ylabel="loss", yaxis=:log, width=3.0)
    savefig(p3,"./loss_iterate.png")
    save_data_l = DataFrame(loss = ll_it)
    CSV.write("./loss.csv", save_data_l)

    E = zeros(Float64, en.t_size, en.HS_size)
    for t_step in 1:en.t_size
        E[t_step,:], v = eigen(VtoM(ag.HF_TL[t_step,:],en))
    end
    p1 = plot(E[1:end-1,1].-E[1,1], xlabel="t_step", ylabel="E of HF_t", width=3.0)
    p1 = plot!(E[1:end-1,2].-E[1,2], width=3.0)
    p1 = plot!(E[1:end-1,3].-E[1,3], width=3.0)
    p1 = plot!(E[1:end-1,4].-E[1,4], width=3.0)
    savefig(p1,"./E_HF_test.png")

    
    println("Drawing Finish!")
    
end

@time main(ARGS)
