using Flux
include("TwoSpin_env.jl")

#考えたい系のパラメータによらない部分をまとめた構造体（機械学習に関するハイパーパラメータ）
struct agtQ
    in_size::Int
    out_size::Int
    n_dense::Int
    ϵ::Float32
    γ::Float32
    t_size::Int
end

function init_nQ(n::Int, γ0::Float32, ϵ0::Float32, t::Int)
    #width of the input layer (\Omega,  ξ, J_z, J_x, h_z, K(t))
    in_size::Int = 5 + 4^2
    #the size of K'(t)
    out_size::Int = 4^2
    #width of the hidden layers
    n_dense::Int = n
    # inverse weight of the time-periodicity 
    ϵ::Float32 = ϵ0
    #discount rate
    γ::Float32 = γ0
    t_size::Int = t

    return in_size, out_size, n_dense, ϵ, γ, t_size
    
end

function init_nQ(arg::Array{String,1})
    #width of the input layer (\Omega,  ξ, J_z, J_x, h_z, K(t))
    in_size::Int = 5 + 4^2

    #the size of K'(t)
    out_size::Int = 4^2

    #width of the hidden layers
    n_dense::Int = parse(Int,arg[1])

    # inverse weight of the time-periodicity 
    ϵ::Float32 = parse(Float32,arg[2])

    #discount rate
    γ::Float32 = parse(Float32,arg[3])

    t_size::Int = parse(Int,arg[4])

    return in_size, out_size, n_dense, ϵ, γ, t_size
end


function Ht(input_parm::Vector{Float32}, t::Int, t_size::Int)
    #input_parm = [J_z, J_x, h_z, xi_t]
    H_0 = Hermitian([ -input_parm[2]-2input_parm[4] 0.0f0im 0 -input_parm[3]; 0 input_parm[2] -input_parm[3] 0; 0 -input_parm[3] input_parm[2] 0; -input_parm[3] 0 0 -input_parm[2]+2input_parm[4]])
    V_t = Hermitian([ 0 -1 -1 0; -1 0 0 -1; -1 0 0 -1; 0.0f0im -1 -1 0]*input_parm[5]*sin(Float32(2pi*t/t_size)))
    return (H_0 + V_t)
end


#calculate H_r(t) from K'(t) & K(t)
#=
function micro_motion2(Kp_t::Vector{Float32},input_parm::Vector{Float32}, Kt_data::Matrix{Float32}, t::Int, t_size::Int)
    dt = Float32(2pi/t_size/input_parm[1])
    if(t==1)
        #Kt = VtoM(K_t, en)
        #KtM = zeros(Float32, 4, 4)
        KtpM = VtoM(Kp_t*dt)
        KtmM = VtoM(-Kp_t*dt)
        Ud = Matrix{ComplexF32}(I, 4, 4)
        #exp(-1f0im*KtM)
        U = Ud'
        Ud_p = (exp(-1f0im*KtpM)-exp(-1f0im*KtmM))/(2dt)
        HF_m = ComplexF32.(U*Ht(input_parm, t, t_size)*Ud -1f0im*U*Ud_p)
        HF_m = Hermitian(HF_m)
    elseif(t==2)
        KtM = VtoM(Kt_data[t-1,:])
        KtpM = VtoM(Kt_data[t-1,:]+Kp_t*dt)
        KtmM = zeros(Float32, 4, 4)
        Ud = exp(-1f0im*KtM)
        U = Ud'
        Ud_p = (exp(-1f0im*KtpM)-exp(-1f0im*KtmM))/(2dt)
        HF_m = ComplexF32.(U*Ht(input_parm, t, t_size)*Ud -1f0im*U*Ud_p)
        HF_m = Hermitian(HF_m)
    else
        #Kt = VtoM(K_t, en)
        KtM = VtoM(Kt_data[t-1,:])
        KtpM = VtoM(Kt_data[t-1,:]+Kp_t*dt)
        KtmM = VtoM(Kt_data[t-2,:])
        Ud = exp(-1f0im*KtM)
        U = Ud'
        Ud_p = (exp(-1f0im*KtpM)-exp(-1f0im*KtmM))/(2dt)
        HF_m = ComplexF32.(U*Ht(input_parm, t, t_size)*Ud -1f0im*U*Ud_p)
        HF_m = Hermitian(HF_m)
    end
    #HF_m = Hermitian(U*(en.H_0 + en.V_t*sin(2pi*t/en.t_size) - Kp)*U')
    HF = MtoV(HF_m)
    return HF
end=#

function calc_hf(en::Env, Kp_t::Vector{Float32},input_parm::Vector{Float32}, Kt_data::Matrix{Float32}, t::Int, t_size::Int)
    dt = Float32(2pi/t_size/input_parm[1])
    
    if(t==1)
        KtpM = VtoM(Kp_t*dt, en)
        KtmM = VtoM(-Kp_t*dt, en)
        Ud = Matrix{ComplexF32}(I, 4, 4)
        U = Ud'
        Ud_p = (exp(-1f0im*KtpM)-exp(-1f0im*KtmM))/(2dt)    
    elseif(t==2)
        KtM = VtoM(Kt_data[t-1,:], en)
        KtpM = VtoM(Kt_data[t-1,:]+Kp_t*dt, en)
        KtmM = zeros(Float32, 4, 4)
        Ud = exp(-1f0im*KtM)
        U = Ud'
        Ud_p = (exp(-1f0im*KtpM)-exp(-1f0im*KtmM))/(2dt)
    else
        KtM = VtoM(Kt_data[t-1,:], en)
        KtpM = VtoM(Kt_data[t-1,:]+Kp_t*dt, en)
        KtmM = VtoM(Kt_data[t-2,:], en)
        Ud = exp(-1f0im*KtM)
        U = Ud'
        Ud_p = (exp(-1f0im*KtpM)-exp(-1f0im*KtmM))/(2dt)
    end

    HF_0 = Hermitian(U*Ht(input_parm, t, t_size)*Ud -1f0im*U*Ud_p)
    HF_m = MtoV(HF_0,en)

    return HF_m
end

function micro_motion2(en::Env, Kp_t::Matrix{Float32},input_parm::Matrix{Float32}, Kt_data::Array{Float32,3}, t::Int, t_size::Int, bs::Int)
    HF = hcat(([calc_hf(en, Kp_t[:,b], input_parm[:,b], Kt_data[:,:,b], t, t_size) for b in 1:bs])...)
    return HF
end

function micro_motion!(en::Env, Kp_t::Vector{Float32},input_parm::Vector{Float32}, Kt_data::Matrix{Float32}, t::Int, t_size::Int)
    dt = Float32(2pi/t_size/input_parm[1])
    if(t==1)
        #Kt = VtoM(K_t, en)
        #KtM = zeros(Float32, 4, 4)
        Kt_data[t,:] = Kp_t*dt
        KtpM = VtoM(Kt_data[t,:],en)
        KtmM = VtoM(-Kp_t*dt,en)
        Ud = Matrix{ComplexF32}(I, 4, 4)
        #exp(-1f0im*KtM)
        U = Ud'
        Ud_p = (exp(-1f0im*KtpM)-exp(-1f0im*KtmM))/(2dt)
        HF_m = ComplexF32.(U*Ht(input_parm, t, t_size)*Ud -1f0im*U*Ud_p)
        HF_m = Hermitian(HF_m)
    elseif(t==2)
        KtM = VtoM(Kt_data[t-1,:],en)
        Kt_data[t,:] = Kt_data[t-1,:]+Kp_t*dt
        KtpM = VtoM(Kt_data[t,:],en)
        KtmM = zeros(Float32, 4, 4)
        Ud = exp(-1f0im*KtM)
        U = Ud'
        Ud_p = (exp(-1f0im*KtpM)-exp(-1f0im*KtmM))/(2dt)
        HF_m = ComplexF32.(U*Ht(input_parm, t, t_size)*Ud -1f0im*U*Ud_p)
        HF_m = Hermitian(HF_m)
    else
        #Kt = VtoM(K_t, en)
        KtM = VtoM(Kt_data[t-1,:],en)
        Kt_data[t,:] = Kt_data[t-1,:]+Kp_t*dt
        KtpM = VtoM(Kt_data[t,:],en)
        KtmM = VtoM(Kt_data[t-2,:],en)
        Ud = exp(-1f0im*KtM)
        U = Ud'
        Ud_p = (exp(-1f0im*KtpM)-exp(-1f0im*KtmM))/(2dt)
        HF_m = ComplexF32.(U*Ht(input_parm, t, t_size)*Ud -1f0im*U*Ud_p)
        HF_m = Hermitian(HF_m)
    end
    #HF_m = Hermitian(U*(en.H_0 + en.V_t*sin(2pi*t/en.t_size) - Kp)*U')
    HF = MtoV(HF_m,en)
    return HF
end



#caclulate L2 norm for matrix
function diff_norm(V::Vector{Float32}, en::Env)
    M = VtoM(V, en)
    n = real(tr(M*M))
    return n
end
function diff_norm(V::Matrix{Float32}, en::Env)
    bs = size(V)[2]
    d = 0.0
    for b in 1:bs
        M = VtoM(V[:,b],en)
        d += real(tr(M*M))
    end
    return d
end

#calculate the loss function in a single cycle 
#=
function loss_calc_hyb(model0, input_parm::Vector{Float32}, Kt_data::Matrix{Float32}, HF_given::Vector{Float32},en::Env, ag::agtQ, bs::Int)
    l::Float32 = 0.0
    kp_sum = zeros(Float32, 4^2)
    for t in 1:ag.t_size
        if(t==1)
            tt=ag.t_size
        else
            tt=t-1
        end
        x = vcat([input_parm, Kt_data[tt,:]]...)
        Kp = model0(x)
        kp_sum += Kp

        HF_calc = micro_motion2(en, Kp, input_parm, Kt_data, t, ag.t_size)
        l += ag.ϵ * diff_norm((HF_calc-HF_given))
        #l += loss_fn_hybrid(en,ag, HF_given, HF_calc,t)
    end
    l += diff_norm(kp_sum)/ag.t_size
    return l 
end=#

function loss_t(en::Env, ag::agtQ, HF_calc::Matrix{Float32}, HF_given::Array{Float32, 3}, t::Int)
    l = 0.0f0
    for tt in 1:ag.t_size
        d = abs(tt-t)
        if(d>ag.t_size/2)
            d = ag.t_size-d
        end
        l += ag.γ^d * diff_norm(HF_calc-HF_given[tt,:,:], en)
    end
    return l/ag.t_size
end

function loss_tt(en::Env, HF_calc::Matrix{Float32}, HF_given::Array{Float32, 3},γ::Float32, t::Int)
    l = 0.0f0
    for tt in 1:t
        l += γ^(t-tt) * diff_norm(HF_calc-HF_given[tt,:,:], en)
    end
    return l/t
end

#function loss_calc_hyb(model0::Chain, input_parm::Matrix{Float32}, Kt_data::Array{Float32, 3}, HF_given::Matrix{Float32}, ag::agtQ)
function loss_calc_hyb(model0::Chain, input_parm::Matrix{Float32}, Kt_data::Array{Float32, 3}, HF_given::Array{Float32, 3},en::Env,  ag::agtQ, bs::Int)
    l::Float32 = 0.0
    #bs = size(input_parm)[2]
    kp_sum = zeros(Float32, 4^2, bs)
    for t in 1:ag.t_size
        if(t==1)
            tt=ag.t_size
        else
            tt=t-1
        end
        x = vcat([input_parm, Kt_data[tt,:,:]]...)
        Kp = model0(x)
        kp_sum += Kp
        #=
        if(t == div(ag.t_size,2))
            @time HF_calc = micro_motion2(en, Kp, input_parm, Kt_data, t, ag.t_size, bs)
        #l += ag.ϵ * diff_norm((HF_calc-HF_given[tt,:,:]), en)/ag.t_size
            @time l += ag.ϵ * loss_tt(en, HF_calc, HF_given,ag.γ, t)/ag.t_size
        else=#
            HF_calc = micro_motion2(en, Kp, input_parm, Kt_data, t, ag.t_size, bs)
            #l += ag.ϵ * diff_norm((HF_calc-HF_given[tt,:,:]), en)/ag.t_size
            l += ag.ϵ * loss_tt(en, HF_calc, HF_given,ag.γ, t)/ag.t_size
            #l += loss_fn_hybrid(en,ag, HF_given, HF_calc,t)
        #end
    end
    l += diff_norm(kp_sum, en)
    return l /bs
end

function loss_calc_hyb_t(model0::Chain, input_parm::Matrix{Float32}, Kt_data::Array{Float32, 3}, HF_given::Array{Float32, 3},en::Env,  ag::agtQ, bs::Int, t::Int)
    l::Float32 = 0.0
    if(t==1)
        tt=ag.t_size
    else
        tt=t-1
    end
    x = vcat([input_parm, Kt_data[tt,:,:]]...)
    Kp = model0(x)
    HF_calc = micro_motion2(en, Kp, input_parm, Kt_data, t, ag.t_size, bs)
    l = ag.ϵ * loss_t(en, ag, HF_calc, HF_given, t)/bs
    return l
end

function check_dynamics(model0::Chain, input::Vector{Float32}, en::Env, ag::agtQ)
    Kt = zeros(Float32, ag.t_size, 4^2)
    HF = zeros(Float32, ag.t_size, 4^2)
    #println("input:$(length(input))")
    #println("Kt:$(length(Kt[1,:]))")
    for t in 1:ag.t_size
        if(t==1)
            tt=ag.t_size
        else
            tt=t-1
        end
        #input = vcat([input, Kt[tt,:]]...)
        input0 = reshape(vcat([input, Kt[tt,:]]...), ag.in_size, 1)
        Kp = model0(input0)
        HF[t,:] = micro_motion!(en, Kp[:,1], input0[1:5], Kt, t, ag.t_size)
    end
    return Kt, HF
end



function inp2string(input::Vector{Float32})
    s = "Ω$(input[1])_Jz$(input[2])_Jx$(input[3])_hz$(input[4])_ξ$(input[5])"
    return s
end

using DataFrames
using CSV
using BSON: @save
using BSON: @load
using Plots
using StatsBase

ENV["GKSwstype"]="nul"

function main(arg::Array{String,1})
    Ω = parse(Float32,arg[1]) 
    Jz = parse(Float32, arg[2])
    Jx = parse(Float32, arg[3])
    hz = parse(Float32, arg[4]) 
    ξ = parse(Float32, arg[5])

    en = set_Env(4)

    ag = agtQ(init_nQ(arg[6:9])...)

    opt = Flux.Optimiser(WeightDecay(1f-6), Adam(1f-3))

    it_MAX = parse(Int,arg[10])
    ll_it = zeros(Float32, it_MAX)
    println("start!")
    batch_num = parse(Int,arg[11])
    depth = parse(Int,arg[12])

    st::Int = 0
    if(arg[13]=="init")
        #model = Chain(Dense(ag.in_size, ag.n_dense), BatchNorm(ag.n_dense), Dense(ag.n_dense, ag.n_dense, relu), Dense(ag.n_dense, ag.n_dense, relu), Dense(ag.n_dense, ag.n_dense, relu), Dense(ag.n_dense, ag.n_dense, relu), Dense(ag.n_dense, ag.out_size))
        model = Chain(Dense(ag.in_size, ag.n_dense), BatchNorm(ag.n_dense), Tuple(Chain(Dense(ag.n_dense, ag.n_dense, relu)) for i in 1:depth)..., Dense(ag.n_dense, ag.out_size))
    else
        @load arg[13] model
    end

    
    input = zeros(Float32, 5, batch_num)
    for i in 1:batch_num
        input[:,i] = generate_parmv(Ω, Jz, Jx, hz, ξ)
    end

    Kt = zeros(Float32, ag.t_size, 4^2, batch_num)
    HF = zeros(Float32, ag.t_size, 4^2, batch_num)
    
    for it in 1:it_MAX
        if(it==1)
            for b in 1:batch_num
                Kt[:,:,b], HF[:,:,b] = check_dynamics(model, input[:,b],en, ag)
            end
            println("HF_calc Finish!")
        end

        if(it_MAX==1)
            E = zeros(Float32, en.t_size, en.HS_size)
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
            break
        end
        #=
        ll_it[it], grads = Flux.withgradient(Flux.params(model)) do
            #loss_calc_hyb(model, en, ag, HF_it)
            loss_calc_hyb(model, input, Kt, HF,en, ag, batch_num)
        end
        Flux.Optimise.update!(opt, Flux.params(model), grads)
        =#
        for t in 1:ag.t_size
            l, grads = Flux.withgradient(Flux.params(model)) do
                loss_calc_hyb_t(model, input, Kt, HF,en, ag, batch_num,t)
            end
            Flux.Optimise.update!(opt, Flux.params(model), grads)
            ll_it[it] += l
            for b in 1:batch_num
                Kt[:,:,b], HF[:,:,b] = check_dynamics(model, input[:,b],en, ag)
            end
        end

        if(it==1) 
            println("First Learning Finish!")
        end

        
        if(it%(div(it_MAX, 10))==0)
            print("#")
            #opt = Flux.Optimiser(WeightDecay(1f-6), Adam(1f-3))
        end
        if(it == it_MAX)
            
            s = 3
            ss = sample(1:batch_num, s, replace=false, ordered=true)
            for b in ss
                E = zeros(Float32, ag.t_size, 4)
                for t_step in 1:ag.t_size
                    E[t_step,:], v = eigen(VtoM(HF[t_step,:,b],en))
                end

                p1 = plot(E[:,1].-E[1,1], xlabel="t_step", ylabel="E of HF_t", width=3.0)
                p1 = plot!(E[:,2].-E[1,2], width=3.0)
                p1 = plot!(E[:,3].-E[1,3], width=3.0)
                p1 = plot!(E[:,4].-E[1,4], width=3.0)
                savefig(p1,"./HF_t$(it)_"*inp2string(input[:,b])*".png")
                println("Drawing Finish!")
                #println(E[:,4])
                p2 = plot(Kt[:,1,b], xlabel="t_step", ylabel="E of K_t", width=2.0)
                for i in 2:16
                    p2 = plot!(Kt[:,i,b], width=2.0)
                end
                savefig(p2,"./K_t$(it)_"*inp2string(input[:,b])*".png")
                save_data1 = DataFrame(Kt[:,:,b], :auto)
                CSV.write("./K_TL$(it)_"*inp2string(input[:,b])*".csv", save_data1)
            end
            @save "mymodel$(st+it).bson" model
        end

    end
    println(":Learning Finish!")
    
    p3 = plot(ll_it, xlabel="it_step", ylabel="loss", yaxis=:log, width=3.0)
    savefig(p3,"./loss_iterate.png")
    save_data_l = DataFrame(loss = ll_it)
    CSV.write("./loss.csv", save_data_l)
    println("Drawing Finish!")
    
end

@time main(ARGS)