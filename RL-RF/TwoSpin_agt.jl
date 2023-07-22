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

function init_nQ(n::Int=32, γ0::Float32=9f-1, ϵ0::Float32=1f0, t::Int)
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
    H_0::Hermitian{ComplexF32, Matrix{ComplexF32}} = Hermitian([ -input_parm[2]-2input_parm[4] 0 0 -input_parm[3]; 0 input_parm[2] -input_parm[3] 0; 0 -input_parm[3] input_parm[2] 0; -input_parm[3] 0 0 -input_parm[2]+2input_parm[4]])
    V_t::Hermitian{ComplexF32, Matrix{ComplexF32}} = Hermitian([ 0 -1 -1 0; -1 0 0 -1; -1 0 0 -1; 0 -1 -1 0]*input_parm[5]* sin(Float32(2pi*t/t_size)))
    return (H_0 + V_t)
end


#calculate H_r(t) from K'(t) & K(t)
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
end

function micro_motion!(Kp_t::Vector{Float32},input_parm::Vector{Float32}, Kt_data::Matrix{Float32}, t::Int, t_size::Int)
    dt = Float32(2pi/t_size/input_parm[1])
    if(t==1)
        #Kt = VtoM(K_t, en)
        #KtM = zeros(Float32, 4, 4)
        Kt_data[t,:] = Kp_t*dt
        KtpM = VtoM(Kt_data[t,:])
        KtmM = VtoM(-Kp_t*dt)
        Ud = Matrix{ComplexF32}(I, 4, 4)
        #exp(-1f0im*KtM)
        U = Ud'
        Ud_p = (exp(-1f0im*KtpM)-exp(-1f0im*KtmM))/(2dt)
        HF_m = ComplexF32.(U*Ht(input_parm, t, t_size)*Ud -1f0im*U*Ud_p)
        HF_m = Hermitian(HF_m)
    elseif(t==2)
        KtM = VtoM(Kt_data[t-1,:])
        Kt_data[t,:] = Kt_data[t-1,:]+Kp_t*dt
        KtpM = VtoM(Kt_data[t,:])
        KtmM = zeros(Float32, 4, 4)
        Ud = exp(-1f0im*KtM)
        U = Ud'
        Ud_p = (exp(-1f0im*KtpM)-exp(-1f0im*KtmM))/(2dt)
        HF_m = ComplexF32.(U*Ht(input_parm, t, t_size)*Ud -1f0im*U*Ud_p)
        HF_m = Hermitian(HF_m)
    else
        #Kt = VtoM(K_t, en)
        KtM = VtoM(Kt_data[t-1,:])
        Kt_data[t,:] = Kt_data[t-1,:]+Kp_t*dt
        KtpM = VtoM(Kt_data[t,:])
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
end

#caclulate L2 norm for matrix
function diff_norm(V::Vector{Float32})
    M = VtoM(V)
    n = tr(M*M)
    return n
end
function diff_norm(V::Vector{Float32})
    M = VtoM(V)
    n = tr(M*M)
    return n
end

#calculate the loss function in a single cycle 
function loss_calc_hyb(model0, input_parm::Vector{Float32}, Kt_data::Matrix{Float32}, HF_given::Vector{Float32}, ag::agtQ)
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

        HF_calc = micro_motion2(Kp, input_parm, Kt_data, t, ag.t_size)
        l += ag.ϵ * diff_norm((HF_calc-HF_given))
        #l += loss_fn_hybrid(en,ag, HF_given, HF_calc,t)
    end
    l += diff_norm(kp_sum)/ag.t_size
    return l 
end

function loss_calc_hyb(model0, input_parm::Matrix{Float32}, Kt_data::Array{Float32, 3}, HF_given::Matrix{Float32}, ag::agtQ)
    l::Float32 = 0.0
    bs = size(input_parm)[2]
    kp_sum = zeros(Float32, 4^2, bs)
    for t in 1:ag.t_size
        if(t==1)
            tt=ag.t_size
        else
            tt=t-1
        end
        x = vcat([input_parm, (Kt_data[:,tt,:])']...)
        Kp = model0(x)
        kp_sum += Kp

        HF_calc = micro_motion2(Kp, input_parm, Kt_data, t, ag.t_size)
        l += ag.ϵ * diff_norm((HF_calc-HF_given))
        #l += loss_fn_hybrid(en,ag, HF_given, HF_calc,t)
    end
    l += diff_norm(kp_sum)/ag.t_size
    return l 
end

#calculate the loss function in a single cycle & update K(t)&H_r(t)
function loss_calc_hyb!(model0, input_parm::Vector{Float32}, Kt_data::Matrix{Float32}, HF_given::Vector{Float32}, ag::agtQ)
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

        HF_calc = micro_motion!(Kp, input_parm, Kt_data, t, ag.t_size)
        l += ag.ϵ * diff_norm((HF_calc-HF_given))
    end
    l += diff_norm(kp_sum)/ag.t_size

    return l, kp_sum/ag.t_size
end

function check_dynamics(model0::Chain, input::Vector{Float32}, ag::agtQ)
    Kt = zeros(Float32, ag.t_size, 4^2)
    HF = zeros(Float32, ag.t_size, 4^2)
    for t in 1:ag.t_size
        if(t==1)
            tt=ag.t_size
        else
            tt=t-1
        end
        input = vcat([input, Kt[tt,:]]...)
        Kp = model0(input)
        HF[t,:] = micro_motion!(Kp, input[1:5], Kt, t, ag.t_size)
    end
    return Kt, HF
end

function check_dynamics(model0::Chain, input::Matrix{Float32}, ag::agtQ)
    bs = size(input)[2]
    Kt = zeros(Float32, bs, ag.t_size, 4^2)
    HF = zeros(Float32, bs, ag.t_size, 4^2)
    for t in 1:ag.t_size
        if(t==1)
            tt=ag.t_size
        else
            tt=t-1
        end
        inputs = vcat([input, (Kt[:,tt,:])']...)
        Kp = model0(input)
        for i in 1:bs
            HF[i,t,:] = micro_motion!(Kp[i,:], input[i,:], Kt[i,:,:], t, ag.t_size)
        end
    end
    return Kt, HF
end

function init_HF(en::TS_env)
    jp = en.Jz + en.hz
    jm = en.Jz - en.hz
    VHmHV::Vector{Float32} = 4*en.ξ*[0.0, 0.0, jp, 0.0, jp, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -jm, 0.0, 0.0, -jm, 0.0]
    init = MtoV(en.H_0) + VHmHV
    return init
end

using DataFrames
using CSV
using BSON: @save
using BSON: @load
using Plots
ENV["GKSwstype"]="nul"

function main(arg::Array{String,1})
    Ω = parse(Float32,arg[1]) 
    Jz = parse(Float32, arg[2])
    Jx = parse(Float32, arg[3])
    hz = parse(Float32, arg[4]) 
    ξ = parse(Float32, arg[5])
    # t::Int, t_size::Int
    #t=100, Ω0=10.0, ξ0=0.2, Jz0=1f0, Jx0=0.7, hz0=0.5
    #en = TS_env(init_env(parse(Int,arg[1]), parse(Float32,arg[2]), parse(Float32,arg[3]), parse(Float32,arg[4]), parse(Float32,arg[5]), parse(Float32,arg[6]))...)

    #en::TS_env, n=32, γ0=0.9, ϵ0=1f0
    ag = agtQ(init_nQ(arg[6:9])...)

    

    if(arg[10]=="clip1")
        opt = Flux.Optimise.Optimiser(ClipValue(1e-1),Adam(1e-1))
    elseif(arg[10]=="clip2")
        opt = Flux.Optimise.Optimiser(ClipValue(1e-2),Adam(1e-2))
    elseif(arg[10]=="clip3")
        opt = Flux.Optimise.Optimiser(ClipValue(1e-3),Adam(1e-3))
    elseif(arg[10]=="rms")
        opt = RMSProp()
    elseif(arg[10]=="gd")
        opt = Descent(parse(Float32, arg[15]))
    elseif(arg[10]=="adab")
        opt = AdaBelief()
    else
        opt = ADAM()
    end

    it_MAX = parse(Int,arg[11])
    ll_it = zeros(Float32, it_MAX)
    println("start!")

    st::Int = 0
    if(arg[12]=="init")
        model = Chain(Dense(ag.in_size, ag.n_dense, relu), Dense(ag.n_dense, ag.n_dense, relu), Dense(ag.n_dense, ag.n_dense, relu), Dense(ag.n_dense, ag.n_dense, relu), Dense(ag.n_dense, ag.out_size))
        ag.K_TL[en.t_size,:] = zeros(Float32, en.HS_size^2)
    else
        @load arg[12] model
        ag.K_TL = Matrix(CSV.read(arg[13], DataFrame))
        st = parse(Int,arg[14])
    end

    batch_num = parse(Int,arg[13])
    input = zeros(Float32, 5, batch_num)
    for i in 1:batch_num
        input[:,i] = generate_parmv(Ω, Jz, Jx, hz, ξ)
    end
    
    for it in 1:it_MAX
        HF_it = zeros(Float32, en.HS_size^2) 
        if(it==1)
            #=
            for t_step in 1:ag.t_size
                if(t_step==1)
                    tt=en.t_size
                else
                    tt=t_step-1
                end
                p = [en.ξ*sin(2pi*t_step/en.t_size), en.Jz, en.Jx, en.hz]
                x = vcat([p, ag.K_TL[tt,:]]...)

                ag.Kp_TL[t_step,:] = model(x)
                ag.K_TL[t_step,:], ag.HF_TL[t_step,:] = micro_motion(ag.Kp_TL[t_step,:], ag.K_TL[tt,:],en,t_step)

                HF_it += ag.HF_TL[t_step,:]/en.t_size
            end=#
            Kt, HF = check_dynamics(model, input, ag)
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

        grads = Flux.gradient(Flux.params(model)) do
            #loss_calc_hyb(model, en, ag, HF_it)
            loss_calc_hyb(model, input, Kt[1,:,:], HF[1,:], ag)
        end
        Flux.Optimise.update!(opt, Flux.params(model), grads)

        if(it==1) 
            println("First Learning Finish!")
        end

        ag.K_TL[en.t_size,:] = zeros(Float32, en.HS_size^2)
        ll_it[it], Kp_av = loss_calc_hyb!(model,en, ag, HF_it)
        if(it%(div(it_MAX, 10))==0)
            print("#")
        end
        if(it%1000 == 0 && it!=0)
            E = zeros(Float32, en.t_size, en.HS_size)
            for t_step in 1:en.t_size
                E[t_step,:], v = eigen(VtoM(ag.HF_TL[t_step,:],en))
            end

            p1 = plot(E[:,1].-E[1,1], xlabel="t_step", ylabel="E of HF_t", width=3.0)
            p1 = plot!(E[:,2].-E[1,2], width=3.0)
            p1 = plot!(E[:,3].-E[1,3], width=3.0)
            p1 = plot!(E[:,4].-E[1,4], width=3.0)
            savefig(p1,"./HF_t$(st+it).png")
            println("Drawing Finish!")
            #println(E[:,4])
            p2 = plot(ag.K_TL[:,1], xlabel="t_step", ylabel="E of K_t", width=2.0)
            for i in 2:en.HS_size^2
                p2 = plot!(ag.K_TL[:,i], width=2.0)
            end
            save_data1 = DataFrame(ag.K_TL, :auto)
            CSV.write("./K_TL$(st+it).csv", save_data1)
            savefig(p2,"./K_t$(st+it).png")
            p4 = plot(ag.Kp_TL[:,1], xlabel="t_step", ylabel="E of Kp_t", width=2.0)
            for i in 2:en.HS_size^2
                p4 = plot!(ag.Kp_TL[:,i], width=2.0)
            end
            savefig(p4,"./Kp_t$(st+it).png")
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