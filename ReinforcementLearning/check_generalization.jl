include("TwoSpin_env.jl")

using Flux
using BSON: @load

function micro_motion(Kp_t::Vector{Float64}, K_t::Vector{Float64}, en::TS_env, t::Int)
    Kp = VtoM(Kp_t,en)
    K_t_new = K_t + (2pi/en.t_size/en.Ω) * Kp_t 
    Kt = VtoM(K_t_new,en)
    HF_m = Hermitian(exp(1.0im*Kt)*(en.H_0 + en.V_t*sin(2pi*t/en.t_size) - Kp)*exp(-1.0im*Kt))
    HF = MtoV(HF_m, en)
    return K_t_new, HF
end


function loss_fn_hybrid(en::TS_env, HF_t::Array{Float64,2}, HF_calc::Vector{Float64}, t::Int)
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
        l += diff_norm((HF_calc-HF_t[lt,:]),en)/en.t_size
    end
    #l += diff_norm(HF_given - HF_calc,en)/en.t_size
    return l
end

function main(arg::Array{String,1})
    en = TS_env(init_env(parse(Int,arg[1]), parse(Float64,arg[2]), parse(Float64,arg[3]), parse(Float64,arg[4]), parse(Float64,arg[5]), parse(Float64,arg[6]))...)

    @load arg[7] model

    HF_t = zeros(Float64, en.t_size, en.HS_size^2)
    K_t = zeros(Float64, en.t_size, en.HS_size^2)
    Kp_t = zeros(Float64, en.t_size, en.HS_size^2)

    #dt = 2pi/en.Ω/en.t_size
    l::Float64 = 0.0

    for t in 1:en.t_size
        if(t==1)
            tt=en.t_size
        else
            tt=t-1
        end
        p = [en.Ω, en.ξ*sin(2pi*t/en.t_size), en.Jz, en.Jx, en.hz]
        x = vcat([p, K_t[tt,:]]...)
        Kp_t[t,:] = model0(x)
        K_t[t,:], HF_t[t,:] = micro_motion(Kp_t[t,:], K_t[tt,:], en, t)
    end

    for t in 1:en.t_size
        l += loss_fn_hybrid(en, HF_t, HF_t[t,:],t)
        if(t==t_size)
            l += diff_norm(HF_t[t,:]-ag.HF_TL[1,:],en)
        end
    end

    E = zeros(Float64, en.t_size, en.HS_size)
    for t_step in 1:en.t_size
        E[t_step,:], v = eigen(VtoM(HF_t[t_step,:],en))
    end

    p1 = plot(E[:,1].-E[1,1], xlabel="t_step", ylabel="E of HF_t", width=3.0)
    p1 = plot!(E[:,2].-E[1,2], width=3.0)
    p1 = plot!(E[:,3].-E[1,3], width=3.0)
    p1 = plot!(E[:,4].-E[1,4], width=3.0)
    savefig(p1,"./HF_t.png")
    println("Drawing Finish!")
    #println(E[:,4])
    p2 = plot(K_t[:,1], xlabel="t_step", ylabel="E of K_t", width=2.0)
    for i in 2:en.HS_size^2
        p2 = plot!(K_t[:,i], width=2.0)
    end
    save_data1 = DataFrame(K_t, :auto)
    CSV.write("./K_TL.csv", save_data1)
    savefig(p2,"./K_t.png")
end

@time main(ARGS)
