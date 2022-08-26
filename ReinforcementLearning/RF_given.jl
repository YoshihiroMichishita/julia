include("TwoSpin_env.jl")

function check_HF(en::TS_env, Kt::Matrix{Float64})
    t_size = size(Kt)[1]
    dt = 2pi/en.Ω/t_size
    HF = zeros(Float64, t_size, en.HS_size^2)

    Kpt = zeros(Float64, t_size, en.HS_size^2)
    for t in 1:t_size
        if(t>1)
            tt = t-1
        else
            tt = t_size
        end
        Kpt[t,:] = (Kt[t,:] - Kt[tt,:])/dt
        KtM = VtoM(Kt[t,:],en)
        KptM = VtoM(Kpt[t,:],en)
        HfM = exp(-1.0im*KtM)*(en.H_0 + en.V_t - 1.0im*KptM)*exp(-1.0im*KtM)
        HF[t,:] = MtoV(HfM, en)
    end
    return HF
end

function set_Kt_res(en::TS_env, mp::vector{Float64})
    Kt = zeros(Float64, en.t_size, en.HS_size^2)

    M1 = [0.0 1.0im 1.0im 0.0; -1.0im 0.0 0.0 0.0; -1.0im 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
    M1v = MtoV(Hermitian(M1),en)

    M2 = [0.0 1.0 1.0 0.0; 1.0 0.0 0.0 0.0; 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
    M2v = MtoV(Hermitian(M2),en)

    M3 = [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 -1.0; 0.0 0.0 0.0 -1.0; 0.0 -1.0 -1.0 0.0]
    M3v = MtoV(Hermitian(M3),en)

    for t in 1:en.t_size
        Kt[t,:] = mp[1]*sin(2pi*t/en.t_size)*M1v + (1.0+cos(2pi*t/en.t_size))*(mp[2]*M2v + mp3*M3v)
    end
    return Ktghj
end


using DataFrames
using CSV
using Plots
ENV["GKSwstype"]="nul"

function main(arg::Array{String,1})

    en = TS_env(init_env(parse(Int,arg[1]), parse(Float64,arg[2]), parse(Float64,arg[3]), parse(Float64,arg[4]), parse(Float64,arg[5]), parse(Float64,arg[6]))...)
    mp = [parse(Float64,arg[7]), parse(Float64,arg[8]), parse(Float64,arg[9])]

    Kt = set_Kt_res(en, mp)

    HFt = check_HF(en, Kt)

    E = zeros(Float64, en.t_size, en.HS_size)
    for t_step in 1:en.t_size
        E[t_step,:], v = eigen(VtoM(HFt[t_step,:],en))
    end

    p1 = plot(E[:,1].-E[1,1], xlabel="t_step", ylabel="E of HF_t", width=3.0)
    p1 = plot!(E[:,2].-E[1,2], width=3.0)
    p1 = plot!(E[:,3].-E[1,3], width=3.0)
    p1 = plot!(E[:,4].-E[1,4], width=3.0)

    savefig(p1,"./HF_t_given.png")

    save_data1 = DataFrame(HFt, :auto)
    CSV.write("./HF_t_given.csv", save_data1)

end

@time main(ARGS)
