include("TwoSpin_env.jl")


using DataFrames
using CSV
using Plots
ENV["GKSwstype"]="nul"

function main(arg::Array{String,1})

    en = TS_env(init_env(parse(Int,arg[1]), parse(Float64,arg[2]), parse(Float64,arg[3]), parse(Float64,arg[4]), parse(Float64,arg[5]), parse(Float64,arg[6]))...)
    read_data = Matrix(CSV.read(arg[7], DataFrame))
    Kt_BI = zeros(Float64, en.t_size, en.HS_size^2)

    e, v = eigen(en.H_0)
    println(e)
    println(v'*en.H_0*v)

    for t in 1:en.t_size
        KtM = VtoM(read_data[t,:],en)
        K_BI = v' * KtM * v
        KK = Hermitian(K_BI)
        Kt_BI[t,:] = MtoV(KK,en)
    end

    
    V_BI = v' * en.V_t * v
    VV = Hermitian(V_BI)
    Vt_BI = MtoV(VV,en)
    println(Vt_BI)

    save_data1 = DataFrame(Kt_BI, :auto)
    CSV.write("./Kt_BI.csv", save_data1)
    p2 = plot(Kt_BI[:,1], xlabel="t_step", ylabel="E of K_t", width=2.0)
    for i in 2:en.HS_size^2
        p2 = plot!(Kt_BI[:,i], width=2.0)
    end
    savefig(p2,"./Kt_BI.png")

end

@time main(ARGS)
