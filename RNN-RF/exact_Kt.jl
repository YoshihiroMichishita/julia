include("TwoSpin_env.jl")

function calc_Uf(en::TS_env)
    U_F = Matrix{Complex{Float64}}(I,en.HS_size,en.HS_size)
    for t in 1:en.t_size
        H_t = en.H_0 + sin(2pi*t/en.t_size)*en.V_t
        U_F = exp(-1.0im*H_t*en.dt)*U_F 
    end
    H_F = en.Ω*Hermitian(1.0im*log(U_F))/(2pi)
    return H_F
end

function Unitarize!(U::Matrix{ComplexF64})
    O = U' * U
    for i in 1:size(U)[1]
        U[:,i] = U[:,i]/sqrt(O[i,i])
    end
end


function calc_Kt(en::TS_env, H_F)
    K_t = zeros(Float64, 100, en.HS_size^2)
    Ud_t = Matrix{Complex{Float64}}(I,en.HS_size,en.HS_size)
    for t in 1:en.t_size
        H_t = en.H_0 + sin(t*en.dt*en.Ω)*en.V_t
        Ud_p = 1.0im*(Ud_t*H_F - H_t*Ud_t)
        Ud_t += en.dt*Ud_p
        Unitarize!(Ud_t)
        if(t%(en.t_size/100)==0)
            if(t%(en.t_size/10)==0)
                O = Ud_t'*Ud_t
                println("Unitary check: $(O[1,1]), $(O[2,2]), $(O[3,3]), $(O[4,4])")
            end
            K = Hermitian(1.0im*log(Ud_t))
            K_t[Int(t/(en.t_size/100)),:] = MtoV(K, en)
        end
    end
    return K_t
end

using DataFrames
using CSV
using Plots
ENV["GKSwstype"]="nul"

function main(arg::Array{String,1})
    en = TS_env(init_env(parse(Int,arg[1]), parse(Float64,arg[2]), parse(Float64,arg[3]), parse(Float64,arg[4]), parse(Float64,arg[5]), parse(Float64,arg[6]))...)

    H_F = calc_Uf(en)
    K_t = calc_Kt(en, H_F)

    p2 = plot(K_t[:,1], xlabel="t_step", ylabel="E of K_t", width=2.0)
    for i in 2:en.HS_size^2
        p2 = plot!(K_t[:,i], width=2.0)
    end
    save_data1 = DataFrame(K_t, :auto)
    CSV.write("./K_TL_exact$(en.Ω).csv", save_data1)
    savefig(p2,"./K_t_exact.png")
end

@time main(ARGS)
