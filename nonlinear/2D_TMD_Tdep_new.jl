using Distributed
addprocs(30)
@everywhere include("./2D_TMD_parm.jl")
@everywhere include("./transport.jl")
@everywhere include("./k_C3.jl")

using DataFrames
using CSV
using Plots

function main(arg::Array{String,1})

    K_SIZE = parse(Int,arg[11])
    kk = get_kk(K_SIZE)

    mu0 = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15]
    #collect(0.00:0.005:0.15)

    Green_mu = zeros(Float64,length(mu0))
    Green_mu_sea = zeros(Float64,length(mu0))

    Drude_mu = zeros(Float64,length(mu0))
    BCD_mu = zeros(Float64,length(mu0))
    ChS_mu = zeros(Float64,length(mu0))
    gBC_mu = zeros(Float64,length(mu0))


    for j in 1:length(mu0)
        p = Parm(set_parm_mudep(arg, mu0[j])...)

        kk = get_kk(p.K_SIZE)
        dk2 = 2.0/(3*sqrt(3.0)*p.K_SIZE*p.K_SIZE)
        
        if j == 1
            println("Parm(t_i, a_u, a_d, Pr, mu, eta, T, hx, hy, hz, K_SIZE, W_MAX, W_in, W_SIZE)")
            println(p)
        end

        Drude_mu[j], BCD_mu[j], gBC_mu[j], ChS_mu[j], Green_mu[j], Green_mu_sea[j] = @distributed (+) for i in 1:length(kk)
            Hamk = Hamiltonian(HandV_fd(kk[i],p)...)
            Drude_mu0, BCD_mu0, gBC_mu0, ChS_mu0 = Green_DC_BI_nonlinear_full(p, Hamk)
            Green_mu0, Green_mu_sea0 = Green_DC_nonlinear(p, Hamk)

            [dk2*Drude_mu0, dk2*BCD_mu0, dk2*gBC_mu0, dk2*ChS_mu0, dk2*Green_mu0, dk2*Green_mu_sea0]
        end
        print("#")
    end
    println("finish the calculation!")
    # headerの名前を(Q,E1,E2)にして、CSVファイル形式を作成
    save_data2 = DataFrame(T=mu0, Drude=Drude_mu, BCD=BCD_mu, gBC=gBC_mu, ChS=ChS_mu, Green=Green_mu, Green_sea=Green_mu_sea)
    CSV.write("./Tdep_"*arg[15]*arg[16]*arg[17]*"_mu"*arg[5]*".csv", save_data2)

    ENV["GKSwstype"]="nul"
    Plots.scalefontsizes(1.4)

    p1 = plot(mu0, Green_mu, label="Green_sur",xlabel="T",ylabel="σ",title="T-dependence", width=4.0, marker=:circle, markersize = 4.8)
    p1 = plot!(mu0, Green_mu_sea, label="Green_sea", width=4.0, marker=:circle, markersize = 4.8)
    p1 = plot!(mu0, Drude_mu, label="Drude", width=4.0, marker=:circle, markersize = 4.8)
    p1 = plot!(mu0, BCD_mu, label="BCD", width=4.0, marker=:circle, markersize = 4.8)
    p1 = plot!(mu0, gBC_mu, label="gBC", width=4.0, marker=:circle, markersize = 4.8)
    p1 = plot!(mu0, ChS_mu, label="ChS", width=4.0, marker=:circle, markersize = 4.8)
    savefig(p1,"./Tdep_"*arg[15]*arg[16]*arg[17]*".png")
end

@time main(ARGS)