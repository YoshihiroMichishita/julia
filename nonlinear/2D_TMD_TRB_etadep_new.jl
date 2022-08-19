using Distributed
addprocs(4)
@everywhere include("./2D_TMD_parm.jl")
@everywhere include("./transport.jl")


using DataFrames
using CSV
using Plots

function main(arg::Array{String,1})

    K_SIZE = parse(Int,arg[11])
    kk = get_kk(K_SIZE)

    eta = collect(0.01:0.01:0.1)
    #collect(0.005:0.005:0.15)
    #[0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08]
    #collect(0.005:0.005:0.1)

    Green_eta = zeros(Float64,length(eta))
    Green_eta_sea = zeros(Float64,length(eta))

    Drude_eta = zeros(Float64,length(eta))
    BCD_eta = zeros(Float64,length(eta))
    gBC_eta = zeros(Float64,length(eta))
    ChS_eta = zeros(Float64,length(eta))


    for j in 1:length(eta)
        p = Parm(set_parm_etadep(arg, eta[j])...)
        kk = get_kk(p.K_SIZE)
        dk2 = 2.0/(3*sqrt(3.0)*p.K_SIZE*p.K_SIZE)
        
        if j == 1
            println("Parm(t_i, a_u, a_d, Pr, mu, eta, T, hx, hy, hz, K_SIZE, W_MAX, W_SIZE)")
            println(p)
        end

        Drude_eta[j], BCD_eta[j], gBC_eta[j], ChS_eta[j], Green_eta[j], Green_eta_sea[j] = @distributed (+) for i in 1:length(kk)
            Hamk = Hamiltonian(HandV(kk[i],p)...)
            Drude_eta0, BCD_eta0, gBC_eta0, ChS_eta0 = Green_DC_BI_nonlinear_full(p, Hamk)
            Green_eta0, Green_eta_sea0 = Green_DC_nonlinear(p, Hamk)

            [dk2*Drude_eta0, dk2*BCD_eta0, dk2*gBC_eta0, dk2*ChS_eta0, dk2*Green_eta0, dk2*Green_eta_sea0]
        end
        print("#")
    end
    println("finish the calculation!")
    # headerの名前を(Q,E1,E2)にして、CSVファイル形式を作成
    save_data2 = DataFrame(η=eta, Drude=Drude_eta, BCD=BCD_eta, gBC=gBC_eta, ChS=ChS_eta, Green=Green_eta, Green_sea=Green_eta_sea)
    CSV.write("./etadep_"*arg[15]*arg[16]*arg[17]*"_T002_M_test.csv", save_data2)

    ENV["GKSwstype"]="nul"
    Plots.scalefontsizes(1.4)

    p1 = plot(eta, Green_eta, label="Green_sur",xlabel="η",ylabel="σ",title="η-dependence", width=4.0, marker=:circle, markersize = 4.8)
    p1 = plot!(eta, Green_eta_sea, label="Green_sea", width=4.0, marker=:circle, markersize = 4.8)
    p1 = plot!(eta, Drude_eta, label="Drude", width=4.0, marker=:circle, markersize = 4.8)
    p1 = plot!(eta, BCD_eta, label="BCD", width=4.0, marker=:circle, markersize = 4.8)
    p1 = plot!(eta, gBC_eta, label="gBC", width=4.0, marker=:circle, markersize = 4.8)
    p1 = plot!(eta, ChS_eta, label="ChS", width=4.0, marker=:circle, markersize = 4.8)
    savefig(p1,"./etadep_"*arg[15]*arg[16]*arg[17]*".png")
end

@time main(ARGS)