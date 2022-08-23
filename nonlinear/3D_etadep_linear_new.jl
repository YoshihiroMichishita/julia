using Distributed
addprocs(32)
include("./3D_Weyl_parm.jl")
include("transport.jl")


using DataFrames
using CSV
using Plots
#gr()

function main(arg::Array{String,1})
    println("t, p0, v, mu, Delta, eta, T, K_SIZE, W_MAX, W_SIZE")
    mu0 = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15]
    Drude_mu = zeros(Float64,length(mu0))
    app_Drude0_mu = zeros(Float64,length(mu0))
    BC_mu = zeros(Float64,length(mu0))
    dQM_mu = zeros(Float64,length(mu0))
    app_dQM_mu = zeros(Float64,length(mu0))
    Green_mu = zeros(Float64,length(mu0))

    for j in 1:length(mu0)
        p = Parm(set_parm_etadep(arg, mu0[j])...)

        k2 = collect(Iterators.product((0:pi/p.K_SIZE:pi)[1:end-1], (pi/p.K_SIZE:pi/p.K_SIZE:(pi+pi/p.K_SIZE))[1:end-1]))
        for kz in collect(0:pi/p.K_SIZE:pi)[1:end]
            
            Dr0::Float64 = 0.0
            app_Dr0::Float64 = 0.0
            BC0::Float64 = 0.0
            dQM0::Float64 = 0.0
            app_QM0::Float64 = 0.0
            Green0::Float64 = 0.0
            #Dr0, Dr00, BC0, dQM0, app_QM0, Green0 
            Dr0, app_Dr0, BC0, dQM0, app_QM0, Green0= @distributed (+) for i in 1:length(k2)
                k = (k2[i][1], k2[i][2], kz)
                Hamk = Hamiltonian(HandV(k,p)...)
                Dr, app_Dr, BC, dQM, app_QM = Green_DC_BI_linear_full(p, Hamk)
                Green_Dr = Green_DC(p, Hamk)
                #Green_Dr
                [Dr/(p.K_SIZE^3), app_Dr/(p.K_SIZE^3), BC/(p.K_SIZE^3), dQM/(p.K_SIZE^3), app_QM/(p.K_SIZE^3), Green_Dr/(p.K_SIZE^3)]
                #, Green_tot/(p.K_SIZE^3)]
            end

            if kz == 0 || kz == pi
                Green_mu[j] += 4Green0
                Drude_mu[j] += 4Dr0
                app_Drude0_mu[j] += 4app_Dr0
                BC_mu[j] += 4BC0
                dQM_mu[j] += 4dQM0
                app_dQM_mu[j] += 4app_QM0
                Green_mu[j] += 4Green0
            else
                Green_mu[j] += 8Green0
                Drude_mu[j] += 8Dr0
                app_Drude0_mu[j] += 8app_Dr0
                BC_mu[j] += 8BC0
                dQM_mu[j] += 8dQM0
                app_dQM_mu[j] += 8app_QM0
                Green_mu[j] += 8Green0
            end
        end
        if j == 1
            println(p)
        end
    end
    println("finish the calculation!")
    # headerの名前を(Q,E1,E2)にして、CSVファイル形式を作成
    save_data = DataFrame(eta=mu0, Drude=DrudeX_mu, BCD=BCX_mu, dQM=dQMX_mu, Green_Dr=Drude0_mu, app_QM=app_QM_mu, Green_tot=Green_mu)
    #「./」で現在の(tutorial.ipynbがある)ディレクトリにファイルを作成の意味、指定すれば別のディレクトリにファイルを作ることも出来る。
    CSV.write("./eta_dep_ZZ.csv", save_data)

    #gr()
    ENV["GKSwstype"]="nul"
    p1=plot(mu0, Drude_mu, label="Drude",xlabel="μ",ylabel="σ",title="linear conductivity", width=2.0, marker=:circle)
    p1=plot!(mu0, BC_mu, label="BC", width=2.0, marker=:circle)
    p1=plot!(mu0, dQM_mu, label="dQM", width=2.0, marker=:circle)
    p1=plot!(mu0, Green_mu-Drude_mu-BC_mu-dQM_mu, label="Matsubara", width=2.0, marker=:circle)
    p1=plot!(mu0, Green_mu, label="Green", width=2.0, marker=:circle)
    savefig(p1,"./eta_dep_ZZ.png")
end

@time main(ARGS)