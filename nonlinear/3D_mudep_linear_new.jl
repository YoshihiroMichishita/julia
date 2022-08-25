using Distributed
addprocs(4)
@everywhere include("./3D_NLSM_parm.jl")
@everywhere include("./transport.jl")


using DataFrames
using CSV
using Plots
#gr()

function main(arg::Array{String,1})
    println("t, p0, v, mu, Delta, eta, T, K_SIZE, W_MAX, W_SIZE")
    mu0 = collect(-0.2:0.02:0.2)
    #[0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15]
    Drude_mu = zeros(Float64,length(mu0))
    app_Drude0_mu = zeros(Float64,length(mu0))
    BC_mu = zeros(Float64,length(mu0))
    dQM_mu = zeros(Float64,length(mu0))
    app_dQM_mu = zeros(Float64,length(mu0))
    Green_mu = zeros(Float64,length(mu0))

    for j in 1:length(mu0)
        #p = Parm(set_parm_etadep(arg, mu0[j])...)
        p = Parm(set_parm_mudep(arg, mu0[j])...)

        for kz in collect(-pi+pi/p.K_SIZE:pi/p.K_SIZE:pi)
            
            Dr0::Float64 = 0.0
            app_Dr0::Float64 = 0.0
            BC0::Float64 = 0.0
            dQM0::Float64 = 0.0
            app_QM0::Float64 = 0.0
            Green0::Float64 = 0.0

            kk = get_kk(kz, p.K_SIZE)
            #Dr0, Dr00, BC0, dQM0, app_QM0, Green0 
            Dr0, app_Dr0, BC0, dQM0, app_QM0, Green0= @distributed (+) for i in 1:length(kk)

                k = [kk[i][1], kk[i][2], kz]
                Hamk = Hamiltonian(HandV(k,p)...)
                Dr, app_Dr, BC, dQM, app_QM = Green_DC_BI_linear_full(p, Hamk)
                Green_Dr = 0
                #, ass = Green_DC(p, Hamk)
                #Green_Dr
                [Dr/(p.K_SIZE^3), app_Dr/(p.K_SIZE^3), BC/(p.K_SIZE^3), dQM/(p.K_SIZE^3), app_QM/(p.K_SIZE^3), Green_Dr/(p.K_SIZE^3)]
                #, Green_tot/(p.K_SIZE^3)]
            end

            Green_mu[j] += Green0
            Drude_mu[j] += Dr0
            app_Drude0_mu[j] += app_Dr0
            BC_mu[j] += BC0
            dQM_mu[j] += dQM0
            app_dQM_mu[j] += app_QM0
            Green_mu[j] += Green0
        end
        if j == 1
            println(p)
        end
        print("#")
    end
    println("finish the calculation!")
    # headerの名前を(Q,E1,E2)にして、CSVファイル形式を作成
    save_data = DataFrame(mu=mu0, Drude=Drude_mu, BCD=BC_mu, dQM=dQM_mu, Green_Dr=app_Drude0_mu, app_QM=app_dQM_mu, Green_tot=Green_mu)
    #「./」で現在の(tutorial.ipynbがある)ディレクトリにファイルを作成の意味、指定すれば別のディレクトリにファイルを作ることも出来る。
    CSV.write("./mu_dep_ZZ.csv", save_data)

    #gr()
    ENV["GKSwstype"]="nul"
    p1=plot(mu0, Drude_mu, label="Drude",xlabel="μ",ylabel="σ",title="linear conductivity", width=2.0, marker=:circle)
    p1=plot!(mu0, BC_mu, label="BC", width=2.0, marker=:circle)
    p1=plot!(mu0, dQM_mu, label="dQM", width=2.0, marker=:circle)
    p1=plot!(mu0, Green_mu-Drude_mu-BC_mu-dQM_mu, label="Matsubara", width=2.0, marker=:circle)
    p1=plot!(mu0, Green_mu, label="Green", width=2.0, marker=:circle)
    savefig(p1,"./mu_dep_ZZ.png")
end

@time main(ARGS)