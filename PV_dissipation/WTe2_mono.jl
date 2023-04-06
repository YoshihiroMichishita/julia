using Distributed
addprocs(32)
@everywhere include("./WTe2_mono_parm.jl")
@everywhere include("./transport.jl")

using DataFrames
using CSV
using Plots

function main(arg::Array{String,1})

    K_SIZE = parse(Int,arg[11])
    kk = get_kk(K_SIZE)
    dk2 = (2pi)^2/(K_SIZE^2)
    η = parse(Float64,arg[9])
    st = parse(Float64,arg[18])
    ed = parse(Float64,arg[19])
    sw = parse(Int,arg[20])
    

    Win0 = collect(st:0.1:ed)

    Green_W = zeros(Float64,length(Win0), 14)
    Velocity_W = zeros(Float64,length(Win0), 12)
    Length_W = zeros(Float64,length(Win0), 12)
    #Green_W_sea = zeros(Float64,length(mu0))

    #Drude_mu = zeros(Float64,length(mu0))
    #BCD_mu = zeros(Float64,length(mu0))
    #ChS_mu = zeros(Float64,length(mu0))
    #gBC_mu = zeros(Float64,length(mu0))


    for j in 1:size(Win0)[1]
        #t, tl, ar, ad, mu, eta, T, hx, dz, K_size, Wmax, Win, Wsize, abc, 
        p = Parm(set_parm_Wdep(arg, Win0[j])...)

        kk = get_kk(p.K_SIZE)
        
        
        if j == 1
            println("Parm(td, tp, td_AB, tp_AB, t0_aB, μd, μp, μf, eta, T, K_size, Wmax, Win, Wsize, abc)")
            println(p)
        end
        if(sw>0)
            Green_W[j,1:12] = @distributed (+) for i in 1:size(kk)[1]
                Hamk = Hamiltonian(HandV_fd(kk[i],p)...)
                HV_BI!(Hamk)
                dk2*Green_PV_BI(p, Hamk)
            end
            if(sw>1)
                Green_W[j,13:14] = @distributed (+) for i in 1:size(kk)[1]
                    Hamk = Hamiltonian(HandV_fd(kk[i],p)...)
                    dk2*Green_PV(p, Hamk)
                end
            end
        end
        
        Velocity_W[j,:] = @distributed (+) for i in 1:size(kk)[1]
            Hamk = Hamiltonian(HandV_fd(kk[i],p)...)
            HV_BI!(Hamk)
            dk2*Velocity_PV_BI(p, Hamk)
        end

        Length_W[j,:] = @distributed (+) for i in 1:size(kk)[1]
            Hamk = Hamiltonian(HandV_fd(kk[i],p)...)
            HV_BI!(Hamk)
            dk2*Length_PV_BI2(p, Hamk)
        end

        print("#")
        if(j%10==0)
            print("||")
        end
    end
    println("finish the calculation!")
    # headerの名前を(Q,E1,E2)にして、CSVファイル形式を作成
    save_data0 = DataFrame(W=Win0)
    CSV.write("./Win.csv", save_data0)

    if(sw>0)
        save_data1 = DataFrame(Green_W, :auto)
        CSV.write("./Wdep_η$(η)_$(arg[14])$(arg[15])$(arg[16])_Green.csv", save_data1)
    end

    save_data2 = DataFrame(Velocity_W, :auto)
    CSV.write("./Wdep_η$(η)_$(arg[14])$(arg[15])$(arg[16])_Velocity.csv", save_data2)

    save_data3 = DataFrame(Length_W, :auto)
    CSV.write("./Wdep_η$(η)_$(arg[14])$(arg[15])$(arg[16])_Length.csv", save_data3)


    ENV["GKSwstype"]="nul"
    Plots.scalefontsizes(1.4)

    p1 = plot(Win0, Green_W[:,1], label="Dr", xlabel="Ω",ylabel="σ",title="Ω-dependence", width=2.0, marker=:circle, markersize = 2.0, color=1)
    p1 = plot!(Win0, Velocity_W[:,1], label=nothing, style=:dash, width=2.0, marker=:square, markersize = 2.0, color=1)
    p1 = plot!(Win0, Length_W[:,1], label=nothing, style=:dash, width=2.0, marker=:utriangle, markersize = 2.0, color=1)

    p1 = plot!(Win0, Green_W[:,3], label="BCD", width=2.0, marker=:circle, markersize = 2.0, color=2)
    p1 = plot!(Win0, Velocity_W[:,3], label=nothing, style=:dash, width=2.0, marker=:square, markersize = 2.0, color=2)
    p1 = plot!(Win0, Length_W[:,3], label=nothing, style=:dash, width=2.0, marker=:utriangle, markersize = 2.0, color=2)

    p1 = plot!(Win0, Green_W[:,5]+Green_W[:,9], label="Shift", width=2.0, marker=:circle, markersize = 2.0, color=3)
    p1 = plot!(Win0, Velocity_W[:,5]+Velocity_W[:,9], label=nothing, style=:dash, width=2.0, marker=:square, markersize = 2.0, color=3)
    p1 = plot!(Win0, Length_W[:,5]+Length_W[:,9], label=nothing, style=:dash, width=2.0, marker=:utriangle, markersize = 2.0, color=3)

    p1 = plot!(Win0, Green_W[:,7], label="Inj", width=2.0, marker=:circle, markersize = 2.0, color=4)
    p1 = plot!(Win0, Velocity_W[:,7], label=nothing, style=:dash, width=2.0, marker=:square, markersize = 2.0, color=4)
    p1 = plot!(Win0, Length_W[:,7], label=nothing, style=:dash, width=2.0, marker=:utriangle, markersize = 2.0, color=4)
    p1 = plot!(Win0, Green_W[:,13], label="tot", width=2.0, marker=:circle, markersize = 2.0, color=5)
    savefig(p1,"./PV_Linear_Wdep_η$(η)_$(arg[14])$(arg[15])$(arg[16]).png")

    p2 = plot(Win0, Green_W[:,2], label="Dr", xlabel="Ω",ylabel="σ",title="Ω-dependence", width=2.0, marker=:circle, markersize = 2.0, color=1)
    p2 = plot!(Win0, Velocity_W[:,2], label=nothing, style=:dash, width=2.0, marker=:square, markersize = 2.0, color=1)
    p2 = plot!(Win0, Length_W[:,2], label=nothing, style=:dash, width=2.0, marker=:utriangle, markersize = 2.0, color=1)

    p2 = plot!(Win0, Green_W[:,4], label="BCD", width=2.0, marker=:circle, markersize = 2.0, color=2)
    p2 = plot!(Win0, Velocity_W[:,4], label=nothing, style=:dash, width=2.0, marker=:square, markersize = 2.0, color=2)
    p2 = plot!(Win0, Length_W[:,4], label=nothing, style=:dash, width=2.0, marker=:utriangle, markersize = 2.0, color=2)

    p2 = plot!(Win0, Green_W[:,6]+Green_W[:,10], label="Shift", width=2.0, marker=:circle, markersize = 2.0, color=3)
    p2 = plot!(Win0, Velocity_W[:,6]+Velocity_W[:,10], label=nothing, style=:dash, width=2.0, marker=:square, markersize = 2.0, color=3)
    p2 = plot!(Win0, Length_W[:,6]+Length_W[:,10], label=nothing, style=:dash, width=2.0, marker=:utriangle, markersize = 2.0, color=3)

    p2 = plot!(Win0, Green_W[:,8], label="Inj", width=2.0, marker=:circle, markersize = 2.0, color=4)
    p2 = plot!(Win0, Velocity_W[:,8], label=nothing, style=:dash, width=2.0, marker=:square, markersize = 2.0, color=4)
    p2 = plot!(Win0, Length_W[:,8], label=nothing, style=:dash, width=2.0, marker=:utriangle, markersize = 2.0, color=4)

    p2 = plot!(Win0, Green_W[:,14], label="tot", width=2.0, marker=:circle, markersize = 2.0, color=5)
    savefig(p2,"./PV_CP_Wdep_η$(η)_$(arg[14])$(arg[15])$(arg[16]).png")
end

@time main(ARGS)