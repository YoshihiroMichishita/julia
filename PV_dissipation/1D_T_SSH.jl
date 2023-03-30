using Distributed
addprocs(26)
@everywhere include("./SSH_T_parm.jl")
@everywhere include("./transport.jl")

using DataFrames
using CSV
using Plots

function main(arg::Array{String,1})

    K_SIZE = parse(Int,arg[8])
    kk = get_kk(K_SIZE)
    dk2 = (2pi)/(K_SIZE)
    η = parse(Float64,arg[6])
    st = parse(Float64,arg[15])
    ed = parse(Float64,arg[16])

    Win0 = collect(st:0.1:ed)

    Green_W = zeros(Float64,length(Win0), 10)
    Velocity_W = zeros(Float64,length(Win0), 10)
    Length_W = zeros(Float64,length(Win0), 10)
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
            println("Parm(t, tl, ar, ad, mu, eta, T, hx, dz, K_size, Wmax, Win, Wsize, abc)")
            println(p)
        end

        Green_W[j,:] = @distributed (+) for i in 1:size(kk)[1]
            Hamk = Hamiltonian(HandV_fd(kk[i],p)...)
            dk2*Green_PV_BI(p, Hamk)
        end

        Velocity_W[j,:] = @distributed (+) for i in 1:size(kk)[1]
            Hamk = Hamiltonian(HandV_fd(kk[i],p)...)
            dk2*Velocity_PV_BI(p, Hamk)
        end

        Length_W[j,:] = @distributed (+) for i in 1:size(kk)[1]
            Hamk = Hamiltonian(HandV_fd(kk[i],p)...)
            dk2*Length_PV_BI(p, Hamk)
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

    save_data1 = DataFrame(Green_W, :auto)
    CSV.write("./Wdep_η$(η)_Green.csv", save_data1)

    save_data2 = DataFrame(Velocity_W, :auto)
    CSV.write("./Wdep_η$(η)_Velocity.csv", save_data2)

    save_data3 = DataFrame(Length_W, :auto)
    CSV.write("./Wdep_η$(η)_Length.csv", save_data3)


    ENV["GKSwstype"]="nul"
    Plots.scalefontsizes(1.4)

    p1 = plot(Win0, Green_W[:,1], label="Dr", xlabel="Ω",ylabel="σ",title="Ω-dependence", width=2.0, marker=:circle, markersize = 2.0, color=1)
    p1 = plot!(Win0, Velocity_W[:,1], label="V_Dr", style=:dash, width=2.0, marker=:square, markersize = 2.0, color=1, legend=nothing)
    p1 = plot!(Win0, Length_W[:,1], label="L_Dr", style=:dash, width=2.0, marker=:utriangle, markersize = 2.0, color=1, legend=nothing)

    p1 = plot!(Win0, Green_W[:,3], label="BCD", width=2.0, marker=:circle, markersize = 2.0, color=2)
    p1 = plot!(Win0, Velocity_W[:,3], label="BCD_Dr", style=:dash, width=2.0, marker=:square, markersize = 2.0, color=2, legend=nothing)
    p1 = plot!(Win0, Length_W[:,3], label="BCD_Dr", style=:dash, width=2.0, marker=:utriangle, markersize = 2.0, color=2, legend=nothing)

    p1 = plot!(Win0, Green_W[:,5]+Green_W[:,9], label="Shift", width=2.0, marker=:circle, markersize = 2.0, color=3)
    p1 = plot!(Win0, Velocity_W[:,5]+Velocity_W[:,9], label="Shift_Dr", style=:dash, width=2.0, marker=:square, markersize = 2.0, color=3, legend=nothing)
    p1 = plot!(Win0, Length_W[:,5]+Length_W[:,9], label="Shift_Dr", style=:dash, width=2.0, marker=:utriangle, markersize = 2.0, color=3, legend=nothing)

    p1 = plot!(Win0, Green_W[:,7], label="Inj", width=2.0, marker=:circle, markersize = 2.0, color=4)
    p1 = plot!(Win0, Velocity_W[:,7], label="Inj_Dr", style=:dash, width=2.0, marker=:square, markersize = 2.0, color=4, legend=nothing)
    p1 = plot!(Win0, Length_W[:,7], label="Inj_Dr", style=:dash, width=2.0, marker=:utriangle, markersize = 2.0, color=4, legend=nothing)
    savefig(p1,"./PV_Linear_Wdep_η$(η).png")

    p2 = plot(Win0, Green_W[:,2], label="Dr", xlabel="Ω",ylabel="σ",title="Ω-dependence", width=2.0, marker=:circle, markersize = 2.0, color=1)
    p2 = plot!(Win0, Velocity_W[:,2], label="V_Dr", style=:dash, width=2.0, marker=:square, markersize = 2.0, color=1, legend=nothing)
    p2 = plot!(Win0, Length_W[:,2], label="L_Dr", style=:dash, width=2.0, marker=:utriangle, markersize = 2.0, color=1, legend=nothing)

    p2 = plot!(Win0, Green_W[:,4], label="BCD", width=2.0, marker=:circle, markersize = 2.0, color=2)
    p2 = plot!(Win0, Velocity_W[:,4], label="BCD_Dr", style=:dash, width=2.0, marker=:square, markersize = 2.0, color=2, legend=nothing)
    p2 = plot!(Win0, Length_W[:,4], label="BCD_Dr", style=:dash, width=2.0, marker=:utriangle, markersize = 2.0, color=2, legend=nothing)

    p2 = plot!(Win0, Green_W[:,6]+Green_W[:,10], label="Shift", width=2.0, marker=:circle, markersize = 2.0, color=3)
    p2 = plot!(Win0, Velocity_W[:,6]+Velocity_W[:,10], label="Shift_Dr", style=:dash, width=2.0, marker=:square, markersize = 2.0, color=3, legend=nothing)
    p2 = plot!(Win0, Length_W[:,6]+Length_W[:,10], label="Shift_Dr", style=:dash, width=2.0, marker=:utriangle, markersize = 2.0, color=3, legend=nothing)

    p2 = plot!(Win0, Green_W[:,8], label="Inj", width=2.0, marker=:circle, markersize = 2.0, color=4)
    p2 = plot!(Win0, Velocity_W[:,8], label="Inj_Dr", style=:dash, width=2.0, marker=:square, markersize = 2.0, color=4, legend=nothing)
    p2 = plot!(Win0, Length_W[:,8], label="Inj_Dr", style=:dash, width=2.0, marker=:utriangle, markersize = 2.0, color=4, legend=nothing)
    savefig(p2,"./PV_CP_Wdep_η$(η).png")
end

@time main(ARGS)