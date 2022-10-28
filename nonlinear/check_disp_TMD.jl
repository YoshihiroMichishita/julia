include("./2D_TMD_parm.jl")
include("./transport.jl")
include("./k_C3.jl")

using Plots

function main(arg::Array{String,1})
    p = Parm(set_parm(arg)...) 
    E = Disp_HSL(p, 2)

    ENV["GKSwstype"]="nul"
    Plots.scalefontsizes(1.4)

    #q_int = 1:4p.K_SIZE

    p1 = plot(E[:,1],xticks=([0, p.K_SIZE/2, p.K_SIZE, 2p.K_SIZE, 3p.K_SIZE, 4p.K_SIZE],["K", "M", "K'", "M'", "Γ", "K"]),xlabel="HSL",ylabel="ϵ",title="Dispersion", linewidth=3.0, yrange=[-1.5,0.5], legend = false)
    p1 = plot!(E[:,2], linewidth=3.0, legend = false)
    savefig(p1,"./disp_check.png")
end
    


@time main(ARGS)