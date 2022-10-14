include("model_2D_IB.jl")

using Plots
function get_HSL(K_SIZE::Int)
    q::Vector{Vector{Float64}} = []
    dk = pi/K_SIZE
    k = range(0, pi-dk, length=K_SIZE)
    for ik in k
        kk::Vector{Float64} = [ik, 0.0]
        push!(q, kk)        
    end
    for ik in k
        kk::Vector{Float64} = [pi-ik, ik]
        push!(q, kk)        
    end
    for ik in k
        kk::Vector{Float64} = [0.0, pi-ik]
        push!(q, kk)        
    end
    for ik in k
        kk::Vector{Float64} = [ik, ik]
        push!(q, kk)        
    end
    return q
end

function main(arg::Array{String,1})
    p = Parm(set_parm_Wdep(arg, 0.01)...)
    q = get_HSL(p.K_SIZE)
    q_size = size(q)[1]
    Aw = zeros(Float64, p.w_size, q_size)
    for id_q in 1:q_size
        H = set_H(q[id_q],p)
        for id_w in 1:p.w_size
            w = p.w_mesh[id_w]
            GR0::Array{ComplexF64,2} = -H + Matrix{Complex{Float64}}(w*I,2,2) + p.eta*Matrix{Complex{Float64}}(1.0im*I,2,2) - p.Σw[id_w]

            GR::Array{ComplexF64,2} = inv(GR0)
            Aw[id_w, id_q] = -imag(tr(GR))/pi
        end
    end
    ENV["GKSwstype"]="nul"
    Plots.scalefontsizes(1.4)

    q_int = 1:4p.K_SIZE

    p1 = plot(q_int, p.w_mesh, Aw, st=:heatmap, label="e",xlabel="HSL",xticks=([0, p.K_SIZE, 2p.K_SIZE, 3p.K_SIZE, 4p.K_SIZE],["Γ", "X", "X'", "Γ", "M"]),ylabel="E",title="Dispersion", width=3.0, range=[0,10])
    savefig(p1,"./disp_int.png")
end
    


@time main(ARGS)