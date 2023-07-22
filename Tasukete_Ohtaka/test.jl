using Plots
using SharedArrays
using Distributed
addprocs(24)
@everywhere using LinearAlgebra
@everywhere using SymPy

ENV["GKSwstype"]="nul"

function main()
    println("start!")
    data = SharedArray{Float64}(24,100)
    @sync @distributed for i in 1:24
        x = symbols("x")
        f = sin(x)
        g = f.integrate(x)
        t = collect(0:2pi/100:2pi)
        data[i,:] = [N(g.subs(x,i*t[j])) for j in 1:100]
    end

    p1 = plot(data[1,:], linewidth=2.0)
    p1 = plot!(data[2,:], linewidth=2.0)
    p1 = plot!(data[3,:], linewidth=2.0)
    p1 = plot!(data[4,:], linewidth=2.0)
    savefig(p1, "test.png")
end

@time main()