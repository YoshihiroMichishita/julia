using LinearAlgebra

module Parm
    export parm1

    t=1.0
    V=0.5
    K_SIZE=100
    function parm1(filename)
        include(pwd()*"/"*filename)
        return t,V,K_SIZE
    end
end

using .Parm
Parm.parm1("./julia/parm.jl")
#Parm.parm1(ARGS[1])

function Ham(k)
    H = [Parm.t*(cos(k[1])+cos(k[2])) Parm.V
    Parm.V -Parm.t*(cos(k[1])+cos(k[2]))]
    return H
end

q = collect(Float64,0:pi/Parm.K_SIZE:4pi)
s = length(q)
ev1 = zeros(s)
ev2 = zeros(s)


for i = 1:s
    if i<(s/4)
        k = [q[i],0]
        H = Ham(k)
        e,d = eigen(H)
        ev1[i] = e[1]
        ev2[i] = e[2]
    
    elseif i<(s/2) && i>=(s/4)
        k = [2pi-q[i],q[i]-pi]
        H = Ham(k)
        e,d = eigen(H)
        ev1[i] = e[1]
        ev2[i] = e[2]
    elseif i<(3*s/4) && i>=(s/2)
        k = [0,3pi-q[i]]
        H = Ham(k)
        e,d = eigen(H)
        ev1[i] = e[1]
        ev2[i] = e[2]
    else
        k = [q[i]-3pi,q[i]-3pi]
        H = Ham(k)
        e,d = eigen(H)
        ev1[i] = e[1]
        ev2[i] = e[2]
    end
end



using Plots

plot(q, ev1)
plot!(q, ev2)
#savefig("plot.png")
