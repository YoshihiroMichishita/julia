using Distributed
addprocs(20)
@everywhere using LinearAlgebra

#ハミルトニアンのパラメータを内包する構造体。パラメーターを振って物理量を計算しグラフを書くときはmutableにした方が良いかも。
@everywhere struct Parm
    t::Float64
    lamda::Float64
    M::Float64
    mu::Float64
    eta::Float64
    T::Float64
    W_MAX::Float64
    K_SIZE::Int
    W_SIZE::Int
end

#ハミルトニアン及びそこから派生して速度演算子を内包する構造体
@everywhere mutable struct Hamiltonian
    Hk::Array{ComplexF64,2}
    Vx::Array{ComplexF64,2}
    Vy::Array{ComplexF64,2}
    Vxx::Array{ComplexF64,2}
    Vyx::Array{ComplexF64,2}
    #後々非エルミートに拡張できるようにComplexF64にしているが、別にFloat64でも良いはず
    E::Array{ComplexF64,1}
end

#遅延及び先進グリーン関数を内包する構造体
@everywhere mutable struct Green
    GR::Array{ComplexF64,2}
    GA::Array{ComplexF64,2}
    dGR::Array{ComplexF64,2}
    dGA::Array{ComplexF64,2}
end


@everywhere function HandV_topo(k::NTuple{2, Float64},p::Parm)
    H::Array{ComplexF64,2} = [-p.t*(cos(k[1])+cos(k[2])-2.0)+p.mu+p.M p.lamda*(-sin(k[1])-im*sin(k[2]))
    p.lamda*(-sin(k[1])+im*sin(k[2])) p.t*(cos(k[1])+cos(k[2])-2.0)+p.mu-p.M]

    Vx::Array{ComplexF64,2} = [p.t*(sin(k[1])) p.lamda*(-cos(k[1]))
    p.lamda*(-cos(k[1])) -p.t*(sin(k[1]))]

    Vy::Array{ComplexF64,2} = [p.t*(sin(k[2])) p.lamda*(-im*cos(k[2]))
    p.lamda*(im*cos(k[2])) -p.t*(sin(k[2]))]

    Vxx::Array{ComplexF64,2} = [p.t*(cos(k[1])) p.lamda*(sin(k[1]))
    p.lamda*(sin(k[1])) -p.t*(cos(k[1]))]

    Vyx::Array{ComplexF64,2} = [0.0 0.0
    0.0 0.0]

    E::Array{ComplexF64,1} = zeros(2)

    return H, Vx, Vy, Vxx, Vyx, E 
end

@everywhere function Gk(w::Float64, p::Parm, Ham::Hamiltonian)
    #Green関数のinverse
    GR0::Array{ComplexF64,2} = -Ham.Hk + Matrix{Complex{Float64}}(w*I,2,2) + p.eta*Matrix{Complex{Float64}}(1.0im*I,2,2)

    GR::Array{ComplexF64,2} = inv(GR0)
    GA::Array{ComplexF64,2} = GR'
    dGR::Array{ComplexF64,2} = - GR * GR
    dGA::Array{ComplexF64,2} = - GA * GA
    return GR, GA, dGR, dGA
end


@everywhere f(e,T) = 1.0/(1.0+exp(e/T))
@everywhere df(e,T) = -1.0/(1.0+exp(e/T))/(1.0+exp(-e/T))/T

@everywhere function Green_LR(p::Parm, H::Hamiltonian)
    Drude::Float64 = 0.0
    QH::Float64 = 0.0

    for w = collect(Float64,-p.W_MAX:2*p.W_MAX/p.W_SIZE:p.W_MAX)
        G = Green(Gk(w,p,H)...)
        Drude += real(tr(H.Vx*G.GR*H.Vx*(G.GR .- G.GA))*df(w,p.T)/p.W_SIZE)
        QH += real(tr(H.Vy*G.dGR*H.Vx*(G.GR .- G.GA))*f(w,p.T)/p.W_SIZE)
    end
    return Drude, QH
end

p1 = Parm(0.4, 1.0, -1.0, 0, 0.01, 0.005, 2.0, 300, 1000)

function main(ARGS)
    k = collect(Iterators.product((-pi:2*pi/p1.K_SIZE:pi)[1:end-1], (-pi:2*pi/p1.K_SIZE:pi)[1:end-1]))
    
    Dr0, QHE0 = @distributed (+) for i in 1:length(k)
        Hamk = Hamiltonian(HandV_topo(k[i],p1)...)
        d, q = Green_LR(p1,Hamk)
        [d/(p1.K_SIZE^2), q/(p1.K_SIZE^2)]
    end

    println("Drude = $(Dr0), QHE = $(QHE0)")
end

#main()
@time main()