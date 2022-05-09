using LinearAlgebra
using BenchmarkTools
using Base.Threads

#ハミルトニアンのパラメータを内包する構造体。パラメーターを振って物理量を計算しグラフを書くときはmutableにした方が良いかも。
struct Parm
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
mutable struct Hamiltonian
    Hk::Array{ComplexF64,2}
    Vx::Array{ComplexF64,2}
    Vy::Array{ComplexF64,2}
    Vxx::Array{ComplexF64,2}
    Vyx::Array{ComplexF64,2}
    #後々非エルミートに拡張できるようにComplexF64にしているが、別にFloat64でも良いはず
    E::Array{ComplexF64,1}
end

#遅延及び先進グリーン関数を内包する構造体
mutable struct Green
    GR::Array{ComplexF64,2}
    GA::Array{ComplexF64,2}
    dGR::Array{ComplexF64,2}
    dGA::Array{ComplexF64,2}
end


function HandV_topo(k::Array{Float64},p::Parm)
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

function Gk(w::Float64, p::Parm, Ham::Hamiltonian)
    #Green関数のinverse
    GR0::Array{ComplexF64,2} = -Ham.Hk + Matrix{Complex{Float64}}(w*I,2,2) + p.eta*Matrix{Complex{Float64}}(1.0im*I,2,2)

    GR::Array{ComplexF64,2} = inv(GR0)
    GA::Array{ComplexF64,2} = GR'
    dGR::Array{ComplexF64,2} = - GR * GR
    dGA::Array{ComplexF64,2} = - GA * GA
    return GR, GA, dGR, dGA
end


f(e,T) = 1.0/(1.0+exp(e/T))
df(e,T) = -1.0/(1.0+exp(e/T))/(1.0+exp(-e/T))/T

function Green_LR(p::Parm, H::Hamiltonian)
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

#ここで並列処理用に配列を用意しておく
Dr1 = zeros(Float64,p1.K_SIZE)
QHE1 = zeros(Float64,p1.K_SIZE)



kx = collect(Float64,-pi:2*pi/p1.K_SIZE:pi)
ky = collect(Float64,-pi:2*pi/p1.K_SIZE:pi)

for x = 1:p1.K_SIZE
    #ちゃんと並列化されてるかの確認。何番目のスレッドで実行しているかを出力。全部出力されるとめんどくさいので50個に１つにしている
    if(x%50==0)
        println("Thread = $(threadid())")
    end
    for y = 1:p1.K_SIZE
        kk = [kx[x],ky[y]]
        Hamk = Hamiltonian(HandV_topo(kk,p1)...)
        d, q = Green_LR(p1,Hamk)
        Dr1[x] += d/(p1.K_SIZE^2)
        QHE1[x] += q/(p1.K_SIZE^2)
    end
end

Dr0 = 0.0
QHE0 = 0.0
#reduction が分からなかったのでここでsumを取る
for x = 1:p1.K_SIZE
    global Dr0 += Dr1[x]
    global QHE0 += QHE1[x]
end
println("Drude = $(Dr0), QHE = $(QHE0)")