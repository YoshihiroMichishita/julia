using LinearAlgebra
using Distributions
#=
struct TS_env
    t_size::Int
    HS_size::Int
    num_parm::Int
    Ω::Float32
    ξ::Float32
    Jz::Float32
    Jx::Float32
    hz::Float32
    #H_0::Matrix{ComplexF32}
    #V_t::Matrix{ComplexF32}
    H_0::Hermitian{ComplexF32, Matrix{ComplexF32}}
    V_t::Hermitian{ComplexF32, Matrix{ComplexF32}}
    #σ_vec::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}}
    #σ_vec2::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}}
    dt::Float32
end=#

#Hermite行列をベクトルに変換するためのベクトルを生成
function generate_M(H_size::Int)
    A::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}} =[]
    for i in 1:H_size
        for j in i:H_size
            if(i==j)
                #l = 2(d+1-i)*(i-1) + (i-1)^2 + 1
                B = zeros(ComplexF32,H_size,H_size)
                B[i,j] = 1.0
                B = Hermitian(B)
                push!(A,B)
                #A[l,:,:] = B[:,:]
            else
                #l = 2(d+1-i)*(i-1) + (i-1)^2 + 2*(j-i)
                B = zeros(ComplexF32,H_size,H_size)
                B[i,j] = 0.5
                B[j,i] = 0.5
                B = Hermitian(B)
                B2 = zeros(ComplexF32,H_size,H_size)
                B2[i,j] = -0.5im
                B2[j,i] = 0.5im
                B2 = Hermitian(B2)
                #A[l,:,:] = B[:,:]
                #A[l+1,:,:] = B2[:,:]
                push!(A,B)
                push!(A,B2)
            end
        end
    end
    return A
end

#ベクトルをHermite行列に変換するためのベクトルを生成
function generate_M2(H_size::Int)
    A::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}} =[]
    for i in 1:H_size
        for j in i:H_size
            if(i==j)
                #l = 2(d+1-i)*(i-1) + (i-1)^2 + 1
                B = zeros(ComplexF32,H_size,H_size)
                B[i,j] = 1f0
                B = Hermitian(B)
                push!(A,B)
                #A[l,:,:] = B[:,:]
            else
                #l = 2(d+1-i)*(i-1) + (i-1)^2 + 2*(j-i)
                B = zeros(ComplexF32,H_size,H_size)
                B[i,j] = 1f0
                B[j,i] = 1f0
                B = Hermitian(B)
                B2 = zeros(ComplexF32,H_size,H_size)
                B2[i,j] = -1f0im
                B2[j,i] = 1f0im
                B2 = Hermitian(B2)
                #A[l,:,:] = B[:,:]
                #A[l+1,:,:] = B2[:,:]
                push!(A,B)
                push!(A,B2)
            end
        end
    end
    return A
end

function init_env(t::Int=100, Ω0::Float32 = 10.0, ξ0::Float32 = 0.2, Jz0::Float32 = 1.0, Jx0::Float32 = 0.7, hz0::Float32 = 0.5)
    t_size::Int=t
    HS_size::Int = 4
    num_parm::Int = 5
    Ω::Float32 = Ω0
    ξ::Float32 = ξ0
    Jz::Float32 = Jz0
    Jx::Float32 = Jx0
    hz::Float32 = hz0

    H_0::Hermitian{ComplexF32, Matrix{ComplexF32}} = Hermitian([ -Jz-2hz 0 0 -Jx; 0 Jz -Jx 0; 0 -Jx Jz 0; -Jx 0 0 -Jz+2hz])
    V_t::Hermitian{ComplexF32, Matrix{ComplexF32}} = Hermitian([ 0 -ξ -ξ 0; -ξ 0 0 -ξ; -ξ 0 0 -ξ; 0 -ξ -ξ 0])

    #σ_vec = generate_M(HS_size)
    #σ_vec2 = generate_M2(HS_size)

    dt = Float32(2pi/t_size/Ω)

    return t_size, HS_size, num_parm, Ω, ξ, Jz, Jx, hz, H_0, V_t, dt
end

#ランダムに周期駆動系のパラメータを生成(とりあえず正規分布としておく)
#=
function init_env_randn(t::Int=100, Ω0::Float32, ξ0::Float32, Jz0::Float32, Jx0::Float32, hz0::Float32)
    t_size::Int=t
    HS_size::Int = 4
    num_parm::Int = 5
    Ω::Float32 = Ω0 + 5randn(Float32)
    ξ::Float32 = ξ0*randn(Float32)
    Jz::Float32 = Jz0*randn(Float32)
    Jx::Float32 = Jx0*randn(Float32)
    hz::Float32 = hz0*randn(Float32)
    println("=========================")
    println("init_env_randn")
    print("Ω = $(Ω), ")
    print("ξ = $(ξ), ")
    print("Jz = $(Jz), ")
    print("Jx = $(Jx), ")
    println("hz = $(hz)")

    H_0::Hermitian{ComplexF32, Matrix{ComplexF32}} = Hermitian([ -Jz-2hz 0 0 -Jx; 0 Jz -Jx 0; 0 -Jx Jz 0; -Jx 0 0 -Jz+2hz])
    V_t::Hermitian{ComplexF32, Matrix{ComplexF32}} = Hermitian([ 0 -ξ -ξ 0; -ξ 0 0 -ξ; -ξ 0 0 -ξ; 0 -ξ -ξ 0])

    #σ_vec = generate_M(HS_size)
    #σ_vec2 = generate_M2(HS_size)

    dt = Float32(2pi/t_size/Ω)

    return t_size, HS_size, num_parm, Ω, ξ, Jz, Jx, hz, H_0, V_t, dt

end=#

function generate_parmv(Ω::Float32, Jz::Float32, Jx::Float32, hz::Float32, ξ::Float32)
    Ω1::Float32 = Ω + 5*Float32(rand(Uniform(-1.0f0, 1.0f0)))
    ξ1::Float32 = ξ + Float32(0.3rand(Uniform(-1.0f0, 1.0f0)))
    Jz1::Float32 = Jz + Float32(rand(Uniform(-1.0f0, 1.0f0)))
    Jx1::Float32 = Jx + Float32(rand(Uniform(-1.0f0, 1.0f0)))
    hz1::Float32 = hz + Float32(rand(Uniform(-1.0f0, 1.0f0)))
    parm_v = [Ω1, Jz1, Jx1, hz1, ξ1]
    return parm_v
end
#structにすればいける？


struct Env
    σ1::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}}
    σ2::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}}
end

function set_Env(HS_size::Int)
    σ_vec = generate_M(HS_size)
    σ_vec2 = generate_M2(HS_size)
    return Env(σ_vec, σ_vec2)
end


#Translate Hermite Matrix to Vector
function MtoV(M::Hermitian{ComplexF32, Matrix{ComplexF32}}, en::Env)
    #V = [real(tr(σ1[i] * M)) for i in 1:16]
    V = real.(tr.(en.σ1 .* (M,)))
    return V
end

#Translate Vector to Hermite Matrix
function VtoM(V::Vector{Float32}, en::Env)
    M = V' * en.σ2
    return M
end
    
    
    