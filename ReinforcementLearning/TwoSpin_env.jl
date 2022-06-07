using LinearAlgebra


#SU(4) generator
#=
G1::Matrix{ComplexF64} = [0.0 1.0 0.0 0.0; 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
G2::Matrix{ComplexF64} = [0.0 -1.0im 0.0 0.0; 1.0im 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
G3::Matrix{ComplexF64} = [1.0 0.0 0.0 0.0; 0.0 -1.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
G4::Matrix{ComplexF64} = [0.0 0.0 1.0 0.0; 0.0 0.0 0.0 0.0; 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
G5::Matrix{ComplexF64} = [0.0 0.0 -1.0im 0.0; 0.0 0.0 0.0 0.0; 1.0im 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
G6::Matrix{ComplexF64} = [0.0 0.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 0.0 0.0]
=#


mutable struct TS_env
    t_size::Int
    HS_size::Int
    num_parm::Int
    Ω::Float64
    ξ::Float64
    Jz::Float64
    Jx::Float64
    hz::Float64
    H_0::Matrix{ComplexF64}
    V_t::Matrix{ComplexF64}
end

function init_env()
    t_size::Int=100
    HS_size::Int = 4
    num_parm::Int = 5
    Ω::Float64 = 10.0
    ξ::Float64 = 0.2
    Jz::Float16 = 1.0
    Jx::Float16 = 0.7
    hz::Float64 = 0.5
    H_0::Matrix{ComplexF64} = [ -Jz-2hz, 0, 0, -Jx; 0, Jz, -Jx, 0; 0, -Jx, Jz, 0; -Jx, 0, 0, -Jz+2hz]
    V_t::Matrix{ComplexF64} = [ 0 , -ξ, -ξ, 0; -ξ, 0, 0, -ξ; -ξ, 0, 0, -ξ; 0, -ξ, -ξ, 0]

    return t_size, HS_size, num_parm, Ω, ξ, Jz, Jx, hz, H_0, V_t
end

function vec_to_matrix(v::Vector{Float64})
    d::Int = sqrt(length(v))
    M = zeros(ComplexF64,d,d)
    for i in 1:d
        for j in i:d
            l = (i-1)*d + 2*(j-1)
            if(i==j)
                M[i,j] = v[l+1]
            else
                M[i,j] = v[l] + im*v[l+1]
            end
        end
    end
    Kt = Hermitian(M)
    
    return Kt
end

function matrix_to_vec(M::Hermitian{ComplexF64, Matrix{ComplexF64}})
    d = sinze(M)[1]
    v::Vector{Float64} = []
    for i in 1:d
        for j in i:d
            if(i==j)
                push!(v, M[i,j])
            else
                push!(v,M[i,j].real)
                push!(v,M[i,j].imag)
            end
        end
        
    end
    return v
end

function reward(v_old::Vector{Float64}, v_new::Vector{Float64}, en::TS_env)
    K_old = vec_to_matrix(v_old)
    K_new = vec_to_matrix(v_new)
    dt = 2pi/(en.Ω*en.t_size)
    Kp = (K_new - K_old)/dt
    U = exp(im*K_new)
    Vp = U * (en.V_t - Kp) * U'
    e, vec = eigen(Vp)
    r = - e'* e
    return r
end
    
    