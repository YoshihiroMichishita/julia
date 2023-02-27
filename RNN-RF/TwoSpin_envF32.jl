using LinearAlgebra


#SU(4) generator
#=
G1::Matrix{ComplexF32} = [0.0 1.0 0.0 0.0; 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
G2::Matrix{ComplexF32} = [0.0 -1.0im 0.0 0.0; 1.0im 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
G3::Matrix{ComplexF32} = [1.0 0.0 0.0 0.0; 0.0 -1.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
G4::Matrix{ComplexF32} = [0.0 0.0 1.0 0.0; 0.0 0.0 0.0 0.0; 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
G5::Matrix{ComplexF32} = [0.0 0.0 -1.0im 0.0; 0.0 0.0 0.0 0.0; 1.0im 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
G6::Matrix{ComplexF32} = [0.0 0.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 0.0 0.0]
=#


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
    σ_vec::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}}
    σ_vec2::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}}
    p2::Float32
    i::ComplexF32
    dt::Float32
end

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

function generate_M2(H_size::Int)
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
                B[i,j] = 1.0
                B[j,i] = 1.0
                B = Hermitian(B)
                B2 = zeros(ComplexF32,H_size,H_size)
                B2[i,j] = -1.0im
                B2[j,i] = 1.0im
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
    #H_0::Matrix{ComplexF32} = [ -Jz-2hz 0 0 -Jx; 0 Jz -Jx 0; 0 -Jx Jz 0; -Jx 0 0 -Jz+2hz]
    #V_t::Matrix{ComplexF32} = [ 0 -ξ -ξ 0; -ξ 0 0 -ξ; -ξ 0 0 -ξ; 0 -ξ -ξ 0]

    H_0::Hermitian{ComplexF32, Matrix{ComplexF32}} = Hermitian([ -Jz-2hz 0 0 -Jx; 0 Jz -Jx 0; 0 -Jx Jz 0; -Jx 0 0 -Jz+2hz])
    V_t::Hermitian{ComplexF32, Matrix{ComplexF32}} = Hermitian([ 0 -ξ -ξ 0; -ξ 0 0 -ξ; -ξ 0 0 -ξ; 0 -ξ -ξ 0])

    σ_vec = generate_M(HS_size)
    σ_vec2 = generate_M2(HS_size)
    p2 = Float32(2pi)
    i = ComplexF32(1.0im)

    dt = p2/t_size/Ω

    return t_size, HS_size, num_parm, Ω, ξ, Jz, Jx, hz, H_0, V_t, σ_vec, σ_vec2, p2, i, dt
end

function VtoM(V::Vector{Float32},en::TS_env)
    M = V' * en.σ_vec2
    return M
end

function MtoV(M::Hermitian{ComplexF32, Matrix{ComplexF32}}, en::TS_env)
    V = real.(tr.(en.σ_vec .* (M,)))
    return V
end

function reward(v_old::Vector{Float32}, v_new::Vector{Float32}, en::TS_env)
    K_old = vec_to_matrix(v_old)
    K_new = vec_to_matrix(v_new)
    dt = en.p2/(en.Ω*en.t_size)
    Kp = (K_new - K_old)/dt
    U = exp(im*K_new)
    Vp = U * (en.V_t - Kp) * U'
    e, vec = eigen(Vp)
    r = - e'* e
    return r
end
    
    