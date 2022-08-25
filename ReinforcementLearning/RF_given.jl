using LinearAlgebra

struct TS_env
    t_size::Int
    H_size::Int
    Ω::Float64
    σ_vec::σ_vec::Vector{Hermitian{ComplexF64, Matrix{ComplexF64}}}
end

function generate_M(H_size::Int)
    A::Vector{Hermitian{ComplexF64, Matrix{ComplexF64}}} =[]
    for i in 1:H_size
        for j in i:H_size
            if(i==j)
                #l = 2(d+1-i)*(i-1) + (i-1)^2 + 1
                B = zeros(ComplexF64,H_size,H_size)
                B[i,j] = 1.0
                B = Hermitian(B)
                push!(A,B)
                #A[l,:,:] = B[:,:]
            else
                #l = 2(d+1-i)*(i-1) + (i-1)^2 + 2*(j-i)
                B = zeros(ComplexF64,H_size,H_size)
                B[i,j] = 0.5
                B[j,i] = 0.5
                B = Hermitian(B)
                B2 = zeros(ComplexF64,H_size,H_size)
                B2[i,j] = 0.5im
                B2[j,i] = -0.5im
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

function init_env(t::Int, Hs::Int, Ω::Float64)
    σ = generate_M(Hs)
    return t, Hs, Ω, σ
end

function VtoM(V::Vector{Float64},en::TS_env)
    M = V' * en.σ_vec
    return M
end

function MtoV(M::Hermitian{ComplexF64, Matrix{ComplexF64}}, en::TS_env)
    V = real.(tr.(en.σ_vec .* (M,)))
    return V
end

function KtoKp(en::TS_env, K::Array{Float64,2}, Ht::Array{Float64,2})
    dt = 2pi/en.Ω/en.t_size
    Kp = zeros(Float64, size(K)[1], size(K)[2])
    HF = zeros(Float64, size(K)[1], size(K)[2])
    for t in 1:size(K)[1]
        if(t>1)
            tt=t-1
        else
            tt=size(K)[1]
        end
        Kp[t,:] = (K[t,:]-K[tt,:])/dt
        KM = VtoM(K[t,:],en)
        HfM = exp(-1.0im*KM)(VtoM(Ht[t,:],en)-1.0im*VtoM(Kp[t,:],en))exp(1.0im*KM)
        HF[t,:] = MtoV(HfM, en)
    end
    return Kp, HF
end

function main(arg::vector{String})
    en = TS_env(init_env(parse(Int,arg[1]),parse(Int,arg[2]),parse(Float64,arg[3]))...)

    Ht = zeros(Float64, en.t_size, en.H_size^2)
    Kt = zeros(Float64, en.t_size, en.H_size^2)
    Kpt = zeros(Float64, en.t_size, en.H_size^2)
    HFt = zeros(Float64, en.t_size, en.H_size^2)
    for t in 1:en.t_size
        
        
    end
end
