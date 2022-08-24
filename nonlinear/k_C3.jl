using LinearAlgebra
#create mesh over the BZ
function get_kk(K_SIZE::Int)
    kk = Vector{NTuple{2, Float64}}(undef,0)
    dk = 4pi/(3K_SIZE)
    #dk2 = 2.0/(3*sqrt(3.0)*K_SIZE*K_SIZE)
    for i in collect(dk:dk:4pi/3)
        
        for j in collect(0:dk:4pi/3)
            k = j*a1 + i*a2
            push!(kk,(k[1],k[2]))
        end
        
        
        for j in collect(dk:dk:4pi/3)
            if (i+j) < (4pi/3+dk)
                k = -j*a1 + i*a3
                push!(kk,(k[1],k[2]))
            end
        end
    end
    l = length(kk)
    for i in 1:l
        k0 = kk[i]
        k0 = -1 .* k0
        push!(kk,k0)
    end
    for i in collect(-4pi/3:dk:4pi/3)
        k = i*a1
        push!(kk,(k[1],k[2]))
    end
    return kk
end

function Disp_HSL(p::Parm, H_SIZE::Int)
    E = zeros(Floart64, 4*p.K_SIZE, H_SIZE)

    for K0 in 1:p.K_SIZE
        KK = 4pi/3*K0/p.K_SIZE
        k = 4pi/3*a1 + KK*a2
        kk = (k[1], k[2])
        H = Hamiltonian(HandV(kk,p)...)
        e, v = eigen(H.Hk)
        E[K0,:] = real.(e)
    end
    for K0 in 1:p.K_SIZE
        KK = 2pi/3*K0/p.K_SIZE
        k = 4pi/3*a3 - KK*a1
        kk = (k[1], k[2])
        H = Hamiltonian(HandV(kk,p)...)
        e, v = eigen(H.Hk)
        E[K0+p.K_SIZE,:] = real.(e)
    end
    for K0 in 1:p.K_SIZE
        KK = 4pi/3*K0/p.K_SIZE
        k = 4pi/3*a3 -2pi/3*a1 + KK*(0.5*a1 - a3) 
        kk = (k[1], k[2])
        H = Hamiltonian(HandV(kk,p)...)
        e, v = eigen(H.Hk)
        E[K0+2p.K_SIZE,:] = real.(e)
    end
    for K0 in 1:p.K_SIZE
        KK = 4pi/3*K0/p.K_SIZE
        k = KK*(a1)
        kk = (k[1], k[2])
        H = Hamiltonian(HandV(kk,p)...)
        e, v = eigen(H.Hk)
        E[K0+3p.K_SIZE,:] = real.(e)
    end
    return E
end