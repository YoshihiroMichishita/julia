using LinearAlgebra
using Plots

struct Parm
    t_step::Int
    n_site::Int
    Ω::Float32
    ξ::Float32
    Jz::Float32
    Jx::Float32
    hz::Float32
    dt::Float32
    Ms::Int
end

function init_parm(arg::Vector{String})
    t_step = parse(Int, arg[1])
    n_site = parse(Int, arg[2])
    Ω = parse(Float32, arg[3])
    ξ = parse(Float32, arg[4])
    Jz = parse(Float32, arg[5])
    Jx = parse(Float32, arg[6])
    hz = parse(Float32, arg[7])
    #H_0 = Hermitian([ -Jz-2hz 0 0 -Jx; 0 Jz -Jx 0; 0 -Jx Jz 0; -Jx 0 0 -Jz+2hz])
    #V_t = Hermitian([ 0 -ξ -ξ 0; -ξ 0 0 -ξ; -ξ 0 0 -ξ; 0 -ξ -ξ 0])
    dt = 2pi/Ω/t_step
    Ms = 2^n_site

    #return Parm(t_step, HS_size, Ω, ξ, Jz, Jx, hz, H_0, V_t, dt)
    return Parm(t_step, n_site, Ω, ξ, Jz, Jx, hz, dt, Ms)
end

function init_parm_ξdep(arg::Vector{String}, ξ0::Float32)
    t_step = parse(Int, arg[1])
    n_site = parse(Int, arg[2])
    Ω = parse(Float32, arg[3])
    ξ = ξ0
    #parse(Float32, arg[4])
    Jz = parse(Float32, arg[5])
    Jx = parse(Float32, arg[6])
    hz = parse(Float32, arg[7])
    #H_0 = Hermitian([ -Jz-2hz 0 0 -Jx; 0 Jz -Jx 0; 0 -Jx Jz 0; -Jx 0 0 -Jz+2hz])
    #V_t = Hermitian([ 0 -ξ -ξ 0; -ξ 0 0 -ξ; -ξ 0 0 -ξ; 0 -ξ -ξ 0])
    dt = 2pi/Ω/t_step
    Ms = 2^n_site

    #return Parm(t_step, HS_size, Ω, ξ, Jz, Jx, hz, H_0, V_t, dt)
    return Parm(t_step, n_site, Ω, ξ, Jz, Jx, hz, dt, Ms)
end

struct System
    H_0::Hermitian{Float32, Matrix{Float32}}
    V_t::Hermitian{Float32, Matrix{Float32}}
end

function vec2ind(n::Int, v::Vector{Bool})
    ind = 0
    for i in 1:n
        ind += Int(v[i])*2^(i-1)
    end
    return ind+1
end

function onebody!(p::Parm,v::Vector{Bool}, H::Matrix{Float32}, V::Matrix{Float32})
    id = vec2ind(p.n_site, v)
    for it in 1:p.n_site
        sw = v[it]
        sz = 2sw-1
        H[id, id] += p.hz*sz
        v2 = copy(v)
        v2[it] = true - sw
        id2 = vec2ind(p.n_site, v2)
        V[id, id2] += p.ξ
    end
end

function twobody!(p::Parm,v::Vector{Bool}, H::Matrix{Float32})
    id = vec2ind(p.n_site, v)
    for it in 1:p.n_site-1
        next = it+1
        sw1 = v[it]
        sw2 = v[next]
        sz1 = 2sw1-1
        sz2 = 2sw2-1
        H[id, id] += p.Jz*sz1*sz2
        v2 = copy(v)
        v2[it] = true - sw1
        v2[next] = true - sw2
        id2 = vec2ind(p.n_site, v2)
        H[id, id2] += p.Jx
    end
end

function update!(v::Vector{Bool})
    for i in 1:length(v)
        if(v[i])
            v[i] = false
        else
            v[i] = true
            break
        end
    end
end

function init_system(p::Parm)
    H = zeros(Float32, p.Ms, p.Ms)
    V = zeros(Float32, p.Ms, p.Ms)
    v::Vector{Bool} = [false for i in 1:p.n_site]
    for i in 1:p.Ms
        onebody!(p, v, H, V)
        twobody!(p, v, H)
        update!(v)
    end
    H0 = Hermitian(H)
    Vt = Hermitian(V)
    return System(H0, Vt)
end

function calc_Kt(history::Vector{Int}, p::Parm, s::System)
    MV = []
    his = copy(history)
    #println(length(his))
    for it in 1:length(his)
        sw = pop!(his)
        if(sw==1)
            push!(MV, s.H_0)
        elseif(sw==2)
            push!(MV, sx*s.V_t)
        elseif(sw==3)
            A = pop!(MV)
            B = pop!(MV)
            C = A + B
            push!(MV, C)
        elseif(sw==4)
            A = pop!(MV)
            B = pop!(MV)
            C = -1im*(A*B - B*A)
            push!(MV, C)
        elseif(sw==5)
            A = pop!(MV)
            B = pop!(MV)
            C = (A*B + B*A)/2
            push!(MV, C)
        elseif(sw==6)
            A = pop!(MV)
            try
                #B = A.integrate((x, 0, x))/env.Ω
                B = A.integrate(x)/p.Ω
            catch
                B = A
            end
            push!(MV, B)
        end
        #@show MV
    end
    t = collect(0:p.Ω*p.dt:2pi)

    Ks = MV[end]
    K0 = N(Ks.subs(x,0.0f0))
    return Ks-K0
    #println(Ks)
    #=
    if(typeof(Ks)==Matrix{Sym})
        K0 = N(Ks.subs(x,t[1]))
        #Kt::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}} = [Hermitian(N(Ks.subs(x,t[i]))) for i in 1:env.t_step]
        Kt::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}} = [Hermitian(N(Ks.subs(x,t[i]))-K0) for i in 1:p.t_step]
        return Kt
    else
        Kh::Vector{Hermitian{ComplexF32, Matrix{ComplexF32}}} = [Hermitian(Ks) for i in 1:p.t_step]
        return Kh
    end=#
    #Kt = [Hermitian(N(Ks.subs(x,t[i]))) for i in 1:env.t_step]
    
end

function calc_Hr(p::Parm, s::System, Kt)
    HFn = Hermitian(zeros(ComplexF32, p.Ms, p.Ms))
    Vp = Hermitian(zeros(ComplexF32, p.Ms, p.Ms))
    Vm = Hermitian(zeros(ComplexF32, p.Ms, p.Ms))
    for i in 1:p.t_step
        t = p.Ω*i*p.dt
        U = exp(1im*N(Kt.subs(x, t)))
        HFt = Hermitian(U*(s.H_0 + s.V_t*sin(t)) * U' - 1im* U*(exp(-1im*N(Kt.subs(x, t+p.Ω*p.dt)))-exp(-1im*N(Kt.subs(x, t-p.Ω*p.dt))))/2p.dt)/p.t_step
        HFn += HFt
        Vp += HFt*exp(1im*p.Ω*i*p.dt)
        Vm += HFt*exp(-1im*p.Ω*i*p.dt)
    end
    return HFn, Vp, Vm
end

function calc_HR_p(p::Parm, HFn::Hermitian{ComplexF32, Matrix{ComplexF32}}, Vp::Hermitian{ComplexF32, Matrix{ComplexF32}}, β::Float32, sts::Vector{Vector{ComplexF32}})
    U = exp(Float32(2pi)*1im*HFn/p.t_step)
    Vdp = Hermitian(-1im*(Vp*HFn - HFn*Vp))
    Vt = Vdp
    C = 0.0f0
    l = length(sts)
    for i in 1:p.t_step
        Vt = Hermitian(U*Vt*U')
        for st in sts
            C += p.dt* real(st'*Vt*Vdp*st) * exp(1im*p.Ω*i*p.dt)/l
        end
    end
    C = (1-exp(-β*p.Ω))/(2p.Ω)*C
    return C
end

function calc_HR_m(p::Parm, HFn::Hermitian{ComplexF32, Matrix{ComplexF32}}, Vm::Hermitian{ComplexF32, Matrix{ComplexF32}}, β::Float32, sts::Vector{Vector{ComplexF32}})
    U = exp(Float32(2pi)*1im*HFn/p.t_step)
    Vdm = Hermitian(-1im*(Vm*HFn - HFn*Vm))
    Vt = Vdm
    C = 0.0f0
    l = length(sts)
    for i in 1:p.t_step
        Vt = Hermitian(U*Vt*U')
        for st in sts
            C += p.dt* real(st'*Vt*Vdm*st) * exp(1im*p.Ω*i*p.dt)/l
        end
    end
    C = (1-exp(-β*p.Ω))/(-2p.Ω)*C
    return C
end


#=
function calc_Udt_orig(old_Udt::Matrix,t::Int, p::Parm, s::System)
    newU = (Matrix(I, p.Ms, p.Ms) - 1.0f0im*(s.H_0 + Float32(sign(cos(p.Ω*t*p.dt)))*s.V_t)*p.dt)*old_Udt
    #newU = newU/sqrt((newU*newU')[1,1])
    return newU
end=#

function therm_U(β::Float32, s::System)
    e,v = eigen(s.H_0)
    println(e[end]-e[1])
    U = v*Diagonal(exp.(-β*e))*v'
    return U
end

function calc_E(st::Vector{ComplexF32}, s::System)
    st2 = s.H_0*st
    E = real(st' * st2)/real(st' * st)
    return E
end

function init_state(p::Parm, thermoU::Matrix{Float32}, β_it::Int)
    st = randn(ComplexF32, p.Ms)
    st = st/sqrt(st'*st)
    for it in 1:β_it
        st = thermoU*st
        st = st/sqrt(st'*st)
    end
    return st
end
#=
function main(ARG)
    p = init_parm(ARG)
    s = init_system(p)

    ϵ1 = parse(Float32, ARG[8])
    ϵ2 = parse(Float32, ARG[9])
    β = parse(Int, ARG[10])

    Udt = Matrix(I, p.Ms, p.Ms)
    for t in 1:p.t_step
        Udt = calc_Udt_orig(Udt, t, p, s)
    end
    Udt = Udt/sqrt((Udt*Udt')[1,1])
    
    thermoU = therm_U(β, s)
    
    κ = 0.0
    ti = zeros(Int,2)
    sample = 1
    for it in 1:sample
        st = init_state(p, thermoU)
        
        #ti = zeros(Int,2)
        for t in 1:30
            st = Udt*st
            st = st/sqrt(st'*st)
            E = calc_E(st, s)
            println(E)
            if(E>ϵ1 && ti[1]==0)
                ti[1] = t
            end
            if(E>ϵ2 && ti[2]==0)
                ti[2] = t
                break
            end
            
        end
        println("============")
        #κ += (ϵ2-ϵ1)/((ti[2]-ti[1])*p.dt*p.t_step)/sample
    end
    #println(ti)
    #println(κ)
end=#

using Plots
ENV["GKSwstype"]="nul"

function calc_HR(ARG)
    p = init_parm(ARG)
    s = init_system(p)

    #ϵ1 = parse(Float32, ARG[8])
    #ϵ2 = parse(Float32, ARG[9])
    β0 = parse(Float32, ARG[8])
    β_it = parse(Float32, ARG[9])
    β = β0*β_it

    hist = parse.(Int, ARG[10:end])

    HFn, Vp, Vm = calc_Hr(p, s, calc_Kt(hist, p, s))
    thermoU = therm_U(β0, s)
    
    sample = 10
    sts::Vector{Vector{ComplexF32}} = []

    for it in 1:sample
        st = init_state(p, thermoU, β_it)
        push!(sts, st)
    end

    κ = calc_HR_p(p, HFn, Vp, β, sts) + calc_HR_m(p, HFn, Vm, β, sts)
    println("κ = $(κ)")
end

function calc_HR_ξdep(ARG, ξ)
    p = init_parm_ξdep(ARG, ξ)
    s = init_system(p)

    β0 = parse(Float32, ARG[8])
    β_it = parse(Float32, ARG[9])
    β = β0*β_it

    hist = parse.(Int, ARG[10:end])

    HFn, Vp, Vm = calc_Hr(p, s, calc_Kt(hist, p, s))
    thermoU = therm_U(β0, s)
    
    sample = 10
    sts::Vector{Vector{ComplexF32}} = []

    for it in 1:sample
        st = init_state(p, thermoU, β_it)
        push!(sts, st)
    end

    κ = calc_HR_p(p, HFn, Vp, β, sts) + calc_HR_m(p, HFn, Vm, β, sts)
    println("κ = $(κ)")
    return κ
end

function calc_HR_βdep(ARG, β_it::Int)
    p = init_parm(ARG)
    s = init_system(p)

    β0 = parse(Float32, ARG[8])
    #β_it = parse(Float32, ARG[9])
    β = β0*β_it

    hist = parse.(Int, ARG[10:end])

    HFn, Vp, Vm = calc_Hr(p, s, calc_Kt(hist, p, s))
    thermoU = therm_U(β0, s)
    
    sample = 10
    sts::Vector{Vector{ComplexF32}} = []

    for it in 1:sample
        st = init_state(p, thermoU, β_it)
        push!(sts, st)
    end

    κ = calc_HR_p(p, HFn, Vp, β, sts) + calc_HR_m(p, HFn, Vm, β, sts)
    println("κ = $(κ)")
    return κ
end

function main_ξ(ARG)
    κs = []
    ξs::Vector{Float32} = [0.025f0, 0.05f0, 0.1f0, 0.2f0, 0.4f0, 0.8f0, 1.6f0, 3.2f0, 6.4f0]
    for ξ in ξs
        κ = calc_HR_ξdep(ARG, ξ)
        push!(κs, κ)
    end
    println(κs)
    p0 = plot(ξs, κs, linewidth=2.0, marker=:circle, xscale=:log10, yscale=:log10)
    savefig(p0, "./HeatingRate_ξdep.png")
end

function main_β(ARG)
    κs = []
    β_its::Vector{Int} = [1, 5, 10, 20, 30, 40, 50, 70, 100]
    for β_it in β_its
        κ = calc_HR_ξdep(ARG, β_it)
        push!(κs, κ)
    end
    println(κs)
    p0 = plot(β_its, κs, linewidth=2.0, marker=:circle)
    savefig(p0, "./HeatingRate_βdep.png")
end

@time main_ξ(ARGS)
    