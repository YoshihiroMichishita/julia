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
    dt = 2pi/Ω/t_step
    Ms = 2^n_site

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

#=
function twobody!(p::Parm,v::Vector{Bool}, H::Matrix{Float32})
    id = vec2ind(p.n_site, v)
    for it in 1:p.n_site
        next = it+1
        if(next>p.n_site)
            next = 1
        end
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
end=#

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

function calc_Udt_orig(old_Udt::Matrix,t::Int, p::Parm, s::System)
    newU = (Matrix(I, p.Ms, p.Ms) - 1.0f0im*(s.H_0 + Float32(sign(cos(p.Ω*t*p.dt)))*s.V_t)*p.dt)*old_Udt
    #newU = newU/sqrt((newU*newU')[1,1])
    return newU
end

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

day=1011

function calc_HR(ARG)
    p = init_parm(ARG)
    s = init_system(p)

    ϵ1 = parse(Float32, ARG[8])
    ϵ2 = parse(Float32, ARG[9])
    β = parse(Float32, ARG[10])
    β_it = parse(Float32, ARG[11])

    
    Udt = Matrix(I, p.Ms, p.Ms)
    for t in 1:p.t_step
        Udt = calc_Udt_orig(Udt, t, p, s)
        #=
        if(t%100==0)
            Udt = Udt/sqrt((Udt*Udt')[1,1])
        end=#
    end
    println((Udt*Udt')[1,1])
    Udt = Udt/sqrt((Udt*Udt')[1,1])
    
    thermoU = therm_U(β, s)
    
    κ = 0.0
    ti = zeros(Int,2)
    sample = 10
    t_size = 100
    Es = zeros(Float32, sample, t_size)
    for it in 1:sample
        st = init_state(p, thermoU, β_it)
        #Es = []
        #ti = zeros(Int,2)
        for t in 1:t_size
            st = Udt*st
            st = st/sqrt(st'*st)
            E = calc_E(st, s)
            Es[it, t] = E
            #push!(Es, E)
            #println(E)
            if(E>ϵ1 && ti[1]==0)
                ti[1] = t
            end
            if(E>ϵ2 && ti[2]==0)
                ti[2] = t
                break
            end
            
        end
        #println("============")
        #push!(Ess, Es)
        #κ += (ϵ2-ϵ1)/((ti[2]-ti[1])*p.dt*p.t_step)/sample
    end
    Ess = [sum(Es[:,t])/sample for t in 1:t_size]
    xx = [sum(Ess[10*(t-1)+1:10t])/10 for t in 1:10]
    println("κ = $((xx[5]-xx[2])*p.Ω/(2pi)/3)")
    p1 = plot(Ess, linewidth=2.0)
    p2 = plot(xx, linewidth=2.0, marker=:circle)
    #for it in 2:sample
    #    p1 = plot!(Ess[it], linewidth=2.0)
    #end
    savefig(p1, "./HeatingDynamics_ξ$(p.ξ).png")
    savefig(p2, "./HeatingDynamics_ξ$(p.ξ)_ave.png")
    #println(ti)
    #println(κ)
end

function calc_HR_ξdep(ARG, ξ)
    p = init_parm_ξdep(ARG, ξ)
    s = init_system(p)

    ϵ1 = parse(Float32, ARG[8])
    ϵ2 = parse(Float32, ARG[9])
    β = parse(Float32, ARG[10])
    β_it = parse(Int, ARG[11])

    
    Udt = Matrix(I, p.Ms, p.Ms)
    for t in 1:p.t_step
        Udt = calc_Udt_orig(Udt, t, p, s)
    end
    println((Udt*Udt')[1,1])
    Udt = Udt/sqrt((Udt*Udt')[1,1])
    
    thermoU = therm_U(β, s)
    
    κ = 0.0
    ti = zeros(Int,2)
    sample = 20
    t_size = 100
    Es = zeros(Float32, sample, t_size)
    for it in 1:sample
        st = init_state(p, thermoU, β_it)
        #Es = []
        #ti = zeros(Int,2)
        for t in 1:t_size
            st = Udt*st
            st = st/sqrt(st'*st)
            E = calc_E(st, s)
            Es[it, t] = E
            #push!(Es, E)
            #println(E)
            if(E>ϵ1 && ti[1]==0)
                ti[1] = t
            end
            if(E>ϵ2 && ti[2]==0)
                ti[2] = t
                break
            end
            
        end
        #println("============")
        #push!(Ess, Es)
        #κ += (ϵ2-ϵ1)/((ti[2]-ti[1])*p.dt*p.t_step)/sample
    end
    Ess = [sum(Es[:,t])/sample for t in 1:t_size]
    xx = [sum(Ess[10*(t-1)+1:10t])/10 for t in 1:10]
    #κ = (xx[5]-xx[2])*p.Ω/(2pi)/3
    κ = (xx[6]-xx[2])*p.Ω/(2pi)/4
    println("κ = $(κ)")
    #p1 = plot(Ess, linewidth=2.0)
    p2 = plot(xx, linewidth=2.0, marker=:circle)
    #for it in 2:sample
    #    p1 = plot!(Ess[it], linewidth=2.0)
    #end
    #savefig(p1, "./HeatingDynamics_ξ$(p.ξ)_$(day).png")
    savefig(p2, "./HeatingDynamics_ξ$(p.ξ)_ave_$(day).png")
    #println(ti)
    #println(κ)
    return κ
end

function calc_HR_βdep(ARG, β_it::Int)
    p = init_parm(ARG)
    s = init_system(p)

    ϵ1 = parse(Float32, ARG[8])
    ϵ2 = parse(Float32, ARG[9])
    β = parse(Float32, ARG[10])

    
    Udt = Matrix(I, p.Ms, p.Ms)
    for t in 1:p.t_step
        Udt = calc_Udt_orig(Udt, t, p, s)
        #=
        if(t%100==0)
            Udt = Udt/sqrt((Udt*Udt')[1,1])
        end=#
    end
    println((Udt*Udt')[1,1])
    Udt = Udt/sqrt((Udt*Udt')[1,1])
    
    thermoU = therm_U(β, s)
    
    
    ti = zeros(Int,2)
    sample = 10
    t_size = 100
    Es = zeros(Float32, sample, t_size)
    for it in 1:sample
        st = init_state(p, thermoU, β_it)
        #Es = []
        #ti = zeros(Int,2)
        for t in 1:t_size
            st = Udt*st
            st = st/sqrt(st'*st)
            E = calc_E(st, s)
            Es[it, t] = E
            #push!(Es, E)
            #println(E)
            if(E>ϵ1 && ti[1]==0)
                ti[1] = t
            end
            if(E>ϵ2 && ti[2]==0)
                ti[2] = t
                break
            end
            
        end
        #println("============")
        #push!(Ess, Es)
        #κ += (ϵ2-ϵ1)/((ti[2]-ti[1])*p.dt*p.t_step)/sample
    end
    Ess = [sum(Es[:,t])/sample for t in 1:t_size]
    xx = [sum(Ess[10*(t-1)+1:10t])/10 for t in 1:10]
    κ = (xx[5]-xx[2])*p.Ω/(2pi)/3
    println("κ = $(κ)")
    p1 = plot(Ess, linewidth=2.0)
    p2 = plot(xx, linewidth=2.0, marker=:circle)
    #for it in 2:sample
    #    p1 = plot!(Ess[it], linewidth=2.0)
    #end
    savefig(p1, "./HeatingDynamics_ξ$(p.ξ).png")
    savefig(p2, "./HeatingDynamics_ξ$(p.ξ)_ave.png")
    #println(ti)
    return κ
end

function main_ξ(ARG)
    κs = []
    ξs::Vector{Float32} = [0.1f0, 0.14f0, 0.2f0, 0.28f0, 0.4f0, 0.56f0,0.7f0, 0.8f0, 0.9f0, 1.0f0, 1.1f0, 1.2f0]
    for ξ in ξs
        κ = calc_HR_ξdep(ARG, ξ)
        push!(κs, κ)
    end
    println(κs)
    p0 = plot(ξs, κs, linewidth=2.0, marker=:circle, xscale=:log10)
    savefig(p0, "./HeatingRate_ξdep_$(day).png")
    p1 = plot(ξs, κs, linewidth=2.0, marker=:circle, xscale=:log10, yscale=:log10)
    savefig(p1, "./HeatingRate_ξdep_log_$(day).png")
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
    