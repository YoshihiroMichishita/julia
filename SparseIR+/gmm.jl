using LinearAlgebra, Distributions
#using Flux

mutable struct gParams
    K::Int
    μ::Vector{Float32}
    Σ::Vector{Float32}
    ϕ::Vector{Float32}
end

function init_params(K::Int)
    μ = rand(Uniform(-1.0,1.0),K)
    Σ = ones(K)
    ϕ = [1.0/K for i in 1:K]
    return gParams(K, μ, Σ, ϕ)
end

function softmax0(x::Vector{Float64})
    p1 = exp.(x)
    return p1/sum(p1)
end

function rand_init_params(K::Int)
    μ = rand(Uniform(-1.0,1.0),K)
    Σ = exp.(rand(Uniform(-4.0,2.0),K))
    ϕ = softmax0(rand(Uniform(-5.0,5.0),K))
    #[1.0/K for i in 1:K]
    return gParams(K, μ, Σ, ϕ)
end

function gparams2data(p::gParams)
    ord = sortperm(p.ϕ)
    return [p.μ[ord]..., log.(p.Σ[ord])..., p.ϕ[ord]...]
    #return [p.μ..., p.Σ..., p.ϕ...]
end

function gparams2data_μperm(p::gParams)
    ord = sortperm(p.μ)
    return [p.μ[ord]..., log.(p.Σ[ord])..., p.ϕ[ord]...]
    #return [p.μ..., p.Σ..., p.ϕ...]
end

function gparams2data_σperm(p::gParams)
    ord = sortperm(p.Σ)
    return [p.μ[ord]..., log.(p.Σ[ord])..., p.ϕ[ord]...]
    #return [p.μ..., p.Σ..., p.ϕ...]
end

function gparams2data_μσperm(p::gParams)
    ord = sortperm(p.μ./p.Σ)
    return [p.μ[ord]..., log.(p.Σ[ord])..., p.ϕ[ord]...]
    #return [p.μ..., p.Σ..., p.ϕ...]
end

function gparams2data_sym2(p::gParams)
    return [sum(p.μ), sum(log.(p.Σ)), sum(p.ϕ), p.μ[1]*p.μ[2], log(p.Σ[1])*log(p.Σ[2]), p.ϕ[1]*p.ϕ[2], p.μ[1]*log(p.Σ[2])+ p.μ[2]*log(p.Σ[1]), p.ϕ[1]*log(p.Σ[2])+ p.ϕ[2]*log(p.Σ[1]), p.μ[1]*p.ϕ[2]+ p.μ[2]*p.ϕ[1]]
end

function gparams2data_sym2_2(p::gParams)
    return [sum(p.μ), 0.1f0sum(log.(p.Σ)), sum(p.ϕ), p.μ[1]*p.μ[2], 0.01f0log(p.Σ[1])*log(p.Σ[2]), p.ϕ[1]*p.ϕ[2], 0.1f0p.μ[1]*log(p.Σ[2])+ 0.1f0p.μ[2]*log(p.Σ[1]), 0.1f0p.ϕ[1]*log(p.Σ[2])+ 0.1f0p.ϕ[2]*log(p.Σ[1]), p.μ[1]*p.ϕ[2]+ p.μ[2]*p.ϕ[1]]
end

using SymPy
@syms x

function create_perms(n::Int)
    perms = []
    init = [1, 1, 1]
    for i in 1:n^3
        push!(perms, copy(init))
        init[end] += 1
        for j in 3:-1:2
            if init[j] > n
                init[j] = 1
                init[j-1] += 1
            end
        end
    end
    return perms
end

function mod3(x::Int)
    t = x % 3
    if t == 0
        return 3
    else
        return t
    end
end

function sym2params(out::Vector{Float32})
    xs = []
    for i in 1:3
        f(x) = x^2 - out[i]*x + out[i+3]
        ans = convert(Vector{Float32}, solve(f(x), x))
        push!(xs, ans)
    end
    scores = Float32[]
    for perm in create_perms(2)[1:4]
        score = 0.0
        x0 = [xs[1][perm[1]], xs[2][perm[2]], xs[3][perm[3]]]
        y0 = [xs[1][3-perm[1]], xs[2][3-perm[2]], xs[3][3-perm[3]]]
        for i in 1:3
            score += (x0[mod3(i)]*y0[mod3(i+1)] + x0[mod3(i+1)]*y0[mod3(i)] - out[6+i])^2
        end
        push!(scores, score)
    end
    id = argmin(scores)
    ans_perm = create_perms(2)[id]
    return [xs[1][ans_perm[1]], xs[1][3-ans_perm[1]], xs[2][ans_perm[2]], xs[2][3-ans_perm[2]], xs[3][ans_perm[3]], xs[3][3-ans_perm[3]]]
end

function sym2params2(out::Vector{Float32})
    xs = []
    for i in 1:3
        f(x) = x^2 - out[i]*x + out[i+3]
        ans = convert(Vector{Float32}, solve(f(x), x))
        push!(xs, ans)
    end
    scores = Float32[]
    for perm in create_perms(2)[1:4]
        score = 0.0
        x0 = [xs[1][perm[1]], xs[2][perm[2]], xs[3][perm[3]]]
        y0 = [xs[1][3-perm[1]], xs[2][3-perm[2]], xs[3][3-perm[3]]]
        for i in 1:3
            score += (x0[mod3(i)]*y0[mod3(i+1)] + x0[mod3(i+1)]*y0[mod3(i)] - out[6+i])^2
        end
        push!(scores, score)
    end
    id = argmin(scores)
    ans_perm = create_perms(2)[id]
    return [xs[1][ans_perm[1]], xs[1][3-ans_perm[1]], exp(xs[2][ans_perm[2]]), exp(xs[2][3-ans_perm[2]]), xs[3][ans_perm[3]], xs[3][3-ans_perm[3]]]
end

function sym2params2_2(out::Vector{Float32})
    xs = []
    for i in 1:3
        f(x) = x^2 - out[i]*x + out[i+3]
        #ans = convert(Vector{Float32}, solve(f(x), x))
        ans = real.(convert(Vector{ComplexF32}, solve(f(x), x)))
        push!(xs, ans)
    end
    scores = Float32[]
    for perm in create_perms(2)[1:4]
        score = 0.0
        x0 = [xs[1][perm[1]], xs[2][perm[2]], xs[3][perm[3]]]
        y0 = [xs[1][3-perm[1]], xs[2][3-perm[2]], xs[3][3-perm[3]]]
        for i in 1:3
            score += (x0[mod3(i)]*y0[mod3(i+1)] + x0[mod3(i+1)]*y0[mod3(i)] - out[6+i])^2
        end
        push!(scores, score)
    end
    id = argmin(scores)
    ans_perm = create_perms(2)[id]
    return [xs[1][ans_perm[1]], xs[1][3-ans_perm[1]], exp(10xs[2][ans_perm[2]]), exp(10xs[2][3-ans_perm[2]]), xs[3][ans_perm[3]], xs[3][3-ans_perm[3]]]
end

function gmm1(w::Float32, p::gParams)
    #μ, Σ, ϕ = p.μ, p.Σ, p.ϕ
    y = sum([p.ϕ[i]*pdf(Normal(p.μ[i], p.Σ[i]), w) for i in 1:p.K])
    return y
end

function gmm1(w::Float64, p::gParams)
    #μ, Σ, ϕ = p.μ, p.Σ, p.ϕ
    y = sum([p.ϕ[i]*pdf(Normal(p.μ[i], p.Σ[i]), w) for i in 1:p.K])
    return y
end

function gmm_rho(ws::Vector{Float64}, p::gParams)
    return [gmm1(w, p) for w in ws]
end

function logpws(ws::Vector{Float32}, true_ρ::Vector{Float32}, p::gParams)
    dw = ws[2]-ws[1]
    return dw*(true_ρ'*[log(gmm1(w, p)) for w in ws])
end

function D_kl(ws::Vector{Float32}, true_ρ::Vector{Float32}, p::gParams)
    f(x) = gmm1(x, p)
    K = length(ws)
    dw = ws[2]-ws[1]
    return sum([dw*true_ρ[i]*(log(true_ρ[i]+1f-8)-log(f(ws[i])+1f-8)) for i in 1:K])
end

function p_now(ws::Vector{Float32}, p::gParams)
    f(x) = gmm1(x, p)
    return Float32[f(w) for w in ws]
end

#EM algorithm
function E_step(ws::Vector{Float32}, p::gParams)
    q = zeros(Float32, size(ws)[1], p.K)
    for (i,w) in enumerate(ws)
        for k in 1:p.K
            q[i,k] = p.ϕ[k]*pdf(Normal(p.μ[k], p.Σ[k]), w)
        end
        q[i,:] = q[i,:]/(sum(q[i,:])+1f-8)
    end
    return q
end

function M_step!(ws::Vector{Float32}, true_ρ::Vector{Float32}, q::Matrix{Float32}, params::gParams)
    Nd = size(ws)[1]
    dw = ws[2] - ws[1]
    for k in 1:params.K
        qsum = true_ρ'*q[:,k]
        params.ϕ[k] = dw*qsum
        params.μ[k] = sum(true_ρ .* q[:,k] .* ws)/qsum
        #params.Σ[k] = sum(true_ρ .* q[:,k] .* (ws .- params.μ[k]) .* (ws .- params.μ[k]))/qsum
        params.Σ[k] = sqrt(sum([true_ρ[i] * q[i,k] * (ws[i] - params.μ[k]) * (ws[i] - params.μ[k]) for i in 1:Nd])/qsum)
        #params.Σ[k] = params.Σ[k]/qsum
    end
    params.ϕ = params.ϕ/sum(params.ϕ)
end

function EM(dkl_max::Float32, X::Int, ws::Vector{Float32}, true_ρ::Vector{Float32})
    #X = 10
    itr_max = 2000
    th = 1e-6
    gp = init_params(X)
    old_logpx = logpws(ws, true_ρ, gp)

    for i in 1:itr_max
        qs = E_step(ws, gp)
        #println(qs)
        M_step!(ws, true_ρ, qs, gp)
        new_logpx = logpws(ws, true_ρ, gp)
        dkl = D_kl(ws, true_ρ, gp)
        if (abs(new_logpx - old_logpx) < th && dkl < dkl_max)
            #dkl = D_kl(ws, true_ρ, gp)
            println("itr: ", i, " logpx: ", new_logpx, " converged!")
            println("D_kl: ", dkl)
            break
        end
        old_logpx = new_logpx
    end
    return gp
end