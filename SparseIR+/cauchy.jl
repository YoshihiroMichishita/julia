using LinearAlgebra, Distributions, Random, StatsBase

struct CauchyParams
    K::Int
    μ::Vector{Float32}
    Σ::Vector{Float32}
    ϕ::Vector{Float32}
end

function init_rand_params(K::Int)
    μ = rand(Uniform(-1.0,1.0),K)
    Σ = exp.(rand(Uniform(-5.0,1.0),K))
    ϕ = softmax0(3.0*randn(K))
    return CauchyParams(K, μ, Σ, ϕ)
end

function cparams2data(p::CauchyParams)
    ord = sortperm(p.ϕ)
    return [p.μ[ord]..., log.(p.Σ[ord])..., p.ϕ[ord]...]
end

function softmax0(x::Vector{Float64})
    p1 = exp.(x)
    return p1/sum(p1)
end

function cparams2data(p::CauchyParams)
    ord = sortperm(p.ϕ)
    return [p.μ[ord]..., log.(p.Σ[ord])..., p.ϕ[ord]...]
end

function ccm(w::Float64, p::CauchyParams)
    y = sum([p.ϕ[i]*pdf(Cauchy(p.μ[i], p.Σ[i]), w) for i in 1:p.K])
    return y
end

function ccm_rho(ws::Vector{Float64}, p::CauchyParams)
    return [ccm(w, p) for w in ws]
end