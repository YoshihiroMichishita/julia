using LinearAlgebra
using Flux
using JLD2
using BSON
using Plots

tanh5(x) = 5tanh(x)
tanh6(x) = 4tanh(x)
tanh8(x) = 8tanh(x)

###############################################
#### Model Initialization
###############################################

function init_model_sp(n_l::Int, n_gauss::Int, width::Int, depth::Int)
    model = Chain(Dense(n_l, width), LayerNorm(width), (Chain(Dense(width, width, softplus), LayerNorm(width)) for i in 1:depth)..., Flux.Parallel(vcat, Dense(width, n_gauss, tanh), Dense(width, n_gauss, tanh6), Dense(width, n_gauss, tanh5)))
    return model
end

function init_model_sp2(n_l::Int, n_gauss::Int, width::Int, depth::Int)
    model = Chain(Dense(n_l, width), LayerNorm(width), (Chain(Dense(width, width), LayerNorm(width), softplus) for i in 1:depth)..., Flux.Parallel(vcat, Dense(width, n_gauss, tanh), Dense(width, n_gauss, tanh6), Dense(width, n_gauss, tanh5)))
    return model
end

function init_model_tanh(n_l::Int, n_gauss::Int, width::Int, depth::Int)
    model = Chain(Dense(n_l, width), LayerNorm(width), (Chain(Dense(width, width, tanh), LayerNorm(width)) for i in 1:depth)..., Flux.Parallel(vcat, Dense(width, n_gauss, tanh), Dense(width, n_gauss, tanh6), Dense(width, n_gauss, tanh5)))
    return model
end

function init_model_tanh_symGP(n_l::Int, n_gauss::Int, width::Int, depth::Int)
    model = Chain(Dense(n_l, width), LayerNorm(width), (Chain(Dense(width, width, tanh), LayerNorm(width)) for i in 1:depth)..., Flux.Parallel(vcat, (Chain(Dense(width, div(width,8), tanh), Dense(div(width,8), div(width,8), tanh), Dense(div(width,8), div(width,8), tanh), Dense(div(width,8), 3, tanh)) for i in 1:n_gauss)... ))
    return model
end

function init_model_sp_symGP(n_l::Int, n_gauss::Int, width::Int, depth::Int)
    model = Chain(Dense(n_l, width), LayerNorm(width), (Chain(Dense(width, width, softplus), LayerNorm(width)) for i in 1:depth)..., Flux.Parallel(vcat, (Chain(Dense(width, div(width,8), tanh), Dense(div(width,8), div(width,8), tanh), Dense(div(width,8), div(width,8), tanh), Dense(div(width,8), 3, tanh)) for i in 1:n_gauss)... ))
    return model
end

function init_model_sym(n_l::Int, n_gauss::Int, width::Int, depth::Int)
    model = Chain(Dense(n_l, width), LayerNorm(width), (Chain(Dense(width, width, softplus), LayerNorm(width)) for i in 1:depth)..., Dense(width, 9))
    return model
end

function init_model_sym2(n_l::Int, n_gauss::Int, width::Int, depth::Int)
    model = Chain(Dense(n_l, width), LayerNorm(width), (Chain(Dense(width, width, tanh), LayerNorm(width)) for i in 1:depth)..., Dense(width, 9))
    return model
end



###################################################
###  Loss function
###################################################

function loss(model::Chain, K::Int, in, ans, λ::Float32)
    out = cpu(model(cu(in)))
    l = size(out)[2]
    phi = softmax(out[2K+1:3K,:])
    mu = out[1:K,:]
    logs = out[K+1:2K,:]
    sigma = exp.(logs)
    logts = ans[K+1:2K, :]
    true_sigma = exp.(logts)
    loss = sum(phi.*(((mu .- ans[1:K,:])./ sigma).^2 + (true_sigma./sigma).^2 - 2logts + 2logs)) + λ * sum(phi.*log.(phi ./ (ans[2K+1:3K, :] .+ 1f-10)))
    return loss/l - 1.0f0
end

function loss2(model::Chain, K::Int, in, ans, λ::Float32)
    out = cpu(model(cu(in)))
    l = size(out)[2]
    phi = softmax(out[2K+1:3K,:])
    true_phi = ans[2K+1:3K, :]
    mu = out[1:K,:]
    sigma = out[K+1:2K,:]
    t_sigma = ans[K+1:2K, :]
    loss = sum(true_phi .* ((mu .- ans[1:K,:]).^2 + (sigma .- t_sigma).^2)) + λ * sum(true_phi.*log.(true_phi ./ (phi .+ 1f-10)))
    return loss/l
end

function loss2_1(model::Chain, K::Int, in, ans, λ::Float32)
    out = cpu(model(cu(in)))
    l = size(out)[2]
    phi = softmax(out[2K+1:3K,:])
    true_phi = ans[2K+1:3K, :]
    mu = out[1:K,:]
    sigma = out[K+1:2K,:]
    t_sigma = ans[K+1:2K, :]
    loss = sum(phi .* ((mu .- ans[1:K,:]).^2 + (sigma .- t_sigma).^2)) + λ * sum(phi.*log.(phi ./ (true_phi .+ 1f-10)))
    return loss/l
end

function loss2_sort(K::Int, λ::Float32, μ::Vector{Float32}, μ_ans::Vector{Float32},σ::Vector{Float32}, σ_ans::Vector{Float32},ϕ::Vector{Float32}, ϕ_ans::Vector{Float32})
    l0 = 0.0f0
    perm = sortperm(ϕ; rev=true)
    for j in 1:K
        i = perm[j]
        l0 += ϕ_ans[j]*(((μ[i]-μ_ans[j]))^2 + (σ_ans[j] - σ[i])^2 + λ*log(ϕ_ans[j]/(ϕ[i]+1f-10)))
    end
    return l0
end

function loss2_symGP(model::Chain, K::Int, in, ans, λ::Float32)
    out = cpu(model(cu(in)))
    l = size(out)[2]
    phi = softmax(5out[3:3:3K,:])
    true_phi = ans[2K+1:3K, :]
    mu = out[1:3:3K,:]
    true_mu = ans[1:K,:]
    sigma = 6out[2:3:3K,:]
    true_sigma = ans[K+1:2K, :]
    loss = 0.0f0
    for i in l
        loss += loss2_sort(K, λ, mu[:,i], true_mu[:,i], sigma[:,i], true_sigma[:,i], phi[:,i], true_phi[:,i])
    end
    return loss/l
end

function loss2_symGP2(model::Chain, K::Int, in, ans, λ::Float32)
    out = cpu(model(cu(in)))
    #l = size(out)[2]
    phi = softmax(5out[3:3:3K,:])
    true_phi = ans[2K+1:3K, :]
    mu = out[1:3:3K,:]
    true_mu = ans[1:K,:]
    sigma = 6out[2:3:3K,:]
    true_sigma = ans[K+1:2K, :]
    loss = mean(phi .* ((mu .- true_mu).^2 + (sigma .- true_sigma).^2)) + λ * sum(phi.*log.(phi ./ (true_phi .+ 1f-10)))
    return loss
end

function loss2_2(model::Chain, K::Int, in, ans, λ::Float32)
    out = cpu(model(cu(in)))
    l = size(out)[2]
    phi = softmax(out[2K+1:3K,:])
    true_phi = ans[2K+1:3K, :]
    mu = out[1:K,:]
    sigma = out[K+1:2K,:]
    t_sigma = ans[K+1:2K, :]
    loss = sum(true_phi .* ((mu .- ans[1:K,:]).^2 ./ exp.(sigma./2) + (sigma .- t_sigma).^2)) + λ * sum(phi.*log.(phi ./ (true_phi .+ 1f-10)))
    return loss/l
end

function loss4(model::Chain, K::Int, in, ans, λ::Float32)
    out = cpu(model(cu(in)))
    l = size(out)[2]
    phi = softmax(out[2K+1:3K,:])
    mu = out[1:K,:]
    logs = out[K+1:2K,:]
    sigma = exp.(logs)
    logts = ans[K+1:2K, :]
    true_sigma = exp.(logts)
    loss = sum(phi.*(((mu .- ans[1:K,:])./ sigma).^2 + (true_sigma./sigma - sigma./true_sigma).^2)) + λ * sum(phi.*log.(phi ./ (ans[2K+1:3K, :] .+ 1f-10)))
    return loss/l
end

function gmm1(K::Int, w::Float32, μ::Vector{Float32}, Σ::Vector{Float32}, ϕ::Vector{Float32})
    #μ, Σ, ϕ = p.μ, p.Σ, p.ϕ
    y = 0.0f0
    for i in 1:K
        y += ϕ[i]*pdf(Normal(μ[i], Σ[i]), w)
    end 
    return y
end

function dkl_p(ws::Vector{Float32}, mu::Vector{Float32}, sigma::Vector{Float32}, phi::Vector{Float32}, true_mu::Vector{Float32}, true_sigma::Vector{Float32}, true_phi::Vector{Float32})
    K = length(mu)
    dw = ws[2]-ws[1]
    dkl= 0.0f0
    for w in ws
        p = gmm1(K,w,mu,sigma,phi)
        q = gmm1(K,w,true_mu,true_sigma,true_phi)
        dkl += dw * p * log(p/(q+1f-10))
    end
    return dkl
end


function loss_KL(model::Chain, K::Int, in, ans, ws::Vector{Float32})
    out = cpu(model(cu(in)))
    l = size(out)[2]
    phi = softmax(out[2K+1:3K,:])
    true_phi = ans[2K+1:3K,:]
    mu = out[1:K,:]
    true_mu = ans[1:K,:]
    logs = out[K+1:2K,:]
    sigma = exp.(logs)
    logts = ans[K+1:2K, :]
    true_sigma = exp.(logts)
    loss = 0.0f0
    for i in 1:l
        loss += dkl_p(ws, mu[:,i], sigma[:,i], phi[:,i], true_mu[:,i], true_sigma[:,i], true_phi[:,i])/l
    end
    return loss
end


function loss_sort(K::Int, λ::Float32, μ::Vector{Float32}, μ_ans::Vector{Float32},σ::Vector{Float32}, σ_ans::Vector{Float32},ϕ::Vector{Float32}, ϕ_ans::Vector{Float32})
    l0 = 0.0f0
    perm = sortperm(ϕ; rev=true)
    for j in 1:K
        i = perm[j]
        l0 += ϕ[i]*(((μ[i]-μ_ans[j])/σ[i])^2 + (σ_ans[j]/σ[i] - σ[i]/σ_ans[j])^2 + λ*log(ϕ[i]/(ϕ_ans[j]+1f-10)))
    end
    return l0
end

function loss4_symGP3(model::Chain, K::Int, in, ans, λ::Float32)
    out = cpu(model(cu(in)))
    l = size(out)[2]
    phi = softmax(out[2K+1:3K,:])
    true_phi = ans[2K+1:3K, :]
    mu = out[1:K,:]
    true_mu = ans[1:K,:]
    logs = out[2K+1:3K,:]
    sigma = exp.(logs)
    logts = ans[K+1:2K, :]
    true_sigma = exp.(logts)
    loss = 0.0f0
    for i in l
        loss += loss_sort(K, λ, mu[:,i], true_mu[:,i], sigma[:,i], true_sigma[:,i], phi[:,i], true_phi[:,i])/l
    end
    return loss
end


function loss4_symGP(model::Chain, K::Int, in, ans, λ::Float32)
    out = cpu(model(cu(in)))
    l = size(out)[2]
    phi = softmax(5out[3:3:3K,:])
    true_phi = ans[2K+1:3K, :]
    mu = out[1:3:3K,:]
    true_mu = ans[1:K,:]
    logs = 6out[2:3:3K,:]
    sigma = exp.(logs)
    logts = ans[K+1:2K, :]
    true_sigma = exp.(logts)
    loss = 0.0f0
    for i in l
        loss += loss_sort(K, λ, mu[:,i], true_mu[:,i], sigma[:,i], true_sigma[:,i], phi[:,i], true_phi[:,i])
    end
    return loss/l
end

function loss4_symGP2(model::Chain, K::Int, in, ans, λ::Float32)
    out = cpu(model(cu(in)))
    l = size(out)[2]
    phi = softmax(5out[3:3:3K,:])
    true_phi = ans[2K+1:3K, :]
    mu = out[1:3:3K,:]
    true_mu = ans[1:K,:]
    logs = 6out[2:3:3K,:]
    sigma = exp.(logs)
    logts = ans[K+1:2K, :]
    true_sigma = exp.(logts)
    loss = mean(phi.*(((mu .- true_mu[1:K,:])./ sigma).^2 + (true_sigma./sigma - sigma./true_sigma).^2)) + λ * sum(phi.*log.(phi ./ (true_phi .+ 1f-10)))
    return loss
end


perms(l) = isempty(l) ? [l] : [[x; y] for x in l for y in perms(setdiff(l, x))]


function loss_perm(perm::Vector{Int}, λ::Float32, μ::Vector{Float32}, μ_ans::Vector{Float32},σ::Vector{Float32}, σ_ans::Vector{Float32},ϕ::Vector{Float32}, ϕ_ans::Vector{Float32})
    l0 = 0.0f0
    for i in 1:K
        j = perm[i]
        l0 += ϕ[i]*(((μ[i]-μ_ans[j])/σ[i])^2 + (σ_ans[j]/σ[i] - σ[i]/σ_ans[j])^2 + λ*log(ϕ[i]/(ϕ_ans[j]+1f-10)))
    end
    return l0
end

function loss_min(K::Int, λ::Float32, μ::Vector{Float32}, μ_ans::Vector{Float32},σ::Vector{Float32}, σ_ans::Vector{Float32},ϕ::Vector{Float32}, ϕ_ans::Vector{Float32})
    ls = 1f5
    for perm in permsv
        l0 = loss_perm(perm, λ, μ, μ_ans, σ, σ_ans, ϕ, ϕ_ans)
        if(ls>l0)
            ls = l0
        end
    end 
    return ls
end

function loss4_sym(model::Chain, K::Int, in, ans, λ::Float32)
    out = cpu(model(cu(in)))
    l = size(out)[2]
    phi = softmax(out[2K+1:3K,:])
    true_phi = ans[2K+1:3K, :]
    mu = out[1:K,:]
    true_mu = ans[1:K,:]
    logs = out[K+1:2K,:]
    sigma = exp.(logs)
    logts = ans[K+1:2K, :]
    true_sigma = exp.(logts)
    loss = 0.0f0
    for i in l
        loss += loss_min(K, λ, mu[:,i], true_mu[:,i], sigma[:,i], true_sigma[:,i], phi[:,i], true_phi[:,i])
    end
    return loss/l
end



########################################################
### 