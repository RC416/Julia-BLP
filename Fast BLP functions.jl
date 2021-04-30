
module BLP_functions
export demand_objective_function

using DataFrames        # for Not() and sample data
using LinearAlgebra     # basic math
using Statistics        # for mean()



function fast_demand_objective_function(θ₂,X,s,Z,v,market_id)

    # initialze δ 
    δ = zeros(size(s))

# 1. Contraction mapping to find δ
# set up contraction mapping for δ
Φ(δ) = δ + log.(s) - log.(σ(δ,θ₂,X,v,market_id))

# contraction mapping parameters
tolerance = 1e-6                      # Matlab code uses 1e-6 or 1e-9
largest_dif = tolerance + 1           # track difference from first value. initialze at a value above tolerance
max_iterations = 1000                 # reasonable limit 
counter = 0                           # track iterations 

while (largest_dif > tolerance)
    # recalculate delta
    δ = Φ(δ)
    # check if all estimates (elements) are within tolerance
    largest_dif = maximum(abs.( δ - Φ(δ) ))
    # check if max iterations is exceeded
    counter += 1
    if counter == max_iterations
        break
    end
end

# back out the θ₁ value implied by delta. Note "space" not used in X. Lines 22-25 of GMM object in Matlab code.
# we have δ = Xθ₁' 
# without instruments: θ₁ = (X'X)⁻¹X'δ                    (OLS)
# with instruments: ̂X = Z (Z'Z)⁻¹Z'X   and θ₁ = (̂X'̂X)⁻¹̂Xδ   (2SLS)
# sub in for 2SLS: θ₁ = (̂X'̂X)⁻¹̂Xδ = ([(Z'Z)⁻¹Z'X]'[(Z'Z)⁻¹Z'X])⁻¹[(Z'Z)⁻¹Z'X]*δ
# θ₁ = inv((Z*inv(Z'Z)*Z'X)'*(Z*inv(Z'Z)*Z'X))*(Z*inv(Z'Z)*Z'X)'*δ
# which reduces to the expression:
θ₁ = inv((X'Z)*inv(Z'Z)*(X'Z)') * (X'Z)*inv(Z'Z)*Z'δ


# 2. using δ, calculate unobserved demand ξⱼ(θ) = δ-xⱼβ for all markets
ξ = δ - X*θ₁


# 3. construct GMM objective function
# recall GMM momement condition: E[h(Z)ξ(θ)] = 0
# where h(Z) = Z' (vector of instruments)
#       ξ(θ) = δⱼ(θ) - xⱼβ
# GMM objective fuction:
# Q(θ) = G(θ)'*W*G(θ)

# weighting matrix. 
W = inv(Z'Z) # Z'Z is optimal if ξ(θ) term is i.i.d. (normally the error term)
# GMM objective function value
Q = (Z'ξ)' * W * (Z'ξ)

# save global value of important variables so that the gradient function can access them
#global ξ_global = ξ
#global δ_global = X*θ₁


# 4. return objective function value and other useful values.
return Q, θ₁, ξ
end























#= Fast Market Share Estimation Function -----------------------------------------------------------
Estimates the market share of each product given delta and θ₂.
Used to solve the contraction mapping of the objective function to find the correct δ.

Market share of product j in a given market is: 
σⱼ = ∫exp(pⱼα + Xβ + pⱼvᵢₖₚσᵛₚ + Σₖ xⱼₖvᵢₖσᵛₖ) / 1 + Σₖ exp(pₖα + Xβ + pₖvᵢₖₚσᵛₚ + Σₖ xⱼₖvᵢₖσᵛₖ) * f(νᵢ)dνᵢ
where δ = pⱼα + Xβ   and   μ = pⱼvᵢₖₚσᵛₚ + Σₖ xⱼₖvᵢₖσᵛₖ = X*(v*θ₂')'
therefore: σⱼ = ∫exp(δⱼ + μⱼᵢ) / 1 + Σₖ exp(δₖ + μₖᵢ) * f(νᵢ)dνᵢ

Key steps:
1. Calculate μⱼᵢ for each individual for each product within each market (2217x50)
2. Using given δ and μ, calculate numerator of σⱼ: exp(δⱼ + μⱼᵢ)
3. Calculate the sum term in the denominator. Build a matrix 2217x50 of
demonominator terms for each individual within a given market.
4. Use divide numerator from 2. by denominator from 3. to get individual shares
for each product across the 50 individuals.
5. Calculate the mean share across the 50 individuals (equivalent to Monty Carlo integration).

Returns vector of predicted market shares for every product in the market. 

Considerably faster than the explicit version of the function since most calculations
    are done on vectors/matrices (3.2 ms versus 112 ms).

Inputs:
δ:  nx1 vector. represents xβ + ξ. One for each product in given market
θ₂: 5x1 vector of σᵛ coefficients. One for each x variable
X:  nx5 matrix of observables for all n products in a market. Drops random coefficient for "space" to aid estimation.
v:  50x5 vector of random draws from joint normal mean 0 variance 1. 
    One random draw per each of the 5 θ₂ coefficents per person.
market_id: 2217x1 vector of market id for each product/observation (cdid, market = years in this dataset)
=#

function σ(δ, θ₂, X, v, market_id)

    # get number of individuals and products
    n_individuals = size(v,2)
    n_products = size(X,1)

    # get delta for each individual (identical): 2217x50
    δ = repeat(δ,1,50) # repeats δ for each of the 50 individuals

    # calculate μⱼᵢ for each product for each individual: 2217x50
    μ = zeros(n_products, n_individuals)
    for market in unique(market_id)
        # μⱼᵢ = ∑ₖ xⱼₖ * vₖᵢ * σₖ   where σₖ is one of the θ₂ coefficients  
        #μ[market_id.==market,:] = X[market_id.==market,Not(6)] * (v[:,:] .* θ₂')' 
        μ[market_id.==market,:] = X[market_id.==market,Not(6)] * (v[market,:,:] .* θ₂')' 
    end

    # the numerator is easily calculated as: exp(δ+μ)

    # for the denominator: calculate the denominator term for each individual in each market: 2217x50
    ∑ₖexp = zeros(size(μ))
    # for each market
    for market in unique(market_id)
        # get the sequence of denominator terms for each individual
        denom_sequence = exp.(δ[market_id.==market,:] + μ[market_id.==market,:])
        # sum over all products in market for each individual
        market_denominator = sum(denom_sequence, dims=1)
        # assign to each row for given individual in given market
        ∑ₖexp[market_id.==market,:] = repeat(market_denominator, sum(market_id.==market))
    end

    # calculate market share for each product for each individual
    individual_shares = exp.(δ+μ) ./ (1 .+ ∑ₖexp)
    σ = mean(individual_shares, dims=2) # average across individuals (Monty Carlo integration)

    # return vector of estimated market shares: 2217x1
    return σ
end

@btime σ(δ, θ₂, X, v_50, market_id)

@btime fs1(δ, θ₂, v_50, X, market_id) # 3.2 ms (35x faster)
@btime old_sig(δ, θ₂, v_50, X, market_id) # 112 ms