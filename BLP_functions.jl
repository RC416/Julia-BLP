#= The two essential BLP functions =#
module BLP_functions
export demand_objective_function

using DataFrames        # for Not() and sample data
using LinearAlgebra     # basic math
using Statistics        # for mean()

#= Demand Objective Function -----------------------------------------------------------
Performs the key steps for BLP demand estimation

Key steps:
1. given θ₂, solve for δ using contraction mapping
2. using δ, calculate ξⱼ(θ) = δ-xⱼβ
3. set up GMM moments: E[Z*ξ(θ)] = G(θ) = 0 and construct GMM function Q(θ) = G(θ)'*W*G(θ)
4. return GMM function value

Find the θ that minimizes objective_function(θ)

Step 1. requires calculating predicted market share σ(δ,θ) using numerical integration.
This is done with the second function here, σ() (below demand_objective_function).

Inputs:

θ₂: 5x1 vector of σᵛ coefficients (for all variables except space). Space random coefficient not estimated to aid estimation. 
X:  2217x6 matrix of observables (including price)
s:  2217x1 vector of product market shares
Z:  2217x15 vector of BLP instruments
v:  2217x250 vector of random draws from joint normal
market_id: 2217x1 vector of market id for each product/observation (cdid, market = years in this dataset)
Used to track which products are in which markets since markets are solved individually.

Does not use θ₁ as an input. Rather, backs out θ₁ from θ₂ in the step 2.
This allows for optimization over only the θ₂ coefficients (5) without including θ₁ (6 others).
=#

function demand_objective_function(θ₂,X,s,Z,v,market_id)

# 1. solve for delta
# initialize vector to hold all market deltas
δ = zeros(length(s))

# get id of each market
markets = unique(market_id)

# solve for delta in each market
Threads.@threads for market in markets # run in parallel with Threads

    # get observables and pre-selected random draws for given market
    xₘ = X[market_id.==market, Not(6) ]        # observed features. excluding "space" the same way as matlab code to aid estimation of random effects.
    sₘ = s[market_id.==market,:]               # market share 
    #vₘ = v[market,:]                      # vector of 250 pre-selected random draws (=> 50 simulated individuals)
    vₘ = v[market,:,:]                         # 50x5 vector of 5 pre-selected random draws for 50 simulated individuals

    n_products = length(sₘ)               # number of products in given market
    δₘ = zeros(n_products)                # geuss value of delta from main vector

    # define the contraction mapping in a function. Note δ, s and σ are vectors.
    # σ is a custom function defined below. It estimates market shares through numerical integration.
    Φ(δ) = δ + log.(sₘ) - log.(σ(δ,θ₂,xₘ,vₘ))

    # contraction mapping
    tolerance = 1e-4                      # Matlab code uses 1e-6 or 1e-9
    largest_dif = tolerance + 1           # track difference from first value. initialze at a value above tolerance
    max_iterations = 1000                 # reasonable limit 
    counter = 0                           # track iterations 

    while (largest_dif > tolerance)
        # recalculate delta
        δₘ = Φ(δₘ)
        # check if all estimates (elements) are within tolerance
        largest_dif = maximum(abs.( δₘ - Φ(δₘ) ))
        # check if max iterations is exceeded
        counter += 1
        if counter == max_iterations
            break
        end
    end

    # append this market's deltas to the vector of all deltas
    δ[market_id.==market] = δₘ
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




#= Market Share Estimation Function -----------------------------------------------------------
Calculates the market share of each product given delta and θ₂.

Key steps:
1. Select sets of 5 draws from the standard joint normal (V) (one for each θ₂)
2. Define a function to calculate the vector of μⱼᵢ values
3. Define a function for the interior of the key integral
4. Solve the integral numerically with Monty Carlo integration
5. Repeat for every product in the market.

Returns vector of predicted market shares for every product in the market. 

Inputs:
δ: nx1 vector. represents xβ + ξ. One for each product in given market
θ₂: 5x1 vector of σᵛ coefficients. One for each x variable
X: nx5 matrix of observables for all n products in a market. Drops random coefficient for "space" to aid estimation.
v: 50x5 vector of random draws from joint normal mean 0 variance 1. 
    One random draw per each of the 5 θ₂ coefficents per person. =#

function σ(δ,θ₂,X,v)

# get number of products in the given market
n_products = length(δ) 
# initialize predicted market share vector
σ = zeros(n_products) 
# get number of simulated individuals 
n_individuals = size(v,1)

# calculate μⱼᵢ values for each individual and each product
μ = X * (v .* θ₂')'     # = μⱼᵢ = ∑ₖ xⱼₖ * vₖᵢ * σₖ      where σₖ is one of the θ₂ coefficients

# estimate market share for each product j in the market.
for j in 1:n_products

    # function for the interior of the integral: σ = ∫exp(δⱼ+μᵢⱼ) / 1 + Σₖ exp(δₖ + μᵢₖ) * f(νᵢ)dνᵢ
    integral_interior(i) = exp(δ[j] + μ[j,i]) / (1 + sum(exp.(δ + μ[:,i])))

    # calculation of new σⱼ using Monty Carlo integration 
    # apply integral_interior to vector of individual indices 
    # σ[j] = sum(integral_interior.(1:n_individuals)) * 1 / length(1:n_individuals)
    σ[j] = mean(integral_interior.(1:n_individuals))
end

# return estimated market shares
return σ
end



end # end module 