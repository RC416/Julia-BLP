#=
Function to calculate the vector of relevant price elasticities.

The key step to solving the supply side is the vector of price elasticities Δ ***
Must solve for own price elasticities ∂σⱼ/∂pⱼ and cross price elastiticities ∂σⱼ/∂pₖ
Market share of product j in a given market is: 
σⱼ = ∫exp(pⱼα + Xβ + pⱼvᵢₖₚσᵛₚ + Σₖ xⱼₖvᵢₖσᵛₖ) / 1 + Σₖ exp(pₖα + Xβ + pₖvᵢₖₚσᵛₚ + Σₖ xⱼₖvᵢₖσᵛₖ) * f(νᵢ)dνᵢ

Simplify as following:
σⱼ = ∫ cⱼ / (1 + Σₖ cₖ ) f(vᵢ)dvᵢ
where cⱼ = exp(pⱼα + Xβ + pⱼvᵢₖₚσᵛₚ + Σₖ xⱼₖvᵢₖσᵛₖ)
and ∂cⱼ/∂pⱼ = (α + vᵢₖₚσᵛₚ)exp(pⱼα + Xβ + pⱼvᵢₖₚσᵛₚ + Σₖ xⱼₖvᵢₖσᵛₖ)

Let 𝒯ⱼ represent the interior of the integral for product j in a given market
Thus 𝒯ⱼ = cⱼ / (1 + Σₖ cₖ ) and σⱼ = ∫ 𝒯ⱼ f(vᵢ)vᵢ

Differentiate using 1) fundamental theorem of calculus (bring derivative inside integral) and
2) quotient rule (for the own price elasticity). 

Own price elasticity:
∂σⱼ/∂pⱼ = ∫(α + vᵢₖₚσᵛₚ) 𝒯ⱼ (1 - 𝒯ⱼ) f(vᵢ)dvᵢ

Cross price elasticity: 
∂σⱼ/∂pₘ = ∫ -(α + vᵢₖₚσᵛₚ) 𝒯ⱼ 𝒯ₘ f(vᵢ)dvᵢ

This module's function solves 1) the own price elasticities (diagnoal of Δ) for all 
products and 2) the cross price elasticities for all products marketed by the same 
company in the same market (all other elements of Δ).

***Solution to the FOC:
S - Δ(P-MC) = 0 

Where S and P are nx1 vetors of observed share and price, respectively.
Δ is a nxn matrix of cross price elastitities.
mc is a nx1 vector of marginal cost for the n products.

Example for company 1 that markets only product 1 and company 2 that markets product 2 and 3.
s₁ - [-∂σ₁/∂p₁     0         0    ] (p₁ - mc₁) = 0
s₂ - [   0     -∂σ₂/∂p₂  -∂σ₃/∂p₂ ] (p₂ - mc₂) = 0
s₂ - [   0     -∂σ₂/∂p₃  -∂σ₃/∂p₃ ] (p₃ - mc₃) = 0 
 ⋮ -  [   ⋮         ⋮          ⋮    ] (   ⋮    ) = ⋮

Assumes that firms set prices simultaneously to maximize static profits. 

note: large time save running loops in parallel. 
=#


# export function in module so that it can be used in main file
module supply_price_elasticities
export price_elasticities

# required modules 
using DataFrames        # for Not() and sample data
using LinearAlgebra     # basic math
using Statistics        # for mean()

function price_elasticities(θ₁, θ₂, X, s, v_diag, v_off_diag, market_id, firm_id)
#= 
θ₁: 6x1 vector of coefficients where the price coefficient is first
θ₂: 5x1 vector of random coefficients where the price coefficient is first
X : 2217x6 matrix of observables where price is the frist column
s : 2217x1 vector of observed market shares
v_diag : 5000x5 vector of pre-selected random draws from joint normal to simulate 5000 or 
    some large number of individuals. 
v_off_diag : 50x5 vector of pre-selected random draws from joint normal to simulate 50 or
    some number of individuals.
market_id: 2217x1 vector of market id for each product/observation  (cdid = market = year in this dataset)
firm_id: 2217x1 vector of firm id for each product.

for v_diag/v_off_diag, the diagonal terms are much larger so may want more precision by simulating a 
larger number of individuals. Off diagnoal terms are smaller and much more numerous, so can use
fewer individuals to reduce computation time.
=#

# get price coefficient
α = θ₁[1] 
# get price random coefficient
σᵛₚ = θ₂[1]

# get number of products in all markets
n_products = size(X,1)

# initialize empty matrix
Δ = zeros(n_products, n_products)

#=
Own price elasticity:
∂σⱼ/∂pⱼ = ∫(α + vᵢₖₚσᵛₚ) 𝒯ⱼ (1 - 𝒯ⱼ) f(vᵢ)vᵢ
Corresponds to the diagonal of Δ
=#

# X is a vector of observables for all products for all markets
# vᵢ is a vector of 5 random draws for a given individual
# j is a particular product
# recall there is no random coefficient for space (index 6 of X)
# note that there are about ~100 products per market

# loop through all products
Threads.@threads for j in 1:n_products # run loop in parallel with Threads. reduced time ~75x.

    # get market id for product j
    market = market_id[j]

    # get observables and indiviudals
    xⱼ = X[j,:]                    # observables for product j 
    xₘ = X[market_id.==market,:]   # observables of all products in market with product j
    vₘ = v_diag[market,:,:]        # matrix of 5000x5 pre-selected random draws (=> 5000 individuals)
   
    # function defining the interior of the sigma function integral 
    𝒯(vᵢ) = exp(xⱼ'θ₁ + xⱼ[Not(6)]'*(θ₂.*vᵢ)) / (1 + sum(exp.(xₘ*θ₁ + xₘ[:,Not(6)]*(θ₂.*vᵢ)))) 

    # interior of the own price elasticity function
    integral_interior(vᵢ) = (α + vᵢ[1]*σᵛₚ) * 𝒯(vᵢ) * (1 - 𝒯(vᵢ))

    # estimate with Monty Carlo integration over all individuals in vₘ
    # integral_interior() is applied to each of the ~5000 sets of 5 vᵢ values in vₘ
    ∂σⱼ_∂pⱼ = mean(integral_interior.(vₘ))
    # equivalently: ∂σⱼ_∂pⱼ = sum(integral_interior.(vₘ)) * 1 / length(vₘ)

    # assign own price elasticitiy to matrix of price elasticities (along the diagonal) 
    Δ[j,j] = -∂σⱼ_∂pⱼ

end


#=
Cross price elasticity: 
∂σⱼ/∂pₖ ∫ - (α + vᵢₖₚσᵛₚ) 𝒯ⱼ 𝒯ₖ f(vᵢ)vᵢ 
=#

# X is a vector of observables for all products for all markets
# vᵢ is a vector of 5 random draws for a given individual
# j and k are particular products
# recall there is no random coefficient for space (index 6 of X)
# note that there are about ~100 products per market

# loop through all columns (σ)
Threads.@threads for j in 1:n_products  # run loop in parallel with Threads. reduced time ~500x. 
    # loop through all rows (price)
    for k in 1:n_products

        # check that the row and column product are both marketed by the same company in the same market
        if (firm_id[j] == firm_id[k]) & (market_id[j] == market_id[k]) & (j != k)

            # get observables and indiviudals in the market x₁
            xⱼ = X[j,:]                         # observables for product j
            xₖ = X[k,:]                         # observables for product k 
            xₘ = X[market_id.==market_id[j],:]  # observables of all products in market with product j and k
            vₘ = v_off_diag[market_id[j],:,:]     # matrix of 50x5 pre-selected random draws (=> 50 individuals)

            # interior of sigma function integral for products j or k
            𝒯(xⱼ,vᵢ) = exp(xⱼ'θ₁ + xⱼ[Not(6)]'*(θ₂.*vᵢ)) / (1 + sum(exp.(xₘ*θ₁ + xₘ[:,Not(6)]*(θ₂.*vᵢ)))) 

            # interior of the own price elasticity function
            integral_interior(vᵢ) = (α + vᵢ[1]*σᵛₚ) * 𝒯(xⱼ,vᵢ) * 𝒯(xₖ,vᵢ)

            # estimate with Monty Carlo integration over all individuals in V
            # integral_interior() is applied to each of the sets of 5 vᵢ values in vₘ
            ∂σⱼ_∂pₖ = mean(integral_interior.(vₘ))
            # equivalently: ∂σⱼ_∂pₖ = sum(integral_interior.(vₘ)) * 1 / length(vₘ)
            
            # assign cross price elasticitiy to matrix of price elasticities 
            Δ[k,j] = -∂σⱼ_∂pₖ
        end
    end
end

# return the completed matrix of price elasticities
return Δ
end 


end # end module