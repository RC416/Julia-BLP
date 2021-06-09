module demand_derivatives
export gradient

using LinearAlgebra     # basic math
using Statistics        # for mean()

#= Gradient of BLP objective function 

recall that the objective function returns Q = [Z'ξ]'W[Z'ξ]

Q(ξ(θ₂)) = [Z'ξ(θ₂)]'W[Z'ξ(θ₂)]

Gradient: take derivative of Q with respect to the 5 coefficients in θ₂
∂Q/∂θ₂ = ∂Q/∂ξ * ∂ξ/∂θ₂

∂Q/∂ξ is the derivative of the quadratic form x'Ax: ∂x'Ax/∂x = x'[A + A'] = 2x'A   (if A is symmetric)
=> ∂Q/∂ξ = 2[Z'ξ]'W 
Which is a 1x2217 matrix

∂ξ/∂θ₂ is complicated. The final solution is:
∂ξ/∂θ₂ = (I-M)-[∂σ/∂δ]⁻¹[∂σ/∂θ₂] ≈ -[∂σ/∂δ]⁻¹[∂σ/∂θ₂]    which is a 2217x5 matrix
    where M = X[(Z(Z'Z)⁻¹Z'X)'(Z(Z'Z)⁻¹Z'X)]⁻¹(Z(Z'Z)⁻¹Z'X)  (2SLS projection matrix)


Derivation (steps i-iv): 
Start with expression for ξ:  ξ = δ - X*θ₁    where: δ depends on θ₂ and θ₁ depends on δ. 

i) Recall θ₁ comes from the regression of X on δ:  δ ~ Xθ₁ 
So θ₁ = (X'X)⁻¹X'δ  (OLS)   or [(Z(Z'Z)⁻¹Z'X)'(Z(Z'Z)⁻¹Z'X)]⁻¹(Z(Z'Z)⁻¹Z'X)δ  (2SLS)
Therefore: 
ξ = δ - X*θ₁  = δ - Mδ = (I-M)δ  and δ depends on θ₂:
ξ(θ₂) = (I-M) δ(θ₂) => [∂ξ(θ₂)/∂θ₂] = (I-M)*[∂δ(θ₂)/∂θ₂]

ii) There is no simple expression for δ, but it can be found through implicit differentiation of σ().
At the correct δ, we should have: observed share (s) = predicted share (σ)
s - σ(δ(θ₂),θ₂) = 0 
differentiate with resepct to θ₂ using chain rule:
0 - [∂σ/∂δ]*[∂δ/θ₂] - [∂σ/∂θ₂] = 0
=> [∂δ/θ₂] = -[∂σ/∂δ]⁻¹[∂σ/∂θ₂]         [2217*2217][2217*5] matrix

 
iii) Find the derivatives ∂σ/∂δ and ∂σ/∂θ₂

recall that the predicted share for product j is given by:
σⱼ = ∫ cⱼ / (1 + Σₖ cₖ ) f(vᵢ)dvᵢ
where cⱼ = exp(pⱼα + Xβ + pⱼvᵢₖₚσᵛₚ + Σₖ xⱼₖvᵢₖσᵛₖ)
therefore: ∂cⱼ/∂δⱼ = cⱼ   and   ∂cⱼ/∂δₘ = 0

Let 𝒯ⱼ represent the interior of the integral for product j in a given market
Thus 𝒯ⱼ = cⱼ / (1 + Σₖ cₖ ) and σⱼ = ∫ 𝒯ⱼ f(vᵢ)vᵢ

Differentiate using 1) fundamental theorem of calculus (bring derivative inside integral) and
2) quotient rule (for the own price elasticity). 

∂σⱼ/∂δₘ:
    where j=m (diagnoal):
        ∂σⱼ/∂δⱼ = ∫ 𝒯ⱼ (1 - 𝒯ⱼ) f(vᵢ)dvᵢ
    where j≠m (off diagnoal):
        ∂σⱼ/∂δₘ = ∫ - 𝒯ⱼ 𝒯ₘ f(vᵢ)dvᵢ

∂σⱼ/∂θ₂:
for the first θ₂ coefficient θ₂₁:
    ∂σⱼ/∂θ₂₁ = ∫ v₁ᵢ 𝒯ⱼ (x₁ⱼₜ - ∑ₖ x₁ⱼₜ 𝒯ⱼ) f(vᵢ)dvᵢ 

note that these derivatives are calculated within a given market t. 

The full gradient is: 
∂Q/∂θ₂ = ∂Q/∂ξ * ∂ξ/∂θ₂ = 2[Z'ξ]'W*(I-M)*-[∂σ/∂δ]⁻¹[∂σ/∂θ₂] ≈ 2[Z'ξ]'W*-[∂σ/∂δ]⁻¹[∂σ/∂θ₂]   
=# 

#=
Inputs
θ₂: 5x1 vector of random coefficients where the price coefficient is first
X : 2217x6 matrix of observables where price is the first column
v : 20x50x5 vector of random draws from joint normal mean 0 variance 1. 
    One random draw per each of the 5 θ₂ coefficents per person.
    Start with 50 sets of 5 draws for each of the 20 markets (50 unique individuals per market).
market_id: 2217x1 vector of market id for each product/observation  (cdid = market = year in this dataset)

ξ : 2217x1 vector of residuals from the objective function
𝒯 : 2217x50 matrix of market share estimates for each product for each individual
the interior of the sigma market share function.

θ₂ must be the only dynamic input. The optimization function will input the current θ₂ values and expect
the gradient of the function at θ₂. ξ and 𝒯 are calculated by the objective function and must be passed
 to the gradient.

2217 = number of products
6  = number of X variables and non-random coefficients
5  = number of estimated random coefficients 
15 = number of instruments (10) + endogenous X variables (5)
20 = number of markets
50 = number of individuals simulated in base case (also considered 100, 500, 1000, ...)
=#


function gradient(θ₂,X,Z,v,market_id,ξ,𝒯)

# Set up
# get number of products, individuals, and θ₂ coefficients
n_products = size(X,1)
n_individuals = size(v,2)
n_coefficients = size(θ₂,1)


# 1 - Take derivative of Q with respect to ξ
# ∂Q/∂θ₂ = ∂Q/∂ξ * ∂ξ/∂θ₂
W = inv(Z'Z)
∂Q_∂ξ = 2*(Z'ξ)'W*Z'


# 2 - find the derivatives ∂σ/∂δ and ∂σ/∂θ₂

    # 2.1 ∂σ/∂δ:
    # array to store the derivatives
    # each slice [:,:,i] is the 2217x2217 matrix for individual i
    ∂σᵢ_∂δ = zeros(n_products, n_products, n_individuals)

    # get the index of the diagonal for slice of ∂σᵢ_∂δ[:,:,individual] 
    diagonal_index = CartesianIndex.(1:n_products, 1:n_products) # object of (1,1) (2,2) ... (2217,2217) indices

    # calculate the derivative given 𝒯(j,i) values from objective function
    for individual in 1:n_individuals

        # derivative for off-diagonal elements: -𝒯ⱼᵢ * 𝒯ₘᵢ
        ∂σᵢ_∂δ[:,:,individual] = -𝒯[:,individual] * 𝒯[:,individual]'

        # derivative for diagonal elements: 𝒯ⱼᵢ * (1 - 𝒯ⱼᵢ)
        ∂σᵢ_∂δ[diagonal_index, individual] = 𝒯[:,individual] .* (1 .- 𝒯[:,individual])

    end

    # calculate mean over all individuals (Monty Carlo integration)
    ∂σ_∂δ = mean(∂σᵢ_∂δ, dims=3)[:,:] # semicolon to remove the dropped third dimension

    # calculate inverse 
    ∂σ_∂δ⁻¹ = zeros(size(∂σ_∂δ))
    # must be done market-by-market: products outside of given market do not affect shares within market (creates a block matrix)
    for market in unique(market_id)
        ∂σ_∂δ⁻¹[market_id.==market, market_id.==market] = inv(∂σ_∂δ[market_id.==market, market_id.==market])
    end


    
    # 2.2 ∂σ/∂θ₂:
    # ∂σⱼ/∂θ₂₁ = ∫ v₁ᵢ 𝒯ⱼ (x₁ⱼₜ - ∑ₖ x₁ⱼₜ 𝒯ⱼ) f(vᵢ)dvᵢ 

    # array to store the derivatives
    # each slice [:,i,c] is the 2217 vector for individual i for θ₂ coefficient c
    ∂σᵢ_∂θ₂ = zeros(n_products, n_individuals, n_coefficients)

    # calculate market-by-market
    for market in unique(market_id)
        # for each of the 5 coefficients
        for coef in 1:n_coefficients

            # calculate sum term for simplicity: ∑ₖ x₁ⱼₜ 𝒯ⱼ
            Σⱼx₁ⱼ𝒯ⱼᵢ = X[market_id.==market, coef]' * 𝒯[market_id.==market,:]

            # calculate derivative for all individuals for given coefficient: v₁ᵢ 𝒯ⱼ (x₁ⱼₜ - ∑ₖ x₁ⱼₜ 𝒯ⱼ)
            ∂σᵢ_∂θ₂[market_id.==market,:,coef] = v[market,:,coef]' .* 𝒯[market_id.==market,:] .* (X[market_id.==market,coef] .- Σⱼx₁ⱼ𝒯ⱼᵢ)
        end
    end

    # calculate mean over all individuals (Monty Carlo integration)
    # dimension 2 indexes the 50 individuals
    ∂σ_∂θ₂ = mean(∂σᵢ_∂θ₂, dims=2)[:,1,:] # semicolons/slicing to removed the dropped dimension



# 3. Combine derivaties to calculate gradient

# The full gradient is: 
# Q/∂θ₂ ∂= ∂Q/∂ξ * ∂ξ/∂θ₂ = (2[Z'ξ]'W*Z') (I-M)*-[∂σ/∂δ]⁻¹[∂σ/∂θ₂] ≈ 2[Z'ξ]'W*Z'(-[∂σ/∂δ]⁻¹)[∂σ/∂θ₂] 

# gradient calculation
# ∂Q_∂θ₂ = (2*(Z'ξ)'W) * Z' * -∂σ_∂δ⁻¹ * ∂σ_∂θ₂
∂Q_∂θ₂ = ∂Q_∂ξ * (-∂σ_∂δ⁻¹ * ∂σ_∂θ₂)

return ∂Q_∂θ₂'
end # end function

# run time = 1.24 s
# @btime gradient($θ₂,$X,$Z,$v_50,$cdid,$ξ,$𝒯)

end # end module
