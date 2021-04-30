module BLP_derivatives
export gradient

using DataFrames        # for Not() and sample data
using LinearAlgebra     # basic math
using Statistics        # for mean()


#= Gradient of BLP objective function 

recall that the objective function retuns Q = [Z'ξ]'W[Z'ξ]

Q(ξ(θ₂)) = [Z'ξ(θ₂)]'W[Z'ξ(θ₂)]

Gradient: take derivative of Q with respect to the 5 coefficients in θ₂
∂Q/∂θ₂ = ∂Q/∂ξ * ∂ξ/∂θ₂

∂Q/∂ξ is the derivative of the quadratic form x'Ax: ∂x'Ax/∂x = x'[A + A'] = 2x'A   (if A is symmetric)
=> ∂Q/∂ξ = 2[Z'ξ]'W 
Which is a 1x2217 matrix

∂ξ/∂θ₂ is complicated. The final solution is:
∂ξ/∂θ₂ = (I-M)[∂σ/∂δ]⁻¹[∂σ/∂θ₂] ≈ [∂σ/∂δ]⁻¹[∂σ/∂θ₂]    which is a 2217x5 matrix
    where M = X[(Z(Z'Z)⁻¹Z'X)'(Z(Z'Z)⁻¹Z'X)]⁻¹(Z(Z'Z)⁻¹Z'X)  (2SLS projection matrix)


Derivation (steps i-iv): 
Start with expression for ξ:  ξ = δ - X*θ₁    where: δ depends on θ₂ and θ₁ depends on δ. 

i) Recall θ₁ comes from the regression of X on δ:  δ ~ Xθ₁ 
So θ₁ = (X'X)⁻¹X'δ  (OLS)   or [(Z(Z'Z)⁻¹Z'X)'(Z(Z'Z)⁻¹Z'X)]⁻¹(Z(Z'Z)⁻¹Z'X)δ  (2SLS)
Therefore: 
ξ = δ - X*θ₁  = δ - Mδ = (I-M)δ  and δ depends on θ₂:
ξ(θ₂) = (I-M) δ(θ₂) => [∂ξ(θ₂)/∂θ₂] = (I-M)*[∂δ(θ₂)/∂θ₂]

ii) There is no easy expression for δ, but it can be found through implicit differentiation of σ().
At the correct δ, we should have: observed share (s) = predicted share (σ)
s - σ(δ(θ₂),θ₂) = 0 
differentiate with resepct to θ₂:
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
        ∂σⱼ/∂δⱼ = ∫ 𝒯ⱼ (1 - 𝒯ⱼ) f(vᵢ)vᵢ
    where j≠m (off diagnoal):
        ∂σⱼ/∂δₘ = ∫ - 𝒯ⱼ 𝒯ₘ f(vᵢ)vᵢ

∂σⱼ/∂θ₂:
for the first θ₂ coefficient θ₂₁:
    ∂σⱼ/∂θ₂₁ = ∫ v₁ᵢ 𝒯ⱼ (x₁ⱼₜ - ∑ₖ x₁ⱼₜ 𝒯ⱼ) f(vᵢ)vᵢ 

note that these derivatives are calculated within a given market t. 

The full gradient is: 
∂Q/∂θ₂ = ∂Q/∂ξ * ∂ξ/∂θ₂ = 2[Z'ξ]'W*(I-M)*-[∂σ/∂δ]⁻¹[∂σ/∂θ₂] ≈ 2[Z'ξ]'W*-[∂σ/∂δ]⁻¹[∂σ/∂θ₂]   
=# 

#=
Inputs
θ₂: 5x1 vector of random coefficients where the price coefficient is first
X : 2217x6 matrix of observables where price is the frist column
s : 2217x1 vector of observed market shares
v : 2217x250 vector of pre-selected random draws from joint normal
market_id: 2217x1 vector of market id for each product/observation  (cdid = market = year in this dataset)

ξ : 2217x1 vector of residuals from the objective function. 
δ : 2217x1 vector δ calculated from X*θ₁ in the objective function

θ₂ must be the only dynamic input. The optimization function will input the current θ₂ values and expect
the gradient of the function at θ₂. ξ and δ are stored global by the objective function so that they can
be accessed by the gradient.
=#



θ₂ = [ 0.172, -2.528, 0.763, 0.589,  0.595]



function gradient(θ₂,X,s,Z,v,market_id, ξ, δ)
   # ∂Q_∂ξ = 2 * [Z'ξ]' * W
    # 1x2217

    # ∂A'x/∂x = A'
    # ∂x'A/∂x = A

end

# Set up

# get ξ and δ values from objective function
ξ = ξ_global
δ = X*θ₁_global





# 1 - Take derivative of Q with respect to the 5 coefficients in θ₂
∂Q/∂θ₂ = ∂Q/∂ξ * ∂ξ/∂θ₂

W = inv(Z'Z)
∂Q_∂ξ = 2*(Z'ξ)'W*Z'

# 2 - find the derivatives ∂σ/∂δ and ∂σ/∂θ₂

# get number of products in all markets and number of simulated individuals
n_products = size(X,1)
n_individuals = size(v,2)

# calculate μᵢⱼₜ for each of the simulated individuals in each market
μ = zeros(n_products, n_individuals)
for market in unique(market_id)
    # μⱼᵢ = ∑ₖ xⱼₖ * vₖᵢ * σₖ   where σₖ is one of the θ₂ coefficients  
    μ[market_id.==market,:] = X[market_id.==market,Not(6)] * (v_50[market,:,:] .* θ₂')' 
end

# initialize empty matrix of derivatives
∂σ_∂δ = zeros(n_products, n_products)


# the interior of the sigma market share integral
𝒯(j,i) = exp(δ[j] + μ[j,i]) / (1 + sum(exp.(δ[market_id.==market] + μ[market_id.==market,i])))

# derivatives of the interior of the integral for ∂σ/∂δ
σ_δ_integral_interior_diag(i) = 𝒯(j,i) * (1 - 𝒯(j,i))   # for diagonal terms where j=k
σ_δ_integral_interior_off_diag(i) = -𝒯(j,i) * 𝒯(k,i)    # for off diagonal terms j≠k


# 2:30 not parallel
# 1:30 parallel

Threads.@threads for market in unique(market_id)

    # get product ids for given market
    products = findall(market_id.==market)

    # get market id
    market = market_id[j]

    # get observables and indiviudals
    xₘ = X[market_id.==market,:]     # observables of all products in market with product j

    for j in products
        for k in products

            if j == k

                ∂σ_∂δ[j,k] = mean(σ_δ_integral_interior_diag.(1:n_individuals))

            end

            if j != k 

                ∂σ_∂δ[j,k] = mean(σ_δ_integral_interior_off_diag.(1:n_individuals))

            end

        end
    end
end








# version 1: ~1:30 with parallel. (3 hours+ without)
# loops through the whole matrix.

# loop through all products (rows)
Threads.@threads for j in 1:n_products    # started at 9:23 - took over 3 hours then failed
    # loop through all products (columns)
    for k in 1:n_products
        # when products are in the same market
        if (market_id[j]==market_id[k])

            # get market id
            market = market_id[j]
            
            # get observables and indiviudals
            xₘ = X[market_id.==market,:]     # observables of all products in market with product j

            # function defining the interior of the sigma function intergral for product j and invidual i. 
            𝒯(j,i) = exp(δ[j] + μ[j,i]) / (1 + sum(exp.(δ[market_id.==market] + μ[market_id.==market,i])))

            # if on the matrix diagonal ∂σⱼ/∂δⱼ:
            if j == k
                #integral_interior_diag(i) = 𝒯(j,i) * (1 - 𝒯(j,i)) 
                d(i) = 𝒯(j,i) * (1 - 𝒯(j,i)) 
                # calculate the value by Monty Carlo integration and assign value to derivative matrix
                #∂σ_∂δ[j,k] = mean(integral_interior_diag.(1:n_individuals))
                ∂σ_∂δ[j,k] = mean(d.(1:n_individuals))
            
            # if off the matrix diagnoal ∂σⱼ/∂δₖ:
            else
                integral_interior_off_diag(i) = -𝒯(j,i) * 𝒯(k,i)
                c(i) = -𝒯(j,i) * 𝒯(k,i)
                # calculate the value by Monty Carlo integration and assign value to derivative matrix
                #∂σ_∂δ[j,k] = mean(integral_interior_off_diag.(1:n_individuals))                
                ∂σ_∂δ[j,k] = mean(c.(1:n_individuals))                
            end


        end
    end
end














end # end module

