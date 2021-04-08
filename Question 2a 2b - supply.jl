#= 
ECO-2404
Problem Set 1
Question 2a and 2b

Estimate the pricing equation for product j in market m: 
    ln(mcⱼₘ) = Xⱼₘγ + ωⱼₘ 

Assume that Xⱼₘ is exogenous: E[ωⱼₘ|Xⱼₘ] = 0

For part a) assume marginal cost pricing: mcⱼₘ = priceⱼₘ
Since Xⱼₘ is exogenous, use OLS to estimate γ coefficeints. 

For part b) use price elasticities and assume firms are in equlibrium.

Use θ₃ in place of γ for consistency. 
=#

using CSV               # loading data
using DataFrames        # loading data
using LinearAlgebra     # basic math

# load data and set up variables
cd("C:\\Users\\Ray\\OneDrive\\Economics\\Course Material\\ECO2404 - Empirical\\Problem Set 1 Julia\\Submission")

# main dataset
blp_data = CSV.read("BLP_product_data.csv", DataFrame)
# construct vector of observables Xⱼₘ
X = Matrix(blp_data[!, ["const","hpwt","air","mpg","space"]]) # exogenous X variables
x₁= Matrix(blp_data[!, ["price", "const","hpwt","air","mpg","space"]]) # all X variables (price included)
# vector of prices pⱼₘ
P = Vector(blp_data[!, "price"])
# observed market shares
S = Vector(blp_data[!, "share"])
# firm and market id numbers
firm_id = Vector(blp_data[!, "firmid"])
market_id = Vector(blp_data[!, "cdid"])

# θ₁ and θ₂ estimates from the demand side
θ₁ = [-0.427, -9.999, 2.801, 1.099, -0.430, 2.795]
θ₂ = [ 0.172, -2.528, 0.763, 0.589,  0.595]

# pre-selected random draws
v = Matrix(CSV.read("BLP_v.csv", DataFrame, header=0))

#= 
Question 2a - marginal cost pricing
Assume firms price at marginal cost (competitive market) and solve for parameters with OLS.
=#

# set marginal cost equal to price
MC = P

# OLS Regression 

# parameter estimates
γ = θ₃ = inv(X'X)X'log.(MC)

# Robust Standard Errors
# residuals
ϵ = log.(MC) - X*θ₃
# covariance matrix
Σ = Diagonal(ϵ*ϵ')
Var_θ₃ = inv(X'X)*(X'*Σ*X)*inv(X'X)
# standard errors
SE_θ₃ = sqrt.(Diagonal(Var_θ₃))

# solution to 2a is θ₃ (γ) and SE_θ₃.
# γ = θ₃ = [1.625  1.534  0.741  -0.133  0.127]
# SE_θ₃  = [0.100  0.099  0.020   0.018  0.042]]



#=
Question 2b - multi-product firms setting prices in equilibrium
Assume firms set prices simultaneously to maximize static profits across all their products.

Solution to the FOC:
S - Δ(P-MC) = 0
solve for marginal cost: MC = P - Δ⁻¹S

Where Δ is a vector of own and cross price elastiticities, P is a vector of prices
and S is a vector of market shares. 

Finding Δ is the challenging part. See "supply elasiticites.jl" for the calculation
and detailed documentation.

Steps:
- Solve for Δ
- Calculate marginal cost MC
- Use MC to estimate θ₃ / γ with OLS

Note: ~10% of marginal cost estimates turn out to be negative in this dataset. 
They are dropped from analysis before the log transformation. This can be avoided
by estimating supply and demand simultaneously.
=#

# load module with function to calculate price elasticities
include("supply_price_elasticities.jl")
using .supply_price_elasticities

# calculate matrix of price elasticities (takes ~2 minutes)
Δ = price_elasticities(θ₁, θ₂, x₁, S, v, market_id, firm_id)

# get inverse
Δ⁻¹ = inv(Δ)

# calculate MC
MC = P - Δ⁻¹*S

# OLS Regression 

# drop any negative marginal cost estimates to allow for log transformation
X = X[MC.>0,:]
MC = MC[MC.>0]

# parameter estimates
γ = θ₃ = inv(X'X)X'*log.(MC)

# Robust Standard Errors
# residuals
ϵ = log.(MC) - X*θ₃
# covariance matrix
Σ = Diagonal(ϵ*ϵ')
Var_θ₃ = inv(X'X)*(X'*Σ*X)*inv(X'X)
# standard errors
SE_θ₃ = sqrt.(Diagonal(Var_θ₃))

# solution to 2b is θ₃ (γ) and SE_θ₃.
# γ = θ₃ = [1.0843  2.2343  0.9117 -0.2172  0.0429]
# SE_θ₃  = [0.2269  0.1945  0.0434  0.0473  0.0980]