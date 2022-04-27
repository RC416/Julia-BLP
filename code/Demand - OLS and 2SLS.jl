#= 
Estimation of the demand parameters using OLS and 2SLS. Contains documentation of the variables.

Part 1 - parameters are estimated with OLS. 
Part 2 - instruments are construsted and used for 2SLS.
=#

using CSV               # loading data
using DataFrames        # loading data
using LinearAlgebra     # basic math

# set working directory
cd("C:\\Users\\Ray\\Documents\\GitHub\\Julia BLP\\Julia-BLP\\data and random draws")

# load data
blp_data = CSV.read("BLP_product_data.csv", DataFrame)

#= blp data contains the following variables

# Identifying Variables
firmid: id number of a car manufacturer. 26 unique firms.
cdid: year since 1970, each year treated as new market. 20 years from 1970-1990.
id: unique vehicle id number. 2217 unique products. 

# Product Characteristics
const: value 1 for every observation
hpwt: horsepower to weight ratio
air: presence of airbags (binary)
mpg: miles per gallon
space: amount of space in the car
price: price of the car

# Market Shares
share: market share of given product in given year (cdid)
outshr: fraction of consumers that did not buy any car that year (same for all cars in a given year)
=#

# Load X variables. 2217x5 and 2217x6 matrices respectively
X = Matrix(blp_data[!, ["const","hpwt","air","mpg","space"]]) # exogenous X variables
x₁ = Matrix(blp_data[!, ["price","const","hpwt","air","mpg","space"]]) # exogenous x variables and price

# Load Y variables. 2217x1 vectors
share = Vector(blp_data[!,"share"])
outshr = Vector(blp_data[!,"outshr"])
Y = log.(share) - log.(outshr) # market share normalized by outshare

# ------------------------------------------------------------------------------------
# Part 1 - Estimation with OLS 

# Estimate demand coefficients in logit model with normal OLS.
θ₁ = inv(x₁'x₁)*x₁'Y

# Calculate robust standard errors 

# get residuals 
ϵ = Y - x₁*θ₁
# covariance matrix 
𝚺 = Diagonal(ϵ*ϵ')
Var_θ = inv(x₁'x₁)*(x₁'*𝚺*x₁)*inv(x₁'x₁)
# get standard errors
SE_θ = sqrt.(diag(Var_θ))

# approximate solution
# θ₁   = [-0.089 -11.352  0.526  0.016  0.501  2.740]
# SE_θ = [ 0.004   0.377  0.297  0.071  0.065  0.156]

# ------------------------------------------------------------------------------------
# Part 2 - Estimating with 2SLC

# construct the BLP instruments

id = Vector(blp_data[!,"id"])
cdid = Vector(blp_data[!,"cdid"])
firmid = Vector(blp_data[!,"firmid"])

#= Two sets of instruments
1. Characteristics of other products from the same company in the same market.
Logic: the characteristics of other products affect the price of a 
given product but not its demand. Alternatively, firms decide product characteristics X 
before observing demand shocks ξ. 
2. Characteristics of other products from different companies in the same market.
Logic: the characteristics of competing products affects the price of a
given product but not its demand. Alternatively, other firms decide their product
characteristics X without observing the demand shock for the given product ξ.
=#

n_products = size(id,1) # number of observations = 2217

# initialize arrays to hold the two sets of 5 instruments. 
IV_others = zeros(n_obs,5)
IV_rivals = zeros(n_obs,5)

# loop through every product in every market (every observation)
for j in 1:n_products

    # 1. Set of instruments from other product characteristics
    # get the index of all different products (id) made by the same firm (firmid)
    # in the same market/year (cdid) 
    other_index = (firmid.==firmid[j]) .* (cdid.==cdid[j]) .* (id.!=id[j])
    # x variable values for other products (excluding price)
    other_x_values = X[other_index,:]
    # sum along columns
    IV_others[j,:] = sum(other_x_values, dims=1)

    # 2. Set of instruments from rival product characteristics
    # get index of all products from different firms (firmid) in the same market/year (cdid)
    rival_index = (firmid.!=firmid[j]) .* (cdid.==cdid[j])
    # x variable values for other products (excluding price)
    rival_x_values = X[rival_index,:]
    # sum along columns
    IV_rivals[j,:] = sum(rival_x_values, dims=1)
    
end

# vector of observations and instruments
IV = [X IV_others IV_rivals]
# restate Y for clarity
Y = log.(share) - log.(outshr)

# estimate 2SLS coefficients 
𝓧 = IV*inv(IV'IV)*IV'x₁      # X hat matrix (stage 1)
θ₁_IV = inv(𝓧'𝓧)*𝓧'Y        # coefficient estimates (stage 2)

# robust standard errors
𝛀 = IV*inv(IV'IV)*IV'           # projection matrix 
ϵ_IV = Y - x₁*θ₁_IV             # residuals 
𝚺_IV = Diagonal(ϵ_IV*ϵ_IV')     # residual variance

Var_θ_IV = inv(x₁'*𝛀*x₁) * (x₁'*𝛀*𝚺_IV*𝛀*x₁) * inv(x₁'*𝛀*x₁)
SE_θ_IV = sqrt.(diag(Var_θ_IV))

# approximate solution
# θ₁_IV   = [-0.139 -11.154  1.831  0.555  0.404  0.270]
# SE_θ_IV = [ 0.011   0.390  0.396  0.128  0.069  0.167]
