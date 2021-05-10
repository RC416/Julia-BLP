#= 
ECO-2404
Problem Set 1
Question 1c

Estimation of demand parameters through BLP method.

Relies on two key functions defined in BLP_functions. 

objective_function() 
given theta, solves for delta using contraction mapping
constructs GMM criterion function and value.

Uses Optim optimization package to find the θ₂ that minimizes the function.
=#

# Load key functions and packages -------------------------------------------------
cd("C:\\Users\\Ray\\Documents\\GitHub\\Julia BLP\\Julia-BLP")

include("Fast BLP functions.jl")    # module with custom BLP functions (objective function and σ())
include("BLP_instruments.jl")       # module to calculate BLP instruments

using .BLP_functions
using .BLP_instrument_module

using CSV               # loading data
using DataFrames        # loading data
using LinearAlgebra     # basic math
using Statistics        # for mean


# Load key data ------------------------------------------------------------------
blp_data = CSV.read("BLP_product_data.csv", DataFrame) # dataframe with all observables 
v_50 = Matrix(CSV.read("random_draws_50_individuals.csv", DataFrame, header=0)) # pre-selected random draws from joint normal to simulate 50 individuals
# reshape to 3-d arrays: v(market, individual, coefficient draw) 
v_50 = reshape(v_50, (20,50,5)) # 20 markets, 50 individuals per market, 5 draws per invididual (one for each θ₂ random effect coefficient)


# Load X variables. 2217x5 and 2217x6 matrices respectively
X = Matrix(blp_data[!, ["price","const","hpwt","air","mpg","space"]]) # exogenous x variables and price

# Load Y variable market share. 2217x1 vector
share = Vector(blp_data[!,"share"])

# product, market, and firm ids 
id = Vector(blp_data[!,"id"])
cdid = Vector(blp_data[!,"cdid"])
firmid = Vector(blp_data[!,"firmid"])

# BLP instruments. Function uses same code as Question 1b to calculate instruments.
# price (column 6) not included in BLP instruments.
Z = BLP_instruments(X[:,Not(1)], id, cdid, firmid)


# Minimize objective function -----------------------------------------------------

using Optim             # for minimization functions. see: http://julianlsolvers.github.io/Optim.jl/v0.9.3/user/config/
using BenchmarkTools    # for timing/benchmarking functions

# θ₂ guess value. Initialze elements as floats.
# this implies starting θ₁ values equal to the IV coefficients (random effects = 0)
θ₂ = [0.0, 0.0, 0.0, 0.0, 0.0]

# test run and timing of objective function
Q, θ₁, ξ, 𝒯 = demand_objective_function(θ₂,X,share,Z,v_50,cdid)   # returns objective function value, and θ₁ and ξ estiamtes 
@btime demand_objective_function($θ₂,$X,$share,$Z,$v_50,$cdid)  
# v1-3: initially 55 seconds per call, later ~20 seconds
# v4: 6.7 seconds per call (issue with adjoints in vector of random draws)
# v5: 2.6 seconds per call (parallelized lower level sigma loop instead of higher-level objective function) 
# v6: 2.2 seconds per call
# fast BLP 1: 0.088 seconds per call

# optimization
# temporary function that takes only θ₂ and returns objective function value
f(θ₂) = demand_objective_function(θ₂,X,share,Z,v_50,cdid)[1]
# set up optimization object
result = optimize(f, θ₂, NelderMead(), Optim.Options(x_tol=1e-2, iterations=500, show_trace=true, show_every=10))
# v4: 6400s / 107 minutes to converge at tol 1e-2ll
# v5: 4700s / 79 minutes / 488 iterations / 841 calls to converge at tol 1e-2
# v6: 4500s / 75 minutes / 488 iterations / 841 calls to converge at tol 1e-2
# note: 100 iterations (<15 minutes) is sufficient get a close estimate.
# fast BLP 1: 205s / 3.5 mintues / 356 iterations / 587 calls to converge at tol 1e-2

# get optimal θ₂ and θ₁ values
θ₂ = Optim.minimizer(result)
θ₁ = demand_objective_function(θ₂,X,share,Z,v_50,cdid)[2] # get corresponding θ₁

# other optimization diagnostics
Optim.minimizer(result)    # optimal input values
Optim.minimum(result)      # minimum value of objective function
Optim.iterations(result)   # number of iterations needed



# using gradient function
include("BLP_derivatives.jl")    # module with custom BLP functions (objective function and σ())

using .BLP_derivatives

# anonomyous objective function 
function f(θ₂)

    # run objective function and get key outputs
    Q, θ₁, ξ, 𝒯 = demand_objective_function(θ₂,X,share,Z,v_50,cdid)

    # assign outputs to globals so that they can be used in gradient
    #global θ₁ = θ₁
    #global ξ = ξ
    #global 𝒯 = 𝒯

    # return objective function value
    return Q
end

function ∇(storage, θ₂)
    Q, θ₁, ξ, 𝒯 = demand_objective_function(θ₂,X,share,Z,v_50,cdid)

    ∇ = gradient(θ₂,X,Z,v_50,cdid,ξ,𝒯)
    storage[1] = ∇[1]
    storage[2] = ∇[2]
    storage[3] = ∇[3]
    storage[4] = ∇[4]
    storage[5] = ∇[5]
end

result = optimize(f, ∇, θ₂, LBFGS(), Optim.Options(x_tol=1e-2, iterations=50, show_trace=true, show_every=1))
Optim.minimizer(result)



function g(θ₂)
    Q, θ₁, ξ, 𝒯 = demand_objective_function(θ₂,X,share,Z,v_50,cdid)
    return norm(gradient(θ₂,X,Z,v_50,cdid,ξ,𝒯))
end
result = optimize(g, θ₂, NelderMead(), Optim.Options(x_tol=1e-1, iterations=500, show_trace=true, show_every=10))
Optim.minimizer(result)


# solution is θ₂ and θ₁ values:
# θ₂ = [ 0.172, -2.528, 0.763, 0.589,  0.595]
# θ₁ = [-0.427, -9.999, 2.801, 1.099, -0.430, 2.795]



function f!(θ₂)
    return θ₂'θ₂
end

function ∇!(storage, θ₂)
    storage[1] = 2 * θ₂[1] 
    storage[2] = 2 * θ₂[2] 
    storage[3] = 2 * θ₂[3] 
    storage[4] = 2 * θ₂[4] 
    storage[5] = 2 * θ₂[5]
end

θ₂ = [ 2000.0, -2000.0, 5000.0, 6000.0, -10000.0]
result = optimize(f!, ∇!, θ₂, LBFGS(),Optim.Options(show_trace=true, show_every=1))
Optim.minimizer(result)



function f2(x)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function g!(G, x)
    G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    G[2] = 200.0 * (x[2] - x[1]^2)
end

function h!(H, x)
    H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    H[1, 2] = -400.0 * x[1]
    H[2, 1] = -400.0 * x[1]
    H[2, 2] = 200.0
end

initial_x = zeros(2)

optimize(f2, g!, h!, initial_x, Newton(), Optim.Options(show_trace=true, show_every=1))