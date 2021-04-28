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

include("BLP_functions.jl")    # module with custom BLP functions (objective function and σ())
include("BLP_instruments.jl")  # module to calculate BLP instruments

using .BLP_functions
using .BLP_instrument_module

using CSV               # loading data
using DataFrames        # loading data
using LinearAlgebra     # basic math
using Statistics        # for mean


# Load key data ------------------------------------------------------------------
blp_data = CSV.read("BLP_product_data.csv", DataFrame) # dataframe with all observables 
v = Matrix(CSV.read("BLP_v.csv", DataFrame, header=0)) # pre-selected random draws from joint normal

# Load X variables. 2217x5 and 2217x6 matrices respectively
X = Matrix(blp_data[!, ["const","hpwt","air","mpg","space"]]) # exogenous X variables
x₁ = Matrix(blp_data[!, ["price","const","hpwt","air","mpg","space"]]) # exogenous x variables and price

# Load Y variable market share. 2217x1 vector
share = Vector(blp_data[!,"share"])

# product, market, and firm ids 
id = Vector(blp_data[!,"id"])
cdid = Vector(blp_data[!,"cdid"])
firmid = Vector(blp_data[!,"firmid"])

# BLP instruments. Function uses same code as Question 1b to calculate instruments.
Z = BLP_instruments(X, id, cdid, firmid)


# Minimize objective function -----------------------------------------------------

using Optim             # for minimization functions. see: http://julianlsolvers.github.io/Optim.jl/v0.9.3/user/config/
using BenchmarkTools    # for timing/benchmarking functions

# θ₂ guess value. Initialze elements as floats.
# this implies starting θ₁ values equal to the IV coefficients (random effects = 0)
θ₂ = [0.0, 0.0, 0.0, 0.0, 0.0]

# test run and timing of objective function
Q, θ₁, ξ = demand_objective_function(θ₂,x₁,share,Z,v,cdid)   # returns objective function value, and θ₁ and ξ estiamtes 
@btime demand_objective_function($θ₂,$x₁,$share,$Z,$v,$cdid) 
# v1-3: initially 55 seconds per call, later ~20 seconds
# v4: 6.7 seconds per call (issue with adjoints in vector of random draws)
# v5: 2.6 seconds per call (parallelized lower level sigma loop instead of higher-level objective function) 
# v6: 2.3 seconds per call

# optimization
# temporary function that takes only θ₂ and returns objective function value
f(θ₂) = demand_objective_function(θ₂,x₁,share,Z,v,cdid)[1]
# set up optimization object
result = optimize(f, θ₂, NelderMead(), Optim.Options(x_tol=1e-2, iterations=100, show_trace=true, show_every=10))
# v4: 6400s / 107 minutes to converge at tol 1e-2
# v5: 4700s / 79 minutes / 488 iterations / 841 calls to converge at tol 1e-2
# v6: 4500s / 75 minutes / 488 iterations / 841 calls to converge at tol 1e-2
# note: 100 iterations (<15 minutes) is sufficient get a close estimate.  

# get optimal θ₂ and θ₁ values
θ₂ = Optim.minimizer(result)
θ₁ = demand_objective_function(θ₂,x₁,share,Z,v,cdid)[2] # get corresponding θ₁

# other optimization diagnostics
Optim.minimizer(result)    # optimal input values
Optim.minimum(result)      # minimum value of objective function
Optim.iterations(result)   # number of iterations needed

# solution is θ₂ and θ₁ values:
# θ₂ = [ 0.172, -2.528, 0.763, 0.589,  0.595]
# θ₁ = [-0.427, -9.999, 2.801, 1.099, -0.430, 2.795]