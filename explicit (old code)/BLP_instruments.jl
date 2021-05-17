#= BLP instruments =#
# function to enclose the calculation of instruments.
# same code as Question 1a 1b.
# used to save space in the Question 1c file. 

module BLP_instrument_module
export BLP_instruments

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

function BLP_instruments(X, id, cdid, firmid)

n_products = size(id,1) # number of observations = 2217

# initialize arrays to hold the two sets of 5 instruments. 
IV_others = zeros(n_products,5)
IV_rivals = zeros(n_products,5)

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

return IV
end

end # end module 

#= sample code
X = Matrix(blp_data[!, ["const","hpwt","air","mpg","space"]]) # exogenous X variables
id = Vector(blp_data[!,"id"])
cdid = Vector(blp_data[!,"cdid"])
firmid = Vector(blp_data[!,"firmid"])

BLP_instruments(X, id, cdid, firmid)
=#