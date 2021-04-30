


v

cdid

findfirst.([cdid], unique(cdid))

findfirst.(cdid, 1)

findfirst(unique(cdid))

cdid


fvs = findfirst.(isequal.(unique(cdid)),[cdid])

v_small = v[fvs,:]


n_markets = 20
n_individuals = 50
n_coefficients = 5

v_50 = zeros(20, 50, 5)
v_50 = zeros(n_markets, n_individuals, n_coefficients)

for market in 1:n_markets
    for i in 1:n_individuals
        v_50[market, i, :] = v_small[market, [i,i+50,i+100,i+150,i+200]]
    end
end



v_5000 = v[1:2000,:]


v_50 = Matrix(CSV.read("random_draws_50_individuals.csv", DataFrame, header=0)) # pre-selected random draws from joint normal to simulate 50 individuals
v_5000 = Matrix(CSV.read("random_draws_5000_individuals.csv", DataFrame, header=0)) # pre-selected random draws from joint normal to simulate 50 individuals

# reshape to 3-d arrays: v(market, individual, coefficient draw) 
# the sets of 50 individuals (v_50) is used in most places to estimate market share. 50 is a compromise between speed and precision.
# the sets of 5000 individuals (v_5000) is used for the diagonal of the price elastiticty matrix in supply price elasticities which
# only needs to be calculated once, so greater precision can be achieved. 
v_50 = reshape(v_50, (20,50,5)) # 20 markets, 50 individuals per market, 5 draws per invididual (one for each θ₂ random effect coefficient)
v_5000 = reshape(v_5000, (20,5000,5)) # 20 markets, 5000 individuals per market, 5 draws per invididual (one for each θ₂ random effect coefficient)






@btime σ($δₘ,$θ₂,$xₘ,$vₘ)