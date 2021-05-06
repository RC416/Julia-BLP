





fvs = fivndfirst.(isequal.(unique(cdid)),[cdid])

v_small = v[fvs,:]


n_markets = 20
n_individuals = 50
n_coefficients = 5

v_50_3 = zeros(20, 50, 5)
v_50_2 = zeros(n_markets, n_individuals, n_coefficients)

for market in 1:n_markets
    for i in 1:n_individuals
        v_50_3[market, i, :] = v_old[fvs[market], [i,i+50,i+100,i+150,i+200]]
        v_50_3[market, i, :] = v_old[market, [i,i+50,i+100,i+150,i+200]]
    end
end

v_100 = v[1:40,:]
v_200 = v[1:80,:]
v_500 = v[1:200,:]
v_1000 = v[1:400,:]
v_5000 = v[1:2000,:]


v_50 = Matrix(CSV.read("random_draws_50_individuals.csv", DataFrame, header=0)) # pre-selected random draws from joint normal to simulate 50 individuals
v_5000 = Matrix(CSV.read("random_draws_5000_individuals.csv", DataFrame, header=0)) # pre-selected random draws from joint normal to simulate 50 individuals

# reshape to 3-d arrays: v(market, individual, coefficient draw) 
# the sets of 50 individuals (v_50) is used in most places to estimate market share. 50 is a compromise between speed and precision.
# the sets of 5000 individuals (v_5000) is used for the diagonal of the price elastiticty matrix in supply price elasticities which
# only needs to be calculated once, so greater precision can be achieved. 
v_50 = reshape(v_50, (20,50,5)) # 20 markets, 50 individuals per market, 5 draws per invididual (one for each θ₂ random effect coefficient)
v_5000 = reshape(v_5000, (20,5000,5)) # 20 markets, 5000 individuals per market, 5 draws per invididual (one for each θ₂ random effect coefficient)

v_100 = reshape(v_100, (20,500))
v_200 = reshape(v_200, (20,1000))
v_500 = reshape(v_500, (20,2500))
v_1000 = reshape(v_1000, (20,5000))
v_5000 = reshape(v_5000, (20,25000))

redone_v_50 = reshape(v_50, (20,250))

reloaded_v_50 = reshape(redone_v_50, (20,50,5))


@btime σ($δₘ,$θ₂,$xₘ,$vₘ)

CSV.write("random_draws_50_individuals2.csv", DataFrame(redone_v_50), writeheader=false)
CSV.write("random_draws_100_individuals.csv", DataFrame(v_100), writeheader=false)
CSV.write("random_draws_200_individuals.csv", DataFrame(v_200), writeheader=false)
CSV.write("random_draws_500_individuals.csv", DataFrame(v_500), writeheader=false)
CSV.write("random_draws_1000_individuals.csv", DataFrame(v_1000), writeheader=false)
CSV.write("random_draws_5000_individuals.csv", DataFrame(v_5000), writeheader=false)

v_50

v_100 = reshape(v_100, (20,100,5))