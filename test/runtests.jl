using DataArrays, DataFrames, PDMats, ReTerms
using Base.Test

include(joinpath(dirname(@__FILE__),"data.jl"))

const sf = ScalarReTerm(ds[:Batch])
const Yield = convert(Vector{Float64}, array(ds[:Yield]))

@test size(sf) == (30, 6)
@test size(sf,1) == 30
@test size(sf,2) == 6
@test size(sf,3) == 1

dd = fill(5., 6)
@test sf'ones(30) == dd
@test ones(30)'sf == dd'
@test sf * ones(6) == ones(30)
@test (sf'sf)\(sf'Yield) == [1505.,1528.,1564.,1498.,1600.,1470.]
pls(sf, Yield - mean(Yield))

X = ones(30,1)
@test X'sf == dd'
update!(sf, 0.5)
@test sf'ones(30) == dd ./ 2.
@test (sf'sf)\(sf'Yield) == 2. .* [1505.,1528.,1564.,1498.,1600.,1470.]
pls(sf, Yield - mean(Yield))
