using ReTerms, DataArrays, PDMats
using Base.Test

Batch = compact(@pdata(rep('A':'F',5)))
sf = SimpleScalarReTerm(Batch,1.)

@test size(sf) == (6,30)
@test size(sf,1) == 6
@test size(sf,2) == 30
@test size(sf,3) == 1

@test sf * ones(30) == fill(5.,6)
@test sf'ones(6) == ones(30)

