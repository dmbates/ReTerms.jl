using ReTerm,DataArrays,PDMats
using Base.Test

Batch = compact(@pdata(rep('A':'F',5)))
sf = SimpleScalarReTerm(Batch,1.)
@test sf * ones(30) == fill(5.,5)
