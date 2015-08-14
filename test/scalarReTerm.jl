const sf = ReMat(ds[:Batch])
const Yield = convert(Vector{Float64}, convert(Array,ds[:Yield]))

@test size(sf) == (30, 6)
@test size(sf,1) == 30
@test size(sf,2) == 6
@test size(sf,3) == 1

dd = fill(5., 6)
@test sf'ones(30) == dd
const crp = sf'sf
@test crp == fill(5.,(1,1,6))
@test sf'Yield == [7525.0,7640.0,7820.0,7490.0,8000.0,7350.0]
@test Diagonal(vec(crp))\(sf'Yield) == [1505.,1528.,1564.,1498.,1600.,1470.]

