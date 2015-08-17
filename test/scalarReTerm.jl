const sf = ReMat(ds[:Batch])
const Yield = convert(Vector{Float64},ds[:Yield])

@test size(sf) == (30, 6)
@test size(sf,1) == 30
@test size(sf,2) == 6
@test size(sf,3) == 1

dd = fill(5., 6)
@test sf'ones(30) == dd

const crp = sf'sf
@test isa(crp, ReTerms.HBlkDiag{Float64})
@test size(crp) == (6,6)
@test crp.arr == fill(5.,(1,1,6))
const rhs = sf'Yield
@test rhs == [7525.0,7640.0,7820.0,7490.0,8000.0,7350.0]
@test A_ldiv_B!(similar(rhs),crp,rhs) == [1505.,1528.,1564.,1498.,1600.,1470.]

const L = ReTerms.ColMajorLowerTriangular(1)
L[:θ] = [0.5]

@test isa(Ac_mul_B!(L,crp),ReTerms.HBlkDiag)
@test crp.arr == fill(2.5,(1,1,6))
