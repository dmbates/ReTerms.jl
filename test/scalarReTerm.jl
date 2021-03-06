const sf = ReMat(ds[:Batch])
# check alternative constructors
@test sf == ReMat(ds[:Batch],ones(length(ds[:Batch])))

const Yield = convert(Array,ds[:Yield])

# check size methods
@test size(sf) == (30, 6)
@test size(sf,1) == 30
@test size(sf,2) == 6
@test size(sf,3) == 1

dd = fill(5., 6)
@test sf'ones(30) == dd

@test ones(30)'sf == dd'
const crp = sf'sf
@test isa(crp, Diagonal{Float64})
const crp1 = copy(crp)
@test crp1 == crp
@test crp[2,6] == 0
@test crp[6,6] == 5
@test size(crp) == (6,6)
@test crp.diag == fill(5.,6)
const rhs = sf'Yield
@test rhs == [7525.0,7640.0,7820.0,7490.0,8000.0,7350.0]
@test A_ldiv_B!(crp,copy(rhs)) == [1505.,1528.,1564.,1498.,1600.,1470.]

const L = ReTerms.LT(sf)
L[:θ] = [0.5]

@test isa(scale!(L,crp),Diagonal)
@test crp.diag == fill(2.5,6)
@test copy!(crp1,crp) == crp

const sf1 = ReMat(psts[:Sample])
const sf2 = ReMat(psts[:Batch])
@test size(sf1) == (60,30)
@test size(sf2) == (60,10)

const crp11 = sf1'sf1
const pr21 = sf2'sf1
const crp22 = sf2'sf2

@test isa(crp11,Diagonal{Float64})
@test isa(crp22,Diagonal{Float64})
@test isa(pr21,SparseMatrixCSC{Float64})
