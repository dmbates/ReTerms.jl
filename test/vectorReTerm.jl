const Days = convert(Vector{Float64},slp[:Days])
const vf = ReTerms.VectorReMat(slp[:Subject],hcat(ones(length(Days)),Days)')
const Reaction = convert(Array,slp[:Reaction])

@test size(vf) == (180,36)
const vrp = vf'vf
@test (vf'ones(size(vf,1)))[1:4] == [10.,45,10,45]
@test isa(vrp,ReTerms.HBlkDiag{Float64})
@test eltype(vrp) == Float64
@test size(vrp) == (36,36)
const rhs1 = ones(36,2)
const x = similar(rhs1)
const b1 = copy(vrp.arr[:,:,1]) + I
@test sub(ReTerms.inflate!(vrp).arr,:,:,1) == b1
const cf = cholfact(b1)
@test A_ldiv_B!(x,vrp,rhs1)[1:2,1:2] == cf\ones(2,2)
@test triu!(sub(cholfact!(vrp).arr,:,:,1)) == cf[:U]
