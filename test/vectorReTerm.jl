const Days = convert(Vector{Float64},slp[:Days])
const vf = ReMat(slp[:Subject],hcat(ones(length(Days)),Days)')
const Reaction = convert(Array,slp[:Reaction])

@test size(vf) == (180,36)
const vrp = vf'vf
@test (vf'ones(size(vf,1)))[1:4] == [10.,45,10,45]
