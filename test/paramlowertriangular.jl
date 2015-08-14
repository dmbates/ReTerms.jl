const m1 = ReTerms.ColMajorLowerTriangular(3)

@test size(m1) == (3,3)
@test m1[:θ] == [1.,0,0,1,0,1]
@test lowerbd(m1) == [0.,-Inf,-Inf,0.,-Inf,0.]

m1[:θ] = [1.:6;]
@test m1[:θ] == [1.:6;]

