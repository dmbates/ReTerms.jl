fm1 = LMM(ds[:Batch],ds[:Yield])

@test size(fm1.A) == (2,2)
@test size(fm1.trms) == (2,)
@test size(fm1.R) == (2,2)
@test size(fm1.Λ) == (1,)
@test lowerbd(fm1) == zeros(1)
@test fm1[:θ] == ones(1)

fm1[:θ] = [0.713]
@test objective(fm1) ≈ 327.34216280955366

fit(fm1)
@test objective(fm1) ≈ 327.3270598811428

fm2 = LMM(ds2[:Batch],ds2[:Yield])
@test lowerbd(fm2) == zeros(1)
fit(fm2)
@test fm2[:θ] == zeros(1)
