const sf1 = ReTerm(psts[:Sample])
const sf2 = ReTerm(psts[:Batch])

@test size(sf1) == (60,30)
@test size(sf2) == (60,10)
const crpr1 = sf1'sf1
const crpr2 = sf2'sf2
const pr12 = sf2'sf1
@test crpr1 == fill(2.,(1,1,30))
@test crpr2 == fill(6.,(1,1,10))
