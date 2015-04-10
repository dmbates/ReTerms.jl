using ReTerms, DataArrays, PDMats
using Base.Test

const Batch = compact(@pdata(rep('A':'F', 1, 5)))
const sf = ScalarReTerm(Batch)
const Yield = [1545.,1440.,1440.,1520.,1580.,1540.,1555.,1490.,1560.,1495.,
               1595.,1550.,1605.,1510.,1560.,1445.,1440.,1595.,1465.,1545.,
               1595.,1630.,1515.,1635.,1625.,1520.,1455.,1450.,1480.,1445.]

@test size(sf) == (30, 6)
@test size(sf,1) == 30
@test size(sf,2) == 6
@test size(sf,3) == 1

dd = fill(5., 6)
@test sf'ones(30) == dd
@test sf * ones(6) == ones(30)
@test crossprod(sf)\(sf'Yield) == [1505.,1528.,1564.,1498.,1600.,1470.]
(crossprod(sf) + I)\(sf' * (Yield - mean(Yield)))

update!(sf, 0.5)
@test sf'ones(30) == dd ./ 2.
@test crossprod(sf)\(sf'Yield) == 2. .* [1505.,1528.,1564.,1498.,1600.,1470.]
(crossprod(sf) + I)\(sf' * (Yield - mean(Yield)))
