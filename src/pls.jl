type FeTerm
    X::Matrix{Float64}
    XtX::Matrix{Float64}
end

function FeTerm(X::Matrix{Float64})
    XtX = X'X
    r = rank(cholfact(XtX; pivot=true))
    r == size(X,2) || throw(Base.RankDeficientException(r))
    FeTerm(X,XtX)
end

type LMM <: StatsBase.RegressionModel
    X::FeTerm
    re::Array{ReTerm,1}
    y::Vector{Float64}  # response vector
    uβ::Vector{Float64} # concatenation of spherical random effects and fixed-effects
    ty::Vector{Float64} # concatenation of Z'y blocks and X'y
end

function LMM(X::Matrix{Float64}, re::Vector{ReTerm{Float64}}, y::Vector{Float64})
    n,p = size(X)
    all(t -> size(t,1) == n, re) && length(y) == n || throw(DimensionMismatch(""))
    LMM(FeTerm(X),re,y)
end

function LMM(X::Matrix{Float64}, re::ReTerm{Float64}, y::Vector{Float64})
    size(re,1) == size(X,1) == length(y) || throw(DimensionMismatch(""))
    LMM(FeTerm(X),ReTerm[re],y)
end

function tune(lmm::LMM)
    r = copy(lmm.y)
    BLAS.gemv!('N',-1.,lmm.X.X,(lmm.X.X\lmm.y),1.,r)
    [optimize(λ -> objective!(re, λ, r), 0., 2.; abs_tol=0.05).minimum for re in lmm.re]
end
