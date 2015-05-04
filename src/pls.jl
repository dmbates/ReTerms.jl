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

Base.size(f::FeTerm) = size(f.X)
Base.size(f::FeTerm,i::Integer) = size(f.X,i)

type LMM <: StatsBase.RegressionModel
    X::FeTerm
    re::Vector
    y::Vector{Float64}  # response vector
    uβ::Vector{Float64} # concatenation of spherical random effects and fixed-effects
    ty::Vector{Float64} # concatenation of Z'y blocks and X'y
end

function LMM{T<:Real}(X::FeTerm, rev::Vector, y::Vector{T})
    n,p = size(X)
    all(t -> size(t,1) == n, rev) && length(y) == n || throw(DimensionMismatch(""))
    yy = convert(Vector{Float64},y)
    ty = vcat([re'yy for re in rev]..., X.X'yy)
    LMM(X,rev,yy,zeros(length(ty)),ty)
end

LMM(X::Matrix, re::ReTerm, y::Vector) =  LMM(FeTerm(convert(Matrix{Float64},X)),ReTerm[re],y)
LMM(re::ReTerm,y::Vector) = LMM(ones((length(y),1)),re,y)
LMM(p::PooledDataVector,y::Vector) = LMM(reterm(p),y)

function tune(lmm::LMM)
    r = copy(lmm.y)
    BLAS.gemv!('N',-1.,lmm.X.X,(lmm.X.X\lmm.y),1.,r)
    [optimize(λ -> objective!(re, λ, r), 0., 2.; abs_tol=0.05).minimum for re in lmm.re]
end
