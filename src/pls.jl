type FeTerm
    X::Matrix{Float64}
    function FeTerm(X::Matrix{Float64})
        rank(cholfact!(X'X,:U,Val{true})) == size(X,2) || error("X is not of full column rank")
        new(X)
    end
end

Base.size(f::FeTerm) = size(f.X)
Base.size(f::FeTerm,i::Integer) = size(f.X,i)

Base.Ac_mul_B!(r::DenseVecOrMat,f::FeTerm,t::ScalarReTerm) = Ac_mul_B!(r,f.X,t)
Base.Ac_mul_B(f::FeTerm,t::ScalarReTerm) = Ac_mul_B!(Array(Float64,(size(f,2),size(t,2))),f,t)
Base.Ac_mul_B!(r::DenseVecOrMat,t::ScalarReTerm,f::FeTerm) = Ac_mul_B!(r,t,f.X)
Base.Ac_mul_B(t::ScalarReTerm,f::FeTerm) = Ac_mul_B!(Array(Float64,(size(t,2),size(f,2))),t,f)
Base.Ac_mul_B(f::FeTerm,g::FeTerm) = Ac_mul_B!(Array(Float64,(size(f,2),size(g,2))),f.X,g.X)
Base.Ac_mul_B!(r::Matrix{Float64},f::FeTerm,g::FeTerm) = Ac_mul_B!(r,f.X,g.X)
Base.Ac_mul_B!(r::DenseVecOrMat, f::FeTerm, v::DenseVecOrMat) = Ac_mul_B!(r,f.X,v)
Base.Ac_mul_B(f::FeTerm,v::DenseVecOrMat) = Ac_mul_B(f.X,v)

type LMM <: StatsBase.RegressionModel
    trms::Vector{Any}
    A::Matrix{Any}
    y::Vector{Float64}  # response vector
    uβ::Vector{Float64} # concatenation of spherical random effects and fixed-effects
    ty::Vector{Float64} # concatenation of Z'y blocks and X'y
end

function LMM{T<:Real}(X::FeTerm, rev::Vector, y::Vector{T})
    n,p = size(X)
    all(t -> size(t,1) == n, rev) && length(y) == n || throw(DimensionMismatch(""))
    yy = convert(Vector{Float64},y)
    trms = Any[rev,X;]
    ntrms = length(trms)
    A = Array(Any,(ntrms,ntrms))
    R = Array(Any,(ntrms,ntrms))
    for j in 1:(ntrms-1),i in (j+1):ntrms
        A[i,j] = R[i,j] = nothing
    end
    for j in 1:ntrms, i in 1:j
        A[i,j] = Ac_mul_B(trms[i],trms[j])
    end
    ty = vcat([t'yy for t in trms]...)
    LMM(trms,A,yy,zeros(length(ty)),ty)
end

LMM(X::Matrix, re::Vector, y::Vector) =  LMM(FeTerm(convert(Matrix{Float64},X)),ReTerm[re],y)
LMM(X::Matrix, p::Vector{PooledDataVector},y::Vector) = LMM(FeTerm(X),map(reterm,p),y)
LMM(X::Matrix, p::Vector{PooledDataVector},y::DataVector) = LMM(FeTerm(X),map(reterm,p),convert(Array,y))
LMM(re::Vector,y::Vector) = LMM(ones((length(y),1)),re,y)
LMM(p::Vector{PooledDataVector},y::Vector) = LMM(FeTerm(ones((length(y),1))),map(reterm,p),y)
LMM(p::Vector{PooledDataVector},y::DataVector) = LMM(FeTerm(ones((length(y),1))),map(reterm,p),convert(Array,y))

function tune(lmm::LMM)
    r = copy(lmm.y)
    BLAS.gemv!('N',-1.,lmm.X.X,(lmm.X.X\lmm.y),1.,r)
    [optimize(λ -> objective!(re, λ, r), 0., 2.; abs_tol=0.05).minimum for re in lmm.re]
end
