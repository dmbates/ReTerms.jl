type LMM <: StatsBase.RegressionModel
    trms::Vector{Any}
    A::Matrix{Any}
    R::Matrix{Any}
    lower::Vector{Float64}              # vector of lower bounds on parameters
    pars::Vector{Float64}               # current parameter vector
    gp::Vector
end

function LMM(X::AbstractMatrix, rev::Vector, y::Vector)
    n,p = size(X)
    all(t -> size(t,1) == n, rev) && length(y) == n || throw(DimensionMismatch(""))
    lower = mapreduce(lowerbd,vcat,rev)
    ntrms = length(rev) + 2
    trms = Array(Any,ntrms)
    for i in eachindex(rev) trms[i] = rev[i] end
    trms[end-1] = X
    trms[end] = reshape(convert(Vector{Float64},y),(n,1))
    A = Array(Any,(ntrms,ntrms))
    R = Array(Any,(ntrms,ntrms))
    for j in 1:(ntrms-1),i in (j+1):ntrms # assign lower triangle
        A[i,j] = R[i,j] = nothing
    end
    for j in 1:ntrms, i in 1:j
        pr = Ac_mul_B(trms[i],trms[j])
        if issparse(pr) && isdenseish(pr)
            pr = full(pr)
        end
        A[i,j] = pr
        if i == j
            mm = A[i,i] + I
            for k in 1:(i - 1)
                mm += A[k,i]'A[k,i]
            end
            R[i,i] = cfactor!(isdiag(mm) && size(mm,1) > 1 ? Diagonal(diag(mm)) : full(mm))
        else
            R[i,j] = copy(pr)
        end
    end
    LMM(trms,A,R,lower,[x == 0. ? 1. : 0. for x in lower],cumsum(vcat(1,map(npar,rev))))
end

LMM(X::AbstractMatrix,re::Vector,y::DataVector) = LMM(X,re,convert(Array,y))
LMM(re::Vector,y::DataVector) = LMM(ones(length(y),1),re,convert(Array,y))
LMM(re::Vector,y::Vector) = LMM(ones((length(y),1)),re,y)

## Slightly modified version of chol! from julia/base/linalg/cholesky.jl

function cfactor!(A::AbstractMatrix)
    n = Base.LinAlg.chksquare(A)
    @inbounds begin
        for k = 1:n
            for i = 1:k - 1
                downdate!(A[k,k],A[i,k])
            end
            cfactor!(A[k,k])
            for j = k + 1:n
                for i = 1:k - 1
                    downdate!(A[k,j],A[i,k],A[i,j])
                end
                Base.LinAlg.Ac_ldiv_B!(A[k,k],A[k,j])
            end
        end
    end
    return UpperTriangular(A)
end

cfactor!(A::DenseMatrix{Float64}) = Base.LinAlg.chol!(A)
cfactor!(x::Number) = sqrt(real(x))
cfactor!(D::Diagonal) = (map!(cfactor!,D.diag); D)
cfactor!(t::UpperTriangular{Float64}) = Base.LinAlg.chol!(t.data,Val{:U})


function setpars!(lmm::LMM,pars::Vector{Float64})
    all(pars .>= lmm.lower) || error("elements of pars violate bounds")
    copy!(lmm.pars,pars)
    gp = lmm.gp
    nt = length(lmm.trms)               # number of terms
    R = lmm.R
    A = lmm.A
    for j in 1:nt, i in 1:j
        inject!(R[i,j],A[i,j])
    end
    ## set parameters in r.e. terms, scale rows and columns, add identity
    for j in 1:(nt-2)
        tj = lmm.trms[j]
        setpars!(tj,sub(pars,gp[j]:(gp[j+1]-1)))
        for jj in j:nt                  # scale the jth row by λ'
            scale!(tj,R[j,jj])
        end
        for i in 1:j                    # scale the ith column by λ
            scale!(tj,R[i,j])
        end
        inflate!(R[j,j])                # R[j,j] += I
    end
    cfactor!(R)
    lmm
end

function Base.LinAlg.Ac_ldiv_B!{T<:FloatingPoint}(D::Diagonal{T},B::DenseMatrix{T})
    m,n = size(B)
    dd = D.diag
    length(dd) == m || throw(DimensionMismatch(""))
    for j in 1:n, i in 1:m
        B[i,j] /= dd[i]
    end
    B
end

Base.LinAlg.A_ldiv_B!{T<:FloatingPoint}(D::Diagonal{T},B::DenseMatrix{T}) =
    Base.LinAlg.Ac_ldiv_B!(D,B)

downdate!(C::UpperTriangular{Float64},A::DenseMatrix{Float64}) = BLAS.syrk!('U','T',-1.0,A,1.0,C.data)

downdate!(C::DenseMatrix{Float64},A::DenseMatrix{Float64},B::DenseMatrix{Float64}) =
    BLAS.gemm!('T','N',-1.0,A,B,1.0,C)

function inflate!(A::UpperTriangular{Float64})
    n = Base.LinAlg.chksquare(A)
    for i in 1:n
        A[i,i] += 1.
    end
    A
end

inflate!(D::Diagonal{Float64}) = (d = D.diag; for i in eachindex(d) d[i] += 1. end; D)

inject!(d,s) = copy!(d,s)

function inject!(d::AbstractMatrix{Float64}, s::Diagonal{Float64})
    fill!(d,0.)
    sd = s.diag
    for i in eachindex(sd)
        d[i,i] = sd[i]
    end
    d
end

function Base.logdet(t::UpperTriangular)
    n = Base.LinAlg.chksquare(t)
    mapreduce(log,(+),diag(t))
end

Base.logdet(lmm::LMM) = 2.*mapreduce(logdet,(+),diag(lmm.R)[1:end-2])

function objective(lmm::LMM)
    n = Float64(length(lmm.trms[end]))
    logdet(lmm) + n*(1.+log(2π*abs2(lmm.R[end,end][1,1])/n))
end

## objective(m) -> negative twice the log-likelihood
function objective(m::LinearMixedModel)
    n,p = size(m)
    REML = m.REML
    fn = @compat(Float64(n - (REML ? p : 0)))
    logdet(m,false) + fn*(1.+log(2π*pwrss(m)/fn)) + (REML ? logdet(m) : 0.)
end
