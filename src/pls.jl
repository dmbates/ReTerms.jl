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
            R[i,i] = isdiag(mm) && size(mm,1) > 1 ? Diagonal(diag(mm)) : full(mm)
        else
            R[i,j] = copy(pr)
        end
    end
    LMM(trms,A,R,lower,[x == 0. ? 1. : 0. for x in lower],cumsum(vcat(1,map(npar,rev))))
end

LMM(X::AbstractMatrix,re::Vector,y::DataVector) = LMM(X,re,convert(Array,y))
LMM(re::Vector,y::DataVector) = LMM(ones(length(y),1),re,convert(Array,y))
LMM(re::Vector,y::Vector) = LMM(ones((length(y),1)),re,y)

## Slightly modified from the version in julia/base/linalg/cholesky.jl

function chol!{T}(A::AbstractMatrix{T}, ::Type{Val{:U}})
    n = Base.LinAlg.chksquare(A)
    @inbounds begin
        for k = 1:n
            for i = 1:k - 1
                downdate!(A[k,k],A[i,k])
            end
            Akk = chol!(A[k,k], Val{:U})
            A[k,k] = Akk
            AkkInv = inv(Akk')
            for j = k + 1:n
                for i = 1:k - 1
                    downdate!(A[k,j],A[i,k],A[i,j])
                end
                A[k,j] = AkkInv*A[k,j]
            end
        end
    end
    return UpperTriangular(A)
end

chol!(D::Diagonal,::Type{Val{:U}}) = map!(D.diag,chol!)

function setpars!(lmm::LMM,pars::Vector{Float64})
    all(pars .>= lmm.lower) || error("elements of pars violate bounds")
    copy!(lmm.pars,pars)
    gp = lmm.gp
    R = copy!(lmm.R,lmm.A)              # initialize R to a copy of A
    nt = size(R,2)                      # total number of terms
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
        R[j,j] += I
    end
    chol!(R,Val{:U})
    lmm
end

factorize!(d::PDiagMat) = map!(inv,d.inv_diag,d.diag)
    
downdate!(d::PDMat,r::Matrix{Float64}) = BLAS.syrk!('U','T',-1.0,r,1.0,d.mat)

function downdate!(d::PDMat,r::SparseMatrixCSC{Float64})
    m,n = size(r)
    size(d,2) == n || throw(DimensionMismatch(""))
    rcp = r.colptr
    rrv = r.rowval
    rnz = r.nzval
    m = d.mat
    for j in 1:n
        for k in rcp[j]:(rcp[j+1]-1)    # downdate the diagonal
            m[j,j] -= abs2(rnz(k))
        end
        for i in 1:(j-1)                # downdate the off-diagonals
            ki = rcp[i]
            kj = rcp[j]
            while (ki < rcp[i+1] && kj < rcp[j+i])
                if rrv[ki] < rrv[kj]
                    ki += 1
                    next
                elseif rrv[ki] > rrv[kj]
                    kj += 1
                    next
                else
                    m[i,j] -= rnz[ki] * rnz[kj]
                    ki += 1
                    kj += 1
                end
            end
        end
    end
    d.mat
end

function downdate!(d::PDiagMat,r::SparseMatrixCSC{Float64})
    n = size(r,2)
    dd = d.diag
    length(dd) == n || throw(DimensionMismatch(""))
    rcp = r.colptr
    rnz = r.nzval
    for j in 1:n
        for k in rcp[j]:(rcp[j+1]-1)
            dd[j] -= abs2(rnz[k])
        end
    end
    d.diag
end

function downdate!(d::PDiagMat,m::DenseMatrix{Float64})
    r,s = size(m)
    dd = d.diag
    s == length(dd) == 1 || error("method only make sense for 1 by 1 PDiagMat")
    dd[1] -= sum(abs2,m)
    d
end

Base.LinAlg.chol!(D::Diagonal) = map!(D.diag,chol!)

Base.copy!(pd::PDiagMat,d::Diagonal{Float64}) = (copy!(pd.diag,d.diag);fill!(pd.inv_diag,NaN);pd)

function Base.copy!(pd::PDMat,d::Diagonal{Float64})
    size(pd) == size(d) || throw(DimensionMismatch(""))
    pd.chol.uplo == 'U' || error("improperly formed PDMat, chol should be upper triangular")
    mm = pd.mat
    rr = pd.chol.factors
    fill!(mm,0.)
    dd = d.diag
    n = length(dd)
    for j in 1:n
        mm[j,j] = dd[j]
        for i in 1:j
            rr[i,j] = NaN
        end
    end
    pd
end

Base.copy!(pd::PDMat,m::DenseMatrix{Float64}) = (copy!(pd.mat,m);pd)

function Base.copy!(pd::PDiagMat,m::DenseMatrix{Float64})
    dd = pd.diag
    di = pd.inv_diag
    r,s = size(m)
    r == s == length(dd) && isdiag(m) || throw(DimensionMismatch(""))
    for j in 1:r
        dd[j] = m[j,j]
        di[j] = NaN
    end
    pd
end
