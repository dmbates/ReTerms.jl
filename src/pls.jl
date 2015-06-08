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
            R[i,i] = isdiag(mm) ? PDiagMat(diag(mm)) : PDMat(full(mm))
        else
            R[i,j] = copy(pr)
        end
    end
    LMM(trms,A,R,lower,[x == 0. ? 1. : 0. for x in lower],cumsum(vcat(1,map(npar,rev))))
end

LMM(X::AbstractMatrix,re::Vector,y::DataVector) = LMM(X,re,convert(Array,y))
LMM(re::Vector,y::DataVector) = LMM(ones(length(y),1),re,convert(Array,y))
LMM(re::Vector,y::Vector) = LMM(ones((length(y),1)),re,y)

function setpars!(lmm::LMM,pars::Vector{Float64})
    all(pars .>= lmm.lower) || error("elements of pars violate bounds")
    copy!(lmm.pars,pars)
    gp = lmm.gp
    A = lmm.A
    R = lmm.R
    nt = size(A,2)                      # total number of terms
    for j in 1:nt, i in 1:j             # initialize R to a copy of A
        copy!(R[i,j],A[i,j])
    end
    ## set parameters in r.e. terms and initialize diagonal blocks of R
    for i in 1:(nt-2)
        ti = lmm.trms[i]
        setpars!(ti,sub(pars,gp[i]:(gp[i+1]-1)),lmm.A[i,i],lmm.R[i,i])
        for j in (i+1):size(lmm.R,2)    # scale the ith row by λ
            scale!(lmm.R[i,j],ti,lmm.A[i,j])
        end
        for k in 1:(i-1)                # scale the ith column by λ, downdate diagonal block
            scale!(lmm.R[k,i],ti)
        end
    end
    for j in 1:nt
        for i in 1:(j-1)
            downdate!(R[j,j],R[i,j])
        end
        @show R[j,j]
        # factor!(R[j,j])  # also adds the identity
        # whiten blocks to the right
    end
    lmm
end

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
