type LMM <: StatsBase.RegressionModel
    trms::Vector{Any}
    A::Matrix{Any}
    R::Matrix{Any}
    lower::Vector{Float64}              # vector of lower bounds on parameters
    pars::Vector{Float64}               # current parameter vector
    gp::Vector
    fit::Bool
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
        pr = trms[i]'trms[j]
        if isdense_ish(pr)
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
    LMM(trms,A,R,lower,[x == 0. ? 1. : 0. for x in lower],cumsum(vcat(1,map(npar,rev))),false)
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

@doc "Subtract, in place, A'A or A'B from C"->
downdate!(C::UpperTriangular{Float64},A::DenseMatrix{Float64}) =
    BLAS.syrk!('U','T',-1.0,A,1.0,C.data)
downdate!(C::DenseMatrix{Float64},A::DenseMatrix{Float64},B::DenseMatrix{Float64}) =
    BLAS.gemm!('T','N',-1.0,A,B,1.0,C)
function downdate!{T<:FloatingPoint}(C::Diagonal{T},A::SparseMatrixCSC{T})
    m,n = size(A)
    dd = C.diag
    length(dd) == n || throw(DimensionMismatch(""))
    nz = nonzeros(A)
    for j in eachindex(dd)
        for k in nzrange(A,j)
            dd[j] -= abs2(nz[k])
        end
    end
    C
end
function downdate!{T<:FloatingPoint}(C::DenseMatrix{T},A::SparseMatrixCSC{T},B::DenseMatrix{T})
    m,n = size(A)
    r,s = size(C)
    r == n && s == size(B,2) && m == size(B,1) || throw(DimensionMismatch(""))
    nz = nonzeros(A)
    rv = rowvals(A)
    for jj in 1:s, j in 1:n, k in nzrange(A,j)
        C[j,jj] -= nz[k]*B[rv[k],jj]
    end
end

@doc "`fit(m)` -> m Optimize the objective using an NLopt optimizer"->
function StatsBase.fit(m::LMM, verbose::Bool=false, optimizer::Symbol=:default)
    m.fit && return m
    th = getpars(m); k = length(th)
    if optimizer == :default
        optimizer = hasgrad(m) ? :LD_MMA : :LN_BOBYQA
    end
    opt = NLopt.Opt(optimizer, k)
    NLopt.ftol_rel!(opt, 1e-12)   # relative criterion on deviance
    NLopt.ftol_abs!(opt, 1e-8)    # absolute criterion on deviance
    NLopt.xtol_abs!(opt, 1e-10)   # criterion on parameter value changes
    NLopt.lower_bounds!(opt, lower(m))
    feval = 0
    geval = 0
    function obj(x::Vector{Float64}, g::Vector{Float64})
        feval += 1
        val = objective(setpars!(m,x))
        if length(g) == length(x)
            geval += 1
            grad!(g,m)
        end
        val
    end
    if verbose
        function vobj(x::Vector{Float64}, g::Vector{Float64})
            feval += 1
            val = objective(setpars!(m,x))
            print("f_$feval: $(round(val,5)), [")
            showcompact(x[1])
            for i in 2:length(x) print(","); showcompact(x[i]) end
            println("]")
            if length(g) == length(x)
                geval += 1
                grad!(g,m)
            end
            val
        end
        NLopt.min_objective!(opt, vobj)
    else
        NLopt.min_objective!(opt, obj)
    end
    fmin, xmin, ret = NLopt.optimize(opt, th)
    ## very small parameter values often should be set to zero
    xmin1 = copy(xmin)
    modified = false
    for i in eachindex(xmin1)
        if 0. < abs(xmin1[i]) < 1.e-5
            modified = true
            xmin1[i] = 0.
        end
    end
        if modified && (ff = objective(setpars!(m,xmin1))) < fmin
            fmin = ff
            copy!(xmin,xmin1)
        end
#        m.opt = OptSummary(th,xmin,fmin,feval,geval,optimizer)
    if verbose println(ret) end
    m.fit = true
    m
end

getpars(lmm::LMM) = lmm.pars

grad!(v,lmm::LMM) = v

hasgrad(lmm::LMM) = false

@doc "Add an identity matrix to the argument, in place"->
inflate!(D::Diagonal{Float64}) = (d = D.diag; for i in eachindex(d) d[i] += 1. end; D)
function inflate!(A::UpperTriangular{Float64})
    n = Base.LinAlg.chksquare(A)
    for i in 1:n
        A[i,i] += 1.
    end
    A
end

@doc "`copy!` allowing for heterogeneous matrix types"
inject!(d,s) = copy!(d,s)
function inject!(d::AbstractMatrix{Float64}, s::Diagonal{Float64})
    fill!(d,0.)
    sd = s.diag
    for i in eachindex(sd)
        d[i,i] = sd[i]
    end
    d
end

Base.logdet(lmm::LMM) = 2.*mapreduce(logdet,(+),diag(lmm.R)[1:end-2])

lower(lmm::LMM) = lmm.lower

@doc "Negative twice the log-likelihood"
function objective(lmm::LMM)
    n = Float64(length(lmm.trms[end]))
    logdet(lmm) + n*(1.+log(2π*abs2(lmm.R[end,end][1,1])/n))
end

@doc "Install new parameter values.  Update `trms` and the Cholesky factor `R`"->
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

function Base.LinAlg.Ac_ldiv_B!{T<:FloatingPoint}(D::Diagonal{T},B::SparseMatrixCSC{T})
    m,n = size(B)
    dd = D.diag
    length(dd) == m || throw(DimensionMismatch(""))
    nzv = nonzeros(B)
    rv = rowvals(B)
    for j in 1:n, k in nzrange(B,j)
        nzv[k] /= dd[rv[k]]
    end
    B
end

Base.LinAlg.A_ldiv_B!{T<:FloatingPoint}(D::Diagonal{T},B::DenseMatrix{T}) =
    Base.LinAlg.Ac_ldiv_B!(D,B)

function Base.logdet(t::UpperTriangular)
    n = Base.LinAlg.chksquare(t)
    mapreduce(log,(+),diag(t))
end
