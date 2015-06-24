type LMM <: StatsBase.RegressionModel
    trms::Vector{Any}
    A::Matrix{Any}   # symmetric cross-product blocks (lower triangle)
    L::LowerTriangular          # left Cholesky factor in blocks.
    lower::Vector{Float64}      # vector of lower bounds on parameters
    pars::Vector{Float64}       # current parameter vector
    gp::Vector
    fit::Bool
end

function LMM(X::AbstractMatrix, rev::Vector, y::Vector)
    n,p = size(X)
    all(t -> size(t,1) == n, rev) && length(y) == n || throw(DimensionMismatch(""))
    lower = mapreduce(lowerbd,vcat,rev)
    nt = length(rev) + 1
    trms = Array(Any,nt)
    for i in eachindex(rev) trms[i] = rev[i] end
    trms[end] = hcat(X,y)
    A = fill!(Array(Any,(nt,nt)),nothing)
    L = LowerTriangular(fill!(Array(Any,(nt,nt)),nothing))
    for j in 1:nt, i in j:nt
        A[i,j] = densify(trms[i]'trms[j])
    end
    for i in 1:nt              # first col is sparse, others are dense
        L[i,1] = copy(A[i,1])
    end
    for k in 2:nt
        L[k,k] = LowerTriangular(tril(full(copy(A[k,k]))))
        ## if isdiag(L[k,k]) # factor k is nested in previous factors
        ##     L[k,k] = Diagonal(diag(L[k,k]))
        ##     for i in (k + 1):nt
        ##         L[i,k] = copy(A[i,k])
        ##     end
        ## else
            for i in (k + 1):nt
                L[i,k] = full(copy(A[i,k]))
            end
        ## end
        ## for j in (k + 1):nt
        ##     for i in 1:(k-1)
        ##         downdate!(L[k,j],L[i,k],L[i,j])
        ##     end
        ## end
    end
    A[end,end] = LowerTriangular(tril(A[end,end]))
    pars = [x == 0. ? 1. : 0. for x in lower]
#    LMM(trms,A,L,lower,pars,cumsum(vcat(1,map(npar,rev))),false)
    setpars!(LMM(trms,A,L,lower,pars,cumsum(vcat(1,map(npar,rev))),false),pars)
end

LMM(X::AbstractMatrix,re::Vector,y::DataVector) = LMM(X,re,convert(Array,y))
LMM(re::Vector,y::DataVector) = LMM(ones(length(y),1),re,convert(Array,y))
LMM(re::Vector,y::Vector) = LMM(ones(length(y),1),re,y)
LMM(re::Vector{Symbol},y::Symbol,df) = LMM([reterm(df[s]) for s in re],df[y])

## Slightly modified version of chol! from julia/base/linalg/cholesky.jl

function cfactor!(A::AbstractMatrix)
    n = Base.LinAlg.chksquare(A)
    @inbounds begin
        for k = 1:n
            for j in 1:(k - 1)
                downdate!(A[k,k],A[k,j])  # A[k,k] -= A[k,j]*A[k,j]'
            end
            cfactor!(A[k,k])   # (lower) Cholesky factor of A[k,k]
            for i in (k + 1):n
                for j in 1:(k - 1)
                    downdate!(A[i,k],A[i,j],A[k,j]) # A[i,k] -= A[i,j]*A[k,j]
                end
                Base.LinAlg.A_rdiv_Bc!(A[i,k],A[k,k])
            end
        end
    end
    return LowerTriangular(A)
end

cfactor!(x::Number) = sqrt(real(x))
cfactor!(D::Diagonal) = (map!(cfactor!,D.diag); D)
cfactor!(L::LowerTriangular{Float64}) = Base.LinAlg.chol!(L.data,Val{:L})

@doc "Subtract, in place, AA' or AB' from C"->
downdate!(C::LowerTriangular{Float64},A::DenseMatrix{Float64}) =
    BLAS.syrk!('L','N',-1.0,A,1.0,C.data)
downdate!(C::DenseMatrix{Float64},A::DenseMatrix{Float64},B::DenseMatrix{Float64}) =
    BLAS.gemm!('N','T',-1.0,A,B,1.0,C)
function downdate!(C::Diagonal{Float64},A::SparseMatrixCSC{Float64,BlasInt})
    m,n = size(A)
    dd = C.diag
    length(dd) == n || throw(DimensionMismatch(""))
    nz = nonzeros(A)
    for j in eachindex(dd)
        for k in nzrange(A,j)
            @inbounds dd[j] -= abs2(nz[k])
        end
    end
    C
end
if Base.blas_vendor() == :mkl
    function downdate!(C::DenseMatrix{Float64},A::SparseMatrixCSC{Float64,BlasInt},B::DenseMatrix{Float64})
        ma,na = size(A); mb,nb = size(B); mc,nc = size(C)
        mc == na && nc == nb && ma == mb || throw(DimensionMismatch(""))
        ccall((:mkl_dcscmm,:libmkl_rt),Void,
              (Ptr{UInt8},Ref{BlasInt},Ref{BlasInt},Ref{BlasInt},
               Ref{Float64},Ptr{UInt8},Ptr{Float64},Ptr{BlasInt},Ptr{BlasInt},Ptr{BlasInt},
               Ptr{Float64},Ref{BlasInt},
               Ref{Float64},Ptr{Float64},Ref{BlasInt}),
              "T",mc,nc,mb,
              -1.0,"G**F",nonzeros(A),rowvals(A),A.colptr,pointer(A.colptr,2),
              B,nb,
              1.0,C,nc)
        C
    end

    function downdate!(C::DenseMatrix{Float64},A::SparseMatrixCSC{Float64,BlasInt},B::SparseMatrixCSC{Float64,BlasInt})
        ma,na = size(A)
        mb,nb = size(B)
        ma == size(C,1) && mb == size(C,2) && na == nb || throw(DimensionMismatch(""))
        cc = similar(C)
        ccall((:mkl_dcsrmultd,:libmkl_rt),Void,
              (Ref{UInt8},Ref{BlasInt},Ref{BlasInt},Ref{BlasInt},
               Ptr{Float64},Ptr{BlasInt},Ptr{BlasInt},
               Ptr{Float64},Ptr{BlasInt},Ptr{BlasInt},
               Ptr{Float64},Ref{BlasInt}),
              "T",na,ma,nb,
              nonzeros(A),rowvals(A),A.colptr,
              nonzeros(B),rowvals(B),B.colptr,
              cc,ma)
        map!(-,C,C,cc)
    end

    function downdate!(C::LowerTriangular{Float64},A::SparseMatrixCSC{Float64,BlasInt})
        m,n = size(A)
        m == size(C,1) || throw(DimensionMismatch(""))
        cd = C.data
        cc = similar(cd)
        ccall((:mkl_dcsrmultd,:libmkl_rt),Void,
              (Ptr{UInt8},Ref{BlasInt},Ref{BlasInt},Ref{BlasInt},
               Ptr{Float64},Ptr{BlasInt},Ptr{BlasInt},
               Ptr{Float64},Ptr{BlasInt},Ptr{BlasInt},
               Ptr{Float64},Ref{BlasInt}),
              "T",n,m,n,
              nonzeros(A),rowvals(A),A.colptr,
              nonzeros(A),rowvals(A),A.colptr,
              cc,m)
        for j in 1:m, i in j:m
            cd[i,j] -= cc[i,j]
        end
        C
    end

    function downdate!{T<:Float64}(C::DenseMatrix{T},A::DenseMatrix{T},B::SparseMatrixCSC{T,BlasInt})
        ma,na = size(A)
        mb,nb = size(B)
        ma == size(C,1) && mb == size(C,2) && na == nb || throw(DimensionMismatch(""))
        rvb = rowvals(B); nzb = nonzeros(B)
        for j in 1:nb
            ptA = pointer(A,1+(j-1)*ma)
            ib = nzrange(B,j)
            rvbj = sub(rvb,ib)
            nzbj = sub(nzb,ib)
            for k in eachindex(ib)
                BLAS.axpy!(ma,-nzbj[k],ptA,1,pointer(C,1+(rvbj[k]-1)*ma),1)
            end
        ptA += ma
        end
        C
    end
else
    function downdate!(C::DenseMatrix{Float64},A::SparseMatrixCSC{Float64,BlasInt},B::DenseMatrix{Float64})
        m,n = size(A)
        r,s = size(C)
        r == n && s == size(B,2) && m == size(B,1) || throw(DimensionMismatch(""))
        nz = nonzeros(A)
        rv = rowvals(A)
        for jj in 1:s, j in 1:n, k in nzrange(A,j)
            C[j,jj] -= nz[k]*B[rv[k],jj]
        end
        C
    end
    
    function downdate!{T<:FloatingPoint}(C::DenseMatrix{T},A::SparseMatrixCSC{T},B::SparseMatrixCSC{T})
        ma,na = size(A)
        mb,nb = size(B)
        ma == size(C,1) && mb == size(C,2) && na == nb || throw(DimensionMismatch(""))
        rva = rowvals(A); nza = nonzeros(A); rvb = rowvals(B); nzb = nonzeros(B)
        for j in 1:nb
            ia = nzrange(A,j)
            ib = nzrange(B,j)
            rvaj = sub(rva,ia)
            rvbj = sub(rvb,ib)
            nzaj = sub(nza,ia)
            nzbj = sub(nzb,ib)
            for k in eachindex(ib)
                krv = rvbj[k]
                knz = nzbj[k]
                for i in eachindex(ia)
                    @inbounds C[rvaj[i],krv] -= nzaj[i]*knz
                end
        end
        end
        C
    end

    function downdate!(C::LowerTriangular{Float64},A::SparseMatrixCSC{Float64,BlasInt})
        m,n = size(A)
        m == size(A,1) || throw(DimensionMismatch(""))
        rv = rowvals(A)
        nz = nonzeros(A)
        cc = C.data
        for k in 1:n
            rng = nzrange(A,k)
            nzk = sub(nz,rng)
            rvk = sub(rv,rng)
            for j in eachindex(rng)
                nzkj = nzk[j]
                rvkj = rvk[j]
                for i in j:length(rng)
                    @inbounds cc[rvk[i],rvkj] -= nzkj*nzk[i]
                end
            end
        end
        C
    end

    function downdate!{T<:Float64}(C::DenseMatrix{T},A::DenseMatrix{T},B::SparseMatrixCSC{T,BlasInt})
        ma,na = size(A)
        mb,nb = size(B)
        ma == size(C,1) && mb == size(C,2) && na == nb || throw(DimensionMismatch(""))
        rvb = rowvals(B); nzb = nonzeros(B)
        for j in 1:nb
            ptA = pointer(A,1+(j-1)*ma)
            ib = nzrange(B,j)
            rvbj = sub(rvb,ib)
            nzbj = sub(nzb,ib)
            for k in eachindex(ib)
                BLAS.axpy!(ma,-nzbj[k],ptA,1,pointer(C,1+(rvbj[k]-1)*ma),1)
            end
            ptA += ma
        end
        C
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
function inflate!(A::LowerTriangular{Float64})
    n = Base.LinAlg.chksquare(A)
    for i in 1:n
        @inbounds A[i,i] += 1.
    end
    A
end

@doc "`copy!` allowing for heterogeneous matrix types"
inject!(d,s) = copy!(d,s)
function inject!(d::LowerTriangular,s::LowerTriangular)
    (n = size(s,2)) == size(d,2) || throw(DimensionMismatch(""))
    for j in 1:n
        copy!(sub(d,j:n,j),sub(s,j:n,j))
    end
    d
end
function inject!(d::AbstractMatrix{Float64}, s::Diagonal{Float64})
    sd = s.diag
    length(sd) == size(d,2) || throw(DimensionMismatch(""))
    fill!(d,0.)
    @inbounds for i in eachindex(sd)
        d[i,i] = sd[i]
    end
    d
end
inject!(d::Diagonal{Float64},s::Diagonal{Float64}) = (copy!(d.diag,s.diag);d)
function inject!(d::SparseMatrixCSC{Float64},s::SparseMatrixCSC{Float64})
    m,n = size(d)
    size(d) == size(s) || throw(DimensionMismatch(""))
    if nnz(d) == nnz(s)
        copy!(nonzeros(d),nonzeros(s))
        return d
    end
    drv = rowvals(d); srv = rowvals(s); dnz = nonzeros(d); snz = nonzeros(s)
    fill!(dnz,0.)
    for j in 1:n
        dnzr = nzrange(d,j)
        dnzrv = sub(drv,dnzr)
        snzr = nzrange(s,j)
        if length(snzr) == length(dnzr) && all(dnzrv .== sub(srv,snzr))
            copy!(sub(dnz,dnzr),sub(snz,snzr))
        else
            for k in snzr
                ssr = srv[k]
                kk = searchsortedfirst(dnzrv,ssr)
                kk > length(dnzrv) || dnzrv[kk] != ssr || error("cannot inject sparse s into sparse d")
                dnz[dnzr[kk]] = snz[k]
            end
        end
    end
    d
end

Base.logdet(lmm::LMM) = 2.*mapreduce(logdet,(+),diag(lmm.L)[1:end-1])

lower(lmm::LMM) = lmm.lower

@doc "Negative twice the log-likelihood"->
function objective(lmm::LMM)
    n = size(lmm.trms[1],1)
    logdet(lmm) + n*(1.+log(2π*abs2(lmm.L[end,end][end,end])/n))
end

@doc "Install new parameter values.  Update `trms` and the Cholesky factor `L`"->
function setpars!(lmm::LMM,pars::Vector{Float64})
    all(pars .>= lmm.lower) || error("elements of pars violate bounds")
    copy!(lmm.pars,pars)
    gp = lmm.gp
    nt = length(lmm.trms)               # number of terms
    L = lmm.L
    A = lmm.A
    for j in 1:nt, i in j:nt
        inject!(L[i,j],A[i,j])
    end
    ## set parameters in r.e. terms, scale rows and columns, add identity
    for j in 1:(nt-1)
        tj = lmm.trms[j]
        setpars!(tj,sub(pars,gp[j]:(gp[j+1]-1)))
        for i in j:nt                   # scale the jth column by λ'
            scale!(L[i,j],tj)
        end
        for jj in 1:j                   # scale the jth row by λ
            scale!(tj,L[j,jj])
        end
        inflate!(L[j,j])                # L[j,j] += I
    end
    cfactor!(L)
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

function Base.LinAlg.Ac_ldiv_B!{T<:Float64}(D::Diagonal{T},B::SparseMatrixCSC{T,BlasInt})
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

function Base.logdet(t::LowerTriangular)
    n = Base.LinAlg.chksquare(t)
    mapreduce(log,(+),diag(t))
end

function Base.LinAlg.A_rdiv_Bc!{T<:FloatingPoint}(A::SparseMatrixCSC{T},B::Diagonal{T})
    m,n = size(A)
    dd = B.diag
    n == length(dd) || throw(DimensionMismatch(""))
    nz = nonzeros(A)
    for j in eachindex(dd)
        @inbounds scale!(sub(nz,nzrange(A,j)),inv(dd[j]))
    end
    A
end

function Base.LinAlg.A_rdiv_Bc!{T<:FloatingPoint}(A::Matrix{T},B::Diagonal{T})
    m,n = size(A)
    dd = B.diag
    n == length(dd) || throw(DimensionMismatch(""))
    for j in eachindex(dd)
        @inbounds scale!(sub(A,:,j),inv(dd[j]))
    end
    A
end
