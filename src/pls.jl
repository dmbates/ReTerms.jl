type LMM <: StatsBase.RegressionModel
    trms::Vector
    Λ::Vector
    A::Matrix        # symmetric cross-product blocks (upper triangle)
    R::Matrix        # right Cholesky factor in blocks.
    fit::Bool
end

function LMM(Rem::Vector, Λ::Vector, X::AbstractMatrix, y::Vector)
    all(x->isa(x,AbstractReMat),Rem) ||
        throw(ArgumentError("Elements of Rem should be AbstractReMat's"))
    all(x->isa(x,ParamLowerTriangular),Λ) ||
        throw(ArgumentError("Elements of Λ should be ParamLowerTriangular"))
    n,p = size(X)
    all(t -> size(t,1) == n,Rem) && length(y) == n || throw(DimensionMismatch("n not consistent"))
    nreterms = length(Rem)
    length(Λ) == nreterms && all([chksz(Rem[i],Λ[i]) for i in 1:nreterms]) ||
        throw(DimensionMismatch("Rem and Λ"))
    nt = nreterms + 1
    trms = Array(Any,nt)
    for i in eachindex(Rem) trms[i] = Rem[i] end
    trms[end] = hcat(X,y)
    A = fill!(Array(Any,(nt,nt)),nothing)
    R = fill!(Array(Any,(nt,nt)),nothing)
    for j in 1:nt, i in 1:j
        A[i,j] = densify(trms[i]'trms[j])
        R[i,j] = copy(A[i,j])
    end
#    A[end,end] = Symmetric(triu(A[end,end]),:U)
#    R[end,end] = cholfact!(R[end,end])
    LMM(trms,Λ,A,R,false)
end

LMM(re::Vector,Λ::Vector,X::AbstractMatrix,y::DataVector) = LMM(re,Λ,X,convert(Array,y))

LMM(re::Vector,X::DenseMatrix,y::DataVector) = LMM(re,map(LT,re),X,convert(Array,y))

LMM(g::PooledDataVector,y::DataVector) = LMM([ReMat(g)],y)

LMM(re::Vector,y::DataVector) = LMM(re,ones(length(y),1),y)

chksz(A::ReMat,λ::ParamLowerTriangular) = size(λ,1) == 1
chksz(A::VectorReMat,λ::ParamLowerTriangular) = size(λ,1) == size(A.z,1)

lowerbd(A::LMM) = mapreduce(lowerbd,vcat,A.Λ)

Base.getindex(m::LMM,s::Symbol) = mapreduce(x->x[s],vcat,m.Λ)

function Base.setindex!(m::LMM,v::Vector,s::Symbol)
    s == :θ || throw(ArgumentError("only ':θ' is meaningful for assignment"))
    lam = m.Λ
    length(v) == sum(nθ,lam) || throw(DimensionMismatch("length(v) should be $(sum(nθ,lam))"))
    A = m.A
    R = m.R
    n = size(A,1)                       # inject upper triangle of A into L
    for j in 1:n, i in 1:j
        inject!(R[i,j],A[i,j])
    end
    offset = 0
    for i in eachindex(lam)
        li = lam[i]
        nti = nθ(li)
        li[:θ] = sub(v,offset + (1:nti))
        offset += nti
        for j in i:size(R,2)
            scale!(li,R[i,j])
        end
        for ii in 1:i
            scale!(R[ii,i],li)
        end
        inflate!(R[i,i])
    end
    cfactor!(R)
end

"""
`fit(m)` -> `m`

Optimize the objective using an NLopt optimizer.
"""
function StatsBase.fit(m::LMM, verbose::Bool=false, optimizer::Symbol=:default)
    m.fit && return m
    th = m[:θ]
    k = length(th)
    if optimizer == :default
        optimizer = hasgrad(m) ? :LD_MMA : :LN_BOBYQA
    end
    opt = NLopt.Opt(optimizer, k)
    NLopt.ftol_rel!(opt, 1e-12)   # relative criterion on deviance
    NLopt.ftol_abs!(opt, 1e-8)    # absolute criterion on deviance
    NLopt.xtol_abs!(opt, 1e-10)   # criterion on parameter value changes
    NLopt.lower_bounds!(opt, lowerbd(m))
    feval = 0
    geval = 0
    function obj(x::Vector{Float64}, g::Vector{Float64})
        feval += 1
        m[:θ] = x
        val = objective(m)
        if length(g) == length(x)
            geval += 1
            grad!(g,m)
        end
        val
    end
    if verbose
        function vobj(x::Vector{Float64}, g::Vector{Float64})
            feval += 1
            m[:θ] = x
            val = objective(m)
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

grad!(v,lmm::LMM) = v

hasgrad(lmm::LMM) = false

"Add an identity matrix to the argument, in place"
inflate!(D::Diagonal{Float64}) = (d = D.diag; for i in eachindex(d) d[i] += 1 end; D)

function inflate!(A::LowerTriangular{Float64})
    n = Base.LinAlg.chksquare(A)
    for i in 1:n
        @inbounds A[i,i] += 1
    end
    A
end


Base.logdet(lmm::LMM) = 2.*mapreduce(logdet,(+),diag(lmm.R)[1:end-1])

"Negative twice the log-likelihood"
function objective(lmm::LMM)
    n = size(lmm.trms[1],1)
    logdet(lmm) + n*(1.+log(2π*abs2(lmm.R[end,end][end,end])/n))
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

function Base.LinAlg.Ac_ldiv_B!{T<:FloatingPoint}(A::HBlkDiag,B::DenseMatrix{T})
    m,n = size(B)
    aa = A.arr
    r,s,k = size(aa)
    m == Base.LinAlg.chksquare(A) || throw(DimensionMismatch())
    scr = Array(T,(r,n))
    for i in 1:k
        bb = sub(B,(i-1)*r+(1:r),:)
        copy!(bb,Base.LinAlg.Ac_ldiv_B!(UpperTriangular(sub(aa,:,:,i)),copy!(scr,bb)))
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

function Base.logdet{T<:FloatingPoint}(R::HBlkDiag{T})
    ra = R.arr
    ret = zero(T)
    r,s,k = size(ra)
    for i in 1:k
        ret += logdet(UpperTriangular(sub(ra,:,:,i)))
    end
    ret
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

function Base.LinAlg.A_rdiv_B!(A::StridedVecOrMat,D::Diagonal)
    m, n = size(A, 1), size(A, 2)
    if n != length(D.diag)
        throw(DimensionMismatch("diagonal matrix is $(length(D.diag)) by $(length(D.diag)) but left hand side has $n columns"))
    end
    (m == 0 || n == 0) && return A
    dd = D.diag
    for j = 1:n
        dj = dd[j]
        if dj == 0
            throw(SingularException(j))
        end
        for i = 1:m
            A[i,j] /= dj
        end
    end
    A
end

Base.LinAlg.A_rdiv_Bc!(A::StridedVecOrMat,D::Diagonal) = A_rdiv_B!(A,D)
