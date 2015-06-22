type SimpleScalarReTerm <: ReTerm
    f::PooledDataVector
    λ::Float64
end

Base.size(t::SimpleScalarReTerm) = (length(t.f.refs),length(t.f.pool))
Base.size(t::SimpleScalarReTerm,i) = size(t)[i]

function Base.A_mul_B!(r::DenseVecOrMat, t::SimpleScalarReTerm, v::DenseVecOrMat)
    n,q = size(t)
    k = size(v,2)
    size(r,1) == n && size(v,1) == q && size(r,2) == k || throw(DimensionMismatch(""))
    rr = t.f.refs
    if k == 1
        r = v[rr]
    else
        for j in 1:k
            sub(r,:,j) = sub(v,:,rr)
        end
    end
    scale!(r,t.λ)
end

function *(t::SimpleScalarReTerm, v::DenseVecOrMat)
    k = size(t,1)
    A_mul_B!(Array(Float64, isa(v,Vector) ? (k,) : (k,size(v,2))), t, v)
end

function Base.Ac_mul_B!(r::DenseVecOrMat, t::SimpleScalarReTerm, v::DenseVecOrMat)
    n,q = size(t)
    k = size(v,2)
    size(r,1) == q && size(v,1) == n && size(r,2) == k || throw(DimensionMismatch(""))
    fill!(r,zero(eltype(r)))
    rr = t.f.refs
    if k == n1
        for i in 1:n
            @inbounds r[rr[i]] += v[i]
        end
    else
        for j in 1:k, i in 1:n
            @inbounds r[rr[i],j] += v[i,j]
        end
    end
    scale!(r,t.λ)
end

function Base.Ac_mul_B(t::SimpleScalarReTerm, v::DenseVecOrMat{Float64})
    k = size(t,2)
    Ac_mul_B!(Array(Float64, isa(v,Vector) ? (k,) : (k, size(v,2))), t, v)
end

type ScalarReTerm <: ReTerm
    f::PooledDataVector                 # grouping factor
    z::Vector{Float64}
    λ::Float64
end

function ScalarReTerm(f::PooledDataVector, z::Vector{Float64})
    length(f) == length(z) || throw(DimensionMismatch(""))
    rr = f.refs
    ScalarReTerm(f, z, one(Float64))
end

Base.size(t::ScalarReTerm) = (length(t.z), length(t.f.pool))
Base.size(t::ScalarReTerm,i::Integer) =
    i < 1 ? throw(BoundsError()) :
    i == 1 ? length(t.z) :
    i == 2 ? length(t.f.pool) : 1

function Base.A_mul_B!(r::DenseVecOrMat, t::ScalarReTerm, v::DenseVecOrMat)
    n,q = size(t)
    k = size(v,2)
    size(r,1) == n && size(v,1) == q && size(r,2) == k || throw(DimensionMismatch(""))
    tz = t.z
    rr = t.f.refs
    if k == 1
        for i in eachindex(r)
            @inbounds r[i] = tz[i] * v[rr[i]]
        end
    else
        for j in 1:k, i in 1:n
            @inbounds r[i,j] = tz[i] * v[rr[i],j]
        end
    end
    r
end

function *(t::ScalarReTerm, v::DenseVecOrMat)
    k = size(t,1)
    A_mul_B!(Array(Float64, isa(v,Vector) ? (k,) : (k,size(v,2))), t, v)
end

function Base.Ac_mul_B!(r::DenseVecOrMat, t::ScalarReTerm, v::DenseVecOrMat)
    n,q = size(t)
    k = size(v,2)
    size(r,1) == q && size(v,1) == n && size(r,2) == k || throw(DimensionMismatch(""))
    fill!(r,zero(eltype(r)))
    rr = t.f.refs
    zz = t.z
    if k == n1
        for i in 1:n
            @inbounds r[rr[i]] += v[i] * zz[i]
        end
    else
        for j in 1:k, i in 1:n
            @inbounds r[rr[i],j] += v[i,j] * zz[i]
        end
    end
    scale!(r,t.λ)
end

function Base.Ac_mul_B(t::ScalarReTerm, v::DenseVecOrMat{Float64})
    k = size(t,2)
    Ac_mul_B!(Array(Float64, isa(v,Vector) ? (k,) : (k, size(v,2))), t, v)
end

function Base.Ac_mul_B!(r::DenseVecOrMat, v::DenseVecOrMat, t::ScalarReTerm)
    n,q = size(t)
    k = size(v,2)
    size(r,2) == q && size(v,1) == n && size(r,1) == k || throw(DimensionMismatch(""))
    fill!(r, zero(eltype(r)))
    rr = t.f.refs
    zz = t.z
    if k == 1
        for i in 1:n
            @inbounds r[rr[i]] += v[i] * zz[i]
        end
    else
## FIXME Change this to use pointers, as in the downdate! method in pls.jl
        for j in 1:n
            BLAS.axpy!(zz[j],sub(v,j,:),sub(r,:,Int(rr[j])))
        end
    end
    t.λ == 1 ? r : scale!(r,t.λ)
end

Base.Ac_mul_B(v::DenseVecOrMat{Float64},t::ScalarReTerm) =
    Ac_mul_B!(Array(Float64,(size(v,2),size(t,2))), v, t)

function Base.Ac_mul_B(t::ScalarReTerm, s::ScalarReTerm)
    if is(s,t)
        crprd = zeros(length(s.f.pool))
        z = s.z
        rr = s.f.refs
        for i in eachindex(z)
            crprd[rr[i]] += abs2(z[i])
        end
        return Diagonal(crprd)
    end
    sparse(convert(Vector{Int32},t.f.refs),convert(Vector{Int32},s.f.refs),t.z .* s.z)
end

lowerbd(t::ScalarReTerm) = zeros(Float64,1)

setpars!(t::ScalarReTerm,x) = (t.λ = convert(Float64,x[1]); t)

Base.scale!(t::ScalarReTerm,m::Diagonal) = (scale!(t.λ,m.diag); m)

Base.scale!(m::Diagonal,t::ScalarReTerm) = scale!(t,m)

Base.scale!(t::ScalarReTerm, m::AbstractMatrix{Float64}) = scale!(t.λ,m)

function Base.scale!(x::Number,t::UpperTriangular{Float64})
    m,n = size(t)
    for j in 1:n, i in 1:j
        @inbounds t[i,j] *= x
    end
end
function Base.scale!{T<:Number}(s::T,t::LowerTriangular{T})
    m,n = size(t)
    for j in 1:n, i in j:m
        @inbounds t.data[i,j] *= s
    end
    t
end

Base.scale!(m::AbstractMatrix{Float64}, t::ScalarReTerm) = scale!(t.λ,m)

Base.logdet(t::ScalarReTerm) = sum(Base.LogFun(), t.plsdiag)

npar(t::ScalarReTerm) = 1

Base.scale!(r::DenseVecOrMat{Float64},t::ScalarReTerm,a::DenseVecOrMat{Float64}) = scale!(t.λ,copy!(r,a))

function Base.scale!(r::SparseMatrixCSC,t::ScalarReTerm,a::SparseMatrixCSC)
    size(r) == size(a) && nnz(r) == nnz(a) || throw(DimensionMismatch(""))
    λ = t.λ
    rcp = r.colptr
    acp = a.colptr
    rrv = r.rowval
    arv = a.rowval
    rnz = r.nzval
    anz = a.nzval
    for j in 1:size(a,2)
        rcp[j+1] = acp[j+1]
        for k in acp[j]:(acp[j+1]-1)
            rrv[k] = arv[k]
            rnz[k] = λ*anz[k]
        end
    end
    r
end

Base.scale!(r::DenseVecOrMat{Float64},t::ScalarReTerm) = scale!(r,t.λ)

function Base.scale!(r::SparseMatrixCSC{Float64},t::ScalarReTerm)
    scale!(r.nzval,t.λ)
    r
end
