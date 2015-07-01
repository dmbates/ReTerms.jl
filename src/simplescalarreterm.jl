@doc """
A `simple scalar random-effects term` is a scalar term in which
the implicit matrix `Z'` is a row of 1's.
"""->
type SimpleScalarReTerm <: ReTerm
    f::PooledDataVector
    λ::Float64
end

SimpleScalarReTerm(f::PooledDataVector) = SimpleScalarReTerm(f,1.0)

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

#Base.scale!(m::AbstractMatrix{Float64}, t::SimpleScalarReTerm) = scale!(t.λ,m)

function Base.scale!(t::SimpleScalarReTerm,r::SparseMatrixCSC{Float64,BlasInt})
    scale!(t.λ,r.nzval)
    r
end

Base.scale!(d::Diagonal{Float64},t::SimpleScalarReTerm) = (scale!(d.diag,t.λ);d)

Base.scale!(m::Matrix{Float64},t::SimpleScalarReTerm) = scale!(m,t.λ)

function Base.scale!(l::LowerTriangular{Float64,Matrix{Float64}},t::SimpleScalarReTerm)
    n = size(l,2)
    tl = t.λ
    ld = l.data
    if tl != 1.0
        for j in 1:n, i in j:n
            @inbounds ld[i,j] *= tl
        end
    end
    l
end

function Base.Ac_mul_B(t::SimpleScalarReTerm, s::SimpleScalarReTerm)
    if is(s,t)
        crprd = zeros(length(s.f.pool))
        rr = s.f.refs
        for i in eachindex(rr)
            crprd[rr[i]] += 1.0
        end
        return Diagonal(crprd)
    end
    sparse(convert(Vector{BlasInt},t.f.refs),convert(Vector{BlasInt},s.f.refs),1.0)
end

function Base.Ac_mul_B!(r::DenseVecOrMat, v::DenseVecOrMat, t::SimpleScalarReTerm)
    n,q = size(t)
    k = size(v,2)
    size(r,2) == q && size(v,1) == n && size(r,1) == k || throw(DimensionMismatch(""))
    fill!(r, zero(eltype(r)))
    rr = t.f.refs
    if k == 1
        for i in 1:n
            @inbounds r[rr[i]] += v[i]
        end
    else
## FIXME Change this to use pointers, as in the downdate! method in pls.jl
        for j in 1:n
            BLAS.axpy!(1.,sub(v,j,:),sub(r,:,Int(rr[j])))
        end
    end
    t.λ == 1 ? r : scale!(r,t.λ)
end

## scale! with a scalar random-effects term is multiplication by a scalar
Base.scale!(t::SimpleScalarReTerm, m::AbstractMatrix{Float64}) = scale!(t.λ,m)

Base.scale!(t::SimpleScalarReTerm,m::Diagonal) = (scale!(t.λ,m.diag); m)

function Base.scale!(r::SparseMatrixCSC{Float64,BlasInt},t::SimpleScalarReTerm)
    scale!(r.nzval,t.λ)
    r
end

Base.Ac_mul_B(v::DenseVecOrMat{Float64},t::SimpleScalarReTerm) =
    Ac_mul_B!(Array(Float64,(size(v,2),size(t,2))), v, t)

npar(t::SimpleScalarReTerm) = 1

lowerbd(t::SimpleScalarReTerm) = zeros(Float64,1)

setpars!(t::SimpleScalarReTerm,x) = (t.λ = convert(Float64,x[1]); t)
