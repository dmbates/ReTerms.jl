type ScalarReTerm{T<:FloatingPoint} <: ReTerm{T}
    f::PooledDataVector                 # grouping factor
    z::Vector{T}
    λ::T
    crprdiag::Vector{T}
    plsdiag::Vector{T}
    plsdinv::Vector{T}
end

function ScalarReTerm{T<:FloatingPoint}(f::PooledDataVector, z::Vector{T})
    (n = length(f)) == length(z) || throw(DimensionMismatch(""))
    q = length(f.pool)
    crprd = zeros(T, q)
    for i in 1:n
        crprd[f.refs[i]] += abs2(z[i])
    end
    plsd = crprd .+ one(T)
    ScalarReTerm(f, z, one(T), crprd, plsd, [inv(x) for x in plsd])
end

ScalarReTerm(f::PooledDataVector) = ScalarReTerm(f, ones(length(f)))

Base.size(t::ScalarReTerm) = (length(t.z), length(t.f.pool))
Base.size(t::ScalarReTerm,i::Integer) =
    i < 1 ? throw(BoundsError()) :
    i == 1 ? length(t.z) :
    i == 2 ? length(t.f.pool) : 1

function Base.A_mul_B!(r::VecOrMat, t::ScalarReTerm, v::VecOrMat)
    n,q = size(t)
    k = size(v,2)
    size(r,1) == n && size(v,1) == q && size(r,2) == k || throw(DimensionMismatch(""))
    if k == 1
        for i in 1:n
            @inbounds r[i] = t.z[i] * v[t.f.refs[i]]
        end
    else
        for j in 1:k, i in 1:n
            @inbounds r[i,j] = t.z[i] * v[t.f.refs[i],j]
        end
    end
    scale!(r,t.λ)
end

function *{T<:FloatingPoint}(t::ScalarReTerm{T}, v::VecOrMat{T})
    k = size(t,1)
    A_mul_B!(Array(T, isa(v,Vector) ? (k,) : (k,size(v,2))), t, v)
end

function Base.Ac_mul_B!(r::VecOrMat, t::ScalarReTerm, v::VecOrMat)
    n,q = size(t)
    k = size(v,2)
    size(r,1) == q && size(v,1) == n && size(r,2) == k || throw(DimensionMismatch(""))
    fill!(r,zero(eltype(r)))
    if k == 1
        for i in 1:n
            @inbounds r[t.f.refs[i]] += v[i] * t.z[i]
        end
    else
        for j in 1:k, i in 1:n
            @inbounds r[t.f.refs[i],j] += v[i,j] * t.z[i]
        end
    end
    scale!(r,t.λ)
end

function Base.Ac_mul_B{T<:FloatingPoint}(t::ScalarReTerm{T}, v::VecOrMat{T})
    k = size(t,2)
    Ac_mul_B!(Array(T, isa(v,Vector) ? (k,) : (k, size(v,2))), t, v)
end

function Base.Ac_mul_B!(r::VecOrMat, v::VecOrMat, t::ScalarReTerm)
    n,q = size(t)
    k = size(v,2)
    size(r,2) == q && size(v,1) == n && size(r,1) == k || throw(DimensionMismatch(""))
    fill!(r, zero(eltype(r)))
    if k == 1
        for i in 1:n
            @inbounds r[t.f.refs[i]] += v[i] * t.z[i]
        end
    else
        for j in 1:k, i in 1:n
            @inbounds r[t.f.refs[i],j] += v[i,j] * t.z[i]
        end
    end
    scale!(r,t.λ)
end

Base.Ac_mul_B{T<:FloatingPoint}(v::VecOrMat{T},t::ScalarReTerm{T}) =
    Ac_mul_B!(Array(T, (size(v,2), size(t,2))), v, t)


function Base.Ac_mul_B{T<:FloatingPoint}(t::ScalarReTerm{T}, s::ScalarReTerm{T})
    is(s,t) && return PDiagMat(abs2(t.λ) .* t.crprdiag)
    (n = size(t,1)) == size(s,1) || throw(DimensionMismatch(""))
    scale!(t.λ * s.λ, sparse(convert(Vector{Int32},t.f.refs),convert(Vector{Int32},s.f.refs),t.z .* s.z))
end

lowerbd{T<:FloatingPoint}(t::ScalarReTerm{T}) = zeros(T,1)

function update!{T<:FloatingPoint}(t::ScalarReTerm{T}, x)
    t.λ = convert(T, x)
    λsq = abs2(t.λ)
    for j in 1:size(t,2)
        t.plsdiag[j] = λsq * t.crprdiag[j] + one(T)
        t.plsdinv[j] = inv(t.plsdiag[j])
    end
    t
end

@doc "Solve u := (t't + I)\(t'r)" ->
pls(t::ScalarReTerm, r::VecOrMat) = PDiagMat(t.plsdiag, t.plsdinv)\(t'r)

Base.logdet(t::ScalarReTerm) = sum(Base.LogFun(), t.plsdiag)

function pwrss{T<:FloatingPoint}(t::ScalarReTerm{T}, y::Vector{T})
    u = pls(t, y)
    res = sumabs2(u)
    pred = t * u
    for i in 1:length(y)
        res += abs2(y[i] - pred[i])
    end
    res
end

function objective!{T<:FloatingPoint}(t::ScalarReTerm{T}, λ::T, r::Vector{T})
    update!(t,λ)
    n = size(t, 1)
    logdet(t) + n * (1.+log(2π * pwrss(t, r)/n))
end

function PDMats.whiten!{T<:FloatingPoint}(r::DenseVector{T}, t::ScalarReTerm{T}, b::DenseVector{T})
    (q = size(t,2)) == length(b) == length(r) || throw(DimensionMismatch(""))
    for i in 1:q
        r[i] = sqrt(t.plsdinv[i]) * v[i]
    end
    r
end

PDMats.whiten!{T<:FloatingPoint}(t::ScalarReTerm{T}, b::DenseVector{T}) = whiten!(b, t, b)

PDMats.whiten!{T<:FloatingPoint}(r::DenseMatrix{T}, t::ScalarReTerm{T}, b::DenseMatrix{T}) =
    broadcast!(*, r, b, sqrt(t.plsdinv))

PDMats.whiten!{T<:FloatingPoint}(t::ScalarReTerm{T}, B::DenseMatrix{T}) = whiten!(B, t, B)

function PDMats.whiten!{T<:FloatingPoint}(t::ScalarReTerm{T}, B::SparseMatrixCSC{T})
    (q = size(t,2)) == size(B,1) || throw(DimensionMismatch(""))
    sc = sqrt(t.plsdinv)
    bv = B.nzval
    rv = B.rowval
    for i in 1:length(rv)
        bv[i] *= sc[rv[i]]
    end
    B
end
