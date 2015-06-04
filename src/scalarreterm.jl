type ScalarReTerm{T<:FloatingPoint} <: ReTerm{T}
    f::PooledDataVector                 # grouping factor
    z::Vector{T}
    λ::T
end

function ScalarReTerm{T<:FloatingPoint}(f::PooledDataVector, z::Vector{T})
    length(f) == length(z) || throw(DimensionMismatch(""))
    rr = f.refs
    ScalarReTerm(f, z, one(T))
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

function *{T<:FloatingPoint}(t::ScalarReTerm{T}, v::DenseVecOrMat{T})
    k = size(t,1)
    A_mul_B!(Array(T, isa(v,Vector) ? (k,) : (k,size(v,2))), t, v)
end

function Base.Ac_mul_B!(r::DenseVecOrMat, t::ScalarReTerm, v::DenseVecOrMat)
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

function Base.Ac_mul_B{T<:FloatingPoint}(t::ScalarReTerm{T}, v::DenseVecOrMat{T})
    k = size(t,2)
    Ac_mul_B!(Array(T, isa(v,Vector) ? (k,) : (k, size(v,2))), t, v)
end

function Base.Ac_mul_B!(r::DenseVecOrMat, v::DenseVecOrMat, t::ScalarReTerm)
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
            @inbounds r[j,t.f.refs[i]] += v[i,j] * t.z[i]
        end
    end
    scale!(r,t.λ)
end

Base.Ac_mul_B{T<:FloatingPoint}(v::DenseVecOrMat{T},t::ScalarReTerm{T}) =
    Ac_mul_B!(Array(T, (size(v,2), size(t,2))), v, t)


function Base.Ac_mul_B{T<:FloatingPoint}(t::ScalarReTerm{T}, s::ScalarReTerm{T})
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

lowerbd{T<:FloatingPoint}(t::ScalarReTerm{T}) = zeros(T,1)

function setpars!{T<:FloatingPoint}(t::ScalarReTerm{T}, x)
    t.λ = convert(T, x)
    λsq = abs2(t.λ)
    for j in 1:size(t,2)
        t.plsdiag[j] = λsq * t.crprdiag[j] + one(T)
        t.plsdinv[j] = inv(t.plsdiag[j])
    end
    t
end

Base.scale!(t::ScalarReTerm,v::DenseVecOrMat) = scale!(t.λ, v)

@doc "Solve u := (t't + I)\(t'r)" ->
pls(t::ScalarReTerm, r::DenseVecOrMat) = PDiagMat(t.plsdiag, t.plsdinv)\(t'r)

@doc "Solve u := (t't + I)\(t'r) in place" ->
function pls!(u::DenseVecOrMat, t::ScalarReTerm, r::DenseVecOrMat)
    scale!(t.plsdinv, Ac_mul_B!(reshape(u,(size(u,1),size(u,2))), t, r))
end

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
    setpars!(t,λ)
    n = size(t, 1)
    logdet(t) + n * (1.+log(2π * pwrss(t, r)/n))
end

function PDMats.whiten!{T<:FloatingPoint}(r::DenseVector{T}, t::ScalarReTerm{T}, b::DenseVector{T})
    (q = size(t,2)) == length(b) == length(r) || throw(DimensionMismatch(""))
    for i in eachindex(b)
        r[i] = sqrt(t.plsdinv[i]) * b[i]
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
    for i in eachindex(bv)
        bv[i] *= sc[rv[i]]
    end
    B
end
