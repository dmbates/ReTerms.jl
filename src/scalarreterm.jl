type ScalarReTerm{T<:FloatingPoint} <: ReTerm
    f::PooledDataVector                 # grouping factor
    z::Vector{T}
    λ::T
end

function ScalarReTerm{T<:FloatingPoint}(f::PooledDataVector, z::Vector{T})
    length(f) == length(z) || throw(DimensionMismatch(""))
    ScalarReTerm(f, z, one(T))
end

ScalarReTerm(f::PooledDataVector) = ScalarReTerm(f, ones(length(f)), 1.)

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
            r[i] = t.z[i] * v[t.f.refs[i]]
        end
    else
        for j in 1:k, i in 1:n
            r[i,j] = t.z[i] * v[t.f.refs[i],j]
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
            r[t.f.refs[i]] += v[i] * t.z[i]
        end
    else
        for j in 1:k, i in 1:n
            r[t.f.refs[i],j] += v[i,j] * t.z[i]
        end
    end
    scale!(r,t.λ)
end

function Base.Ac_mul_B{T<:FloatingPoint}(t::ScalarReTerm{T}, v::VecOrMat{T})
    k = size(t,2)
    Ac_mul_B!(Array(T, isa(v,Vector) ? (k,) : (k, size(v,2))), t, v)
end

function Base.Ac_mul_B{T<:FloatingPoint}(t::ScalarReTerm{T}, s::ScalarReTerm{T})
    if is(s,t)
        z = t.z
        refs = t.f.refs
        res = zeros(eltype(z), size(t, 2))
        for i in 1:length(z)
            res[refs[i]] += abs2(z[i])
        end
        return PDiagMat(scale!(res, abs2(t.λ)))
    end
    (n = size(t,1)) == size(s,1) || throw(DimensionMismatch(""))
    I = Int[]
    J = Int[]
    V = T[]
    tr = t.f.refs
    sr = s.f.refs
    for i in 1:n
        push!(I, tr[i])
        push!(J, sr[i])
        push!(V, t.z[i] * s.z[i])
    end
    sparse(I,J,V)
end

lowerbd(t::ScalarReTerm) = zeros(1)

function update!(t::ScalarReTerm, x) 
    t.λ = convert(eltype(t.z), x)
    t
end

@doc "Solve u := (t't + I)\(t'r)" ->
pls(t::ScalarReTerm, r::VecOrMat) = (t't + I)\(t'r)
