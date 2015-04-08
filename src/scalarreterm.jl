abstract ScalarReTerm <: ReTerm

type SimpleScalarReTerm <: ScalarReTerm
    f:PooledDataVector                  # grouping factor
    λ::FloatingPoint
end

Base.size(t::SimpleScalarReTerm) = (length(t.f.pool),length(t.f.refs))
Base.size(t::SimpleScalarReTerm,i::Integer) = i < 1 ? throw(BoundsError()) :
   i == 1 ? length(t.f.pool) :
   i == 2 ? length(t.f.refs) : 1

function Base.A_mul_B!(r::VecOrMat, t::SimpleScalarReTerm, v::VecOrMat)
    m,n = size(t)
    k = size(v,2)
    size(r,1) == m && size(v,1) == n && size(r,2) == k || throw(DimensionMismatch(""))
    fill!(r,zero(eltype(r)))
    if k == 1
        for i in 1:n
            r[t.f.refs[i]] += v[i]
        end
    else
        for j in 1:k, i in 1:n
            r[t.f.refs[i],j] += v[i,j]
        end
    end
    scale!(r,t.λ)
end

function *(t::SimpleScalarReTerm, v::VecOrMat{Float64})
    k = size(t,1)
    A_mul_B!(Array(v, isa(v,Vector) ? k : (k,size(v,2))), t, v)
end

function Base.Ac_mul_B!(r::VecOrMat, t::SimpleScalarReTerm, v::VecOrMat)
    m,n = size(t)
    k = size(v,2)
    size(r,1) == n && size(v,1) == m && size(r,2) == k || throw(DimensionMismatch(""))
    fill!(r,zero(eltype(r)))
    if k == 1
        for i in 1:n
            r[i] += v[t.f.refs[i]]
        end
    else
        for j in 1:k, i in 1:n
            r[i,j] += v[t.f.refs[i],j]
        end
    end
    scale!(r,t.λ)
end

function Base.Ac_mul_B(t::SimpleScalarReTerm, v::VecOrMat)
    k = size(t,2)
    Ac_mul_B!(Array(v, isa(v,Vector) ? k : (k, size(v,2))))
end
