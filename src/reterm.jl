abstract ReTerm{T<:FloatingPoint}

function StatsBase.counts{T<:FloatingPoint}(x::ReTerm{T},y::ReTerm{T})
    Ti = max(length(x.f.pool),length(y.f.pool)) < typemax(Int32) ? Int32 : Int64
    sparse(convert(Vector{Ti},x.f.refs),convert(Vector{Ti},y.f.refs),one(T))
end

reterm(p::PooledDataVector) = ScalarReTerm(p,ones(length(p)))

reterm{T<:Integer}(v::Vector{T}) = ScalarReTerm(compact(pool(v)),ones(length(v)))
