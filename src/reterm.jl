abstract ReTerm{T<:FloatingPoint}

reterm(p::PooledDataVector) = SimpleScalarReTerm(p,1.0)

reterm{T<:Integer}(v::Vector{T}) = SimpleScalarReTerm(compact(pool(v)),1.0)
