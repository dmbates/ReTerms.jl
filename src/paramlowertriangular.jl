"""
Parameterized lower triangular matrices.
"""
abstract ParamLowerTriangular{T,S<:AbstractMatrix} <: Base.LinAlg.AbstractTriangular{T,S}

"""
Parameterized lower triangular matrix in which each element of the lower triangle is a parameter
"""
immutable ColMajorLowerTriangular{T,S<:AbstractMatrix} <: ParamLowerTriangular{T,S}
    Lambda::LowerTriangular{T,S}
end

ColMajorLowerTriangular(n::Integer) = ColMajorLowerTriangular(LowerTriangular(eye(n)))

Base.convert(::Type{LowerTriangular},A::ColMajorLowerTriangular) = A.Lambda

Base.size(A::ColMajorLowerTriangular, args...) = size(A.Lambda, args...)

Base.copy(A::ColMajorLowerTriangular) = ColMajorLowerTriangular(copy(A.Lambda))

Base.full(A::ColMajorLowerTriangular) = full(A.Lambda)

@inline nlower(n::Integer) = (n*(n+1))>>1

function Base.getindex{T}(A::ColMajorLowerTriangular{T},s::Symbol)
    s == :θ || throw(KeyError(s))
    Ad = A.Lambda.data
    n = size(Ad,1)
    res = Array(T,nlower(n))
    k = 0
    for j = 1:n, i in j:n
        @inbounds res[k += 1] = Ad[i,j]
    end
    res
end

Base.getindex(A::ColMajorLowerTriangular,i::Integer,j::Integer) = A.Lambda[i,j]

function Base.setindex!{T}(A::ColMajorLowerTriangular{T},v::Vector{T},s::Symbol)
    s == :θ || throw(KeyError(s))
    Ad = A.Lambda.data
    n = size(Ad,1)
    length(v) == nlower(n) || throw(DimensionMismatch("length(v) = $(length(v)), should be $(nlower(n))"))
    k = 0
    for j in 1:n, i in j:n
        Ad[i,j] = v[k += 1]
    end
    A
end

"""
lower bounds on the parameters
"""
function lowerbd{T}(A::ColMajorLowerTriangular{T})
    n = size(A.Lambda.data,1)
    res = fill(convert(T,-Inf),nlower(n))
    k = -n
    for j in n+1:-1:2
        res[k += j] = zero(T)
    end
    res
end

"""
size of the parameter vector
"""
nθ(A::ColMajorLowerTriangular) = nlower(size(A.Lambda.data,1))


