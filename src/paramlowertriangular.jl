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

ColMajorLowerTriangular(typ,n::Integer) = ColMajorLowerTriangular(LowerTriangular(eye(typ,n)))

ColMajorLowerTriangular(n::Integer) = ColMajorLowerTriangular(LowerTriangular(eye(n)))

Base.convert(::Type{LowerTriangular},A::ColMajorLowerTriangular) = A.Lambda

Base.size(A::ColMajorLowerTriangular, args...) = size(A.Lambda, args...)

Base.size(A::ColMajorLowerTriangular) = size(A.Lambda)

Base.copy(A::ColMajorLowerTriangular) = ColMajorLowerTriangular(copy(A.Lambda))

Base.copy!(A::ColMajorLowerTriangular,B::ColMajorLowerTriangular) = (copy!(A.Lambda.data,B.Lambda.data);A)

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

function Base.setindex!{T}(A::ColMajorLowerTriangular{T},v::StridedVector{T},s::Symbol)
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

function Base.scale!(A::ColMajorLowerTriangular,B::HBlkDiag)
    Ba = B.arr
    r,s,k = size(Ba)
    Al = A.Lambda
    n = Base.LinAlg.chksquare(Al)
    n == r || throw(DimensionMismatch())
    if r == 1
        scale!(Ba,Al[1,1])
    else
        Ac_mul_B!(Al,reshape(Ba,(r,s*k)))
    end
    B
end

function Base.scale!(A::HBlkDiag,B::ColMajorLowerTriangular)
    aa = A.arr
    r,s,k = size(aa)
    bl = B.Lambda
    (l = Base.LinAlg.chksquare(bl)) == r == s || throw(DimensionMismatch())
    for i in 1:k
        A_mul_B!(sub(aa,:,:,i),bl)
    end
    A
end

function Base.Ac_mul_B!{T}(A::ColMajorLowerTriangular{T},B::Diagonal{T})
    size(A,1) == 1 || throw(DimensionMismatch())
    scale!(A.Lambda.data[1,1],B.diag)
    B
end

LT(A::ReMat) = ColMajorLowerTriangular(eltype(A.z),1)

LT(A::VectorReMat) = (Az = A.z; ColMajorLowerTriangular(eltype(Az),size(Az,1)))

function Base.scale!{T}(A::ColMajorLowerTriangular{T},B::DenseVecOrMat{T})
    al = A.Lambda
    (l = Base.LinAlg.chksquare(al)) == 1 && return scale!(al.data[1],B)
    m,n = size(B,1),size(B,2)
    Base.LinAlg.Ac_mul_B!(al,reshape(B,(l,div(m,l)*n)))
    B
end

function Base.scale!{T}(A::AbstractMatrix{T},B::ColMajorLowerTriangular{T})
    bl = B.Lambda
    (l = Base.LinAlg.chksquare(bl)) == 1 && return scale!(A,bl.data[1])
    m,n = size(A,1),size(A,2)
    q,r = divrem(n,l)
    r == 0 || throw(DimensionMismatch())
    for b in 0:(q-1)
        A_mul_B!(sub(A,:,b*q + (1:q)),bl)
    end
    A
end
