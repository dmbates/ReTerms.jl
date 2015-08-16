"""
`ReMat` - a random effects matrix

The matrix consists of the grouping factor, `f`, and the transposed dense model
matrix `z`.  The length of `f` must be equal to the number of columns of `z`
"""
immutable ReMat
    f::PooledDataVector                 # grouping factor
    z::Matrix
    function ReMat(p::PooledDataVector,z::Matrix)
        length(p) == size(z,2) || throw(DimensionMismatch())
        new(p,z)
    end
end

ReMat(p::PooledDataVector,v::Vector) = ReMat(p,v')

ReMat(p::PooledDataVector) = ReMat(p,ones(1,length(p)))

ReMat{T<:Integer}(v::Vector{T}) = ReMat(compact(pool(v)))

Base.eltype(A::ReMat) = eltype(A.z)

Base.size(A::ReMat) = (length(A.f),(size(A.z,1)*length(A.f.pool)))

Base.size(A::ReMat,i::Integer) =
    i < 1 ? throw(BoundsError()) :
    i == 1 ? length(A.f) :
    i == 2 ? (length(A.f.pool)*size(A.z,1)) : 1

function Base.Ac_mul_B!{T}(R::DenseVecOrMat{T},A::ReMat,B::DenseVecOrMat{T})
    n,q = size(A)
    k = size(B,2)
    size(R,1) == q && size(B,1) == n && size(R,2) == k || throw(DimensionMismatch(""))
    fill!(R,zero(T))
    rr = A.f.refs
    zz = A.z
    l = size(zz,1)
    rt = reshape(R,(l,div(q,l),k))
    for j in 1:k, i in 1:n
            Base.axpy!(B[i,j],sub(zz,:,i),sub(rt,:,Int(rr[i]),j))
    end
    R
end

function Base.Ac_mul_B{T}(A::ReMat,B::DenseVecOrMat{T})
    k = size(A,2)
    Ac_mul_B!(Array(Float64, isa(B,Vector) ? (k,) : (k, size(B,2))), A, B)
end

function Base.Ac_mul_B(A::ReMat, B::ReMat)
    if is(A,B)
        k = size(A.z,1)
        nl = length(A.f.pool)
        T = promote_type(eltype(A),eltype(B))
        crprd = zeros(T,(k,k,nl))
        z = A.z
        rr = A.f.refs
        for i in eachindex(rr)
            BLAS.syr!('L',one(T),sub(z,:,i),sub(crprd,:,:,Int(rr[i])))
        end
        for j in 1:nl
            Base.LinAlg.copytri!(sub(crprd,:,:,j),'L')
        end
        return HBlkDiag(crprd)
    end
    Az = A.z
    k,l = size(Az)
    Bz = B.z
    m,n = size(Bz)
    l == n || throw(DimensionMisMatch())
    (r,s) = promote(A.f.refs,B.f.refs)
    sparse(r,s,[sub(Az,:,i)*sub(Bz,:,i)' for i in 1:n])
end

