"""
`HBlkDiag` - a homogeneous block diagonal matrix, i.e. all the diagonal blocks are the same size

A matrix consisting of k diagonal blocks of size `r×s` is stored as an `r×s×k` array.
"""
immutable HBlkDiag{T} <: AbstractMatrix{T}
    arr::Array{T,3}
end

Base.eltype{T}(A::HBlkDiag{T}) = T

Base.size(A::HBlkDiag) = ((r,s,k) = size(A.arr); (r*k,s*k))

function Base.size(A::HBlkDiag,i::Integer)
    i < 1 && throw(BoundsError())
    i > 2 && return 1
    r,s,k = size(A.arr)
    (i == 1 ? r : s)*k
end

Base.copy!{T}(d::HBlkDiag{T},s::HBlkDiag{T}) = copy!(d.arr,s.arr)

Base.copy{T}(s::HBlkDiag{T}) = HBlkDiag(copy(s.arr))

function Base.LinAlg.A_ldiv_B!(R::DenseVecOrMat,A::HBlkDiag,B::DenseVecOrMat)
    Aa = A.arr
    r,s,k = size(Aa)
    r == s || throw(ArgumentError("A must be square"))
    (m = size(B,1)) == size(R,1) || throw(DimensionMismatch())
    (n = size(B,2)) == size(R,2) || throw(DimensionMismatch())
    r*k == m || throw(DimensionMismatch())
    if r == 1
        for j in 1:n, i in 1:m
            R[i,j] = B[i,j]/Aa[i]
        end
    else
        rows = (1:r)+(b-1)*k
        Base.LinAlg.A_ldiv_B!(sub(R,rows,:),sub(A.arr,:,:,b),sub(B,rows,:))
    end
    R
end

function Base.getindex{T}(A::HBlkDiag{T},i::Integer,j::Integer)
    Aa = A.arr
    r,s,k = size(Aa)
    bi,ri = divrem(i,r)
    bj,rj = divrem(j,s)
    bi == bj || return zero(T)
    Aa[ri+1,rj+1,bi]
end
