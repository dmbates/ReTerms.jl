type VectorReTerm{T<:FloatingPoint} <: ReTerm{T}
    f::PooledDataVector                 # length n grouping factor
    zt::Matrix{T}                       # k × n transposed model matrix
    λ::AbstractMatrix{T}                # k × k lower triangular
    ## can't be more specific about the type of λ with v"0.3.x"/v"0.4-" differences
    splits::Vector{Int}                 # splits in columns of λ; can be empty
    crprdiag::Array{T,3}                # k × k × q arrays
    plsdiag::Array{T,3}
end

function VectorReTerm{T<:FloatingPoint}(f::PooledDataVector, zt::Matrix{T})
    (n = length(f)) == size(zt,2) || throw(DimensionMismatch(""))
    k = size(zt,1)
    q = length(f.pool)
    crprd = zeros(T, (k, k, q))
    for i in 1:n
        BLAS.syr!('L', one(T), sub(zt,:,i), sub(crprd,:,:,Int(f.refs[i])))
    end
    plsdiag = copy(crprd)
    for j in 1:q, i in 1:k
        plsdiag[i, i, j] += one(T)
    end
    λ = VERSION < v"0.4-" ? Triangular(eye(k),:L,false) : LowerTriangular(eye(k))
    VectorReTerm(f, zt, λ, Int[], crprd, plsdiag)
end
