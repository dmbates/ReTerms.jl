@doc "Create the pattern of the Cholesky factor based on the upper triangle of A"->
function cholpattern{Tv,Ti}(A::SparseMatrixCSC{Tv,Ti})
    m,n = size(A)
    m == n || error("A must be square")
    parent = etree(A)
    I = Ti[]
    J = Ti[]
    sizehint!(I, nnz(A))
    sizehint!(J, nnz(A))
    for j in Ti[1:n;]
        cj = Base.SparseMatrix.ereach(A, j, parent)
        append!(J,cj)
        push!(J,j)
        append!(I,fill(j,length(cj) + 1))
    end
    sparse(I,J,one(Tv))
end

@doc """
Convert sparse to dense if the proportion of nonzeros exceeds a threshold.
A no-op for other matrix types.
"""->
densify(S,threshold=0.3) = issparse(S) && nnz(S)/(*(size(S)...)) > threshold ? full(S) : S

@doc """Convert a group in an HDF5File to a Dict{Symbol,Any} using readmmap"""->
function g2dict(fid::HDF5File,gnm)
    res = Dict{Symbol,Any}()
    g = fid[gnm]
    for nm in names(g)
        dd = readmmap(g[nm])
        if eltype(dd) <: Unsigned && isperm(unique(dd))
            dd = PooledDataArray(DataArrays.RefArray(dd),convert(Vector{Int32},[1:maximum(dd);]))
        end
        res[Symbol(nm)] = dd
    end
    res
end
