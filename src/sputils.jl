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

@doc "Return sparsity pattern for X'X when X is sparse"->
function crprsppat(X::SparseMatrixCSC)
    m,n = size(X)
    ss = [IntSet(nzrange(X,j)) for j in 1:n]
    I = Int32[]
    J = Int32[]
    for j in 2:n, i in 1:(j-1)
        if length(intersect(ss[i],ss[j])) > 0
            push!(I,i)
            push!(J,j)
        end
    end
    sparse(I,J,1.)
end
