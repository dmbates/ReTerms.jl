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

function isdenseish(S::SparseMatrixCSC,threshold=0.3)
    0. < threshold < 1. || error("threshold must be in (0.,1.)")
    nnz(S)/(*(size(S)...)) > threshold
end
