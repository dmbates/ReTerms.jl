"Create the pattern of the Cholesky factor based on the upper triangle of A"
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

"""
Convert sparse to dense if the proportion of nonzeros exceeds a threshold.
A no-op for other matrix types.
"""
densify(S,threshold=0.3) = issparse(S) && nnz(S)/(*(size(S)...)) > threshold ? full(S) : S

"""Create a value of the pool for a PooledDataArray from an unsigned vector"""
function getpool(f::HDF5.HDF5Dataset,dd)
    uu = unique(dd)
    isperm(uu) || error("unique values are not a permutation")
    if exists(attrs(f),"pool")
        pool = read(attrs(f)["pool"])
        if eltype(pool) == UInt8
            pool = convert(Vector{Char},pool)
        end
        return pool
    end
    nu = length(uu)
    return convert(Vector{nu > typemax(Int32) ? Int64 :
                          nu > typemax(Int16) ? Int32 :
                          nu > typemax(Int8) ? Int16 : Int8}, [1:nu;])
end
        
"""Convert a group in an HDF5File to a Dict{Symbol,Any} using readmmap"""
function g2dict(fid::HDF5File,gnm)
    res = Dict{Symbol,Any}()
    g = fid[gnm]
    for nm in names(g)
        dd = readmmap(g[nm])
        if eltype(dd) <: Unsigned
            dd = PooledDataArray(DataArrays.RefArray(dd),getpool(g[nm],dd))
        end
        res[Symbol(nm)] = dd
    end
    res
end

function Base.scale!(x::Number,t::UpperTriangular{Float64})
    m,n = size(t)
    td = t.data
    for j in 1:n, i in 1:j
        @inbounds td[i,j] *= x
    end
end

function Base.scale!{T<:Number}(s::T,t::LowerTriangular{T})
    m,n = size(t)
    td = t.data
    for j in 1:n, i in j:m
        @inbounds td[i,j] *= s
    end
    t
end

function df2g(fid::HDF5File,nm::ByteString,df::DataFrame)
    gg = g_create(fid,nm)
    for sym in names(df)
        d = df[sym]
        if isa(d,DataArrays.PooledDataArray)
            gg[string(sym)] = d.refs
            if eltype(d.pool) <: Char
                attrs(gg[string(sym)])["pool"] = convert(Vector{UInt8},d.pool)
            else
                attrs(gg[string(sym)])["pool"] = d.pool
            end
        else
            gg[string(sym)] = convert(Array,d)
        end
    end
end
