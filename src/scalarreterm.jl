type ScalarReTerm <: ReTerm
    f::PooledDataVector                 # grouping factor
    z::Vector{Float64}
    λ::Float64
end

function ScalarReTerm(f::PooledDataVector, z::Vector{Float64})
    length(f) == length(z) || throw(DimensionMismatch(""))
    rr = f.refs
    ScalarReTerm(f, z, one(Float64))
end

Base.size(t::ScalarReTerm) = (length(t.z), length(t.f.pool))
Base.size(t::ScalarReTerm,i::Integer) =
    i < 1 ? throw(BoundsError()) :
    i == 1 ? length(t.z) :
    i == 2 ? length(t.f.pool) : 1

function Base.A_mul_B!(r::DenseVecOrMat, t::ScalarReTerm, v::DenseVecOrMat)
    n,q = size(t)
    k = size(v,2)
    size(r,1) == n && size(v,1) == q && size(r,2) == k || throw(DimensionMismatch(""))
    tz = t.z
    rr = t.f.refs
    if k == 1
        for i in eachindex(r)
            @inbounds r[i] = tz[i] * v[rr[i]]
        end
    else
        for j in 1:k, i in 1:n
            @inbounds r[i,j] = tz[i] * v[rr[i],j]
        end
    end
    r
end

function *(t::ScalarReTerm, v::DenseVecOrMat)
    k = size(t,1)
    A_mul_B!(Array(Float64, isa(v,Vector) ? (k,) : (k,size(v,2))), t, v)
end

function Base.Ac_mul_B!(r::DenseVecOrMat, t::ScalarReTerm, v::DenseVecOrMat)
    n,q = size(t)
    k = size(v,2)
    size(r,1) == q && size(v,1) == n && size(r,2) == k || throw(DimensionMismatch(""))
    fill!(r,zero(eltype(r)))
    if k == 1
        for i in 1:n
            @inbounds r[t.f.refs[i]] += v[i] * t.z[i]
        end
    else
        for j in 1:k, i in 1:n
            @inbounds r[t.f.refs[i],j] += v[i,j] * t.z[i]
        end
    end
    scale!(r,t.λ)
end

function Base.Ac_mul_B(t::ScalarReTerm, v::DenseVecOrMat{Float64})
    k = size(t,2)
    Ac_mul_B!(Array(Float64, isa(v,Vector) ? (k,) : (k, size(v,2))), t, v)
end

function Base.Ac_mul_B!(r::DenseVecOrMat, v::DenseVecOrMat, t::ScalarReTerm)
    n,q = size(t)
    k = size(v,2)
    size(r,2) == q && size(v,1) == n && size(r,1) == k || throw(DimensionMismatch(""))
    fill!(r, zero(eltype(r)))
    if k == 1
        for i in 1:n
            @inbounds r[t.f.refs[i]] += v[i] * t.z[i]
        end
    else
        for j in 1:k, i in 1:n
            @inbounds r[j,t.f.refs[i]] += v[i,j] * t.z[i]
        end
    end
    scale!(r,t.λ)
end

Base.Ac_mul_B(v::DenseVecOrMat{Float64},t::ScalarReTerm) =
    Ac_mul_B!(Array(Float64,(size(v,2),size(t,2))), v, t)


function Base.Ac_mul_B(t::ScalarReTerm, s::ScalarReTerm)
    if is(s,t)
        crprd = zeros(length(s.f.pool))
        z = s.z
        rr = s.f.refs
        for i in eachindex(z)
            crprd[rr[i]] += abs2(z[i])
        end
        return Diagonal(crprd)
    end
    sparse(convert(Vector{Int32},t.f.refs),convert(Vector{Int32},s.f.refs),t.z .* s.z)
end

lowerbd(t::ScalarReTerm) = zeros(Float64,1)

function setpars!(t::ScalarReTerm, x, Aii::Diagonal{Float64}, Rii::PDiagMat)
    t.λ = convert(Float64,x[1])
    λsq = abs2(t.λ)
    rd = Rii.diag
    ad = Aii.diag
    for j in eachindex(rd)
        rd[j] = λsq * ad[j] + 1.
    end
    t
end

function setpars!(t::ScalarReTerm, x, Aii::Diagonal{Float64}, Rii::PDMat)
    t.λ = convert(Float64,x[1])
    λsq = abs2(t.λ)
    rm = fill!(Rii.mat,0.)
    ad = Aii.diag
    for j in 1:size(t,2)
        rm[j,j] = λsq * ad[j] + 1.
    end
    t
end

Base.scale!(t::ScalarReTerm,v::DenseVecOrMat) = scale!(t.λ, v)

@doc "Solve u := (t't + I)\(t'r)" ->
pls(t::ScalarReTerm, r::DenseVecOrMat) = PDiagMat(t.plsdiag, t.plsdinv)\(t'r)

@doc "Solve u := (t't + I)\(t'r) in place" ->
function pls!(u::DenseVecOrMat, t::ScalarReTerm, r::DenseVecOrMat)
    scale!(t.plsdinv, Ac_mul_B!(reshape(u,(size(u,1),size(u,2))), t, r))
end

Base.logdet(t::ScalarReTerm) = sum(Base.LogFun(), t.plsdiag)

function pwrss(t::ScalarReTerm, y)
    u = pls(t, y)
    res = sumabs2(u)
    pred = t * u
    for i in 1:length(y)
        res += abs2(y[i] - pred[i])
    end
    res
end

function objective!(t::ScalarReTerm, λ::Float64, r::Vector)
    setpars!(t,λ)
    n = size(t, 1)
    logdet(t) + n * (1.+log(2π * pwrss(t, r)/n))
end

function PDMats.whiten!(r::DenseVector{Float64}, t::ScalarReTerm, b::DenseVector{Float64})
    (q = size(t,2)) == length(b) == length(r) || throw(DimensionMismatch(""))
    for i in eachindex(b)
        r[i] = sqrt(t.plsdinv[i]) * b[i]
    end
    r
end

PDMats.whiten!(t::ScalarReTerm, b::DenseVector{Float64}) = whiten!(b, t, b)

PDMats.whiten!(r::DenseMatrix{Float64}, t::ScalarReTerm, b::DenseMatrix{Float64}) =
    broadcast!(*, r, b, sqrt(t.plsdinv))

PDMats.whiten!(t::ScalarReTerm, B::DenseMatrix{Float64}) = whiten!(B, t, B)

function PDMats.whiten!(t::ScalarReTerm, B::SparseMatrixCSC{Float64})
    (q = size(t,2)) == size(B,1) || throw(DimensionMismatch(""))
    sc = sqrt(t.plsdinv)
    bv = B.nzval
    rv = B.rowval
    for i in eachindex(bv)
        bv[i] *= sc[rv[i]]
    end
    B
end

npar(t::ScalarReTerm) = 1

Base.scale!(r::DenseVecOrMat{Float64},t::ScalarReTerm,a::DenseVecOrMat{Float64}) = scale!(t.λ,copy!(r,a))

function Base.scale!(r::SparseMatrixCSC,t::ScalarReTerm,a::SparseMatrixCSC)
    size(r) == size(a) && nnz(r) == nnz(a) || throw(DimensionMismatch(""))
    λ = t.λ
    rcp = r.colptr
    acp = a.colptr
    rrv = r.rowval
    arv = a.rowval
    rnz = r.nzval
    anz = a.nzval
    for j in 1:size(a,2)
        rcp[j+1] = acp[j+1]
        for k in acp[j]:(acp[j+1]-1)
            rrv[k] = arv[k]
            rnz[k] = λ*anz[k]
        end
    end
    r
end

Base.scale!(r::DenseVecOrMat{Float64},t::ScalarReTerm) = scale!(r,t.λ)

function Base.scale!(r::SparseMatrixCSC{Float64},t::ScalarReTerm)
    scale!(r.nzval,t.λ)
    r
end
