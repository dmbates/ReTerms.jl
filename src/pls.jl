type LMM <: StatsBase.RegressionModel
    trms::Vector{Any}
    A::Matrix{Any}
    R::Matrix{Any}
    lower::Vector{Float64}              # vector of lower bounds on parameters
    pars::Vector{Float64}               # current parameter vector
end

function LMM(X::AbstracMatrix, rev::Vector, y::Vector)
    n,p = size(X)
    all(t -> size(t,1) == n, rev) && length(y) == n || throw(DimensionMismatch(""))
    lower = Float64[]
    for r in rev
        k = size(r.z,1)
        for i in 1:k
            push!(lower,0.0)
            for j in i:k
                push!(lower,-Inf)
            end
        end
    end
    ntrms = length(rev) + 2
    trms = Array(Any,ntrms)
    for i in eachindex(rev) trms[i] = rev[i] end
    trms[end-1] = X
    trms[end] = reshape(convert(Vector{Float64},y),(n,1))
    A = Array(Any,(ntrms,ntrms))
    R = Array(Any,(ntrms,ntrms))
    for j in 1:(ntrms-1),i in (j+1):ntrms
        A[i,j] = R[i,j] = nothing
    end
    for j in 1:ntrms, i in 1:j
        pr = Ac_mul_B(trms[i],trms[j])
        if issparse(pr) && isdenseish(pr)
            pr = full(pr)
        end
        A[i,j] = pr
        if i == j
            mm = A[i,i]
            for k in 1:i
                mm += A[k,i]'A[k,i]
            end
            R[i,i] = isdiag(mm) ? PDiagMat(diag(mm)) : PDMat(full(mm))
        else
            R[i,j] = copy(pr)
        end
    end
    LMM(trms,A,R,lower,[x == 0. ? 1. : 0. for x in lower])
end

LMM(re::Vector,y::Vector) = LMM(ones((length(y),1)),re,y)
