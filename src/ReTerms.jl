module ReTerms

    using PDMats, DataArrays.PooledDataVector, Mamba, Distributions

    export ReTerm, ScalarReTerm, SimpleScalarReTerm

    abstract ReTerm

    ## until the PR for PDMats is incorporated
    +(a::AbstractPDMat,b::UniformScaling) = a + ScalMat(dim(a), @compat(Float64(b.λ)))

    include("scalarreterm.jl")
    include("mamba.jl")

end # module
