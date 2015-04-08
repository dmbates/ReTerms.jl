module ReTerms

    using PDMats, DataArrays.PooledDataVector, Mamba, Distributions

    export ReTerm, ScalarReTerm, SimpleScalarReTerm

    abstract ReTerm
    
    include("scalarreterm.jl")
    include("mamba.jl")

end # module
