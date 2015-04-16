module ReTerms

    using Compat, PDMats, DataArrays.PooledDataVector #, Mamba, Distributions

if VERSION < v"0.4-"
    using Docile                        # for the @doc macro
end

    export ReTerm, ScalarReTerm         # types

    export lowerbd, pls, update!

    abstract ReTerm

    include("scalarreterm.jl")
#    include("mamba.jl")

end # module
