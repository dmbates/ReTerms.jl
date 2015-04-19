module ReTerms

    using Compat, DataArrays.PooledDataVector, Optim, PDMats, StatsBase

if VERSION < v"0.4-"
    using Docile                        # for the @doc macro
end

    export FeTerm, LMM, ReTerm, ScalarReTerm         # types

    export lowerbd, pls, update!

    abstract ReTerm

    include("scalarreterm.jl")
    include("pls.jl")

end # module
