module ReTerms

    using Compat, DataArrays.PooledDataVector, Optim, PDMats, StatsBase

if VERSION < v"0.4-"
    using Docile                        # for the @doc macro
end

    export FeTerm, LMM, ReTerm, ScalarReTerm, VectorReTerm  # types

    export lowerbd, pls, update!

    abstract ReTerm{T<:FloatingPoint}

    include("scalarreterm.jl")
    include("vectorreterm.jl")
    include("pls.jl")

end # module
