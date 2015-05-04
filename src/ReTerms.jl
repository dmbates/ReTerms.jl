module ReTerms

    using Compat, DataArrays.PooledDataVector, PDMats, StatsBase

if VERSION < v"0.4-"
    using Docile                        # for the @doc macro
end

    export FeTerm, LMM, ReTerm, ScalarReTerm, VectorReTerm  # types

    export getpars!, lowerbd, pls, pls!, reterm, setpars!

    include("reterm.jl")
    include("scalarreterm.jl")
    include("vectorreterm.jl")
    include("pls.jl")

end # module
