module ReTerms

    using DataArrays, DataFrames, NLopt, StatsBase

    export LMM, ReTerm, ScalarReTerm, VectorReTerm  # types

    export getpars, lowerbd, reterm, setpars!

    include("sputils.jl")
    include("reterm.jl")
    include("scalarreterm.jl")
    include("vectorreterm.jl")
    include("pls.jl")

end # module
