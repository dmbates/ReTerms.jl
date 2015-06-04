module ReTerms

    using DataArrays, PDMats, StatsBase

    export LMM, ReTerm, ScalarReTerm, VectorReTerm  # types

    export getpars!, lowerbd, pls, pls!, reterm, setpars!

    include("sputils.jl")
    include("reterm.jl")
    include("scalarreterm.jl")
    include("vectorreterm.jl")
    include("pls.jl")

end # module
