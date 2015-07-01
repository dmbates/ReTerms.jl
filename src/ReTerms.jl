module ReTerms

    using DataArrays, DataFrames, HDF5, NLopt, StatsBase

    export LMM, ReTerm, ScalarReTerm, VectorReTerm  # types

    export g2dict, getpars, lowerbd, objective, reterm, setpars!

    using Base.LinAlg.BlasInt

    include("utils.jl")
    include("reterm.jl")
    include("simplescalarreterm.jl")
    include("scalarreterm.jl")
    include("vectorreterm.jl")
    include("pls.jl")

end # module
