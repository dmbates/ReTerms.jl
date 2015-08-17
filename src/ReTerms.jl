module ReTerms

    using DataArrays, DataFrames, HDF5, NLopt, StatsBase

    export ReMat

    export g2dict, lowerbd, objective, reterm

    using Base.LinAlg.BlasInt

    include("blockmats.jl")
    include("remat.jl")
    include("paramlowertriangular.jl")
    include("pls.jl")

end # module
