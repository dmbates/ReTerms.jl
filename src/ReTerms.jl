module ReTerms

    using DataArrays, DataFrames, HDF5, NLopt, StatsBase

    export ReMat

    export g2dict, lowerbd, objective, reterm

    using Base.LinAlg.BlasInt

#    include("utils.jl")
    include("paramlowertriangular.jl")
    include("remat.jl")
#    include("simplescalarreterm.jl")
#    include("scalarreterm.jl")
#    include("vectorreterm.jl")
#    include("pls.jl")

end # module
