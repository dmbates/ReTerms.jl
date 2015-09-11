using DataArrays, DataFrames, ReTerms
using Base.Test

include(joinpath(dirname(@__FILE__),"data.jl"))

include("paramlowertriangular.jl")
include("scalarReTerm.jl")
include("vectorReTerm.jl")
include("pls.jl")
include("simulation.jl")
