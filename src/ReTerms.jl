module ReTerm

    using PDMats,DataArrays.PooledDataVector

    export ReTerm, ScalarReTerm, SimpleScalarReTerm

    abstract ReTerm
    
    include("./scalarreterm")

end # module
