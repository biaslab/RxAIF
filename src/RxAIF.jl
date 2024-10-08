module RxAIF

using RxInfer, LinearAlgebra, ReactiveMP, BayesBase, DomainSets

using ReactiveMP: clamplog
using ForwardDiff: jacobian
using TupleTools: deleteat
using ReactiveMP: FunctionalDependencies, messagein, setmessage!, getlocalclusters, clusterindex, getmarginals
using Base.Broadcast: BroadcastFunction
using BayesBase: AbstractContinuousGenericLogPdf
using LogExpFunctions: xlogx

import ReactiveMP: functional_dependencies
import DomainSets: dimension, in
import RxInfer: mean


include("distributions/predefined.jl")
include("nodes/predefined.jl")
include("rules/predefined.jl")

end