# This file implements any hot-fixes for external dependencies.
# This file can be empty, which is fine.

using Random
using BayesBase
using ReactiveMP
using ExponentialFamily
using SpecialFunctions: digamma
using StatsFuns: gammainvcdf, loggamma
using LoopVectorization: vmap, @turbo

import Base: size
import RxInfer: mean
import BayesBase: rand!, eltype
import Distributions: entropy, logpdf, insupport, _rand!
import ExponentialFamily: rand!, size


#-----------
# SampleList
#-----------

function mean_h(d::SampleList)
    s = d.samples
    w = d.weights
    N = length(w)
    s_vec = reshape(s, (ndims(d)..., N))

    m   = mapreduce(i->s_vec[:,:,i].*w[i], +, 1:N)
    m_h = mapreduce(i->h(s_vec[:,:,i]).*w[i], +, 1:N)

    return (m, m_h)
end


#----------------
# MatrixDirichlet
#----------------

# These are hacks to make _rand! work with matrix variate logpfds
eltype(::LinearizedProductOf) = Float64
eltype(::ContinuousMatrixvariateLogPdf) = Float64

size(d::MatrixDirichlet) = size(d.a)

function logpdf(d::MatrixDirichlet, x::AbstractMatrix)
    return sum(sum((d.a.-1).*log.(x),dims=1) - sum(loggamma.(d.a), dims=1) + loggamma.(sum(d.a,dims=1)))
end

mean(::typeof(clamplog), dist::MatrixDirichlet) = 
    digamma.(clamplog.(dist.a)) .- digamma.(sum(clamplog.(dist.a)); dims = 1)

# Average energy definition for SampleList marginal
@average_energy MatrixDirichlet (q_out::SampleList, q_a::PointMass) = begin
    H = mapreduce(+, zip(eachcol(mean(q_a)), eachcol(mean(log, q_out)))) do (q_a_column, logmean_q_out_column)
        return -loggamma(sum(q_a_column)) + sum(loggamma.(q_a_column)) - sum((q_a_column .- 1.0) .* logmean_q_out_column)
    end
    return H
end

# Patch rand! as defined in ExponentialFamily
function rand!(rng::AbstractRNG, dist::MatrixDirichlet, container::AbstractMatrix{T}) where {T <: Real}
    samples = vmap(d -> rand(rng, Dirichlet(convert(Vector, d))), eachcol(dist.a))
    @views for col in 1:size(container)[2]
        b = container[:, col]
        b[:] .= samples[col]
    end

    return container
end

function _rand!(rng::AbstractRNG, dist::MatrixDirichlet, container::Array{Any, 3})
    for i = 1:size(container)[3]
        samples = vmap(d -> rand(rng, Dirichlet(convert(Vector, d))), eachcol(dist.a))
        @views for col in 1:size(container)[2]
            b = container[:, col, i]
            b[:] .= samples[col]
        end
    end

    return container
end

function mean_h(d::MatrixDirichlet)
    n_samples = 20 # Fixed number of samples
    s = [rand(d) for i=1:n_samples]

    return (sum(s)./n_samples, sum(h.(s))./n_samples)
end


#-----------
# Transition
#-----------

@rule Transition(:out, Marginalisation) (q_in::PointMass, q_a::Any) = begin
    a = clamp.(exp.(mean(log, q_a) * probvec(q_in)), tiny, Inf)
    return Categorical(a ./ sum(a))
end

@rule Transition(:a, Marginalisation) (q_out::Any, q_in::PointMass) = begin
    return MatrixDirichlet(collect(probvec(q_out)) * probvec(q_in)' .+ 1)
end


#-------
# Probit
#-------

using StatsFuns: normcdf
using ReactiveMP: getp, clamplog

# Overload average energy for Probit node with safe computation
@average_energy Probit (q_out::Union{PointMass, Bernoulli}, q_in::UnivariateNormalDistributionsFamily, meta::ProbitMeta) = begin

    # extract parameters
    p = mean(q_out)
    m, v = mean_var(q_in)

    # specify function
    h = (x) -> -p * clamplog(normcdf(x)) - (1 - p) * clamplog(normcdf(-x))

    # calculate average average energy (default of 32 points)
    gh_cubature = GaussHermiteCubature(getp(meta))
    U = 0.0
    tmp = sqrt(2 * v)
    for k in 1:getp(meta)
        U += gh_cubature.witer[k] * h(gh_cubature.piter[k] * tmp + m)
    end
    U /= sqrt(pi)

    # return average energy
    return U
end