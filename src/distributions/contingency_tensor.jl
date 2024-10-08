export ContingencyTensor
export mean, entropy


const TensorVariate = ArrayLikeVariate{3}
const DiscreteTensorVariateDistribution = Distribution{TensorVariate,  Discrete}

struct ContingencyTensor{T<:Real, P<:AbstractArray{T}} <: DiscreteTensorVariateDistribution
    p::P
end

mean(dist::ContingencyTensor) = dist.p # Assumes normalized tensor

entropy(dist::ContingencyTensor) = -sum(xlogx.(dist.p))
