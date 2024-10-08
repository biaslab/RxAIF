@rule GoalObservation(:c, Marginalisation) (q_c::Union{Dirichlet, PointMass},
                                            q_x::Union{Bernoulli, Categorical, PointMass}, 
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::BetheMeta{Missing}) = begin
    log_c = mean(BroadcastFunction(log), q_c)
    x     = probvec(q_x)
    log_A = mean(BroadcastFunction(log), q_A)

    # Compute internal marginal
    y = softmax(log_A*x + log_c)

    return Dirichlet(y .+ 1)
end

@rule GoalObservation(:c, Marginalisation) (q_c::Union{Dirichlet, PointMass}, # Unused
                                            q_x::Union{Bernoulli, Categorical, PointMass}, 
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::BetheMeta{<:AbstractVector}) = begin
    return Dirichlet(meta.y .+ 1)
end

@rule GoalObservation(:c, Marginalisation) (q_x::Union{Bernoulli, Categorical, PointMass}, 
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::GeneralizedMeta{Missing}) = begin
    x = probvec(q_x)
    A = mean(q_A)

    return Dirichlet(A*x .+ 1)
end

@rule GoalObservation(:c, Marginalisation) (q_x::Union{Bernoulli, Categorical, PointMass}, # Unused
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, # Unused
                                            meta::GeneralizedMeta{<:AbstractVector}) = begin
    return Dirichlet(meta.y .+ 1)
end
