@rule GoalObservation(:A, Marginalisation) (q_c::Union{Dirichlet, PointMass},
                                            q_x::Union{Bernoulli, Categorical, PointMass}, 
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::BetheMeta{Missing}) = begin
    log_c = mean(BroadcastFunction(log), q_c)
    x     = probvec(q_x)
    log_A = mean(BroadcastFunction(log), q_A)

    # Compute internal marginal
    y = softmax(log_A*x + log_c)

    return MatrixDirichlet(y*x' .+ 1)
end

@rule GoalObservation(:A, Marginalisation) (q_c::Union{Dirichlet, PointMass},
                                            q_x::Union{Bernoulli, Categorical, PointMass}, 
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, # Unused
                                            meta::BetheMeta{<:AbstractVector}) = begin
    x = probvec(q_x)

    return MatrixDirichlet(meta.y*x' .+ 1)
end

@rule GoalObservation(:A, Marginalisation) (q_c::Union{Dirichlet, PointMass},
                                            q_x::Union{Bernoulli, Categorical, PointMass}, 
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass},
                                            meta::GeneralizedMeta{Missing}) = begin
    log_c = mean(BroadcastFunction(log), q_c)
    x     = probvec(q_x)
    A_bar = mean(q_A)
    M, N  = size(A_bar)

    log_mu(A) = (A*x)'*(log_c - clamplog.(A_bar*x)) - x'*h(A)

    return ContinuousMatrixvariateLogPdf((RealNumbers()^M, RealNumbers()^N), log_mu)
end

@rule GoalObservation(:A, Marginalisation) (q_c::Union{Dirichlet, PointMass}, # Unused
                                            q_x::Union{Bernoulli, Categorical, PointMass}, 
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, # Unused
                                            meta::GeneralizedMeta{<:AbstractVector}) = begin
    x = probvec(q_x)

    return MatrixDirichlet(meta.y*x' .+ 1)
end
