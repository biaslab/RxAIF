@rule GoalObservation(:x, Marginalisation) (q_c::Union{Dirichlet, PointMass},
                                            q_x::Union{Bernoulli, Categorical}, 
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::BetheMeta{Missing}) = begin
    log_c = mean(BroadcastFunction(log), q_c)
    x     = probvec(q_x)
    log_A = mean(BroadcastFunction(log), q_A)

    # Compute internal marginal
    y = softmax(log_A*x + log_c)

    return Categorical(softmax(log_A'*y))
end

@rule GoalObservation(:x, Marginalisation) (q_c::Union{Dirichlet, PointMass},
                                            q_x::Union{Bernoulli, Categorical}, # Unused
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::BetheMeta{<:AbstractVector}) = begin
    log_A = mean(BroadcastFunction(log), q_A)

    return Categorical(softmax(log_A'*meta.y))
end

@rule GoalObservation(:x, Marginalisation) (m_x::Union{Bernoulli, Categorical},
                                            q_c::Union{Dirichlet, PointMass},
                                            q_x::Union{Bernoulli, Categorical},
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::GeneralizedMeta{Missing}) = begin
    log_d    = clamplog.(probvec(m_x))
    log_c    = mean(BroadcastFunction(log), q_c)
    x_0      = probvec(q_x)
    (A, h_A) = mean_h(q_A)
    (N, M)   = size(A)

    # Root-finding problem for marginal statistics
    g(x)  = x - softmax(-h_A + A'*log_c - A'*clamplog.(A*x) + log_d)
    L(xi) = g(softmax(xi))
    J(xi) = jacobian(L, xi)

    xi_k = zeros(M)
    for k=1:meta.newton_iterations
        xi_k = xi_k - pinv(J(xi_k))*L(xi_k) # Newton step for multivariate root finding
    end

    # Compute outbound message statistics
    rho = softmax(xi_k - log_d)

    return Categorical(rho)
end

@rule GoalObservation(:x, Marginalisation) (m_x::Union{Bernoulli, Categorical}, # Unused
                                            q_c::Union{Dirichlet, PointMass}, # Unused
                                            q_x::Union{Bernoulli, Categorical}, # Unused
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::GeneralizedMeta{<:AbstractVector}) = begin
    
    log_A = clamp.(mean(BroadcastFunction(log), q_A), -12, 12)

    return Categorical(softmax(log_A'*meta.y))
end
