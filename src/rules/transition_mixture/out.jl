@rule TransitionMixture(:out, Marginalisation) (
    m_in::Union{Bernoulli, DiscreteNonParametric, PointMass}, 
    m_switch::Union{Bernoulli, DiscreteNonParametric, PointMass},
    q_a1::PointMass, 
    q_a2::PointMass,
    q_a3::PointMass, 
    q_a4::PointMass) = begin

    p_in     = probvec(m_in)
    p_switch = probvec(m_switch)
    
    A_bar = exp.([mean(BroadcastFunction(clamplog), q_a1);;; 
                  mean(BroadcastFunction(clamplog), q_a2);;;
                  mean(BroadcastFunction(clamplog), q_a3);;;
                  mean(BroadcastFunction(clamplog), q_a4)])

    # Contraction
    (M, N, L) = size(A_bar)
    a = zeros(M)
    for i=1:N
        for k=1:L
            @inbounds a += A_bar[:,i,k]*p_in[i]*p_switch[k]
        end
    end      

    return Categorical(a ./ sum(a))
end
