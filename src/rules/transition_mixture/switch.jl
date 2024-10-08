@rule TransitionMixture(:switch, Marginalisation) (
    m_out::Union{Bernoulli, DiscreteNonParametric, PointMass}, 
    m_in::Union{Bernoulli, DiscreteNonParametric, PointMass},
    q_a1::PointMass, 
    q_a2::PointMass,
    q_a3::PointMass, 
    q_a4::PointMass) = begin

    p_out = probvec(m_out)
    p_in  = probvec(m_in)
    
    A_bar = exp.([mean(BroadcastFunction(clamplog), q_a1);;; 
                  mean(BroadcastFunction(clamplog), q_a2);;;
                  mean(BroadcastFunction(clamplog), q_a3);;;
                  mean(BroadcastFunction(clamplog), q_a4)])

    # Contraction
    (M, N, L) = size(A_bar)
    a = zeros(L)
    for j=1:M
        for i=1:N
            @inbounds a += A_bar[j,i,:]*p_out[j]*p_in[i]
        end
    end      

    return Categorical(a ./ sum(a))
end

@rule TransitionMixture(:switch, Marginalisation) (
    m_in::Union{Bernoulli, DiscreteNonParametric, PointMass},
    q_out::PointMass, 
    q_a1::PointMass, 
    q_a2::PointMass,
    q_a3::PointMass, 
    q_a4::PointMass) = begin 
    
    @call_rule TransitionMixture(:switch, Marginalisation) (
        m_out = q_out,
        m_in  = m_in,
        q_a1  = q_a1,
        q_a2  = q_a2,
        q_a3  = q_a3,
        q_a4  = q_a4
    )
end
