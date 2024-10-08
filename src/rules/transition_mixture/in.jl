@rule TransitionMixture(:in, Marginalisation) (
    m_out::Union{Bernoulli, DiscreteNonParametric, PointMass}, 
    m_switch::Union{Bernoulli, DiscreteNonParametric, PointMass},
    q_a1::PointMass, 
    q_a2::PointMass,
    q_a3::PointMass, 
    q_a4::PointMass) = begin

    p_out    = probvec(m_out)
    p_switch = probvec(m_switch)
    
    A_bar = exp.([mean(BroadcastFunction(clamplog), q_a1);;; 
                  mean(BroadcastFunction(clamplog), q_a2);;;
                  mean(BroadcastFunction(clamplog), q_a3);;;
                  mean(BroadcastFunction(clamplog), q_a4)])

    # Contraction
    (M, N, L) = size(A_bar)
    a = zeros(N)
    for j=1:M
        for k=1:L
            @inbounds a += A_bar[j,:,k]*p_out[j]*p_switch[k]
        end
    end      

    return Categorical(a ./ sum(a))
end

@rule TransitionMixture(:in, Marginalisation) (
    m_switch::Union{Bernoulli, DiscreteNonParametric, PointMass}, 
    q_out::PointMass, 
    q_a1::PointMass, 
    q_a2::PointMass,
    q_a3::PointMass, 
    q_a4::PointMass) = begin
    
    @call_rule TransitionMixture(:in, Marginalisation) (
        m_out    = q_out, 
        m_switch = m_switch,
        q_a1     = q_a1, 
        q_a2     = q_a2,
        q_a3     = q_a3,
        q_a4     = q_a4
    )
end
