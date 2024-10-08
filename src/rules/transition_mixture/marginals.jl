@marginalrule TransitionMixture(:out_in_switch) (
    m_out::Union{Bernoulli, DiscreteNonParametric, PointMass}, 
    m_in::Union{Bernoulli, DiscreteNonParametric, PointMass}, 
    m_switch::Union{Bernoulli, DiscreteNonParametric, PointMass}, 
    q_a1::PointMass, 
    q_a2::PointMass,
    q_a3::PointMass, 
    q_a4::PointMass) = begin

    p_out    = probvec(m_out)
    p_in     = probvec(m_in)
    p_switch = probvec(m_switch)

    A_bar = exp.([mean(BroadcastFunction(clamplog), q_a1);;; 
                  mean(BroadcastFunction(clamplog), q_a2);;;
                  mean(BroadcastFunction(clamplog), q_a3);;;
                  mean(BroadcastFunction(clamplog), q_a4)])

    B_tilde = cat(map(z->z*p_out*p_in', p_switch)..., dims=3) # Construct message tensor
    B       = B_tilde.*A_bar

    return ContingencyTensor(B ./ sum(B))
end

@marginalrule TransitionMixture(:in_switch) (
    m_in::Union{Bernoulli, DiscreteNonParametric, PointMass}, 
    m_switch::Union{Bernoulli, DiscreteNonParametric, PointMass}, 
    q_out::PointMass, 
    q_a1::PointMass, 
    q_a2::PointMass,
    q_a3::PointMass, 
    q_a4::PointMass) = begin
    
    @call_marginalrule TransitionMixture(:out_in_switch) (
        m_out    = q_out, 
        m_in     = m_in, 
        m_switch = m_switch, 
        q_a1     = q_a1, 
        q_a2     = q_a2,
        q_a3     = q_a3, 
        q_a4     = q_a4
    )
end
