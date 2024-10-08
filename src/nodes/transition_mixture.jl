export TransitionMixture


struct TransitionMixture end

@node TransitionMixture Stochastic [out, in, switch, a1, a2, a3, a4]

@average_energy TransitionMixture (q_out_in_switch::ContingencyTensor, 
                                   q_a1::PointMass, 
                                   q_a2::PointMass,
                                   q_a3::PointMass,
                                   q_a4::PointMass) = begin
    # Build a tensor object                                    
    log_A_bar = [mean(BroadcastFunction(clamplog), q_a1);;;
                 mean(BroadcastFunction(clamplog), q_a2);;;
                 mean(BroadcastFunction(clamplog), q_a3);;;
                 mean(BroadcastFunction(clamplog), q_a4)]

    B = mean(q_out_in_switch)

    sum(-tr.(transpose.(eachslice(B, dims=3)) .* eachslice(log_A_bar, dims=3)))
end

@average_energy TransitionMixture (q_out::PointMass,
                                   q_in_switch::ContingencyTensor, 
                                   q_a1::PointMass, 
                                   q_a2::PointMass,
                                   q_a3::PointMass, 
                                   q_a4::PointMass) = begin
    score(AverageEnergy(), 
          TransitionMixture, 
          Val{(:out_in_switch, :a1, :a2, :a3, :a4)}(), 
          map((q) -> Marginal(q, false, false, nothing), (q_in_switch, q_a1, q_a2, q_a3, q_a4)), 
          nothing)
end
