export GoalObservation
export BetheMeta, GeneralizedMeta
export BethePipeline, GeneralizedPipeline


struct GoalObservation end

@node GoalObservation Stochastic [c, x, A]


#--------
# Helpers
#--------

h(A) = -diag(A'*clamplog.(A))

mean_h(d::PointMass) = (d.point, h(d.point))


#----------
# Modifiers
#----------

# Metas
struct BetheMeta{P} # Meta parameterized by y type for rule overloading
    y::P # Pointmass value for observation
end
BetheMeta() = BetheMeta(missing) # Absent observation

struct GeneralizedMeta{P}
    y::P # Pointmass value for observation
    newton_iterations::Int64
end
GeneralizedMeta() = GeneralizedMeta(missing, 20)
GeneralizedMeta(point) = GeneralizedMeta(point, 20)

# Pipelines
struct BethePipeline <: FunctionalDependencies end
struct GeneralizedPipeline <: FunctionalDependencies
    init_message::Union{Bernoulli, Categorical}

    GeneralizedPipeline() = new() # If state is clamped, then no inital message is required
    GeneralizedPipeline(init_message::Union{Bernoulli, Categorical}) = new(init_message)
end

function functional_dependencies(::BethePipeline, factornode, interface, iindex)
    message_dependencies = ()
    
    clusters = getlocalclusters(factornode)
    marginal_dependencies = getmarginals(clusters) # Include all node-local marginals

    return message_dependencies, marginal_dependencies
end

function functional_dependencies(pipeline::GeneralizedPipeline, factornode, interface, iindex)
    clusters = getlocalclusters(factornode)
    cindex = clusterindex(clusters, iindex) # Find the index of the cluster for the current interface

    # Message dependencies
    if (iindex === 2) # Message towards state
        output = messagein(interface)
        setmessage!(output, pipeline.init_message)
        message_dependencies = (interface, )
    else
        message_dependencies = ()
    end

    # Marginal dependencies
    if (iindex === 2) || (iindex === 3) # Message towards state or parameter
        marginal_dependencies = getmarginals(clusters) # Include all marginals
    else
        marginal_dependencies = skipindex(getmarginals(clusters), cindex) # Skip current cluster
    end

    return message_dependencies, marginal_dependencies
end


#-----------------
# Average Energies
#-----------------

@average_energy GoalObservation (q_c::Union{Dirichlet, PointMass}, 
                                 q_x::Union{Bernoulli, Categorical, PointMass}, 
                                 q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                 meta::BetheMeta{Missing}) = begin
    log_c = mean(BroadcastFunction(log), q_c)
    x = probvec(q_x)
    log_A = mean(BroadcastFunction(log), q_A)

    # Compute internal marginal
    y = softmax(log_A*x + log_c)

    return -y'*(log_A*x + log_c - clamplog.(y))
end

@average_energy GoalObservation (q_c::Union{Dirichlet, PointMass}, 
                                 q_x::Union{Bernoulli, Categorical, PointMass}, 
                                 q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                 meta::BetheMeta{<:AbstractVector}) = begin
    log_c = mean(BroadcastFunction(log), q_c)
    x = probvec(q_x)
    log_A = mean(BroadcastFunction(log), q_A)

    return -meta.y'*(log_A*x + log_c)
end

@average_energy GoalObservation (q_c::Union{Dirichlet, PointMass}, 
                                 q_x::Union{Bernoulli, Categorical, PointMass}, 
                                 q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                 meta::GeneralizedMeta{Missing}) = begin
    log_c = mean(BroadcastFunction(log), q_c)
    x = probvec(q_x)
    (A, h_A) = mean_h(q_A)

    return x'*h_A - (A*x)'*(log_c - clamplog.(A*x))
end

@average_energy GoalObservation (q_c::Union{Dirichlet, PointMass}, 
                                 q_x::Union{Bernoulli, Categorical, PointMass}, 
                                 q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                 meta::GeneralizedMeta{<:AbstractVector}) = begin
    log_c = mean(BroadcastFunction(log), q_c)
    x = probvec(q_x)
    log_A = clamp.(mean(BroadcastFunction(log), q_A), -12, 12)

    return -meta.y'*(log_A*x + log_c)
end