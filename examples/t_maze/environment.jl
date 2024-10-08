using Random

function initializeWorld(env)
    x_t_min = env.x_0
    function execute(a_t::Int64) # Execute a move to position a_t
        x_t = env.B[a_t]*x_t_min # State transition
        y_t = env.A*x_t # Observation probabilities

        x_t_min = x_t # Reset state for next step
    end

    y_t = env.A*env.x_0
    function observe()
        k = rand(Categorical(y_t))
        o_t = zeros(16)
        o_t[k] = 1.0

        return o_t # One-hot observation
    end

    return (execute, observe)
end