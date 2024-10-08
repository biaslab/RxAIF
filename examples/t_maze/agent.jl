function initializeAgent(params, stats)
    pol_t = nothing
    function plan(t::Int64)
        tau = params.T - t + 1  # Current planning horizon 
        res = infer(model          = t_maze_plan(tau=tau, params=params, stats=stats),
                    constraints    = structured(), 
                    data           = (c = [params.C for k=1:tau],),
                    initialization = init_marginals(),
                    iterations     = 50,
                    returnvars     = (u = KeepLast(),),
                    free_energy    = false)

        return pol_t = mean.(res.posteriors[:u])
    end

    function estimate(o_t::Vector, a_t::Int64)
        res = infer(model       = t_maze_estimate(params=params, stats=stats, u_t=a_t),
                    data        = (y_t = o_t,),
                    returnvars  = (x_t = KeepLast(),),
                    free_energy = false)
        
        return stats[:D_t_min] = res.posteriors[:x_t].p  # Reset for next timestep
    end
    
    act() = first(pol_t)

    return (plan, act, estimate)
end