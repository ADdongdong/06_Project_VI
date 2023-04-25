import jax
import jax.numpy as jnp


def leapfrog(params, momentum, log_prob_fn, step_size, n_steps):
    """Approximates Hamiltonian dynamics using the leapfrog algorithm."""

    # define a single step
    def step(i, args):
        params, momentum = args
        
        # update momentum
        grad = jax.grad(log_prob_fn)(params)
        momentum += 0.5 * step_size * grad

        # update params
        params += momentum * step_size

        # update momentum
        grad = jax.grad(log_prob_fn)(params)
        momentum += 0.5 * step_size * grad
        
        return params, momentum

    # do 'n_steps'
    '''
    lax.fori_loop() 循环函数，将函数应用于一个范围内的这个数
    参数一：循环的下限
    参数二：循环的上限
    参数三：英语与循环变量的函数
    参数四：循环变量的初始值
    '''
    new_params, new_momentum = jax.lax.fori_loop(0, n_steps, step, (params, momentum))
    print(new_params, new_momentum)
    return new_params, new_momentum


def hmc_sampler(params, log_prob_fn, n_steps, n_leapfrog_steps, step_size, key):
    """
    Runs HMC and returns the full Markov chain as a list.
    - params: array
    - log_prob_fn: function that takes params as the only argument and returns a scalar value
    """

    # define a single step
    def step(i, args):
        params, chain, total_accept_prob, key = args
        key, normal_key, uniform_key = jax.random.split(key, 3)

        # generate random momentum 初始化动量
        momentum = jax.random.normal(normal_key, shape=params.shape)

        # leapfrog 得到新的参数θ和动量v
        new_params, new_momentum = leapfrog(params, momentum, log_prob_fn, step_size, n_leapfrog_steps)
        

        # MH correction
        potentaial_energy_diff = log_prob_fn(new_params) - log_prob_fn(params)
        kinetic_energy_diff = 0.5*(momentum**2 - new_momentum**2).sum()
        log_accept_prob = potentaial_energy_diff + kinetic_energy_diff
        accept_prob = jnp.minimum(1, jnp.exp(log_accept_prob))
        total_accept_prob += accept_prob
        accept = jax.random.uniform(uniform_key) < accept_prob
        #where 返回一个数组，第一个参数是条件，如果是true返回后面两个数组中的第一个，否则返回第二个
        #这个函数支持向量运算
        params = jnp.where(accept, new_params, params)
        
        # store params 保存参数
        chain = chain.at[i].set(params)
         
        return params, chain, total_accept_prob, key
    
    # do 'n_steps'
    chain = jnp.zeros([n_steps, len(params)])
    _, chain, total_accept_prob, key = jax.lax.fori_loop(0, n_steps, step, (params, chain, 0, key))
    
    print(f'Avg. accept. prob.: {(total_accept_prob/n_steps):.2%}')
    return chain
