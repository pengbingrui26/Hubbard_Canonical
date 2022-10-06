import jax
import jax.numpy as jnp
from free_model import Hubbard_1d 
from ED_1d import Hubbard_ED
import time 

from set_up import init_fn, \
                   flip, flip_vmapped, random_init_sigma, random_init_sigma_vmapped           

 
jax.config.update("jax_enable_x64", True)

I = complex(0., 1.)

# =============================================================================

# sample W and W_sign
def sample_simple(model, psi0, key, batch, taus, nthermal, nsample, ninterval):
    make_W, _, _ = init_fn(model)
    make_W_vmapped = jax.vmap(make_W, in_axes = (None, 0, None), out_axes = (0, 0))

    key_init, key_flip = jax.random.split(key, 2)
    key_init = jax.random.split(key_init, batch)
    sigma = random_init_sigma_vmapped(model.L*2, key_init)  # shape: (batch, L*2)

    sigma_sampled = []
    W_sampled = []
    sign_sampled = []

    start = time.time()

    W_real, W_imag = make_W_vmapped(psi0, sigma, taus)
    W = W_real + I * W_imag
    W_norm = abs(W)
    W_sign = W / W_norm

    for imove in range(nthermal + nsample * ninterval):
        print("imove:", imove)

        if (imove > nthermal) and ((imove - nthermal) % ninterval == 0):
            sigma_sampled.append(sigma)
            W_sampled.append(W)
            sign_sampled.append(W_sign)
        
        ## flip
        key_uniform, key_proposal, key_flip = jax.random.split(key_flip, 3)
        key_proposal = jax.random.split(key_proposal, batch)

        sigma_proposal = flip_vmapped(sigma, key_proposal)

        W_proposal_real, W_proposal_imag = make_W_vmapped(psi0, sigma_proposal, taus)
        W_proposal = W_proposal_real + I * W_proposal_imag
        W_proposal_norm = abs(W_proposal)

        #W_ratio = abs(W_proposal) / abs(W)
        W_ratio = W_proposal_norm / W_norm
        ratio = jnp.where(W_ratio < 1., W_ratio, 1.)
        proposal = jax.random.uniform(key_uniform, shape = (batch, ))

        accept = ratio > proposal
        sigma = jnp.where(accept[:, None], sigma_proposal, sigma)
        W = jnp.where(accept, W_proposal, W)
        W_norm = abs(W)
        W_sign = W / W_norm
        ##

    end = time.time()
    print("time for MCMC:", end - start)

    sigma_sampled = jnp.array(sigma_sampled).reshape(-1, model.L*2)
    W_sampled = jnp.array(W_sampled).flatten()
    sign_sampled = jnp.array(sign_sampled).flatten()

    return sigma_sampled, W_sampled, sign_sampled



# test ====================================================================

def test_sample_simple():
    L = 2
    N = int(L/2)
    t = 1.
    U = 1.

    tau1, tau2 = 0.2, 0.2
    taus = jnp.array([tau1, tau2])

    nthermal = 10
    nsample = 5
    ninterval = 1
    batch = 3

    key = jax.random.PRNGKey(21)

    model = Hubbard_1d(L, N, t, U)
    psi0 = model.get_ground_state()

    sigma, W, sign = sample_simple(model, psi0, key, batch, taus, nthermal, nsample, ninterval)

    print("sigma:", sigma.shape)   
    print("W:", W.shape)
    print("sign:", sign.shape)



def test_sample_vmapped():
    L = 4
    N = int(L/2)
    t = 1.
    U = 1.

    tau1, tau2 = 0.2, 0.2
    taus = jnp.array([tau1, tau2])

    nthermal = 10
    nsample = 6
    ninterval = 5
    batch = 500

    model = Hubbard_1d(L, N, t, U)
    psi0_set = model.get_psi0_full()
    psi0_set = jnp.array(psi0_set) 
    
    key = jax.random.PRNGKey(21)
    key_vmapped = jax.random.split(key, psi0_set.shape[0])

    sigma_sampled_vmapped, W_sampled_vmapped, sign_sampled_vmapped = \
                sample_vmapped(model, psi0_set, key_vmapped, batch, taus, nthermal, nsample, ninterval)

    print("sigma_sample.shape:", sigma_sampled_vmapped.shape)


# run ===========================================================================

#test_sample_simple()
#test_sample_vmapped()
t
