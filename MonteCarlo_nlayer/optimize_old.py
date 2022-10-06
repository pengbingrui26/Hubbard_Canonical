import jax
import jax.numpy as jnp
from free_model import Hubbard_1d 
from ED_dimer import dimer
from ED_1d import Hubbard_ED
import time 

from mc_1d import make_expU, evolve_nlayer_vmapped, make_W, make_W_vmapped, \
                  make_Eloc_old_vmapped, make_Eloc, make_Eloc_vmapped, make_eloc, make_eloc_vmapped, \
                     flip_vmapped, random_init_sigma, random_init_sigma_vmapped, \
                     make_direct_nlayer
            
jax.config.update("jax_enable_x64", True)

I = complex(0., 1.)

# =============================================================================

def make_expU_nlayer_det(model, sigma, tau1s):
    output_L = jnp.identity(model.L*2)
    output_R = jnp.identity(model.L*2)

    sigmaL = sigma[:model.L]
    sigmaR = sigma[model.L:]

    for tau1 in tau1s:
        expU_L = make_expU(model, sigmaL, tau1)
        expU_R = make_expU(model, sigmaR, tau1)
        output_L = jnp.dot(expU_L, output_L)
        output_R = jnp.dot(expU_R, output_R)
 
    output = jnp.dot(jnp.conjugate(output_L.T), output_R)
    return jnp.linalg.det(output)


make_expU_nlayer_det_vmapped = jax.vmap(make_expU_nlayer_det, in_axes = (None, 0, None), out_axes = 0)


def make_W_ratio(model, psi0, sigma, sigma_proposal, tau1):
    psi0_dagger = jnp.conjugate(psi0.T)
    psi0_inv = jnp.linalg.pinv(psi0)
    psi0_dagger_inv = jnp.linalg.pinv(psi0_dagger)

    sigmaL = sigma[:model.L]
    sigmaR = sigma[model.L:]
    sigmaL_proposal = sigma_proposal[:model.L]
    sigmaR_proposal = sigma_proposal[model.L:]

    expU_L = make_expU(model, sigmaL, tau1)
    expU_R = make_expU(model, sigmaR, tau1)
    expU_L_proposal = make_expU(model, sigmaL_proposal, tau1)
    expU_R_proposal = make_expU(model, sigmaR_proposal, tau1)

    expUs = [ expU_L, jnp.conjugate(expU_R.T), psi0_inv, psi0_dagger_inv, expU_R_proposal, jnp.conjugate(expU_L_proposal.T) ]

    #print(jnp.dot(psi0_inv, psi0_dagger_inv))
    #print(jnp.dot(jnp.conjugate(expU_L.T), expU_L))

    #output = psi0
    output = jnp.identity(model.L*2)
    for expU in expUs:
        #print("expU:")
        #print(expU)
        output = jnp.dot(expU, output)

    #print("output:")
    #print(output)

    output = jnp.dot(jnp.conjugate(psi0.T), output)

    return jnp.linalg.det(output) 




def sample(model, psi0, key, batch, taus, nthermal, nsample, ninterval):

    key_init, key_flip = jax.random.split(key, 2)

    key_init = jax.random.split(key_init, batch)
    sigma = random_init_sigma_vmapped(model.L*2, key_init)  # shape: (batch, L*2)

    sigma_sampled = []
    W_sampled = []
    sign_sampled = []

    start = time.time()

    W_real, W_imag = make_W_vmapped(model, psi0, sigma, taus)
    W = W_real + I * W_imag
    W_norm = abs(W)
    W_sign = W / W_norm

    for imove in range(nthermal + nsample * ninterval):
        #print("imove:", imove)

        if (imove > nthermal) and ((imove - nthermal) % ninterval == 0):
            sigma_sampled.append(sigma)
            W_sampled.append(W)
            sign_sampled.append(W_sign)

        ## flip
        key_uniform, key_proposal, key_flip = jax.random.split(key_flip, 3)
        key_proposal = jax.random.split(key_proposal, batch)

        sigma_proposal = flip_vmapped(sigma, key_proposal)

        W_proposal_real, W_proposal_imag = make_W_vmapped(model, psi0, sigma_proposal, taus)
        W_proposal = W_proposal_real + I * W_proposal_imag
        W_proposal_norm = abs(W_proposal)

        #tau1s = taus[0: :2]
        #W = make_expU_nlayer_det_vmapped(model, sigma, tau1s)
        #W_proposal = make_expU_nlayer_det_vmapped(model, sigma_proposal, tau1s)

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

    start = time.time()

    sigma_sampled = jnp.array(sigma_sampled).reshape(-1, model.L*2)
    W_sampled = jnp.array(W_sampled).flatten()
    sign_sampled = jnp.array(sign_sampled).flatten()

    end = time.time()
    print("time for flattening:", end - start)

    return sigma_sampled, W_sampled, sign_sampled


sample_vmapped = jax.vmap(sample, in_axes = (None, 0, 0, None, None, None, None, None), out_axes = (0, 0, 0))

# =====================================================================================================


make_grad_W = jax.jacrev(make_W, argnums = -1)
make_grad_W_vmapped = jax.vmap(make_grad_W, in_axes = (None, None, None, 0, None), out_axes = (0, 0))

make_grad_Eloc = jax.jacrev(make_Eloc, argnums = -1)
make_grad_Eloc_vmapped = jax.vmap(make_grad_Eloc, in_axes = (None, None, None, 0, None), out_axes = (0, 0))


make_grad_eloc = jax.jacrev(make_eloc, argnums = -1)
make_grad_eloc_vmapped = jax.vmap(make_grad_eloc, in_axes = (None, None, None, 0, None), out_axes = 0)



# sample Sign, eloc, W, d_eloc/d_t, d_W/d_t 
def sample_new(model, psi0, key, batch, taus, nthermal, nsample, ninterval):
    key_init, key_flip = jax.random.split(key, 2)
    key_init = jax.random.split(key_init, batch)
    sigma = random_init_sigma_vmapped(model.L*2, key_init)  # shape: (batch, L*2)

    Tmatr = model.get_Hfree()
    expT0 = jax.scipy.linalg.expm(-I * Tmatr)
    expT0 = list(expT0)
    print("expT0:")
    print(expT0)

    #sigma_sampled = []
    W_sampled = []
    sign_sampled = []
    eloc_sampled = []

    grad_W_sampled = []
    grad_eloc_sampled = []

    start = time.time()

    W_real, W_imag = make_W_vmapped(model, psi0, expT0, sigma, taus)
    W = W_real + I * W_imag
    W_norm = abs(W)
    W_sign = W / W_norm

    for imove in range(nthermal + nsample * ninterval):
        #print("imove:", imove)

        ## collect
        if (imove > nthermal) and ((imove - nthermal) % ninterval == 0):
            W_sampled.append(W)
            sign_sampled.append(W_sign)

            eloc_real, eloc_imag = make_eloc_vmapped(model, psi0, expT0, sigma, taus) 
            eloc = eloc_real + I * eloc_imag
            grad_W_real, grad_W_imag = make_grad_W_vmapped(model, psi0, expT0, sigma, taus)     
            grad_W = grad_W_real + I * grad_W_imag

            grad_eloc_real, grad_eloc_imag = make_grad_eloc_vmapped(model, psi0, expT0, sigma, taus)     
            grad_eloc = grad_eloc_real + I * grad_eloc_imag

            eloc_sampled.append(eloc)
            grad_W_sampled.append(grad_W)
            grad_eloc_sampled.append(grad_eloc)

        ## flip
        key_uniform, key_proposal, key_flip = jax.random.split(key_flip, 3)
        key_proposal = jax.random.split(key_proposal, batch)

        sigma_proposal = flip_vmapped(sigma, key_proposal)

        W_proposal_real, W_proposal_imag = make_W_vmapped(model, psi0, expT0, sigma_proposal, taus)
        W_proposal = W_proposal_real + I * W_proposal_imag
        W_proposal_norm = abs(W_proposal)

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
    print("time for MCMC by sample_new:", end - start)

    W_sampled = jnp.array(W_sampled).flatten()
    sign_sampled = jnp.array(sign_sampled).flatten()
    eloc_sampled = jnp.array(eloc_sampled).flatten()

    grad_W_sampled = jnp.array(grad_W_sampled)
    grad_eloc_sampled = jnp.array(grad_eloc_sampled)

    #print("grad_W_sampled.shape:", grad_W_sampled.shape)
    #print("grad_eloc_sampled.shape:", grad_eloc_sampled.shape)

    grad_W_sampled = jnp.concatenate(grad_W_sampled, axis =0).T
    grad_eloc_sampled = jnp.concatenate(grad_eloc_sampled, axis =0).T

    return W_sampled, sign_sampled, eloc_sampled, grad_W_sampled, grad_eloc_sampled


sample_new_vmapped = jax.vmap(sample_new, in_axes = (None, 0, 0, None, None, None, None, None), out_axes = (0, 0, 0))


# ==================================================================================

def make_En(model, psi0, key, batch, taus, nthermal, nsample, ninterval):
    #start = time.time()
    sigma, W, sign = sample(model, psi0, key, batch, taus, nthermal, nsample, ninterval)
    #end = time.time()
    #print("time for sampling:", end - start)

    start = time.time()
    #Eloc_real, Eloc_imag = make_Eloc_vmapped(model, psi0, sigma, taus)
    Eloc_real, Eloc_imag = make_Eloc_old_vmapped(model, psi0, sigma, taus)
    Eloc = Eloc_real + I * Eloc_imag
    
    eloc = Eloc / W

    sign_dot_eloc = jnp.multiply(sign, eloc)
    E_n = sign_dot_eloc.sum() / sign.sum()
    end = time.time()
    print("time for computing En:", end - start)

    return E_n

make_En_vmapped = jax.vmap(make_En, in_axes = (None, 0, 0, None, None, None, None, None), out_axes = 0)



def make_En_new(model, psi0, key, batch, taus, nthermal, nsample, ninterval):
    W, sign, eloc, grad_W, grad_eloc = sample_new(model, psi0, key, batch, taus, nthermal, nsample, ninterval)

    start = time.time()

    sign_dot_eloc = jnp.multiply(sign, eloc)
    E_n = sign_dot_eloc.sum() / sign.sum()
    end = time.time()
    print("time for computing En:", end - start)

    return E_n


# ==================================================================================


def make_En_and_grad(model, psi0, key, batch, taus, nthermal, nsample, ninterval):
    sigma, W, sign = sample(model, psi0, key, batch, taus, nthermal, nsample, ninterval)

    #Eloc_real, Eloc_imag = make_Eloc_vmapped(model, psi0, sigma, taus)
    Eloc_real, Eloc_imag = make_Eloc_old_vmapped(model, psi0, sigma, taus)
    Eloc = Eloc_real + I * Eloc_imag
    
    eloc = Eloc / W

    sign_dot_eloc = jnp.multiply(sign, eloc)
    E_mean = sign_dot_eloc.sum() / sign.sum()

    grad_W_real, grad_W_imag = make_grad_W_vmapped(model, psi0, sigma, taus)
    grad_W = grad_W_real + I * grad_W_imag
    #print("grad_W:", grad_W.shape)
 
    #grad_Eloc_real, grad_Eloc_imag = make_grad_Eloc_vmapped(model, psi0, sigma, taus)
    #grad_Eloc = grad_Eloc_real + I * grad_Eloc_imag
    ##print("grad_Eloc:", grad_Eloc.shape)
    
    #grad_eloc = (jnp.multiply(grad_Eloc.T, W) - jnp.multiply(Eloc, grad_W.T)) / jnp.multiply(W, W)
    ##print("grad_eloc:", grad_eloc.shape) 

    grad_eloc_real, grad_eloc_imag = make_grad_eloc_vmapped(model, psi0, sigma, taus)
    grad_eloc = grad_eloc_real + I * grad_eloc_imag
    #print("grad_eloc:", grad_eloc.shape) 

    O = grad_W.T / W

    dominator = jnp.multiply( sign, grad_eloc.T + jnp.multiply(eloc - E_mean, O) )
    #print("dominator:", dominator.shape)
    dominator = dominator.mean(axis = -1).real
    numerator = sign.mean().real

    gradient= dominator / numerator

    return E_mean.real, gradient 


make_En_and_grad_vmapped = jax.vmap(make_En_and_grad, in_axes = (None, 0, 0, None, None, None, None, None), out_axes = (0, 0))



def make_En_and_grad_new(model, psi0, key, batch, taus, nthermal, nsample, ninterval):
    W, sign, eloc, grad_W, grad_eloc = sample_new(model, psi0, key, batch, taus, nthermal, nsample, ninterval)

    start = time.time()

    sign_dot_eloc = jnp.multiply(sign, eloc)
    E_mean = sign_dot_eloc.sum().real / sign.sum().real

    O = grad_W / W

    dominator = jnp.multiply( sign, grad_eloc + jnp.multiply(eloc - E_mean, O) )
    dominator = dominator.mean(axis = -1).real
    numerator = sign.mean().real

    gradient= dominator / numerator

    end = time.time()
    print("time for computing loss and grad:", end - start)

    return E_mean, gradient 

make_En_and_grad_new_vmapped = jax.vmap(make_En_and_grad_new, in_axes = (None, 0, 0, None, None, None, None, None), out_axes = (0, 0))




def make_E_and_grad_total(model, psi0_set, key, batch, taus, nthermal, nsample, ninterval):
    # psi0_set has type of list
    #E_set = 0.
    #grad_set = jnp.zeros(2)
    #for psi0 in psi0_set:
    #    E, grad =  make_E_and_grad(model, psi0, key, batch, taus, nthermal, nsample, ninterval)
    #    E_set += E
    #    grad_set += grad

    #E_set = jnp.array(E_set)
    #grad_set = jnp.array(grad_set)
 
    E, grad = make_En_and_grad_vmapped(model, psi0_set, key, batch, taus, nthermal, nsample, ninterval)
    E = E.sum()
    grad = grad.sum(axis = 0)

    return E, grad


def make_p(qq):
    pp = jax.nn.softmax(qq)
    return pp

def make_logp(qq):
    pp = jax.nn.softmax(qq)
    logpp = jnp.log(pp)
    return logpp

make_grad_logp = jax.jacrev(make_logp, argnums = 0)


def make_loss_grad(beta, model, psi0_set, key, batch, params, nthermal, nsample, ninterval):
    qq = params[:psi0_set.shape[0]]
    pp = jax.nn.softmax(qq)
    #grad_pp = jax.jacrev(make_p, argnums = 0)  # the Jacobi has shape pp.shape, qq.shape), with the (i, j)-th element being dp_i / dq_j
    grad_log_pp = make_grad_logp(qq)

    S_all = 1./beta * jnp.log(pp) 

    key_set = jax.random.split(key, psi0_set.shape[0])
    taus = params[psi0_set.shape[0]:]

    #E_all, grad_E_all = make_En_and_grad_vmapped(model, psi0_set, key_set, batch, taus, nthermal, nsample, ninterval)
    E_all, grad_E_all = make_En_and_grad_new_vmapped(model, psi0_set, key_set, batch, taus, nthermal, nsample, ninterval)

    F = jnp.dot(pp, S_all + E_all)

    grad_F_qq = jnp.dot(jnp.multiply(S_all + E_all, grad_log_pp.T), pp) # shape of (qq.shape, )
    #print("grad_F_qq:", grad_F_qq.shape)
    grad_F_taus = jnp.dot(pp, grad_E_all)
    #print("grad_F_taus:", grad_F_taus.shape)

    grad_F = jnp.hstack((grad_F_qq, grad_F_taus))
    
    return F, grad_F



def optimize_En(model, psi0, batch, nthermal, nsample, ninterval, Nlayer):
    
    taus = jnp.array([0.1] * 2 * Nlayer) 
    params = taus

    learning_rate = 1e-2

    import optax
    optimizer = optax.adam(learning_rate = learning_rate)
    opt_state = optimizer.init(params)

    key = jax.random.PRNGKey(42)

    def step(params, opt_state):
        loss, grad = make_En_and_grad(model, psi0, key, batch, params, nthermal, nsample, ninterval)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grad

    opt_nstep = 2000

    dimer_ED = dimer(model.t, model.U)
    E_exact = dimer_ED.eigs()[0]

    loss_all = []
    for istep in range(opt_nstep):
        start = time.time()
        params, opt_state, loss, grad = step(params, opt_state)
        end = time.time()
        print("time:", end - start)
        print('istep:', istep)
        print('grad:')
        print(grad)
        print('params:')
        print(params)
        print('loss, exact:', loss, E_exact)
        loss_all.append(loss)
        #if abs(loss - E_exact[0]) < 1e-2:
        #    break;
        print('\n')



def optimize_E_total(model, psi0_set, batch, nthermal, nsample, ninterval, Nlayer):
    
    taus = jnp.array([0.1] * 2 * Nlayer) 
    params = taus

    learning_rate = 1e-2

    import optax
    optimizer = optax.adam(learning_rate = learning_rate)
    opt_state = optimizer.init(params)

    key = jax.random.PRNGKey(42)

    def step(params, opt_state, key):
        key_old, key = jax.random.split(key, 2)
        key_set = jax.random.split(key, psi0_set.shape[0])

        loss, grad = make_E_and_grad_total(model, psi0_set, key_set, batch, params, nthermal, nsample, ninterval)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grad, key

    opt_nstep = 2000

    dimer_ED = dimer(model.t, model.U)
    E_exact = dimer_ED.eigs()[0].sum()

    loss_all = []
    for istep in range(opt_nstep):
        params, opt_state, loss, grad, key = step(params, opt_state, key)
        print('istep:', istep)
        print('grad:')
        print(grad)
        print('params:')
        print(params)
        print('loss, exact:', loss, E_exact)
        loss_all.append(loss)
        #if abs(loss - E_exact[0]) < 1e-2:
        #    break;
        print('\n')


# =================================================================================================

def make_free_energy_ED(beta, L, N, t, U):
    model = Hubbard_ED(L, N, t, U)
    F = model.free_energy(beta)
    return F


def optimize_F(beta, model, psi0_set, batch, nthermal, nsample, ninterval, Nlayer):
    qq = jnp.array([0.1] * psi0_set.shape[0])    
    taus = jnp.array([0.1] * 2 * Nlayer) 
    params = jnp.hstack((qq, taus))

    learning_rate = 1e-1

    import optax
    optimizer = optax.adam(learning_rate = learning_rate)
    opt_state = optimizer.init(params)

    key = jax.random.PRNGKey(42)

    def step(params, opt_state, key):
        key_old, key = jax.random.split(key, 2)

        loss, grad = make_loss_grad(beta, model, psi0_set, key, batch, params, nthermal, nsample, ninterval)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grad, key

    opt_nstep = 200

    F_exact = make_free_energy_ED(beta, model.L, model.N, model.t, model.U)

    loss_all = []
    for istep in range(opt_nstep):
        params, opt_state, loss, grad, key = step(params, opt_state, key)
        print('istep:', istep)
        print('grad:')
        print(grad)
        print('params:')
        print(params)
        print('loss, exact:', loss, F_exact)
        loss_all.append(loss)
        #if abs(loss - F_exact) < 1e-2 * 5:
        #    break
        print('\n')

    datas = {"F_exact": F_exact, "U": model.U, "beta": beta, \
             "learning_rate": learning_rate, "opt_nstep":opt_nstep, "loss": loss_all}

    import pickle as pk
    fp = open('./optimize_F', 'wb')
    pk.dump(datas, fp)
    fp.close()

# test =======================================================================


def test_make_expU_nlayer_det():
    L = 2
    N = int(L/2)
    t = 1.
    U = 4.
    model = Hubbard_1d(L, N, t, U)

    psi0 = model.get_psi0()
    psi0 = model.get_psi0()[:,:model.N*2]

    tau1s = jnp.array([1., 2.])
   
    sigma = jnp.array([1, -1, 1, -1])
    sigma_vmapped = jnp.array([ sigma, sigma, sigma ])
    print(sigma_vmapped.shape)

    det_expU_vmapped = make_expU_nlayer_det_vmapped(model, sigma_vmapped, tau1s) 
    print(det_expU_vmapped)


   
def test_make_W_ratio():
    L = 2
    N = int(L/2)
    t = 1.
    U = 4.
    model = Hubbard_1d(L, N, t, U)

    psi0 = model.get_psi0()[:,:model.N*2]
    psi0_dagger = jnp.conjugate(psi0.T)

    psi0_inv = jnp.linalg.pinv(psi0)
    psi0_dagger_inv = jnp.linalg.pinv(psi0_dagger)
    #print(psi0_dagger_inv)

    #print(jnp.dot(psi0_inv, psi0))
    #print(jnp.dot(psi0, psi0_inv))

    #psi0_norm = jnp.dot(jnp.conjugate(psi0.T), psi0)
    #psi0_norm = jnp.dot(psi0, jnp.conjugate(psi0.T))
    #print(psi0_norm)

    sigma = jnp.array([1, 1, -1, -1])
    sigma_proposal = jnp.array([1, 1, -1, -1])

    tau1 = 1.
    tau2 = 0.8
    taus = jnp.array([tau1, tau2])

    W_real, W_imag = make_W(model, psi0, sigma, taus)
    W = W_real + I * W_imag

    W_proposal_real, W_proposal_imag = make_W(model, psi0, sigma_proposal, taus)
    W_proposal = W_proposal_real + I * W_proposal_imag

    W_ratio = W_proposal / W

    W_ratio_new = make_W_ratio(model, psi0, sigma, sigma_proposal, tau1)

    print(W_ratio)
    print(W_ratio_new)



def test_make_W():
    L = 2
    N = int(L/2)
    t = 1.
    U = 4.
    model = Hubbard_1d(L, N, t, U)

    psi0 = model.get_psi0()[:,:model.N*2]

    key_init = jax.random.PRNGKey(21)
    sigma = random_init_sigma(L*2, key_init)   
 
    batch = 3  
    key_init_vmapped = jax.random.split(key_init, batch)
    sigma_vmapped = random_init_sigma_vmapped(L*2, key_init_vmapped)   

    taus = jnp.array([1., 1.])

    W_real, W_imag = make_W(model, psi0, sigma, taus)
    W_real, W_imag = make_W_vmapped(model, psi0, sigma_vmapped, taus)
    print(W_real)
    print(W_imag)


def test_make_grad_W():
    L = 2
    N = int(L/2)
    t = 1.
    U = 4.
    model = Hubbard_1d(L, N, t, U)

    psi0 = model.get_psi0()[:,:model.N*2]

    key_init = jax.random.PRNGKey(21)
    sigma = random_init_sigma(L*2, key_init)   
 
    batch = 3  
    key_init_vmapped = jax.random.split(key_init, batch)
    sigma_vmapped = random_init_sigma_vmapped(L*2, key_init_vmapped)   

    taus = jnp.array([1., 1.])

    make_grad_W = jax.jacrev(make_W, argnums = -1)
    grad_W = make_grad_W(model, psi0, sigma, taus)
    print(grad_W)
    #print(grad_W.shape)

    #make_grad_W_vmapped = jax.jacrev(make_W_vmapped, argnums = -1)
    make_grad_W_vmapped = jax.vmap(make_grad_W, in_axes = (None, None, 0, None), out_axes = (0, 0))
    grad_W_vmapped = make_grad_W_vmapped(model, psi0, sigma_vmapped, taus)
    print(grad_W_vmapped)
    #print(grad_W_vmapped.shape)



def test_make_eloc_vmapped():
    L = 2
    N = int(L/2)
    t = 1.
    U = 4.
    model = Hubbard_1d(L, N, t, U)

    psi0 = model.get_psi0()[:,:model.N*2]
 
    batch = 3  
    key_init = jax.random.PRNGKey(21)
    key_init = jax.random.split(key_init, batch)
    sigma = random_init_sigma_vmapped(L*2, key_init)   

    taus = jnp.array([1., 1.])

    eloc = make_eloc_vmapped(model, psi0, sigma, taus)
    print(eloc)


 
def test_sample():
    L = 2
    N = int(L/2)
    t = 1.
    U = 1.

    tau1, tau2 = 0.2, 0.2
    taus = jnp.array([tau1, tau2])

    nthermal = 100
    nsample = 4
    ninterval = 10
    batch = 5

    key = jax.random.PRNGKey(21)

    model = Hubbard_1d(L, N, t, U)
    psi0 = model.get_psi0()[:,:model.N*2]
    psi0_set = model.get_psi0_set()

    sigma_sampled, W_sampled, sign_sampled = sample(model, psi0, key, batch, taus, nthermal, nsample, ninterval)
    
    return sigma_sampled, W_sampled, sign_sampled


def test_sample_new():
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

    W, sign, eloc, grad_W, grad_eloc = sample_new(model, psi0, key, batch, taus, nthermal, nsample, ninterval)
   
    print("W:", W.shape)
    print("sign:", sign.shape)
    print("eloc:", eloc.shape)
    print("grad_W:", grad_W.shape)
    print("grad_eloc:", grad_eloc.shape)



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


def test_make_En():
    L = 6
    N = int(L/2)
    t = 1.
    U = 0.

    tau1, tau2 = 1., 1.
    taus = jnp.array([tau1, tau2])

    nthermal = 100
    nsample = 11
    ninterval = 5
    batch = 1000

    key = jax.random.PRNGKey(21)

    model = Hubbard_1d(L, N, t, U)
    psi0 = model.get_ground_state()

    E = make_En(model, psi0, key, batch, taus, nthermal, nsample, ninterval)

    model_ED = Hubbard_ED(L, N, t, U)
    eigvals, _ = model_ED.eigs()
 
    print("E:", E)
    print("E_ED:", eigvals[0])


def test_make_En_vmapped():
    L = 2
    N = int(L/2)
    t = 1.

    tau1, tau2 = 0.2, 0.2
    taus = jnp.array([tau1, tau2])

    nthermal = 100
    nsample = 11
    ninterval = 10
    batch = 1000

    key = jax.random.PRNGKey(21)
    key_vmapped = jax.random.split(key, 4)

    for U in jnp.arange(0., 5., 1.):
        model = Hubbard_1d(L, N, t, U)

        psi_1 = model.get_psi0()[:, [0, 1]]
        psi_2 = model.get_psi0()[:, [0, 2]]
        psi_3 = model.get_psi0()[:, [1, 3]]
        psi_4 = model.get_psi0()[:, [2, 3]]

        psi0_vmapped = jnp.array([psi_1, psi_2, psi_3, psi_4])

        E_n = make_En_vmapped(model, psi0_vmapped, key_vmapped, batch, taus, nthermal, nsample, ninterval)
        E_ED = make_direct_nlayer(t, U, taus) 
 
        print("E:", E_n)
        print("E_ED:", E_ED)
        print("\n")


def test_make_En_and_grad():
    L = 2
    N = int(L/2)
    t = 1.

    tau1, tau2 = 0.2, 0.2
    taus = jnp.array([tau1, tau2])

    nthermal = 50
    nsample = 11
    ninterval = 5
    batch = 1000

    key = jax.random.PRNGKey(21)

    for U in jnp.arange(0., 5., 1.):
        model = Hubbard_1d(L, N, t, U)

        psi_1 = model.get_psi0()[:, [0, 1]]
        psi_2 = model.get_psi0()[:, [0, 2]]
        psi_3 = model.get_psi0()[:, [1, 3]]
        psi_4 = model.get_psi0()[:, [2, 3]]

        E_n, grad = make_En_and_grad(model, psi_1, key, batch, taus, nthermal, nsample, ninterval)
        #E_ED = make_direct_nlayer(t, U, taus) 
 
        print("E:")
        print(E_n)
        print("grad:")
        print(grad)
        #print("E_ED:", E_ED)
        print("\n")


def test_make_loss_grad():
    L = 2
    N = int(L/2)
    t = 1.

    qq = jnp.array([0.25, 0.25, 0.25, 0.25])

    tau1, tau2 = 0.2, 0.2
    taus = jnp.array([tau1, tau2])

    params = jnp.hstack((qq, taus))

    nthermal = 50
    nsample = 6
    ninterval = 5
    batch = 1000

    key = jax.random.PRNGKey(21)

    U = 1.
    model = Hubbard_1d(L, N, t, U)

    psi_1 = model.get_psi0()[:, [0, 1]]
    psi_2 = model.get_psi0()[:, [0, 2]]
    psi_3 = model.get_psi0()[:, [1, 3]]
    psi_4 = model.get_psi0()[:, [2, 3]]

    psi0_set = jnp.array([ psi_1, psi_2, psi_3, psi_4 ])

    beta = 1.
    loss, grad = make_loss_grad(beta, model, psi0_set, key, batch, params, nthermal, nsample, ninterval)
    print("loss:", loss) 
 
def test_make_grad_eloc():
    L = 2
    N = int(L/2)
    t = 1.
    U = 4.
    model = Hubbard_1d(L, N, t, U)
    psi0 = model.get_psi0()[:,:model.N*2]

    key_init = jax.random.PRNGKey(21)
    sigma = random_init_sigma(L*2, key_init)   
 
    batch = 3  
    key_init_vmapped = jax.random.split(key_init, batch)
    sigma_vmapped = random_init_sigma_vmapped(L*2, key_init_vmapped)   

    taus = jnp.array([1., 1., 1., 1.])

    grad_eloc = make_grad_eloc(model, psi0, sigma, taus) 
    print(grad_eloc)

    grad_eloc_vmapped = make_grad_eloc_vmapped(model, psi0, sigma_vmapped, taus) 
    print(grad_eloc_vmapped)


 
def test_optimize_En():
    L = 2
    N = int(L/2)
    t = 1.
    U = 4.
    model = Hubbard_1d(L, N, t, U)

    psi_1 = model.get_psi0()[:, [0, 1]]
    psi_2 = model.get_psi0()[:, [0, 2]]
    psi_3 = model.get_psi0()[:, [1, 3]]
    psi_4 = model.get_psi0()[:, [2, 3]]

    Nlayer = 1
 
    nthermal = 100
    nsample = 11
    ninterval = 5
  
    batch = 1000

    optimize_En(model, psi_1, batch, nthermal, nsample, ninterval, Nlayer)
 
 
def test_optimize_E_total():
    L = 2
    N = int(L/2)
    t = 1.
    U = 2.
    model = Hubbard_1d(L, N, t, U)

    psi_1 = model.get_psi0()[:, [0, 1]]
    psi_2 = model.get_psi0()[:, [0, 2]]
    psi_3 = model.get_psi0()[:, [1, 3]]
    psi_4 = model.get_psi0()[:, [2, 3]]

    Nlayer = 1
 
    nthermal = 100
    nsample = 11
    ninterval = 5
  
    batch = 1000

    psi0_set = [psi_1, psi_2, psi_3, psi_4]
    psi0_set = jnp.array(psi0_set)

    optimize_E_total(model, psi0_set, batch, nthermal, nsample, ninterval, Nlayer)
 

def test_make_free_energy_ED():
    L = 4
    N = int(L/2)
    t = 1.
    U = 1.
    beta = 1.
    F = make_free_energy_ED(beta, L, N, t, U)
    print(F) 


def test_optimize_F():
    L = 4
    N = int(L/2)
    t = 1.
    U = 1.
    model = Hubbard_1d(L, N, t, U)

    psi0_set = model.get_psi0_full()
    print(len(psi0_set))
    #num_psi = int(len(psi0_set) / 4)
    #psi0_set = jnp.array(psi0_set[:num_psi])
    psi0_set = jnp.array(psi0_set)
    print(psi0_set.shape[0])

    nthermal = 50
    nsample = 11
    ninterval = 5
  
    batch = 500

    Nlayer = 1
    beta = 1. 
    optimize_F(beta, model, psi0_set, batch, nthermal, nsample, ninterval, Nlayer)

 
# run ========================================================================

#test_make_expU_nlayer_det()

#test_make_W_ratio()

#test_make_W()
#test_make_grad_W()
#test_make_eloc_vmapped()
#test_grad_t()
#test_sample_old()

#test_sample()
#test_sample_new()


#test_sample_vmapped()

#test_make_En()
#test_make_En_vmapped()

#test_make_grad_eloc()

#test_make_En_and_grad()
#test_make_En_and_grad_vmapped()

#test_make_loss_grad()

#test_optimize_En()
#test_optimize_E_total()

#test_make_free_energy_ED()
test_optimize_F()

