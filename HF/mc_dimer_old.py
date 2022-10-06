import jax
import jax.numpy as jnp
from free_model import Hubbard_1d 
from ED_dimer import dimer
import time 

jax.config.update("jax_enable_x64", True)

I = complex(0., 1.)
#I = 1.

def make_expU_new(model, sigma, tau):
    # exp(-i V tau) = Gamma^L * exp(alpha * \sum_i sigma_i * (n_{i,up} - n_{i, down}))
    # Gamma = 1/2 * exp(-i tau U/4)
    # alpha = arccosh(exp(i tau U/2))

    Gamma = 1/2 * jnp.exp(-I*tau*model.U/4) 
    alpha = jnp.arccosh(jnp.exp(I*tau*model.U/2))

    nspin_arr = jnp.hstack((jnp.ones(model.L), -jnp.ones(model.L)))
    sigma = jnp.hstack((sigma, sigma))
    U_diags = jnp.multiply(sigma, nspin_arr)
    U = jnp.diag(U_diags) 
    expU = jnp.power(Gamma, model.L) * jax.scipy.linalg.expm(alpha * U)

    return expU


def make_expU(model, sigma, tau):
    # exp(-i*V*tau) = (1/2)^L * exp{-\tau*U*N/2} * \sum_{{sigma}} exp{alpha \sum_i \sigma_i (n_{i,up} - n_{i,down})}
    # alpha = arccosh(exp{i*tau*U/2})

    #print('model.U:', model.U)
    Gamma = jnp.exp(-I*tau*model.U*model.N*2/2) 
    alpha = jnp.arccosh(jnp.exp(I*tau*model.U/2))
    #print("Gamma:", Gamma)
    #print("1/cosh(alpha):", 1/jnp.cosh(alpha))

    nspin_arr = jnp.hstack((jnp.ones(model.L), -jnp.ones(model.L)))
    sigma = jnp.hstack((sigma, sigma))
    V_diags = jnp.multiply(sigma, nspin_arr)
    V = jnp.diag(V_diags) 
    expU = jnp.power(0.5, model.L) * Gamma * jax.scipy.linalg.expm(alpha * V)

    return expU


def evolve(model, psi0, sigma, tau1, tau2):
    Tmatr = model.get_Hfree()
    expT = jax.scipy.linalg.expm(-I * Tmatr * tau2)
    expU = make_expU(model, sigma, tau1)
    return jnp.dot(expT, jnp.dot(expU, psi0))

evolve_vmapped = jax.vmap(evolve, in_axes = (None, None, 0, None, None), out_axes = 0)


def evolve_nlayer(model, psi0, sigma, taus):
    Tmatr = model.get_Hfree()
    lenth = taus.shape[-1]
    nlayer = int(lenth/2)
    psi = psi0
    for ilayer in range(nlayer):
        tau1 = taus[lenth - ilayer*2 - 1]
        tau2 = taus[lenth - ilayer*2 - 2]
        expT = jax.scipy.linalg.expm(-I * Tmatr * tau2)
        expU = make_expU(model, sigma, tau1)
        psi = jnp.dot(expT, jnp.dot(expU, psi))
    return psi
       
evolve_nlayer_vmapped = jax.vmap(evolve_nlayer, in_axes = (None, None, 0, None), out_axes = 0)


 
def make_W_old(model, psi0, sigma_long, tau1, tau2):
    sigmaL = sigma_long[:model.L*2]
    sigmaR = sigma_long[model.L*2:]
    psiL = evolve(model, psi0, sigmaL, tau1, tau2)
    psiR= evolve(model, psi0, sigmaR, tau1, tau2)

    W = jnp.dot(jnp.conjugate(psiL.T), psiR)
    W_norm = jnp.linalg.norm(W)
    W_sign = W / W_norm
    return W_norm, W_sign


def make_W(psiL, psiR):
    W = jnp.dot(jnp.conjugate(psiL.T), psiR)
    W = jnp.linalg.det(W)
    W_norm = jnp.linalg.norm(W)
    W_sign = W / W_norm
    return W_norm, W_sign
 

make_W_vmapped = jax.vmap(make_W, in_axes = (0, 0), out_axes = (0, 0, 0))


def make_Eloc_old(model, psiL, psiR):
    psiL_up = psiL[:model.L, :]
    psiL_down = psiL[model.L:, :]
    psiR_up = psiR[:model.L, :]
    psiR_down = psiR[model.L:, :]

    S = jnp.dot(jnp.conjugate(psiL.T), psiR)
    S_inv = jnp.linalg.pinv(S)

    Sup = jnp.dot(jnp.conjugate(psiL_up.T), psiR_up)
    Sup_inv = jnp.linalg.pinv(Sup)
    Sdown = jnp.dot(jnp.conjugate(psiL_down.T), psiR_down)
    Sdown_inv = jnp.linalg.pinv(Sdown)

    Tmatr = model.get_Hfree()
    T_loc = jnp.dot(jnp.conjugate(psiL.T), jnp.dot(Tmatr, psiR))
    T_loc = jnp.trace(jnp.dot(S_inv, T_loc))

    U_loc = 0.
    for k in range(model.L):
        psiL_up_k = jnp.conjugate(psiL_up)[k,:]
        psiR_up_k = psiR_up[k,:]
        A_up = jnp.outer(psiL_up_k, psiR_up_k)
        k_spin_up = jnp.trace(jnp.dot(Sup_inv, A_up))
         
        psiL_down_k = jnp.conjugate(psiL_down)[k,:]
        psiR_down_k = psiR_down[k,:]
        A_down = jnp.outer(psiL_down_k, psiR_down_k)
        k_spin_down = jnp.trace(jnp.dot(Sdown_inv, A_down))

        U_loc += k_spin_up * k_spin_down

    Eloc = jnp.linalg.det(S) * (T_loc + model.U * U_loc)
    return Eloc


def make_Eloc(model, psiL, psiR):
    # ========================================================================================================================
    # Eloc = T + V 
    # V = U \sum_i (n_{i,uparrow} - 1/2) (n_{i,downarrow} - 1/2) = U \sum_i ( n_{i,uparrow} * n_{i,downarrow} - 1/2 * n_i + 1/4 )
    # n_{i, s} = M_{i,i}, where M = psiR_s * (psiL_s^{\dagger} * psiR_s)^{-1} * psiL_s^{\dagger}
    # ========================================================================================================================
    psiL_up = psiL[:model.L, :]
    psiL_down = psiL[model.L:, :]
    psiR_up = psiR[:model.L, :]
    psiR_down = psiR[model.L:, :]

    """
    psiL_up = psiL[:, :model.L]
    psiL_down = psiL[:, model.L:]
    psiR_up = psiR[:, :model.L]
    psiR_down = psiR[:, model.L:]
    """

    S = jnp.dot(jnp.conjugate(psiL.T), psiR)
    Sup = jnp.dot(jnp.conjugate(psiL_up.T), psiR_up)
    Sdown = jnp.dot(jnp.conjugate(psiL_down.T), psiR_down)

    Sup_inv = jnp.linalg.pinv(Sup)
    Sdown_inv = jnp.linalg.pinv(Sdown)

    G_up = jnp.dot(psiR_up, jnp.dot(Sup_inv, jnp.conjugate(psiL_up.T))) 
    G_down = jnp.dot(psiR_down, jnp.dot(Sdown_inv, jnp.conjugate(psiL_down.T))) 

    hopping_up  = jnp.diagonal(G_up, offset = 1) + jnp.diagonal(G_up, offset = -1)
    hopping_down  = jnp.diagonal(G_down, offset = 1) + jnp.diagonal(G_down, offset = -1)
    T_loc = jnp.sum(hopping_up) + jnp.sum(hopping_down)

    n_up = jnp.diagonal(G_up)
    n_down = jnp.diagonal(G_down)
    U_loc = jnp.sum(jnp.multiply(n_up, n_down)) 

    Eloc = (-model.t * T_loc + model.U * U_loc) * jnp.linalg.det(S)

    return Eloc


make_Eloc_vmapped = jax.vmap(make_Eloc, in_axes = (None, 0, 0), out_axes = 0)

   
def flip(sigma_long, key):
    # sigma_long has shape of (model.L*2, )
    L2 = sigma_long.shape[-1]
    isite = jax.random.choice(key = key, a = jnp.arange(L2))
    #print('isite: ', isite)
    sigma_new = sigma_long.at[isite].set(-sigma_long[isite]) 
    return sigma_new 


def flip_new(sigma_long, key):
    # sigma_long has shape of (model.L*2, )
    L2 = sigma_long.shape[-1]
    L = int(L2/2)
    keyL, keyR = jax.random.split(key, 2)
    isiteL = jax.random.choice(key = keyL, a = jnp.arange(L))
    isiteR = jax.random.choice(key = keyR, a = jnp.arange(L))
    sigma_new = sigma_long.at[jnp.array([isiteL, isiteR])].set(jnp.array([-sigma_long[isiteL], -sigma_long[isiteR]]))
    return sigma_new 


flip_vmapped = jax.vmap(flip, in_axes = (0, 0), out_axes = 0)



def random_init_sigma(L2, key):
    return jax.random.choice(key = key, a = jnp.array([1, -1]), shape = (L2, )) 

random_init_sigma_vmapped = jax.vmap(random_init_sigma, in_axes = (None, 0), out_axes = 0)


def metropolis(model, sigma_init, tau1, tau2, nthermal, nsample):
    sigma = sigma_init
    psi0 = model.get_psi0()[:, :model.N*2]
    #print('psi0:')
    #print(psi0)

    # ==================================================================================================
    # E_mean = A / B
    # A = \sum_{sigma', sigma} W_norm(sigma', sigma) eloc(sigma', sigma) W_sign(sigma', sigma)
    #   = \sum_{sigma', sigma} W_norm(sigma', sigma) <psi(sigma')| H |psi(sigma)> W_sign(sigma', sigma)
    # B = \sum_{sigma', sigma} W_norm(sigma', sigma) W_sign(sigma', sigma)
    # ==================================================================================================

    #sign_dot_Eloc_sampled = jnp.zeros(nsample)
    #sign_sampled = jnp.zeros(nsample)
    sign_dot_Eloc_sampled = complex(0., 0.)
    sign_sampled = complex(0., 0.)

    key = jax.random.PRNGKey(42)

    ninterval = 10
    for imove in range(nthermal + nsample * ninterval):
        #print('sigma:', sigma)
        sigmaL = sigma[:model.L]
        sigmaR = sigma[model.L:]
        psiL = evolve(model, psi0, sigmaL, tau1, tau2)
        psiR = evolve(model, psi0, sigmaR, tau1, tau2)

        Eloc = make_Eloc(model, psiL, psiR)
        W_norm, W_sign = make_W(psiL, psiR)
        #print('W_norm, W_sign:', W_norm, W_sign)

        key_uniform, key = jax.random.split(key, 2)
        sigma_proposal = flip(sigma, key)

        sigmaL_proposal = sigma_proposal[:model.L]
        sigmaR_proposal = sigma_proposal[model.L:]
        psiL_proposal = evolve(model, psi0, sigmaL_proposal, tau1, tau2)
        psiR_proposal = evolve(model, psi0, sigmaR_proposal, tau1, tau2)

        W_norm_proposal, W_sign_proposal = make_W(psiL_proposal, psiR_proposal)
        W_ratio = W_norm_proposal / W_norm
        ratio = min(1., W_ratio)

        proposal = jax.random.uniform(key_uniform)
        #print('proposal:', proposal)
        
        #if ratio > proposal:
        sigma = jnp.where(ratio > proposal, sigma_proposal, sigma)

        if (imove > nthermal) and ((imove - nthermal) % ninterval == 0):
            isample = int((imove - nthermal) / ninterval)
            #print('isample:', isample)
            #print('Eloc * W_sign:', Eloc * W_sign)
            #print('W_sign:', W_sign)
            #sign_dot_Eloc_sampled = sign_dot_Eloc_sampled.at[isample].set(Eloc*W_sign) 
            #sign_smapled = sign_sampled.at[isample].set(W_sign)
            sign_dot_Eloc_sampled += Eloc*W_sign
            sign_sampled += W_sign
            #print('sign_sampled:', sign_sampled)
 
    #print('sign_dot_Eloc_sampled:', sign_dot_Eloc_sampled)
    #print('sign_sampled:', sign_sampled)
    #print('sign_dot_Eloc_sampled_mean, sign_sampled_mean:', sign_dot_Eloc_sampled.mean(), sign_sampled.mean())
    #E_mean = sign_dot_Eloc_sampled.mean() / sign_sampled.mean() 
    E_mean = sign_dot_Eloc_sampled / sign_sampled
    return E_mean

      
def metropolis_vmapped(model, sigma_init, taus, nthermal, nsample):
    sigma = sigma_init  # sigma has shape (batch, L*2)
    batch = sigma.shape[0]
    psi0 = model.get_psi0()[:, :model.N*2]

    # ==================================================================================================
    # E_mean = A / B
    # A = \sum_{sigma', sigma} W_norm(sigma', sigma) eloc(sigma', sigma) W_sign(sigma', sigma)
    #   = \sum_{sigma', sigma} W_norm(sigma', sigma) <psi(sigma')| H |psi(sigma)> W_sign(sigma', sigma)
    # B = \sum_{sigma', sigma} W_norm(sigma', sigma) W_sign(sigma', sigma)
    # ==================================================================================================

    #sign_dot_Eloc_sampled = complex(0., 0.)
    #sign_sampled = complex(0., 0.)

    sign_dot_Eloc_sampled = []
    sign_sampled = []

    key = jax.random.PRNGKey(42)

    ninterval = 10
    for imove in range(nthermal + nsample * ninterval):
        print("imove:", imove)
        #print('sigma:', sigma)
        sigmaL = sigma[:, :model.L]
        sigmaR = sigma[:, model.L:]
        #psiL = evolve_vmapped(model, psi0, sigmaL, tau1, tau2)
        #psiR = evolve_vmapped(model, psi0, sigmaR, tau1, tau2)
        psiL = evolve_nlayer_vmapped(model, psi0, sigmaL, taus)
        psiR = evolve_nlayer_vmapped(model, psi0, sigmaR, taus)

        Eloc = make_Eloc_vmapped(model, psiL, psiR)
        W_norm, W_sign = make_W_vmapped(psiL, psiR)

        #print('Eloc:', Eloc)
        #print('W_norm:', W_norm)
        #print('W_sign:', W_sign)

        Eloc = Eloc / jnp.multiply(W_norm, W_sign)

        key_uniform, key_proposal, key = jax.random.split(key, 3)
        key_proposal = jax.random.split(key_proposal, batch)

        sigma_proposal = flip_vmapped(sigma, key_proposal)
        sigmaL_proposal = sigma_proposal[:, :model.L]
        sigmaR_proposal = sigma_proposal[:, model.L:]
        psiL_proposal = evolve_nlayer_vmapped(model, psi0, sigmaL_proposal, taus)
        psiR_proposal = evolve_nlayer_vmapped(model, psi0, sigmaR_proposal, taus)

        W_norm_proposal, W_sign_proposal = make_W_vmapped(psiL_proposal, psiR_proposal)
        W_ratio = W_norm_proposal / W_norm
        ratio = jnp.where(W_ratio < 1., W_ratio, 1.)
        #print('ratio:', ratio)
        proposal = jax.random.uniform(key_uniform, shape = (batch, ))
        #print('proposal:', proposal)   

        accept = ratio > proposal
        #print("accept:", accept)
        sigma = jnp.where(accept[:, None], sigma_proposal, sigma)

        if (imove > nthermal) and ((imove - nthermal) % ninterval == 0):
            #isample = int((imove - nthermal) / ninterval)
            sign_dot_Eloc_sampled.append(jnp.multiply(Eloc, W_sign))
            sign_sampled.append(W_sign)

    #print('sign_dot_Eloc_sampled:', sign_dot_Eloc_sampled)
    #print('sign_sampled:', sign_sampled)
    #print('sign_dot_Eloc_sampled_mean, sign_sampled_mean:', sign_dot_Eloc_sampled.mean(), sign_sampled.mean())
    #E_mean = sign_dot_Eloc_sampled.mean() / sign_sampled.mean() 

    sign_dot_Eloc_sampled = jnp.array(sign_dot_Eloc_sampled)
    sign_sampled = jnp.array(sign_sampled)

    E_mean = sign_dot_Eloc_sampled.sum() / sign_sampled.sum()
    sign_mean = sign_sampled.mean()

    sign_dot_Eloc_errorbar = sign_dot_Eloc_sampled.std()/jnp.sqrt(sign_dot_Eloc_sampled.size)
    sign_errorbar = sign_sampled.std()/jnp.sqrt(sign_sampled.size)
 
    return E_mean, sign_mean, sign_dot_Eloc_errorbar, sign_errorbar
 
# test ====================================================================

def make_direct(t, U, tau1, tau2):
    model = dimer(t, U)
    Hmatr = model.H
    Umatr = model.Umatr
    expU = jax.scipy.linalg.expm(-I*Umatr * tau1)
    Tmatr = model.Tmatr
    expT = jax.scipy.linalg.expm(-I*Tmatr * tau2)

    model_free = dimer(t, 0.)
    _, psi = model_free.eigs()
    psi0 = psi[:, 0]
    #print(psi0)
    #print('psi0 norm:')
    #print(jnp.linalg.norm(psi0))
    psi0_new = jnp.dot(expT, jnp.dot(expU, psi0))
    E = jnp.dot(jnp.conjugate(psi0_new.T), jnp.dot(Hmatr, psi0_new))
    E = E / jnp.dot(jnp.conjugate(psi0_new.T), psi0_new)
    return E


def make_direct_nlayer(t, U, taus):
    model = dimer(t, U)
    Hmatr = model.H
    Tmatr = model.Tmatr
    Umatr = model.Umatr
 
    model_free = dimer(t, 0.)
    _, psi = model_free.eigs()
    psi0 = psi[:, 0]
    #print(psi0)
    #print('psi0 norm:')
    #print(jnp.linalg.norm(psi0))

    lenth = taus.shape[-1]
    nlayer = int(lenth/2)
    for ilayer in range(nlayer):
        tau1 = taus[lenth - ilayer*2 - 1]
        tau2 = taus[lenth - ilayer*2 - 2]
        expT = jax.scipy.linalg.expm(-I * Tmatr * tau2)
        expU = jax.scipy.linalg.expm(-I * Umatr * tau1)
        psi0 = jnp.dot(expT, jnp.dot(expU, psi0))
 
    E = jnp.dot(jnp.conjugate(psi0.T), jnp.dot(Hmatr, psi0))
    E = E / jnp.dot(jnp.conjugate(psi0.T), psi0)
    return E


def test_Nspin():
    L = 2
    N = int(L/2)
    t = 1.
    U = 4.
    model = Hubbard_1d(L, N, t, U)
    psi0 = model.get_psi0()[:,:model.N*2]
    
    nspin_arr = jnp.hstack((jnp.ones(model.L), -jnp.ones(model.L)))
    sigma = jnp.array([1, -1])
    sigma = jnp.hstack((sigma, sigma))
    expU_diags = jnp.multiply(sigma, nspin_arr)
 
    expU = jnp.diag(expU_diags) 
    expU = jax.scipy.linalg.expm(expU)
    expU = jnp.dot(jnp.conjugate(psi0.T), jnp.dot(expU, psi0))
    expU = jnp.linalg.det(expU)
    print(expU)
   
 
def test_make_expU():
    L = 2
    N = int(L/2)
    t = 1.
    U = 4.
    model = Hubbard_1d(L, N, t, U)

    eigvals = model.get_eigvals()
    print(eigvals)

    psi0 = model.get_psi0()
    print(psi0)
    #eixt()
    psi0 = model.get_psi0()[:,:model.N*2]
    #print(psi0)
    #print(jnp.linalg.det(jnp.dot(jnp.conjugate(psi0.T), psi0)))

    tau = 1.
   
    expU = jnp.zeros((model.L*2, model.L*2))
    for sigma1 in [1, -1]:
        for sigma2 in [1, -1]:
            sigma = jnp.array([sigma1, sigma2])
            #print(make_expU(model, sigma, tau))
            expU += make_expU(model, sigma, tau) 

    #print(expU)
    #exit()
    expU = jnp.dot(jnp.conjugate(psi0.T), jnp.dot(expU, psi0))
    expU = jnp.linalg.det(expU)
    print(expU)

    #
    model_1 = dimer(t, U)
    Umatr_1 = model_1.Umatr
    expU_1 = jax.scipy.linalg.expm(-Umatr_1 * tau)
    model_1_free = dimer(t, 0.)
    _, psi_1 = model_1.eigs()
    psi0_1 = psi_1[:, 0]
    print("psi0 by ED:")
    print(psi0_1)
    #print('psi0_1 norm:')
    #print(jnp.linalg.norm(psi0_1))
    expU_1 = jnp.dot(jnp.conjugate(psi0_1.T), jnp.dot(expU_1, psi0_1))
    print(expU_1)

    
    """
    make_expU_vmapped = jax.vmap(make_expU, in_axes = (None, 0, None), out_axes = 0)
    sigma_vmapped = jnp.array([ [1, 1], [-1, -1], [1, -1] ])
    expU_vmapped = make_expU_vmapped(model, sigma_vmapped, tau)
    print(expU_vmapped)
    print(expU_vmapped.shape)
    """

def test_make_W():
    L = 2
    N = int(L/2)
    t = 1.
    U = 5.
    model = Hubbard_1d(L, N, t, U)
    psi0 = model.get_psi0()[:,:model.N*2]
    #print(psi0)
    #exit()

    tau1, tau2 = 1., 1.
    taus = 
    sigma_vmapped = jnp.array([ [1, 1], [1, -1], [1, -1] ])
    
    #evolve_vmapped = jax.vmap(evolve, in_axes = (None, None, 0, None, None), out_axes = 0)
    psi0_new = evolve_vmapped(model, psi0, sigma_vmapped, tau1, tau2)
    print(psi0_new)
    print(psi0_new.shape)

    #exit()

    psiL = psi0_new
    psiR = psi0_new

    print(make_W_vmapped(psiL, psiR))

    print(make_Eloc_vmapped(model, psiL, psiR))


def test_evolve_nlayer():
    L = 2
    N = int(L/2)
    t = 1.
    U = 2.
    model = Hubbard_1d(L, N, t, U)
    psi0 = model.get_psi0()[:,:model.N*2]
    print(psi0)
    #exit()

    tau1, tau2 = 1., 1.
    tau3, tau4 = 1., 1.

    taus = jnp.array([tau1, tau2, tau3, tau4])
    sigma_vmapped = jnp.array([ [1, 1], [1, -1], [1, -1] ])
    
    psi0_new = evolve_nlayer_vmapped(model, psi0, sigma_vmapped, taus)
    print(psi0_new)
    print(psi0_new.shape)

    #exit()

    """
    psiL = psi0_new
    psiR = psi0_new

    print(make_W_vmapped(psiL, psiR))

    print(make_Eloc_vmapped(model, psiL, psiR))
    """

def test_make_Eloc():
    L = 2
    N = int(L/2)
    t, U = 1., 2.
    model = Hubbard_1d(L, N, t, U)
    eigvals = model.get_eigvals()
    #print(eigvals)
    psi0 = model.get_psi0()
    #print(psi0) 

    #psi = psi0[:, :N*2]
    psi = jnp.vstack((psi0[:, 0], psi0[:, 1])).T  
    #print(psi)
    print(jnp.linalg.norm(psi))
    psi_norm = jnp.linalg.det(jnp.dot(jnp.conjugate(psi.T), psi))
 
    eloc = make_Eloc(model, psi, psi)
    eloc = eloc / psi_norm
    print(eloc)

    #
    model_ED_free = dimer(t, 0.)
    _, psi_ed = model_ED_free.eigs()
    psi0_ed = psi_ed[:, 0]
    #print(psi0_ed)
    #print('psi0 norm:', jnp.linalg.norm(psi0_ed))
    model_ED = dimer(t, U)
    Hmatr = model_ED.H
    Umatr = model_ED.Umatr
    E = jnp.dot(jnp.conjugate(psi0_ed.T), jnp.dot(Hmatr, psi0_ed))
    E = E / jnp.dot(jnp.conjugate(psi0_ed.T), psi0_ed)
    print(E)
 

def test_flip():
    sigma = jnp.array([1, 1, 1, 1])
    key = jax.random.PRNGKey(42)

    """
    for i in range(20):
        key_old, key = jax.random.split(key, 2)
        print(key)
        sigma_new = flip(sigma, key)
        print(sigma_new)
    """

    sigma_vmapped = jnp.array([ [1, 1], [-1, -1], [1, -1] ])
    key_vmapped = jax.random.split(key, 3)
    sigma_vmapped_new = flip_vmapped(sigma_vmapped, key_vmapped)
    print(sigma_vmapped_new)  
 

def test_random_init_sigma():
    L = 2
    L2 = L * 2
    key = jax.random.PRNGKey(42)
    sigma_init = random_init_sigma(L2, key)
    #print(sigma_init)

    batch = 5
    key_vmapped = jax.random.split(key, batch)
    sigma_init_vmapped = random_init_sigma_vmapped(L2, key_vmapped)
    print(sigma_init_vmapped)


def test_make_direct():
    t, U = 1., 2.5
    tau1, tau2 = 1., 1.
    E = make_direct(t, U, tau1, tau2)
    print(E)


def test_metropolis():
    L = 2
    N = int(L/2)
    t = 1.
    U = 4.
    model = Hubbard_1d(L, N, t, U)
    eigvals = model.get_eigvals()
    psi0 = model.get_psi0()

    tau1, tau2 = 1., 1.
   
    nthermal = 50
    nsample = 400

    key_init_sigma = jax.random.PRNGKey(21)
    sigma_init = random_init_sigma(model.L*2, key_init_sigma)   

    E_mean = metropolis(model, sigma_init, tau1, tau2, nthermal, nsample)
    print(E_mean)

    print(make_direct(t, U, tau1, tau2)) 
 

def test_metropolis_vmapped():
    L = 2
    N = int(L/2)
    t = 1.

    tau1, tau2 = 0.2, 0.2
    taus = jnp.array([tau1, tau2])

    #tau1, tau2, tau3, tau4 = 2., 2., 2., 2.
    #taus = jnp.array([tau1, tau2, tau3, tau4])

    nthermal = 100
    nsample = 10

    batch = 5000

    key_init = jax.random.PRNGKey(21)
    key_init = jax.random.split(key_init, batch)
    sigma_init = random_init_sigma_vmapped(L*2, key_init)   

    E_MC_real = []
    E_MC_imag = []
    E_ED_real = []
    E_ED_imag = []

    sign_real = []
    sign_imag = []

    sign_dot_Eloc_errorbar_real = []
    sign_dot_Eloc_errorbar_imag = []

    sign_errorbar_real = []
    sign_errorbar_imag = []

    start = time.time()
    for U in jnp.arange(0., 1.1, 1.):
        model = Hubbard_1d(L, N, t, U)
        E_MC, sign, sign_dot_Eloc_errorbar, sign_errorbar = metropolis_vmapped(model, sigma_init, taus, nthermal, nsample)
        E_ED = make_direct_nlayer(t, U, taus) 
        print("E_MC:", E_MC)
        print("E_ED:", E_ED)

        E_MC_real.append(E_MC.real)
        E_ED_real.append(E_ED.real)
        E_MC_imag.append(E_MC.imag)
        E_ED_imag.append(E_ED.imag)

        sign_real.append(sign.real)
        sign_imag.append(sign.imag)
      
        sign_dot_Eloc_errorbar_real.append(sign_dot_Eloc_errorbar.real)
        sign_dot_Eloc_errorbar_imag.append(sign_dot_Eloc_errorbar.imag)

        sign_errorbar_real.append(sign_errorbar.real)
        sign_errorbar_imag.append(sign_errorbar.imag)
 
    #
    end = time.time()
    #print("time:", end - start)

    #print(E_MC_real)
    #print(E_MC_imag)
    #print(E_ED_real)
    #print(E_ED_imag)

    #print(sign_real)
    #print(sign_imag)
    #exit() 

    datas = { "MC_real": E_MC_real, "ED_real": E_ED_real, "MC_imag": E_MC_imag, "ED_imag": E_ED_imag, \
              "sign_real": sign_real, "sign_imag": sign_imag, \
              "sign_dot_Eloc_errorbar_real": sign_dot_Eloc_errorbar_real, "sign_dot_Eloc_errorbar_imag": sign_dot_Eloc_errorbar_imag, \
              "sign_errorbar_real": sign_errorbar_real, "sign_errorbar_imag": sign_errorbar_imag }

    import pickle as pk   
    fp = open('./test_E_and_sign_t=0.2', 'wb')  
    pk.dump(datas, fp)   
    fp.close()

# run ======================================================================

#test_Nspin()
#test_make_expU()
#test_evolve()
#test_evolve_nlayer()
#test_make_Eloc()
#test_random_init_sigma()
#test_flip()
#test_make_direct()
#test_metropolis()
#test_metropolis_vmapped()



