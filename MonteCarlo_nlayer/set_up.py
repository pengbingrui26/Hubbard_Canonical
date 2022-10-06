import jax
import jax.numpy as jnp
from free_model import Hubbard_1d 
from ED_dimer import dimer
from ED_1d import Hubbard_ED
from functools import partial
import time 

jax.config.update("jax_enable_x64", True)

I = complex(0., 1.)

# ================================================================================================

def init_fn(model):
    
    Tmatr = model.get_Hfree_half()
    expT0 = jax.scipy.linalg.expm(-I * Tmatr)

    # act on spin SD
    def _make_expU(spin, sigma, tau):
        # exp(-i*V*tau) = (1/2)^L * exp{-\tau*U*N/2} * \sum_{{sigma}} exp{alpha \sum_i \sigma_i (n_{i,up} - n_{i,down})}
        # alpha = arccosh(exp{i*tau*U/2})
        Gamma = jnp.exp(-I*tau*model.U*model.N*2/2) 
        alpha = jnp.arccosh(jnp.exp(I*tau*model.U/2))

        #nspin_arr = jnp.hstack((jnp.ones(model.L), -jnp.ones(model.L)))
        nspin_arr = spin * jnp.ones(model.L)
        V_diags = jnp.multiply(sigma, nspin_arr)
        expU_diags = jnp.power(0.5, model.L) * Gamma * jnp.exp(alpha * V_diags)

        return expU_diags

    # act on spin SD
    def _evolve(psi0, spin, sigmas, taus):
        # sigmas has the shape (nlayer, model.L)
        lenth = taus.shape[-1]
        nlayer = int(lenth/2)
        psi = psi0

        def body_fun(ilayer, psi):
            tau1 = taus[lenth - ilayer*2 - 1]
            tau2 = taus[lenth - ilayer*2 - 2]
            sigma = sigmas[ilayer]

            #expT = jax.lax.pow(expT0, tau2.astype(complex))
            expT = jax.scipy.linalg.expm(-I * Tmatr * tau2)
            expU_diag = _make_expU(spin, sigma, tau1)

            #psi = jnp.multiply(expU_diag, psi.T).T
            #psi = jnp.dot(expT, psi)

            return jnp.dot(expT, jnp.multiply(expU_diag, psi.T).T)

        #psi = jax.lax.fori_loop(0, nlayer, body_fun, psi.astype('complex'))

        for ilayer in range(nlayer):
            tau1 = taus[lenth - ilayer*2 - 1]
            tau2 = taus[lenth - ilayer*2 - 2]

            sigma = sigmas[ilayer]
            #expT = jax.lax.pow(expT0, tau2.astype(complex))
            expT = jax.scipy.linalg.expm(-I * Tmatr * tau2)
            expU_diag = _make_expU(spin, sigma, tau1)

            psi = jnp.multiply(expU_diag, psi.T).T
            psi = jnp.dot(expT, psi)

        return psi

    # act on full SD
    def make_W(psi0, sigmas_long, taus):
        # sigmas_long has the shape of (nlayer, model.L*2) 
        sigmasL, sigmasR = sigmas_long[:, :model.L], sigmas_long[:, model.L:]
        psi0_up, psi0_down = psi0[:model.L, :], psi0[model.L:, :]

        psi_up_L = _evolve(psi0_up, 1, sigmasL, taus)
        psi_down_L = _evolve(psi0_down, -1, sigmasL, taus)

        psi_up_R = _evolve(psi0_up, 1, sigmasR, taus)
        psi_down_R = _evolve(psi0_down, -1, sigmasR, taus)

        ##
        #psi_L = jnp.vstack((psi_up_L, psi_down_L))
        #psi_R = jnp.vstack((psi_up_R, psi_down_R))

        W_up = jnp.dot(jnp.conjugate(psi_up_L.T), psi_up_R)

        W_down = jnp.dot(jnp.conjugate(psi_down_L.T), psi_down_R)

        W = W_up + W_down
        W = jnp.linalg.det(W)

        return W.real, W.imag


    # act on full SD
    def make_Eloc_no_evolve(psiL, psiR, taus):
        psiL_up, psiL_down = psiL[:model.L, :], psiL[model.L:, :]
        psiR_up, psiR_down = psiR[:model.L, :], psiR[model.L:, :]
        
        #S = jnp.dot(jnp.conjugate(psiL.T), psiR)
        S = jnp.dot(jnp.conjugate(psiL_up.T), psiR_up) + jnp.dot(jnp.conjugate(psiL_down.T), psiR_down)
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
        return Eloc.real, Eloc.imag

 
    # act on full SD
    def make_Eloc(psi0, sigmas_long, taus):
        # sigmas_long has shape (nlayer, model.L*2)
        sigmasL = sigmas_long[:, :model.L]
        sigmasR = sigmas_long[:, model.L:]

        psi0_up, psi0_down = psi0[:model.L, :], psi0[model.L:, :]
        
        psiL_up = _evolve(psi0_up, 1, sigmasL, taus)
        psiL_down = _evolve(psi0_down, -1, sigmasL, taus)

        psiR_up = _evolve(psi0_up, 1, sigmasR, taus)
        psiR_down = _evolve(psi0_down, -1, sigmasR, taus)

        psiL = jnp.vstack((psiL_up, psiL_down))
        psiR = jnp.vstack((psiR_up, psiR_down))

        #S = jnp.dot(jnp.conjugate(psiL.T), psiR)
        S = jnp.dot(jnp.conjugate(psiL_up.T), psiR_up) + jnp.dot(jnp.conjugate(psiL_down.T), psiR_down)
        S_inv = jnp.linalg.pinv(S)

        Sup = jnp.dot(jnp.conjugate(psiL_up.T), psiR_up)
        Sup_inv = jnp.linalg.pinv(Sup)
        Sdown = jnp.dot(jnp.conjugate(psiL_down.T), psiR_down)
        Sdown_inv = jnp.linalg.pinv(Sdown)

        Tmatr = model.get_Hfree()
        T_loc = jnp.dot(jnp.conjugate(psiL.T), jnp.dot(Tmatr, psiR))
        T_loc = jnp.trace(jnp.dot(S_inv, T_loc))

        U_loc = 0. + 1j * 0.

        def body_fun(k, uloc):
            psiL_up_k = jnp.conjugate(psiL_up)[k,:]
            psiR_up_k = psiR_up[k,:]
            A_up = jnp.outer(psiL_up_k, psiR_up_k)
            k_spin_up = jnp.trace(jnp.dot(Sup_inv, A_up))
         
            psiL_down_k = jnp.conjugate(psiL_down)[k,:]
            psiR_down_k = psiR_down[k,:]
            A_down = jnp.outer(psiL_down_k, psiR_down_k)
            k_spin_down = jnp.trace(jnp.dot(Sdown_inv, A_down))

            return uloc + k_spin_up * k_spin_down

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

        #U_loc = jax.lax.fori_loop(0, model.L, body_fun, U_loc)

        Eloc = jnp.linalg.det(S) * (T_loc + model.U * U_loc)
        return Eloc.real, Eloc.imag

    # act on full SD
    def make_eloc(psi0, sigmas_long, taus):
        # sigmas_long has shape (nlayer, model.L*2)
        Eloc_real, Eloc_imag = make_Eloc(psi0, sigmas_long, taus)
        Eloc = Eloc_real + Eloc_imag * I

        W_real, W_imag = make_W(psi0, sigmas_long, taus)
        W = W_real + W_imag * I

        eloc = Eloc / W
        return eloc.real, eloc.imag

    return make_W, make_Eloc, make_eloc
    #return make_W, make_eloc
   

# ========================================================================== 

def flip(sigmas_long, key):
    # sigmas_long has shape (nlayer, model.L*2)
    Lsite_full = sigmas_long.shape[-1]
    sigmas_reshaped = sigmas_long.flatten()
    L_all = sigmas_reshaped.shape[-1]
    isite = jax.random.choice(key = key, a = jnp.arange(L_all))
    sigmas_reshaped = sigmas_reshaped.at[isite].set(-sigmas_reshaped[isite]) 
    sigmas_new = sigmas_reshaped.reshape(-1, Lsite_full)
    return sigmas_new

flip_vmapped = jax.vmap(flip, in_axes = (0, 0), out_axes = 0)


def random_init_sigma(Lsite_full, nlayer, key):
    return jax.random.choice(key = key, a = jnp.array([1, -1]), shape = (nlayer, Lsite_full)) 

#random_init_sigma_vmapped = jax.vmap(random_init_sigma, in_axes = (None, None, 0), out_axes = 0)


def random_init_sigma_vmapped(batch, nlayer, Lsite_full, key):
    sigma_vmapped = jax.random.choice(key = key, a = jnp.array([1, -1]), \
                                        shape = (batch, nlayer, Lsite_full)) 
    return sigma_vmapped
   



# test ====================================================================

def make_direct_nlayer(L, N, t, U, taus):
    model = Hubbard_ED(L, N, t, U)
    Hmatr = model.get_T()
    Tmatr = model.get_U()
    Umatr = model.get_Hamiltonian()
 
    model_free = Hubbard_ED(L, N, t, U)
    _, psi = model_free.eigs()
    #psi0 = psi[:, 0]
    #print(psi0)
    #print('psi0 norm:')
    #print(jnp.linalg.norm(psi0))

    lenth = taus.shape[-1]
    nlayer = int(lenth/2)

    Es = []
    for i in range(psi.shape[-1]):
        psi0 = psi[:, i]
        for ilayer in range(nlayer):
            tau1 = taus[lenth - ilayer*2 - 1]
            tau2 = taus[lenth - ilayer*2 - 2]
            expT = jax.scipy.linalg.expm(-I * Tmatr * tau2)
            expU = jax.scipy.linalg.expm(-I * Umatr * tau1)
            psi0 = jnp.dot(expT, jnp.dot(expU, psi0))
 
        E = jnp.dot(jnp.conjugate(psi0.T), jnp.dot(Hmatr, psi0))
        E = E / jnp.dot(jnp.conjugate(psi0.T), psi0)
        Es.append(E)

    return Es


def test_make_W():
    L = 2
    N = int(L/2)
    t, U = 1., 2.
    model = Hubbard_1d(L, N, t, U)
    psi0 = model.get_ground_state()

    psi_norm = jnp.linalg.det(jnp.dot(jnp.conjugate(psi0.T), psi0))
 
    sigma_long = jnp.array([1, -1, -1, 1])
    taus = jnp.array([1., 1.])

    make_W, _ = init_fn(model)
    W = make_W(psi0, sigma_long, taus)

    print("W:", W)

def test_make_Eloc():
    L = 4
    N = int(L/2)
    t, U = 1., 0.
    model = Hubbard_1d(L, N, t, U)
    psi0 = model.get_ground_state()

    psi_norm = jnp.linalg.det(jnp.dot(jnp.conjugate(psi0.T), psi0))
    print("psi_norm:")
    print(psi_norm)
 
    #Eloc = make_Eloc_old_no_evolve(model, psi0, psi0)
    #print("Eloc:", Eloc)
    #Eloc = Eloc / psi_norm
    #print("Eloc:", Eloc)

    sigma_long = jnp.ones(L*2)
    taus = jnp.array([0., 0.])

    _, make_Eloc, _ = init_fn(model)
    Eloc, _ = make_Eloc(model, psi0, sigma_long, taus)
    print("Eloc:", Eloc)
    Eloc = Eloc / psi_norm
    print("Eloc:", Eloc)

    model_ED = Hubbard_ED(L, N, t, U)
    eigvals, _ = model_ED.eigs()
 
    print("E_ED:", eigvals[0])



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
    print(sigma_init)

    batch = 5
    key_vmapped = jax.random.split(key, batch)
    sigma_init_vmapped = random_init_sigma_vmapped(L2, key_vmapped)
    print(sigma_init_vmapped)


def test_make_direct():
    t, U = 1., 2.5
    tau1, tau2 = 1., 1.
    E = make_direct_nlayer(t, U, tau1, tau2)
    print(E)



# run ======================================================================

#test_Nspin()
#test_make_expU()
#test_evolve()
#test_evolve_nlayer()
#test_make_W()
#test_make_Eloc()
#test_random_init_sigma()
#test_flip()
#test_make_direct()
#test_metropolis()
#test_metropolis_vmapped()



