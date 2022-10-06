import jax
import jax.numpy as jnp
from free_model import Hubbard_1d 
from ED_1d import Hubbard_ED
import time 

from set_up import init_fn

jax.config.update("jax_enable_x64", True)

I = complex(0., 1.)


# =============================================================================


def make_direct(L, N, t, U, taus):
    model = Hubbard_ED(L, N, t, U)
    
    Tmatr = model.get_T()
    Umatr = model.get_U()
    Hmatr = model.get_Hamiltonian()

    model_free = Hubbard_ED(L, N, t, 0.)
    es, psi = model_free.eigs()
    #print("es:")
    #print(es)
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
        psi0 = jnp.dot(expU, jnp.dot(expT, psi0))

    E = jnp.dot(jnp.conjugate(psi0.T), jnp.dot(Hmatr, psi0))
    E = E / jnp.dot(jnp.conjugate(psi0.T), psi0)
    return E


def test_evolve():
    L = 2
    N = int(L/2)
    t, U = 1., 5.
    model = Hubbard_1d(L, N, t, U)
    psi0 = model.get_ground_state()

    psi_norm = jnp.linalg.det(jnp.dot(jnp.conjugate(psi0.T), psi0))
    #print("psi_norm:")
    #print(psi_norm)
 
    sigma_long = jnp.array([1, 1, -1, 1])
    taus = jnp.array([1., 1.])

    evolve, make_W, _, _ = init_fn(model)

    W = make_W(psi0, sigma_long, taus)
    #print("W:", W)


def test_make_W():
    L = 2
    N = int(L/2)
    t, U = 1., 5.
    model = Hubbard_1d(L, N, t, U)
    psi0 = model.get_ground_state()

    psi_norm = jnp.linalg.det(jnp.dot(jnp.conjugate(psi0.T), psi0))
    #print("psi_norm:")
    #print(psi_norm)
 
    sigma_long = jnp.array([1, 1, -1, 1])
    taus = jnp.array([1., 1.])

    make_W, _, _ = init_fn(model)

    W = make_W(psi0, sigma_long, taus)
    print("W:", W)


def test_make_Eloc():
    L = 4
    N = int(L/2)
    t, U = 1., 5.
    model = Hubbard_1d(L, N, t, U)
    psi0 = model.get_ground_state()

    psi_norm = jnp.linalg.det(jnp.dot(jnp.conjugate(psi0.T), psi0))
    #print("psi_norm:")
    #print(psi_norm)
 
    sigma_long = jnp.array([1, 1, -1, -1] * 2)
    taus = jnp.array([0.7, 0.2])

    _, make_Eloc, _ = init_fn(model)

    Eloc, _ = make_Eloc(psi0, sigma_long, taus)

    print("Eloc:", Eloc)
    #Eloc = Eloc / psi_norm
    #print("Eloc:", Eloc)

    #E_ED = make_direct(L, N, t, U, taus)
    #print("E_ED:", E_ED)


# run =====================================================================

#test_make_W()
test_make_Eloc()

