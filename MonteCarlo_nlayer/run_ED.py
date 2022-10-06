import jax
import jax.numpy as jnp

from ED_1d import Hubbard_ED


def free_energy():
    L = 6
    N = int(L/2)
    t = 1.
    U = 1.
    model = Hubbard_ED(L, N, t, U)
    beta = 0.25
    F = model.free_energy(beta)
    print("F:", F) 


def finite_T():
    L = 8
    N = int(L/2)
    t = 1.
    U = 1.
    model = Hubbard_ED(L, N, t, U)
   
    es, _ = model.eigs()

    import pickle as pk
    fd = open('./Es_txt', 'wb')
    pk.dump(es, fd)
    fd.close()


# run ==============================================

free_energy()
#finite_T()


 
