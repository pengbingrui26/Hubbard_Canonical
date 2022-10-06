import jax
import jax.numpy as jnp
from free_model import Hubbard_1d
from ED_1d import Hubbard_ED

jax.config.update("jax_enable_x64", True)

# test =================================================================

def test_Nspin():
    L = 4
    N = int(L/2)
    t = 1.
    U = 2.
    model = Hubbard_1d(L, N, t, U)
    Nspin_up = model.get_Nspin_up()
    print(Nspin_up)
    Nspin_down = model.get_Nspin_down()
    print(Nspin_down)


def test_get_psi0_half():
    L = 4
    N = int(L/2)
    t = 1.
    U = 2.
    model = Hubbard_1d(L, N, t, U)
    _, eigvecs_half = model.get_eigs_half()
    print(eigvecs_half)
    print("\n")
    psi0_set = model.get_psi0_half()
    for psi0 in psi0_set:
        print(psi0)
       
def test_get_psi0_full():
    L = 6
    N = int(L/2)
    t = 1.
    U = 2.
    model = Hubbard_1d(L, N, t, U)
    _, eigvecs_half = model.get_eigs_half()
    #print(eigvecs_half)
    #print("\n")
    psi0_set = model.get_psi0_full()
    for psi0 in psi0_set:
        print(psi0)
 
def test_get_ground_energy():
    L = 4
    N = int(L/2)
    t = 1.
    U = 0.
    model = Hubbard_1d(L, N, t, U)
    gs_E = model.get_ground_energy()

    model_ED = Hubbard_ED(L, N, t, U)
    eigvals, _ = model_ED.eigs()
    
    print(gs_E)
    print(eigvals[0])     

# run ==================================================================

#test_Nspin()
#test_get_psi0_half()
#test_get_psi0_full()
test_get_ground_energy()
