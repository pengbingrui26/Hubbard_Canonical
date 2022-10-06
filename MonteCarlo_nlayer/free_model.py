import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

class Hubbard_1d(object):
    def __init__(self, L, N, t, U):
        self.L = L
        self.N = N
        self.t = t
        self.U = U

    def get_Hfree_half(self):
        H_free_half = -self.t*jnp.eye(self.L, k = 1) - self.t*jnp.eye(self.L, k = -1)
        H_free_half = H_free_half.at[0, self.L-1].set(-self.t)
        H_free_half = H_free_half.at[self.L-1, 0].set(-self.t)  
        return H_free_half
       
    def get_Hfree(self):
        H_free_half = self.get_Hfree_half()
        H_free = jnp.zeros((self.L*2, self.L*2))
        H_free = H_free.at[ :self.L, :self.L ].set(H_free_half)
        H_free = H_free.at[ self.L:, self.L: ].set(H_free_half)
        return H_free

    def get_eigs_half(self):
        H_free_half = self.get_Hfree_half()
        eigvals, eigvecs = jnp.linalg.eigh(H_free_half)
        sorted_idx = jnp.argsort(eigvals)
        eigvals = eigvals[sorted_idx]
        eigvecs = eigvecs[:, sorted_idx]
        return eigvals, eigvecs

    def get_eigvals(self):
        h_free = self.get_Hfree()
        eigvals, eigvecs = jnp.linalg.eigh(h_free)
        return jnp.sort(eigvals)

    def get_psi0_half(self):
        _, eigvecs_half = self.get_eigs_half()
        psi0_set = []
        npsi = eigvecs_half.shape[-1]
        import itertools
        list_idx = list(itertools.combinations(range(npsi), int(npsi/2)))
        for idx in list_idx:
            idx = list(idx)
            psi = eigvecs_half[:, idx]
            psi0_set.append(psi)
        return psi0_set 

    def get_ground_energy(self):
        eigvals_half, eigvecs_half = self.get_eigs_half()
        return jnp.sum(eigvals_half[:self.N]) * 2

    def get_ground_state(self):
        _, eigvecs_half = self.get_eigs_half()
        psi_up = eigvecs_half[:, :self.N]
        psi_down = eigvecs_half[:, :self.N]

        psi_full = jnp.zeros((self.L*2, self.N*2))
        psi_full = psi_full.at[:self.L, :self.N].set(psi_up)
        psi_full = psi_full.at[self.L:, self.N:].set(psi_down)
        return psi_full 

    def get_psi0_full(self):
        psi0_full_set = []
        psi0_half_set = self.get_psi0_half()
        n_psi0 = len(psi0_half_set)
        for i in range(n_psi0):
            for j in range(n_psi0):
                psi0_up = psi0_half_set[i]                
                psi0_down = psi0_half_set[j]
                #psi0_full = jnp.vstack((psi0_up, psi0_down))                
                psi0_full = jnp.zeros((self.L*2, self.N*2))
                psi0_full = psi0_full.at[:self.L, :self.N].set(psi0_up)
                psi0_full = psi0_full.at[self.L:, self.N:].set(psi0_down)
                psi0_full_set.append(psi0_full)
        return psi0_full_set

    def get_psi0_nset(self, n_psi0):
        psi0_full_set = []
        psi0_half_set = self.get_psi0_half()
        npsi0_half = len(psi0_half_set)
        stop = False
        for i in range(npsi0_half):
            if stop == True:
                break
            for j in range(npsi0_half):
                if len(psi0_full_set) == n_psi0:
                    stop = True
                    break
                psi0_up = psi0_half_set[i]                
                psi0_down = psi0_half_set[j]
                #psi0_full = jnp.vstack((psi0_up, psi0_down))                
                psi0_full = jnp.zeros((self.L*2, self.N*2))
                psi0_full = psi0_full.at[:self.L, :self.N].set(psi0_up)
                psi0_full = psi0_full.at[self.L:, self.N:].set(psi0_down)
                psi0_full_set.append(psi0_full)
        return psi0_full_set


    def get_psi0(self):
        h_free = self.get_Hfree()
        eigvals, eigvecs = jnp.linalg.eigh(h_free)
        idx = jnp.argsort(eigvals)
        eigvecs = eigvecs[:,idx]
        return eigvecs 


"""
    def get_U(self):
        return self.U * jnp.identity(self.L*2)

    def get_V(self):
        diags = jnp.hstack((jnp.ones(self.L), -jnp.ones(self.L)))
        return jnp.diag(diags)

    def get_Nspin_up(self):
        matr_init = jnp.zeros((self.L*2, self.L*2))
        Nspin_up = matr_init.at[:self.L, :self.L].set(jnp.identity(self.L))
        return Nspin_up

    def get_Nspin_down(self):
        matr_init = jnp.zeros((self.L*2, self.L*2))
        Nspin_down = matr_init.at[self.L:, self.L:].set(jnp.identity(self.L))
        return Nspin_down
"""





 
