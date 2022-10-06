import jax
import jax.numpy as jnp

class Hubbard_ED(object):
    def __init__(self, L, N, t, U):
        self.Lsite = L # number of sites
        self.N = N # number of particles of each spin
        self.t = t # hopping amptitude
        self.U = U # on-site repulsive potential

    def Hilbert_space_size(self):
        from itertools import combinations
        up = combinations(range(self.Lsite), self.N)
        up = [ list(xx) for xx in up ]
        return len(up) * len(up)
 
    def get_basis_non_ordered(self):
        from itertools import combinations
        up = combinations(range(self.Lsite), self.N)
        up = [ sorted(list(xx)) for xx in up ]
        up_basis = []
        for i, occ in enumerate(up):
            up_basis.append(occ)
        down_basis = up_basis
        basis = []
        for x in up_basis:
            for y in down_basis:
                y_new = [ (yy+self.Lsite) for yy in y ]
                basis.append( x+y_new )
        return basis

    def get_basis(self):
        basis = self.get_basis_non_ordered()
        occ_dic = {}
        for double_occ in range(self.N+1):
            occ_dic[double_occ] = []

        for ib, ba in enumerate(basis):
            num_double_occ = len([ xx for xx in ba[:self.N] if (xx+self.Lsite) in ba[self.N:] ])
            assert num_double_occ <= self.N+1, num_double_occ
            occ_dic[num_double_occ].append(ib)      
     
        basis_idx = []
        for double_occ in range(self.N+1):
            basis_idx = basis_idx + occ_dic[double_occ]
 
        basis_ordered = []
        for iba in basis_idx:
            basis_ordered.append(basis[iba])
        
        return basis_ordered
  

    def hopping(self, state, x, dire): 
        assert dire in [1, -1]
        assert x in range(self.Lsite)
        assert type(state) == list, state
        assert len(state) == self.N, len(state)

        spin = ''
        if all( [uu in range(self.Lsite) for uu in state] ):
            spin = 'up'
        elif  all( [dd in range(self.Lsite, self.Lsite*2) for dd in state] ):
            spin = 'down'
        assert spin in ['up', 'down'], spin

        state_new = state.copy()
        parity = None

        if spin == 'down':
            state_new = [ (xx - self.Lsite) for xx in state_new ]

        x1 = x+dire
        x11 = (x+dire) % self.Lsite

        if (x not in state_new) or (x11 in state_new):
            state_new = None
        else:
            alpha = x11
            beta = x
            l = state_new.index(beta) + 1
            parity1 = l-1
            state_new.remove(beta)
            state_new = sorted(state_new + [alpha])
            s = state_new.index(alpha) + 1
            parity2 = s-1 
            #parity3 = 1
            ##if x1 < 0 or x1 >= self.Lx or y1 < 0 or y1 >= self.Ly:
            #if (x1, y1) not in self.idx.values():
            #    parity3 = -1
            import math
            #parity = int(math.pow(-1, parity1 + parity2)*parity3)
            parity = int(math.pow(-1, parity1 + parity2))
            if spin == 'down':
                state_new = [ (xx + self.Lsite) for xx in state_new ]

        return state_new, parity


    def all_hoppings(self, state):
        assert len(state) == self.N*2, len(state)
        up = state[:self.N]
        down = state[self.N:]
 
        state_hopped = []
        for x in range(self.Lsite):
            for dire in [1, -1]:
                tmp_up, parity_up = self.hopping(up, x, dire)
                tmp_down, parity_down = self.hopping(down, x, dire)
 
                up_hopped, down_hopped  = '', ''
                if tmp_up == None:
                    assert parity_up == None
                    up_hopped = None
                else:
                    up_hopped = tmp_up + down
                if tmp_down == None:
                    assert parity_down == None
                    down_hopped = None
                else:
                    down_hopped = up + tmp_down

                up_hopped = (up_hopped, parity_up)
                down_hopped = (down_hopped, parity_down)

                if up_hopped not in state_hopped:
                    state_hopped.append( up_hopped )
                if down_hopped not in state_hopped:
                    state_hopped.append( down_hopped )

        return state_hopped

 
    def get_T(self):
        #basis = self.get_basis_non_ordered()
        basis = self.get_basis()
        overlap_matrix = {} 

        for ib, bas in enumerate(basis):
            bas_hopped = self.all_hoppings(bas)
            bas_hopped = [ xx for xx in bas_hopped if xx != (None, None) ]
            overlap = []
            for (xx, parity) in bas_hopped:
                for ibb, bass in enumerate(basis):
                    if bass == xx:
                        overlap.append( (ibb, parity) )
            overlap_matrix[ib] = overlap

        T_matr = jnp.zeros((len(basis), len(basis)))
        for ii in overlap_matrix:
            for (jj, jj_parity) in overlap_matrix[ii]:
                #T_matr[ii][jj] = -self.t * jj_parity
                T_matr = T_matr.at[ii, jj].set(-self.t * jj_parity)
        
        #return T_matr, overlap_matrix  
        return T_matr
    
    def get_U(self):
        #basis = self.get_basis_non_ordered()
        basis = self.get_basis()

        def count_double_occ(state): # count the number of double occupation 
            assert len(state) == self.N*2
            return len([ i for i in state[:self.N] if (i+self.Lsite) in state[self.N:] ])

        U_matr = jnp.zeros((len(basis), len(basis)))
        for ib, ba in enumerate(basis):
            #U_matr[ib][ib] = count_double_occ(ba) * self.U
            U_matr = U_matr.at[ib, ib].set(count_double_occ(ba) * self.U)
        return U_matr
 
    def get_Hamiltonian(self):
        #T_matr, T_dic = self.get_T()
        T_matr = self.get_T()
        U_matr = self.get_U()
        return T_matr + U_matr

    def eigs(self):
        hamiltonian = self.get_Hamiltonian()
        eigvals, eigvecs = jnp.linalg.eigh(hamiltonian)
        sorted_idx = jnp.argsort(eigvals)
        eigvecs = eigvecs[:, sorted_idx]
        return eigvals, eigvecs

    def free_energy(self, beta):
        E, _ = self.eigs()       
        F = 0.
        Z = sum([ jnp.exp(-beta*ee) for ee in E ])
        for i in range(len(E)):
            p = jnp.exp(-beta * E[i]) / Z
            f = p * (1/beta * jnp.log(p) + E[i])
            F += f
        return F



 
