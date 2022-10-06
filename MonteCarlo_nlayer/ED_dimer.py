import jax
import jax.numpy as jnp
#import sympy as sp

jax.config.update("jax_enable_x64", True)

class dimer(object):

    def __init__(self, t, U):
        self.t = t
        self.U = U

        self.H1 = jnp.array([ [ -U/2, 0, -t, -t ], \
               [ 0, -U/2, -t, -t ], \
               [ -t, -t, U/2, 0 ], \
               [ -t, -t, 0, U/2 ] ])

        self.hfree = jnp.array([ [ 0, 0, -t, -t ], \
               [ 0, 0, -t, -t ], \
               [ -t, -t, 0, 0 ], \
               [ -t, -t, 0, 0 ] ])

        self.Tmatr = jnp.array([ [ 0, 0, -t, -t ], \
               [ 0, 0, -t, -t ], \
               [ -t, -t, 0, 0 ], \
               [ -t, -t, 0, 0 ] ])

        self.Umatr1 = jnp.array([ [ 0., 0, 0, 0 ], \
               [ 0, 0., 0, 0 ], \
               [ 0, 0, U, 0 ], \
               [ 0, 0, 0, U ] ])

        self.Umatr = jnp.array([ [ -U/2., 0, 0, 0 ], \
               [ 0, -U/2., 0, 0 ], \
               [ 0, 0, U/2., 0 ], \
               [ 0, 0, 0, U/2. ] ])

        self.Tmatr_new = jnp.array([ [ 0, -t, -t, 0 ], \
               [ -t, 0, 0, -t ], \
               [ -t, 0, 0, -t ], \
               [ 0, -t, -t, 0 ] ])

        self.Umatr_new = jnp.array([ [ 0, 0, 0, 0 ], \
               [ 0, U, 0, 0 ], \
               [ 0, 0, U, 0 ], \
               [ 0, 0, 0, 0 ] ])

        self.H = self.Tmatr + self.Umatr1
        #self.H = self.Tmatr_new + self.Umatr_new


    def eigs(self):
        E, V = jnp.linalg.eigh(self.H)
        sorted_idx = jnp.argsort(E)
        V = V[:, sorted_idx]
        return E, V

    def GS(self):
        E, V = self.eigs()
        idx = jnp.argsort(E)
        return V[:, idx[0]]

    def gwf(self, g):
        t = self.t
        H_free = jnp.array([ [ 0, 0, -t, -t ], \
               [ 0, 0, -t, -t ], \
               [ -t, -t, 0, 0 ], \
               [ -t, -t, 0, 0 ] ])
        E_free, V_free = jnp.linalg.eigh(H_free)
        idx = jnp.argsort(E_free)
        GS_WF = V_free[:, idx[0]]
        #print('GS_WF:', GS_WF)
        Gutz_weight = jnp.array([ 1, 1, g, g ] )
        Gutz_WF = jnp.multiply(Gutz_weight, GS_WF)
        return Gutz_WF

    def qgt(self, g):
        matr_double_occ = jnp.array([ [0,0,0,0], \
                                     [0,0,0,0], \
                                     [0,0,1,0], \
                                     [0,0,0,1] ])
        grad_g = jnp.power(g, -1) * matr_double_occ
        grad_g_square = jnp.dot(grad_g, grad_g)
        gwf = self.gwf(g)
        basis = jnp.array([1,1,1,1])
        A = jnp.dot(gwf, jnp.dot(grad_g_square, gwf)) / jnp.dot(gwf, gwf)
        b = jnp.dot(gwf, jnp.dot(grad_g, gwf)) / jnp.dot(gwf, gwf)
        B = b * b
        qgt = A - B
        return qgt    

    def free_energy(self, beta):
        E, _ = self.eigs()       
        F = 0.
        Z = sum([ jnp.exp(-beta*ee) for ee in E ])
        for i in range(4):
            p = jnp.exp(-beta * E[i]) / Z
            f = p * (1/beta * jnp.log(p) + E[i])
            F += f
        return F
    


