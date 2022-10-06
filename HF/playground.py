import numpy as np
import jax.numpy as jnp
import jax.scipy as scipy 
I = 1

class Model():
	L = 4
	U = 1

tau = 0.05

Umatr_1 = jnp.array([ [ -U/2., 0, 0, 0 ], \
               [ 0, -U/2., 0, 0 ], \
               [ 0, 0, U/2., 0 ], \
               [ 0, 0, 0, U/2. ] ])

xpU_1 = jax.scipy.linalg.expm(-Umatr_1 * tau)


def make_expU(model, sigma, tau): 
	# exp(-i V tau) = Gamma^L * exp(alpha * \sum_i sigma_i * (n_{i,up} - n_{i, down}))
	# Gamma = 1/2 * exp(-i tau U/4)
	# alpha = arccosh(exp(i tau U/2))
	Gamma = 1/2 * jnp.exp(-I*tau*model.U/4)
	#print('x:', I*tau*model.U/2) 
	alpha = jnp.arccosh(jnp.exp(I*tau*model.U/2))
	
	#print('Gamma:', Gamma)
	#print('alpha:', alpha)
	
	nspin_arr = jnp.hstack((jnp.ones(model.L), -jnp.ones(model.L)))
	sigma = jnp.hstack((sigma, sigma))
	U_diags = jnp.multiply(sigma, nspin_arr)
	U = jnp.diag(U_diags)
	expU = jnp.power(Gamma, model.L) * scipy.linalg.expm(alpha * U)
	
	return expU

m = Model()
sigma = np.random.randint(0,2,4)

print(make_expU(m,sigma,tau))
