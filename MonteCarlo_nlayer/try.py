import numpy as np
import jax
import jax.numpy as jnp

def func(arr):
    arr = np.dot(arr, np.array([1,2,4]))
    return arr

"""
arr_test = np.array([[1,2], [5,6], [7,8]])
print(arr_test)
print(arr_test.T)
print(func(arr_test))
print(arr_test)
"""

"""
arr_test = np.array([[1,2], [5,6], [7,8]])
print(id(arr_test))
arr_test = arr_test.T
#print(arr_test)
print(id(arr_test))
"""

def test01():
    a = jnp.array([1, 1, 1])
    b = jnp.array([2, 2, 2])
    tr = True
    a = jnp.where(tr, b, a)
    print(a)


def test02():
    a = jnp.array([1, 1, 1, 1, 1])
    a = a.at[jnp.array([0, 3])].set(jnp.array([-a[0], -a[3]]))
    print(a)
 
def test03():
    a = jnp.array([1, 1, 1])
    b = jnp.array([2, 2, 3])
    c = a / b
    print(c) 

def test04():
    a = jnp.array([[0.7, 1.2, 1.5], [2.1, 0.2, 0.9]])
    aa = jnp.array([[-1.7, -10.2, -1.5], [-2.1, -0.2, -0.9]])

    #b = jnp.array([True, False])
    tr1 = jnp.array([1, 12])
    tr2 = jnp.array([2, 11])
    b = tr1 < tr2
    print(b)

    c = jnp.where(b[:, None], a, aa)
    print(c) 

def test05():
    a = jnp.array([1, 1, 1])
    s = a.shape[-1]
    print(s)

def dot(matr, arr):
    arr = jnp.dot(matr, arr)
    return arr

def test06():
    matr = jnp.array([[1, 2], [-2, 3]])
    arr = jnp.array([10, 20])
    brr = dot(matr, arr)
    print(brr)
    print(arr)

def test07():
    arr = jnp.array([1., 2., 3.])
    arr = arr.at[1].set(complex(0., 2.5))
    print(arr)

def test08():
    I = complex(0., 1.)
    print(jnp.linalg.norm(I))
    arr = [ I, 1 + I, 2 + I, 3 + I ]
    arr = jnp.array(arr)
    #print(arr[-2:])
    norm = abs(arr)
    print(norm)


def test09():
    arr = jnp.array([ [1, 2], [3, 4] ])
    print(arr.mean())
    print(arr.size)

def test09():
    arr = jnp.array([ 100, 200 ])
    brr = jnp.array([ [1, 1], \
                      [10, 10],
                      [100, 100] ])
    crr = jnp.multiply(arr, brr)
    print(crr)

def test10():
    brr = jnp.array([ [1, 1], \
                      [10, 10],
                      [100, 100] ])
    crr = brr.flatten()
    print(crr)

def test10():
    brr = jnp.array([ [1, 2, 3, 4], \
                      [10, 20, 30, 40],
                      [100, 200, 300, 400] ])
    print(brr[:, [0, 2]])

def test11():
    arr = jnp.array([1, 1])
    brr = jnp.hstack((arr, arr, arr))
    print(brr)
    crr = arr ** 2
    print(crr)

def test12():
    arr = jnp.zeros((5, ))
    brr = jnp.zeros((5, 2)).T
    print(brr)
    print(arr)
    crr = jnp.multiply(brr, arr)
    print(crr)

def test13():
    sample = []
    arr = jnp.zeros((3, 2)) 
    sample.append(arr)
    sample.append(arr)
    sample.append(arr)
    sample.append(arr)
    arr = jnp.concatenate(sample)
    print(arr)


def test14():
    brr = jnp.array([ [1, 2, 3, 4], \
                      [10, 20, 30, 40],
                      [100, 200, 300, 400] ])
    m = brr.mean(axis = -1)
    s = brr.sum(axis = 0)
    print(m) 
    print(s)

def test15():
    arr = jnp.array([1, 2, 3, 4, 5, 6])
    print(arr.shape[0])
    brr = arr[0: :2]
    crr = arr[1: :2]
    print(brr)
    print(crr)

def test16():
    import itertools
    list1 = range(6)
    list2 = list(itertools.combinations(list1, 2))
    for a in list2:
        print(a)
        print(type(a))
        b = list(a)
        print(b)

def test17():
    l = [1, 2, 3, 4]
    print(l[:2])


def test18():
    arr = jnp.array([1.1, 2.1, 3.1])
    print(arr)
    brr = func(arr)
    print(arr)            
    
    #jax.lax.fori_loop(lower, upper, body_fun, init_val)


def test19():
    import itertools
    arr = range(10)
    arr = itertools.combinations(arr, 5)
    arr = list(arr)
    b = len(arr)
    print(b)
    print(b**2)


def test20():
    def body_fun(i, sampled):
        sampled = sampled.at[i,:].set(jnp.array([1, 1, 1])) 
        return sampled

    sampled = jnp.zeros((10, 3))
    sampled = jax.lax.fori_loop(0, 20, body_fun, sampled)
    print(sampled)

def test21():
    a = 1. + 1. * 1j
    b = 3. + 4. * 1j
    lis = [a, b]
    arr = jnp.array(lis) 
    arr_norm = abs(arr)
    print(arr_norm)
    arr_sign = arr / arr_norm
    print(arr_sign)
    c = 0.76507492 + 0.03546794 * 1j
    print(abs(c))



def fn(a):
    return a + 3


fn_pmapped = jax.pmap(fn, in_axes = 0, out_axes = 0)

def test22():
    arr = jnp.array([2, 3])
    brr = fn_pmapped(arr)
    print(brr)

def test23():
    arr = jnp.array([ [1, 1], [2, 2], [3, 3] ])
    brr = jnp.concatenate(arr, axis = 0)
    print(brr)
    crr = jnp.array([ [ [1, 1], [2, 2], [3, 3] ], \
                      [ [11, 11], [22, 22], [33, 33] ], \
                      [ [111, 111], [222, 222], [333, 333] ], \
                      [ [1111, 1111], [2222, 2222], [3333, 3333] ] ])
    print(crr.shape)
    drr = jnp.concatenate(crr, axis = 0)
    print(drr)

    err = jnp.split(crr, 2)
    print(err)


def test24():

# ======================================================================

#test01()
#test02()
#test03()
#test04()
#test05()
#test06()
#test07()
#test08()
#test09()
#test10()
#test11()
#test12()
#test13()
#test14()
#test15()
#test16()
#test17()
#test18()
#test19()
#test20()
#test21()
#test22()
test23()

