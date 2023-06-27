
import numpy as np

def local_newton_raphson(f, J, x0, epsilon):
    '''
    Find roots of non-linear equations using the local Newton-Raphson method

    f - function
    J - Jacobian matrix
    x0 - initial point
    epsilon - tolerance

    x - root
    k - number of iterations
    '''

    # Initialization
    x = x0
    k = 0
    
    j = J(x)

    f_x = f(x)
    f_norm = np.linalg.norm(f_x)
    while f_norm > epsilon:
        k += 1
        x = x - np.linalg.solve(j, f_x)
        j = J(x)
        f_x = f(x)
        f_norm = np.linalg.norm(f_x)
        print('Iteration', k, ':', x)
    return x, k

def f(x):
    return np.array([
        x[0]**2 + x[1]**2 - 4,
        4*x[0]**2 - x[1]**2 - 4
        ])

def J(x):
    return np.array([
        [2*x[0], 2*x[1]],
        [8*x[0], -2*x[1]]
        ])

x0 = np.array([1, 1])
epsilon = 1e-4
x, k = local_newton_raphson(f, J, x0, epsilon)