
import numpy as np

def local_newton_raphson(f, g, h, x0, epsilon):
    '''
    minimze a quadratic approximation of a function.
    
    f - function
    g - gradient
    h - hessian
    x0 - initial point
    epsilon - tolerance

    x - root
    k - number of iterations
    '''
    x = x0
    k = 0
    g_x = g(x)
    g_norm = np.linalg.norm(g_x)
    h_x = h(x)
    while g_norm > epsilon:
        k += 1  
        x = x - np.linalg.solve(h_x, g_x)
        g_x = g(x)
        g_norm = np.linalg.norm(g_x)
        h_x = h(x)
        print('Iteration', k, ':', x)
    return x, k

def f(x):
    return lambda x : 100*x[0]**4 + 0.01*x[1]**4 

def g(x):
    return np.array([
        400*x[0]**3,
        0.04*x[1]**3
        ])

def h(x):
    return np.array([
        [1200*x[0]**2, 0],
        [0, 0.12*x[1]**2]
        ])

x0 = np.array([1, 1])
epsilon = 1e-4
x, k = local_newton_raphson(f, g, h, x0, epsilon)