import numpy as np
import matplotlib.pyplot as plt
import pickle
import cvxpy as cp
import networkx as nx

# defining kernel function


def kernel_matrix(x, y):
    # Euclidean Kernel
    n = len(x)
    m = len(y)
    K = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            K[i, j] = np.exp(-np.linalg.norm(x[i]-y[j])**2)
    return K


def kernel_im(xi, xm):
    # Euclidean Kernel
    # xi is a vector and xm is a list of vector
    n = len(xm)
    K = np.zeros(n)
    for j in range(n):
        K[j] = np.exp(-np.linalg.norm(xi-xm[j])**2)
    return K
