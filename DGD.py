import numpy as np
import matplotlib.pyplot as plt
import pickle
import cvxpy as cp
import networkx as nx
from utils import kernel_matrix, kernel_im, compute_alpha, get_agents_from_pickle, grad_alpha


def DGD_revisited(x, mu, sigma, a, adjacency_matrix,alpha_opt,lr):
    # stacked points
    alpha = np.zeros(len(x)) # initial alpha
    n = len(x)
    W = 1/a*(adjacency_matrix)  # define the weight matrix
    # define the kronecker product of the weight matrix
    W_bar = np.kron(W, np.eye(n))
    j = 0
    while np.linalg.norm(alpha - alpha_opt) > 0.001 and j<1000: :
        j += 1
        alpha = W_bar * alpha+ lr * grad_alpha(x, mu, sigma, alpha)
    return alpha_optim
