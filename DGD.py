import numpy as np
import matplotlib.pyplot as plt
import pickle
import cvxpy as cp
import networkx as nx
from utils import kernel_matrix, kernel_im


def DGD_revisited(x, mu, sigma, a, adjacency_matrix):
    # stacked points
    y_0 = x
    n = len(x)
    W = 1/a*(adjacency_matrix)  # define the weight matrix
    # define the kronecker product of the weight matrix
    W_bar = np.kron(W, np.eye(n))

    return alpha_optim
