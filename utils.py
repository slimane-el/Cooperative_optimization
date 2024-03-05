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
    K = np.zeros((n, 1))
    for j in range(n):
        K[j] = np.exp(-np.linalg.norm(xi-xm[j])**2)
    return K


def compute_alpha(x, y, x_selected, sigma):
    n = len(x)
    m = len(x_selected)
    Kmm = kernel_matrix(x_selected, x_selected)
    Knm = kernel_matrix(x[0:n], x_selected)
    alpha_exact = np.linalg.inv(
        sigma**2*Kmm + np.eye(m) + np.transpose(Knm) @ Knm) @ np.transpose(Knm) @ y[0:n]
    return alpha_exact


def get_agents_from_pickle(pickle_name, a, n, m):
    # summary :
    # a : number of agents
    # n : number of data points
    # m : number of selected points
    # pickle_name : name of the pickle file
    # Load the data
    with open(pickle_name, 'rb') as f:
        x, y = pickle.load(f)

    agent_x = []
    agent_y = []
    # Randomly select m points
    # the points should be shared between the different agents
    selected_points = np.random.choice(np.array(range(n)), m, replace=False)
    x_selected = x[selected_points]
    y_selected = y[selected_points]
    for j in range(a):
        agent_x.append(x[j*20:j*20+20])
        agent_y.append(y[j*20:j*20+20])

    # # Data visualization
    # for j in range(a):
    #     plt.plot(agent_x[j], agent_y[j], 'o', label='Data')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    return agent_x, agent_y, x_selected, y_selected


def grad_alpha(sigma, mu, y_agent, x_agent, x_selected, alpha):
    Kmm = kernel_matrix(x_selected, x_selected)
    a = len(x_agent)
    grad = [0 for i in range(a)]
    for i in range(a):
        grad[i] = (sigma**2*Kmm @ alpha[i] + mu*np.tranpose(alpha[i]))/a
        for j in range(len(x_agent[i])):
            grad[i] += - y_agent[i][j]*kernel_im(x_agent[i][j], x_selected) + (
                kernel_im(x_agent[i][j], x_selected)) @ np.tranpose(kernel_im(x_agent[i][j], x_selected)) @ alpha[i]
    return grad
