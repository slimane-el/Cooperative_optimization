import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from utils import *
# ADMM Distributed Algorithm


def compute_alpha_admm(x, y, selected_points, selected_points_agent, sigma, mu, K, lamb, z, Beta, adj_matrix, l):
    n = len(x)
    a = adj_matrix.shape[0]
    m = len(selected_points)
    Kmm = get_Kij(selected_points, selected_points, K)
    Kim = get_Kij(selected_points_agent, selected_points, K)
    A = sigma**2 * Kmm/5 + np.eye(m)*mu/5 + np.transpose(Kim) @ Kim
    b = np.transpose(Kim) @ y[selected_points_agent]
    for j in range(a):
        if adj_matrix[l, j] != 0 and l != j:
            A += Beta * np.eye(m)
            b += Beta * z[l, j, :] - lamb[l, j, :]
    return (np.linalg.solve(A, b))


def ADMM(mu, sigma, adjacency_matrix, y, x, selected_points, selected_points_agent, K, Beta, n_iter):
    # we intialize the lambda(i,j) and the z(i,j) to zero
    m = len(selected_points)
    a = len(selected_points_agent)
    n = len(x)
    lamb = np.zeros((a, a, m))
    z = np.zeros((a, a, m))
    alpha = np.zeros((a, m))
    list_alpha = []
    for k in range(n_iter):
        for i in range(a):
            alpha[i] = compute_alpha_admm(x, y, selected_points, selected_points_agent[i],
                                          sigma, mu, K, lamb, z, Beta, adjacency_matrix, i)
        for i in range(a):
            for j in range(a):
                if adjacency_matrix[i, j] > 0:
                    z[i, j, :] = (alpha[i] - alpha[j])/2 + \
                        (lamb[i, j, :]+lamb[j, i, :])/(2*Beta)
                    lamb[i, j, :] = lamb[i, j, :] + \
                        Beta * (alpha[i] - z[i, j, :])
        # print(lamb)
        list_alpha.append(alpha)
    return (alpha, list_alpha)


# main
if __name__ == "__main__":
    with open('first_database.pkl', 'rb') as f:
        x, y = pickle.load(f)
    # # Data visualization
    # plt.plot(x, y, 'o', label='Data')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()
    # Generate the data
    a = 5
    n = 100
    m = 10
    agent_x, agent_y, x_selected, y_selected, selected_points, selected_points_agents, K, x, y = get_agents_from_pickle(
        'first_database.pkl', a, n, m)
    print(f'Nb agents : {a}')
    print(f'Nb data points : {n}')
    print(f'Nb selected points : {m}')
    print(f'Points per agent : {n/a}\n')

    sigma = 0.5
    mu = 0.1

    # Compute the alpha optimal
    print("Compute the alpha optimal....")
    start = time.time()
    alpha_optimal = compute_alpha(x, y, x_selected, sigma)
    end = time.time()
    print(f'Time to compute alpha optimal : {end - start}\n')
    # # Export alpha optimal to a file
    # with open('alpha_optim.pkl', 'wb') as f:
    #     pickle.dump(alpha_optim, f)

    # create the weight matrix
    ind = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    W = create_W(ind, 5, auto=False)
    print(W)
    visual_graph(ind)
    # W = np.array([[1/3, 1/3, 0, 0, 1/3],
    #               [1/3, 1/3, 1/3, 0, 0],
    #               [0, 1/3, 1/3, 1/3, 0],
    #               [0, 0, 1/3, 1/3, 1/3],
    #               [1/3, 0, 0, 1/3, 1/3]])
    print("TEST MATRICE DOUBLE STO : ", is_double_sto(W))
    # Compute the alpha optimal with the dual decomposition algorithm
    start = time.time()
    Beta = 0.2
    n_iter = 10000
    alpha_optim, alpha_list_agent = ADMM(
        mu, sigma, W, y, x, selected_points, selected_points_agents, K, Beta, n_iter)
    end = time.time()
    print(f'Time to compute alpha optimal with ADMM : {end - start}\n')
    print(f'alpha optimal : {alpha_optimal}\n')
    print(f'alpha optimal using ADMM : {np.mean(alpha_optim,axis=0)}\n')
