from utils import *
import numpy as np


def coef_edge(i, j, adj_matrix):
    if adj_matrix[i, j]==0:
        return 0
    elif adj_matrix[i, j]==1 and i>=j:
        return 0
    else:
        return -1
    
def search_A(W):
    # Trouver la matrice A
    edges_list = []
    nb_agents = W.shape[0] # nombre de noeuds
    for i in range(nb_agents):
        for j in range(i):
            if W[i,j]>0:
                edges_list.append([i, j])
    E = len(edges_list) # nombre d'arêtes
    A_dd=np.zeros((m*E, m*nb_agents))
    for e in range(E) : # pour chaque arête
        edge = edges_list[e]
        A_dd[e*m:(e+1)*m, edge[0]*m:(edge[0]+1)*m] = np.eye(m)
        A_dd[e*m:(e+1)*m, edge[1]*m:(edge[1]+1)*m] = -np.eye(m)
    return A_dd
    
def solve_alpha_dualdec(x, y, selected_points, selected_points_agent, sigma, mu, K, adj_matrix, lamb):
    # lambda should be shape (E, 1) TODO which shape ??
    # TODO : reshape lamb ?
    #lamb = lamb.reshape
    # E is the set of edges 
    E = len(adj_matrix)
    print("lamb shape : ", lamb.shape)
    n = len(x)
    a = len(selected_points_agent)
    m = len(selected_points)
    Kmm = get_Kij(selected_points, selected_points, K)
    alpha = []
    for i in range(a):
        Kim = get_Kij(selected_points_agent[i], selected_points, K)
        coefs_edge = list(map(lambda i, j: coef_edge(i, j, adj_matrix), [i]*a, range(a))) # shape (1, ...)
        lambi = lamb[i, :]
        A = sigma**2 * Kmm + np.eye(m)*mu + np.transpose(Kim) @ Kim
        b = np.transpose(Kim) @ y[selected_points_agent[i]] - coefs_edge @ lambi 
        alpha.append(np.linalg.solve(A, b))
    return np.array(alpha)

def dualDec(x, y, selected_points, selected_points_agent, K, sigma, mu, lr, W, max_iter=1000, lamb0=0):
    # graph is the adjacency matrix
    # W is the weight matrix
    graph = 1 * (W>0)
    m = len(selected_points)
    a = len(selected_points_agent)
    for i in range(a):
        graph[i, i] = 0
    lambda_ij = lamb0*np.ones((a, a, m))
    alpha_mean_list = []
    alpha_list_agent = []
    A = search_A(W)
    for n_iter in tqdm(range(max_iter)):
        # Calcul de x_i_star pour tous les noeuds
        alpha_optim = np.zeros((a,m))
        for agent in range(a): 
            alpha_optim[agent, : ] = solve_alpha_dualdec(
                x, y, selected_points, selected_points_agent, sigma, mu,
                K, graph, lambda_ij[agent, : , : ])
        for i in range(a):
            for j in range(i):
                lambda_ij[i, j, : ] += lr * (alpha_optim[i, :] - alpha_optim[j, :])
        
        # f_res.append(f_a(x_star.mean(axis=0),A,sigma,K_mm,K_im,N,y))
        alpha_mean_list.append(alpha_optim.mean(axis=0))
        alpha_list_agent.append(alpha_optim)

    return alpha_optim, alpha_list_agent, alpha_mean_list


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

    sigma=0.5
    mu=1

    adj_matrix = np.array([[1/3, 1/3, 0, 0, 1/3],
                  [1/3, 1/3, 1/3, 0, 0],
                  [0, 1/3, 1/3, 1/3, 0],
                  [0, 0, 1/3, 1/3, 1/3],
                  [1/3, 0, 0, 1/3, 1/3]])
    adj_matrix *= 3.0

    # lamb is matrix shape (a, a)
    lamb = np.zeros((a, a))
    alphatest = solve_alpha_dualdec(
        x, y, selected_points, selected_points_agents, sigma, mu,
        K, adj_matrix, lamb)
    print(alphatest.shape)

    dualDec(
        x, y, selected_points, selected_points_agents,
        K, sigma, mu, 0.1, adj_matrix, max_iter=1000, lamb0=0
    )

    