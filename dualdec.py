from utils import *
import numpy as np

# def compute_alpha(x, y, x_selected, sigma):
#     n = len(x)
#     m = len(x_selected)
#     Kmm = kernel_matrix(x_selected, x_selected)
#     Knm = kernel_matrix(x[0:n], x_selected)
#     alpha_exact = np.linalg.inv(
#         sigma**2*Kmm + np.eye(m)*mu + np.transpose(Knm) @ Knm) @ np.transpose(Knm) @ y[0:n]
#     return alpha_exact

def coef_edge(i, j, adj_matrix):
    if adj_matrix[i, j]==0:
        return 0
    elif adj_matrix[i, j]==1 and i>=j:
        return 0
    else:
        return -1
    
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

def dualDec(x, y, selected_points, selected_points_agent, K, sigma, mu, lr, W, max_iter=1000):

    return alpha_optim, alpha_list


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

    