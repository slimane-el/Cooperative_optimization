from utils import *


def gradient_tracking(x, y, x_selected, m, sigma, mu, lr, max_iter=1000):
    """
    This function implements the gradient tracking algorithm.

    Parameters
    ----------
    x : list of numpy array
        The x coordinates of the data points for each agent.
    y : list of numpy array
        The y coordinates of the data points for each agent.
    n : int
        The number of data points.
    m : int
        The number of selected data points.
    sigma : float
        The kernel parameter.
    lr : float
        The learning rate.
    alpha_optim : numpy array
        The optimal alpha.

    Returns
    -------
    alpha : numpy array
        The final alpha.

    """
    a = len(x)  # number of agents
    # stacked points
    alpha = [np.arange(m) for i in range(a)]  # initial alpha for each agent
    # initial gradient for each agent
    gradient = [np.arange(m) for i in range(a)]
    alpha = np.array(alpha).reshape(a*m, 1)
    gradient = grad_alpha(sigma, mu, y, x, x_selected,
                          alpha.reshape(a, m)).reshape(a*m, 1)
    W = np.array([[1/3, 1/3, 0, 0, 1/3],
                  [1/3, 1/3, 1/3, 0, 0],
                  [0, 1/3, 1/3, 1/3, 0],
                  [0, 0, 1/3, 1/3, 1/3],
                  [1/3, 0, 0, 1/3, 1/3]])  # 1/a*(np.ones((a, a)))  # define the weight matrix
    # define the kronecker product of the weight matrix
    W_bar = np.kron(W, np.eye(m))
    j = 0
    while np.linalg.norm(alpha.reshape(a, m)[0] - alpha_mean) > 0.0001 and j < 10000:
        j += 1
        # for i in range(a):
        #     alpha[i] = W_bar * alpha[i]+ lr * gradient[i]
        #     gradient[i] = grad_alpha(x, y, sigma, alpha[i])
        alpha_new = W_bar @ alpha - lr * gradient
        # IMPORTANT : in grad_alpha alpha should be a 2D array
        gradient = W_bar @ gradient + grad_alpha(sigma, mu, y, x, x_selected, alpha_new.reshape(a, m)).reshape(a*m, 1) - \
            grad_alpha(sigma, mu, y, x, x_selected,
                       alpha.reshape(a, m)).reshape(a*m, 1)
        alpha = alpha_new
        alpha_mean = np.mean(alpha.reshape(a, m), axis=0)
        alpha_list.append(alpha.reshape(a, m))
        alpha_mean_list.append(alpha_mean)
        # print(f'Iteration {j} : {np.linalg.norm(alpha_mean)}')

    alpha_optim = alpha.reshape(a, m)
    alpha_optim = np.mean(alpha_optim, axis=0)

    return alpha_optim, j, alpha_list


if __name__ == "__main__":

    # # Load the data x and y
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
    agent_x, agent_y, selected_points, x_selected, y_selected = get_agents_from_pickle(
        'first_database.pkl', 5, 100, 10)
    print(x.shape)
    print(y.shape)
    print(type(x_selected))
    print(x_selected.shape)
    print(x_selected)

    # Compute the alpha optimal
    sigma = 0.5
    mu = 0
    lr = 0.1
    alpha_optim = gradient_tracking(
        agent_x, agent_y, x_selected, m, sigma, mu, lr)

    """
     # Data visualization
    for j in range(a):
        plt.plot(agent_x[j], agent_y[j], 'o', label='Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # Compute the alpha optimal
    sigma = 0.5
    alpha_optim = compute_alpha(x , y, x_selected, sigma)
    

    # define the graph connection of the agents (a undirected star graph for now):
    Gx = nx.star_graph(a-1).to_undirected()
    nx.draw(Gx, with_labels=True)
    plt.show()
    Adj = nx.adjacency_matrix(Gx).todense()
    print(Adj)

    """
