from utils import *
import time

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
    alpha_list = []
    alpha_mean_list = []
    a = len(x) # number of agents
    # stacked points
    # initial alpha random with gaussian distribution
    alpha = np.random.normal(0, 1, (a, m))
    alpha = np.array(alpha).reshape(a*m, 1)
    gradient = grad_alpha(sigma, mu, y, x, x_selected, alpha.reshape(a, m)).reshape(a*m, 1)
    W = np.array([[1/3, 1/3, 0, 0, 1/3], 
                  [1/3, 1/3, 1/3, 0, 0], 
                  [0, 1/3, 1/3, 1/3, 0], 
                  [0, 0, 1/3, 1/3, 1/3],
                  [1/3, 0, 0, 1/3, 1/3]] ) # 1/a*(np.ones((a, a)))  # define the weight matrix
    # define the kronecker product of the weight matrix
    W_bar = np.kron(W, np.eye(m))
    j = 0
    while j<max_iter: # np.linalg.norm(alpha.reshape(a, m)[0] - alpha_mean) > 0.001 and
        j += 1
        # for i in range(a):
        #     alpha[i] = W_bar * alpha[i]+ lr * gradient[i]
        #     gradient[i] = grad_alpha(x, y, sigma, alpha[i])
        alpha_new = W_bar @ alpha - lr * gradient
        # IMPORTANT : in grad_alpha alpha should be a 2D array
        gradient = W_bar @ gradient + grad_alpha(sigma, mu, y, x, x_selected, alpha_new.reshape(a, m)).reshape(a*m, 1) - \
            grad_alpha(sigma, mu, y, x, x_selected, alpha.reshape(a, m)).reshape(a*m, 1)
        alpha = alpha_new
        alpha_mean = np.mean(alpha.reshape(a, m), axis=0)
        alpha_list.append(alpha.reshape(a, m))
        alpha_mean_list.append(alpha_mean)
        # print(f'Iteration {j} : {np.linalg.norm(alpha_mean)}')
    
    alpha_optim = alpha.reshape(a, m)
    alpha_optim = np.mean(alpha_optim, axis=0)

    return alpha_optim, j, alpha_list

if __name__ == "__main__":
    ALPHA = False
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
    agent_x, agent_y, selected_points, x_selected, y_selected = get_agents_from_pickle('first_database.pkl', 5, 100, 10)
    print(f'Nb agents : {a}')
    print(f'Nb data points : {n}')
    print(f'Nb selected points : {m}')
    print(f'Points per agent : {n/a}\n')
    """
    # Data visualization
    # for j in range(a):
    #     plt.plot(agent_x[j], agent_y[j], 'o', label='Data')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    # Define the graph connection of the agents (a undirected star graph for now):
    # Gx = nx.star_graph(a-1).to_undirected()
    # nx.draw(Gx, with_labels=True)
    # plt.show()
    # Adj = nx.adjacency_matrix(Gx).todense()
    # print(Adj)

    if ALPHA == True:
        # Compute the alpha optimal
        print("Compute the alpha optimal....")
        sigma = 0.5
        start = time.time()
        alpha_optim = compute_alpha(x , y, x_selected, sigma)
        end = time.time()
        print(f'Time to compute alpha optimal : {end - start}\n')

        # Export alpha optimal to a file
        with open('alpha_optim.pkl', 'wb') as f:
            pickle.dump(alpha_optim, f)

    else :
        # Load alpha optimal from a file
        with open('alpha_optim.pkl', 'rb') as f:
            alpha_optim = pickle.load(f)

    print(f'alpha optimal : {alpha_optim}\n')


    # Compute the alpha optimal with the gradient tracking algorithm
    print("Compute the alpha optimal with the gradient tracking algorithm....")
    sigma = 0.5
    mu = 10
    lr = 0.002
    max_iter = 2000
    start = time.time()
    alpha_optim_gt, tot_ite, alpha_list = gradient_tracking(
        agent_x, agent_y, x_selected, m, sigma, mu, lr, max_iter)
    end = time.time()
    print(f'alpha optimal with gradient tracking : {alpha_optim_gt}')
    print(f'Time to compute alpha optimal with gradient tracking : {end - start}')
    print(f'Total iterations : {tot_ite}\n')

    # Data visualization
    # Y = np.linalg.norm(alpha_list - alpha_optim, axis=1)
    # unpack the list of alpha to get for each agent the evolution of alpha
    agent_1 = np.linalg.norm(np.array([alpha_list[i][0] for i in range(len(alpha_list))]) - alpha_optim, axis=1)
    agent_2 = np.linalg.norm(np.array([alpha_list[i][1] for i in range(len(alpha_list))]) - alpha_optim, axis=1)
    agent_3 = np.linalg.norm(np.array([alpha_list[i][2] for i in range(len(alpha_list))]) - alpha_optim, axis=1)
    agent_4 = np.linalg.norm(np.array([alpha_list[i][3] for i in range(len(alpha_list))]) - alpha_optim, axis=1)
    agent_5 = np.linalg.norm(np.array([alpha_list[i][4] for i in range(len(alpha_list))]) - alpha_optim, axis=1)

    plt.plot(agent_1, label='Agent 1')
    plt.plot(agent_2, label='Agent 2')
    plt.plot(agent_3, label='Agent 3')
    plt.plot(agent_4, label='Agent 4')
    plt.plot(agent_5, label='Agent 5')
    plt.xlabel('Iterations')
    plt.ylabel('Optimality gap (norm)') 
    plt.show()
    
  
    
"""
    