from utils import *
import time
# from sinkhorn_knopp import sinkhorn_knopp as skp

def is_double_sto(A):
  """
  Vérifie si A est doublement stochastique
  """
  if A.shape[0]!=A.shape[1]:
    print("Mauvaise forme")
    return False
  reponse=True
  ligne = np.zeros(A.shape[0])
  col = np.zeros(A.shape[0])
  for i in range (A.shape[0]):
    ligne[i] = A[i,:].sum()
    if not np.isclose(A[i,:].sum(),1): 
      reponse=False
    col[i] = A[:,i].sum()
    if not np.isclose(A[:,i].sum(),1,rtol=0.01): 
      reponse = False
  if reponse == False:
    print("ligne : ",ligne)
    print("Colonne : ",col) 

  return reponse

# def create_W(liste,taille,auto=True):
#     """
#     liste des arrêtes
#     taille du graphe
#     auto : bool : diagonale mise à 1
#     """
#     sk = skp.SinkhornKnopp()
#     if auto==True:
#         res=np.eye(taille)
#     else:
#         res=np.zeros((taille,taille))
#     for i,j in liste: 
#         res[i,j]=1
#         res[j,i]=1
#     return sk.fit(res)


def visual_graph(liste_indice):
    G = nx.Graph()
    for i,j in liste_indice:
        G.add_edge(i+1, j+1)
    # explicitly set positions
    pos = {1: (-1, 0), 2: (np.cos(3*np.pi/5), np.sin(3*np.pi/5)), 3: (np.cos(np.pi/5), np.sin(np.pi/5))
           , 4: (np.cos(-np.pi/5), np.sin(-np.pi/5)), 5: (np.cos(-3*np.pi/5),np.sin(-3*np.pi/5))}
    options = {
    "font_size": 36,
    "node_size": 3000,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 1,
    "width": 1,
    }
    nx.draw_networkx(G, pos, **options)
    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.1)
    plt.axis("off")
    plt.show()
    return 0


def gradient_tracking(x, y, x_selected, sigma, mu, lr, W, max_iter=1000):
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
    m = len(x_selected) # number of selected points
    # stacked points
    # initial alpha random with 0
    alpha = np.zeros((a*m, 1))
    # alpha = np.array(alpha).reshape(a*m, 1)
    gradient = grad_alpha(sigma, mu, y, x, x_selected, alpha.reshape(a, m)).reshape(a*m, 1)
    # define the kronecker product of the weight matrix
    W_bar = np.kron(W, np.eye(m))
    j = 0
    while j<max_iter: # np.linalg.norm(alpha.reshape(a, m)[0] - alpha_mean) > 0.001 and
        j += 1
        alpha_new = W_bar @ alpha - lr * gradient
        # IMPORTANT : in grad_alpha alpha should be a 2D array
        gradient = (W_bar @ gradient) + (grad_alpha(sigma, mu, y, x, x_selected, alpha_new.reshape(a, m)).reshape(a*m, 1) - \
            grad_alpha(sigma, mu, y, x, x_selected, alpha.reshape(a, m)).reshape(a*m, 1))
        alpha = alpha_new
        alpha_mean = np.mean(alpha.reshape(a, m), axis=0)
        alpha_list.append(alpha.reshape(a, m))
        alpha_mean_list.append(alpha_mean)
        # print(f'Iteration {j} : {np.linalg.norm(alpha_mean)}')
    
    alpha_optim = alpha.reshape(a, m)
    alpha_optim = np.mean(alpha_optim, axis=0)
    return alpha_optim, j, alpha_list

if __name__ == "__main__":
    ALPHA = True

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
       'first_database.pkl', a, n, m)
    print(f'Nb agents : {a}')
    print(f'Nb data points : {n}')
    print(f'Nb selected points : {m}')
    print(f'Points per agent : {n/a}\n')

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

    print(f'alpha optimal : {alpha_optim}\n')


    # Compute the alpha optimal with the gradient tracking algorithm
    print("Compute the alpha optimal with the gradient tracking algorithm....")
    sigma = 0.5
    mu = 10
    lr = 0.002
    max_iter = 5000
    W = np.array([[1/3, 1/3, 0, 0, 1/3], 
                  [1/3, 1/3, 1/3, 0, 0], 
                  [0, 1/3, 1/3, 1/3, 0], 
                  [0, 0, 1/3, 1/3, 1/3],
                  [1/3, 0, 0, 1/3, 1/3]] ) 
    start = time.time()
    alpha_optim_gt, tot_ite, alpha_list = gradient_tracking(
        agent_x, agent_y, x_selected, sigma, mu, lr, W, max_iter)
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
    
  
    

    