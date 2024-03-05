from utils import *

if __name__ == "__main__":
    
    # # Load the data x and y
    with open('first_database.pkl', 'rb') as f:
        x, y = pickle.load(f)

    print(x)
    print(x[0])
    test = kernel_im(x[0], x)
    print(type(test))
    print(test.shape)
    # print(np.expand_dims(test, axis=1).shape)
    print((test@np.transpose(test)).shape)
    # print(np.transpose(np.expand_dims(test, axis=1)) @ np.expand_dims(test, axis=1).shape)

    # # Data visualization
    # plt.plot(x, y, 'o', label='Data')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()
    """
    # Generate the data
    a = 5
    n = 100
    m = 10
    agent_x, agent_y, selected_points, x_selected, y_selected = get_agents_from_pickle('first_database.pkl', 5, 100, 10)

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