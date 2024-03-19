import numpy as np
import matplotlib.pyplot as plt
import pickle
import cvxpy as cp
import networkx as nx

# defining kernel function


def kernel_matrix(x, y):
    # Euclidean Kernel
    # x and y are numpy arrays
    n = len(x)
    m = len(y)
    K = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            K[i, j] = np.exp(-np.linalg.norm(x[i]-y[j])**2)
    return K


def kernel_im(xi, xm):
    # Euclidean Kernel
    # xi is a point and xm is a numpy array of points
    n = len(xm)
    K = np.zeros((1, n))
    for j in range(n):
        K[0][j] = np.exp(-np.linalg.norm(xi-xm[j])**2)
    # K should be of shape (1,n)
    return K


def compute_alpha(x, y, x_selected, sigma):
    n = len(x)
    m = len(x_selected)
    Kmm = kernel_matrix(x_selected, x_selected)
    Knm = kernel_matrix(x[0:n], x_selected)
    alpha_exact = np.linalg.inv(
        sigma**2*Kmm + np.eye(m) + np.transpose(Knm) @ Knm) @ np.transpose(Knm) @ y[0:n]
    return alpha_exact


def get_agents_from_pickle(pickle_name, a, n, m, plot=False):
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

    # Data visualization
    if plot:
        for j in range(a):
            plt.plot(agent_x[j], agent_y[j], 'o', label='Data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    # agent_x is a list of numpy arrays from x and agent_y is a list of numpy arrays from y
    # len of agent_x and agent_y are equal to a
    # x_selected is a numpy array of selected points from x and y_selected is a numpy array of selected points from y
    # selected_points is a numpy array of the indices of the selected points
    # x_selected and y_selected are of length m
    return agent_x, agent_y, x_selected, y_selected, selected_points


def grad_alpha(sigma, mu, y_agent, x_agent, x_selected, alpha):
    # summary :
    # sigma : float
    # mu : float
    # y_agent : list of numpy arrays
    # x_agent : list of numpy arrays
    # x_selected : numpy array
    # alpha : list of numpy arrays
    # return : numpy array of shape (a, m)
    Kmm = kernel_matrix(x_selected, x_selected)
    a = len(x_agent)
    grad = [0 for i in range(a)]  # list of numpy arrays
    for i in range(a):
        # print(f'alpha[i] shape : {alpha[i].shape}')
        # print(f'x_agent[i] shape should be n/nb agents (= 20): {x_agent[i].shape}')
        # print(f'x_selected shape should be m (=10): {x_selected.shape}')
        big_kernel_im = kernel_matrix(x_agent[i], x_selected)
        # print(f'big_kernel_im shape (should be 20, 10 ) : {big_kernel_im.shape}')
        big_kernel_im_transpose = np.transpose(big_kernel_im)
        # print(f'big_kernel_im_transpose shape : {big_kernel_im_transpose.shape}')
        grad[i] = (1.0/a) * (sigma**2 * Kmm + mu * np.eye(len(x_selected))) @ alpha[i] + \
            big_kernel_im_transpose @ (big_kernel_im @ alpha[i] - y_agent[i])
        # print(f'grad[i] shape : {grad[i].shape}')
    return grad


def grad_alpha2(sigma, mu, y_agent, x_agent, x_selected, alpha):
    Kmm = kernel_matrix(x_selected, x_selected)
    a = len(x_agent)
    grad = [0 for i in range(a)]  # list of numpy arrays
    for i in range(a):
        grad[i] = (sigma**2*Kmm @ alpha[i] + mu*alpha[i]) / a
        # print(f'\ngrad[i] shape : {grad[i].shape}')
        for j in range(len(x_agent[i])):
            # print(f'kernel_im shape (should be 1, 10 ) : {kernel_im(x_agent[i][j], x_selected).shape}')
            # print(f'kernel_im_transpose shape (should be 10, 1) : {np.transpose(kernel_im(x_agent[i][j], x_selected)).shape}')
            # print(f'alpha[i] shape : {alpha[i].shape}')
            term_toadd = - y_agent[i][j]*np.transpose(kernel_im(x_agent[i][j], x_selected)) + \
                np.transpose(kernel_im(
                    x_agent[i][j], x_selected)) * (kernel_im(x_agent[i][j], x_selected) @ alpha[i])
            grad[i] += term_toadd.squeeze()
    return np.array(grad).reshape(a, len(x_selected))


# main
if __name__ == "__main__":
    # Test kernel matrix :
    x = np.array([1, 1, 1])
    y = np.array([1, 1, 1, 1])
    test = kernel_matrix(x, y)
    print(test.shape == (3, 4))
    print(test == np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]))

    # Test kernel_im
    xi = 1
    xm = np.array([1, 1, 1, 1])
    test = kernel_im(xi, xm)
    print(test.shape == (1, 4))
    print(test)

    # test kernel matrix with x of size 1
    x = np.array([1])
    y = np.array([1, 1, 1, 1])
    test = kernel_matrix(x, y)
    print(test.shape == (1, 4))
    print(test)

    x_agent, y_agent, x_selected, y_selected, selected_points = get_agents_from_pickle(
        'first_database.pkl', a=5, n=100, m=10)

    # test grad_alpha
    sigma = 0.5
    mu = 1
    alpha = [np.zeros(10) for i in range(5)]
    grad = grad_alpha(sigma, mu, y_agent, x_agent, x_selected, alpha)
    # test grad_alpha2
    grad2 = grad_alpha2(sigma, mu, y_agent, x_agent, x_selected, alpha)
    # test grad_alpha and grad_alpha2 with 10e-1 precision
    grad[0] = np.round(grad[0], 1)
    grad2[0] = np.round(grad2[0], 1)
    grad[1] = np.round(grad[1], 1)
    grad2[1] = np.round(grad2[1], 1)
    print(f'TEST : {grad[0] == grad2[0]}')

    # time the two functions
    import time
    start = time.time()
    grad = grad_alpha(sigma, mu, y_agent, x_agent, x_selected, alpha)
    end = time.time()
    print(f'grad_alpha time : {end - start}')
    start = time.time()
    grad2 = grad_alpha2(sigma, mu, y_agent, x_agent, x_selected, alpha)
    end = time.time()
    print(f'grad_alpha2 time : {end - start}')
