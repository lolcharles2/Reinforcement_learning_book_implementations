import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import cm

# GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MountainCar:
    """
    The mountain car environment as described in Example 10.1 of the book.
    """

    def __init__(self, g, a):
        """ Initializes the environment.

            @type g: float
                The gravitational strength.
            @type a: float
                The acceleration of the car.
        """

        self.g = g
        self.a = a

    def getNextState(self, state, action):
        """ The state transition function

            The action is represented as -1, 0, 1.
            -1 represents full throttle backwards.
            0 represents no throttle.
            1 represents full throttle forwards.

            @type state: tuple[float]
                A tuple (x, x_dot) representing the current state)
            @type action: int
                An integer -1, 0, 1 representing the action.
            @rtype: tuple(tuple(float))
                A tuple (new_x, new_x_dot), done representing the new state and
                if the episode is finished.

        """
        x, x_dot = state
        new_x_dot = max(-0.07, min(0.07, x_dot + self.a*action - self.g*np.cos(3*x)))
        new_x = max(-1.2, min(0.5, x + new_x_dot))

        # If car reaches left bound, set velocity to 0.
        if new_x == -1.2: new_x_dot = 0.0

        # If car reaches right bound, episode finishes.
        done = (new_x == 0.5)

        return (new_x, new_x_dot), done


class Agent:
    """ Agent object that uses semi-gradient SARSA(n) to find optimal
        policy.
    """

    def __init__(self, Car, NN, optimizer, criterion, grid_length, sigma_x, sigma_xdot):
        """ Initializes the agent.

            @type Car: MountainCar
                The MountainCar environment.
            @type NN: NeuralNet
                Neural network for computing the state-action values
            @type optimizer: optimizer
                Optimizer from the torch.optim module.
            @type criterion: criterion
                Criterion from the torch.nn module.
            @type grid_length: int
                The side length of the grid for discretizing the state space.
            @type sigma_x: float
                The standard deviation of the gaussian RBF in x.
            @type sigma_xdot: float
                The standard deviation of the gaussian RBF in x_dot.
        """
        self.Car = Car
        self.NN = NN
        self.op = optimizer
        self.criterion = criterion
        self.sigma_x = sigma_x
        self.sigma_xdot = sigma_xdot

        # Constructs a gaussian radial basis on a grid in the state space.
        self.x_vals = torch.zeros(grid_length**2)
        self.x_dot_vals = torch.zeros(grid_length**2)
        ind = 0
        for x in np.linspace(-1.2, 0.5, grid_length):
            for x_dot in np.linspace(-0.07, 0.07, grid_length):
                self.x_vals[ind] = x
                self.x_dot_vals[ind] = x_dot
                ind += 1

    def chooseAction(self, state, epsilon):
        """ Chooses action according to an epsilon greedy policy.
            @type state: tuple[float]
                A tuple (x, x_dot) representing the current state.
            @type epsilon: float
                Exploration probability.
            @rtype: int
                An integer -1, 0, 1 representing the action.
        """

        if random.uniform(0, 1) < epsilon:
            return random.randint(-1, 1)
        else:
            with torch.no_grad():
                _, action = torch.max(self.NN(self.state_to_basis(state)), 1)
                return int(action.item())-1

    def oneStep(self, state, action, target):
        """ Performs one training step on observed transition.
            @type state: tuple[int]
                A tuple (x,x_dot) denoting the state.
            @type action: int
                An integer -1, 0, 1 representing the action.
            @type target: float
                The target action value.
        """
        pred = self.NN(self.state_to_basis(state))

        with torch.no_grad():
            pred_target = torch.clone(pred)
            pred_target[0][action+1] = target

        loss = self.criterion(pred, pred_target)

        loss.backward()

        self.op.step()

        self.op.zero_grad()

    def train(self, episodes, n, epsilon):
        """ Trains the agent using on-policy semi-gradient SARSA(n).

            @type episodes: int
                The number of episodes to train.
            @type n: int
                The n parameter in SARSA(n).
            @type epsilon: float
                Exploration action probability.
        """
        tot = 0
        for episode in range(episodes):

            if (episode + 1) % 100 == 0:
                print(f'Episode {episode + 1}/{episodes} completed!')
                self.plotPolicy(episode + 1)
                self.plotTrajectory(episode + 1)
                torch.save(self.NN.state_dict(), 'mountain_car_NN_model')
                print(f'Average steps per episode: {tot/100}')
                tot = 0

            state = (random.uniform(-0.6, 0.4), 0)
            action = self.chooseAction(state, epsilon)
            states = [state]
            actions = [action]

            time = 0
            tau = 0
            T = float('inf')

            while tau < T - 1:

                if time < T:
                    state, done = self.Car.getNextState(state, action)

                    states.append(state)

                    if done:
                        T = time + 1
                        tot += T
                    else:
                        action = self.chooseAction(state, epsilon)
                        actions.append(action)

                tau = time - n + 1

                if tau >= 0:

                    G = tau - min(tau + n, T)
                    if tau + n < T:
                        with torch.no_grad():
                            G += self.NN(self.state_to_basis(states[tau + n]))[0][actions[tau + n]+1]

                    self.oneStep(states[tau], actions[tau], G)

                time += 1

    def state_to_basis(self, state):
        x, x_dot = state
        basis = torch.exp(-(self.x_vals - x) ** 2 / (2*self.sigma_x**2) - (self.x_dot_vals - x_dot) ** 2 / (2*self.sigma_xdot**2))
        return basis.view(1, -1).to(device)

    def plotPolicy(self, episode):
        """
        Plots and saves the current policy in x and y.
        """
        X, XDOT = np.meshgrid(np.linspace(-1.2, 0.6), np.linspace(-0.07, 0.07))

        cost_to_go = np.zeros(shape=np.shape(X))

        with torch.no_grad():
            for i in range(len(X)):
                for j in range(len(X[0])):
                    state = (X[i][j], XDOT[i][j])
                    value = -torch.max(self.NN(self.state_to_basis(state)))
                    cost_to_go[i][j] = value

        # Plotting and saving results
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, XDOT, cost_to_go, cmap='viridis', edgecolor='k', linewidth = 0.005, antialiased=False)
        ax.set_xlabel('x')
        ax.set_ylabel('x_dot')
        ax.set_zlabel('cost_to_go')
        ax.set_zlim(0)
        plt.title(f'Episode {episode}')
        plt.savefig('cost_to_go.png')
        plt.close()

    def plotTrajectory(self, episode):
        """
        Plots a sample trajectory x as a function of steps in one episode
        using the current policy.
        """
        xlocs = []
        actions = []
        x, xdot = random.uniform(-0.6, -0.4), 0.0
        done = False
        while not done:
            with torch.no_grad():
                _, action = torch.max(self.NN(self.state_to_basis((x, xdot))), 1)
                action = int(action) - 1
                actions.append(action)
                state, done = self.Car.getNextState((x, xdot), action)
                x, xdot = state
            xlocs.append(x)

        plt.figure()
        plt.plot(xlocs, label='Trajectory')
        plt.plot(actions, label='Action')
        plt.xlabel('Steps')
        plt.ylabel('x')
        plt.xlim(0)
        plt.axhline(y=-1.2, color='k', label='x bounds')
        plt.axhline(y=0.5, color='k')
        plt.legend()
        plt.title(f'Episode {episode}')
        plt.savefig('trajectory.png')
        plt.close()


if __name__ == "__main__":

    # Mountain car parameters
    g = 0.0025
    a = 0.001

    # Agent parameters
    grid_length = 25
    sigma_x = 1.7/grid_length
    sigma_xdot = 0.14/grid_length

    # Training parameters
    episodes = 10000
    n = 8
    epsilon = 0.0
    learning_rate = 0.08

    Car = MountainCar(g, a)
    model = nn.Linear(grid_length**2, 3).to(device)

    model.load_state_dict(torch.load('mountain_car_NN_model'))

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    car_agent = Agent(Car, model, optimizer, criterion, grid_length, sigma_x, sigma_xdot)

    car_agent.train(episodes, n, epsilon)

