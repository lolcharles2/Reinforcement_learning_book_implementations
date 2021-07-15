import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

# GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ServerQueue:
    """
    The server queuing task as described in Example 10.2.
    """

    def __init__(self, num_servers, customer_rewards, free_prob):
        """ Initializes the environment.

            @type num_servers: int
                The number of servers.
            @type customer_rewards: list[float]
                A list of the rewards of each customer
                if they are given access to a server.
                The reward paid by each customer is also
                equal to their priority.
            @type free_prob: float
                The probability of a busy server freeing up at each
                time step.
        """

        self.num_servers = num_servers
        self.customer_rewards = customer_rewards
        self.free_prob = free_prob

    def getInitialState(self):
        """
        Returns the initial state.
        @rtype: tuple[int]
            A tuple (free_servers, customer) representing the initial state.
        """
        return self.num_servers, random.randint(0, len(self.customer_rewards) - 1)

    def getNextState(self, state, action):
        """ The state transition function

            The action is represented as 0 or 1, representing
            whether to reject or accept the current customer.

            @type state: tuple[int]
                A tuple (num_free_servers, customer) representing the current state.
            @type action: int
                An integer 0, 1 representing the action.
            @rtype: tuple(tuple(float))
                A tuple (new_free_servers, next_customer) representing the new state
                and the reward received.

        """
        free_servers, customer = state

        busy_servers = self.num_servers - free_servers
        servers_freed = 0
        for server in range(busy_servers):
            if random.uniform(0, 1) < self.free_prob:
                servers_freed += 1

        free_servers += servers_freed

        next_customer = random.randint(0, len(self.customer_rewards) - 1)

        if free_servers == 0:
            return (0, next_customer), 0

        if action:
            return (free_servers - 1, next_customer), self.customer_rewards[customer]
        else:
            return (free_servers, next_customer), 0


class Agent:
    """ Agent object that uses differential semi-gradient SARSA to find optimal
        policy.
    """

    def __init__(self, server, NN, optimizer, criterion):
        """ Initializes the agent.

            @type server: ServerQueue
                Server queue environment.
            @type NN: NeuralNet
                Neural network for computing the state-action values
            @type optimizer: optimizer
                Optimizer from the torch.optim module.
            @type scheduler: lr_scheduler
                Learning rate scheduler from the torch.optim module.
            @type criterion: criterion
                Criterion from the torch.nn module.
        """
        self.server = server
        self.NN = NN
        self.op = optimizer
        self.criterion = criterion

        # Constructs a one-hot vector for each state.
        num_servers = server.num_servers
        num_customers = len(server.customer_rewards)
        self.state_to_basis = {}
        for free_servers in range(num_servers + 1):
            for customer in range(num_customers):
                basis = torch.zeros((num_servers + 1) * num_customers).to(device)
                basis[free_servers * num_customers + customer] = 1.0
                self.state_to_basis[(free_servers, customer)] = basis

    def chooseAction(self, state, epsilon):
        """ Chooses action according to an epsilon greedy policy.
            @type state: tuple[int]
                A tuple (free_servers, customer) representing the current state.
            @type epsilon: float
                Exploration probability.
            @rtype: int
                An integer 0, 1 representing the action.
        """

        if random.uniform(0, 1) < epsilon:
            return random.randint(0, 1)
        else:
            with torch.no_grad():
                _, action = torch.max(self.NN(self.state_to_basis[state]), 0)
                return int(action.item())

    def oneStep(self, state, action, target):
        """ Performs one training step on observed transition.
            @type state: tuple[int]
                A tuple (free_servers, customer) denoting the state.
            @type action: int
                An integer 0, 1 representing the action.
            @type target: float
                The target action value.
        """
        pred = self.NN(self.state_to_basis[state])

        with torch.no_grad():
            pred_target = torch.clone(pred)
            pred_target[action] = target

        loss = self.criterion(pred, pred_target)

        loss.backward()

        self.op.step()

        self.op.zero_grad()

    def train(self, steps, epsilon, beta):
        """ Trains the agent using on-policy differential semi-gradient SARSA.

            @type steps: int
                The number of steps to train.
            @type epsilon: float
                Exploration action probability.
            @type beta: float
                The step size in updating average reward.
            @type gamma: float
                A number between 0 and 1 that is multiplied to
                beta at each step.
        """

        state = self.server.getInitialState()
        action = self.chooseAction(state, epsilon)
        average_reward = 0.0

        for step in range(steps):

            if (step + 1) % 10000 == 0:
                torch.save(self.NN.state_dict(), 'server_queue_NN_model')
                print(f'Step {step + 1}/{steps} completed!')
                self.plotPolicy(step + 1)
                print(f'Average estimated reward: {average_reward:.3f}')

            next_state, reward = self.server.getNextState(state, action)
            next_action = self.chooseAction(next_state, epsilon)

            with torch.no_grad():
                target = reward - average_reward + self.NN(self.state_to_basis[next_state])[next_action]

                average_reward += beta * (target - self.NN(self.state_to_basis[state])[action])

            # Update network after reaching steady state
            if step > 5000:
                self.oneStep(state, action, target)

            state, action = next_state, next_action

    def plotPolicy(self, step):
        """
        Plots the current policy.
        """
        num_servers, num_customers = self.server.num_servers, len(self.server.customer_rewards)
        pi = np.zeros(shape=(num_servers, num_customers))

        with torch.no_grad():
            for free_servers in range(1, num_servers+1):
                for customer in range(num_customers):
                    vals = self.NN(self.state_to_basis[(free_servers, customer)])
                    if vals[0] == vals[1]:
                        action = 1
                    else:
                        _, action = torch.max(self.NN(self.state_to_basis[(free_servers, customer)]), 0)
                        action = action.item()
                    pi[free_servers-1, customer] = action

        plt.figure()
        plt.imshow(pi.T)
        plt.xlabel('Number of free servers')
        plt.ylabel('Customer priority')
        plt.xticks(ticks=range(0, 10), labels=range(1, 11))
        plt.yticks(ticks=range(num_customers), labels=self.server.customer_rewards)
        plt.colorbar(ticks=[0, 1])
        plt.title(f'Step {step}')
        plt.savefig('policy.png')
        plt.close()



if __name__ == "__main__":
    # Server Queue parameters
    num_servers = 10
    customer_rewards = [1, 2, 4, 8]
    free_prob = 0.06

    # Training parameters
    steps = 2000000
    epsilon = 0.1
    learning_rate = 0.02
    beta = 0.01

    server = ServerQueue(num_servers, customer_rewards, free_prob)

    model = nn.Linear((num_servers + 1) * len(customer_rewards), 2, bias=False).to(device)
    model.weight.data.fill_(0.0)

    model.load_state_dict(torch.load('server_queue_NN_model'))

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    server_agent = Agent(server, model, optimizer, criterion)

    server_agent.train(steps, epsilon, beta)

