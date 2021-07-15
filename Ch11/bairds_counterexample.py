import copy

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

# GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BairdsCounterexample:
    """
    Baird's Counterexample as described in Fig 11.1.
    """

    def getInitialState(self):
        """
        Returns a random initial state.
        @rtype: int
            An integer 0 - 6 representing the state.
        """
        return random.randint(0, 6)

    def getNextState(self, state, action):
        """ The state transition function

            The action is represented as 0 or 1, representing
            the dashed and solid action.

            @type state: int
                An integer 0 - 6 representing the current state.
            @type action: int
                An integer 0, 1 representing the action.
            @rtype: int
                An integer 0 - 6 representing the next state.

        """
        if action == 0:
            x = random.randint(0, 5)
            return x if x < state else x+1
        else:
            return 6


class Agent:
    """ Agent object that uses semi-gradient Q learning
    to estimate the Q function.
    """
    def __init__(self, bairds):
        """
        Initializes the agent.

        @type bairds: BairdsCounterexample
            The counterexample environment.
        """
        self.bairds = bairds

    def train(self, steps, alpha, gamma, b_dashed):
        """ Trains the agent using off-policy semi-gradient Q learning.

            @type steps: int
                The number of steps to train.
            @type alpha: float
                The step size.
            @type gamma: float
                A number between 0 and 1 that is multiplied to
                beta at each step.
            @type b_dashed
                A number between 0 and 1 representing the
                probability of choosing the dashed action
                under the behavioral policy.
        """
        # Initialize weights
        w = np.ones(8)
        w[6] = 10

        # This is computed by taking the expectation of the state
        # value on the next state and then extracting the corresponding
        # state vector.
        convert_to_vector = {}
        for state in range(7):
            for action in range(2):
                if action == 1:
                    vec = np.zeros(8)
                    vec[6] = gamma
                    vec[7] = 2 * gamma
                else:
                    vec = gamma * np.full(8, 1/3)
                    if state == 6:
                        vec[7] = gamma
                        vec[6] = 0
                    else:
                        vec[6] -= gamma*1/6
                        vec[7] += gamma*5/6
                        vec[state] -= gamma*1/3

                convert_to_vector[(state, action)] = vec

        state = self.bairds.getInitialState()

        weight_history = []

        for step in range(steps):
            # Choosing action based on behavioral policy:
            if random.uniform(0, 1) < b_dashed:
                action = 0
            else:
                action = 1

            next_state = self.bairds.getNextState(state, action)

            target = gamma * max(np.dot(convert_to_vector[(next_state, 0)], w),
                                 np.dot(convert_to_vector[(next_state, 1)], w))

            sa_vector = convert_to_vector[(state, action)]
            w += alpha * (target - np.dot(sa_vector, w)) * sa_vector

            weight_history.append(copy.deepcopy(w))

            state = next_state

        # Plotting w values as a function of training step
        plt.figure()
        weight_history = np.array(weight_history).T
        for i in range(len(weight_history)):
            plt.plot(weight_history[i], label = f'w_{i+1}')
        plt.xlabel('step')
        plt.ylabel('weights')
        plt.legend()
        plt.savefig('weights')
        plt.show()


if __name__ == "__main__":

    # Training parameters
    steps = 1000
    alpha = 0.01
    gamma = 0.99

    # Policy parameter
    b_dashed = 6/7 # This cannot be 0

    counterexample = BairdsCounterexample()

    agent = Agent(counterexample)

    agent.train(steps, alpha, gamma, b_dashed)


