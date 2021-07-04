import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import collections

class Env:
    """ Implementation of the enviornment described in Fig 8.8.

        0 is the starting state, N_states is the ending state.
    """
    def __init__(self, N_states, b):
        """ Initializes the enviornment.

            @type N_states: int
                Number of states
            @type b: int
                Branching factor
        """
        
        self.N_states = N_states
        self.b = b
        
        self.transitions = np.random.randint(N_states, size = (N_states, 2, b))

        self.rewards = np.random.normal(0, 1, size = (N_states, 2, b))

        
    def resetTask(self):
        """ Resets the task enviornment """

        self.transitions = np.random.randint(self.N_states, size = (self.N_states, 2, self.b))
        self.rewards = np.random.randn(self.N_states, 2, self.b)

                    
    def getNextState(self, state, action):
        """ Given the current state and an action, gets the next state.

            On each transition, there is a 0.1 probability of termination.

            @type state: int
                The current state.
            @type action: int
                The chosen action
            @rtype: tuple
                A tuple of (next_state, reward)
        """

        if random.uniform(0, 1) < 0.1:
            return self.N_states, 0
        else:
            c = np.random.randint(self.b)
            
            return self.transitions[state, action, c], self.rewards[state, action, c]
        

class Agent:
    """ An agent that performs uniform updates by cycling through all
        state-action pairs, updating each in place

    """
    def __init__(self, N_states, b, env):
        """ Initializes the agent

            @type N_states: int
                The number of states.
            @type transitions: dict[tuple] -> tuple
                A dictionary that maps (state, action) to a list of possible (next_state, reward)
            @type env: Env
                Enviornment that the agent will interact with.
        """

        self.N_states = N_states
        self.b = b
        self.Q = np.zeros((N_states, 2))
        self.env = env

    def reset(self):
        """ Resets the Q values back to 0 """
        self.Q = np.zeros((N_states, 2))

        
    def trainUniform(self, runs, iterations, eval_s0_interval):
        """ Evaluates the Q function for a number of iterations.
            The evaluation will be performed by sweeping through all
            state action pairs repeatly.
            
            @type runs: int
                The number of runs to average performance over.
            @type iterations: int
                The number of expected updates to perform per run.
            @type eval_s0_interval: int
                The number of expected updates inbetween before
                evaluating the state value of the starting state (1)
                under the current greedy policy.
            @rtype: tuple[array]
                Two arrays iteration_vals, avg_s0_value. avg_s0_value[i] denotes the average
                starting state value following the greedy policy at iteration iteration_vals[i].
        """

        iteration_vals = [iteration for iteration in range(0, iterations, eval_s0_interval)]

        avg_s0_value = np.zeros(len(iteration_vals))
        
        print("Evaluating uniform training...")

        for run in range(runs):

            s0_vals = []
            iteration = 0

            # Reset agent and enviornment before each run
            self.reset()
            self.env.resetTask()
            
            while iteration < iterations:
                for state in range(self.N_states):
                    for action in [0,1]:

                        # Performing expected update
                        self.Q[state, action] =  0.9 * np.mean(self.env.rewards[state, action] + np.max(self.Q[self.env.transitions[state, action]], axis = 1))

                        if iteration % eval_s0_interval == 0:
                            s0_vals.append(self.evaluateStartState())
                            
                        iteration += 1

                        if iteration >= iterations: break
                        
                    if iteration >= iterations: break

            print("Run {} completed!".format(run+1))

            avg_s0_value += np.array(s0_vals) / runs

        return iteration_vals, avg_s0_value

    def trainOnPolicy(self, runs, iterations, eval_s0_interval, epsilon):
        """ Evaluates the Q function for a number of iterations.
            The evaluation will be performed by following an
            epsilon greedy policy based on the current estimate of the
            Q function.
            
            @type runs: int
                The number of runs to average performance over.
            @type iterations: int
                The number of expected updates to perform per run.
            @type eval_s0_interval: int
                The number of expected updates inbetween before
                evaluating the state value of the starting state (1)
                under the current greedy policy.
            @type epsilon: float
                The probability of exploration.
            @rtype: tuple[array]
                Two arrays iteration_vals, avg_s0_value. avg_s0_value[i] denotes the average
                starting state value following the greedy policy at iteration iteration_vals[i].
        """

        iteration_vals = [iteration for iteration in range(0, iterations, eval_s0_interval)]

        avg_s0_value = np.zeros(len(iteration_vals))

        print("Evaluating on-policy training...")
        
        for run in range(runs):

            # Reset agent and enviornment before each run
            self.reset()
            self.env.resetTask()
            
            s0_vals = []
            iteration = 0

            state = 0
            
            while iteration < iterations:

                    action = self.chooseAction(state, epsilon)

                    # Performing expected update
                    
                    self.Q[state, action] =  0.9 * np.mean(self.env.rewards[state, action] + np.max(self.Q[self.env.transitions[state, action]], axis = 1))
                    
                    if iteration % eval_s0_interval == 0:
                        s0_vals.append(self.evaluateStartState())

                    # Moving to next state. If terminal, restart at 0.
                    state, _ = self.env.getNextState(state, action)
                    if state == self.N_states: state = 0
                            
                    iteration += 1
                        

            print("Run {} completed!".format(run+1))
            
            avg_s0_value += np.array(s0_vals) / runs

        return iteration_vals, avg_s0_value

    def chooseAction(self, state, epsilon):
        """ Chooses an action based on an epsilon-greedy policy using the current
            Q function.

            @type state: int
                The current state
            @type epsilon: float
                The probability of exploration.
            @rtype: int
                The chosen action.
        """
        if self.Q[state][0] == self.Q[state][1]:
            return random.choice([0,1])
        
        if random.uniform(0, 1) < epsilon:
            return int(np.argmin(self.Q[state]))
        else:
            return int(np.argmax(self.Q[state]))
        
    def evaluateStartState(self):
        """ Evaluates the state value of the state 0 based on the current
            greedy policy using Monte Carlo.

        """
    
        rewards = []
        
        for _ in range(1000):
            reward = 0
            state = 0
            while state != self.N_states:
                
                action = np.argmax(self.Q[state])
                
                state, r = self.env.getNextState(state, action)
                
                reward += r
                            
            rewards.append(reward)

        return np.mean(rewards)
            
        
if __name__ == "__main__":

    # Number of states
    N_states = 10000

    # Braching factor
    b = 3

    # Runs to average performance over
    runs = 30

    # Iterations to train per run
    iterations = 250000

    # Interval between iterations to evaluate V(s_0)
    eval_s0_interval = 200

    # Exploration probability
    epsilon = 0.1

    
    enviornment = Env(N_states, b)
    agent = Agent(N_states, b, enviornment)

    # Uniform updates
    iteration_vals, avg_s0_value_uniform = agent.trainUniform(runs, iterations, eval_s0_interval)

    # On-policy updates
    iteration_vals, avg_s0_value_policy = agent.trainOnPolicy(runs, iterations, eval_s0_interval, epsilon)
    
    plt.figure()
    plt.plot(iteration_vals, avg_s0_value_uniform, label = "Uniform", color = 'r')
    plt.plot(iteration_vals, avg_s0_value_policy, label = "Policy", color = 'b')
    plt.xlabel("Computation time, in expected updates")
    plt.ylabel("Value of starting state under greedy policy")
    plt.legend()
    plt.savefig('onPolicy_vs_uniform.png')
    plt.show()
    
