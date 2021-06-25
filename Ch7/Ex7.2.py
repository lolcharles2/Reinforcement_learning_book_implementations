import random
import numpy as np
import matplotlib.pyplot as plt
import copy

class RandomWalk:
    """ 1D random walk enviornment, example 7.1 in the book.
    """
    def __init__(self, L):
        """ Initializes the enviornment.

            @type L: int
                The length of the 1D chain.
        """

        
        self.L = L
       

    def getNextState(self, x, action):
        """ Given the current position and an action, gets the next state.

            Actions are represented by an integer 0 and 1, with 0 denoting
            moving left and 1 denoting moving right.

            @type x: int
                The current position.
            @type action: int
                The action performed. 
            @rtype: tuple[int]
                A tuple (new_x, reward).
        """

        if not 0 < x < self.L - 1:
            raise ValueError("That's not a valid position!")
        if not 0 <= action <= 1:
            raise ValueError("That's not a valid action!")

        new_x = x - 1 if action == 0 else x + 1
        reward = 1 if new_x == self.L - 1 else 0
        
        return new_x, reward

        
class Agent:
    """ Agent object that evaluates the value function for a given
        policy using TD(n).
    """

    def __init__(self, L, t):
        """ Initializes the agent.

            @type L: int
                The length of the 1D chain.
            @type t: function(x, action) -> new_x
                The state transition function giving the new position
                given the current position and the action taken.
        """
        self.L = L
        self.t = t
        self.V = np.full(L, 0.5)
        self.V[0] = self.V[L-1] = 0


    def evaluatePolicy(self, episodes, n, alpha, gamma, pi, start, sum_TD_err):
        """ Trains the agent using on-policy TD control via TD(n).

            @type episodes: int
                The number of episodes to train.
            @type n: int
                The n parameter in TD(n).
            @type alpha: float
                The step size in TD.
            @type gamma: float
                The discount factor.
            @type pi: function(x) -> action
                The policy whose value function is to be evaluated.
            @type start: int
                The starting position in the chain.
            @type sum_TD_err: boolean
                If true, then the value function is updated using the sum
                of TD errors calculated from the value function from the previous
                episode.
            @rtype: list[float]
                A list denoting the mean squared error of the value estimates
                from the true values for each episode.

        """
        
        mean_errors = []
        true_values = [1/(self.L-1) * i for i in range(self.L)]
        true_values[-1] = 0
        
        
        for episode in range(episodes):

            mean_errors.append(np.sqrt(np.mean((self.V-np.array(true_values))**2)))

            V_copy = copy.deepcopy(self.V)
            
            position = start
            rewards = []
            states = [start]
                
            time = 0
            tau = 0
            T = float('inf')

            while tau <= T - 1:
                if time < T:
                    
                    action = pi(position)
                    new_position, reward = self.t(position, action)

                    rewards.append(reward)
                    states.append(new_position)
                    
                    if new_position == 0 or new_position == self.L-1:
                        T = time + 1

                    position = new_position
                        
                tau = time - n + 1
                
                if tau >= 0:
                    
                    if sum_TD_err:
                        err = 0
                        for i in range(tau, min(tau+n-1, T-1) + 1):
                            err += gamma**(i-tau) * (rewards[i] + gamma*V_copy[states[i+1]] - V_copy[states[i]])
                        self.V[states[tau]] += alpha * err
                    else:
                        G = 0
                        for i in range(tau+1, min(tau+n, T) + 1):
                            G += gamma**(i-tau-1) * rewards[i-1]
                        if tau + n < T: G += gamma**n * self.V[states[tau + n]]
                        
                        self.V[states[tau]] += alpha * (G - self.V[states[tau]])
                    
                    
                time += 1
                

        return np.array(mean_errors)   

   

        
if __name__ == "__main__":

    # 1D random walk chain length
    L = 19

    # Start in the middle of the chain
    start = L//2


    alpha = 0.05
    gamma = 1.0
    episodes = 200
    n = 4

    # Runs to average results over
    runs = 100
    
    chain = RandomWalk(L)


    # Defining random policy
    def pi(x):
        r = random.uniform(0, 1)
        return 0 if r < 0.5 else 1
    
    TDn = np.zeros(episodes)
    non_TDn = np.zeros(episodes)

    for run in range(runs):
        agent = Agent(L, chain.getNextState)
        TDn += agent.evaluatePolicy(episodes, n, alpha, gamma, pi, start, sum_TD_err = False)/runs
        
    
    for run in range(runs):
        agent = Agent(L, chain.getNextState)
        non_TDn += agent.evaluatePolicy(episodes, n, alpha, gamma, pi, start, sum_TD_err = True)/runs
    

    
    plt.figure()
    plt.plot(TDn, label = 'TD(n)')
    plt.plot(non_TDn, label = 'Sum TD Errs')
    plt.xlabel("Episodes")
    plt.ylabel("Mean squared error")
    plt.legend()
    plt.show()
