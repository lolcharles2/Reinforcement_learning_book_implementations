import random
import matplotlib.pyplot as plt
import numpy as np

class kArmedBandit:
    """ k armed bandit class

        Simulates a k armed bandit, where there is an reward associated with each of k different actions.
        When an action is presented to the bandit, the bandit will give a reward drawn from a normal distribution
        with a mean depending on the action and standard deviation 1. The value of the mean for each action may
        stay constant or change over time.
        For each action, the mean reward is initialized by drawing from a normal distribution with mean 0 and
        standard deviation 1.
        
    """
    def __init__(self, k, stationary = False):
        """ Initializes the bandit.

            @type k: int
                number of actions
            @type stationary: boolean
                If true, the list of rewards change when the updateReward method
                is called. The mean value of each reward is incremented
                independently by a value drawn from a gaussian with mean 0 and
                standard deviation 0.01. This causes the rewards to perform a random
                walk over time.
                If false, the rewards stay the same over time.
        """
        self.rewards = [np.random.normal(0, 1) for _ in range(k)]
        self.stationary = stationary
        self.k = k

    def getReward(self, action):
        """ Gets the reward associated with an action

            @type action: int
                an integer from 0 to k-1 denoting the action
            @rtype: float
                the reward associated with the action
        """
        if action >= self.k:
            raise ValueError("That is not a valid action!")
        reward = np.random.normal(self.rewards[action], 1)
        
        return reward

    def updateRewards(self):
        """ Updates all rewards if stationary is false """
        if not self.stationary:
            for i in range(self.k):
                self.rewards[i] += np.random.normal(0, 0.01)
    
class EpsilonGreedyAgent:
    """ The AI agent attempting to estimate the rewards of the k-armed bandit.
        
        The agent chooses actions and obtains rewards from the bandit. It then
        refines the estimates of the reward for each action based on the reward
        it has obtained. The goal of the agent is to accurately estimate the true
        rewards and use it to maximize the reward of its actions.

        In choosing the action, the agent may choose to be exploratory with probability
        epsilon or greedy. If exploratory is chosen, then the agent chooses an action
        at random. If greedy is chosen, then the agent chooses the action with the
        current maximum estimated reward.
    """
    def __init__(self, k, initial_estimates, epsilon, alpha = 0.1):
        """ Initializes the agent.

            @type k: int
                number of actions
            @type initial_estimates: list[float]
                list of length k with the initial estimates of the rewards
            @type epsilon: float
                a number between 0 and 1. Denotes the probability of an exploratory
                action.
            @type alpha: float
                the step size used to update the reward estimates
    
        """
        assert len(initial_estimates) == k, "Initial rewards array does not have length k!"
        super().__init__()
        self.estimates = initial_estimates
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha

    def chooseAction(self):
        """ Chooses an action

            With probabiltiy epsilon, the agent chooses a random action.
            With probability 1-epsilon, the agent chooses an action with the highest
            current estimated reward. Ties are broken at random

            @rtype: int
                An integer between 0 and k-1 that denotes the chosen action
        """
        x = random.uniform(0,1)
        if x > self.epsilon:
            best_estimate = max(self.estimates)
            candidate_actions = []
            for i in range(self.k):
                if self.estimates[i] == best_estimate:
                    candidate_actions.append(i)
            return random.choice(candidate_actions)
        return random.randrange(self.k)

    def updateEstimates(self, action, reward):
        """ Updates the agent's estimated reward for an action

            After the agent has recieved a reward for an action, it updates its
            estimate of the reward for that action by bringing it closer to the
            reward recieved. The estimate is updated by a number alpha times
            the difference between the recieved reward and the estimated reward.

            @type action: int
                an integer between 0 and k-1 representing the action
            @type reward: float
                the reward obtained from the bandit for the action

        """
        if action >= self.k:
            raise ValueError("That is not a valid action!")
        self.estimates[action] += self.alpha * (reward - self.estimates[action])
    


class GradientAgent:
    """ The AI agent attempting to choose the best action of the k-armed bandit.
        
        The agent chooses actions and obtains rewards from the bandit. The agent
        chooses the action with the highest "preference" and recieves an reward.
        Based on the reward, the agent then updates the preferences of each action
        using a stochastic gradient descent algorithm. If the reward was high, then
        the preference for that action will be increased, and vice versa.
        
    """
    def __init__(self, k, alpha):
        """ Initializes the agent.

            @type k: int
                number of actions
            @type alpha: float
                the step size used to update the probability estimates
    
        """
        super().__init__()
        self.H = np.zeros(k)
        self.baseline = [0]*k
        self.alpha = alpha
        self.frequencies = [1]*k
        self.k = k
        
    def chooseAction(self):
        """ Chooses an action with the highest preference.

            Ties are broken randomly.

            @rtype: int
                An integer between 0 and k-1 that denotes the chosen action
        """
        best_preference = max(self.H)
        candidate_actions = []
        for i in range(len(self.H)):
            if self.H[i] == best_preference:
                candidate_actions.append(i)
        return random.choice(candidate_actions)

    def updateEstimates(self, action, reward):
        """ Updates the agent's preference for an action

            After the agent has recieved a reward for an action, it updates its
            preference for that action based on the reward. If the reward is high,
            the preference for that action is increased and the preference for all
            other actions are decreased. If the reward is low, the opposite happens.

            @type action: int
                an integer between 0 and k-1 representing the action
            @type reward: float
                the reward obtained from the bandit for the action

        """
        if action >= self.k:
            raise ValueError("That is not a valid action!")

        # Update baseline reward value as average of all rewards recieved for that action
        self.baseline[action] += 1.0/self.frequencies[action] * (reward - self.baseline[action])
        self.frequencies[action] += 1

        # For numerical stability
        self.H = self.H - max(self.H)
        
        normalization = np.sum(np.exp(self.H))
        
        for i in range(len(self.H)):
            if i == action:
                self.H[i] += self.alpha * (reward - self.baseline[i]) * (1 - np.exp(self.H[i]) / normalization)
            else:
                self.H[i] -= self.alpha * (reward - self.baseline[i]) * np.exp(self.H[i]) / normalization
        


class UCBAgent:
    """ The AI agent attempting to estimate the rewards of the k-armed bandit.
        
        The agent chooses actions and obtains rewards from the bandit. It then
        refines the estimates of the reward for each action based on the reward
        it has obtained. The goal of the agent is to accurately estimate the true
        rewards and use it to maximize the reward of its actions.

        The agent chooses the action based on the current highest current estimated
        reward and the uncertainty in the estimate.
        
    """
    def __init__(self, k, c, alpha = 0.1):
        """ Initializes the agent.

            @type k: int
                number of actions
            @type initial_estimates: list[float]
                list of length k with the initial estimates of the rewards
            @type c: float
                a number controlling the degree of exploration
            @type alpha: float
                the step size used to update the reward estimates
    
        """
        super().__init__()
        self.estimates = [0]*k
        self.k = k
        self.alpha = alpha
        self.c = c
        self.frequencies = [0]*k
        self.iteration = 1
        
    def chooseAction(self):
        """ Chooses an action

           The agent chooses an action based on its current estimate of the
           reward for each action and the uncertainty in that estimate.

            @rtype: int
                An integer between 0 and k-1 that denotes the chosen action
        """

        Scores = []

        for i in range(self.k):
            if self.frequencies[i] == 0:
                Scores.append(float('inf'))
            else:
                Scores.append(self.estimates[i] + self.c * np.sqrt(np.log(self.iteration) / self.frequencies[i]))
        
                              
        best_estimate = max(Scores)
        candidate_actions = []
        for i in range(self.k):
            if Scores[i] == best_estimate:
                candidate_actions.append(i)
                              
        chosen_action = random.choice(candidate_actions)
        self.frequencies[chosen_action] += 1
        self.iteration += 1
        
        return chosen_action
        

    def updateEstimates(self, action, reward):
        """ Updates the agent's estimated reward for an action

            After the agent has recieved a reward for an action, it updates its
            estimate of the reward for that action by bringing it closer to the
            reward recieved. The estimate is updated by a number alpha times
            the difference between the recieved reward and the estimated reward.

            @type action: int
                an integer between 0 and k-1 representing the action
            @type reward: float
                the reward obtained from the bandit for the action

        """
        if action >= self.k:
            raise ValueError("That is not a valid action!")
        self.estimates[action] += self.alpha * (reward - self.estimates[action])
        

class CompareAgents:
    """ Compares the performance of various bandit agents

    """
    def __init__(self, AgentList, bandit):
        """ Initializes the comparison.

            As the agents are trained, the rewards they obtain are recorded
            during the second half of the training

            @type AgentList: list[agents]
                list of various agent classes
            @type bandit: kArmedBandit
                the bandit class
        """
        self.AgentList = AgentList
        self.bandit = bandit
        self.performance = [[] for _ in range(len(AgentList))]

    def trainAgents(self, iterations):
        """ Trains the agents

            Trains the agent by making it repeatly perform actions and obtaining
            rewards from the bandit. Returns the mean reward during the second half
            of the training for each agent.

            @type iterations: int
                the number of training iterations
            @rtype: list[float]
                the mean reward obtained during the second half
                of the training for each agent.
        """
 
        for i in range(iterations):
            for j in range(len(self.AgentList)):
                agent = self.AgentList[j]
                
                action = agent.chooseAction()
            
                reward = self.bandit.getReward(action)
           
                agent.updateEstimates(action, reward)
            
                if i > iterations / 2:
                    self.performance[j].append(reward)

            self.bandit.updateRewards()

        return [np.mean(rewards) for rewards in self.performance]

        
if __name__ == '__main__':

    k = 10
    iterations_per_run = 10000
    runs = 200

    param_vals = [2.0**i for i in range(-7, 3)]

    Labels = ['Epsilon Greedy', 'Greedy Optimal (epsilon = 0.1)', 'Gradient', 'UCB']
    
    Performances = [[],[],[],[]]

    # For each parameter in param_vals, train 4 different agents (labeled above)
    # for iterations_per_run iterations on the bandit and record their average
    # reward obtained in the last half the training. This procedure is repeated
    # for several runs and the results are averaged.
    
    for parameter in param_vals:

        # Array to store the average reward for each agent for each run.
        # avg_rewards[i] is a list of average rewards for each run of agent i
        avg_rewards = [[],[],[],[]]

        for j in range(runs):

            # For each run, re-initialize the agent and bandit
            GreedyAgent = EpsilonGreedyAgent(k, [0]*k, epsilon = parameter)
            GreedyOptimal = EpsilonGreedyAgent(k, [parameter]*k, epsilon = 0.1)
            GradAgent = GradientAgent(k, alpha = parameter)
            UCB = UCBAgent(k, c = parameter)
            
            Agents = [GreedyAgent, GreedyOptimal, GradAgent, UCB]

            bandit = kArmedBandit(k)

            # Train agents, record avg rewards
            comparison = CompareAgents(Agents, bandit)

            mean_rewards = comparison.trainAgents(iterations_per_run)

            for i in range(len(avg_rewards)):
                avg_rewards[i].append(mean_rewards[i])

        for t in range(len(Performances)):
            Performances[t].append(np.mean(avg_rewards[t]))

    # Plotting results

    plt.figure()
    
    for i in range(len(Performances)):
        plt.plot(param_vals, Performances[i], label = Labels[i])
    
    plt.legend()
    plt.xscale('log')
    plt.ylabel('Average reward')
    plt.xlabel(r'$\epsilon$' + ' (greedy), ' + r'$Q_0$' + ' (greedy optimal), ' + r'$\alpha$' + ' (gradient), ' + r'$c$' + ' (UCB)')
    plt.show()
