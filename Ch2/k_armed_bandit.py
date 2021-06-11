import random
import matplotlib.pyplot as plt
import numpy as np

class kArmedBandit:
    """ k armed bandit class

        Simulates a k armed bandit, where there is an reward associated with each of k different actions.
        When an action is presented to the bandit, the bandit will give a reward drawn from a normal distribution
        with a mean depending on the action and standard deviation 1. The value of the mean for each action may
        stay constant or change over time.
    """
    def __init__(self, k, initial_rewards, stationary):
        """ Initializes the bandit.

            @type k: int
                number of actions
            @type initial_rewards: list[float]
                a list of length k that contains the initial rewards for each action
            @type stationary: boolean
                If true, the list of rewards change after every iteration. Each time
                a reward is requested of an action, the mean value of each reward is incremented
                independently by a value drawn from a gaussian with mean 0 and
                standard deviation 0.01. This causes the rewards to perform a random
                walk over time.
                If false, the list of rewards stay as initial_rewards over time.
        """
        assert len(initial_rewards) == k, "Initial rewards array does not have length k!"
        self.rewards = initial_rewards
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
        if not self.stationary:
            for i in range(k):
                self.rewards[i] += np.random.normal(0, 0.01)
        return reward

    def getBestAction(self):
        """ Gets the best action at this moment

            @rtype: int
                the action (represented as an integer from 0 to k-1)
                corresponding to the maximum current reward
        """
        return self.rewards.index(max(self.rewards))

class kArmedBanditAgent:
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
    def __init__(self, k, initial_estimates, epsilon):
        """ Initializes the agent.

            @type k: int
                number of actions
            @type initial_estimates: list[float]
                list of length k with the initial estimates of the rewards
            @type epsilon: float
                a number between 0 and 1. Denotes the probability of an exploratory
                action.
    
        """
        assert len(initial_estimates) == k, "Initial rewards array does not have length k!"
        self.estimates = initial_estimates
        self.k = k
        self.epsilon = epsilon

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
            for i in range(k):
                if self.estimates[i] == best_estimate:
                    candidate_actions.append(i)
            return random.choice(candidate_actions)
        return random.randrange(k)

    def updateEstimates(self, action, reward, alpha):
        """ Updates the agent's estimated reward for an action

            After the agent has recieved a reward for an action, it updates its
            estimate of the reward for that action by bringing it closer to the
            reward recieved. The estimate is updated by a number alpha times
            the difference between the recieved reward and the estimated reward.

            @type action: int
                an integer between 0 and k-1 representing the action
            @type reward: float
                the reward obtained from the bandit for the action
            @type alpha: float
                the step size used to update the estimate of the reward
        """
        if action >= self.k:
            raise ValueError("That is not a valid action!")
        self.estimates[action] += alpha * (reward - self.estimates[action])
    

class TrainAgent:
    """ Class for training of the agent

        The agent repeatly chooses actions and obtains rewards from the bandit.
        In the process, it updates its estimates of the true rewards of each action.
        At each step, the agent's chosen action, the best possible action, and the reward
        obtained is recorded.

        The agent may update its estimate of the reward in two different ways. The first method is
        to set alpha = 1/n at step n. This technique is called "sample averaging", and amounts
        to estimating the reward of an action by the average value of all rewards recieved in the past
        for that action. The second method is to set alpha to be a constant value. This amounts to
        estimating the rewards using a weighted sum, placing more weight on the most recently recieved
        rewards. 
        
    """
    def __init__(self, k, initial_rewards, initial_estimates, stationary, sample_avg, iterations = 10000, alpha = 0.1, epsilon = 0.1):
        """ Initializes the trainer

            @type k: int
                the number of actions
            @type initial_rewards: list[float]
                the initial list of rewards for the bandit
            @type initial_estimates: list[float]
                the initial list of estimated rewards for the agent
            @type stationary: boolean
                if false, then the rewards for the bandit change over time.
                if true, then the rewards for the bandit stays as initial_rewards.
            @type sample_avg: boolean
                if true, then the agent will use the sample averaging method
                to estimate the rewards. This amounts to setting alpha = 1/n at
                step n.
                if false, the agent will use a constant alpha to update the
                reward estimates.
            @type iterations: int
                the number of actions the agent will perform and use to estimate
                the rewards.
            @type alpha: float
                the step size the agent will use to update the reward estimates if
                sample_avg is set to false.
            @type epsilon: float
                a number between 0 and 1 representing the probability of an exploratory
                action.
        """
        
        self.bandit = kArmedBandit(k, initial_rewards, stationary)
        self.agent = kArmedBanditAgent(k, initial_estimates, epsilon)
        self.alpha = alpha
        self.iterations = iterations
        self.sample_avg = sample_avg
        self.best_action = []
        self.chosen_action = []
        self.rewards_obtained = []

    def train(self):
        """ Trains the agent

            Trains the agent by making it repeatly perform actions and obtaining
            rewards from the bandit. It will then update the agent's estimate of the
            rewards based on the reward it recieved. At each step, the chosen action,
            best possible action, and reward obtained is recorded.
        """
        for i in range(self.iterations):
            action = self.agent.chooseAction()
            best_action = self.bandit.getBestAction()
            
            reward = self.bandit.getReward(action)
            if self.sample_avg:
                self.agent.updateEstimates(action, reward, 1.0/(1+i))
            else:
                self.agent.updateEstimates(action, reward, self.alpha)

            
            self.best_action.append(best_action)
            self.chosen_action.append(action)
            self.rewards_obtained.append(reward)

    def getBestActions(self):
        """ Gets the list of best actions at each step

            @rtype: list[int]
                a list represnting the best possible action
                at each step in the training
        """
        return np.array(self.best_action)

    def getChosenActions(self):
        """ Gets the list of chosen actions at each step

            @rtype: list[int]
                a list represnting the chosen action
                at each step in the training
        """
        return np.array(self.chosen_action)

    def getRewardsObtained(self):
        """ Gets the list of rewards obtained at each step

            @rtype: list[float]
                a list represnting the rewards obtained
                at each step in the training
        """
        return np.array(self.rewards_obtained)

            
        
def collectStatistics(k, runs, iterations_per_run, stationary, sample_avg):
    """ Function to train agents multiple times in order to average their performance.

        @type k: int
            number of actions
        @type runs: int
            number of agents to train in order to average their performance
        @type iterations_per_run: int
            number of steps each agent is to be trained
        @type stationary: boolean
            if true, then the bandit's rewards will stay the same over time
            if false, then the bandit's rewards will change over time
        @type sample_avg: boolean
            if true, then the agent will use the sample averaging method
            to estimate the rewards. This amounts to setting alpha = 1/n at
            step n.
            if false, the agent will use a constant alpha to update the
            reward estimates.

        @rtype: tuple[list[float]]
            a tuple (best_action_prob, avg_rewards).
            best_action_prob is a list of the average probability that an agent
            will choose the best possible action at each step in the training.
            avg_rewards is a list of the average reward an agent obtains at each
            step in the training.
            
    """

    best_action_prob = np.zeros(iterations_per_run)
    avg_rewards = np.zeros(iterations_per_run)
    
    for _ in range(runs):

        initial_rewards = [np.random.normal(0,1)]*k
        initial_estimates = [0]*k
        training = TrainAgent(k, initial_rewards, initial_estimates, stationary = stationary, sample_avg = sample_avg, iterations = iterations_per_run)
        training.train()
        
        best_actions = training.getBestActions()
        chosen_actions = training.getChosenActions()
        rewards = training.getRewardsObtained()

        best_action_prob += (best_actions == chosen_actions)
        avg_rewards += rewards

    best_action_prob /= runs
    avg_rewards /= runs

    return best_action_prob, avg_rewards
    
if __name__ == '__main__':

    k = 10
    runs = 2000
    iterations_per_run = 5000

    # Goal here is to compare the peformance of sample averaging vs constant alpha
    # in the case of a non-stationary reward system. If the reward system is stationary, then
    # sample averaging should perform very well. Due to the law of large numbers, as the number of
    # iterations approach infinity, the average reward for each action converges to the true mean.
    # In an non-stationary reward system however, we expect the constant alpha update method to
    # perform better due to its ability to adapt to changes over time by virtue of weighing more
    # recent rewards more heavily.

    # Sample averaging
    best_action_prob_sample_avg, avg_rewards_sample_avg = collectStatistics(k, runs, iterations_per_run, stationary = False, sample_avg = True)

    # Constant alpha
    best_action_prob_alpha, avg_rewards_alpha = collectStatistics(k, runs, iterations_per_run, stationary = False, sample_avg = False)


    # Plotting results
    plt.figure()
    plt.plot(best_action_prob_sample_avg, label = 'Sample average')
    plt.plot(best_action_prob_alpha, label = 'Constant alpha')
    plt.xlabel('Iteration')
    plt.ylabel('Probability of choosing best action')
    plt.ylim(0, 1)
    plt.legend()

    plt.figure()
    plt.plot(avg_rewards_sample_avg, label = 'Sample average')
    plt.plot(avg_rewards_alpha, label = 'Constant alpha')
    plt.xlabel('Iteration')
    plt.ylabel('Average reward')
    plt.legend()

    plt.show()
