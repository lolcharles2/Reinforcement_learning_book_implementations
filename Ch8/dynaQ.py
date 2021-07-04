import random
import numpy as np
import matplotlib.pyplot as plt
import copy

class GridWorld:
    """ 2D grid world enviornment, example 8.3 in the book.
    """
    def __init__(self, L, W, obstacles, goal):
        """ Initializes the enviornment.

            @type L: int
                The length of the grid.
            @type W: int
                The width of the grid.
            @type obstacles: set[tuple]
                A set of tuples (x,y) denoting obstacles at the
                location (x,y) on the grid.
            @type goal: tuple[int]
                A tuple (x, y) denoting the goal position on the grid.
        """

        
        self.L = L
        self.W = W
        self.obstacles = obstacles
        self.goal_x, self.goal_y = goal

        self.action_to_coords = { 0 : (1, 0),
                                 1 : (0, 1),
                                 2 : (-1, 0),
                                 3 : (0, -1)
            }
       

    def getNextState(self, x, y, action):
        """ Given the current position and an action, gets the next state.

            Actions are represented by an integer 0-3, with 0,1,2,3 denoting
            moving right, up, left, and down respectively.

            @type x: int
                The current x position.
            @type y: int
                The current y position.
            @type action: int
                The action performed. 
            @rtype: tuple[int]
                A tuple (new_x, new_y, reward).
        """

        if not 0 <= x <= self.L - 1 or not 0 <= y <= self.W - 1:
            raise ValueError("That's not a valid position!")
        if not 0 <= action <= 3:
            raise ValueError("That's not a valid action!")

        dx, dy = self.action_to_coords[action]

        new_x = max(0, min(self.L - 1, x + dx))
        new_y = max(0, min(self.W - 1, y + dy))

        if (new_x, new_y) in self.obstacles:
            new_x, new_y = x, y
        
        reward = 1 if new_x == self.goal_x and new_y == self.goal_y else 0
        
        return new_x, new_y, reward

        
class Agent:
    """ Dyna-Q agent that attempts to learn the optimal policy.
    """

    def __init__(self, L, W):
        """ Initializes the agent.

            @type L: int
                The length of the grid.
            @type W: int
                The width of the grid.
        """
        self.L = L
        self.W = W
        self.Q = np.zeros((L, W, 4))
        self.model = {}

        # To keep track of the last time an action was performed.
        self.tau = np.zeros((L, W, 4))
        

    def updateQ(self, x, y, action, reward, new_x, new_y, alpha, gamma, t):
        """ Performs the direct RL update in the Dyna-Q algorithm and updates
            the learned model based on a real interaction with the enviornment.

            @type x: int
                Current x position.
            @type y: int
                Current y position.
            @type action: int
                Action performed.
            @type reward: float
                The reward for performing the action.
            @type new_x: int
                New x position.
            @type new_y: int
                New y position.
            @type alpha: float
                The step size for the update.
            @type gamma: float
                The discount factor.
            @type t: int
                The time at which the action was taken.

        """

        # Performing direct RL update.
        self.Q[x,y,action] += alpha * (reward + gamma * max(self.Q[new_x][new_y]) - self.Q[x,y,action])
        self.tau[x,y,action] = t

        # Updating learned model
        self.model[(x, y, action)] = (new_x, new_y, reward)
        for action in range(4):
            if (x, y, action) not in self.model:
                self.model[(x, y, action)] = (x, y, 0)

    def plan(self, n, alpha, gamma, kappa, t):
        """ Performs n iterations of planning updates

            @type n: int
                The number of planning updates.
            @type alpha: float
                The step size in the update.
            @type gamma: float
                The discount factor.
            @type kappa: float
                The parameter in the Dyna-Q+ agent that rewards
                exploring transitions that have not been explored in a while.
            @type t: int
                The current time step.
        """
        
        if self.model:
            keys = list(self.model.keys())
            for iteration in range(n):
                x, y, action = random.choice(keys)
                new_x, new_y, reward = self.model[(x, y, action)]
                self.Q[x,y,action] += alpha * (reward + kappa * np.sqrt(t - self.tau[x,y,action]) + gamma * max(self.Q[new_x][new_y]) - self.Q[x,y,action])
                
            

    def chooseAction(self, x, y, epsilon, kappa, t):
        """ Chooses an action based on an epsilon greedy policy.

            @type x: int
                Current x position.
            @type y: int
                Current y position.
            @type epsilon: float
                Probability of exploratory action.
            @type kappa: float
                The parameter in the Dyna-Q+ agent that rewards
                exploring transitions that have not been explored in a while.
            @type t: int
                The current time step.  
            @rtype: int
                An integer 0-3 representing the action chosen.
        """
        r = random.uniform(0, 1)
        if r < epsilon:
            return random.randint(0, 3)
        else:
            m = max(self.Q[x][y] + kappa * np.sqrt(t - self.tau[x][y]))
            return random.choice([i for i in range(4) if self.Q[x][y][i] + kappa * np.sqrt(t - self.tau[x][y][i]) == m])
   

        
if __name__ == "__main__":

    # 2D grid size
    L = 9
    W = 6

    start = (3, 0)
    goal = (L-1, W-1)
    obstacles = set([(x, 2) for x in range(1, L)])

    # Parameters
    gamma = 0.95
    alpha = 0.1
    epsilon = 0.1
    
    # Number of planning updates
    n = 100

    # Number of runs to average performance over
    runs = 30

    # Number of time steps per run
    time_steps = 6000


    def evaluatePerformance(epsilon, kappa_choose, kappa_plan, runs):
        """ Evaluates the performance of the agent by averaging
            the cumulative rewards over time over a number of runs.

            @type epsilon: float
                Probability of choosing an exploratory action.
            @type kappa_choose: float
                Exploratory reward for choosing actions.
            @type kappa_plan: float
                Exploratory reward for planning actions.
            @type runs: int
                The number of runs to average performance over.
            @rtype: array[float]
                The average cumulative rewards over time.
        """
        
        avg_rewards = np.zeros(time_steps)

        for run in range(runs):
            
            
            env = GridWorld(L, W, copy.deepcopy(obstacles), goal)
            agent = Agent(L, W)

            rewards = np.zeros(time_steps)
            cumu_rewards = 0
            x, y = start
            
            for t in range(time_steps):

                # At t = 3000, remove the right most obstacle.
                if t == 3000:
                    env.obstacles.remove((L-1, 2))
                
                action = agent.chooseAction(x, y, epsilon, kappa_choose, t)
                new_x, new_y, reward = env.getNextState(x, y, action)
                
                cumu_rewards += reward
                rewards[t] = cumu_rewards

                # Direct RL update and model update
                agent.updateQ(x, y, action, reward, new_x, new_y, alpha, gamma, t)

                # Planning
                agent.plan(n, alpha, gamma, kappa_plan, t)
                
                # Start at the beginning if goal is reached. Else continue to next state.
                if reward == 1:
                    x, y = start
                else:
                    x, y = new_x, new_y

            print("Run number {} completed!".format(run+1))
            avg_rewards += rewards / runs

        return avg_rewards
    
    # Classic dynaQ algorithm with epsilon greedy actions and  no exploration reward.
    print("Evaluating classic DynaQ...")
    dynaQ_perf = evaluatePerformance(epsilon = epsilon, kappa_choose = 0.0, kappa_plan = 0.0, runs = runs)

    # DynaQ+ algorithm with epsilon greedy actions and exploration reward on the planning stage.
    print("Evaluating DynaQ+...")
    dynaQPlus_perf = evaluatePerformance(epsilon = epsilon, kappa_choose = 0.0, kappa_plan = 0.001, runs = runs)

    # DynaQ+ algorithm with no greedy actions but actions are chosen with additioal exploration reward.
    # No exploration reward on the planning stage.
    print("Evaluating DynaQ+ Choose...")
    dynaQPlusChoose_perf = evaluatePerformance(epsilon = 0.0, kappa_choose = 0.008, kappa_plan = 0.0, runs = runs)
    
    plt.figure()
    plt.plot(dynaQ_perf, label = "DynaQ")
    plt.plot(dynaQPlus_perf, label = "DynaQ+")
    plt.plot(dynaQPlusChoose_perf, label = "DynaQ+ Choose")
    plt.axvline(x = 3000, color = 'k')
    plt.xlabel("Time step")
    plt.ylabel("Average cumulative reward")
    plt.legend()
    plt.savefig('dynaQ_compare.png')
    plt.show()
