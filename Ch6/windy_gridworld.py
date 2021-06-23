import random
import numpy as np
import matplotlib.pyplot as plt

class WindyGridWorld:
    """ An implementation of Example 6.9 and 6.10 in
        "Reinforcement Learning, an Introduction by Sutton
        and Barto.

        Valid moves in the grid consists of 1 cell in any of the
        8 adjacent directions: 4 diagonals and 4 left/right up/down
        directions. Along each column of the grid, there is an upwards
        wind that slightly alters the outcome of any action.
        An option of stochastic winds can be turned on where the
        wind behaves stochasticaly by +/- 1 in magnitude.
    """
    def __init__(self, L, W, winds, goal, stochastic_winds):
        """ Initializes the enviornment.

            @type L: int
                The length of the grid.
            @type W: int
                The width of the grid.
            @type winds: list[int]
                A list denoting the wind speed in each column. Positive
                numbers denote winds blowing up the grid. Negative numbers
                denote winds blowing down. Length must be equal to L.
            @type goal: tuple[int]
                 A tuple (x,y) denoting the goal.
            @type stochastic_winds: boolean
                If true, then the wind magnitudes behave stochastically
                by +/- 1 in magnitude on each move.
        """
        if len(winds) != L:
            raise ValueError("The length of the winds list must be equal to the length of the grid!")
        
        self.L = L
        self.W = W
        self.winds = winds
        self.goal = goal
        self.stochastic_winds = stochastic_winds

        self.actions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (0, 0)]

    def getNextState(self, x, y, action):
        """ Given the current x,y position and an action, gets the next state.

            Actions are represented by an integer from 0-8, with 0 denoting
            moving right, and then numbered going counterclockwise. The action
            corresponding to 8 denotes no action apart from that caused by wind.

            @type x: int
                The current x position.
            @type y: int
                The current y position.
            @type action: int
                The action performed. Actions that move off the grid result in no
                movement.
            @rtype: tuple
                A triplet (new_x, new_y, finished) denoting the new x and y positions
                and if the goal has been reached.
        """

        if not (0 <= x <= L-1 and 0 <= y <= W-1):
            raise ValueError("That's not a valid position!")
        if not 0 <= action <= 8:
            raise ValueError("That's not a valid action!")

        if self.stochastic_winds:
            r = random.uniform(0, 1)
            if r < 1/3:
                wind = self.winds[x] - 1
            elif r < 2/3:
                wind = self.winds[x]
            else:
                wind = self.winds[x] + 1
        else:
            wind = self.winds[x]

        dx, dy = self.actions[action]
        
        new_x = int(max(0, min(L-1, x + dx)))
        new_y = int(max(0, min(W-1, y + dy + wind)))
        finished = True if (new_x, new_y) == self.goal else False

        return new_x, new_y, finished
        
class Agent:
    """ An agent for finding the optimal policy of windy grid world using
        on-policy TD control.
    
    """

    def __init__(self, L, W, start, goal, t):
        """ Initializes the agent.

            @type L: int
                The length of the grid.
            @type W: int
                The width of the grid.
            @type start: tuple[int]
                A tuple (x, y) denoting the starting position.
            @type goal: tuple[int]
                A tuple (x, y) denoting the goal.
            @type t: function(x, y, action) -> new_x, new_y, finished
                The state transition function giving the new x and y positions
                and if the episode has finished given the current x and y positions
                and the action taken.
        """
        self.L = L
        self.W = W
        self.start = start
        self.goal = goal
        self.t = t
        self.Q = np.full((L, W, 9), 0.0)


    def train(self, episodes, alpha, epsilon):
        """ Trains the agent using on-policy TD control.

            @type episodes: int
                The number of episodes to train.
            @type alpha: float
                The step size for updating the value function. Between 0 and 1.
            @type epsilon: float
                A number between 0 and 1 denoting the probability of taking
                an exploratory action.
            @rtype: list[int]
                A list T_steps denoting the time step
                at which each episode is completed.

        """
        time = 0

        T_steps = []
        
        for episode in range(episodes):

            
            if episode % 100 == 0:
                print("Episode {} completed!".format(episode))
                
            T_steps.append(time)
            
            x, y = self.start
            action = self.chooseAction(x, y, epsilon)
            finished = False
            
            while not finished:
  
                
                new_x, new_y, finished = self.t(x, y, action)
                new_action = self.chooseAction(new_x, new_y, epsilon)

                self.Q[x][y][action] += alpha*(-1 + self.Q[new_x][new_y][new_action] - self.Q[x][y][action])

                x, y, action = new_x, new_y, new_action

                time += 1

        return T_steps

    def buildPolicy(self):
        """ Builds the policy based on the current action value function
            and plots it along with a trajectory following the policy.

        """
        # Building policy
        pi = np.full((self.L, self.W), 0, dtype = int)

        for x in range(self.L):
            for y in range(self.W):
                pi[x][y] = np.argmax(self.Q[x][y])

        # Generating policy vector field
        x_vals, y_vals = np.meshgrid(np.linspace(0, self.L-1, self.L), np.linspace(0, self.W-1, self.W))
        
        v_x = np.zeros(np.shape(x_vals))
        v_y = np.zeros(np.shape(y_vals))

        for x in range(L):
            for y in range(W):
                dx, dy = grid.actions[pi[x][y]]
                v_x[y][x] = dx
                v_y[y][x] = dy

        # Building trajectory based on policy
        x, y = self.start
        
        trajectory_x, trajectory_y = [x],[y]
        finished = False
        while not finished:
            new_x, new_y, finished = self.t(x, y, pi[x][y])
            trajectory_x.append(new_x)
            trajectory_y.append(new_y)

            x, y = new_x, new_y


        # Plotting policy and trajectory
        plt.figure()
        plt.quiver(x_vals, y_vals, v_x, v_y)
        plt.plot(trajectory_x, trajectory_y, color = 'r', linewidth = 2)
        plt.plot([self.start[0]], [self.start[1]], marker = 's', markersize = 10, color = 'green')
        plt.plot([self.goal[0]], [self.goal[1]], marker = '*', markersize = 20, color = 'blue')
        plt.title("Final policy")
        plt.xlabel("x position")
        plt.ylabel("y position")

    def chooseAction(self, x, y, epsilon):
        """ Chooses an epsilon-greedy action based on the current estimate
            of the action value function.

            @ type x: int
                The current x position.
            @type y: int
                The current y position
            @type epsilon: float
                A number between 0 and 1 denoting the probability of an
                exploratory action.
            @rtype: int
                An integer between 0 and 8 denoting the action.
        """
        r = random.uniform(0, 1)
        if r < epsilon:
            return random.randint(0, 8)
        else:
            return np.argmax(self.Q[x][y])
        
if __name__ == "__main__":
    L = 10
    W = 7
    winds = [0,0,0,1,1,1,2,2,1,0]
    start = (0, 3)
    goal = (7, 3)
    stochastic_winds = False

    alpha = 0.2
    epsilon = 0.1
    episodes = 10000
    
    grid = WindyGridWorld(L, W, winds, goal, stochastic_winds)

    agent = Agent(L, W, start, goal, grid.getNextState)

    T_steps = agent.train(episodes, alpha, epsilon)

    print("Mean number of moves to get to goal in second half of training: {}".format(np.mean([T_steps[i]-T_steps[i-1] for i in range(len(T_steps)//2, len(T_steps))])))
    
    agent.buildPolicy()

    
    plt.figure()
    plt.plot(T_steps, range(len(T_steps)))
    plt.xlabel("Time steps")
    plt.ylabel("Episodes")
    plt.show()
