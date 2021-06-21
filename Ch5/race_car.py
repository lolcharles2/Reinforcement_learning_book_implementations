import random
import numpy as np
import matplotlib.pyplot as plt

class Point:
    """ A point object with an x and y coordinate """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
class RaceCar:
    """ The race car enviornment. """
    
    def __init__(self, L1, L2, W1, W2, no_action_prob):
        """ Initializes the enviornment.
            
            The race track is as follows:


            -  ###########################################&
            |  ###########################################&
            |  ###########################################& 
            W2 ###########################################&
            |  ###########################################&  
            |  ###########################################&  
            -  ###########################################& 
            |  ######################                     |
            |  ######################                     |
            |  ######################                     |
            |  ######################                     |  
            W1 ######################                     |
            |  ######################                     |
            |  ######################                     |  
            |  ######################                     |
            |  ######################                     |
            |  ######################                     |  
            -  $$$$$$$$$$$$$$$$$$$$$$                     |  
               |--------L1----------|---------L2----------|

            The $ symbols is the starting line, and the & symbols is the finish line.

            @type L1: int
            @type L2: int
            @type W1: int
            @type W2: int
            @type no_action_prob: float
                A number between 0 and 1 representing the probability of an action
                failing to register.
        """
        
        self.L1 = L1
        self.L2 = L2
        self.W1 = W1
        self.W2 = W2
        self.no_action_prob = no_action_prob
        
        self.Vx = 0
        self.Vy = 0

        # Borders of the race track
        self.borders = []
        self.borders.append((Point(-1, -1), Point(L1+1, -1)))
        self.borders.append((Point(L1+1, -1), Point(L1+1, W1-1)))
        self.borders.append((Point(L1+1, W1-1), Point(L1+L2, W1-1)))
        self.borders.append((Point(L1+L2, W1+W2+1), Point(-1, W1+W2+1)))
        self.borders.append((Point(-1, W1+W2+1), Point(-1, -1)))

        # End points of the finish line
        self.finish_p = Point(L1+L2, W1-1)
        self.finish_q = Point(L1+L2, W1+W2+1)
        
    def getNextState(self, x, y, action):
        """ The state transition function

            The action is an integer from 0 to 8, converted into one of
            nine different actions as described in the convertActionToVelocity
            function. The velocity is updated first, then the new position is
            calculated. If the trajectory intersects a track boundary, then
            the car is sent back to a random position on the starting line with
            the velocities set to 0. If the trajectory crosses the finish line, then
            the episode ends.

            @type x: int
                x coordinate of current position.
            @type y: int
                y coordinate of current position.
            @type action: int
                An integer from 0 to 8 representing the action.
            @rtype: tuple(int, int, boolean)
                A triplet representing the new x, y positions, and if the
                episode has ended (car crossed finish line).

        """
        
        dV_x, dV_y = self.convertActionToVelocity(action)

        # Updating velocities with probability 1-no_action_prob
        # if the update does not result in both velocities equal to 0
        r = random.uniform(0, 1)
        if r > self.no_action_prob:
            if not (self.Vx + dV_x == 0 and self.Vy + dV_y == 0):
                self.Vx = max(0, min(5, self.Vx + dV_x))
                self.Vy = max(0, min(5, self.Vy + dV_y))
        
        new_x = int(x + self.Vx)
        new_y = int(y + self.Vy)

        # Check if car has crossed a boundary
        for p, q in self.borders:
            if self.doIntersect(Point(x,y), Point(new_x, new_y), p, q):
                self.Vx = 0
                self.Vy = 0
                return random.randint(0, L1), 0, False

        # Check if car has crossed finish line
        if self.doIntersect(Point(x,y), Point(new_x, new_y), self.finish_p, self.finish_q):
            self.Vx = 0
            self.Vy = 0
            return random.randint(0, L1), 0, True

        return new_x, new_y, False

        
        
        

    def convertActionToVelocity(self, action):
        """ Converts the action (an integer from 0 to 8)
            to the changes in velocity in x and y
            according to the following table

                          dV_x
                    | -1    0    1
                  ----------------
                 -1 |  0    1    2
                    |
              dVy 0 |  3    4    5
                    |  
                  1 |  6    7    8

            @type action: int
                A number between 0 and 8.
            @type: tuple[int]
                The changes in x velocity and y velocity
                represented as (dV_x, dV_y)
            
        """
        if not 0 <= action <= 8:
            raise ValueError("That's not a valid action!")
        
        dV_x = action % 3 - 1
        dV_y = action // 3 - 1
        
        return dV_x, dV_y

    def doIntersect(self, p1, q1, p2, q2):
        """ Function that returns True if the line segment 'p1q1'
            and 'p2q2' intersect.

            This piece of code was taken from:
            https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
            
            @type p1: Point
                Point 1 of line segment 1.
            @type q1: Point
                Point 2 of line segment 1.
            @type p2: Point
                Point 1 of line segment 2.
            @type q2: Point
                Point 2 of line segment 2.
            @rtype: boolean
        """
                    
        # Find the 4 orientations required for 
        # the general and special cases
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)
      
        # General case
        if ((o1 != o2) and (o3 != o4)):
            return True
      
        # Special Cases
      
        # p1 , q1 and p2 are colinear and p2 lies on segment p1q1
        if ((o1 == 0) and self.onSegment(p1, p2, q1)):
            return True
      
        # p1 , q1 and q2 are colinear and q2 lies on segment p1q1
        if ((o2 == 0) and self.onSegment(p1, q2, q1)):
            return True
      
        # p2 , q2 and p1 are colinear and p1 lies on segment p2q2
        if ((o3 == 0) and self.onSegment(p2, p1, q2)):
            return True
      
        # p2 , q2 and q1 are colinear and q1 lies on segment p2q2
        if ((o4 == 0) and self.onSegment(p2, q1, q2)):
            return True
  
        # If none of the cases
        return False
    
    def orientation(self, p, q, r):
        """ Finds the orientation of an ordered triplet (p,q,r)
            function returns the following values:
            0 : Colinear points
            1 : Clockwise points
            2 : Counterclockwise

            This piece of code was taken from:
            https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
            
            @type p: Point
            @type q: Point
            @type r: Point
            @rtype: int
          
        """
        
        val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
        if (val > 0):
              
            # Clockwise orientation
            return 1
        elif (val < 0):
              
            # Counterclockwise orientation
            return 2
        else:
              
            # Colinear orientation
            return 0
    
    def onSegment(self, p, q, r):
        """ Given three colinear points p, q, r, the function checks if
            point q lies on line segment 'pr'

            This piece of code was taken from:
            https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
            
            @type p: Point
            @type q: Point
            @type r: Point
            @rtype: boolean
            
        """
        if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
               (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
            return True
        return False

class CarAgent:
    """ Agent for learning to drive the car on the race track. """
    
    def __init__(self, L1, L2, W1, W2, t, gamma):
        """ Initializes the agent.

            The race track is as follows:


            -  ###########################################&
            |  ###########################################&
            |  ###########################################& 
            W2 ###########################################&
            |  ###########################################&  
            |  ###########################################&  
            -  ###########################################& 
            |  ######################                     |
            |  ######################                     |
            |  ######################                     |
            |  ######################                     |  
            W1 ######################                     |
            |  ######################                     |
            |  ######################                     |  
            |  ######################                     |
            |  ######################                     |
            |  ######################                     |  
            -  $$$$$$$$$$$$$$$$$$$$$$                     |  
               |--------L1----------|---------L2----------|

            The $ symbols is the starting line, and the & symbols is the finish line.

            @type L1: int
            @type L2: int
            @type W1: int
            @type W2: int
            @type t: function(x, y, action) -> new_x, new_y, finished
                A state transition function taking the current x,y position
                and action and giving the new x, y positions and a boolean
                finished representing if the episode has ended.
            @type gamma: float
                The discount factor.
            
        """
        self.t = t
        self.L1 = L1

        self.gamma = gamma
        self.Q = np.full((W1+W2+1, L1+L2+1, 9), -1e6)
        self.C = np.zeros((W1+W2+1, L1+L2+1, 9))
        self.pi = np.full((W1+W2+1, L1+L2+1), 4, dtype = int)

    def train(self, episodes, epsilon):
        """ Trains the agent using off-policy MC control.

            Each episode is generated using the current estimated
            best policy convered to an epsilon-soft policy, where with
            probability epsilon it will choose a random action instead.
            This episode is then used to train the agent.

            @type episodes: int
                The number of episodes to train.
            @type epsilon: float
                A number between 0 and 1 for the epsilon-soft policy.
        
        """

        for episode in range(episodes):

            if episode % 10000 == 0:
                np.save("Q.npy", self.Q)
                np.save("C.npy", self.C)
                np.save("pi.npy", self.pi)
                print("Episode {} completed!".format(episode))
                
            # Generate an episode by turning current policy into epsilon-soft
            pi_copy = np.copy(self.pi)
            
            history = self.generateEpisode(pi_copy, epsilon)
            
            G = 0.0
            W = 1.0
            
            for i in range(len(history)-1, -1, -1):
                x, y, action, reward = history[i]
                G = self.gamma*G + reward
                self.C[y][x][action] += W
                self.Q[y][x][action] += W/self.C[y][x][action] * (G - self.Q[y][x][action])
                self.pi[y][x] = np.argmax(self.Q[y][x])

                if action != self.pi[y][x]:
                    break

                denominator = 1.0-epsilon+epsilon/9.0 if action == pi_copy[y][x] else epsilon/9.0
                W *= 1.0/denominator
        
    def generateEpisode(self, pi, epsilon):
        """ Generates an episode given a policy

            @type pi: array[int]
                2D array of size (W1+W2+1, L1+L2+1) where
                pi[i][j] is the determinstic action at position
                x = j, y = i
            @type epsilon: float
                A number between 0 and 1 representing the probability
                of choosing a random action
            @rtype: list[tuple]
                A list representing the history of the episode. Each
                element is 
        """
        # Random initial starting position on the starting line
        x, y = random.randint(0, self.L1), 0
        finished = False

        history = []
        
        while not finished:

            r = random.uniform(0, 1)
            if r < epsilon: action = random.randint(0,8)
            else: action = int(pi[y][x])
            
            new_x, new_y, finished = self.t(x, y, action)

            history.append((x, y, action, -1))

            x, y = new_x, new_y
            
        return history

    def getPolicy(self):
        """ Gets the current policy

            @rtype: array[int]
                A 2D array pi representing the policy.
                pi[y][x] is the action to be taken at position (x,y)
        """
        return self.pi
        
        
if __name__ == "__main__":
    L1 = 10
    L2 = 10
    W1 = 20
    W2 = 5
    no_action_prob = 0.1
    gamma = 1.0


    episodes = 10000000
    epsilon = 0.1
    
    env = RaceCar(L1, L2, W1, W2, no_action_prob)
    agent = CarAgent(L1, L2, W1, W2, env.getNextState, gamma)

    # Load pre-trained files
    agent.pi = np.load("pi.npy")
    agent.Q = np.load("Q.npy")
    agent.C = np.load("C.npy")

    # Train the agent
    agent.train(episodes, epsilon)


    pi = agent.getPolicy()

    pi_x = np.zeros((W1+W2+1, L1+L2+1))
    pi_y = np.zeros((W1+W2+1, L1+L2+1))
    
    # Change all the positions outside the track to some negative number for contrast
    for y in range(len(pi)):
        for x in range(len(pi[0])):
            if not ((0<= x <= L1 and 0<= y <= W1+W2) or (L1 <= x <= L1+L2 and W1 <= y <= W1+W2)):
                pi_x[y][x] = -10
                pi_y[y][x] = -10
            else:
                dV_x, dV_y = env.convertActionToVelocity(pi[y][x])
                pi_x[y][x] = dV_x
                pi_y[y][x] = dV_y
                
                
    # Plotting results
    plt.figure()
    plt.imshow(pi_x, origin = 'lower', interpolation = 'none')
    plt.title('X velocity final policy')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.figure()
    plt.imshow(pi_y, origin = 'lower', interpolation = 'none')
    plt.title('Y velocity final policy')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.show()
