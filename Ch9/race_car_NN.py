import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

# GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Point:
    """ A point object with an x and y coordinate """

    def __init__(self, x, y):
        self.x = x
        self.y = y


class RaceCar:
    """ The race car environment. """

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
        self.borders.append((Point(-1, -1), Point(L1 + 1, -1)))
        self.borders.append((Point(L1 + 1, -1), Point(L1 + 1, W1 - 1)))
        self.borders.append((Point(L1 + 1, W1 - 1), Point(L1 + L2, W1 - 1)))
        self.borders.append((Point(L1 + L2, W1 + W2 + 1), Point(-1, W1 + W2 + 1)))
        self.borders.append((Point(-1, W1 + W2 + 1), Point(-1, -1)))

        # End points of the finish line
        self.finish_p = Point(L1 + L2, W1 - 1)
        self.finish_q = Point(L1 + L2, W1 + W2 + 1)

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
            if self.doIntersect(Point(x, y), Point(new_x, new_y), p, q):
                self.Vx = 0
                self.Vy = 0
                return random.randint(0, L1), 0, False

        # Check if car has crossed finish line
        if self.doIntersect(Point(x, y), Point(new_x, new_y), self.finish_p, self.finish_q):
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
        if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
                (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
            return True
        return False


class Agent:
    """ Agent object that uses semi-gradient SARSA(n) to find optimal
        policy.
    """

    def __init__(self, Car, NN, optimizer, criterion, state_to_basis):
        """ Initializes the agent.

            @type Car: RaceCar
                The RaceCar environment.
            @type NN: NeuralNet
                Neural network for computing the state-action values
            @type optimizer: optimizer
                Optimizer from the torch.optim module.
            @type criterion: criterion
                Criterion from the torch.nn module.
            @type state_to_basis: dict[tuple] -> torch.tensor
                A dictionary that converts a position tuple (x,y) to
                a corresponding torch tensor as input to the neural
                network.
        """
        self.Car = Car
        self.NN = NN
        self.op = optimizer
        self.criterion = criterion
        self.state_to_basis = state_to_basis

    def chooseAction(self, x, y, epsilon):
        """ Chooses action according to an epsilon greedy policy.
            @type x: int
                The x position.
            @type y: int
                The y position.
            @type epsilon: float
                Exploration probability.
            @rtype: int
                An integer from 0 to 8 representing the action.
        """

        if random.uniform(0, 1) < epsilon:
            return random.randint(0, 8)
        else:
            with torch.no_grad():
                _, action = torch.max(self.NN(self.state_to_basis[(x,y)]), 1)
                return int(action.item())

    def oneStep(self, state, action, target):
        """ Performs one training step on observed transition.
            @type state: tuple[int]
                A tuple (x,y) denoting the position.
            @type action: int
                An integer from 0-8 representing the action.
            @type target: float
                The target action value.
        """
        pred = self.NN(self.state_to_basis[state])

        with torch.no_grad():
            pred_target = torch.clone(pred)
            pred_target[0][action] = target

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
        tot_steps = 0

        for episode in range(episodes):

            if (episode + 1) % 1000 == 0:
                print(f'Episode {episode + 1}/{episodes} completed!')
                self.plotPolicy()
                torch.save(self.NN.state_dict(), 'race_car_NN_model')
                print(f'Average steps per episode: {tot_steps/1000}')
                tot_steps = 0

            x, y = random.randint(0, self.Car.L1), 0
            action = self.chooseAction(x, y, epsilon)
            states = [(x, y)]
            actions = [action]

            time = 0
            tau = 0
            T = float('inf')

            while tau < T - 1:
                if time < T:
                    x, y, done = self.Car.getNextState(x, y, action)

                    states.append((x, y))

                    if done:
                        T = time + 1
                        tot_steps += T
                    else:
                        action = self.chooseAction(x, y, epsilon)
                        actions.append(action)

                tau = time - n + 1

                if tau >= 0:

                    G = tau - min(tau + n, T)
                    if tau + n < T:
                        with torch.no_grad():
                            G += self.NN(state_to_basis[states[tau + n]])[0][actions[tau + n]]

                    self.oneStep(states[tau], actions[tau], G)

                time += 1

    def plotPolicy(self):
        """
        Plots and saves the current policy in x and y.
        """
        pi_x = np.zeros((self.Car.W1 + self.Car.W2 + 1, self.Car.L1 + self.Car.L2 + 1))
        pi_y = np.zeros((self.Car.W1 + self.Car.W2 + 1, self.Car.L1 + self.Car.L2 + 1))

        for y in range(len(pi_x)):
            for x in range(len(pi_x[0])):
                # Change all the positions outside the track to -10 for contrast
                if not ((0 <= x <= L1 and 0 <= y <= W1 + W2) or (L1 <= x <= L1 + L2 and W1 <= y <= W1 + W2)):
                    pi_x[y][x] = -10
                    pi_y[y][x] = -10
                else:
                    with torch.no_grad():
                        _, action = torch.max(self.NN(self.state_to_basis[(x,y)]), 1)
                        action = int(action)
                    dV_x, dV_y = self.Car.convertActionToVelocity(action)
                    pi_x[y][x] = dV_x
                    pi_y[y][x] = dV_y

        # Plotting and saving results
        plt.figure()
        plt.imshow(pi_x, origin='lower', interpolation='none')
        plt.title('X velocity final policy')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.savefig('x_policy.png')
        plt.close()

        plt.figure()
        plt.imshow(pi_y, origin='lower', interpolation='none')
        plt.title('Y velocity final policy')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.savefig('y_policy.png')
        plt.close()


def constructRBFStates(L1, L2, W1, W2, sigma):
    """
    Constructs a dictionary dict[tuple] -> torch.tensor that converts
    tuples (x,y) representing positions to torch tensors used as input to the
    neural network. The tensors have an entry for each valid position on the
    race track. For each position (x,y), the tensor is constructed using the gaussian
    radial basis function with standard deviation sigma. In other words, if entry i corresponds
    to the position p2 = (x2, y2), then the tensor for a point p1 = (x1,y1) will have
    tensor[i] = Gaussian_RBF(p1, p2).

    @type L1: int
        See description in the @RaceCar class.
    @type L2: int
        See description in the @RaceCar class.
    @type W1: int
        See description in the @RaceCar class.
    @type W2: int
        See description in the @RaceCar class.
    @type sigma: float
        The standard deviation of the gaussian radial basis function.
    """
    N_states = (L1+1)*(W1+W2+1)+L2*(W2+1)
    x_coords = torch.zeros(N_states, dtype=torch.float32)
    y_coords = torch.zeros(N_states, dtype=torch.float32)
    state_to_basis = {}
    ind = 0
    for x in range(L1+L2+1):
        for y in range(W1+W2+1):
            if (0<=x<=L1 and 0<=y<=W1+W2) or (0<=x<=L1+L2 and W1<=y<=W1+W2):
                x_coords[ind] = x
                y_coords[ind] = y
                ind += 1

    for x in range(L1 + L2 + 1):
        for y in range(W1 + W2 + 1):
            if (0 <= x <= L1 and 0 <= y <= W1 + W2) or (0 <= x <= L1 + L2 and W1 <= y <= W1 + W2):
                basis = torch.exp(-((x_coords-x)**2 + (y_coords-y)**2)/(2*sigma**2))
                state_to_basis[(x,y)] = basis.view(1, -1).to(device)

    return state_to_basis


if __name__ == "__main__":

    # Track parameters
    L1 = 10
    L2 = 10
    W1 = 20
    W2 = 5
    no_action_prob = 0.1

    # Gaussian RBF standard deviation
    sigma = 1

    # Training parameters
    episodes = 1000000
    n = 20
    epsilon = 0.1
    learning_rate = 0.002

    state_to_basis = constructRBFStates(L1, L2, W1, W2, sigma)

    Car = RaceCar(L1, L2, W1, W2, no_action_prob)
    model = nn.Linear((L1+1)*(W1+W2+1)+L2*(W2+1), 9).to(device)

    #model.load_state_dict(torch.load('race_car_NN_model'))

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    car_agent = Agent(Car, model, optimizer, criterion, state_to_basis)

    car_agent.train(episodes, n, epsilon)

