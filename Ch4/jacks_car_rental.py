
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import itertools
import copy

class CarRental:
    """ Implementation of Jack's Car Rental enviornment, example 4.2 of
        Chapter 4 of "Reinforcement Learning, an Introduction" by Sutton and
        Barto. 

        Jack manages a car rental at two locations. Each day, a random
        number of customers arrive at each location to rent cars, and a
        random number of cars get returned at each location. The numbers are
        drawn from a Poisson distribution. Jack makes profit for each car that is
        rented, but if he does not have a car at a location when requested, then
        that business is lost. In order to ensure cars are avaliable when requested,
        he can move some number of cars between the two locations over night. Cars
        that are returned are avaliable for rental the following day.
    """
    def __init__(self, max_cars, max_move, move_car_cost, free_move, parking_penalty,
                 rent_credit, lmbda1_rent, lmbda2_rent, lmbda1_rtn,
                 lmbda2_rtn, max_limit):
        """ Initializes the car rental.

            @type max_cars: int
                Maximum number of cars at each rental location at any given time.
                Any extra cars that arrive are returned to a distribution center and
                disappear from the problem.
            @type max_move: int
                The maximum number of cars Jack can move from one place to another per
                night.
            @type move_car_cost: int
                The cost for moving each car between locations over night.
            @type free_move: int
                The number of free moves from location 1 to location 2 per night.
            @type parking_penalty: int
                Penalty incurred for each location when the number of cars there
                exceeds 10.
            @type rent_credit: int
                The profit per car rented out.
            @type lmbda1_rent: int
                The average value of the poission distribution for rentals at location 1.
            @type lmbda2_rent: int
                The average value of the poission distribution for rentals at location 2.
            @type lmbda1_rtn: int
                The average value of the poission distribution for returns at location 1.
            @type lmbda2_rtn: int
                The average value of the poission distribution for returns at location 2.
            @type max_limit: int
                The maximum number of rentals and returns considered when computing state
                transition probabilities.

        """
        self.max_cars = max_cars
        self.max_move = max_move
        self.free_move = free_move
        self.parking_penalty = parking_penalty
        self.move_car_cost = move_car_cost
        self.rent_credit = rent_credit
        self.lmbda1_rent = lmbda1_rent
        self.lmbda2_rent = lmbda2_rent
        self.lmbda1_rtn = lmbda1_rtn
        self.lmbda2_rtn = lmbda2_rtn
        self.probs = np.zeros((max_limit+1,max_limit+1,max_limit+1,max_limit+1))

        for rent1, rent2, rtn1, rtn2 in itertools.product(range(max_limit+1), repeat = 4):
            p_rent1 = poisson.pmf(rent1, self.lmbda1_rent)
            p_rent2 = poisson.pmf(rent2, self.lmbda2_rent)
            p_rtn1 = poisson.pmf(rtn1, self.lmbda1_rtn)
            p_rtn2 = poisson.pmf(rtn2, self.lmbda2_rtn)
            self.probs[rent1, rent2, rtn1, rtn2] = p_rent1 * p_rent2 * p_rtn1 * p_rtn2



    def getProb(self, rent1, rent2, rtn1, rtn2):
        """ Gets the probability of the state transition given the number of
            rentals and returns at each location during a day.

            @type rent1: int
                Number of cars rented out in location 1.
            @type rent2: int
                Number of cars rented out in location 2.
            @type rtn1: int
                Number of cars returned in location 1.
            @type rtn2: int
                Number of cars returned in location 2.
            @rtype: float
                Probability of transition.
        """
        return self.probs[rent1, rent2, rtn1, rtn2]
        

    def getNextState(self, rent1, rent2, rtn1, rtn2, n1, n2, moved):
        """ Gets the next state given the current state, number of
            cars moved over night, and the number of rentals and returns
            at each location during the following day.

            @type rent1: int
                Number of cars rented out in location 1.
            @type rent2: int
                Number of cars rented out in location 2.
            @type rtn1: int
                Number of cars returned in location 1.
            @type rtn2: int
                Number of cars returned in location 2.
            @type n1: int
                Number of cars currently at location 1.
            @type n2: int
                Number of cars currently at location 2.
            @type moved: int
                Number of cars moved from location 1 to 2 over night.
                If moved < 0, then it representents moving from 2 to 1.
            @rtype: tuple[int]
                A tuple of 2 numbers representing the number of cars
                at location 1 and 2 at the end of the following day.
            
        """
        
        if moved >= 0: moved = min(n1, moved)
        else: moved = - min(n2, -moved)

        new_n1 = max(0, n1 - moved - rent1)
        new_n2 = max(0, min(n2 + moved, self.max_cars) - rent2)

        return int(min(new_n1 + rtn1, self.max_cars)), int(min(new_n2 + rtn2, self.max_cars))

    def getReward(self, rent1, rent2, n1, n2, moved):
        """ Gets the reward given the current state, tne number of cars moved over night,
            and the number of rentals in the following day.

            @type rent1: int
                Number of cars rented out in location 1.
            @type rent2: int
                Number of cars rented out in location 2.
            @type n1: int
                Number of cars currently at location 1.
            @type n2: int
                Number of cars currently at location 2.
            @type moved: int
                Number of cars moved from location 1 to 2 over night.
                If moved < 0, then it representents moving from 2 to 1.
            @rtype: int
                Reward value
        """
        if moved >= 0: moved = min(n1, moved)
        else: moved = - min(n2, -moved)
        
        reward = 0
        reward += min(rent1, n1 - moved)*self.rent_credit
        reward += min(rent2, min(n2 + moved, self.max_cars))*self.rent_credit

        if moved > self.free_move: reward -= (moved - self.free_move)*self.move_car_cost
        elif moved < 0: reward -= abs(moved)*self.move_car_cost

        if n1 - moved > 10: reward -= self.parking_penalty
        if n2 + moved > 10: reward -= self.parking_penalty

        return reward
        
class PolicyEvaluation:
    """Iterative Policy Evaluation

        A class to find the state value function given a determinstic policy.
        
    """
    
    def __init__(self, max_cars, pi, R, t, p, gamma, threshold, max_limit, V0):
        """ Initializes the IPE

            @type max_cars: int
                Maximum number of cars at each location
            @type pi: function(n1, n2) -> int
                The policy for which the state value function is to
                be evaluated. The policy takes as input n1, n2 representing
                the number of cars at locations 1 and 2, and outputs the number
                of cars to move from 1 to 2.
            @type R: function(rent1, rent2, n1, n2, moved) -> int
                The reward function. Outputs the reward.
            @type t: function(rent1, rent2, rtn1, rtn2, n1, n2, moved) -> tuple[int]
                The state transition function. Outputs the next state.
            @type p: function(rent1, rent2, rtn1, rtn2) -> float
                The probability of a state transition.
            @type gamma: float
                The discount factor
            @type threshold: float
                The threshold tolerance for the state value function. The
                iteration will terminate when the maximum difference in the
                state value function across all states between two successive
                iterations is smaller than threshold.
            @type max_limit: int
                The maximum number of rentals and returns considered when computing state
                transition probabilities.
            @type V0: array[float]
                The initial state values, where V0[n1][n2] is the state value for n1 and n2
                cars at location 1 and 2 respectively.
       
        """
        self.t = t
        self.R = R
        self.p = p
        self.pi = pi
        self.V = V0
        self.gamma = gamma
        self.threshold = threshold
        self.max_limit = max_limit

    def train(self):
        """ Performs iterative policy evaluation until the difference in
            values across all states between two successive iterations is less
            than a threshold.

        """
        delta = float('inf')

        tuples = [(a,b,c,d) for a,b,c,d in itertools.product(range(self.max_limit+1), repeat = 4)]
        
        while delta >= self.threshold:
            delta = 0
            for n1 in range(max_cars+1):
                for n2 in range(max_cars+1):
                    v = self.V[n1][n2]
                    
                    moved = self.pi[n1][n2]

                    new_v = 0.0                    
                    for rent1, rent2, rtn1, rtn2 in tuples:
                        
                        new_n1, new_n2 = self.t(rent1, rent2, rtn1, rtn2, n1, n2, moved)
                        prob = self.p(rent1, rent2, rtn1, rtn2)
                        r = self.R(rent1, rent2, n1, n2, moved)
                        
                        new_v += prob*(r + self.gamma * self.V[new_n1][new_n2])


                    self.V[n1][n2] = new_v
                    delta = max(delta, abs(v-new_v))
            print('Current Delta is: ', delta)

    

        
    def getValues(self):
        """ Returns the current values for each state
            @rtype: list[float]
        """
        return self.V

class PolicyImprovement:
    """ Given a state value function, improves the policy greedily by choosing
        an action providing the highest value for the subsequent state.

    """
    def __init__(self, V, max_move, R, t, p, gamma, max_limit):
        """ Initializes the policy improvement

            @type V: array[float]
                The state values, where V[n1][n2] is the state value for n1 and n2
                cars at location 1 and 2 respectively.
            @type max_move: int
                The maximum number of cars that can be moved from one location to
                the other in one night.
            @type R: function(rent1, rent2, n1, n2, moved) -> int
                The reward function. Outputs the reward.
            @type t: function(rent1, rent2, rtn1, rtn2, n1, n2, moved) -> tuple[int]
                The state transition function. Outputs the next state.
            @type p: function(rent1, rent2, rtn1, rtn2) -> float
                The probability of a state transition.
            @type gamma: float
                The discount factor
            @type max_limit: int
                The maximum number of rentals and returns considered when computing state
                transition probabilities.
                
        """
        self.V = V
        self.pi = np.zeros(np.shape(self.V))
        self.max_move = max_move
        self.R = R
        self.t = t
        self.p = p
        self.gamma = gamma
        self.max_limit = max_limit

    def improve(self):
        """ Constructs a policy by taking the action at each state that maximizes the
            state value for the subsequent state.
        """
        
        tuples = [(a,b,c,d) for a,b,c,d in itertools.product(range(self.max_limit+1), repeat = 4)]
        
        for n1 in range(len(self.V)):
            for n2 in range(len(self.V[0])):
                rewards = []
                for action in range(-min(self.max_move, n2), min(self.max_move, n1)+1):
                    v = 0
                    for rent1, rent2, rtn1, rtn2 in tuples:

                        new_n1, new_n2 = self.t(rent1, rent2, rtn1, rtn2, n1, n2, action)
                        prob = self.p(rent1, rent2, rtn1, rtn2)
                        r = self.R(rent1, rent2, n1, n2, action)
                        
                        v += prob * (r + self.gamma * self.V[new_n1][new_n2])
                                     
                    rewards.append(v)
                                     
                self.pi[n1][n2] = rewards.index(max(rewards)) - min(self.max_move, n2)

    def getPolicy(self):
        """ Gets the current policy

            @rtype: array[int]
                The current policy pi, where pi[n1][n2] is the number of cars
                to be moved over night when the current number of cars is n1, n2
                at locations 1 and 2 respectively.
        """
        return self.pi


                    
        
        
if __name__ == "__main__":
    
    max_cars = 20
    max_move = 5
    free_move = 0
    parking_penalty = 0
    move_car_cost = 2
    rent_credit = 10
    
    lmbda1_rent = 3
    lmbda2_rent = 4
    lmbda1_rtn = 3
    lmbda2_rtn = 2

    gamma = 0.9
    threshold = 1
    max_limit = 10

    # Car rental enviornment
    rental = CarRental(max_cars, max_move, free_move, parking_penalty,
                 move_car_cost, rent_credit, lmbda1_rent, lmbda2_rent, lmbda1_rtn,
                 lmbda2_rtn, max_limit)


    # Initializing state values and policy
    V = np.zeros((max_cars+1, max_cars+1))
    pi = np.zeros((max_cars+1, max_cars+1))

    # Number of cycles to run
    cycles = 5
    
    for cycle in range(cycles):

        print("Current cycle: ", cycle + 1)

        # Policy Evaluation
        PE = PolicyEvaluation(max_cars, pi, rental.getReward, rental.getNextState, rental.getProb, gamma, threshold, max_limit, copy.deepcopy(V))

        PE.train()

        print("Policy evaluated!")

        # Policy Improvement
        new_V = PE.getValues()

        PI = PolicyImprovement(new_V, max_move, rental.getReward, rental.getNextState, rental.getProb, gamma, max_limit)

        PI.improve()

        print("Policy improved!")
        
        pi = PI.getPolicy()

        V = new_V

    # Plotting final results
    plt.figure()
    plt.imshow(pi, origin='lower', interpolation='none')
    plt.title("Final Policy")
    plt.xlabel("# of cars at location 1")
    plt.ylabel("# of cars at location 2")

    plt.figure()
    plt.imshow(V, origin='lower', interpolation='none')
    plt.title("Final Value Function")
    plt.xlabel("# of cars at location 1")
    plt.ylabel("# of cars at location 2")
    
    plt.show()
