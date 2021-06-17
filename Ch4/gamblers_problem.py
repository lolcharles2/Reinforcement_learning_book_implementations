
import numpy as np
import matplotlib.pyplot as plt

class GamblersProblem:
    """ Implementation of Gambler's Problem, example 4.3 of
        Chapter 4 of "Reinforcement Learning, an Introduction" by Sutton and
        Barto. 

        A gambler with some starting capital will bet on the result of a coin flip.
        If the coin ends up heads, then he wins his stake, whereas if it ends up tails,
        he loses his stake.
        At each flip he must choose how much of his capital to bet. The game terminates
        if he loses all of his money, or his total capital exceeds some maximum.
        His goal is to make his capital exceed some maximum value.
        
    """
    def __init__(self, max_capital):
        """ Initializes the car rental.

            @type max_capital: int
                The capital at which the gambler has won.

        """
        self.max_capital = max_capital
        

    def getNextState(self, C, bet, heads):
        """ Gets the next state given the current capital C,
            the bet that is placed, and if the coin came up heads.

            @type C: int
                The current capital of the gambler.
            @type bet: int
                The amount which the gambler has bet.
            @type heads: boolean
                True if the coin came up heads, false otherwise.
            @rtype: int
                The gambler's capital after this round.
            
        """
        if not 0 <= bet <= min(C, self.max_capital - C):
            raise ValueError("That's not a valid bet!")
        
        return C + bet if heads else C - bet
        

    def getReward(self, C, C_next):
        """ Gets the reward given the current capital and the next capital.

            @type C: int
                Current capital of the gambler.
            @type C_next: int
                Next capital of the gambler
            @rtype: int
                Reward value
        """
        return 1.0 if C_next == self.max_capital else 0.0
        
class ValueIteration:
    """

        A class using value iteration to find the optimal policy for the gambler.
        
    """
    
    def __init__(self, max_capital, p_h, R, t, gamma):
        """ Initializes the value iteration

            @type max_capital: int
                The capital at which the gambler will win.
            @type p_h: float
                The probability of the coin landing on heads.
            @type R: function(C, C_next) -> float
                A reward function that gives an reward based on the
                current capital C and the next capital C_next.
            @type t: function(C, bet, heads) -> int
                A state transition function that gives the next
                capital given the current capital C, the current bet,
                and if the coin came up heads.
            @type gamma: float
                The discount factor.
        """
        self.max_capital = max_capital
        self.p_h = p_h
        self.R = R
        self.t = t
        self.gamma = gamma
        self.V = np.zeros(max_capital + 1)

    def train(self, iterations):
        """ Performs value iteration until the difference in
            values across all states between two successive iterations is less
            than a threshold.

        """
        delta = float('inf')
        
        for _ in range(iterations):
            delta = 0
            for capital in range(1, max_capital):
                v = self.V[capital]

                new_v = []

                for bet in range(1, min(capital, self.max_capital - capital) + 1):
                    
                    heads_value = self.p_h*(self.R(capital, self.t(capital, bet, True)) + self.gamma * self.V[self.t(capital, bet, True)])
                    tails_value = (1 - self.p_h)*(self.R(capital, self.t(capital, bet, False)) + self.gamma * self.V[self.t(capital, bet, False)])

                    new_v.append(heads_value + tails_value)
                        

                self.V[capital] = max(new_v)
                delta = max(delta, abs(v-self.V[capital]))

            
        print("Value iteration terminated!")

    def getValues(self):
        """ Returns the value function

            @rtype: list[float]
                A list representing the value for each capital.
        """
        return self.V
    
    def buildPolicy(self):
        """ Builds the policy based on the current value function.

            @rtype: list[int]
                A list representing the optimal betting values for each
                value of capital.
        """
        pi = np.zeros(self.max_capital + 1)
        for capital in range(1, self.max_capital):
            values = []
            for bet in range(1, min(capital, self.max_capital - capital) + 1):
                
                heads_value = self.p_h*(self.R(capital, self.t(capital, bet, True)) + self.gamma * self.V[self.t(capital, bet, True)])
                tails_value = (1 - self.p_h)*(self.R(capital, self.t(capital, bet, False)) + self.gamma * self.V[self.t(capital, bet, False)])

                values.append(heads_value + tails_value)

            pi[capital] = values.index(max(values)) + 1

        return pi    
        
if __name__ == "__main__":
    
    max_capital = 127
    p_h = 0.4
    gamma = 1.0
    
    env = GamblersProblem(max_capital)

    agent = ValueIteration(max_capital, p_h, env.getReward, env.getNextState, gamma)

    agent.train(200)

    V = agent.getValues()

    pi = agent.buildPolicy()

    

    # Plotting final results
    plt.figure()
    plt.plot(V)
    plt.title("Final Policy")
    plt.xlabel("Current capital")
    plt.ylabel("Probability of winning")
    plt.ylim(0, 1)
    plt.xlim(1, max_capital-1)

    plt.figure()
    plt.plot(pi)
    plt.title("Final Value Function")
    plt.xlabel("Current capital")
    plt.ylabel("Bet amount")
    plt.xlim(1, max_capital-1)
    plt.ylim(0)
    
    plt.show()
