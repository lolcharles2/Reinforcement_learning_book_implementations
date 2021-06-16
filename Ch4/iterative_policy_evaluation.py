import numpy as np

class IPE:
    """Iterative Policy Evaluation

        A class to find the state value function given a stochastic policy.
        Here, we assume the MDP to have deterministic transitions. That is,
        given a current state S and an action A, there is a deterministic function
        t that yields the next state S': S' = t(S, A).
    """
    
    def __init__(self, n_states, actions, pi, R, t, gamma, threshold):
        """ Initializes the IPE

            @type n_states: int
                The number of states in the MDP.
                The states are represented by numbers from 0 to n_states-1, with 0
                being the terminal state.
            @type actions: list[int]
                A list of valid actions represented by integers.
                We assume the list of valid actions
                is identical for all non-terminal states.
            @type pi: function(action, state) -> float
                The policy for which the state value function is to
                be evaluated. The policy takes as input a (action, state)
                pair, and outputs the probability of taking that action
                in that state.
            @type R: function(action, state) -> float
                The reward function, which yields the reward for taking
                an action in a given state.
            @type t: function(action, state) -> int
                The deterministic state transition function, which yields
                the next state after taking an action in a current state.
            @type gamma: float
                The discount factor
            @type threshold: float
                The threshold tolerance for the state value function. The
                iteration will terminate when the maximum difference in the
                state value function across all states between two successive
                iterations is smaller than threshold.
       
        """
        self.actions = actions
        self.t = t
        self.R = R
        self.pi = pi
        self.V = [0]*n_states
        self.gamma = gamma
        self.threshold = threshold

    def train(self):
        """ Performs iterative policy evaluation until the difference in
            values across all states between two successive iterations is less
            than a threshold.

        """
        delta = float('inf')
        while delta >= self.threshold:
            delta = 0
            for state in range(1, len(self.V)):
                v = self.V[state]
                self.V[state] = sum([pi(action, state)*(self.R(action, state)+self.gamma*self.V[self.t(action, state)]) for action in actions])
                delta = max(delta, abs(v-self.V[state]))

    def getValues(self):
        """ Returns the current values for each state
            @rtype: list[float]
        """
        return self.V


class GridWorld:
    """ Implementation of the grid world example as described in
        Example 4.1 in Chapter 4 of "Reinforcement Learning, An
        Introducton" by Sutton and Barto.

        The enviornment is a grid of size N_ROWS x N_COLS. The states
        in the grid are numbered from 0 to N_ROWS x N_COLS - 1 from
        left to right, top to bottom. State 0 and N_ROWS x N_COLS - 1
        are both (the same) terminal states. In each non-terminal state,
        one can move up, right, down, and left. These actions are represented
        by the numbers 0,1,2,3. If an action moves off the grid, then the state does
        not change. Each transition results in a reward of -1 (even those that move
        off the grid). 

    """
    def __init__(self, N_ROWS, N_COLS):
        self.N_ROWS = N_ROWS
        self.N_COLS = N_COLS
        self.N_STATES = N_COLS*N_ROWS


    def getNextState(self, action, state):
        """ Get the next state given the current (non-terminal) state and an action.
            If the action moves off the board, then the state is unchanged.

            @type action: int
                A number from 0 to 3
            @type state: int
                A number from 1 to N_COLS*N_ROWS - 2
            @rtype: int
                A number from 0 to N_COLS*N_ROWS - 1
        """
        if action not in [0,1,2,3]:
            raise ValueError("That's not a valid action!")
        
        if state <= 0 or state >= N_COLS*N_ROWS - 1:
            raise ValueError("That's not a valid non-terminal state!")
        
        if action == 0:
            new_state = state - N_COLS
            return new_state % (self.N_STATES - 1) if 0 <= new_state <= self.N_STATES - 1 else state
        elif action == 1:
            new_state = state + 1
            return new_state % (self.N_STATES - 1) if new_state <= (state//self.N_COLS) * self.N_COLS + self.N_COLS - 1 else state
        elif action == 2:
            new_state = state + N_COLS
            return new_state % (self.N_STATES - 1) if 0 <= new_state <= self.N_STATES - 1 else state
        else:
            new_state = state - 1
            return new_state % (self.N_STATES - 1) if new_state >= (state//self.N_COLS) * self.N_COLS else state

    def getReward(self, action, state):
        """ Returns the reward from performing an action in a state
            @rtype: float
        """
        return -1.0


if __name__ == "__main__":
    
    N_ROWS = 4
    N_COLS = 4
    gamma = 1
    threshold = 0.0001
    
    actions = [0,1,2,3]
    
    grid = GridWorld(N_ROWS, N_COLS)

    # Uniformly random policy
    def pi(action, state):
        return 1.0/len(actions)

    estimator = IPE(N_ROWS*N_COLS - 1, actions, pi, grid.getReward, grid.getNextState, gamma, threshold)
    estimator.train()

    V = estimator.getValues()

    print(np.reshape(V+[0], (N_ROWS, N_COLS)))
