import random
import matplotlib.pyplot as plt
import numpy as np


NUM_COLS = 3
NUM_ROWS = 3
BOARD_LENGTH = NUM_COLS*NUM_ROWS
NUM_STATES = 3**BOARD_LENGTH

def boardToHash(board):
    """ Converts a board to a hash value
        @type board: list[int]
            The game board
        @rtype: int
            The corresponding hash
    """
    
    hash_value = 0
    for i in range(BOARD_LENGTH):
        hash_value += board[i]*3**i
    return hash_value

def hashToBoard(hash_value):
    """ Converts a hash value to a game board
        @type hash_value: int
            The hash value to be converted
        @rtype: list[int]
            The converted game board
    """
    board = [0]*BOARD_LENGTH
    for i in range(BOARD_LENGTH):
        rem = hash_value % 3
        board[i] = rem
        hash_value = (hash_value - rem) / 3

    return board



def checkWin(board, piece):
    """ Checks if the person with piece has won. If not, also checks if the game has ended or not.
        @type board: list[int]
            the board in the form of a list[int]
        @type piece: int
            integer 1 or 2 
        @rtype: tuple[boolean]
            (win, finished) 
    """
    # Checking rows
    for r in range(NUM_ROWS):
        if all([board[t] == piece for t in range(NUM_COLS*r, NUM_COLS*(r+1))]):
            return (1,1)
    # Checking cols
    for c in range(NUM_COLS):
        if all([board[t] == piece for t in range(c, BOARD_LENGTH, NUM_COLS)]):
            return (1,1)
    # Checking diagonals
    for i in range(NUM_ROWS):
        if all([board[t] == piece for t in range(0, BOARD_LENGTH, NUM_ROWS+1)]):
            return (1,1)
    for i in range(NUM_ROWS):
        if all([board[t] == piece for t in range(NUM_COLS-1, BOARD_LENGTH-1, NUM_ROWS-1)]):
            return (1,1)

    # Count how many pieces are on the board:
    tot_pieces = 0
    for i in range(BOARD_LENGTH):
        if board[i] != 0: tot_pieces += 1
    
    return (0, tot_pieces == BOARD_LENGTH)

def testHash():
    """ Tests the logic of the hashing functions.
        Generates random boards and then proceeds to hash and unhash it to compare to the original.
    """
    for _ in range(10000):
        board = np.random.randint(3, size = 9)
        if list(hashToBoard(boardToHash(board))) != list(board):
            print('Hash check failed!')
            break
    print('Hash check success!')

def testWin():
    """ Tests the logic of the checkWin function.
        Generates random 2D boards. Checks winning condition using cleaner row/column logic and comparing
        the results to the 1D logic in the checkWin function.
    """
    def lookForWin(piece, board):
        for i in range(len(board)):
            if all([board[i][j] == piece for j in range(len(board[0]))]): return (1,1)
        for j in range(len(board[0])):
            if all([board[i][j] == piece for i in range(len(board))]): return (1,1)
        if all([board[i][i] == piece for i in range(len(board))]): return (1,1)
        if all([board[i][len(board)-i-1] == piece for i in range(len(board))]): return (1,1)
        tot = 0
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j]!=0: tot += 1
        return (0,tot == len(board)*len(board[0]))
    
    for _ in range(10000):
        board = np.random.randint(3, size = (3,3))
        if lookForWin(1, board) != checkWin(board.flatten(), 1) or lookForWin(2, board) != checkWin(board.flatten(), 2):
            print('Win check failed!')
            break
    print('Win check success!')


# Testing hash function and winCheck logic
testHash()
testWin()


class Board:
    """ The game board.

        The board is represented as a 1D array list[int] of length NUM_ROWS*NUM_COLS where board[i] == 0
        means that position i is empty. A (0 indexed) position (i,j) on the 2D board has (0 indexed) position
        NUM_COLS * i + j in the 1D board. In other words, the indices in the 1D board are obtained by numbering
        the cells in the 2D board from left to right, top to bottom.
    
    """
    def __init__(self):
        """ Initializes empty board"""
        self.board = [0]*BOARD_LENGTH

    def getBoard(self):
        """Returns the current board"""
        return self.board

    def putPiece(self, piece, position):
        """ Places a piece on the board at a certain position.
            @type piece: int
                the piece to be placed
            @type position: int
                the position on the board to place the piece
            @rtype: tuple[boolean]
                the status (win, finished) after the piece has been placed
        """
        if self.board[position] == 0:
            self.board[position] = piece
            win, finished = checkWin(self.board, piece)
            return win, finished
        else:
            raise AttributeError("That's not a valid move!")

    def resetBoard(self):
        """ Resets the board to an empty board"""
        self.board = [0]*BOARD_LENGTH

class Agent:
    """ AI Agent
    """
    def __init__(self, piece, opponent_piece, lr, epsilon):
        """ Initializes agent.

            Intializes the initial win probabilities for all game states.
            A won state gets probability 1, a loss gets 0. All other states is assigned 0.5.
            State winning probabilities are stored in an array with the hash of the board state as the
            array index.
            
            @type piece: int
                piece the agent is using
            @type opponent_piece: int
                piece the opponent is using
            @type lr: float
                learning rate
            @type epsilon: float
                exploration probability. Between 0 and 1.
        """
        self.piece = piece
        self.lr = lr
        self.epsilon = epsilon
        self.states = []
        self.greedy = []
        self.prob = [0]*NUM_STATES
        for i in range(NUM_STATES):
            board = hashToBoard(i)
            win, finished = checkWin(board, piece)
            if win:
                self.prob[i] = 1.0
                continue
            win, finished = checkWin(board, opponent_piece)
            if win:
                self.prob[i] = 0
                continue
            self.prob[i] = 0.5

    def decideAction(self, board):
        """ With probability 1-epsilon, find the move with the maximum winning probability out of all valid moves.
            In case of ties, a random action is chosen.
            With probability epsilon, choose a random move.

            
            @type board: list[int]
                1D representation of the board
            @rtype: int
                representing the best position to put the piece
        """

        x = random.uniform(0,1)

        if x > self.epsilon:
            # Finding best position
            best_pos = []
            cur_max_prob = -2
            for i in range(BOARD_LENGTH):
                if board[i] == 0:
                    win_prob = self.prob[boardToHash(board) + self.piece*3**i]
                    if win_prob > cur_max_prob:
                        cur_max_prob = win_prob
                        best_pos = [i]
                    elif win_prob == cur_max_prob:
                        best_pos.append(i)
                        
            return random.choice(best_pos)
        

        self.greedy[-1] = 0
        valid_pos = []
        for i in range(BOARD_LENGTH):
            if board[i] == 0:
                valid_pos.append(i)
            
        return random.choice(valid_pos)
        
    def setState(self, board):
        """ Places the board state into the agent's history

            @type board: list[int]
                1D representation of the board
        """
        self.greedy.append(1)
        self.states.append(boardToHash(board))

    def updateProbs(self):
        """ Updates the probabilities of winning using the agent's history of states """
        for i in range(len(self.states)-2, -1, -1):
            self.prob[self.states[i]] += self.greedy[i] * self.lr * (self.prob[self.states[i+1]] - self.prob[self.states[i]])

    def resetState(self):
        """ Resets the agent's history of states """
        self.states = []
        self.greedy = []

    def getProbs(self):
        """ Returns the current estimated win probabilities for each state"""
        return self.prob


class AgentSelfPlay:
    """ A class handling the training of agents through self-play """
    
    def __init__(self, epochs, agent1_piece = 1, agent2_piece = 2, lr = 0.1, epsilon = 0.1):
        """ Initializes the training procedure

            Two agents are created to play against each other. After each game, their
            estimated win probabilities are updated based on that game using the
            temporal-difference learning method. Their win/loss track records during
            training are also recorded, where we assign 1 for a win, 0 for a loss, and
            0.5 for a tie.
            
            @type epochs: int
                Number of games to play for training
            @type agent1_piece: int
                piece of agent 1
            @type agent2_piece: int
                piece of agent 2
            @type lr: float
                learning rate
            @type epsilon: float
                exploration probability
            
        """
        self.Agent1 = Agent(agent1_piece, agent2_piece, lr, epsilon)
        self.Agent2 = Agent(agent2_piece, agent1_piece, lr, epsilon)
        self.agent1_piece = agent1_piece
        self.agent2_piece = agent2_piece
        self.board = Board()
        self.epochs = epochs
        self.Agent1_record = []
        self.Agent2_record = []

    def trainAgents(self):
        """ Trains the two agents by making them play against each other.
            
        """
        for epoch in range(self.epochs):
            self.Agent1.setState(self.board.getBoard())
            self.Agent2.setState(self.board.getBoard())
            alternator = self.alternate()
            
            while True:
                current_agent, current_piece = next(alternator)

                best_position = current_agent.decideAction(self.board.getBoard())

                win, finished = self.board.putPiece(current_piece, best_position)
                
                self.Agent1.setState(self.board.getBoard())
                self.Agent2.setState(self.board.getBoard())

                if win or finished:
                    self.board.resetBoard()
                    if win:
                        self.Agent1_record.append(current_piece == self.agent1_piece)
                        self.Agent2_record.append(current_piece == self.agent2_piece)
                    else:
                        self.Agent1_record.append(0.5)
                        self.Agent2_record.append(0.5)
                    break
                
            self.Agent1.updateProbs()
            self.Agent2.updateProbs()
    
            self.Agent1.resetState()
            self.Agent2.resetState()
                
        return self.Agent1, self.Agent2
    
    def alternate(self):
        """ An alternator that alternates between two agents.
        """
        while True:
            yield self.Agent1, self.agent1_piece
            yield self.Agent2, self.agent2_piece

    
    def getAgentRecords(self):
        """ Get the win/loss/tie records of each agent

            @rtype: tuple[list[float]]
        """
        
        return self.Agent1_record, self.Agent2_record



class Compete:
    """ Class to handle competition between two trained agents """
    
    def __init__(self, Agent1, Agent2, agent1_piece = 1, agent2_piece = 2, num_games = 1000):
        """ Initializes the competition

            Each agent's exploration probability is set to 0, so they always play based on learned policy.
            Their win/loss/tie records are recorded, with a 1 for win, 0 for loss, and 0.5 for tie.

            @type Agent1: Agent
                trained agent 1
            @type Agent2: Agent
                trained agent 2
            @type agent1_piece: int
                piece of agent 1
            @type agent2_piece: int
                piece of agent 2
            @type num_games: int
                number of games to play
        """
        self.Agent1 = Agent1
        self.Agent2 = Agent2
        self.Agent1.epsilon = 0
        self.Agent2.epsilon = 0
        self.num_games = num_games
        self.agent1_piece = agent1_piece
        self.agent2_piece = agent2_piece
        self.Agent1_record = []
        self.Agent2_record = []
        self.board = Board()
        

    def playGames(self):
        """ Make the two agents play against each other """
        
        for game in range(self.num_games):
            alternator = self.alternate()

            while True:
                current_agent, current_piece = next(alternator)

                best_position = current_agent.decideAction(self.board.getBoard())

                win, finished = self.board.putPiece(current_piece, best_position)
            
                
                if win or finished:
                    self.board.resetBoard()
                    if win:
                        self.Agent1_record.append(current_piece == self.agent1_piece)
                        self.Agent2_record.append(current_piece == self.agent2_piece)
                    else:
                        self.Agent1_record.append(0.5)
                        self.Agent2_record.append(0.5)
                    break

    def alternate(self):
        """ Alternator that alternates between the two agents """
        while True:
            yield self.Agent1, self.agent1_piece
            yield self.Agent2, self.agent2_piece

    def getAgentRecords(self):
        """ Gets agent win/loss/tie records

            @rtype: tuple[list[float]]

        """
        return self.Agent1_record, self.Agent2_record

class PlotRecords:
    """ Class for plotting agents' track records """
    
    def __init__(self, Agent1_records, Agent2_records, plot_title, look_back = 50):
        """ Calculates the average score over the last look_back number of games for each agent

            @type Agent1_records: list[float]
                agent1's records
            @type Agent2_records: list[float]
                agent2's records
            @type plot_title: str
                title of the plot
            @type look_back: int
                number of games to average for the score
        """
        
        self.Agent1_runningAvg = []
        self.Agent2_runningAvg = []
        self.plot_title = plot_title
        self.look_back = look_back
        for i in range(len(Agent1_records)):
            self.Agent1_runningAvg.append(sum(Agent1_records[max(0,i+1-look_back):i+1])/float(min(i+1, look_back)))
            self.Agent2_runningAvg.append(sum(Agent2_records[max(0,i+1-look_back):i+1])/float(min(i+1, look_back)))


    def plot(self):
        """ Plots track records """
        plt.figure()
        plt.plot(self.Agent1_runningAvg, color = 'r', label = 'Agent 1')
        plt.plot(self.Agent2_runningAvg, color = 'b', label = 'Agent 2')
        plt.ylabel('Average score of last ' + str(self.look_back) + ' games')
        plt.xlabel('Epochs')
        plt.ylim(0, 1)
        plt.title(self.plot_title)
        plt.legend()
        plt.show()


        
if __name__ == '__main__':

    # Training agents through self-play
    selfPlay = AgentSelfPlay(epochs = int(10000))
    
    Agent1, Agent2 = selfPlay.trainAgents()

    Agent1_record, Agent2_record = selfPlay.getAgentRecords()

    PlotRecords(Agent1_record, Agent2_record, 'Training Record').plot()
    
    # Make trained agent compete with each other
    compete = Compete(Agent1, Agent2)

    compete.playGames()

    Agent1_record, Agent2_record = compete.getAgentRecords()
    
    PlotRecords(Agent1_record, Agent2_record, 'Competition Record').plot()

