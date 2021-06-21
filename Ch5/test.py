import random
import numpy as np
import matplotlib.pyplot as plt

def convertActionToVelocity(action):
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
        
if __name__ == "__main__":
    L1 = 10
    L2 = 10
    W1 = 20
    W2 = 5


    pi = np.load("pi.npy")

    pi_x = np.zeros((W1+W2+1, L1+L2+1))
    pi_y = np.zeros((W1+W2+1, L1+L2+1))

    # Change all the positions outside the track to some negative number for contrast
    for y in range(len(pi)):
        for x in range(len(pi[0])):
            if not ((0<= x <= L1 and 0<= y <= W1+W2) or (L1 <= x <= L1+L2 and W1 <= y <= W1+W2)):
                pi_x[y][x] = -10
                pi_y[y][x] = -10
            else:
                dV_x, dV_y = convertActionToVelocity(pi[y][x])
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
