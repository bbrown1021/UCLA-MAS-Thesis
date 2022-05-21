from collections import namedtuple

# four directions
VonNeumannMotion = namedtuple('VonNeumannMotion', 
                              ['north', 'south', 'west', 'east'], 
                              defaults=[[-1, 0], [1, 0], [0, -1], [0, 1]])