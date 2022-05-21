""" 
Generates a N x N maze with random rectangular shapes in a labeled, bounded box. 
"""

import numpy as np
from skimage.draw import random_shapes

def random_shape_maze(width, height, max_shapes, max_size, allow_overlap, shape=None):
    
    # creates a matrix with non-zero values (colored spaces) and the value 255 (black spaces)
    x, _ = random_shapes(
    	[height, width] # set number of rows and columns to generate
    	, max_shapes # set maximum number of shapes to fit into defined space
    	, max_size=max_size # maximum dimension of each shape
    	, channel_axis=None # new in recent skimage version: 'None' sets a single channel
    	, shape='rectangle' # only generates rectangle shapes of various sizes
    	, allow_overlap=allow_overlap # shapes cannot overlap
    	)
    
    # set grayscale for custom maze environment
    x[x == 255] = 0 # empty spaces
    x[np.nonzero(x)] = 1 # obstacles
    
    # manually create borders for the defined space
    x[0, :] = 1
    x[-1, :] = 1
    x[:, 0] = 1
    x[:, -1] = 1
    
    return x

maze_size = 20

# generate and save random mazes with accessible start and end locations
for i in range(4):
    search = True
    while (search):
        x = random_shape_maze(width=maze_size, height=maze_size, max_shapes=15, max_size=20, allow_overlap=False)
        if (x[2,2] == 1 or x[maze_size-3,maze_size-3] == 1):
            search = True
        else:
            search = False
    
    file_name = "sample_envs/maze%d.npy" % i
    np.save(file_name, x)
