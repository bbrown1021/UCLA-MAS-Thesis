""" 
Defines the basic maze structure and environment setup. 
"""

# abc module provides the infrastructure for defining custom abstract base classes
from abc import ABC, abstractmethod

# create tuple subclasses with named fields
from collections import namedtuple

# toolkit for environments to devlop/compare reinforcement learning algorithms
import gym
from gym.utils import seeding

# supports large, multi-dimensional arrays and matrices & performs high-level mathematical functions
import numpy as np

# python imaging library to support opening, manipulating, and saving different image formats
from PIL import Image

# imports the Object class
from .utils import Object

class BaseMaze(ABC):
    def __init__(self, **kwargs):

        # **kwargs allows us to pass in an unspecified amount of named key-value parameters for configuration purposes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # abstract method is defined when creating custom environment
        objects = self.make_objects()
        # raises error is all objects do not have the specified Object class 
        assert all([isinstance(obj, Object) for obj in objects])
        # creates a tuple with Objects as a named field
        self.objects = namedtuple('Objects', map(lambda x: x.name, objects), defaults=objects)()
          
    @property
    @abstractmethod # method declared without implementation
    def size(self):
        r"""Returns the dimensions of maze (height, width). """
        pass
        
    @abstractmethod # method declared without implementation
    def make_objects(self):
        r"""Returns the list of defined objects. """
        pass

    def _convert(self, x, name):
        r"""Returns the value of the named attribute for each object. """
        for obj in self.objects:
            pos = np.asarray(obj.positions)
            x[pos[:, 0], pos[:, 1]] = getattr(obj, name, None)
        return x
    
    def to_name(self):
        r"""Returns the name for each object. """
        x = np.empty(self.size, dtype=object)
        return self._convert(x, 'name')
    
    def to_value(self):
        r"""Returns the assigned value for each object in their current maze location. """
        x = np.empty(self.size, dtype=int)
        return self._convert(x, 'value')
    
    def to_rgb(self):
        r"""Returns the rgb color tuple for each object. """
        x = np.empty((*self.size, 3), dtype=np.uint8)
        return self._convert(x, 'rgb')
    
    def to_impassable(self):
        r"""Returns boolean of whether an object is impassable. """
        x = np.empty(self.size, dtype=bool)
        return self._convert(x, 'impassable')
    
    # DELETE? 
    def __repr__(self):
        r"""Returns the string representation of an object. """
        return f'{self.__class__.__name__}{self.size}'

class BaseEnv(gym.Env, ABC):
    """ 
    Render modes for the environment: 
    - human: renders to the current display/terminal, returns nothing (designed for human consumption)
    - rgb_array: returns a numpy array with shape (x,y,3) to represent RGB values for a (x,y) pixel image (designed for videos)
    """
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second' : 10}
    reward_range = (-float('inf'), float('inf')) 
    
    def __init__(self):
        self.viewer = None # called when rendering
        self.seed() # for random number generation
    
    @abstractmethod # method declared without implementation
    def step(self, action):
        r"""environment updates given the agent's action"""
        pass
    
    def seed(self, seed=None):
        r"""initializes the random number generator if the seed is manually passed"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    @abstractmethod # method declared without implementation
    def reset(self):
        r"""resets environment to an inital state and returns an initial observation"""
        pass
    
    @abstractmethod # method declared without implementation
    def get_image(self):
        r"""function defined to retrieve the maze image for rendering"""
        pass
    
    def render(self, mode='human', max_width=500):
        r"""renders the maze environment"""
        
        # RGB tuple as 8-bit unsigned integer
        img = self.get_image()
        img = np.asarray(img).astype(np.uint8) 

        # resize img while maintaining aspect ratio
        img_height, img_width = img.shape[:2] # extract height and width of image
        ratio = max_width/img_width
        img = Image.fromarray(img).resize([int(ratio*img_width), int(ratio*img_height)])
        img = np.asarray(img)

        if mode == 'rgb_array':
            return img # render numpy array 
        elif mode == 'human':
            from gym.envs.classic_control.rendering import SimpleImageViewer
            if self.viewer is None:
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img) # renders in standard image formats
            
            return self.viewer.isopen
            
    def close(self):
        r"""releases the viewer in human mode"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
