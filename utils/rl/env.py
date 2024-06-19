import gym
import numpy as np
import gym.spaces as spaces

class CustomEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}  
    def __init__(self, df):
        super(CustomEnv, self).__init__()
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
    
    def reset(self):
        return super().reset()
    
    
    def step(self, action):
        return super().step(action)

    
    
    
    