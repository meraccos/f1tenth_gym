from stable_baselines3.common.monitor import Monitor

from wrappers import RewardWrapper, FrenetObsWrapper
from wrappers import ActionRandomizer, LidarRandomizer

from gymnasium.experimental.wrappers import RescaleActionV0
from gymnasium.wrappers import NormalizeReward
from gymnasium.wrappers import FilterObservation, TimeLimit
from gymnasium.wrappers import FlattenObservation, FrameStack

import numpy as np
import gymnasium as gym

def create_env(maps, seed=5):
    env = gym.make(
        "f110_gym:f110-v0",
        num_agents=1,
        maps=maps,
        seed=seed,
    )

    env = FrenetObsWrapper(env)
    env = RewardWrapper(env)
    
    # env = FilterObservation(env, filter_keys=["scans", "linear_vel"])
    env = FilterObservation(env, filter_keys=["scans", "linear_vel", "prev_action"])

    env = TimeLimit(env, max_episode_steps=10000)
    env = RescaleActionV0(env, min_action = np.array([-1.0, 0.0]), 
                             max_action = np.array([1.0, 1.0]))
    
    # env = LidarRandomizer(env)
    # env = ActionRandomizer(env)
    
    env = FlattenObservation(env)
    # env = FrameStack(env, 4)
            
    env = Monitor(env, info_keywords=("is_success", ))
    
    env = NormalizeReward(env)    

    return env


def linear_schedule(initial_learning_rate: float):
    def schedule(progress_remaining: float):
        return initial_learning_rate * progress_remaining

    return schedule
