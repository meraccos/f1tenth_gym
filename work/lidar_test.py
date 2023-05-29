import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from utils import create_env
import random

maps = list(range(1, 250))
random.seed(2)

env = create_env(maps=maps, seed=5)

model_path = "/Users/meraj/workspace/f1tenth_gym/work/models/lr0005/lr0005_50k.zip"
model = PPO.load(path=model_path, env=env)

obs, _ = env.reset()
done = False

# Set up the LiDAR data plot
plt.ion()
fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

while not done:
    action, _state = model.predict(obs)
    print(f'{action=}')
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Get the LiDAR data from obs
    lidar_data = obs[0:2055]

    # Convert LiDAR data to polar coordinates
    num_angles = lidar_data.size
    full_lidar_angle = np.pi * 270 / 180  # degrees
    angles = np.linspace(-full_lidar_angle / 2, full_lidar_angle / 2, num_angles)

    # Update the LiDAR data plot
    ax.clear()
    ax.plot(angles, lidar_data.flatten(), marker="o", markersize=2, linestyle="None")
    ax.set_title("Real-time LiDAR data")
    ax.set_ylim(0, np.max(lidar_data))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    plt.draw()
    plt.pause(0.00001)

    env.render()

# Close the LiDAR data plot
plt.ioff()
plt.close(fig)
