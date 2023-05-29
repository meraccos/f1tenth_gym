from stable_baselines3 import PPO
from utils import create_env

maps = list(range(1, 450))

env = create_env(maps=maps, seed=5)

model_path = "/Users/meraj/workspace/f1tenth_gym/work/models/new_reward_50k"
model = PPO.load(path=model_path)

obs, _ = env.reset()
done = False

while not done:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
