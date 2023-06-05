from stable_baselines3 import PPO
from utils import create_env

# maps = list(range(1, 450))
maps = ['final']
# maps = list(range(50))
maps = [51]
env = create_env(maps=maps, seed=5)

# model_path = "/Users/meraj/workspace/f1tenth_gym/work/models/act_obs_handmade_stack4_real/act_obs_handmade_stack4_real_4550k.zip"
# model_path = "/Users/meraj/workspace/f1tenth_gym/work/models/act_obs_4stack_hm_train_hm_eval/act_obs_4stack_hm_train_hm_eval_5600k.zip"
model_path = "/Users/meraj/workspace/f1tenth_gym/work/models/yaml_fix/yaml_fix_100k.zip"
model_path = "/Users/meraj/workspace/f1tenth_gym/work/models/experiment/experiment_2500k.zip"
model = PPO.load(path=model_path)

obs, _ = env.reset()
done = False

while not done:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
