from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from utils import create_env, linear_schedule
from callbacks import TensorboardCallback, CustomEvalCallback

from torch.nn import Mish

if __name__ == "__main__":
    save_interval = 5e4
    eval_freq = 5e4
    n_eval_episodes = 20
    learn_steps = 1e7
    log_name = "lr00002_clip_01"
    print(f"\nMODEL: {log_name}\n")

    save_path = f"./models/{log_name}/{log_name}"
    log_dir = "./metrics/"
    maps = list(range(1, 450))

    # lr_schedule = linear_schedule(0.00015)

    env = create_env(maps=maps, seed=8)
    eval_env = create_env(maps=maps, seed=8)

    policy_kwargs = dict(activation_fn=Mish,
                         net_arch=dict(pi=[32, 32], vf=[64, 64]))

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=1024,
        ent_coef=0.0,
        learning_rate=0.00002,
        # learning_rate=lr_schedule,
        batch_size=256,
        # max_grad_norm=0.5,
        # gae_lambda=0.95,
        gamma=0.999,
        n_epochs=15,
        clip_range=0.1,
        tensorboard_log=log_dir,
        device="cpu",
        policy_kwargs=policy_kwargs
    )
        
    callbacks = CallbackList([TensorboardCallback(save_interval, save_path), 
                              CustomEvalCallback(eval_env,
                                                best_model_save_path="./best_models/",
                                                log_path=log_dir,
                                                n_eval_episodes=n_eval_episodes,
                                                eval_freq=eval_freq)])
    
    model.learn(
        total_timesteps=learn_steps,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=log_name,
    )

    env.close()
