from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CallbackList
from utils import create_env, linear_schedule
from callbacks import TensorboardCallback, CustomEvalCallback

from torch.nn import Mish

if __name__ == "__main__":
    save_interval = 5e4
    eval_freq = 5e4
    n_eval_episodes = 25
    learn_steps = 1e7
    # log_name = "act_obs_4stack_hm_train_hm_eval"
    log_name = "cleanup"

    print(f"{log_name=}")

    save_path = f"./models/{log_name}/{log_name}"
    log_dir = "./metrics/"
    tr_maps = list(range(0, 300))
    ev_maps = list(range(0, 300))#+ list(range(300, 350))

    # lr_schedule = linear_schedule(0.0003)

    env = create_env(maps=tr_maps, seed=8)
    eval_env = create_env(maps=ev_maps, seed=8)
    policy_kwargs = dict(activation_fn=Mish,
                         net_arch=dict(pi=[32, 32], vf=[64, 64]))

    # model = RecurrentPPO(
    model = PPO(
        "MlpPolicy",
        # "MlpLstmPolicy",
        env,
        verbose=0,
        n_steps=1024,
        ent_coef=0,
        learning_rate=0.0001,
        # learning_rate=lr_schedule,
        batch_size=256,
        # max_grad_norm=0.5,
        # gae_lambda=0.95,
        gamma=0.998,
        n_epochs=10,
        clip_range=0.02,
        tensorboard_log=log_dir,
        device="cpu",
        policy_kwargs=policy_kwargs
    )
        
    callbacks = CallbackList([TensorboardCallback(save_interval, save_path), 
                              CustomEvalCallback(eval_env,
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
