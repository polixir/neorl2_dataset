import os
import shutil
import numpy as np


def train_plicy(env_name, algo_name="sac",mode="train", **kwargs):
    seed = kwargs.get("seed", 42)
    if 'DMSD' in env_name:
        from neorl2.utils.pid import pid_train
        log_path = f"./logs/{env_name}/{env_name}_tune_log.json"
        
        if mode !="train":
            capital = 10
        else:
            capital = 1000
        pid_train(env_name, log_path, horizon=100, rollouts=50, dim=2, dt=0.2, obs_index = [0,1], target_index=[4,5],capital=capital)
        
    elif "Simglucose" in env_name:
        from neorl2.utils.g2p2c import G2P2C
        if mode =="train":
            total_timesteps = 30000
        else:
            total_timesteps = 3
        log_dir = f"./logs/{env_name}"
        
        model = G2P2C(env_name, verbose=1, log_dir=log_dir, seed=seed)
        model.learn(total_timesteps=total_timesteps)
        
    else: 
        if algo_name == "sac":
            from neorl2.utils.ts import train_sac
            kwargs["task"] = env_name

            kwargs["epoch"] = 200
            kwargs["step_per_epoch"] = 1000
            
            if "Pipeline" in env_name:
                kwargs["step_per_epoch"] = 200
                kwargs["epoch"] = 200
                
            if "RocketRecovery" in env_name:
                kwargs["step_per_epoch"] = 500
                kwargs["epoch"] = 300
                
            if "Fusion" in env_name:
                kwargs['training_num'] = 1
                kwargs['test_num'] = 1

            if "Salespromotion" in env_name:
                kwargs['training_num'] = 1
                kwargs['test_num'] = 100
                kwargs["step_per_epoch"] = 5000
                kwargs["epoch"] = 200
                

            if "RandomFrictionHopper" in env_name:
                kwargs["epoch"] = 500
                
            if mode !="train":
                kwargs["epoch"] = 2
                kwargs["step_per_epoch"] = 100
                
            if mode !="train":
                kwargs["epoch"] = 2
                kwargs["step_per_epoch"] = 100
                
            train_sac(**kwargs)
        
        
        elif algo_name == "ppo" or algo_name == "ppo_rnn":
            import neorl2
            import gymnasium as gym
            from stable_baselines3 import PPO
            from sb3_contrib import RecurrentPPO
            from neorl2.utils.sb3 import EvalCallback
            
            seed=42
            total_timesteps = kwargs.get("total_timesteps", 1e6)
            if mode !="train":
                total_timesteps = 1e4
            env = gym.make(env_name)
            eval_env = gym.make(env_name)
            
            log_path=f"./logs/{env_name}"
            model_save_path = f"./logs/{env_name}/models"
            tensorboard_log_path = f"./logs/{env_name}/tensorboard"
            
            if os.path.exists(model_save_path):
                shutil.rmtree(model_save_path)
                
            if os.path.exists(tensorboard_log_path):
                shutil.rmtree(tensorboard_log_path)
            
            checkpoint_callback = EvalCallback(eval_env, 
                                            best_model_save_path=model_save_path,
                                            log_path=log_path, 
                                            n_eval_episodes=100,
                                            eval_freq=1000,
                                            deterministic=True, 
                                            render=False)

            if algo_name == "ppo":
                model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_path,seed=seed)
            else:
                # Reference: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html
                model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=tensorboard_log_path,seed=seed)
            
            model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
            
        else:
            raise NotImplementedError