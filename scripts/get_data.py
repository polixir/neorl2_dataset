import os
import json
import ray
import glob
import fire
import torch
import neorl2
import numpy as np
import gymnasium as gym

from neorl2.utils.utils import set_seed, mkdir_p
from neorl2.utils.g2p2c_utils.policy_wrapper import PolicyWrapper
from neorl2.utils.fusion_policy import SB2_model


set_seed(42)

@ray.remote(num_cpus=1, num_gpus=0.1)
def collect_data(policy, env_name, seed=None):
    env = neorl2.make(env_name)
    data = {
        "obs": [],
        "action": [],
        "next_obs": [],
        "reward": [],
        "done": [],
        "truncated" : [],
    }
    set_seed(seed)
    obs,_ = env.reset(seed=seed)
    done = False
    truncated = False
    rewards = 0
    lengths = 0
    while not done and not truncated:
        action = policy.predict(obs)
        # clip action
        if isinstance(env.action_space, gym.spaces.Box):      
            action = np.clip(action, env.action_space.low, env.action_space.high)
        if isinstance(env.action_space, gym.spaces.Discrete):
            action = action[0]
        next_obs, reward, done, truncated, _ = env.step(action)
        data['obs'].append(obs)
        data['action'].append(action)
        data['next_obs'].append(next_obs)
        data['reward'].append(reward)
        data['done'].append(done)
        data['truncated'].append(truncated)
        rewards += reward
        lengths += 1
        obs = next_obs

    for k,v in data.items():
        data[k] = np.array(v,np.float32)

    return data

def _collect_trajectory_data(trj_id, env, policy, env_name):
    result = []
    for index in range(trj_id[0],trj_id[1]):
        if isinstance(policy, list):
            result.append(collect_data.remote(policy[index%len(policy)],env_name,seed=index))
        else:
            result.append(collect_data.remote(policy,env_name,seed=index))
        
    results = ray.get(result)

    dataset = {}
    for k, v in results[0].items():
        if k not in  ["done", "truncated"]:
            dataset[k] = np.concatenate([r[k] for r in results], axis=0).astype(np.float32)
        else:
            dataset[k] = np.concatenate([r[k] for r in results], axis=0)
        
    return dataset

def collect_trajectory_data(samples, env, policy, env_name):
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True)
    dataset = None
    trj_id = [0,100]
    while True:
        collect_dataset = _collect_trajectory_data(trj_id, env, policy, env_name)
        if dataset is None:
            dataset = collect_dataset
        else:
            dataset = {k:np.concatenate([dataset[k],v]) for k,v in collect_dataset.items()}

        if len(dataset["reward"]) >= samples:
            break
        trj_id[0] += 100
        trj_id[1] += 100
        
    index_list = list(np.where(np.logical_or(dataset["done"], dataset["truncated"]))[0]+1)
    end_index = next((x for x in index_list if x >= samples), None)
    assert end_index >= samples
    print(end_index, samples)
    trj_nums = index_list.index(end_index) + 1
    dataset = {k:v[:end_index] for k,v in dataset.items()}
    
    if env_name == "SafetyHalfCheetah":
        dataset_indices = np.where((dataset["reward"].reshape(-1,1000) <- 99).sum(axis=1) == 0)[0]
        dataset = {k:np.concatenate([v[i*1000:(i+1)*1000] for i in dataset_indices]) for k,v in dataset.items()}
        trj_nums = len(dataset_indices)

    for k, v in dataset.items():
        print(f"Data {k} shape:", v.shape)
        
    mean_rewards = np.sum(dataset["reward"]) / trj_nums
    #var_rewards = np.var(dataset["reward"].reshape(trj_nums,-1))
    mean_length = len(dataset["reward"]) / trj_nums

    print("Traj Nums:", trj_nums)
    print("Mean Rewards:", mean_rewards,)
    print("Mean Length:", mean_length)

    return dataset

def get_pid_models(file_path, level_ratio):

    with open(file_path, 'r') as file: 
        json_lines = file.readlines()
    
    data_dict = {'reward':[],
                    'PID':[]}
                    
    for _line in json_lines:
        _data = json.loads(_line)
        data_dict['reward'].append(_data['target'])
        data_dict['PID'].append(_data['params'])
    
    def select_PID(percent, num_policy):
        _max = np.max(data_dict['reward'])
        _min = np.min(data_dict['reward'])
        _scale = _max - _min
        _level_rewards = _scale*np.array([percent]) + _min
        print(f'MAX return: {_max:.2f}; MIN return: {_min:.2f}')
        print(f'{percent*100:.0f}% of the return is {_level_rewards.item():.2f}')
        policy_index = np.argsort(np.abs(data_dict['reward']-_level_rewards))[:num_policy]
        _list_return = np.array(data_dict['reward'])[policy_index].tolist()
        _formatted_string = ', '.join([f'{item:.2f}' for item in _list_return])
        print(f'{num_policy} PIDs are with return {_formatted_string}')
        return policy_index

    policy_index = select_PID(level_ratio, 2)
    PIDs = np.array(data_dict['PID'])[policy_index].tolist()
    return PIDs

def get_models(env_name, ratio):
    file_pattern = f"./logs/{env_name}/models/mean_reward_*"
    file_paths = glob.glob(file_pattern)
    file_paths.sort()
    rewards = [float(file_path.split('/')[-1].split('mean_reward_')[1][:-4]) for file_path in file_paths]
    percentile_value = (np.max(rewards) - np.min(rewards)) * ratio + np.min(rewards)
    closest_elements = sorted(rewards, key=lambda x: abs(x - percentile_value))[:4]
    if env_name.lower() in 'salespromotion':
        closest_elements = sorted(rewards)[:int(len(rewards)*ratio)][-4:]
    file_type = [file_path[-4:] for file_path in file_paths if "mean_reward_" in file_path][0]
    models_path = [file_pattern[:-1]+str(i)+file_type for i in closest_elements]

    return models_path

def load_policy(model_path, env_name):
    if "Simglucose" in env_name:
        assert model_path.endswith(".pth")
        config_path = './logs/Simglucose/config.json'
        # Load the JSON file
        with open(config_path, "r") as json_file:
            config = json.load(json_file)
        return PolicyWrapper(config, model_path)
    elif 'Fusion' in env_name:
        level = model_path
        dir, _ = os.path.split(os.path.abspath(neorl2.__file__))
        rl_designer_model_path = os.path.join(dir, 'envs/data/fusion_lstm/weights/best_model.zip')
        _low_state   = [0.35, 1.68, 0.2, 0.5, 1.265, 2.18, 1.1, 3.8, 0.84, 1.1, 3.8, 0.84, 1.15, 1.15, 0.45]
        _high_state  = [0.75, 1.9, 0.5, 0.8, 1.34, 2.29, 2.1, 6.2, 1.06, 2.1, 6.2, 1.06, 1.75, 1.75, 0.6]
        _low_action  = [0.35, 1.68, 0.2, 0.5, 1.265, 2.18]
        _high_action = [0.75, 1.9, 0.5, 0.8, 1.34, 2.29]

        level_map = {
            "low" : 0.1,
            "high" : 0.05,
            # "high" : 0.00,
            }
        level_ratio = level_map[level]
        
        designer = SB2_model(
                model_path = rl_designer_model_path, 
                low_state = _low_state, 
                high_state = _high_state, 
                low_action = _low_action, 
                high_action = _high_action,
                level = level_ratio
            )
        return designer
    # PID
    elif isinstance(model_path, dict):
        from neorl2.utils.pid import MultiPIDController
        dim = int(len(model_path)/3)
        if dim==2:
            _policy = MultiPIDController((model_path['p1'],model_path['p2']),
                                        (model_path['i1'],model_path['i2']),
                                        (model_path['d1'],model_path['d2']),
                                        (0,0))
        elif dim==1:
            _policy = MultiPIDController([model_path['p']],
                                        [model_path['i']],
                                        [model_path['d']],
                                        [0])
        class Policy:
            def __init__(self, pid_policy, dim, env_name):
                self.policy = pid_policy
                self.dim = dim
                if env_name == 'DMSD':
                    self.obs_index = [0,1]
                    self.target_index = [4,5]
                    self.dt = 0.2
                elif env_name == 'Fusion':
                    self.obs_index = [1]
                    self.target_index = [2]
                    self.dt = 0.025

                self.err_p = np.zeros((self.dim, ))
                self.err_i = np.zeros((self.dim, ))
                self.err_d = np.zeros((self.dim, ))
                self.err_hist = [self.err_p]

            def predict(self, obs):
                _positions = obs[..., self.obs_index]    
                _targets =   obs[..., self.target_index]
                self.err_p = _targets-_positions
                self.err_hist.append(self.err_p)

                self.err_i = np.sum(np.stack(self.err_hist), axis=0) *  self.dt
                self.err_d = (self.err_hist[-1] - self.err_hist[-2]) / self.dt
                _temp_pid = np.hstack([self.err_p, self.err_i, self.err_d])      

                action, _ = self.policy.get_actions(_temp_pid)
                return action[0]
        
        return Policy(_policy, dim, env_name)
    # SB3
    elif model_path.endswith(".zip"):
        from stable_baselines3 import PPO
        class Policy:
            def __init__(self, model_path):
                self.policy = PPO.load(model_path, device="cpu")
                
            def predict(self, obs):
                return self.policy.predict(obs, deterministic=True)
            
        return Policy(model_path)
    
    #TS for salespromoton
    elif 'Salespromotion' in env_name:
        import torch
        class Policy:
            def __init__(self, model_path):
                self.policy = torch.load(model_path)
                
            def predict(self, obs):
                if obs.ndim == 1:
                    obs = obs.reshape(1,-1)  
                    batch = 1
                else:
                    assert obs.ndim == 2, obs.shape
                    batch = obs.shape[0]

                with torch.no_grad():
                    mu, state = self.policy(torch.from_numpy(obs))
                squashed_action = torch.tanh(mu[0]).cpu().detach().numpy()
                act = np.clip(squashed_action, -1.0, 1.0)

                low, high = -1 ,1
                act_out = low + (high - low) * (act + 1.0) / 2.0
                if batch == 1:
                    act_out = act_out.reshape(-1)   
                return act_out

        return Policy(model_path)

    # TS
    elif isinstance(model_path, str) and model_path.endswith(".pth"):
        import torch
        class Policy:
            def __init__(self, model_path):
                self.policy = torch.load(model_path)
                
            def predict(self, obs):
                if obs.ndim == 1:
                    obs = obs.reshape(1,-1)  
                    batch = 1
                else:
                    assert obs.ndim == 2, obs.shape
                    batch = obs.shape[0]
                mu, state = self.policy(torch.from_numpy(obs))
                if isinstance(mu,torch.Tensor):
                    action = mu.argmax(axis=-1).cpu().detach().numpy()[0]
                else:
                    mu = mu[0]
                    action = mu.cpu().detach().numpy()
                if batch == 1:
                    action = action.reshape(-1)                
                return action
        return Policy(model_path)
    
    
    else:
        raise NotImplementedError
        

def get_task_datasets(env_name, level, train_trj_nums, val_trj_nums=None):
    level_map = {
        # "low" : 0.35,
        # "medium" : 0.5,
        "high" : 0.75,
    }
    
    level_ratio = level_map[level]
    
    if 'DMSD' in env_name:
        _json_path = f'./logs/{env_name}/{env_name}_tune_log.json'
        models_path = get_pid_models(_json_path, level_ratio)
        # train_models_path = models_path[:1] + models_path[2:]
        train_models_path = models_path[:-1]
        val_models_path= models_path[-1:]
    elif env_name in 'Fusion':
        models_path = 'None' 
        train_models_path = [level]
        val_models_path= [level]
    else:
        if os.path.exists(f"./logs/{env_name}/models/{level}"):
            file_pattern = f"./logs/{env_name}/models/{level}/mean_reward_*"
            models_path = glob.glob(file_pattern)
            models_path.sort()
        else:
            models_path = get_models(env_name, level_ratio)
        train_models_path = models_path[:1] + models_path[2:]
        val_models_path= models_path[1:2]
    print(f"Use train models: {train_models_path}")
    print(f"Use validate models: {val_models_path}")

    env = neorl2.make(env_name)
    train_policy = [load_policy(model_path, env_name) for model_path in train_models_path]
    train_dataset = collect_trajectory_data(train_trj_nums, env, train_policy, env_name)
    
    if val_trj_nums:
        env = neorl2.make(env_name)
        val_policy = [load_policy(model_path, env_name) for model_path in val_models_path]
        val_dataset = collect_trajectory_data(val_trj_nums, env, val_policy, env_name)
        return train_dataset, val_dataset
        
    return train_dataset

def main(env_name=None, 
         level=None, 
         train_samples=100000, 
         val_samples=20000, 
         data_dir="./dataset/"):
    if env_name is None:
        env_name_list = neorl2.__all__
    else:
        if not isinstance(env_name, list):
            env_name_list = [env_name,]
        else:
            env_name_list = env_name
            
    if level is None:
        # level_list = ["medium", "low", "high"]
        level_list = ["high"]
    else:
        if not isinstance(level, list):
            level_list = [level,]
        else:
            level_list = level
            
    for env_name in env_name_list:
        if 'Fusion'==env_name:
            pass
        elif not os.path.exists(f"./logs/{env_name}"):
            print(f"Not find {env_name} models. ")
            continue
        
        env_data_dir = os.path.join(data_dir, env_name)
        mkdir_p(env_data_dir)
        for level in level_list:
            print(f"---------------------------------------{env_name}-{level}--------------------------------------------")
            train_dataset, val_dataset = get_task_datasets(env_name, level, train_samples, val_samples)
            np.savez_compressed(os.path.join(env_data_dir, f"{env_name}-{level}-{train_samples}-train.npz"), **train_dataset)
            np.savez_compressed(os.path.join(env_data_dir, f"{env_name}-{level}-{val_samples}-val.npz"), **val_dataset)
            print(f"-----------------------------------------------------------------------------------------------------")


if __name__ == "__main__":
    fire.Fire(main)