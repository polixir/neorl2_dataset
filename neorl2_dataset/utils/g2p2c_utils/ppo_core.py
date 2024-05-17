import math
import torch
import numpy as np
from collections import deque
import neorl2.utils.g2p2c_utils.core as core
from neorl2.utils.g2p2c_utils.core import custom_reward, custom_reward_traj


class Memory:
    def __init__(self, config, device):
        self.size = config["n_step"]
        self.device = device
        self.feature_hist = config["feature_history"]
        self.features = config["n_features"]
        self.gamma = config["gamma"]
        self.lambda_ = config["lambda_"]
        self.observation = np.zeros(core.combined_shape(self.size, (self.feature_hist, self.features)), dtype=np.float32)
        self.actions = np.zeros(self.size, dtype=np.float32)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.state_values = np.zeros(self.size + 1, dtype=np.float32)
        self.logprobs = np.zeros(self.size, dtype=np.float32)
        self.first_flag = np.zeros(self.size + 1, dtype=np.bool_)
        self.cgm_target = np.zeros(self.size, dtype=np.float32)
        self.ptr, self.path_start_idx, self.max_size = 0, 0, self.size

    def store(self, obs, act, rew, val, logp, cgm_target, counter):
        assert self.ptr < self.max_size
        self.observation[self.ptr] = obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.state_values[self.ptr] = val
        self.logprobs[self.ptr] = logp
        self.first_flag[self.ptr] = True if counter == 0 else False
        self.cgm_target[self.ptr] = cgm_target
        self.ptr += 1

    def finish_path(self, final_v):
        self.state_values[self.ptr] = final_v
        self.first_flag[self.ptr] = False

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        data = dict(obs=self.observation, act=self.actions, v_pred=self.state_values,
                    logp=self.logprobs, first_flag=self.first_flag, reward=self.rewards, cgm_target=self.cgm_target)
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in data.items()}


class StateSpace:
    def __init__(self, config):
        self.feature_history = config["feature_history"]
        self.glucose = deque(self.feature_history*[0], self.feature_history)
        self.insulin = deque(self.feature_history*[0], self.feature_history)
        self.glucose_max = config["glucose_max"]
        self.glucose_min = config["glucose_min"]
        self.insulin_max = config["insulin_max"]
        self.insulin_min = config["insulin_min"]

        self.state = np.stack((self.glucose, self.insulin), axis=-1).astype(np.float32)

    def update(self, cgm=0, ins=0):
        cgm = core.linear_scaling(x=cgm, x_min=self.glucose_min, x_max=self.glucose_max)
        ins = core.linear_scaling(x=ins, x_min=self.insulin_min, x_max=self.insulin_max)

        self.glucose.append(cgm)  # self.glucose.appendleft(cgm)
        self.insulin.append(ins)

        self.state = np.stack((self.glucose, self.insulin), axis=-1).astype(np.float32)

        #handcraft_features = [cgm, ins, ins_20, ins_60, ins_120, hour, t_to_meal, snack, main_meal]
        # handcraft_features = [hour]
        return self.state

