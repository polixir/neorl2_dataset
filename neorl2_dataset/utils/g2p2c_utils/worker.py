import os
from pathlib import Path
import csv
import gymnasium as gym
import neorl2
import numpy as np
import pandas as pd
from collections import deque

from neorl2.utils.g2p2c_utils.pumpAction import Pump
from neorl2.utils.g2p2c_utils.core import get_env, time_in_range, custom_reward, combined_shape, linear_scaling, inverse_linear_scaling
from neorl2.utils.g2p2c_utils.ppo_core import Memory, StateSpace


class Worker:
    def __init__(self, config, mode, env_name, seed, worker_id, device):
        self.config = config
        self.episode = 0
        self.worker_mode = mode
        self.worker_id = worker_id
        self.update_timestep = config["n_step"]
        self.max_test_epi_len = config["max_test_epi_len"]
        self.max_epi_length = config["max_epi_length"]
        self.calibration = config["calibration"]
        self.simulation_seed = seed + 100

        self.env = get_env(env_name=env_name, mode=mode, custom_reward=custom_reward, seed=self.simulation_seed)
        self.state_space = StateSpace(self.config)
        self.pump = Pump(self.config)
        self.std_basal = self.pump.get_basal()
        self.memory = Memory(self.config, device)
        self.episode_history = np.zeros(combined_shape(self.max_epi_length, 11), dtype=np.float32)
        self.reinit_flag = False
        if mode == "training":
            self.init_env()
        self.log1_columns = ['epi', 't', 'cgm', 'meal', 'ins', 'rew', 'rl_ins', 'mu', 'sigma',
                             'prob', 'state_val']
        self.log2_columns = ['epi', 't', 'reward', 'normo', 'hypo', 'sev_hypo', 'hyper', 'lgbi',
                             'hgbi', 'ri', 'sev_hyper']
        self.save_log([self.log1_columns], '/'+self.worker_mode+'/data/logs_worker_')
        self.save_log([self.log2_columns], '/'+self.worker_mode+'/data/'+self.worker_mode+'_episode_summary_')

    def init_env(self):
        if not self.reinit_flag:
            self.episode += 1
        self.counter = 0
        self.init_state, _ = self.env.reset()
        self.cur_state = self.state_space.update(cgm=self.init_state[0], ins=0)

    def rollout(self, policy):
        ri, alive_steps, normo, hypo, sev_hypo, hyper, lgbi, hgbi, sev_hyper = 0, 0, 0, 0, 0, 0, 0, 0, 0
        rollout_steps = self.update_timestep

        for n_steps in range(0, rollout_steps):
            policy_step = policy.get_action(self.cur_state)
            selected_action = policy_step['action'][0]
            rl_action, pump_action = self.pump.action(agent_action=selected_action)
            state, reward, is_done, trunc, info = self.env.step(pump_action)
            scaled_cgm = linear_scaling(x=state[0], x_min=self.config["glucose_min"], x_max=self.config["glucose_max"])
            self.memory.store(self.cur_state, policy_step['action'][0],
                                reward, policy_step['state_value'], policy_step['log_prob'], scaled_cgm, self.counter)
            # update -> state.
            # self.cur_state, self.feat = self.state_space.update(cgm=state[0], ins=pump_action,
            #                                                     meal=info['remaining_time'], hour=(self.counter+1),
            #                                                     meal_type=info['meal_type'], carbs=info['future_carb']) #info['day_hour']
            self.cur_state = self.state_space.update(cgm=state[0], ins=pump_action) #info['day_hour']
            self.episode_history[self.counter] = [self.episode, self.counter, state[0], info['meal'] * info['sample_time'],
                                                  pump_action, reward, rl_action, policy_step['mu'][0], policy_step['std'][0],
                                                  policy_step['log_prob'][0], policy_step['state_value'][0]]
            self.counter += 1
            stop_factor = self.max_epi_length - 1

            criteria = trunc or is_done
            if criteria:  # episode termination criteria.
                final_val = policy.get_final_value(self.cur_state)
                self.memory.finish_path(final_val)

                df = pd.DataFrame(self.episode_history[0:self.counter], columns=self.log1_columns)
                df.to_csv(self.config["log_dir"] + '/' + self.worker_mode + '/data/logs_worker_' + str(self.worker_id) + '.csv',
                          mode='a', header=False, index=False)
                alive_steps = self.counter
                normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper = time_in_range(df['cgm'], df['meal'], df['ins'],
                                                                             self.episode, self.counter, display=False)
                self.save_log([[self.episode, self.counter, df['rew'].sum(), normo, hypo, sev_hypo, hyper, lgbi,
                                hgbi, ri, sev_hyper]],
                              '/' + self.worker_mode + '/data/' + self.worker_mode + '_episode_summary_')

                self.init_env()

        data = self.memory.get()
        return data

    def rollout_test(self, policy):
        ri, alive_steps, normo, hypo, sev_hypo, hyper, lgbi, hgbi, sev_hyper = 0, 0, 0, 0, 0, 0, 0, 0, 0
        self.init_env()
        rollout_steps = self.max_test_epi_len

        for n_steps in range(0, rollout_steps):
            policy_step = policy.get_action(self.cur_state)
            selected_action = policy_step['action'][0]
            rl_action, pump_action = self.pump.action(agent_action=selected_action)
            state, reward, is_done, trunc, info = self.env.step(pump_action)
            self.cur_state = self.state_space.update(cgm=state[0], ins=pump_action) #info['day_hour']
            self.episode_history[self.counter] = [self.episode, self.counter, state[0], info['meal'] * info['sample_time'],
                                                  pump_action, reward, rl_action, policy_step['mu'][0], policy_step['std'][0],
                                                  policy_step['log_prob'][0], policy_step['state_value'][0]]
            self.counter += 1
            stop_factor = self.max_test_epi_len - 1

            criteria = trunc or is_done
            if criteria:  # episode termination criteria.
                df = pd.DataFrame(self.episode_history[0:self.counter], columns=self.log1_columns)
                df.to_csv(self.config["log_dir"] + '/' + self.worker_mode + '/data/logs_worker_' + str(self.worker_id) + '.csv',
                          mode='a', header=False, index=False)
                alive_steps = self.counter
                normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper = time_in_range(df['cgm'], df['meal'], df['ins'],
                                                                             self.episode, self.counter, display=False)
                # print(df['rew'].sum())
                self.save_log([[self.episode, self.counter, df['rew'].sum(), normo, hypo, sev_hypo, hyper, lgbi,
                                hgbi, ri, sev_hyper]],
                              '/' + self.worker_mode + '/data/' + self.worker_mode + '_episode_summary_')

                break  # stop rollout if this is a testing worker!

        data = [ri, alive_steps, normo, hypo, sev_hypo, hyper, lgbi, hgbi, sev_hyper]
        return data, df['rew'].sum()

    def save_log(self, log_name, file_name):
        full_path = self.config["log_dir"] + file_name + str(self.worker_id) + '.csv'
        mk_data_path = '/'.join(full_path.split('/')[:-1])
        Path(os.path.join(os.getcwd(), mk_data_path)).mkdir(parents=True, exist_ok=True)
        with open(full_path, 'a+') as f:
            csvWriter = csv.writer(f, delimiter=',')
            csvWriter.writerows(log_name)
            f.close()
