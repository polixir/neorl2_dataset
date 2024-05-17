import os
import json

import math
import torch

import gymnasium
import neorl2

from neorl2.utils.g2p2c_utils.pumpAction import Pump
from neorl2.utils.g2p2c_utils.ppo_core import StateSpace


class PolicyWrapper:
    def __init__(self, config, model_path):
        self.config = config

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy = torch.load(model_path)
        self.policy = self.policy.to(self.device)
        self.policy.eval()

        self.pump = Pump(self.config)
        self.state_space = StateSpace(self.config)

        self.env_init = True
        self.state = None
        self.last_action = None

    def reset(self):
        self.env_init = True
        self.state = None
        self.last_action = None

    def get_action(self, state):
        policy_step = self.policy.get_action(state)
        selected_action = policy_step['action'][0]
        rl_action, pump_action = self.pump.action(agent_action=selected_action)
        return rl_action, pump_action
    
    def predict(self, obs):
        if self.env_init:
            self.last_action = 0
            self.state = self.state_space.update(cgm=obs[0], ins=self.last_action)
            _, pump_action = self.get_action(self.state)
            self.last_action = pump_action
            self.env_init = False
            # return pump_action  # pump_action in [0, 5]
            return 2 * pump_action / 5 - 1  # in [-1, 1]

        self.state = self.state_space.update(cgm=obs[0], ins=self.last_action)
        _, pump_action = self.get_action(self.state)
        self.last_action = pump_action
        # return pump_action  # pump_action in [0, 5]
        return 2 * pump_action / 5 - 1  # in [-1, 1]


if __name__ == "__main__":
    patient_names = ["adolescent#006",
                        "adolescent#007", 
                        "adolescent#008", 
                        "adolescent#009",
                        "adolescent#010", 
                        "adult#006",
                        "adult#007", 
                        "adult#008", 
                        "adult#009", 
                        "adult#010",
                        "child#006", 
                        "child#007", 
                        "child#008", 
                        "child#009", 
                        "child#010"]
    
    for patient_name in patient_names:
        config_path = "/home/ubuntu/chenjiawei/NeoRL2/scripts/logs/Simglucose/config.json"
        # Load the JSON file
        with open(config_path, "r") as json_file:
            config = json.load(json_file)

        pump = Pump(config)
        state_space = StateSpace(config)

        model_path = "/home/ubuntu/chenjiawei/NeoRL2/scripts/logs/Simglucose/models/best_model.pth"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        policy = torch.load(model_path)
        policy = policy.to(device)
        policy.eval()
        
        env = gymnasium.make("Simglucose", mode='train')
        obs, _ = env.reset()

        init_basal_action = 0
        state = state_space.update(cgm=obs[0], ins=init_basal_action)
        # for _ in range(config["feature_history"]):
        #     basal_action = pump.get_basal()
        #     obs, reward, is_done, _, info = env.step(basal_action)
        #     state = state_space.update(cgm=obs[0], ins=basal_action)
        
        rew = 0
        a_action = []
        for t in range(0, 1000):
            policy_step = policy.get_action(state)
            selected_action = policy_step['action'][0]
            _, pump_action = pump.action(agent_action=selected_action)

            a_action.append(pump_action)

            next_obs, reward, done, trunc, info = env.step(pump_action)
            state = state_space.update(cgm=next_obs[0], ins=pump_action)
            rew += reward
            if done or trunc:
                # print(next_obs[0])
                break
        print(t)
        print(rew)

        # *****************************************************

        policy = PolicyWrapper(config, model_path)
        # env = gymnasium.make("Simglucose", mode='test')
        obs, _ = env.reset()

        rew = 0
        b_action = []
        for t in range(0, 1000):
            pump_action = policy.predict(obs)
            b_action.append(pump_action)
            next_obs, reward, done, trunc, info = env.step(pump_action)
            rew += reward
            obs = next_obs
            if done or trunc:
                # print(next_obs[0])
                break
        print(t)
        print(rew)
        
        # breakpoint()
        # print(rew)
        print('*'*20)





