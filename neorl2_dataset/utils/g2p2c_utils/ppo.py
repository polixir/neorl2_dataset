import os
import gc
from pathlib import Path
import gym
import random
import csv
import time
import math
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

from neorl2.utils.g2p2c_utils.core import f_kl, r_kl
from neorl2.utils.g2p2c_utils.reward_normalizer import RewardNormalizer
from neorl2.utils.g2p2c_utils.worker import Worker
from neorl2.utils.g2p2c_utils.models import ActorCritic


class PPO:
    def __init__(self, config, device, load, path1, path2):
        self.config = config
        self.n_step = config["n_step"]
        self.feature_history = config["feature_history"]
        self.n_features = config["n_features"]
        self.grad_clip = config["grad_clip"]
        self.gamma = config["gamma"]
        self.lambda_ = config["lambda_"]
        self.entropy_coef = config["entropy_coef"]
        self.eps_clip = config["eps_clip"]
        self.train_v_iters = config["n_vf_epochs"]
        self.train_pi_iters = config["n_pi_epochs"]
        self.target_kl = config["target_kl"]
        self.pi_lr = config["pi_lr"]
        self.vf_lr = config["vf_lr"]
        self.batch_size = config["batch_size"]
        self.n_training_workers = config["n_training_workers"]
        self.n_testing_workers = config["n_testing_workers"]
        self.device = device
        self.best_rew = 0

        self.policy = ActorCritic(config, load, path1, path2, device).to(self.device)
        self.optimizer_Actor = torch.optim.Adam(self.policy.Actor.parameters(), lr=self.pi_lr)
        self.optimizer_Critic = torch.optim.Adam(self.policy.Critic.parameters(), lr=self.vf_lr)
        self.value_criterion = nn.MSELoss()
        self.shuffle_rollout = config["shuffle_rollout"]
        self.normalize_reward = config["normalize_reward"]
        self.reward_normaliser = RewardNormalizer(num_envs=self.n_training_workers, cliprew=10.0,
                                                  gamma=self.gamma, epsilon=1e-8, per_env=False)
        self.return_type = config["return_type"]
        self.rollout_buffer = {}

        self.old_states = torch.rand(self.n_training_workers, self.n_step, self.feature_history, self.n_features,
                                     device=self.device)
        self.old_actions = torch.rand(self.n_training_workers, self.n_step, device=self.device)
        self.old_logprobs = torch.rand(self.n_training_workers, self.n_step, device=self.device)
        self.reward = torch.rand(self.n_training_workers, self.n_step, device=self.device)
        self.v_targ = torch.rand(self.n_training_workers, self.n_step, device=self.device)
        self.adv = torch.rand(self.n_training_workers, self.n_step, device=self.device)
        self.cgm_target = torch.rand(self.n_training_workers, self.n_step, device=self.device)
        self.v_pred = torch.rand(self.n_training_workers, self.n_step + 1, device=self.device)
        self.first_flag = torch.rand(self.n_training_workers, self.n_step + 1, device=self.device)

        self.save_log([['policy_grad', 'value_grad', 'val_loss', 'exp_var', 'true_var', 'pi_loss']], '/model_log')
        self.model_logs = torch.zeros(7, device=self.device)
        self.save_log([['status', 'rollout', 't_rollout', 't_update', 't_test']], '/experiment_summary')
        self.save_log([[1, 0, 0, 0, 0]], '/experiment_summary')
        self.completed_interactions = 0
        self.distribution = torch.distributions.Normal
        self.start_planning = False

        if self.config["verbose"]:
            print('Policy Network Parameters: {}'.format(
                sum(p.numel() for p in self.policy.Actor.parameters() if p.requires_grad)))
            print('Value Network Parameters: {}'.format(
                sum(p.numel() for p in self.policy.Critic.parameters() if p.requires_grad)))

    def save_log(self, log_name, file_name):
        Path(os.path.join(os.getcwd(), self.config["log_dir"])).mkdir(parents=True, exist_ok=True)
        with open(self.config["log_dir"] + file_name + '.csv', 'a+') as f:
            csvWriter = csv.writer(f, delimiter=',')
            csvWriter.writerows(log_name)
            f.close()

    def compute_gae(self):
        orig_device = self.v_pred.device
        assert orig_device == self.reward.device == self.first_flag.device
        vpred, reward, first = (x.cpu() for x in (self.v_pred, self.reward, self.first_flag))
        first = first.to(dtype=torch.float32)
        assert first.dim() == 2
        nenv, nstep = reward.shape
        assert vpred.shape == first.shape == (nenv, nstep + 1)
        adv = torch.zeros(nenv, nstep, dtype=torch.float32)
        lastgaelam = 0
        for t in reversed(range(nstep)):
            notlast = 1.0 - first[:, t + 1]
            nextvalue = vpred[:, t + 1]
            # notlast: whether next timestep is from the same episode
            delta = reward[:, t] + notlast * self.gamma * nextvalue - vpred[:, t]
            adv[:, t] = lastgaelam = delta + notlast * self.gamma * self.lambda_ * lastgaelam
        vtarg = vpred[:, :-1] + adv
        return adv.to(device=orig_device), vtarg.to(device=orig_device)

    def prepare_rollout_buffer(self):
        '''concat data from different workers'''
        s_hist = self.old_states.view(-1, self.feature_history, self.n_features)
        act = self.old_actions.view(-1, 1)
        logp = self.old_logprobs.view(-1, 1)
        v_targ = self.v_targ.view(-1)
        adv = self.adv.view(-1)
        cgm_target = self.cgm_target.view(-1)
        first_flag = self.first_flag.view(-1)
        buffer_len = s_hist.shape[0]

        if self.shuffle_rollout:
            rand_perm = torch.randperm(buffer_len)
            s_hist = s_hist[rand_perm, :, :]  # torch.Size([batch, n_steps, features])
            act = act[rand_perm, :]  # torch.Size([batch, 1])
            logp = logp[rand_perm, :]  # torch.Size([batch, 1])
            v_targ = v_targ[rand_perm]  # torch.Size([batch])
            adv = adv[rand_perm]  # torch.Size([batch])
            cgm_target = cgm_target[rand_perm]

        self.rollout_buffer = dict(s_hist=s_hist, act=act, logp=logp, ret=v_targ,
                                   adv=adv, len=buffer_len, cgm_target=cgm_target)

    def train_pi(self):
        print('Running pi update...')
        temp_loss_log = torch.zeros(1, device=self.device)
        policy_grad, pol_count = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        continue_pi_training, buffer_len = True, self.rollout_buffer['len']
        for i in range(self.train_pi_iters):
            start_idx, n_batch = 0, 0
            while start_idx < buffer_len:
                n_batch += 1
                end_idx = min(start_idx + self.batch_size, buffer_len)
                old_states_batch = self.rollout_buffer['s_hist'][start_idx:end_idx, :, :]
                old_actions_batch = self.rollout_buffer['act'][start_idx:end_idx, :]
                old_logprobs_batch = self.rollout_buffer['logp'][start_idx:end_idx, :]
                advantages_batch = self.rollout_buffer['adv'][start_idx:end_idx]
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-5)
                self.optimizer_Actor.zero_grad()
                logprobs, dist_entropy, _, _ = self.policy.evaluate_actor(old_states_batch, old_actions_batch)
                ratios = torch.exp(logprobs - old_logprobs_batch)
                ratios = ratios.squeeze()
                surr1 = ratios * advantages_batch
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * dist_entropy.mean()
                # print('\nPPO debug ratio: {}, adv_mean {}, adv_sigma {}'.format(ratios.mean().detach().cpu().numpy(),
                #       advantages_batch.mean().detach().cpu().numpy(), advantages_batch.std().detach().cpu().numpy()))

                # early stop: approx kl calculation
                log_ratio = logprobs - old_logprobs_batch
                approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).detach().cpu().numpy()
                if approx_kl > 1.5 * self.target_kl:
                    if self.config["verbose"]:
                        print('Early stop => Epoch {}, Batch {}, Approximate KL: {}.'.format(i, n_batch, approx_kl))
                    continue_pi_training = False
                    break
                if torch.isnan(policy_loss):  # for debugging only!
                    print('policy loss: {}'.format(policy_loss))
                    exit()
                temp_loss_log += policy_loss.detach()
                policy_loss.backward()
                policy_grad += torch.nn.utils.clip_grad_norm_(self.policy.Actor.parameters(), self.grad_clip)
                pol_count += 1
                self.optimizer_Actor.step()
                start_idx += self.batch_size
            if not continue_pi_training:
                break
        mean_pi_grad = policy_grad / pol_count if pol_count != 0 else 0
        print('The policy loss is: {}'.format(temp_loss_log))
        return mean_pi_grad, temp_loss_log

    def train_vf(self):
        print('Running vf update...')
        explained_var = torch.zeros(1, device=self.device)
        val_loss_log = torch.zeros(1, device=self.device)
        val_count = torch.zeros(1, device=self.device)
        value_grad = torch.zeros(1, device=self.device)
        true_var = torch.zeros(1, device=self.device)
        buffer_len = self.rollout_buffer['len']
        for i in range(self.train_v_iters):
            start_idx = 0
            while start_idx < buffer_len:
                end_idx = min(start_idx + self.batch_size, buffer_len)
                old_states_batch = self.rollout_buffer['s_hist'][start_idx:end_idx, :, :]
                returns_batch = self.rollout_buffer['ret'][start_idx:end_idx]

                self.optimizer_Critic.zero_grad()
                state_values, _, _ = self.policy.evaluate_critic(old_states_batch, action=None, cgm_pred=False)
                value_loss = self.value_criterion(state_values, returns_batch)
                value_loss.backward()
                value_grad += torch.nn.utils.clip_grad_norm_(self.policy.Critic.parameters(), self.grad_clip)
                self.optimizer_Critic.step()
                val_count += 1
                start_idx += self.batch_size

                # logging.
                val_loss_log += value_loss.detach()
                y_pred = state_values.detach().flatten()
                y_true = returns_batch.flatten()
                var_y = torch.var(y_true)
                true_var += var_y
                explained_var += 1 - torch.var(y_true - y_pred) / (var_y + 1e-5)
        #print('\nvalue update: explained varience is {} true variance is {}'.format(explained_var / val_count, true_var / val_count))
        return value_grad / val_count, val_loss_log, explained_var / val_count, true_var / val_count

    def update(self, rollout):
        if self.return_type == 'discount':
            if self.normalize_reward:  # reward normalisation
                self.reward = self.reward_normaliser(self.reward, self.first_flag)
            self.adv, self.v_targ = self.compute_gae()  # # calc returns

        if self.return_type == 'average':
            self.reward = self.reward_normaliser(self.reward, self.first_flag, type='average')
            self.adv, self.v_targ = self.compute_gae()

        self.prepare_rollout_buffer()
        self.model_logs[0], self.model_logs[5] = self.train_pi()
        self.model_logs[1], self.model_logs[2], self.model_logs[3], self.model_logs[4]  = self.train_vf()

        self.save_log([self.model_logs.detach().cpu().flatten().numpy()], '/model_log')

    def decay_lr(self):
        self.entropy_coef = 0  # self.entropy_coef / 100
        self.pi_lr = self.pi_lr / 10
        self.vf_lr = self.vf_lr / 10
        for param_group in self.optimizer_Actor.param_groups:
            param_group['lr'] = self.pi_lr
        for param_group in self.optimizer_Critic.param_groups:
            param_group['lr'] = self.vf_lr

    def run(self, config, env_name, total_timesteps, seed):
        MAX_INTERACTIONS = 4000 if config["debug"] == 1 else 800000
        LR_DECAY_INTERACTIONS = 2000 if config["debug"] == 1 else 600000
        experiment_done, job_status, last_lr_update = False, 1, 0
        stop_criteria_len, stop_criteria_threshold = 10, 5
        ri_arr = np.ones(stop_criteria_len, dtype=np.float32) * 1000

        # setting up the testing arguments
        worker_agents = [Worker(config, 'training', env_name, i+5, i, self.device) for i in range(self.n_training_workers)]
        testing_agents = [Worker(config, 'training', env_name, i+5000, i+5000, self.device) for i in range(self.n_testing_workers)]

        # ppo learning
        for rollout in range(0, total_timesteps):  # steps * n_workers * epochs
            t1 = time.time()
            for i in range(self.n_training_workers):
                data = worker_agents[i].rollout(self.policy)
                self.old_states[i] = data['obs']
                self.old_actions[i] = data['act']
                self.old_logprobs[i] = data['logp']
                self.v_pred[i] = data['v_pred']
                self.reward[i] = data['reward']
                self.first_flag[i] = data['first_flag']
                self.cgm_target[i] = data['cgm_target']

            t2 = time.time()

            t3 = time.time()
            self.update(rollout)
            # self.policy.save(rew=self.reward[i].mean().item())
            # # save best model
            # if self.reward[i].mean().item() > self.best_rew:
            #     self.best_rew = self.reward[i].mean().item()
            #     self.policy.save_best()
            t4 = time.time()

            t5 = time.time()
            ri = 0

            # testing
            # if self.completed_interactions > 200000:
            #     self.policy.is_testing_worker = True
            mean_rew = 0
            best_rew = 0
            for i in range(self.n_testing_workers):
                res, rew = testing_agents[i].rollout_test(self.policy)
                mean_rew += rew
                print(f"epoch: {rollout}, worker {i}, reward: {rew}")
                ri += res[0]
            mean_rew = mean_rew / self.n_testing_workers
            print(f"epoch: {rollout}, mean reward: {mean_rew}")
            # save policy
            self.policy.save(rew=mean_rew)
            if mean_rew > best_rew:
                best_rew = mean_rew
                self.policy.save_best()

            ri_arr[rollout % stop_criteria_len] = ri / self.n_testing_workers  # mean ri of that rollout.
            t6 = time.time()
            # self.policy.is_testing_worker = False
            gc.collect()

            # decay lr
            self.completed_interactions += (self.n_step * self.n_training_workers)
            if (self.completed_interactions - last_lr_update) > LR_DECAY_INTERACTIONS:
                self.decay_lr()
                last_lr_update = self.completed_interactions

            if self.completed_interactions > MAX_INTERACTIONS:
                experiment_done = True
                job_status = 2

            # logging and termination
            if self.config["verbose"]:
                print('\nExperiment: {}, Rollout {}: Time for rollout: {}, update: {}, '
                      'testing: {}'.format(self.config["log_dir"], rollout, (t2 - t1), (t4 - t3), (t6 - t5)))
            self.save_log([[job_status, rollout, (t2 - t1), (t4 - t3), (t6 - t5)]], '/experiment_summary')

            if experiment_done:
                print('################## starting the validation trials #######################')
                n_val_trials = 3 if config["debug"] == 1 else 500
                validation_agents = [Worker(config, 'testing', env_name, i + 6000, i + 6000, self.device) for i in range(n_val_trials)]
                for i in range(n_val_trials):
                    res, _ = validation_agents[i].rollout_test(self.policy)
                print('Algo RAN Successfully')
                exit()

    def evaluate(self, config, patients, env_ids):
        # setting up the testing arguments
        testing_config = deepcopy(config)
        testing_config["meal_amount"] = [45, 30, 85, 30, 80, 30]

        # testing_config["meal_variance = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]
        # testing_config["time_variance = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]
        # testing_config["meal_prob = [1, -1, 1, -1, 1, -1]

        testing_config["meal_variance"] = [5, 3, 5, 3, 10, 3]
        testing_config["time_variance"] = [60, 30, 60, 30, 60, 30]
        testing_config["meal_prob"] = [0.95, -1, 0.95, -1, 0.95, -1]

        print('################## starting the validation trials #######################')
        n_val_trials = 3 if config["debug"] == 1 else 500
        validation_agents = [Worker(testing_config, 'testing', patients, env_ids, i + 6000, i + 6000, self.device) for i
                             in range(n_val_trials)]
        for i in range(n_val_trials):
            res, _ = validation_agents[i].rollout_test(self.policy)
        print('Algo RAN Successfully')
        exit()

