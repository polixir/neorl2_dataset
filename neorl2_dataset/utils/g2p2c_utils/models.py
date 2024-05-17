import os
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import neorl2.utils.g2p2c_utils.core as core


class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super(FeatureExtractor, self).__init__()
        self.n_features = config["n_features"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_rnn_layers"]
        self.bidirectional = config["bidirectional"]
        self.directions = config["rnn_directions"]
        self.LSTM = nn.LSTM(input_size=self.n_features, hidden_size=self.n_hidden, num_layers=self.n_layers,
                            batch_first=True, bidirectional=self.bidirectional)  # (seq_len, batch, input_size)

    def forward(self, s, mode):
        if mode == 'batch':
            output, (hid, cell) = self.LSTM(s)
            lstm_output = hid.view(hid.size(1), -1)  # => batch , layers * hid
        else:
            s = s.unsqueeze(0)  # add batch dimension
            output, (hid, cell) = self.LSTM(s)  # hid = layers * dir, batch, hidden
            lstm_output = hid.squeeze(1)  # remove batch dimension
            lstm_output = torch.flatten(lstm_output)  # hid = layers * hidden_size

        extract_states = lstm_output
        return extract_states, lstm_output


class GlucoseModel(nn.Module):
    def __init__(self, config, device):
        super(GlucoseModel, self).__init__()
        self.n_features = config["n_features"]
        self.device = device
        self.output = config["n_action"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_rnn_layers"]
        self.bidirectional = config["bidirectional"]
        self.directions = config["rnn_directions"]
        self.feature_extractor = self.n_hidden * self.n_layers * self.directions
        self.last_hidden = self.feature_extractor #* 2
        self.fc_layer1 = nn.Linear(self.feature_extractor + self.output, self.last_hidden)
        # self.fc_layer2 = nn.Linear(self.last_hidden, self.last_hidden)
        # self.fc_layer3 = nn.Linear(self.last_hidden, self.last_hidden)
        self.cgm_mu = NormedLinear(self.last_hidden, self.output, scale=0.1)
        self.cgm_sigma = NormedLinear(self.last_hidden, self.output, scale=0.1)
        self.normal = torch.distributions.Normal(0, 1)

    def forward(self, extract_state, action, mode):
        concat_dim = 1 if (mode == 'batch') else 0
        concat_state_action = torch.cat((extract_state, action), dim=concat_dim)
        fc_output1 = F.relu(self.fc_layer1(concat_state_action))
        fc_output = fc_output1
        # fc_output2 = F.relu(self.fc_layer2(fc_output1))
        # fc_output = F.relu(self.fc_layer3(fc_output2))
        cgm_mu = F.tanh(self.cgm_mu(fc_output))
        # deterministic
        # cgm_sigma = torch.zeros(1, device=self.device, dtype=torch.float32)
        # cgm = cgm_mu
        # probabilistic
        cgm_sigma = F.softplus(self.cgm_sigma(fc_output) + 1e-5)
        z = self.normal.sample()
        cgm = cgm_mu + cgm_sigma * z
        cgm = torch.clamp(cgm, -1, 1)
        return cgm_mu, cgm_sigma, cgm


class ActionModule(nn.Module):
    def __init__(self, config, device):
        super(ActionModule, self).__init__()
        self.device = device
        self.config = config
        self.output = config["n_action"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_rnn_layers"]
        self.directions = config["rnn_directions"]
        self.feature_extractor = self.n_hidden * self.n_layers * self.directions
        self.last_hidden = self.feature_extractor * 2
        self.fc_layer1 = nn.Linear(self.feature_extractor, self.last_hidden)
        self.fc_layer2 = nn.Linear(self.last_hidden, self.last_hidden)
        self.fc_layer3 = nn.Linear(self.last_hidden, self.last_hidden)
        self.mu = NormedLinear(self.last_hidden, self.output, scale=0.1)
        self.sigma = NormedLinear(self.last_hidden, self.output, scale=0.1)
        self.normalDistribution = torch.distributions.Normal

    def forward(self, extract_states, action_type='N'):
        fc_output1 = F.relu(self.fc_layer1(extract_states))
        fc_output2 = F.relu(self.fc_layer2(fc_output1))
        fc_output = F.relu(self.fc_layer3(fc_output2))
        mu = F.tanh(self.mu(fc_output))
        sigma = F.sigmoid(self.sigma(fc_output) + 1e-5)
        z = self.normalDistribution(0, 1).sample()
        action = mu + sigma * z
        action = torch.clamp(action, -1, 1)
        try:
            dst = self.normalDistribution(mu, sigma)
            log_prob = dst.log_prob(action[0])
        except ValueError:
            print('\nCurrent mu: {}, sigma: {}'.format(mu, sigma))
            print('shape: {}. {}'.format(mu.shape, sigma.shape))
            print(extract_states.shape)
            print(extract_states)
            log_prob = torch.ones(2, 1, device=self.device, dtype=torch.float32) * self.glucose_target
        # log_prob = dst.log_prob(action[0])
        # sigma = torch.ones(1, device=self.device, dtype=torch.float32) * 0.01
        # clamp the sigma
        # sigma = F.softplus(self.sigma(fc_output) + 1e-5)
        # sigma = torch.clamp(sigma, 1e-5, 0.33)
        return mu, sigma, action, log_prob


class ValueModule(nn.Module):
    def __init__(self, config, device):
        super(ValueModule, self).__init__()
        self.device = device
        self.output = config["n_action"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_rnn_layers"]
        self.directions = config["rnn_directions"]
        self.feature_extractor = self.n_hidden * self.n_layers * self.directions
        self.last_hidden = self.feature_extractor * 2
        self.fc_layer1 = nn.Linear(self.feature_extractor, self.last_hidden)
        self.fc_layer2 = nn.Linear(self.last_hidden, self.last_hidden)
        self.fc_layer3 = nn.Linear(self.last_hidden, self.last_hidden)
        self.value = NormedLinear(self.last_hidden, self.output, scale=0.1)

    def forward(self, extract_states):
        fc_output1 = F.relu(self.fc_layer1(extract_states))
        fc_output2 = F.relu(self.fc_layer2(fc_output1))
        fc_output = F.relu(self.fc_layer3(fc_output2))
        value = (self.value(fc_output))
        return value


class ActorNetwork(nn.Module):
    def __init__(self, config, device):
        super(ActorNetwork, self).__init__()
        self.device = device
        self.config = config
        self.FeatureExtractor = FeatureExtractor(config)
        self.GlucoseModel = GlucoseModel(config, self.device)
        self.ActionModule = ActionModule(config, self.device)
        self.distribution = torch.distributions.Normal
        self.glucose_target = core.linear_scaling(x=112.5, x_min=self.config["glucose_min"], x_max=self.config["glucose_max"])
        self.t_to_meal = core.linear_scaling(x=0, x_min=0, x_max=self.config["t_meal"])

    def forward(self, s, old_action, mode):
        extract_states, lstmOut = self.FeatureExtractor.forward(s, mode)
        mu, sigma, action, log_prob = self.ActionModule.forward(extract_states)
        if mode == 'forward':
            cgm_mu, cgm_sigma, cgm = self.GlucoseModel.forward(lstmOut, action.detach(), mode)
        else:
            cgm_mu, cgm_sigma, cgm = self.GlucoseModel.forward(lstmOut, old_action.detach(), mode)
        return mu, sigma, action, log_prob, cgm_mu, cgm_sigma, cgm

    def update_state(self, s, cgm_pred, action, batch_size):
        if batch_size == 1:
            if self.config["n_features"] == 2:
                s_new = torch.cat((cgm_pred, action), dim=0)
            if self.config["n_features"] == 3:
                s_new = torch.cat((cgm_pred, action, self.t_to_meal * torch.ones(1, device=self.device)), dim=0)
            s_new = s_new.unsqueeze(0)
            s = torch.cat((s[1:self.config["feature_history"], :], s_new), dim=0)
        else:
            if self.config["n_features"] == 3:
                s_new = torch.cat((cgm_pred, action, self.t_to_meal * torch.ones(batch_size, 1, device=self.device)), dim=1)
            if self.config["n_features"] == 2:
                s_new = torch.cat((cgm_pred, action), dim=1)
            s_new = s_new.unsqueeze(1)
            s = torch.cat((s[:, 1:self.config["feature_history"], :], s_new), dim=1)
        return s


class CriticNetwork(nn.Module):
    def __init__(self, config, device):
        super(CriticNetwork, self).__init__()
        self.FeatureExtractor = FeatureExtractor(config)
        self.ValueModule = ValueModule(config, device)
        self.GlucoseModel = GlucoseModel(config, device)
    def forward(self, s, action, cgm_pred=True, mode='forward'):
        extract_states, lstmOut = self.FeatureExtractor.forward(s, mode)
        value = self.ValueModule.forward(extract_states)
        cgm_mu, cgm_sigma, cgm = self.GlucoseModel.forward(lstmOut, action.detach(), mode) if cgm_pred else (None, None, None)
        return value, cgm_mu, cgm_sigma, cgm


class ActorCritic(nn.Module):
    def __init__(self, config, load, actor_path, critic_path, device):
        super(ActorCritic, self).__init__()
        self.device = device
        self.experiment_dir = config["log_dir"]
        self.Actor = ActorNetwork(config, device)
        self.Critic = CriticNetwork(config, device)
        if load:
            self.Actor = torch.load(actor_path, map_location=device)
            self.Critic = torch.load(critic_path, map_location=device)
        self.distribution = torch.distributions.Normal

    def predict(self, s):
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)

        mean, std, action, log_prob, a_cgm_mu, a_cgm_sigma, a_cgm = self.Actor(s, None, mode='forward')
        state_value, c_cgm_mu,  c_cgm_sigma, c_cgm = self.Critic(s, action, cgm_pred=True, mode='forward' )
        return (mean, std, action, log_prob, a_cgm_mu, a_cgm_sigma, a_cgm), (state_value, c_cgm_mu,  c_cgm_sigma, c_cgm)

    def get_action(self, s):
        (mu, std, act, log_prob, a_cgm_mu, a_cgm_sig, a_cgm), (s_val, c_cgm_mu, c_cgm_sig, c_cgm) = self.predict(s)
        data = dict(mu=mu, std=std, action=act, log_prob=log_prob, state_value=s_val, a_cgm_mu=a_cgm_mu,
                    a_cgm_sigma=a_cgm_sig, c_cgm_mu=c_cgm_mu, c_cgm_sigma=c_cgm_sig, a_cgm=a_cgm, c_cgm=c_cgm)
        return {k: v.detach().cpu().numpy() for k, v in data.items()}

    def get_final_value(self, s):
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        state_value, _,  _, _ = self.Critic(s, action=None, cgm_pred=False, mode='forward' )
        return state_value.detach().cpu().numpy()

    def evaluate_actor(self, state, action):  # evaluate batch
        action_mean, action_std, _, _, a_cgm_mu, a_cgm_sigma, _ = self.Actor(state, action, mode='batch')
        dist = self.distribution(action_mean, action_std)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy, a_cgm_mu, a_cgm_sigma

    def evaluate_critic(self, state, action, cgm_pred):  # evaluate batch
        state_value, c_cgm_mu,  c_cgm_sigma, _ = self.Critic(state, action, cgm_pred, mode='batch')
        return torch.squeeze(state_value), c_cgm_mu,  c_cgm_sigma

    # def save(self, rew):
    #     share_path = self.experiment_dir + '/models/mean_reward_'
    #     mk_share_path = '/'.join(share_path.split('/')[:-1])
    #     Path(os.path.join(os.getcwd(), mk_share_path)).mkdir(parents=True, exist_ok=True)
    #     actor_path = self.experiment_dir + '/models/mean_reward_' + str(rew) + '_Actor.pth'
    #     critic_path = self.experiment_dir + '/models/mean_reward_' + str(rew) + '_Critic.pth'
    #     torch.save(self.Actor, actor_path)
    #     torch.save(self.Critic, critic_path)

    def save(self, rew):
        share_path = self.experiment_dir + '/models/mean_reward_'
        mk_share_path = '/'.join(share_path.split('/')[:-1])
        Path(os.path.join(os.getcwd(), mk_share_path)).mkdir(parents=True, exist_ok=True)
        model_path = self.experiment_dir + '/models/mean_reward_' + str(rew) + '.pth'
        torch.save(self, model_path)

    def save_best(self, ):
        share_path = self.experiment_dir + '/models/best_model_'
        mk_share_path = '/'.join(share_path.split('/')[:-1])
        Path(os.path.join(os.getcwd(), mk_share_path)).mkdir(parents=True, exist_ok=True)
        model_path = self.experiment_dir + '/models/best_model.pth'
        torch.save(self, model_path)

            
def NormedLinear(*args, scale=1.0):
    out = nn.Linear(*args)
    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    return out
