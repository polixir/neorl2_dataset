import os
from pathlib import Path
import json
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union
from neorl2.utils.g2p2c_utils.ppo import PPO


class G2P2C:
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    def _initial(self):
        self.parser.add_argument('--agent', type=str, default='ppo', help='agent used for the experiment.')
        self.parser.add_argument('--restart', type=str, default='1', help='')
        self.parser.add_argument('--m', type=str, default='', help='message about the experiment')
        self.parser.add_argument('--device', type=str, default='cpu', help='give device name')
        self.parser.add_argument('--verbose', type=bool, default=True, help='')
        self.parser.add_argument('--seed', type=int, default=0, help='')
        self.parser.add_argument('--debug', type=int, default=0, help='if debug ON => 1')
        self.parser.add_argument('--kl', type=int, default=1, help='if debug ON => 1')  # experimenting KL implementation

        # simulation
        self.parser.add_argument('--patient_id', type=int, default=0,
                                 help='patient_id = [adolescent child adults] hence 0 - 9 indexes adolescents likewise')
        self.parser.add_argument('--sensor', type=str, default='GuardianRT', help='Dexcom, GuardianRT, Navigator')
        self.parser.add_argument('--pump', type=str, default='Insulet', help='Insulet, Cozmo')

        # for training: # ideal benchmark adult and adolescent doesnt have snacks though => set prob '-1' to remove
        self.parser.add_argument('--meal_prob', type=list, default=[0.95, -1, 0.95, -1, 0.95, -1], help='')
        self.parser.add_argument('--meal_amount', type=list, default=[45, 30, 85, 30, 80, 30], help='')
        self.parser.add_argument('--meal_variance', type=list, default=[5, 3, 5, 3, 10, 3], help='')
        self.parser.add_argument('--time_variance', type=list, default=[60, 30, 60, 30, 60, 30], help='in mins')

        # insulin action limits
        self.parser.add_argument('--action_type', type=str, default='exponential',
                                 help='normal, quadratic, proportional_quadratic, exponential, sparse')
        self.parser.add_argument('--action_scale', type=int, default=1, help='This is the max insulin')
        self.parser.add_argument('--insulin_max', type=int, default=5, help='')
        self.parser.add_argument('--insulin_min', type=int, default=0, help='')
        self.parser.add_argument('--glucose_max', type=int, default=600, help='')  # the sensor range would affect this
        self.parser.add_argument('--glucose_min', type=int, default=39, help='')

        # algorithm training settings.
        self.parser.add_argument('--target_glucose', type=float, default=140, help='target glucose')  # param for pid
        self.parser.add_argument('--use_bolus', type=bool, default=True, help='')  # param for BB
        self.parser.add_argument('--use_cf', type=bool, default=False, help='')  # param for BB
        self.parser.add_argument('--glucose_cf_target', type=float, default=150, help='glucose correction target')  # param for BB
        self.parser.add_argument('--expert_bolus', type=bool, default=False, help='')
        self.parser.add_argument('--expert_cf', type=bool, default=False, help='')
        self.parser.add_argument('--carb_estimation_method', type=str, default='real', help='linear, quadratic, real, rand')
        self.parser.add_argument('--t_meal', type=int, default=20,  # 20 is what i use here
                                 help='if zero, assume no announcmeent; announce meal x min before, '
                                      'Optimal prandial timing of bolus insulin in diabetes management: a review,'
                                      'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5836969/')

        # Actor / Critic Network params.
        self.parser.add_argument('--n_features', type=int, default=3, help='# of features in the state space')
        self.parser.add_argument('--feature_history', type=int, default=48, help='')
        self.parser.add_argument('--calibration', type=int, default=48, help='should be same as feature_hist')
        self.parser.add_argument('--max_epi_length', type=int, default=2000, help='')  # 30days, 5 min, 8640
        self.parser.add_argument('--n_action', type=int, default=1, help='number of control actions')
        self.parser.add_argument('--n_hidden', type=int, default=12, help='hidden units in lstm')
        self.parser.add_argument('--n_rnn_layers', type=int, default=2, help='layers in the lstm')
        self.parser.add_argument('--rnn_directions', type=int, default=1, help='')
        self.parser.add_argument('--rnn_only', type=bool, default=False, help='')
        self.parser.add_argument('--bidirectional', type=bool, default=False, help='')
        self.parser.add_argument('--n_step', type=int, default=6, help='n step TD learning, consider selected sensor!')
        self.parser.add_argument('--gamma', type=float, default=0.99, help='1 if continous')
        self.parser.add_argument('--lambda_', type=float, default=0.95, help='')
        self.parser.add_argument('--max_test_epi_len', type=int, default=1, help='n time max ep trained.')

        # ppo params
        self.parser.add_argument('--eps_clip', type=float, default=0.2, help=' (Usually small, 0.1 to 0.3.) 0.2')
        self.parser.add_argument('--n_vf_epochs', type=int, default=80, help='')
        self.parser.add_argument('--n_pi_epochs', type=int, default=80, help='')
        self.parser.add_argument('--target_kl', type=float, default=0.05, help='# (Usually small, 0.01 or 0.05.)')
        self.parser.add_argument('--pi_lr', type=float, default=1e-4, help='')
        self.parser.add_argument('--vf_lr', type=float, default=1e-4, help='')
        self.parser.add_argument('--batch_size', type=int, default=64, help='')
        self.parser.add_argument('--n_training_workers', type=int, default=20, help='')
        self.parser.add_argument('--n_testing_workers', type=int, default=5, help='')
        self.parser.add_argument('--entropy_coef', type=float, default=0.01, help='')
        self.parser.add_argument('--grad_clip', type=float, default=20, help='')
        self.parser.add_argument('--normalize_reward', type=bool, default=False, help='')
        self.parser.add_argument('--shuffle_rollout', type=bool, default=False, help='')
        self.parser.add_argument('--return_type', type=str, default='discount', help='discount | average')

        # deprecated todo: refactor
        self.parser.add_argument('--bgp_pred_mode', type=bool, default=False, help='future bg prediction')
        self.parser.add_argument('--n_bgp_steps', type=int, default=0, help='future eprediction horizon')
        self.parser.add_argument('--pretrain_period', type=int, default=5760, help='')
    """

    # policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
    #     "MlpPolicy": MlpPolicy,
    #     "CnnPolicy": CnnPolicy,
    #     "MultiInputPolicy": MultiInputPolicy,
    # }
    # policy: SACPolicy
    # actor: Actor
    # critic: ContinuousCritic
    # critic_target: ContinuousCritic

    def __init__(
        self,
        env_name: str,
        restart: str = '1',
        m: str = '',
        device: str = 'cuda',
        verbose: bool = True,
        seed: int = 3,
        debug: int = 0,
        kl: int = 1,
        # directories
        log_dir: str = '',
        # simulation
        patient_id: int = 0,
        sensor: str = 'GuardianRT',
        pump: str = 'Insulet',
        # insulin action limits
        action_scale: int = 5,
        insulin_max: int = 5,
        insulin_min: int = 0,
        glucose_max: int = 600,
        glucose_min: int = 39,
        # algorithm training settings.
        target_glucose: float = 140,
        glucose_cf_target: float = 150,
        expert_bolus: bool = False,
        expert_cf: bool = False,
        t_meal: int = 20,
        # Actor / Critic Network params.
        n_features: int = 2,
        feature_history: int = 12,
        calibration: int = 12,
        max_epi_length: int = 480 * 10,
        n_action: int = 1,
        n_hidden: int = 16,
        n_rnn_layers: int = 1,
        rnn_directions: int = 1,
        rnn_only: bool = False,
        bidirectional: bool = False,
        n_step: int = 256,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        max_test_epi_len: int = 480,
        # ppo params
        eps_clip: float = 0.1,
        n_vf_epochs: int = 5,
        n_pi_epochs: int = 5,
        target_kl: float = 0.01,
        pi_lr: float = 1e-4 * 3,
        vf_lr: float = 1e-4 * 3,
        batch_size: int = 1024,
        n_training_workers: int = 16,
        n_testing_workers: int = 100,
        entropy_coef: float = 0.001,
        grad_clip: float = 20,
        normalize_reward: bool = True,
        shuffle_rollout: bool = True,
        return_type: str = 'average',
        **kwargs,
    ):
        self.config = dict()
        self.config['restart'] = restart
        self.config['m'] = m
        self.config['device'] = device
        self.config['verbose'] = verbose
        self.config['seed'] = seed
        self.config['debug'] = debug
        self.config['kl'] = kl
        self.config['patient_id'] = patient_id
        self.config['sensor'] = sensor
        self.config['pump'] = pump
        self.config['action_scale'] = action_scale
        self.config['insulin_max'] = insulin_max
        self.config['insulin_min'] = insulin_min
        self.config['glucose_max'] = glucose_max
        self.config['glucose_min'] = glucose_min
        self.config['target_glucose'] = target_glucose
        self.config['glucose_cf_target'] = glucose_cf_target
        self.config['expert_bolus'] = expert_bolus
        self.config['expert_cf'] = expert_cf
        self.config['t_meal'] = t_meal
        self.config['n_features'] = n_features
        self.config['feature_history'] = feature_history
        self.config['calibration'] = calibration
        self.config['max_epi_length'] = max_epi_length
        self.config['n_action'] = n_action
        self.config['n_hidden'] = n_hidden
        self.config['n_rnn_layers'] = n_rnn_layers
        self.config['rnn_directions'] = rnn_directions
        self.config['rnn_only'] = rnn_only
        self.config['bidirectional'] = bidirectional
        self.config['n_step'] = n_step
        self.config['gamma'] = gamma
        self.config['lambda_'] = lambda_
        self.config['max_test_epi_len'] = max_test_epi_len
        self.config['eps_clip'] = eps_clip
        self.config['n_vf_epochs'] = n_vf_epochs
        self.config['n_pi_epochs'] = n_pi_epochs
        self.config['target_kl'] = target_kl
        self.config['pi_lr'] = pi_lr
        self.config['vf_lr'] = vf_lr
        self.config['batch_size'] = batch_size
        self.config['n_training_workers'] = n_training_workers
        self.config['n_testing_workers'] = n_testing_workers
        self.config['entropy_coef'] = entropy_coef
        self.config['grad_clip'] = grad_clip
        self.config['normalize_reward'] = normalize_reward
        self.config['shuffle_rollout'] = shuffle_rollout
        self.config['return_type'] = return_type
        self.config.update(kwargs)

        self.env_name = env_name
        self.config["env_name"] = env_name
        self.config['log_dir'] = log_dir

        # save self.config as json
        Path(os.path.join(os.getcwd(), self.config['log_dir'])).mkdir(parents=True, exist_ok=True)
        json_path = os.path.join(self.config['log_dir'], 'config.json')
        with open(json_path, "w") as outfile:
            json.dump(self.config, outfile)

        self.agent = PPO(self.config, device, False, '', '')

    def learn(self, total_timesteps):
        self.agent.run(self.config, self.env_name, total_timesteps, self.config['seed'])
