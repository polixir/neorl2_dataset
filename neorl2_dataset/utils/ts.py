import time
import datetime
import os
import pprint
import shutil

import neorl2
import gymnasium as gym
import numpy as np
import torch

from typing import Any, Callable, Dict, Optional, Union, Tuple
from tianshou.env import DummyVectorEnv, ShmemVectorEnv
from tianshou.utils import BaseLogger, LazyLogger
from tianshou.trainer.utils import  test_episode
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer, AsyncCollector
from tianshou.policy import BasePolicy,SACPolicy,DiscreteSACPolicy
from tianshou.trainer.base import BaseTrainer as _BaseTrainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net


class BaseTrainer(_BaseTrainer):
    def reset(self) -> None:
        """Initialize or reset the instance to yield a new iterator from zero."""
        self.is_run = False
        self.env_step = 0
        if self.resume_from_log:
            self.start_epoch, self.env_step, self.gradient_step = \
                self.logger.restore_data()

        self.last_rew, self.last_len = 0.0, 0
        self.start_time = time.time()
        if self.train_collector is not None:
            self.train_collector.reset_stat()

            if self.train_collector.policy != self.policy:
                self.test_in_train = False
            elif self.test_collector is None:
                self.test_in_train = False

        if self.test_collector is not None:
            assert self.episode_per_test is not None
            assert not isinstance(self.test_collector, AsyncCollector)  # Issue 700
            self.test_collector.reset_stat()
            test_result = test_episode(
                self.policy, self.test_collector, self.test_fn, self.start_epoch,
                self.episode_per_test, self.logger, self.env_step, self.reward_metric
            )
            self.best_epoch = self.start_epoch
            self.best_reward, self.best_reward_std = \
                test_result["rew"], test_result["rew_std"]

        self.epoch = self.start_epoch
        self.stop_fn_flag = False
        self.iter_num = 0
    
    def test_step(self) -> Tuple[Dict[str, Any], bool]:
        """Perform one testing step."""
        assert self.episode_per_test is not None
        assert self.test_collector is not None
        stop_fn_flag = False
        test_result = test_episode(
            self.policy, self.test_collector, self.test_fn, self.epoch,
            self.episode_per_test, self.logger, self.env_step, self.reward_metric
        )
        rew, rew_std = test_result["rew"], test_result["rew_std"]
        if self.best_epoch < 0 or self.best_reward < rew:
            self.best_epoch = self.epoch
            self.best_reward = float(rew)
            self.best_reward_std = rew_std
            if self.save_best_fn:
                self.save_best_fn(self.policy, rew)
        if self.save_best_fn:
            self.save_best_fn(self.policy, rew)
                
        if self.verbose:
            print(
                f"Epoch #{self.epoch}: test_reward: {rew:.6f} ± {rew_std:.6f},"
                f" best_reward: {self.best_reward:.6f} ± "
                f"{self.best_reward_std:.6f} in #{self.best_epoch}",
                flush=True
            )
        if not self.is_run:
            test_stat = {
                "test_reward": rew,
                "test_reward_std": rew_std,
                "best_reward": self.best_reward,
                "best_reward_std": self.best_reward_std,
                "best_epoch": self.best_epoch
            }
        else:
            test_stat = {}
        if self.stop_fn and self.stop_fn(self.best_reward):
            stop_fn_flag = True

        return test_stat, stop_fn_flag


class OffpolicyTrainer(BaseTrainer):
    """Create an iterator wrapper for off-policy training procedure.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param Collector train_collector: the collector used for training.
    :param Collector test_collector: the collector used for testing. If it's None,
        then no testing will be performed.
    :param int max_epoch: the maximum number of epochs for training. The training
        process might be finished before reaching ``max_epoch`` if ``stop_fn`` is
        set.
    :param int step_per_epoch: the number of transitions collected per epoch.
    :param int step_per_collect: the number of transitions the collector would
        collect before the network update, i.e., trainer will collect
        "step_per_collect" transitions and do some policy network update repeatedly
        in each epoch.
    :param episode_per_test: the number of episodes for one policy evaluation.
    :param int batch_size: the batch size of sample data, which is going to feed in
        the policy network.
    :param int/float update_per_step: the number of times the policy network would
        be updated per transition after (step_per_collect) transitions are
        collected, e.g., if update_per_step set to 0.3, and step_per_collect is 256
        , policy will be updated round(256 * 0.3 = 76.8) = 77 times after 256
        transitions are collected by the collector. Default to 1.
    :param function train_fn: a hook called at the beginning of training in each
        epoch. It can be used to perform custom additional operations, with the
        signature ``f(num_epoch: int, step_idx: int) -> None``.
    :param function test_fn: a hook called at the beginning of testing in each
        epoch. It can be used to perform custom additional operations, with the
        signature ``f(num_epoch: int, step_idx: int) -> None``.
    :param function save_best_fn: a hook called when the undiscounted average mean
        reward in evaluation phase gets better, with the signature
        ``f(policy: BasePolicy) ->  None``. It was ``save_fn`` previously.
    :param function save_checkpoint_fn: a function to save training process and
        return the saved checkpoint path, with the signature ``f(epoch: int,
        env_step: int, gradient_step: int) -> str``; you can save whatever you want.
    :param bool resume_from_log: resume env_step/gradient_step and other metadata
        from existing tensorboard log. Default to False.
    :param function stop_fn: a function with signature ``f(mean_rewards: float) ->
        bool``, receives the average undiscounted returns of the testing result,
        returns a boolean which indicates whether reaching the goal.
    :param function reward_metric: a function with signature
        ``f(rewards: np.ndarray with shape (num_episode, agent_num)) ->
        np.ndarray with shape (num_episode,)``, used in multi-agent RL. We need to
        return a single scalar for each episode's result to monitor training in the
        multi-agent RL setting. This function specifies what is the desired metric,
        e.g., the reward of agent 1 or the average reward over all agents.
    :param BaseLogger logger: A logger that logs statistics during
        training/testing/updating. Default to a logger that doesn't log anything.
    :param bool verbose: whether to print the information. Default to True.
    :param bool show_progress: whether to display a progress bar when training.
        Default to True.
    :param bool test_in_train: whether to test in the training phase.
        Default to True.
    """

    __doc__ = BaseTrainer.gen_doc("offpolicy") + "\n".join(__doc__.split("\n")[1:])

    def __init__(
        self,
        policy: BasePolicy,
        train_collector: Collector,
        test_collector: Optional[Collector],
        max_epoch: int,
        step_per_epoch: int,
        step_per_collect: int,
        episode_per_test: int,
        batch_size: int,
        update_per_step: Union[int, float] = 1,
        train_fn: Optional[Callable[[int, int], None]] = None,
        test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        save_best_fn: Optional[Callable[[BasePolicy], None]] = None,
        save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
        resume_from_log: bool = False,
        reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        logger: BaseLogger = LazyLogger(),
        verbose: bool = True,
        show_progress: bool = True,
        test_in_train: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            learning_type="offpolicy",
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=max_epoch,
            step_per_epoch=step_per_epoch,
            step_per_collect=step_per_collect,
            episode_per_test=episode_per_test,
            batch_size=batch_size,
            update_per_step=update_per_step,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            resume_from_log=resume_from_log,
            reward_metric=reward_metric,
            logger=logger,
            verbose=verbose,
            show_progress=show_progress,
            test_in_train=test_in_train,
            **kwargs,
        )

    def policy_update_fn(self, data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Perform off-policy updates."""
        assert self.train_collector is not None
        for _ in range(round(self.update_per_step * result["n/st"])):
            self.gradient_step += 1
            losses = self.policy.update(self.batch_size, self.train_collector.buffer)
            self.log_update_data(data, losses)


def offpolicy_trainer(*args, **kwargs) -> Dict[str, Union[float, str]]:  # type: ignore
    """Wrapper for OffPolicyTrainer run method.

    It is identical to ``OffpolicyTrainer(...).run()``.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    return OffpolicyTrainer(*args, **kwargs).run()


def get_args(**kwargs):
    args = {
        'seed': 42,
        'buffer_size': 1000000,
        'hidden_sizes': [256, 256],
        'actor_lr': 0.001,
        'critic_lr': 0.001,
        'gamma': 0.99,
        'tau': 0.005,
        'alpha': 0.2,
        'auto_alpha': False,
        'alpha_lr': 0.0003,
        'start_timesteps': 10000,
        'epoch': 100,
        'step_per_epoch': 5000,
        'step_per_collect': 1,
        'update_per_step': 1,
        'n_step': 1,
        'batch_size': 256,
        'training_num': 10,
        'test_num': 100,
        'logdir': 'logs',
        'render': 0.0,
        'rew_norm': False,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'resume_path': None,
        'resume_id': None,
        'logger': 'tensorboard',
        'wandb_project': 'mujoco.benchmark',
        'watch': False
    }
    
    return args

class DotDict(dict):
    def __getattr__(self, attr):
        value = self.get(attr)
        if isinstance(value, dict):
            return DotDict(value)
        return value
    
    def __setattr__(self, attr, value):
        self[attr] = value

def train_sac(**kwargs):
    args = DotDict(get_args())
    args.update(kwargs)
    
    env = gym.make(args.task)
    train_envs = ShmemVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)]
    )
    test_envs = ShmemVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)]
    )
    
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # model
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    
    if isinstance(env.action_space, gym.spaces.Discrete):
        from tianshou.utils.net.discrete import Actor, Critic
        actor = Actor(net, 
                      args.action_shape, 
                      softmax_output=False,
                      device=args.device).to(args.device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
        net_c1 = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
        critic1 = Critic(net_c1, last_size=args.action_shape,
                        device=args.device).to(args.device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
        net_c2 = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
        critic2 = Critic(net_c2, last_size=args.action_shape,
                        device=args.device).to(args.device)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
        
        if args.auto_alpha:
            target_entropy = 0.98 * np.log(np.prod(args.action_shape))
            log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
            args.alpha = (target_entropy, log_alpha, alpha_optim)
            
        policy = DiscreteSACPolicy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            args.tau,
            args.gamma,
            args.alpha,
            estimation_step=args.n_step,
            reward_normalization=args.rew_norm
        )
        
    else:
        from tianshou.utils.net.continuous import ActorProb, Critic
        actor = ActorProb(
            net,
            args.action_shape,
            max_action=args.max_action,
            device=args.device,
            unbounded=True,
            conditioned_sigma=True,
        ).to(args.device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
        net_c1 = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            concat=True,
            device=args.device,
        )
        net_c2 = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            concat=True,
            device=args.device,
        )
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
        critic1 = Critic(net_c1, device=args.device).to(args.device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
        critic2 = Critic(net_c2, device=args.device).to(args.device)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

        if args.auto_alpha:
            target_entropy = -np.prod(env.action_space.shape)
            log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
            args.alpha = (target_entropy, log_alpha, alpha_optim)

        policy = SACPolicy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau=args.tau,
            gamma=args.gamma,
            alpha=args.alpha,
            estimation_step=args.n_step,
            action_space=env.action_space,
        )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    train_collector.collect(n_step=args.start_timesteps, random=True)

    # log
    args.algo_name = "sac"
    log_name = os.path.join(args.task, "tensorboard", args.algo_name.upper())
    tensorboard_log_path = os.path.join(args.logdir, log_name)
    if os.path.exists(tensorboard_log_path):
        shutil.rmtree(tensorboard_log_path)
    log_name = os.path.join(args.task, "models")
    model_save_path = os.path.join(args.logdir, log_name)
    if os.path.exists(model_save_path):
        shutil.rmtree(model_save_path)
    os.makedirs(model_save_path)
        

    # logger
    if args.logger == "wandb":
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
    writer = SummaryWriter(tensorboard_log_path)
    writer.add_text("args", str(args))
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)
        
    def save_fn(policy, rew):
        actor = policy.actor
        torch.save(actor, os.path.join(model_save_path, f"mean_reward_{rew}.pth"))

    if not args.watch:
        # trainer
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            save_best_fn=save_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False,
        )
        pprint.pprint(result)

    # Let's watch its performance!
    policy.eval()
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')
