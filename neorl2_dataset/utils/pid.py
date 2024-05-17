import neorl2
import numpy as np
import gymnasium as gym
from typing import Sequence, Tuple
from functools import partial
from bayes_opt import BayesianOptimization
from typing import Sequence, Tuple
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import argparse
import os

class Env_Pool():
    def __init__(self, 
                env_pool: list, 
                dim:int, 
                dt: float,
                obs_index: list,
                target_index: list):

        self.env_pool = env_pool
        self.dim = dim
        self.obs_index = obs_index
        self.target_index = target_index
        self.dt = dt

    def reset(self,):
        self.init_obs = []
        for _e in self.env_pool:
            self.init_obs.append(_e.reset()[0])
        self.init_obs = np.stack(self.init_obs)


        self.err_hist = [np.zeros((len(self.env_pool), self.dim))]

        _positions = self.init_obs[..., self.obs_index]    
        _targets =   self.init_obs[..., self.target_index]

        self.err_p = _targets-_positions
        self.err_hist.append(self.err_p) 
        self.err_i = np.sum(np.stack(self.err_hist), axis=0) *  self.dt
        self.err_d = (self.err_hist[-1] - self.err_hist[-2]) / self.dt
        _temp_pid = np.hstack([self.err_p, self.err_i, self.err_d])      
        return _temp_pid

    def step(self, action):
        next_obs = []
        reward = []
        done = []
        terminate = []
        other_infor = []

        assert len(action.shape)==2

        for i in range(action.shape[0]):
            _temp_next_obs, _temp_reward, _temp_done, _temp_terminate, _temp_other_infor = self.env_pool[i].step(action[i])
            next_obs.append(_temp_next_obs)
            reward.append(_temp_reward)

        next_obs = np.vstack(next_obs)
        reward = np.vstack(reward)

        _positions = next_obs[..., self.obs_index]    
        _targets = next_obs[..., self.target_index]
        self.err_p = _targets-_positions
        self.err_hist.append(self.err_p)

        self.err_i = np.sum(np.stack(self.err_hist), axis=0) *  self.dt
        self.err_d = (self.err_hist[-1] - self.err_hist[-2]) / self.dt

        
        return np.hstack([self.err_p, self.err_i, self.err_d]), reward

class MultiPIDController():
    def __init__(
        self,
        p_gains: Sequence[float],
        i_gains: Sequence[float],
        d_gains: Sequence[float],
        baselines: Sequence[float] = (0.0, 0.0),
    ):
        """Constructor."""
        self.dims = len(p_gains)
        assert len(i_gains) == self.dims
        assert len(d_gains) == self.dims
        assert len(baselines) == self.dims
        self.gains = np.array([
            np.array(p_gains),
            np.array(i_gains),
            np.array(d_gains),
        ])
        self.baselines = np.array(baselines)

    def get_actions(self, observations: np.ndarray) -> Tuple[np.ndarray]:

        if len(observations.shape) == 1:
            observations = observations[np.newaxis]
        assert observations.shape[1] == 3 * self.dims
        observations = observations.reshape(observations.shape[0], 3, self.dims)
        acts = np.sum(self.gains * observations, axis=1) + self.baselines
        # print((self.gains * observations).shape)
        return acts, np.ones(len(observations))


def twod_pid_opt_function(
    p1: float,
    p2: float,
    i1: float,
    i2: float,
    d1: float,
    d2: float,
    env: Env_Pool,
    horizon: int,
):
    controller = MultiPIDController([p1, p2],
                                    [i1, i2],
                                    [d1, d2],
                                    [0.0, 0.0])

    rew_list = []
    observations = env.reset()  
    for h in range(horizon):
        actions,_ = controller.get_actions(observations)
        observations, rew = env.step(actions)
        rew_list.append(rew)    

    return np.stack(rew_list).sum(axis=0).mean()

def threed_pid_opt_function(
    p1: float,
    p2: float,
    p3: float,
    i1: float,
    i2: float,
    i3: float,
    d1: float,
    d2: float,
    d3: float,
    env: Env_Pool,
    horizon: int,
):
    controller = MultiPIDController([p1, p2, p3],
                                    [i1, i2, i3],
                                    [d1, d2, d3],
                                    [0.0, 0.0, 0.0])

    rew_list = []
    observations = env.reset()     
    for h in range(horizon):
        actions,_ = controller.get_actions(observations)
        observations, rew = env.step(actions)
        rew_list.append(rew)    

    return np.stack(rew_list).sum(axis=0).mean()


def oned_pid_opt_function(
    p: float,
    i: float,
    d: float,
    env: Env_Pool,
    horizon: int,
    ):
    controller = MultiPIDController([p],[i],[d],[0])

    rew_list = []
    observations = env.reset()     
    for h in range(horizon):
        actions,_ = controller.get_actions(observations)
        observations, rew = env.step(actions)
        rew_list.append(rew)    

    return np.stack(rew_list).sum(axis=0).mean()


def pid_train(env_name, log_path, horizon=100, rollouts=50, dim=2, dt=0.2, obs_index = [0,1], target_index=[4,5], capital=1000):
    env_pool = Env_Pool([gym.make(f'{env_name}') for _ in range(rollouts)], dim, dt, obs_index, target_index)
    if 'DMSD' in env_name:
        base_obj = twod_pid_opt_function
        pbounds = {'p1':[0,5],'p2':[0,5],'i1':[0, 1],'i2':[0, 1],'d1':[0,5],'d2':[0,5]}
    else:
        base_obj = oned_pid_opt_function
        pbounds = None

    folder_path = os.path.dirname(log_path)
    if not os.path.exists(folder_path):
        # 如果不存在，则创建文件夹
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 创建成功.")
    else:
        print(f"文件夹 '{folder_path}' 已经存在.")

    obj_function = partial(
            base_obj,
            env=env_pool,
            horizon=horizon
        )
    eval_dict = {'target':[],
                'params':[]}
    for seed in range(1):
        optimizer = BayesianOptimization(
            f=obj_function,
            pbounds=pbounds,
            random_state=seed,
            allow_duplicate_points=True,
        )
        logger = JSONLogger(path=log_path)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.maximize(
            init_points=30,
            n_iter=capital,
        )
        print(optimizer.max)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=f'PID policy BayesianOptimization')
    parser.add_argument('--env', type=str, default='', help="PID task")
    args = parser.parse_known_args()[0].__dict__

    env_name = args['env'] #'Fusion-v0' #
    if 'DMSD' in env_name:
        HORIZON = 100
        ROLLOUTS = 50
        IDNAME = env_name

        env_pool = Env_Pool([gym.make(f'{IDNAME}') for _ in range(ROLLOUTS)], 2, 0.2, [0,1], [4,5])
        base_obj = twod_pid_opt_function
        obj_function = partial(
                base_obj,
                env=env_pool,
                horizon=HORIZON
            )
        # pbounds = {'p1':[1,4],'p2':[1,4],'i1':[0, 0.5],'i2':[0, 0.5],'d1':[1,4],'d2':[1,4]}
        pbounds = {'p1':[0,5],'p2':[0,5],'i1':[0, 1],'i2':[0, 1],'d1':[0,5],'d2':[0,5]}
        capital = 1000
        eval_dict = {'target':[],
                    'params':[]}
        for seed in range(1):
            optimizer = BayesianOptimization(
                f=obj_function,
                pbounds=pbounds,
                random_state=seed,
                allow_duplicate_points=True,
            )
            logger = JSONLogger(path=f"./data/pid/{IDNAME}_tune_log.json")
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
            optimizer.maximize(
                init_points=30,
                n_iter=capital,
            )
            print(optimizer.max)
        
    elif env_name == 'Fusion-v0':
        HORIZON = 100
        ROLLOUTS = 50
        IDNAME = 'Fusion-v0'

        env_pool = Env_Pool([gym.make(f'{IDNAME}') for _ in range(ROLLOUTS)], 1, 0.025, [1], [2])
        base_obj = oned_pid_opt_function
        obj_function = partial(
                base_obj,
                env=env_pool,
                horizon=HORIZON
            )
        # pbounds = {'p':[0,2],'i':[0, 0.25],'d':[0,0.5]}
        pbounds = {'p':[0,5],'i':[0, 0.5],'d':[0,1]}
        capital = 1000
        eval_dict = {'target':[],
                    'params':[]}
        for seed in range(1):
            optimizer = BayesianOptimization(
                f=obj_function,
                pbounds=pbounds,
                random_state=seed,
                allow_duplicate_points=True,
            )
            logger = JSONLogger(path=f"./data/pid/{IDNAME}_tune_log.json")
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
            optimizer.maximize(
                init_points=30,
                n_iter=capital,
            )
            print(optimizer.max)

    elif env_name == 'FusionEnv-v0':
        HORIZON = 100
        ROLLOUTS = 2
        IDNAME = env_name

        env_pool = Env_Pool([gym.make(f'{IDNAME}',mode='pid') for _ in range(ROLLOUTS)], 3, 4, [6,7,8], [9,10,11])
        base_obj = threed_pid_opt_function
        obj_function = partial(
                base_obj,
                env=env_pool,
                horizon=HORIZON
            )
        # pbounds = {'p':[0,2],'i':[0, 0.25],'d':[0,0.5]}
        pbounds =  {'p1':[0,5],'p2':[0,5],'p3':[0,5],
                    'i1':[0, 1],'i2':[0, 1],'i3':[0, 1],
                    'd1':[0,5],'d2':[0,5],'d3':[0,5]}

        capital = 1000
        eval_dict = {'target':[],
                    'params':[]}
        for seed in range(1):
            optimizer = BayesianOptimization(
                f=obj_function,
                pbounds=pbounds,
                random_state=seed,
                allow_duplicate_points=True,
            )
            # logger = JSONLogger(path=f"./data/pid/{IDNAME}_tune_log.json")
            # optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
            optimizer.maximize(
                init_points=30,
                n_iter=capital,
            )
            print(optimizer.max)

