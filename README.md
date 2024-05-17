# NeoRL2 Dataset

**NeoRL2 Dataset** is the repository to train policies and generate datasets for [NeoRL2](https://github.com/polixir/NeoRL2) benchmarks. 

## Install 

#### 1. Install neorl2

Please install `neorl2` for getting environments:

```
pip isntall neorl2
```

#### 2. Install neorl2_dataset

```
git clone https://jg.gitlab.polixir.site/polixir/neorl2_dataset.git
cd neorl2_dataset
pip install -e .
```

After installation neorl2, Pipeline、Simglucose、RocketRecover、DMSD and Fusion environments will be available. However, the "RandomFrictionHopper" and "SafetyHalfCheetah" tasks rely on MuJoCo. If you need to use these two environments, it is necessary to obtain a [license](https://www.roboti.us/license.html) and follow the setup instructions, and then run:

```
pip install -e .[mujoco]
```



## Envs

You can use `neorl2` to get all standardized environments, like:

```
import neorl2
import gymnasium as gym

# Create an environment
env = gym.make("Pipeline")
env.reset()
env.step(env.action_space.sample())
```

You can use the following environments now:


| Env Name             | observation shape | action shape | have done | max timesteps |
| -------------------- | ----------------- | ------------ | --------- | ------------- |
| Pipeline             | 52                | 1            | False     | 1000          |
| Simglucose           | 31                | 1            | True      | 480           |
| RocketRecovery       | 7                 | 2            | True      | 500           |
| RandomFrictionHopper | 13                | 3            | True      | 1000          |
| DMSD                 | 6                 | 2            | False     | 100           |
| Fusion               | 15                | 6            | False     | 100           |
| SafetyHalfCheetah    | 18                | 6            | False     | 1000          |



## Usage

### 1.Train policy

The policy training script will automatically utilize reinforcement learning algorithms or PID tuning to train expert policies based on different tasks. During the training process, suboptimal policies will also be preserved.

```
cd scripts

python train_expert_policy.py --env_name Pipeline
```

### 2.Sample data

The data sampling script automatically retrieves suboptimal policies from the training process for sample collection. It selects three policies to collect approximately 100,000 data points for training data and one policy to collect around 20,000 data points for validation data.

```
python get_data.py --env_name Pipeline
                 
```

