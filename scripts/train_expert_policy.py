import os
import fire
import shutil

from neorl2.utils.utils import set_seed
from neorl2.utils.trainer import train_plicy

def train(env_name,mode):
    seed = 42
    set_seed(seed)
    
    folder_path = f"./logs/{env_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        try:
            shutil.rmtree(f"./logs/{env_name}")
        except FileNotFoundError as e:
            print(e)
    
    train_plicy(env_name,mode=mode)

def main(env_name=None, mode="train"):
    print(f"Mode : {mode}")
    if env_name is None:
        env_name_list = ["Pipeline", "Simglucose", 
                         "RocketRecovery", "RandomFrictionHopper", 
                         "DMSD", "Fusion", 
                         "Salespromotion","SafetyHalfCheetah"]
    else:
        if isinstance(env_name, tuple):
            env_name_list = list(env_name)
        elif not isinstance(env_name, list):
            env_name_list = [env_name,]
        else:
            raise NotImplementedError
            
    for env_name in env_name_list:
        print(f"Start train task : {env_name}")
        train(env_name,mode=mode)
    
if __name__ == "__main__":
    fire.Fire(main)
    