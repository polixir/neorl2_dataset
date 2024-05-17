from typing import Optional
import random
import os
import numpy as np
import torch
from torch.backends import cudnn


def set_seed(seed: Optional[int]):
    if seed is None:
        return

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    
    
def mkdir_p(dir_path):
    """
    mkdir -p functionality in python
    :param dir_path:
    :return:
    """
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e