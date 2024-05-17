import os
import sys
from setuptools import setup
from setuptools import find_packages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'neorl2_dataset'))

assert sys.version_info.major == 3, \
    "This repo is designed to work with Python 3." \
    + "Please install it before proceeding."

setup(
    name='neorl2_dataset',
    author='Polixir Technologies Co., Ltd.',
    py_modules=['neorl2_dataset'],
    version="0.0.1",
    packages=find_packages(),
    install_requires=[        
        'neorl2',
        'stable_baselines3==2.2.1',
        'tqdm==4.66.1',
        'tianshou==0.4.11',
        'sb3_contrib==2.2.1',
    ],

)