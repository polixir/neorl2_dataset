import os
import numpy as np
import json, zipfile

input_mins = np.array([0.3, 1.6,0.1,0.5, 1.265,2.18 ])
input_maxs = np.array([0.8, 2.0,0.5,0.9, 1.36, 2.29 ])

def actv(x, method):
    if method == 'relu':
        return np.max([np.zeros_like(x), x], axis=0)
    elif method == 'tanh':
        return np.tanh(x)
    elif method == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif method == 'linear':
        return x


class SB2_model():
    def __init__(self, model_path, low_state, high_state, low_action, high_action, 
                activation='relu', last_actv='tanh', norm=True, bavg=0., level=0):
        zf = zipfile.ZipFile(model_path)
        data = json.loads(zf.read('data').decode("utf-8"))
        self.parameter_list = json.loads(zf.read('parameter_list').decode("utf-8"))
        self.parameters = dict(np.load(zf.open('parameters')))
        self.layers = data['policy_kwargs']['layers'] if 'layers' in data['policy_kwargs'].keys() else [64, 64]
        self.low_state, self.high_state = low_state, high_state
        self.low_action, self.high_action = low_action, high_action
        self.activation, self.last_actv = activation, last_actv
        self.norm = norm
        self.bavg = bavg
        self.level = level

    def predict(self, x, yold=None):
        xnorm = 2 * (x - self.low_state) / np.subtract(self.high_state, self.low_state) - 1 if self.norm else x
        ynorm = xnorm
        for i, layer in enumerate(self.layers):
            w, b = self.parameters[f'model/pi/fc{i}/kernel:0'], self.parameters[f'model/pi/fc{i}/bias:0']
            ynorm = actv(np.matmul(ynorm, w) + b, self.activation)
        w, b = self.parameters[f'model/pi/dense/kernel:0'], self.parameters[f'model/pi/dense/bias:0']
        ynorm = actv(np.matmul(ynorm, w) + b, self.last_actv)

        y = 0.5 * np.subtract(self.high_action, self.low_action) * (ynorm + 1) + self.low_action if self.norm else ynorm
        if yold is None:
            yold = x[:len(y)]
        y =  self.bavg * yold + (1 - self.bavg) * y

        y = (y-input_mins)/(input_maxs-input_mins)
        y = y*2-1
        y = np.random.normal(y, self.level)
        y = y.astype(np.float32)
        return y#, None