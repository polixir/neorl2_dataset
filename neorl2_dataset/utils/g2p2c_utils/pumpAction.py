import math
import pandas as pd
import pkg_resources


class Pump:
    def __init__(self, config):
        self.action_scale = config["action_scale"]
        self.pump_min = config["insulin_min"]
        self.pump_max = config["action_scale"]
        self.config = config

        # set patient specific basal
        # patient_params = pd.read_csv(PATIENT_PARA_FILE)
        # params = patient_params[patient_params.Name.str.match(patient_name)]
        # u2ss = params.u2ss.values.item()
        # BW = params.BW.values.item()
        # self.basal = u2ss * BW / 6000

    def get_basal(self):
        self.basal = 0.013698
        return self.basal

    def action(self, agent_action=None):
        agent_action = self.action_scale * (math.exp((agent_action - 1) * 4))

        rl_action = max(self.pump_min, agent_action)  # check if greater than 0
        rl_action = min(rl_action, self.pump_max)

        pump_action = rl_action
        return rl_action, pump_action


# def get_basal(patient_name='none'):
#     if patient_name == 'none':
#         print('Patient name not provided')
#     quest = pd.read_csv(CONTROL_QUEST)
#     patient_params = pd.read_csv(PATIENT_PARA_FILE)
#     q = quest[quest.Name.str.match(patient_name)]
#     params = patient_params[patient_params.Name.str.match(patient_name)]
#     u2ss = params.u2ss.values.item()
#     BW = params.BW.values.item()
#     basal = u2ss * BW / 6000
#     return basal

# patient:adolescent#001, Basal: 0.01393558889998341
# patient:adolescent#002, Basal: 0.01529933523331466
# patient:adolescent#003, Basal: 0.0107966168000268
# patient:adolescent#004, Basal: 0.01456052239999348
# patient:adolescent#005, Basal: 0.012040315333360101
# patient:adolescent#006, Basal: 0.014590183333350241
# patient:adolescent#007, Basal: 0.012943099999997907
# patient:adolescent#008, Basal: 0.009296317679986218
# patient:adolescent#009, Basal: 0.010107192533314517
# patient:adolescent#010, Basal: 0.01311652320003506

# patient:child#001, Basal: 0.006578422760004344
# patient:child#002, Basal: 0.006584850490398568
# patient:child#003, Basal: 0.004813171311526304
# patient:child#004, Basal: 0.008204957581639397
# patient:child#005, Basal: 0.00858548873873053
# patient:child#006, Basal: 0.006734515005432704
# patient:child#007, Basal: 0.007786704078078988
# patient:child#008, Basal: 0.005667427170273473
# patient:child#009, Basal: 0.006523757656342553
# patient:child#010, Basal: 0.006625406512238658

# patient:adult#001, Basal: 0.02112267499992533
# patient:adult#002, Basal: 0.022825539499994
# patient:adult#003, Basal: 0.023755205833326954
# patient:adult#004, Basal: 0.014797182203265
# patient:adult#005, Basal: 0.01966383496660751
# patient:adult#006, Basal: 0.028742228666635828
# patient:adult#007, Basal: 0.022858123833300104
# patient:adult#008, Basal: 0.01902372999996952
# patient:adult#009, Basal: 0.018896863133377337
# patient:adult#010, Basal: 0.01697815740005382
