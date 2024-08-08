import pandas as pd
import numpy as np
from env_SL import gen_initstate


class SimEnv:
    def __init__(self,
                supply_df: pd.DataFrame,
                device,
                result_filename="result285.csv"):

        self.device = device
        self.result_filename = result_filename
        if not isinstance(supply_df, pd.DataFrame): raise TypeError("Supply info must be a dataframe.")
        self.supply_df = supply_df
        self.start_num = 1
        self.n_units = self.supply_df.shape[0]

        self.p_min_vec = self.supply_df["P_min"].to_numpy()
        self.p_max_vec = self.supply_df["P_max"].to_numpy()
        self.ramp_dn_vec = self.supply_df["Ramp_down"].to_numpy()
        self.ramp_up_vec = self.supply_df["Ramp_up"].to_numpy()
        self.ac_vec = self.supply_df["aCost"].to_numpy()
        self.bc_vec = self.supply_df["bCost"].to_numpy()
        self.cc_vec = self.supply_df["cCost"].to_numpy()
        self.incomplete_episode = False
        self.done = False
        self.period_demand, self.day_num, self.day, self.node_demand_list = gen_initstate(self.start_num, self.n_units, self.device)
        self.n_timesteps = len(self.node_demand_list)
        self.timestep = 0
        self.tolerance = 0.3
        self.pen_factor = 1000

    def get_current_state(self):
        state_dict = {
            "timestep": self.timestep,
            "demand": self.period_demand[self.timestep],
            "day_num": self.day_num,
            "day": self.day,
            # "power": self.power,
            "p_min": self.p_min_vec,
            "p_max": self.p_max_vec,
            "ramp_dn": self.ramp_dn_vec,
            "ramp_up": self.ramp_up_vec,
            "aCost": self.ac_vec,
            "bCost": self.bc_vec,
            "cCost": self.cc_vec,
        }
        Gnode_state = self.node_demand_list[self.timestep]
        return Gnode_state, state_dict

    def get_next_state(self, action):
        self.timestep += 1
        next_state, next_state_dict = self.get_current_state()
        return next_state, next_state_dict

    def count_demand_prod_balance(self, power, demand, tolerance=0.3):
        differences = np.abs(power - demand)
        within_range = differences <= tolerance
        count_within_range = np.sum(within_range)
        return count_within_range

    def cost_evaluation(self, demand: float, power: np.ndarray):
        prod_cost_funs_vec = (self.ac_vec/2) * power ** 2 + self.bc_vec * power + self.cc_vec
        prod_cost = np.sum(prod_cost_funs_vec)
        prod_demand_cost = np.sum(np.sum(power, axis=1) - demand)
        return prod_cost, prod_demand_cost

    def cal_reward(self, power, demand):
        prod_cost, prod_demand_cost = self.cost_evaluation(demand, power)
        reward = -prod_cost #- self.pen_factor * prod_demand_cost
        return reward, prod_cost, prod_demand_cost

    def is_terminal(self):
        if self.timestep >= (self.n_timesteps-1):
            return True
        return False

    def step(self, action: np.ndarray):
        if not isinstance(action, np.ndarray):
            raise TypeError("Action vector must be a NumPy array.")
        state, state_dict = self.get_current_state()
        reward, prod_cost, prod_demand_cost = self.cal_reward(action, state_dict["demand"])
        is_done = self.is_terminal()
        if not is_done:
            next_state, next_state_dict = self.get_next_state(action.T)
        else:
            next_state, next_state_dict = self.reset()
        return next_state, reward, is_done, state, prod_cost, prod_demand_cost, state_dict, state_dict["demand"]

    def reset(self):
        self.timestep = 0
        self.p_min_vec = self.supply_df["P_min"].to_numpy()
        self.p_max_vec = self.supply_df["P_max"].to_numpy()
        self.ramp_dn_vec = self.supply_df["Ramp_down"].to_numpy()
        self.ramp_up_vec = self.supply_df["Ramp_up"].to_numpy()
        self.ac_vec = self.supply_df["aCost"].to_numpy()
        self.bc_vec = self.supply_df["bCost"].to_numpy()
        self.cc_vec = self.supply_df["cCost"].to_numpy()
        self.incomplete_episode = False
        self.done = False
        self.period_demand, self.day_num, self.day, self.node_demand_list = gen_initstate(
            self.start_num, self.n_units, self.device)
        state, state_dict = self.get_current_state()

        return state, state_dict
