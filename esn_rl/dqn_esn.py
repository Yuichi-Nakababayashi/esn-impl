import random
from typing import List, Type
import gym
import hydra
import numpy as np
from cmaes import CMA
from omegaconf.dictconfig import DictConfig
from scipy.sparse.csr import csr_matrix
import matplotlib.pyplot as plt
from torch import optim


class ESN:
    def __init__(
        self,
        num_unit: int,
        w_in_dist: List[float],
        w_internal_dist: List[float],
        in_dim: int,
        nu_dist: List[float],
        zero_density: float = 0.05,
        seed: int = 123,
    ) -> None:
        """
        Args:
            num_unit: the number of reservoir units
            w_in_dist: min, max of W_{in} matrix
            w_internal_dist: min, max of reservoir matrix weights
            in_dim: input dimension (in case cartpole, this value is 4)
            internal_density: the ratio of non-zero components of w_internal_dist
            nu_dist: distribution of white noise added to the output
            save_path: path to save the reservoir internal weights
            seed: random seed
        Instance Variables
            w_in: weights of w_in layer
            w_internal: weights of reservoir layer
            nu_dist: distribution of white noise added to the output
            internal_state: internal_state of the reservoir
        """
        np.random.seed(seed)
        # generate W_{in} weights
        self.w_in = np.random.uniform(*w_in_dist, size=(num_unit, in_dim))
        self.nu_dist = nu_dist
        self.num_unit = num_unit
        non_zero_num = int(num_unit ** 2 * (1 - zero_density))
        self.w_internal = np.zeros([num_unit, num_unit])
        # non_zero_val: randomly generate the non-zero value
        non_zero_val = np.random.uniform(*w_internal_dist, non_zero_num)
        # non_zero_idx: randomly choose the location of non-zero value
        non_zero_idx = random.sample(range(num_unit ** 2), non_zero_num)
        # fill out the nonzero value
        for idx, val in zip(non_zero_idx, non_zero_val):
            self.w_internal[idx // num_unit][idx % num_unit] = val
        eig_max = np.max(np.abs(np.linalg.eigvals(self.w_internal)))
        # re-scale
        self.w_internal *= 0.8 / eig_max
        # convert to csr matrix (for boosting)
        self.w_internal = csr_matrix(self.w_internal)
        self.internal_state = np.zeros(num_unit)

    def forward_impl(self, prev_data: np.array) -> np.array:
        """function to update and return the reservoir unit state
        Args:
            - prev_data: previous data
        Return:
            - reservoir unit state after the update
        """
        nu = np.random.uniform(*self.nu_dist)
        # udpate the internal state
        self.internal_state = np.tanh(
            self.w_internal @ self.internal_state + self.w_in @ prev_data + nu
        )
        return self.internal_state

    def reset(self):
        self.internal_state = np.zeros(self.num_unit)


def cma_objective(env, w_out: np.array, esn: Type[ESN]) -> float:
    """objective function
    Args:
        env: environment
        w_out: weights of read-out layer
        esn: echo state network object
    Return:
        reward: reward from the environment
    """
    def gen_action(action: float) -> int:
        """function to generate the action from float value
        if action >= 0, take 0 action, 1 otherwise 
        """
        return int(action >= 0)
    time_step = 0
    esn.reset()
    state = env.reset()
    while 1:
        esn_state = esn.forward_impl(state)
        action = np.array(esn_state.T @ w_out)
        # NOTE: convert to discrete action space
        action = gen_action(action)
        next_state, _, done, _ = env.step(action)
        state = next_state
        if done or time_step == 100:
            break
        time_step += 1
    return time_step * (-1)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    env_cfg = cfg["environment"]
    env = gym.make(env_cfg["name"])
    o_space = env.observation_space.shape[0]
    esn_cfg = cfg["esn_model"]
    esn = ESN(in_dim=o_space, **esn_cfg)
    cma_cfg = cfg["cma_es"]
    # NOTE: shape of the readout layer
    optimizer = CMA(
        mean=np.zeros(esn_cfg["num_unit"]),
        sigma=cma_cfg["sigma"],
        population_size=cma_cfg["population_size"],
    )
    num_gen = cma_cfg["max_num_gen"]
    avg_rewards = []
    best_score = -1
    for generation in range(num_gen):
        solutions = []
        tot_reward = 0
        out_mean = np.zeros(esn_cfg["num_unit"])
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            out_mean += x
            reward = cma_objective(env, x, esn)
            tot_reward += reward * -1
            solutions.append((x, reward))
        out_mean /= optimizer.population_size
        optimizer.tell(solutions)
        avg_score = tot_reward / optimizer.population_size
        avg_rewards.append(avg_score)
        if best_score < avg_score:
            best_score = avg_score
            np.save("w_out_avg.npy", out_mean)
        print(f"tot reward: {avg_score}")
    plt.plot(list(range(len(avg_rewards))), avg_rewards)
    plt.savefig("avg_rewards.png")


if __name__ == "__main__":
    main()
