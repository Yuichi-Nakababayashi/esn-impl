import random
import ray
from typing import List, Type

import gym
import hydra
import matplotlib.pyplot as plt
import numpy as np
from cmaes import CMA
from matplotlib import animation
from omegaconf.dictconfig import DictConfig
from scipy.sparse.csr import csr_matrix
from torch import optim
from tqdm import tqdm


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
        np.save("w_in.npy", self.w_in)
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
        np.save("w_internal.npy", self.w_internal)
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

    def load(self, load_path: str):
        self.w_internal = np.load(f"{load_path}/w_internal.npy")
        self.w_in = np.load(f"{load_path}/w_in.npy")
        self.reset()


def save_frames_as_gif(frames: List[np.array], filename: str) -> None:
    # Mess with this to change frame size
    print(f"start saving {filename}")
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(filename, writer="imagemagick", fps=60)
    print(f"save done for {filename}")


@ray.remote
def cma_objective(
    env,
    w_out: np.array,
    esn: Type[ESN],
    t_max: int,
    gif_idx: int = -1,
    is_render: bool = False,
) -> float:
    """objective function
    Args:
        env: environment
        w_out: weights of read-out layer
        esn: echo state network object
        t_max: the number of max steps
        is_render: do rendering or not
    Return:
        reward: reward from the environment
    """

    def gen_action(action: float) -> int:
        """function to generate the action from float value
        if action >= 0, take 0 action, 1 otherwise
        TODO: in case acrobot, need to multiply by 2
        """
        return int(action >= 0) * 2

    time_step = 0
    reward = 0
    esn.reset()
    state = env.reset()
    frames = []
    while 1:
        if is_render:
            frames.append(env.render(mode="rgb_array"))
        esn_state = esn.forward_impl(state)
        action = np.array(esn_state.T @ w_out)
        # NOTE: convert to discrete action space
        action = gen_action(action)
        next_state, r, done, _ = env.step(action)
        reward += r
        state = next_state
        if done or time_step == t_max:
            break
        time_step += 1
    if is_render:
        filename = f"render_{gif_idx}.gif"
        env.close()
        save_frames_as_gif(frames, filename=filename)
    return reward * (-1)


def gen_reward_graph(
    reward_list: List[float], file_path: str, title: str = "training..."
):
    """function to generate png and save it to the specified path"""
    plt.clf()
    plt.plot(list(range(len(reward_list))), reward_list)
    plt.title(title)
    plt.xlabel("num epoch")
    plt.ylabel("avg reward")
    plt.title("training...")
    plt.savefig(file_path)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    env_cfg = cfg["environment"]
    t_max = env_cfg["t_max"]
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
        out_mean = np.zeros(esn_cfg["num_unit"])
        print(f"generation {generation  +1} starts ")
        res_list = []
        x_list = []
        for _ in tqdm(range(optimizer.population_size)):
            x = optimizer.ask()
            out_mean += x
            res = cma_objective.remote(env, x, esn, t_max=t_max)
            res_list.append(res)
            x_list.append(x)
        # wait until all the processes are over
        res_list = ray.get(res_list)
        for x, reward in zip(x_list, res_list):
            solutions.append((x, reward))
        out_mean /= optimizer.population_size
        optimizer.tell(solutions)
        # avg_scoreは、w_outのavgで再度計算する
        avg_score = cma_objective.remote(
            env,
            out_mean,
            esn,
            t_max=t_max,
            gif_idx=(generation + 1),
            is_render=False,
        )
        avg_score = ray.get(avg_score) * (-1)
        avg_rewards.append(avg_score)
        if best_score > avg_score:
            best_score = avg_score
            np.save("w_out_avg.npy", out_mean)
        print(f"tot reward: {avg_score}")
        gen_reward_graph(avg_rewards, "avg_rewards.png")
    gen_reward_graph(avg_rewards, "avg_rewards.png", "training done")


if __name__ == "__main__":
    main()
