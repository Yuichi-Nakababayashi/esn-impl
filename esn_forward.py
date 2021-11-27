import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from omegaconf.dictconfig import DictConfig
from scipy.sparse import csr_matrix
from tqdm import tqdm

from narma_data_gen import narma_gen


def esn_forward(
    narma_data: Tuple[np.array],
    save_path: str = "test.npy",
    num_unit: int = 100,
    ignore_num: int = 200,
    w_in_dist: List[float] = [-0.1, 0.1],
    seed: int = 123,
    w_internal_dist: List[float] = [-1.0, 1.0],
    nu_std: List[float] = [-1e-3, 1e-3],
    w_internal_path: str = None
):
    """function to calcualte the internal state
    Args:
        - narma_data (Tuple[np.array]): narma data (input, output)
        - num_unit (int): the number of internal units
        - w_dist (List[float]): the argument for generation noise (low, high)
        - seed (int): seed for np.random
        - nu_std (float): std for nu (white noise for internal state)
    Return:
        - x_mat (np.array): matrix containing all the internal states at each time step
    """
    # 0. set seed
    np.random.seed(seed)
    # 1. generate w_in
    w_in_dist = w_in_dist + [num_unit]
    w_in = np.random.uniform(*w_in_dist)
    # 2. generate internal weight (annotated with W in a paper)
    if w_internal_path is None:
        non_zero_num = int(num_unit ** 2 * 0.05)
        w_internal = np.zeros([num_unit, num_unit])
        w_internal_dist = w_internal_dist + [non_zero_num]
        # non_zero_val: randomly generate the non-zero value
        non_zero_val = np.random.uniform(*w_internal_dist)
        # non_zero_idx: randomly choose the location of non-zero value
        non_zero_idx = random.sample(range(num_unit ** 2), non_zero_num)
        # fill out the nonzero value
        for idx, val in zip(non_zero_idx, non_zero_val):
            w_internal[idx // num_unit][idx % num_unit] = val
        eig_max = np.max(np.abs(np.linalg.eigvals(w_internal)))
        # re-scale
        w_internal *= 0.8 / eig_max
    # convert to csr mat
    else:
        w_internal = np.load(w_internal_path)
    if(w_internal_path is None):
        np.save("w_mat.npy", w_internal)
    w_internal = csr_matrix(w_internal)
    # 3. calculate the internal state one by one
    narma_in, _ = narma_data
    length = len(narma_in)
    x_t = np.zeros([length, num_unit])
    w_internal_dist[-1] = num_unit
    nu_std = nu_std + [num_unit]
    # generate the first internal state
    x_t[0] = np.random.uniform(*w_internal_dist) + np.random.uniform(*nu_std)
    for t in range(length - 1):
        nu = np.random.uniform(*nu_std)
        x_t[t + 1] = np.tanh(w_internal @ x_t[t] + w_in * narma_in[t] + nu)
    # drop the first {ignore_num} steps
    x_t = x_t[ignore_num:]
    np.save(save_path, x_t)
    return x_t


if __name__ == "__main__":
    length = 1000
    narma_data = narma_gen(length=1000, save_path="test.csv")
    esn_forward(narma_data)
