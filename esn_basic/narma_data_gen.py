from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def narma_gen(
    length: int,
    save_path: str,
    narma_num: int = 10,
    uniform_arg: List[float] = [0.0, 0.5],
    ignore_num: int = 200,
    seed: int = 123,
) -> Tuple[np.array]:
    """function to output the data for narma 10 task
    Args:
        length (int): length of narma (narma_num) data
        save_path (str): the path to the narma (narma_num) csv data
        uniform_arg (List): the argument for generation noise (low, high)
        ignore_num (int):the number of steps to be ignored
        seed (int): seed for np.random
    Return:
        uniform_in, unnarm_array (np.array): the narma data(input and output)
    *** narma (narma_num) data is going to be generated ***
    """
    np.random.seed(seed)
    narma_array = np.array([0.0] * (length + ignore_num))
    uniform_arg = uniform_arg + [length + ignore_num]
    uniform_in = np.random.uniform(*uniform_arg)
    for t in range(narma_num, length + ignore_num):
        prev_sum = np.sum(narma_array[t - narma_num: t])
        prev_val = narma_array[t - 1]
        narma_array[t] = (
            0.3 * prev_val
            + 0.05 * prev_val * prev_sum
            + 1.5 * uniform_in[t - narma_num] * uniform_in[t - 1]
            + 0.1
        )
    narma_df = pd.DataFrame()
    narma_df["input"] = uniform_in[ignore_num:]
    narma_df["output"] = narma_array[ignore_num:]
    narma_df.to_csv(save_path, index=False)
    return uniform_in, narma_array


if __name__ == "__main__":
    # for test only
    length = 1000
    narma_in, narma_out = narma_gen(length, "test.csv")
