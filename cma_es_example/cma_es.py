import matplotlib.pyplot as plt
import numpy as np
from cmaes import CMA
from numpy.core.fromnumeric import mean


def quadratic(x):
    """objective function"""
    x1 = x[0]
    x2 = x[1]
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2


if __name__ == "__main__":
    optimizer = CMA(mean=np.zeros(2), sigma=1.3)
    num_gen = 50
    loss_list = [0] * num_gen
    for generation in range(num_gen):
        solutions = []
        # population_size: いくつの候補を生成するか
        tot_loss = 0
        for _ in range(optimizer.population_size):
            # 値を生成する
            x = optimizer.ask()
            value = quadratic(x)
            tot_loss += value
            solutions.append((x, value))
        # これでoptimizeされる
        optimizer.tell(solutions)
        loss_list[generation] = tot_loss / optimizer.population_size
    plt.plot(list(range(num_gen)), loss_list)
    plt.savefig("cma_es_eg.png")
