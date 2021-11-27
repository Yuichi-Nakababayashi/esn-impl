import hydra
import numpy as np
import matplotlib.pyplot as plt

from narma_data_gen import narma_gen
from esn_forward import esn_forward


def train_offline(
    narma_data: np.array,
    esn_states: np.array,
    ignore_num: int = 200,
    is_square: bool = True,
):
    narma_in, narma_out = narma_data
    narma_in, narma_out = np.array(narma_in), np.array(narma_out)
    # 1.drop the first {ignore_num} data, then apply arctanh function (preprocessing)
    narma_in, narma_out = narma_in[ignore_num:], np.arctanh(
        narma_out[ignore_num:])
    # 2. solve the problem
    narma_in = np.expand_dims(narma_in, -1)
    narma_out = np.expand_dims(narma_out, -1)
    m_mat = np.concatenate([narma_in, esn_states], axis=1)
    if is_square:
        m_mat = np.concatenate([m_mat, m_mat ** 2], axis=1)
    w_out = (np.linalg.pinv(m_mat) @ narma_out).T
    # training done
    return w_out


def evaluate_offline(
    w_out: np.array,
    data_cfg: dict,
    esn_cfg: dict,
    length: int,
    seed: int,
    is_square: bool,
    img_path: str
):
    data_cfg["length"] = length
    data_cfg["ignore_num"] = int(length * 0.1)
    data_cfg["seed"] = seed
    data_cfg["save_path"] = data_cfg["save_path"].replace("train", "test")
    esn_cfg["save_path"] = esn_cfg["save_path"].replace("train", "test")
    ignore_num = data_cfg["ignore_num"]
    # 1. generate data
    narma_data = narma_gen(**data_cfg)
    # 2. calculate reservoir state (***w_mat should be the save***)
    esn_states = esn_forward(
        narma_data, w_internal_path="w_mat.npy", **esn_cfg)
    # 3. preprocess
    narma_in, narma_out = narma_data
    narma_in, narma_out = np.array(narma_in), np.array(narma_out)
    narma_in, narma_out = narma_in[ignore_num:], narma_out[ignore_num:]
    narma_in = np.expand_dims(narma_in, -1)
    narma_out = np.expand_dims(narma_out, -1)
    m_mat = np.concatenate([narma_in, esn_states], axis=1)
    if is_square:
        m_mat = np.concatenate([m_mat, m_mat ** 2], axis=1)
    # 4. inference
    y_hat = np.tanh(m_mat @ w_out.T)
    # 5. calculate error
    err = np.linalg.norm(narma_out - y_hat) / len(y_hat) / np.var(narma_out)
    plt.plot(list(range(len(y_hat))), y_hat, label="inference")
    plt.plot(list(range(len(y_hat))), narma_out, label="gt")
    plt.legend()
    plt.savefig(img_path)
    print(f"mse is :{err}")
