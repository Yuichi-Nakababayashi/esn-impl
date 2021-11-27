import hydra
from omegaconf.dictconfig import DictConfig

from esn_forward import esn_forward
from esn_offline import evaluate_offline, train_offline
from narma_data_gen import narma_gen


@hydra.main(config_path="config", config_name="baseline")
def main(cfg: DictConfig) -> None:
    # load config
    data_cfg = cfg["data"]
    esn_cfg = cfg["esn_model"]
    train_cfg = cfg["train"]
    evaluate_cfg = cfg["evaluation"]
    narma_data = narma_gen(**data_cfg)
    esn_states = esn_forward(narma_data, **esn_cfg)
    w_out = train_offline(narma_data, esn_states, **train_cfg)
    evaluate_offline(w_out, data_cfg, esn_cfg, **evaluate_cfg)


if __name__ == "__main__":
    main()
