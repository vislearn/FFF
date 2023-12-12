
import pytest
from lightning_trainable.launcher.utils import parse_config_dict

config_files_collection = [
    # M-FFF models
    # Toy data on circle
    ['../configs/m-fff/von-mises-circle.yaml'],
    # SO(3) models
    ['../configs/m-fff/so3.yaml'],
    ['../configs/m-fff/so3.yaml', '../configs/m-fff/so3-64.yaml'],
    ['../configs/m-fff/so3.yaml', '../configs/m-fff/so3-32.yaml'],
    # Earth models
    ['../configs/m-fff/earth.yaml', '../configs/m-fff/earthquake.yaml'],
    ['../configs/m-fff/earth.yaml', '../configs/m-fff/fire.yaml'],
    ['../configs/m-fff/earth.yaml', '../configs/m-fff/volcano.yaml'],
    ['../configs/m-fff/earth.yaml', '../configs/m-fff/flood.yaml'],
    # Toric models
    ['../configs/m-fff/rna.yaml'],
    ['../configs/m-fff/protein.yaml'],

    # FIF models
    ['../configs/fif/tabular.yaml', '../configs/fif/tabular-power.yaml'],
    ['../configs/fif/tabular.yaml', '../configs/fif/tabular-miniboone.yaml'],
    ['../configs/fif/tabular.yaml', '../configs/fif/tabular-gas.yaml'],
    ['../configs/fif/tabular.yaml', '../configs/fif/tabular-hepmass.yaml'],
    ['../configs/fif/toy.yaml'],
    ['../configs/fif/celeba.yaml'],
    ['../configs/fif/mnist.yaml'],
    ['../configs/fif/mnist.yaml', '../configs/fif/mnist-conditional.yaml'],

    # FFF models
    ['../configs/fff/sbi_base.yaml', '../configs/fff/sbi_gaussian_mixture_example.yaml'],
    ['../configs/fff/dw4.yaml'],
    ['../configs/fff/lj55.yaml'],
    ['../configs/fff/lj13.yaml'],
    ['../configs/fff/qm9.yaml'],
]


@pytest.mark.parametrize("config_files", config_files_collection)
def test_configs(config_files):
    config = parse_config_dict(config_files)

    if "model" not in config:
        return

    model = config.pop("model")
    module_name, class_name = model.rsplit(".", 1)
    if "data_set" in config and "root" in config["data_set"]:
        config["data_set"]["root"] = "../" + config["data_set"]["root"]

    # Smallest memory footprint
    config["batch_size"] = 2
    # config["exact_chunk_size"] = 1

    try:
        model = getattr(__import__(module_name, fromlist=[class_name]), class_name)(config)
    except FileNotFoundError:
        pytest.skip(f"Data not found: {config_files}")

    model.hparams.max_steps = 1
    model.fit()


if __name__ == '__main__':
    pass
