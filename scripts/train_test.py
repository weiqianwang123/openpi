import dataclasses
import os
import pathlib

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

from . import train


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)


def test_validation_is_also_scored_by_owner():
    """With repos of two owners, the scores without a suffix mix them all and each owner's repos are
    scored alone (on half the batches, `/<owner>`), their weights kept; with one owner, only the mix."""
    config = _config.TrainConfig(
        name="val",
        val_repo_ids=["sim_val/a", "sim_val/b", "real_val/c"],
        val_sampling_weights={"sim_val/a": 0.1, "sim_val/b": 0.4, "real_val/c": 0.5},
        val_batches=32,
    )
    sets = train.validation_sets(config)
    assert list(sets) == ["", "/sim_val", "/real_val"]
    assert sets[""][2] == 32
    assert sets["/sim_val"] == (["sim_val/a", "sim_val/b"], {"sim_val/a": 0.1, "sim_val/b": 0.4}, 16)
    single = dataclasses.replace(config, val_repo_ids=["sim_val/a"], val_sampling_weights={})
    assert list(train.validation_sets(single)) == [""]
