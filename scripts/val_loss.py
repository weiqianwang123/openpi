"""The loss of trained checkpoints on held-out data (a validation split, say), comparable across them:
every checkpoint sees the same batches, drawn the same way, with the same noise.

    uv run scripts/val_loss.py --config-name abc_yam --checkpoints CKPT/10000 CKPT/20000 \
        --repo-ids abc_sim_224_val/put_relative abc_sim_224_val/conveyor_pick --batches 64

The data is the config's, read from ``--repo-ids`` instead of its own repos (sampled as the config
mixes them), normalized by the first checkpoint's statistics. Prints one JSON line per checkpoint:
its mean loss over the batches, and the loss by repo with ``--per-repo``.
"""

import dataclasses
import json
import logging
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import openpi.models.model as _model
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.streaming_dataset as _streaming_dataset


@dataclasses.dataclass
class Args:
    config_name: str
    checkpoints: list[pathlib.Path]  # each a step's directory, with params/ and assets/
    repo_ids: list[str]
    batches: int = 64
    batch_size: int = 16
    seed: int = 0
    per_repo: bool = False  # also each repo's loss, its own batches drawn alone
    shuffle_buffer: int = 1_000


def _data(config: _config.TrainConfig, args: Args, repos: list[str]) -> _config.TrainConfig:
    """``config`` reading ``repos``, its statistics the first checkpoint's, one process, fixed seed."""
    data = dataclasses.replace(
        config.data,
        repo_id=repos,
        assets=dataclasses.replace(config.data.assets, assets_dir=str(args.checkpoints[0] / "assets")),
        streaming_shuffle_buffer_size=args.shuffle_buffer,
    )
    return dataclasses.replace(config, data=data, batch_size=args.batch_size, num_workers=0, seed=args.seed)


def _mean_loss(config: _config.TrainConfig, loss_fn, args: Args) -> float:
    """The mean loss over ``args.batches`` batches, the noise of batch i seeded by i."""
    loader = _data_loader.create_data_loader(config, shuffle=True, num_batches=args.batches)
    losses = []
    for i, (observation, actions) in enumerate(loader):
        rng = jax.random.fold_in(jax.random.key(args.seed), i)
        losses.append(float(jnp.mean(loss_fn(rng, observation, actions))))
    return float(np.mean(losses))


def main(args: Args) -> None:
    config = _config.get_config(args.config_name)
    groups = {"all": list(args.repo_ids)}
    if args.per_repo:
        groups.update({repo: [repo] for repo in args.repo_ids})
    for checkpoint in args.checkpoints:
        model = config.model.load(_model.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
        loss_fn = nnx_utils.module_jit(model.compute_loss)
        found = {name: _mean_loss(_data(config, args, repos), loss_fn, args) for name, repos in groups.items()}
        print(json.dumps({"checkpoint": str(checkpoint), "loss": found}), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _streaming_dataset.run_main(lambda: main(tyro.cli(Args)))
