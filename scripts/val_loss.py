"""The scores of trained checkpoints on held-out data (a validation split, say), comparable across
them: every checkpoint sees the same batches, drawn the same way, with the same noise.

    uv run scripts/val_loss.py --config-name abc_yam --checkpoints CKPT/10000 CKPT/20000 \
        --repo-ids abc_sim_224_val/put_relative abc_sim_224_val/conveyor_pick --batches 64

The data is the config's, read from ``--repo-ids`` instead of its own repos (sampled as the config
mixes them), normalized by the first checkpoint's statistics. Prints one JSON line per checkpoint:
its mean loss and action error (training_utils.validation_scores, over the first ``--action-dims``
dims) over the batches of every repo mixed (``all``), of each owner's repos alone when there are
several (the part of a repo id before its ``/``), and of each repo alone with ``--per-repo``.
"""

import dataclasses
import json
import logging
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tyro

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.streaming_dataset as _streaming_dataset
import openpi.training.utils as training_utils


@dataclasses.dataclass
class Args:
    config_name: str
    checkpoints: list[pathlib.Path]  # each a step's directory, with params/ and assets/
    repo_ids: list[str]
    batches: int = 64
    batch_size: int = 16
    seed: int = 0
    per_repo: bool = False  # also each repo's scores, its own batches drawn alone
    shuffle_buffer: int = 1_000
    action_dims: int | None = None  # the robot's: the action error's dims (None: all, padding included)


def _data(config: _config.TrainConfig, args: Args, repos: list[str]) -> _config.TrainConfig:
    """``config`` reading ``repos``, its statistics the first checkpoint's, one process, fixed seed;
    mixed by the config's validation weights where it has one for each repo, else by their frames, and
    every episode of each (the training repos' caps and weights are not theirs)."""
    weights = config.val_sampling_weights
    data = dataclasses.replace(
        config.data,
        repo_id=repos,
        sampling_weights={repo: weights[repo] for repo in repos} if all(r in weights for r in repos) else {},
        max_episodes={},
        assets=dataclasses.replace(config.data.assets, assets_dir=str(args.checkpoints[0] / "assets")),
        streaming_shuffle_buffer_size=args.shuffle_buffer,
    )
    return dataclasses.replace(config, data=data, batch_size=args.batch_size, num_workers=0, seed=args.seed)


def _scorer(model: _model.BaseModel, action_dims: int | None):
    """``model``'s scores of a batch, jitted: (rng, observation, actions) -> (loss, action error)."""
    graphdef, state = nnx.split(model)
    jitted = jax.jit(
        lambda state, rng, observation, actions: training_utils.validation_scores(
            nnx.merge(graphdef, state), rng, observation, actions, action_dims
        )
    )
    return lambda rng, observation, actions: jitted(state, rng, observation, actions)


def _scores(config: _config.TrainConfig, score, args: Args) -> tuple[float, float]:
    """The mean loss and action error over ``args.batches`` batches, the noise of batch i seeded by i."""
    loader = _data_loader.create_data_loader(config, shuffle=True, num_batches=args.batches)
    found = [
        score(jax.random.fold_in(jax.random.key(args.seed), i), observation, actions)
        for i, (observation, actions) in enumerate(loader)
    ]
    losses, errors = zip(*found, strict=True)
    return float(np.mean(losses)), float(np.mean(errors))


def main(args: Args) -> None:
    config = _config.get_config(args.config_name)
    groups = {"all": list(args.repo_ids)}
    owners: dict[str, list[str]] = {}
    for repo in args.repo_ids:
        owners.setdefault(repo.split("/", 1)[0], []).append(repo)
    if len(owners) > 1:
        groups.update(owners)
    if args.per_repo:
        groups.update({repo: [repo] for repo in args.repo_ids})
    for checkpoint in args.checkpoints:
        model = config.model.load(_model.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
        score = _scorer(model, args.action_dims)
        found = {name: _scores(_data(config, args, repos), score, args) for name, repos in groups.items()}
        print(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "loss": {name: loss for name, (loss, _) in found.items()},
                    "action_error": {name: error for name, (_, error) in found.items()},
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _streaming_dataset.run_main(lambda: main(tyro.cli(Args)))
