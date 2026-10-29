import dataclasses
import functools
import logging
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.streaming_dataset as _streaming_dataset
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

# Set on every host of a multi-host TPU slice (by its launcher).
MULTIHOST_ENV = "OPENPI_MULTIHOST"


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled or jax.process_index() != 0:  # one host of a multi-host slice logs
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    # The run id lives beside the checkpoints, so a job that resumes from them rejoins its own W&B
    # run instead of opening a second one. On a preemptible TPU that is the difference between one
    # continuous curve and a new chart per preemption.
    id_file = ckpt_dir / "wandb_id.txt"
    if resuming and id_file.exists():  # a run that started without W&B starts its W&B run now
        run_id = id_file.read_text().strip()
        # `allow`, not `must`: the id can outlive the run it named. A first attempt that was
        # preempted before its first log, or that ran with WANDB_MODE=offline, leaves an id W&B has
        # never heard of -- and `must` turns that into a hard failure that kills the retry, on a run
        # that is otherwise perfectly resumable from its checkpoint. `allow` continues the run when
        # it exists and (re)creates it under the same id when it doesn't.
        logging.info(f"Resuming W&B run {run_id}")
        wandb.init(id=run_id, resume="allow", name=config.exp_name, project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        id_file.write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


@at.typecheck
def val_step(
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    action_dims: int | None = None,
) -> tuple[at.Float[at.Array, ""], at.Float[at.Array, ""]]:
    """The scores of the parameters a checkpoint holds (the EMA's, when kept) on a batch: the mean loss
    and the action error over the first `action_dims` dims (training_utils.validation_scores)."""
    params = state.params if state.ema_params is None else state.ema_params
    observation, actions = batch
    return training_utils.validation_scores(nnx.merge(state.model_def, params), rng, observation, actions, action_dims)


def validation_sets(config: _config.TrainConfig) -> dict[str, tuple[list[str], dict[str, float], int]]:
    """What a validation scores, by the suffix its scores are logged with (`val_loss<suffix>`,
    `val_action_error<suffix>`): every validation repo mixed by `val_sampling_weights` over
    `val_batches` (no suffix); and when the repos come from several owners (the part of a repo id before
    its `/`: a simulator's and a robot's datasets, say), each owner's repos alone over half as many
    batches (`/<owner>`), to tell which one a change comes from. Each: its repos, their weights and its
    batches."""
    weights = dict(config.val_sampling_weights)
    sets = {"": (list(config.val_repo_ids), weights, config.val_batches)}
    owners: dict[str, list[str]] = {}
    for repo in config.val_repo_ids:
        owners.setdefault(repo.split("/", 1)[0], []).append(repo)
    if len(owners) > 1:
        for owner, repos in owners.items():
            own = {repo: weights[repo] for repo in repos if repo in weights}
            sets[f"/{owner}"] = (repos, own, max(1, config.val_batches // 2))
    return sets


def validate(config: _config.TrainConfig, pval_step, val_rng, train_state, data_sharding) -> dict[str, float]:
    """Every validation set's scores (`validation_sets`) for `train_state`, by the names they are logged
    as, batch i's noise drawn from `val_rng` folded with i."""
    scores = {}
    for suffix, (repos, weights, batches) in validation_sets(config).items():
        loader = validation_loader(config, data_sharding, repos, weights, batches)
        found = [pval_step(jax.random.fold_in(val_rng, i), train_state, batch) for i, batch in enumerate(loader)]
        losses, errors = zip(*found, strict=True)
        scores[f"val_loss{suffix}"] = float(jnp.mean(jnp.stack(losses)))
        scores[f"val_action_error{suffix}"] = float(jnp.mean(jnp.stack(errors)))
    return scores


def validation_loader(
    config: _config.TrainConfig,
    sharding: jax.sharding.Sharding,
    repo_ids: list[str],
    weights: dict[str, float],
    batches: int,
):
    """A loader of `batches` batches of `repo_ids` mixed by `weights`, the same batches at every call:
    streamed in this process (no workers to spawn) with the config's seed, and a small shuffle buffer,
    so a validation costs minutes, not a worker pool."""
    data = dataclasses.replace(
        config.data,
        repo_id=repo_ids,
        sampling_weights=weights,
        max_episodes={},
        streaming_shuffle_buffer_size=min(config.data.streaming_shuffle_buffer_size, 2000),
    )
    val_config = dataclasses.replace(config, data=data, num_workers=0)
    return _data_loader.create_data_loader(val_config, sharding=sharding, shuffle=True, num_batches=batches)


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # A resumed run (each preemption of a spot TPU's) goes on through its data from the step it resumes
    # at, rather than over the data it began with: a streamed dataset from where it was then.
    resumed_at = (checkpoint_manager.latest_step() or 0) if resuming else 0
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        start_step=resumed_at,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check (this host's first device's share of it: on a
    # multi-host slice the batch is not all here).
    local = [np.asarray(img.addressable_data(0)) for img in batch[0].images.values()]
    images_to_log = [
        wandb.Image(np.concatenate([img[i] for img in local], axis=1)) for i in range(min(5, len(local[0])))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    if config.val_interval:
        if not config.val_repo_ids:
            raise ValueError("val_interval is set but val_repo_ids names no repo to validate on.")
        pval_step = jax.jit(
            functools.partial(val_step, action_dims=config.val_action_dims),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(replicated_sharding, replicated_sharding),
        )
        val_rng = jax.random.key(config.seed + 1)

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        # A step's validation before its log and its checkpoint: a run preempted between them resumes
        # from the checkpoint before and does the step again, validation and all (saved first, a step
        # cut short there was never validated: 2026-10-28, abc_yam_v3's 10,000). A resumed run's logs to
        # steps W&B already has are dropped, so the validation goes with the step's training log.
        scores = {}
        if config.val_interval and (
            (step % config.val_interval == 0 and step > start_step) or step == config.num_train_steps - 1
        ):
            try:
                scores = validate(config, pval_step, val_rng, train_state, data_sharding)
            except Exception:  # a report: if it fails, the training goes on
                logging.exception("Step %d: the validation failed", step)
            else:
                pbar.write(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in scores.items()))
        if step % config.log_interval == 0 or scores:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log({**reduced_info, **scores}, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


def _entry() -> None:
    if os.environ.get(MULTIHOST_ENV):
        # Every host of a multi-host TPU slice runs this script; JAX joins them into one runtime.
        jax.distributed.initialize()
    main(_config.cli())


if __name__ == "__main__":
    _streaming_dataset.run_main(_entry)
