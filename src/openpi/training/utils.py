from collections.abc import Callable
from typing import Any

from flax import nnx
from flax import struct
import jax
import jax.numpy as jnp
import optax

from openpi.models import model as _model
from openpi.shared import array_typing as at


@at.typecheck
@struct.dataclass
class TrainState:
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State
    model_def: nnx.GraphDef[_model.BaseModel]
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    ema_decay: float | None = struct.field(pytree_node=False)
    ema_params: nnx.State | None = None


def validation_scores(
    model: _model.BaseModel,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    actions: _model.Actions,
    action_dims: int | None = None,
) -> tuple[at.Array, at.Array]:
    """A batch's scores for `model` (in eval mode): its mean loss, and its action error: the mean squared
    error of the actions it samples in full (its sampler's flow steps from noise, as served) against the
    batch's, in the normalized space it is trained in, over the first `action_dims` dims (the robot's;
    None: all, padding included). The loss draws its noise from `rng`, the sampler from its fold with 1."""
    model.eval()
    loss = jnp.mean(model.compute_loss(rng, observation, actions, train=False))
    sampled = model.sample_actions(jax.random.fold_in(rng, 1), observation)
    return loss, jnp.mean(jnp.square(sampled - actions)[..., :action_dims])


@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    """Converts a PyTree into a human-readable string for logging. Optionally, `interp_func` can be provided to convert
    the leaf values to more meaningful strings.
    """
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    """Converts a PyTree of arrays into a human-readable string for logging."""
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")
