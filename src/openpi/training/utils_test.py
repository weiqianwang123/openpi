import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0_config
from openpi.training import utils as training_utils


def test_validation_scores_are_the_loss_and_the_error_of_sampled_actions_over_the_robots_dims():
    """The loss is the model's (eval mode) with the rng given; the action error compares what the model
    samples, from the rng folded with 1, with the batch's actions over the first `action_dims` dims."""
    config = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(jax.random.key(0))
    obs, actions = config.fake_obs(2), config.fake_act(2)
    rng = jax.random.key(1)
    loss, error = training_utils.validation_scores(model, rng, obs, actions, action_dims=3)
    assert np.isclose(loss, jnp.mean(model.compute_loss(rng, obs, actions, train=False)))
    sampled = model.sample_actions(jax.random.fold_in(rng, 1), obs)
    assert np.isclose(error, jnp.mean(jnp.square(sampled - actions)[..., :3]))
    _, every = training_utils.validation_scores(model, rng, obs, actions)
    assert np.isclose(every, jnp.mean(jnp.square(sampled - actions)))
