import optax
import equinox as eqx
from jax import Array, numpy as jnp, random as jr, nn, lax, vmap

from rssm import forward_model


@eqx.filter_jit
def step(params, obs_seq, action_seq, optimizer, opt_state, key):
    def loss_fn(params):
        _forward = lambda o, a, k: forward_model(params, o, a, k)
        subkeys = jr.split(key, obs_seq.shape[0])
        pred_seq, post, prior = vmap(_forward)(obs_seq, action_seq, subkeys)
        return mse_loss(obs_seq, pred_seq) + kl_loss(post.logits, prior.logits)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss


def kl_loss(
    prior_logits: Array, post_logits: Array, free_nats: float = 0.0, alpha: float = 0.8
) -> Array:
    kl_lhs = optax.losses.kl_divergence_with_log_targets(
        lax.stop_gradient(post_logits), prior_logits
    ).sum(axis=-1)
    kl_rhs = optax.losses.kl_divergence_with_log_targets(
        post_logits, lax.stop_gradient(prior_logits)
    ).sum(axis=-1)

    kl_lhs, kl_rhs = jnp.mean(kl_lhs), jnp.mean(kl_rhs)
    if free_nats > 0.0:
        kl_lhs = jnp.maximum(kl_lhs, free_nats)
        kl_rhs = jnp.maximum(kl_rhs, free_nats)
    return (alpha * kl_lhs) + ((1 - alpha) * kl_rhs)


def mse_loss(out_seq: Array, obs_seq: Array) -> Array:
    return jnp.mean(jnp.sum((out_seq - obs_seq) ** 2, axis=-1))
