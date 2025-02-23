from typing import NamedTuple, Tuple

import equinox as eqx
from jax import Array, numpy as jnp, random as jr, nn, lax, vmap


class Prior(eqx.Module):
    rnn_cell: eqx.nn.GRUCell
    fc_input: eqx.nn.Linear
    fc_state: eqx.nn.Linear
    fc_latent: eqx.nn.Linear


class Posterior(eqx.Module):
    fc_input: eqx.nn.Linear
    fc_latent: eqx.nn.Linear


class Encoder(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear


class Decoder(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear


class Model(eqx.Module):
    prior: Prior
    posterior: Posterior
    encoder: Encoder
    decoder: Decoder
    latent_dim: int
    state_dim: int


class State(NamedTuple):
    mean: Array
    std: Array
    latent: Array
    state: Array


def forward_prior(
    prior: Prior,
    prev_post: State,
    action: Array,
    key: jr.PRNGKey,
) -> State:
    feat = jnp.concatenate([action, prev_post.latent], axis=-1)
    hidden = nn.elu(prior.fc_input(feat))

    state = prior.rnn_cell(hidden, prev_post.state)
    hidden = nn.elu(prior.fc_state(state))

    latent = prior.fc_latent(hidden)
    mean, std, sample = forward_normal(latent, key)
    return State(mean, std, sample, state)


def forward_posterior(
    posterior: Posterior,
    obs_emb: Array,
    prior_state: State,
    key: jr.PRNGKey,
) -> State:
    inp = jnp.concatenate([prior_state.state, obs_emb], axis=-1)
    hidden = nn.elu(posterior.fc_input(inp))

    latent = posterior.fc_latent(hidden)
    mean, std, sample = forward_normal(latent, key)
    return State(mean, std, sample, prior_state.state)


def forward_model(
    model: Model,
    obs_seq: Array,
    action_seq: Array,
    key: jr.PRNGKey,
) -> Tuple[Array, State, State]:

    obs_emb_seq = vmap(lambda o: forward_encoder(model.encoder, o))(obs_seq)
    init_post = init_post_state(model)
    post_seq, prior_seq = rollout_dynamics(
        model.prior, model.posterior, obs_emb_seq, init_post, action_seq, key
    )
    out_seq = vmap(lambda s: forward_decoder(model.decoder, s))(post_seq)
    return out_seq, post_seq, prior_seq


def forward_dynamics(
    prior: Prior,
    post: Posterior,
    obs: Array,
    prev_post: State,
    action: Array,
    key: jr.PRNGKey,
) -> State:
    keys = jr.split(key, 2)
    prior_ = forward_prior(prior, prev_post, action, keys[0])
    post_ = forward_posterior(post, obs, prior_, keys[1])
    return post_, prior_


def forward_encoder(encoder: Encoder, obs: Array) -> Array:
    hidden = nn.elu(encoder.fc1(obs))
    return encoder.fc2(hidden)


def forward_decoder(decoder: Decoder, post: State) -> Array:
    inp = jnp.concatenate([post.latent, post.state], axis=-1)
    hidden = nn.elu(decoder.fc1(inp))
    return decoder.fc2(hidden)


def forward_normal(out: Array, key: jr.PRNGKey) -> Tuple[Array, Array, Array]:
    mean, std = jnp.split(out, 2, axis=-1)
    std = nn.softplus(std) + 0.1
    sample = mean + std * jr.normal(key, mean.shape)
    return mean, std, sample


def rollout_dynamics(
    prior: Prior,
    post: Posterior,
    obs_emb_seq: Array,
    init_post: State,
    action_seq: Array,
    key: jr.PRNGKey,
) -> Tuple[State, State]:
    def step(prev_post, step):
        k_, ob_, act_ = step
        post_, prior_ = forward_dynamics(prior, post, ob_, prev_post, act_, k_)
        return post_, (post_, prior_)

    keys = jr.split(key, action_seq.shape[0])
    final_post, (post_seq, prior_seq) = lax.scan(
        step,
        init_post,
        (keys, obs_emb_seq, action_seq),
    )
    return post_seq, prior_seq


def rollout_dynamics_prior(
    prior: Prior,
    init_post: State,
    action_seq: Array,
    key: jr.PRNGKey,
) -> State:
    def step(prev_s, step):
        k_, act_ = step
        new_s = forward_prior(prior, prev_s, act_, k_)
        return new_s, new_s

    keys = jr.split(key, action_seq.shape[0])
    _, states = lax.scan(step, init_post, (keys, action_seq))
    return states


def init_post_state(model: Model, batch_shape: tuple = ()) -> State:
    return State(
        mean=jnp.zeros(batch_shape + (model.latent_dim,)),
        std=jnp.ones(batch_shape + (model.latent_dim,)),
        latent=jnp.zeros(batch_shape + (model.latent_dim,)),
        state=jnp.zeros(batch_shape + (model.state_dim,)),
    )
