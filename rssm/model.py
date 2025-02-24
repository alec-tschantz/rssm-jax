import optax
import equinox as eqx
from typing import NamedTuple, Tuple
from jax import Array, numpy as jnp, random as jr, nn, lax, vmap


class Prior(eqx.Module):
    rnn_cell: eqx.nn.GRUCell
    fc_input: eqx.nn.Linear
    fc_state: eqx.nn.Linear
    fc_logits: eqx.nn.Linear
    norm_input: eqx.nn.RMSNorm
    norm_state: eqx.nn.RMSNorm
    num_discrete: int
    discrete_dim: int


class Posterior(eqx.Module):
    fc_input: eqx.nn.Linear
    fc_logits: eqx.nn.Linear
    norm_input: eqx.nn.RMSNorm
    num_discrete: int
    discrete_dim: int


class Encoder(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    norm1: eqx.nn.RMSNorm
    norm2: eqx.nn.RMSNorm


class Decoder(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    norm1: eqx.nn.RMSNorm
    norm2: eqx.nn.RMSNorm


class Model(eqx.Module):
    prior: Prior
    posterior: Posterior
    encoder: Encoder
    decoder: Decoder
    logit_dim: int
    state_dim: int


class State(NamedTuple):
    logits: Array
    sample: Array
    state: Array


def forward_prior(
    prior: Prior, prev_post: State, action: Array, key: jr.PRNGKey
) -> State:
    prev_sample = prev_post.sample.reshape(-1)
    feat = jnp.concatenate([action, prev_sample], axis=-1)
    hidden = prior.norm_input(nn.silu(prior.fc_input(feat)))
    state = prior.rnn_cell(hidden, prev_post.state)
    hidden = prior.norm_state(nn.silu(prior.fc_state(state)))
    logits = prior.fc_logits(hidden).reshape(prior.num_discrete, prior.discrete_dim)
    logits, sample = sample_logits(logits, key)
    return State(logits, sample, state)


def forward_posterior(
    post: Posterior, obs_emb: Array, prior_state: State, key: jr.PRNGKey
) -> State:
    inp = jnp.concatenate([obs_emb, prior_state.state], axis=-1)
    hidden = post.norm_input(nn.silu(post.fc_input(inp)))
    logits = post.fc_logits(hidden).reshape(post.num_discrete, post.discrete_dim)
    logits, sample = sample_logits(logits, key)
    return State(logits, sample, prior_state.state)


def forward_encoder(encoder: Encoder, obs: Array) -> Array:
    hidden = encoder.norm1(nn.silu(encoder.fc1(obs)))
    out = encoder.fc2(hidden)
    out = encoder.norm2(out)
    return out


def forward_decoder(decoder: Decoder, post: State) -> Array:
    inp = jnp.concatenate([post.sample.reshape(-1), post.state], axis=-1)
    hidden = decoder.norm1(nn.silu(decoder.fc1(inp)))
    out = decoder.fc2(hidden)
    out = decoder.norm2(out)
    return out


def forward_model(
    model: Model, obs_seq: Array, action_seq: Array, key: jr.PRNGKey
) -> Tuple[Array, State, State]:
    obs_emb_seq = vmap(lambda o: forward_encoder(model.encoder, o))(obs_seq)
    init_post = init_post_state(model)
    post_seq, prior_seq = rollout(
        model.prior, model.posterior, obs_emb_seq, init_post, action_seq, key
    )
    out_seq = vmap(lambda s: forward_decoder(model.decoder, s))(post_seq)
    return out_seq, post_seq, prior_seq


def rollout(
    prior: Prior,
    post: Posterior,
    obs_emb_seq: Array,
    init_post: State,
    action_seq: Array,
    key: jr.PRNGKey,
) -> Tuple[Array, Array]:
    def step(prev_post, step_data):
        k_, ob_, act_ = step_data
        keys = jr.split(k_, 2)
        prior_ = forward_prior(prior, prev_post, act_, keys[0])
        post_ = forward_posterior(post, ob_, prior_, keys[1])
        return post_, (post_, prior_)

    keys = jr.split(key, action_seq.shape[0])
    final_post, (post_seq, prior_seq) = lax.scan(
        step, init_post, (keys, obs_emb_seq, action_seq)
    )
    return post_seq, prior_seq


def rollout_prior(
    prior: Prior, init_post: State, action_seq: Array, key: jr.PRNGKey
) -> Array:
    def step(prev_s, step_data):
        k_, act_ = step_data
        new_s = forward_prior(prior, prev_s, act_, k_)
        return new_s, new_s

    keys = jr.split(key, action_seq.shape[0])
    _, states = lax.scan(step, init_post, (keys, action_seq))
    return states


def sample_logits(
    logits: Array, key: jr.PRNGKey, unimix: float = 0.01
) -> Tuple[Array, Array]:
    probs = nn.softmax(logits, axis=-1)
    uniform = jnp.ones_like(probs) / probs.shape[-1]
    probs = (1.0 - unimix) * probs + unimix * uniform
    dist_logits = jnp.log(probs + 1e-8)
    sample = jr.categorical(key, dist_logits, axis=-1)
    onehot = nn.one_hot(sample, probs.shape[-1])
    st_sample = onehot + (probs - lax.stop_gradient(probs))
    return dist_logits, st_sample


def init_post_state(model: Model, batch_shape: tuple = ()) -> State:
    post = model.posterior
    return State(
        logits=jnp.zeros(batch_shape + (post.num_discrete, post.discrete_dim)),
        sample=jnp.zeros(batch_shape + (post.num_discrete, post.discrete_dim)),
        state=jnp.zeros(batch_shape + (model.state_dim,)),
    )
