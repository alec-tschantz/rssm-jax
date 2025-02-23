from typing import NamedTuple, Tuple

import equinox as eqx
from jax import Array, numpy as jnp, random as jr, nn, lax, vmap


class Prior(eqx.Module):
    rnn_cell: eqx.nn.GRUCell
    fc_input: eqx.nn.Linear
    fc_rnn_hidden: eqx.nn.Linear
    fc_output: eqx.nn.Linear


class Posterior(eqx.Module):
    fc_input: eqx.nn.Linear
    fc_output: eqx.nn.Linear


class Model(eqx.Module):
    prior: Prior
    posterior: Posterior
    state_dim: int
    rnn_hidden_dim: int


class Encoder(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear


class Decoder(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear


class State(NamedTuple):
    mean: Array
    std: Array
    sample: Array
    rnn_hidden: Array


def forward_prior(
    prior: Prior,
    prev_post: State,
    action: Array,
    key: jr.PRNGKey,
) -> State:

    feat = jnp.concatenate([action, prev_post.sample], axis=-1)

    hidden = nn.elu(prior.fc_input(feat))
    new_rnn_hidden = prior.rnn_cell(hidden, prev_post.rnn_hidden)
    hidden = nn.elu(prior.fc_rnn_hidden(new_rnn_hidden))

    out = prior.fc_output(hidden)
    mean, std, sample = forward_normal(out, key)
    return State(mean, std, sample, new_rnn_hidden)


def forward_posterior(
    posterior: Posterior,
    obs_emb: Array,
    prior_state: State,
    key: jr.PRNGKey,
) -> State:
    inp = jnp.concatenate([prior_state.rnn_hidden, obs_emb], axis=-1)

    hidden = nn.elu(posterior.fc_input(inp))
    out = posterior.fc_output(hidden)

    mean, std, sample = forward_normal(out, key)
    return State(mean, std, sample, prior_state.rnn_hidden)


def forward_model(
    model: Model,
    obs_emb: Array,
    prev_post: State,
    prev_action: Array,
    key: jr.PRNGKey,
) -> Tuple[State, State]:
    k1, k2 = jr.split(key, 2)
    prior_s = forward_prior(model.prior, prev_post, prev_action, k1)
    post_s = forward_posterior(model.posterior, obs_emb, prior_s, k2)
    return post_s, prior_s


def forward_normal(out: Array, key: jr.PRNGKey) -> Tuple[Array, Array, Array]:
    mean, std = jnp.split(out, 2, axis=-1)
    std = nn.softplus(std) + 0.1
    sample = sample_normal(key, mean, std)
    return mean, std, sample


def forward_encoder(encoder: Encoder, obs: Array) -> Array:
    hidden = nn.elu(encoder.fc1(obs))
    return encoder.fc2(hidden)


def forward_decoder(decoder: Decoder, post_state: State) -> Array:
    inp = jnp.concatenate([post_state.sample, post_state.rnn_hidden], axis=-1)
    hidden = nn.elu(decoder.fc1(inp))
    return decoder.fc2(hidden)


def rollout_model(
    model: Model,
    obs_emb_seq: Array,
    init_post: State,
    action_seq: Array,
    key: jr.PRNGKey,
) -> Tuple[State, State]:

    def step(prev_post, step_data):
        k_, ob_, act_ = step_data
        post_s, prior_s = forward_model(model, ob_, prev_post, act_, k_)
        return post_s, (post_s, prior_s)

    T = action_seq.shape[0]
    keys = jr.split(key, T)
    final_post, (post_seq, prior_seq) = lax.scan(
        step,
        init_post,
        (keys, obs_emb_seq, action_seq),
    )
    return post_seq, prior_seq


def rollout_model_prior(
    model: Model,
    init_post: State,
    action_seq: Array,
    key: jr.PRNGKey,
) -> State:
    def step(prev_s, step_data):
        k_, act_ = step_data
        new_s = forward_prior(model.prior, prev_s, act_, k_)
        return new_s, new_s

    T = action_seq.shape[0]
    keys = jr.split(key, T)
    _, states = lax.scan(step, init_post, (keys, action_seq))
    return states


def rssm_loss(
    params: Tuple[Model, Encoder, Decoder],
    obs_seq: Array,
    action_seq: Array,
    key: jr.PRNGKey,
) -> Tuple[Array, Array]:
    model, encoder, decoder = params
    B, T, _ = obs_seq.shape

    obs_emb_seq = vmap(vmap(lambda o: forward_encoder(encoder, o)))(obs_seq)
    init_post = init_post_state(model, batch_shape=(B,))
    rollout_fn = lambda obs_emb, post, act, k: rollout_model(
        model, obs_emb, post, act, k
    )
    post_seq, prior_seq = vmap(rollout_fn)(
        obs_emb_seq, init_post, action_seq, jr.split(key, B)
    )
    out_seq = vmap(vmap(lambda s: forward_decoder(decoder, s)))(post_seq)

    obs_loss = mse_loss(out_seq, obs_seq)
    kl_loss = kl_divergence(post_seq, prior_seq).mean()

    return obs_loss, kl_loss


def init_prior(
    action_dim: int,
    state_dim: int,
    rnn_hidden_dim: int,
    mlp_hidden_dim: int,
    key: jr.PRNGKey,
) -> Prior:
    k1, k2, k3, k4 = jr.split(key, 4)
    return Prior(
        fc_input=eqx.nn.Linear(
            in_features=action_dim + state_dim,
            out_features=mlp_hidden_dim,
            key=k1,
        ),
        rnn_cell=eqx.nn.GRUCell(
            input_size=mlp_hidden_dim,
            hidden_size=rnn_hidden_dim,
            key=k2,
        ),
        fc_rnn_hidden=eqx.nn.Linear(
            in_features=rnn_hidden_dim,
            out_features=mlp_hidden_dim,
            key=k3,
        ),
        fc_output=eqx.nn.Linear(
            in_features=mlp_hidden_dim,
            out_features=2 * state_dim,
            key=k4,
        ),
    )


def init_posterior(
    obs_emb_dim: int,
    state_dim: int,
    rnn_hidden_dim: int,
    mlp_hidden_dim: int,
    key: jr.PRNGKey,
) -> Posterior:
    k1, k2 = jr.split(key, 2)
    return Posterior(
        fc_input=eqx.nn.Linear(
            in_features=rnn_hidden_dim + obs_emb_dim,
            out_features=mlp_hidden_dim,
            key=k1,
        ),
        fc_output=eqx.nn.Linear(
            in_features=mlp_hidden_dim,
            out_features=2 * state_dim,
            key=k2,
        ),
    )


def init_model(
    obs_emb_dim: int,
    action_dim: int,
    state_dim: int,
    rnn_hidden_dim: int,
    mlp_hidden_dim: int,
    key: jr.PRNGKey,
) -> Model:
    k1, k2 = jr.split(key, 2)
    return Model(
        prior=init_prior(action_dim, state_dim, rnn_hidden_dim, mlp_hidden_dim, k1),
        posterior=init_posterior(
            obs_emb_dim, state_dim, rnn_hidden_dim, mlp_hidden_dim, k2
        ),
        state_dim=state_dim,
        rnn_hidden_dim=rnn_hidden_dim,
    )


def init_encoder(
    obs_dim: int,
    obs_embed_dim: int,
    mlp_hidden_dim: int,
    key: jr.PRNGKey,
) -> Encoder:
    k1, k2 = jr.split(key, 2)
    return Encoder(
        fc1=eqx.nn.Linear(in_features=obs_dim, out_features=mlp_hidden_dim, key=k1),
        fc2=eqx.nn.Linear(
            in_features=mlp_hidden_dim, out_features=obs_embed_dim, key=k2
        ),
    )


def init_decoder(
    state_dim: int,
    rnn_hidden_dim: int,
    obs_dim: int,
    mlp_hidden_dim: int,
    key: jr.PRNGKey,
) -> Decoder:
    k1, k2 = jr.split(key, 2)
    return Decoder(
        fc1=eqx.nn.Linear(
            in_features=state_dim + rnn_hidden_dim,
            out_features=mlp_hidden_dim,
            key=k1,
        ),
        fc2=eqx.nn.Linear(
            in_features=mlp_hidden_dim,
            out_features=obs_dim,
            key=k2,
        ),
    )


def init_post_state(model: Model, batch_shape: tuple = ()) -> State:
    return State(
        mean=jnp.zeros(batch_shape + (model.state_dim,)),
        std=jnp.ones(batch_shape + (model.state_dim,)),
        sample=jnp.zeros(batch_shape + (model.state_dim,)),
        rnn_hidden=jnp.zeros(batch_shape + (model.rnn_hidden_dim,)),
    )


def sample_normal(key: jr.PRNGKey, mean: Array, std: Array) -> Array:
    return mean + std * jr.normal(key, mean.shape)


def kl_divergence(post: State, prior: State) -> Array:
    kl = (
        jnp.log(prior.std / post.std)
        + (post.std**2 + (post.mean - prior.mean) ** 2) / (2 * prior.std**2)
        - 0.5
    )
    return jnp.sum(kl, axis=-1)


def mse_loss(out_seq: Array, obs_seq: Array) -> Array:
    return jnp.mean(jnp.sum((out_seq - obs_seq) ** 2, axis=-1))
