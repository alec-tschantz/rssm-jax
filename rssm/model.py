from typing import Tuple, NamedTuple, Optional

import equinox as eqx
from jax import Array, numpy as jnp, random as jr, nn, lax, vmap


class State(NamedTuple):
    mean: Array
    std: Array
    sample: Array
    rnn_hidden: Array


class Prior(eqx.Module):
    rnn_cell: eqx.nn.GRUCell
    fc_input: eqx.nn.Linear
    fc_rnn_hidden: eqx.nn.Linear
    fc_out: eqx.nn.Linear

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        rnn_hidden_dim: int,
        mlp_hidden_dim: int,
        key: jr.PRNGKey,
    ):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.fc_input = eqx.nn.Linear(
            in_features=action_dim + state_dim, out_features=mlp_hidden_dim, key=k1
        )
        self.rnn_cell = eqx.nn.GRUCell(
            input_size=mlp_hidden_dim, hidden_size=rnn_hidden_dim, key=k2
        )
        self.fc_rnn_hidden = eqx.nn.Linear(
            in_features=rnn_hidden_dim, out_features=mlp_hidden_dim, key=k3
        )
        self.fc_out = eqx.nn.Linear(
            in_features=mlp_hidden_dim, out_features=2 * state_dim, key=k4
        )

    def __call__(self, prev_post: State, action: Array, key: jr.PRNGKey) -> State:
        inp = jnp.concatenate([action, prev_post.sample], axis=-1)
        h = nn.elu(self.fc_input(inp))
        new_rnn_hidden = self.rnn_cell(h, prev_post.rnn_hidden)
        h2 = nn.elu(self.fc_rnn_hidden(new_rnn_hidden))
        out = self.fc_out(h2)
        mean, std = jnp.split(out, 2, axis=-1)
        std = nn.softplus(std) + 0.1
        new_sample = sample_normal(key, mean, std)
        return State(mean, std, new_sample, new_rnn_hidden)


class Posterior(eqx.Module):
    fc_in: eqx.nn.Linear
    fc_out: eqx.nn.Linear

    def __init__(
        self,
        obs_emb_dim: int,
        state_dim: int,
        rnn_hidden_dim: int,
        mlp_hidden_dim: int,
        key: jr.PRNGKey,
    ):
        k1, k2 = jr.split(key, 2)
        self.fc_in = eqx.nn.Linear(
            in_features=rnn_hidden_dim + obs_emb_dim,
            out_features=mlp_hidden_dim,
            key=k1,
        )
        self.fc_out = eqx.nn.Linear(
            in_features=mlp_hidden_dim, out_features=2 * state_dim, key=k2
        )

    def __call__(self, obs_emb: Array, prior: State, key: jr.PRNGKey) -> State:
        x = jnp.concatenate([prior.rnn_hidden, obs_emb], axis=-1)
        h = nn.elu(self.fc_in(x))
        out = self.fc_out(h)
        mean, std = jnp.split(out, 2, axis=-1)
        std = nn.softplus(std) + 0.1
        new_sample = sample_normal(key, mean, std)
        return State(mean, std, new_sample, prior.rnn_hidden)


class Model(eqx.Module):
    prior: Prior
    posterior: Posterior
    state_dim: int
    rnn_hidden_dim: int

    def __init__(
        self,
        obs_emb_dim: int,
        action_dim: int,
        state_dim: int,
        rnn_hidden_dim: int,
        mlp_hidden_dim: int,
        key: jr.PRNGKey,
    ):
        k1, k2 = jr.split(key, 2)
        self.prior = Prior(action_dim, state_dim, rnn_hidden_dim, mlp_hidden_dim, k1)
        self.posterior = Posterior(
            obs_emb_dim, state_dim, rnn_hidden_dim, mlp_hidden_dim, k2
        )
        self.state_dim = state_dim
        self.rnn_hidden_dim = rnn_hidden_dim

    def __call__(
        self, obs_emb: Array, prev_post: State, prev_action: Array, key: jr.PRNGKey
    ) -> Tuple[State, State]:
        k1, k2 = jr.split(key, 2)
        prior_s = self.prior(prev_post, prev_action, k1)
        post_s = self.posterior(obs_emb, prior_s, k2)
        return post_s, prior_s

    def rollout(
        self, obs_emb_seq: Array, init_post: State, action_seq: Array, key: jr.PRNGKey
    ) -> Tuple[State, State]:
        def step(prev_post, step_data):
            k_, ob_, act_ = step_data
            post_s, prior_s = self(ob_, prev_post, act_, k_)
            return post_s, (post_s, prior_s)

        T = action_seq.shape[0]
        keys = jr.split(key, T)
        _, (post_seq, prior_seq) = lax.scan(
            step, init_post, (keys, obs_emb_seq, action_seq)
        )
        return post_seq, prior_seq

    def rollout_prior(
        self, init_post: State, action_seq: Array, key: jr.PRNGKey
    ) -> State:
        def step(prev_s, step_data):
            k_, act_ = step_data
            new_s = self.prior(prev_s, act_, k_)
            return new_s, new_s

        T = action_seq.shape[0]
        keys = jr.split(key, T)
        _, states = lax.scan(step, init_post, (keys, action_seq))
        return states

    def init_post(self, batch_shape: tuple = ()) -> State:
        return State(
            jnp.zeros(batch_shape + (self.state_dim,)),
            jnp.ones(batch_shape + (self.state_dim,)),
            jnp.zeros(batch_shape + (self.state_dim,)),
            jnp.zeros(batch_shape + (self.rnn_hidden_dim,)),
        )


class Encoder(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(
        self, obs_dim: int, obs_embed_dim: int, mlp_hidden_dim: int, key: jr.PRNGKey
    ):
        k1, k2 = jr.split(key, 2)
        self.fc1 = eqx.nn.Linear(
            in_features=obs_dim, out_features=mlp_hidden_dim, key=k1
        )
        self.fc2 = eqx.nn.Linear(
            in_features=mlp_hidden_dim, out_features=obs_embed_dim, key=k2
        )

    def __call__(self, obs: Array) -> Array:
        h = nn.elu(self.fc1(obs))
        return self.fc2(h)


class Decoder(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(
        self,
        state_dim: int,
        rnn_hidden_dim: int,
        obs_dim: int,
        mlp_hidden_dim: int,
        key: jr.PRNGKey,
    ):
        k1, k2 = jr.split(key, 2)
        self.fc1 = eqx.nn.Linear(
            in_features=state_dim + rnn_hidden_dim, out_features=mlp_hidden_dim, key=k1
        )
        self.fc2 = eqx.nn.Linear(in_features=mlp_hidden_dim, out_features=obs_dim, key=k2)

    def __call__(self, post: Array) -> Array:
        inp = jnp.concatenate([post.sample, post.rnn_hidden], axis=-1)
        h = nn.elu(self.fc1(inp))
        return self.fc2(h)


def rssm_loss(
    params: Tuple[Model, Encoder, Decoder],
    obs_seq: Array,
    action_seq: Array,
    key: jr.PRNGKey,
) -> Tuple[Array, Array]:
    model, encoder, decoder = params
    B, T, D_obs = obs_seq.shape

    obs_seq_flat = obs_seq.reshape((B * T, D_obs))
    obs_emb_flat = vmap(encoder)(obs_seq_flat)
    obs_emb = obs_emb_flat.reshape((B, T, -1))

    keys = jr.split(key, B)
    init_post = model.init_post((B,))
    post_seq, prior_seq = vmap(model.rollout)(obs_emb, init_post, action_seq, keys)
    out_seq = vmap(vmap(decoder))(post_seq)

    obs_loss = mse_loss(model, out_seq, obs_seq)
    kl_loss = kl_divergence(post_seq, prior_seq).mean()
    return obs_loss, kl_loss


def mse_loss(model: Model, out_seq: Array, obs_seq: Array) -> Array:
    return jnp.mean(jnp.sum((out_seq - obs_seq) ** 2, axis=-1))


def sample_normal(key: jr.PRNGKey, mean: Array, std: Array) -> Array:
    return mean + std * jr.normal(key, mean.shape)


def kl_divergence(post: State, prior: State) -> Array:
    kl = (
        jnp.log(prior.std / post.std)
        + (post.std**2 + (post.mean - prior.mean) ** 2) / (2 * prior.std**2)
        - 0.5
    )
    return jnp.sum(kl, axis=-1)
