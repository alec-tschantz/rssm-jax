from typing import Tuple, NamedTuple

import equinox as eqx
from jax import Array, numpy as jnp, random as jr, nn, lax, vmap


class State(NamedTuple):
    mean: Array
    std: Array
    stoch: Array
    deter: Array


class Prior(eqx.Module):
    cell: eqx.nn.GRUCell
    input_proj: eqx.nn.Linear
    stoch_proj1: eqx.nn.Linear
    stoch_proj2: eqx.nn.Linear

    def __init__(
        self,
        action_size: int,
        embed_size: int,
        stoch_size: int,
        deter_size: int,
        mlp_hidden_size: int,
        key: jr.PRNGKey,
    ):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.input_proj = eqx.nn.Linear(
            in_features=action_size + stoch_size, out_features=embed_size, key=k1
        )
        self.cell = eqx.nn.GRUCell(
            input_size=embed_size, hidden_size=deter_size, key=k2
        )
        self.stoch_proj1 = eqx.nn.Linear(
            in_features=deter_size, out_features=mlp_hidden_size, key=k3
        )
        self.stoch_proj2 = eqx.nn.Linear(
            in_features=mlp_hidden_size, out_features=2 * stoch_size, key=k4
        )

    def __call__(self, state: State, action: Array, key: jr.PRNGKey) -> State:
        inp = jnp.concatenate([action, state.stoch], axis=-1)  # (B, A + S)
        hidden = nn.elu(self.input_proj(inp))  # (B, E)

        new_deter = self.cell(hidden, state.deter)  # (B, D)

        out = nn.elu(self.stoch_proj1(new_deter))  # (B, H)
        out = self.stoch_proj2(out)  # (B, 2 * S)

        mean, std = jnp.split(out, 2, axis=-1)  # (B, S), (B, S)
        std = nn.softplus(std) + 0.1
        stoch = sample_normal(key, mean, std)
        return State(mean, std, stoch, new_deter)


class Posterior(eqx.Module):
    post_proj1: eqx.nn.Linear
    post_proj2: eqx.nn.Linear

    def __init__(
        self,
        obs_size: int,
        stoch_size: int,
        deter_size: int,
        mlp_hidden_size: int,
        key: jr.PRNGKey,
    ):
        k1, k2 = jr.split(key, 2)
        self.post_proj1 = eqx.nn.Linear(
            in_features=deter_size + obs_size, out_features=mlp_hidden_size, key=k1
        )
        self.post_proj2 = eqx.nn.Linear(
            in_features=mlp_hidden_size, out_features=2 * stoch_size, key=k2
        )

    def __call__(self, obs: Array, prior: State, key: jr.PRNGKey) -> State:
        inp = jnp.concatenate([prior.deter, obs], axis=-1)  # (B, D + O)
        hidden = nn.elu(self.post_proj1(inp))  # (B, H)
        out = self.post_proj2(hidden)  # (B, 2 * S)

        mean, std = jnp.split(out, 2, axis=-1)  # (B, S), (B, S)
        std = nn.softplus(std) + 0.1
        stoch = sample_normal(key, mean, std)
        return State(mean, std, stoch, prior.deter)


class Model(eqx.Module):
    prior: Prior
    posterior: Posterior
    output_proj: eqx.nn.Linear

    stoch_size: int
    deter_size: int

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        stoch_size: int,
        deter_size: int,
        embed_size: int,
        mlp_hidden_size: int,
        key: jr.PRNGKey,
    ):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.prior = Prior(
            action_size, embed_size, stoch_size, deter_size, mlp_hidden_size, k1
        )
        self.posterior = Posterior(
            obs_size, stoch_size, deter_size, mlp_hidden_size, k2
        )
        self.output_proj = eqx.nn.Linear(
            in_features=stoch_size + deter_size, out_features=obs_size, key=k4
        )

        self.stoch_size = stoch_size
        self.deter_size = deter_size

    def __call__(
        self, obs: Array, state: State, action: Array, key: jr.PRNGKey
    ) -> Tuple[State, State]:
        keys = jr.split(key, 2)
        prior = self.prior(state, action, keys[0])
        post = self.posterior(obs, prior, keys[1])
        return post, prior

    def rollout_prior(
        self, init_state: State, action_seq: Array, key: jr.PRNGKey
    ) -> State:
        def step(state, step):
            key, action = step
            new_state = self.prior(state, action, key)
            return new_state, new_state

        T, _ = action_seq.shape
        keys = jr.split(key, T)
        _, states = lax.scan(step, init_state, (keys, action_seq))
        return states

    def rollout_posterior(
        self, obs_seq: Array, init_state: State, action_seq: Array, key: jr.PRNGKey
    ) -> Tuple[State, State]:
        def step(state, step):
            key, obs, action = step
            post, prior = self(obs, state, action, key)
            return post, (post, prior)

        T, _ = action_seq.shape
        keys = jr.split(key, T)
        _, (post_seq, prior_seq) = lax.scan(
            step, init_state, (keys, obs_seq, action_seq)
        )
        return post_seq, prior_seq

    def initial_state(self, B: int) -> State:
        return State(
            jnp.zeros((B, self.stoch_size)),
            jnp.zeros((B, self.stoch_size)),
            jnp.zeros((B, self.stoch_size)),
            jnp.zeros((B, self.deter_size)),
        )


def rssm_loss(
    model: Model, obs_seq: Array, action_seq: Array, key: jr.PRNGKey
) -> Tuple[Array, Array]:
    B, _, _ = obs_seq.shape
    init_state = model.initial_state(B)

    keys = jr.split(key, B)
    post_seq, prior_seq = vmap(model.rollout_posterior)(
        obs_seq, init_state, action_seq, keys
    )

    feature_seq = jnp.concatenate([post_seq.stoch, post_seq.deter], axis=-1)
    out_seq = vmap(vmap(model.output_proj))(feature_seq)

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
