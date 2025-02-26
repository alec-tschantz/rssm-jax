from typing import Tuple, NamedTuple

import equinox as eqx
from jax import Array, numpy as jnp, random as jr, nn, lax


class State(NamedTuple):
    logits: Array
    deter: Array
    stoch: Array


class Model(eqx.Module):
    fc_act: eqx.nn.Linear
    fc_deter: eqx.nn.Linear
    fc_stoch: eqx.nn.Linear
    norm_act: eqx.nn.RMSNorm
    norm_deter: eqx.nn.RMSNorm
    norm_stoch: eqx.nn.RMSNorm
    gru_cell: eqx.nn.GRUCell

    norm_prior: eqx.nn.RMSNorm
    fc_prior: eqx.nn.Linear
    fc_logits_prior: eqx.nn.Linear

    norm_post: eqx.nn.RMSNorm
    fc_post: eqx.nn.Linear
    fc_logits_post: eqx.nn.Linear

    deter_dim: int
    num_discrete: int
    num_classes: int

    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        deter_dim: int,
        num_discrete: int,
        num_classes: int,
        hidden_dim: int,
        key: jr.PRNGKey,
    ):
        stoch_dim = num_discrete * num_classes
        keys = jr.split(key, 9)

        """ deterministic model """
        self.fc_act = eqx.nn.Linear(
            in_features=action_dim, out_features=hidden_dim, key=keys[2]
        )
        self.fc_deter = eqx.nn.Linear(
            in_features=deter_dim, out_features=hidden_dim, key=keys[0]
        )
        self.fc_stoch = eqx.nn.Linear(
            in_features=stoch_dim, out_features=hidden_dim, key=keys[1]
        )

        self.norm_act = eqx.nn.RMSNorm(shape=hidden_dim)
        self.norm_deter = eqx.nn.RMSNorm(shape=hidden_dim)
        self.norm_stoch = eqx.nn.RMSNorm(shape=hidden_dim)

        self.gru_cell = eqx.nn.GRUCell(
            input_size=hidden_dim * 3, hidden_size=deter_dim, key=keys[3]
        )

        """ prior model """
        self.fc_prior = eqx.nn.Linear(
            in_features=deter_dim, out_features=hidden_dim, key=keys[4]
        )
        self.fc_logits_prior = eqx.nn.Linear(
            in_features=hidden_dim, out_features=stoch_dim, key=keys[5]
        )
        self.norm_prior = eqx.nn.RMSNorm(shape=hidden_dim)

        """ posterior model """
        self.fc_post = eqx.nn.Linear(
            in_features=deter_dim + embed_dim, out_features=hidden_dim, key=keys[6]
        )
        self.fc_logits_post = eqx.nn.Linear(
            in_features=hidden_dim, out_features=stoch_dim, key=keys[7]
        )
        self.norm_post = eqx.nn.RMSNorm(shape=hidden_dim)

        self.deter_dim = deter_dim
        self.num_discrete = num_discrete
        self.num_classes = num_classes

    def __call__(self, deter: Array, stoch: Array, action: Array) -> Array:
        x0 = nn.silu(self.norm_deter(self.fc_deter(deter)))
        x1 = nn.silu(self.norm_stoch(self.fc_stoch(stoch)))
        x2 = nn.silu(self.norm_act(self.fc_act(action)))
        x = jnp.concatenate([x0, x1, x2], axis=-1)
        return self.gru_cell(x, deter)

    def prior(self, deter: Array) -> Array:
        x = nn.silu(self.norm_prior(self.fc_prior(deter)))
        return self._uniform_logits(self.fc_logits_prior(x))

    def posterior(self, deter: Array, embed: Array) -> Array:
        x = jnp.concatenate([deter, embed], axis=-1)
        x = nn.silu(self.norm_post(self.fc_post(x)))
        return self._uniform_logits(self.fc_logits_post(x))

    def rollout(
        self,
        init_state: Tuple[Array, Array],
        embed_seq: Array,
        action_seq: Array,
        key: jr.PRNGKey,
    ) -> Tuple[Array, Array, Array]:
        def step(carry, step_data):
            stoch, deter = carry
            k, embed, act = step_data
            deter = self(deter, stoch, act)
            prior_logits = self.prior(deter)
            post_logits = self.posterior(deter, embed)
            stoch = self._sample_logits(post_logits, k)
            return (stoch, deter), (post_logits, prior_logits, deter, stoch)

        keys = jr.split(key, action_seq.shape[0])
        _, outputs = lax.scan(step, init_state, (keys, embed_seq, action_seq))
        return outputs

    def rollout_prior(
        self,
        init_state: Tuple[Array, Array],
        action_seq: Array,
        key: jr.PRNGKey,
    ) -> Tuple[Array, Array, Array]:
        def step(carry, step_data):
            stoch, deter = carry
            k, act = step_data
            deter = self(deter, stoch, act)
            prior_logits = self.prior(deter)
            stoch = self._sample_logits(prior_logits, k)
            return (stoch, deter), (prior_logits, deter, stoch)

        keys = jr.split(key, action_seq.shape[0])
        _, outputs = lax.scan(step, init_state, (keys, action_seq))
        return outputs

    def init_state(self, batch_shape: Tuple = ()) -> Tuple[Array, Array]:
        stoch = jnp.zeros(batch_shape + (self.num_discrete * self.num_classes,))
        deter = jnp.zeros(batch_shape + (self.deter_dim,))
        return stoch, deter

    def _uniform_logits(
        self, logits: Array, uniform_mix: float = 0.01, eps: float = 1e-8
    ) -> Array:
        logits = logits.reshape(self.num_discrete, self.num_classes)
        probs = nn.softmax(logits, axis=-1)
        uniform = jnp.ones_like(probs) / self.num_classes
        probs = (1.0 - uniform_mix) * probs + uniform_mix * uniform
        return jnp.log(probs + eps).reshape(-1)

    def _sample_logits(self, logits: Array, key: jr.PRNGKey) -> Tuple[Array, Array]:
        logits = logits.reshape(self.num_discrete, self.num_classes)
        probs = nn.softmax(logits, axis=-1)

        sample = jr.categorical(key, logits, axis=-1)
        onehot = nn.one_hot(sample, self.num_classes)
        sample = onehot + (probs - lax.stop_gradient(probs))
        return sample.reshape(-1)
