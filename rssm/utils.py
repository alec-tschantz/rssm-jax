import optax
import equinox as eqx
from jax import Array, numpy as jnp, random as jr, nn, lax, vmap

from rssm.model import Model, Prior, Posterior, Encoder, Decoder, State


def init_model(
    obs_dim: int,
    action_dim: int,
    embed_dim: int,
    state_dim: int,
    num_discrete: int,
    discrete_dim: int,
    hidden_dim: int,
    key: jr.PRNGKey,
) -> Model:

    k1, k2, k3, k4 = jr.split(key, 4)
    return Model(
        prior=init_prior(
            action_dim, num_discrete, discrete_dim, state_dim, hidden_dim, k1
        ),
        posterior=init_posterior(
            embed_dim, num_discrete, discrete_dim, state_dim, hidden_dim, k2
        ),
        encoder=init_encoder(obs_dim, embed_dim, hidden_dim, k3),
        decoder=init_decoder(
            num_discrete, discrete_dim, state_dim, obs_dim, hidden_dim, k4
        ),
        logit_dim=num_discrete * discrete_dim,
        state_dim=state_dim,
    )


def init_prior(
    action_dim: int,
    num_discrete: int,
    discrete_dim: int,
    state_dim: int,
    hidden_dim: int,
    key: jr.PRNGKey,
) -> Prior:

    k1, k2, k3, k4 = jr.split(key, 4)
    logit_dim = num_discrete * discrete_dim
    return Prior(
        fc_input=eqx.nn.Linear(
            in_features=action_dim + logit_dim,
            out_features=hidden_dim,
            key=k1,
        ),
        rnn_cell=eqx.nn.GRUCell(
            input_size=hidden_dim,
            hidden_size=state_dim,
            key=k2,
        ),
        fc_state=eqx.nn.Linear(
            in_features=state_dim,
            out_features=hidden_dim,
            key=k3,
        ),
        fc_logits=eqx.nn.Linear(
            in_features=hidden_dim,
            out_features=logit_dim,
            key=k4,
        ),
        num_discrete=num_discrete,
        discrete_dim=discrete_dim,
    )


def init_posterior(
    embed_dim: int,
    num_discrete: int,
    discrete_dim: int,
    state_dim: int,
    hidden_dim: int,
    key: jr.PRNGKey,
) -> Posterior:
    k1, k2 = jr.split(key, 2)
    logit_dim = num_discrete * discrete_dim
    return Posterior(
        fc_input=eqx.nn.Linear(
            in_features=state_dim + embed_dim,
            out_features=hidden_dim,
            key=k1,
        ),
        fc_logits=eqx.nn.Linear(
            in_features=hidden_dim,
            out_features=logit_dim,
            key=k2,
        ),
        num_discrete=num_discrete,
        discrete_dim=discrete_dim,
    )


def init_encoder(
    obs_dim: int,
    embed_dim: int,
    hidden_dim: int,
    key: jr.PRNGKey,
) -> Encoder:
    k1, k2 = jr.split(key, 2)
    return Encoder(
        fc1=eqx.nn.Linear(in_features=obs_dim, out_features=hidden_dim, key=k1),
        fc2=eqx.nn.Linear(in_features=hidden_dim, out_features=embed_dim, key=k2),
    )


def init_decoder(
    num_discrete: int,
    discrete_dim: int,
    state_dim: int,
    obs_dim: int,
    hidden_dim: int,
    key: jr.PRNGKey,
) -> Decoder:
    logit_dim = num_discrete * discrete_dim
    k1, k2 = jr.split(key, 2)
    return Decoder(
        fc1=eqx.nn.Linear(
            in_features=state_dim + logit_dim,
            out_features=hidden_dim,
            key=k1,
        ),
        fc2=eqx.nn.Linear(
            in_features=hidden_dim,
            out_features=obs_dim,
            key=k2,
        ),
    )
