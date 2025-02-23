import equinox as eqx
from jax import Array, numpy as jnp, random as jr, nn, lax, vmap

from rssm.model import Model, Prior, Posterior, Encoder, Decoder, State


def init_model(
    obs_dim: int,
    action_dim: int,
    obs_embed_dim: int,
    latent_dim: int,
    state_dim: int,
    hidden_dim: int,
    key: jr.PRNGKey,
) -> Model:
    k1, k2, k3, k4 = jr.split(key, 4)
    return Model(
        prior=init_prior(action_dim, latent_dim, state_dim, hidden_dim, k1),
        posterior=init_posterior(obs_embed_dim, latent_dim, state_dim, hidden_dim, k2),
        encoder=init_encoder(obs_dim, obs_embed_dim, hidden_dim, k3),
        decoder=init_decoder(latent_dim, state_dim, obs_dim, hidden_dim, k4),
        latent_dim=latent_dim,
        state_dim=state_dim,
    )


def init_prior(
    action_dim: int,
    latent_dim: int,
    state_dim: int,
    hidden_dim: int,
    key: jr.PRNGKey,
) -> Prior:
    k1, k2, k3, k4 = jr.split(key, 4)
    return Prior(
        fc_input=eqx.nn.Linear(
            in_features=action_dim + latent_dim,
            out_features=hidden_dim,
            key=k1,
        ),
        rnn_cell=eqx.nn.GRUCell(
            input_size=hidden_dim,
            hidden_size=state_dim,
            key=k2,
        ),
        fc_rnn_hidden=eqx.nn.Linear(
            in_features=state_dim,
            out_features=hidden_dim,
            key=k3,
        ),
        fc_output=eqx.nn.Linear(
            in_features=hidden_dim,
            out_features=2 * latent_dim,
            key=k4,
        ),
    )


def init_posterior(
    obs_emb_dim: int,
    latent_dim: int,
    state_dim: int,
    hidden_dim: int,
    key: jr.PRNGKey,
) -> Posterior:
    k1, k2 = jr.split(key, 2)
    return Posterior(
        fc_input=eqx.nn.Linear(
            in_features=state_dim + obs_emb_dim,
            out_features=hidden_dim,
            key=k1,
        ),
        fc_output=eqx.nn.Linear(
            in_features=hidden_dim,
            out_features=2 * latent_dim,
            key=k2,
        ),
    )


def init_encoder(
    obs_dim: int,
    obs_embed_dim: int,
    hidden_dim: int,
    key: jr.PRNGKey,
) -> Encoder:
    k1, k2 = jr.split(key, 2)
    return Encoder(
        fc1=eqx.nn.Linear(in_features=obs_dim, out_features=hidden_dim, key=k1),
        fc2=eqx.nn.Linear(in_features=hidden_dim, out_features=obs_embed_dim, key=k2),
    )


def init_decoder(
    latent_dim: int,
    state_dim: int,
    obs_dim: int,
    hidden_dim: int,
    key: jr.PRNGKey,
) -> Decoder:
    k1, k2 = jr.split(key, 2)
    return Decoder(
        fc1=eqx.nn.Linear(
            in_features=state_dim + latent_dim,
            out_features=hidden_dim,
            key=k1,
        ),
        fc2=eqx.nn.Linear(
            in_features=hidden_dim,
            out_features=obs_dim,
            key=k2,
        ),
    )


def kl_divergence(post: State, prior: State) -> Array:
    kl = (
        jnp.log(prior.std / post.std)
        + (post.std**2 + (post.mean - prior.mean) ** 2) / (2 * prior.std**2)
        - 0.5
    )
    return jnp.sum(kl, axis=-1)


def mse_loss(out_seq: Array, obs_seq: Array) -> Array:
    return jnp.mean(jnp.sum((out_seq - obs_seq) ** 2, axis=-1))
