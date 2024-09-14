import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.preprocessing import Normalizer
from sqlalchemy import create_engine, text
from torch.utils.data import DataLoader
from beat_net.beat_net.data_loader import PhysionetECG
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax.random import PRNGKey
from jax.nn import softmax
from numpyro.distributions import MultivariateNormal, MixtureSameFamily, Normal, Categorical

class NN(NearestNeighbors):
    def __init__(self, radius_nn=20, **kwargs):
        super().__init__(**kwargs)
        self.radius_nn = radius_nn

    def transform(self, X):
        return self.kneighbors(X, n_neighbors=self.radius_nn)


def plot_outlier_ecg(outlier_ecg, normal_ecgs, ax,
                     leads=('aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')):
    for i, track in enumerate(outlier_ecg):
        ax.plot(track - 2 * i, color='red', alpha=1)
    for ecg in normal_ecgs:
        for i, track in enumerate(ecg):
            ax.plot(track - 2 * i, alpha=.05, color='blue')
    ax.set_yticks(-2*np.arange(len(outlier_ecg)), leads[:len(outlier_ecg)], fontsize=18)
    ax.set_ylim([-18, 1])
    return ax


def gaussian_posterior_batch_diagonal(y,
                                      likelihood_A,
                                      likelihood_bias,
                                      likelihood_precision_diag,
                                      prior_loc,
                                      prior_covar_diag):
    prior_precision_diag = 1 / prior_covar_diag
    posterior_precision_diag = prior_precision_diag.clone()
    posterior_precision_diag = posterior_precision_diag.at[likelihood_A != 0].set(posterior_precision_diag[likelihood_A != 0] + (likelihood_A[likelihood_A != 0]**2) * likelihood_precision_diag)
    posterior_covariance_diag = 1 / posterior_precision_diag
    mean_residue = y - likelihood_bias
    mean_projected_residue = jnp.zeros_like(prior_loc)
    mean_projected_residue = mean_projected_residue.at[likelihood_A != 0].set((likelihood_A[likelihood_A != 0] * likelihood_precision_diag * mean_residue))
    mean_prior = prior_precision_diag[None, :] * prior_loc
    posterior_mean = posterior_covariance_diag[None, :] * (mean_projected_residue[None, :] + mean_prior)
    return Normal(loc=posterior_mean, scale=jnp.repeat(posterior_covariance_diag[None], repeats=posterior_mean.shape[0], axis=0)**.5)



def get_posterior(obs_full, dataset, dataset_nn, indexes_observed, observation_std, prior_std):
    n_channels, t = dataset[0].shape
    distances, neighbours = dataset_nn.transform(obs_full[None, ...])

    likelihood_A = jnp.zeros(n_channels*t)
    for i in indexes_observed:
        likelihood_A = likelihood_A.at[176*i + jnp.arange(176)].set(1)

    neighbours = neighbours[0]
    log_weights = []
    components = []
    for loc in dataset[neighbours]:
        new_dist = gaussian_posterior_batch_diagonal(y=obs_full[indexes_observed].flatten(),
                                                     likelihood_A=likelihood_A,
                                                     likelihood_bias=jnp.zeros(t*len(indexes_observed)),
                                                     likelihood_precision_diag=jnp.ones(t*len(indexes_observed)) / (observation_std**2),
                                                     prior_loc=loc.flatten(),
                                                     prior_covar_diag=jnp.ones_like(loc.flatten()) * (prior_std**2))
        residue = (obs_full[indexes_observed].flatten() - new_dist.loc.flatten()[likelihood_A.astype(bool)]) / observation_std
        log_constant = -jnp.linalg.norm(residue)**2 / 2 + \
                       -jnp.linalg.norm((loc.flatten() - new_dist.loc) / prior_std)**2 / 2 - \
                       new_dist.log_prob(new_dist.loc).sum()
        log_weights.append(log_constant.item())
        components.append(new_dist)
    component_dist = MultivariateNormal(loc=jnp.stack([l.loc[0] for l in components], axis=0), covariance_matrix=jnp.stack([jnp.diag(l.scale[0]**2) for l in components], axis=0))
    return MixtureSameFamily(Categorical(logits=jnp.array(log_weights)),
                             component_dist)


if __name__ == '__main__':
    database_path = '/mnt/data/gabriel/physionet.org/beats_db'
    label = 'MI-NSR-abQRS'#'LAD-LAnFB-NSR'#'LBBB'#'MI-NSR-abQRS'#'LVH-NSR'#'LBBB'#'LVH-NSR'
    tracks_type = 'Left-from-right'
    if tracks_type == 'Near-from-far':
        tracks_to_keep = [0, 1, 2]
    if tracks_type == 'Inside V leads':
        tracks_to_keep = [0, 1, 2, 3, 8]
    elif tracks_type == 'Left-from-right':
        tracks_to_keep = [0, 1, 2, 3, 4]


    dist_dataloader = DataLoader(dataset=PhysionetECG(database_path=database_path,
                                                      categories_to_filter=["NSR", "SB", "STach", "SA"],
                                                      normalized=False),
                                 batch_size=256,
                                 shuffle=True,
                                 num_workers=10)

    out_dist_dataloader = DataLoader(dataset=PhysionetECG(database_path=database_path, categories_to_filter=[label, ],
                                                          normalized=False),
                                     batch_size=256,
                                     shuffle=True,
                                     num_workers=10)
    prior_std = 1e-2
    measurement_std = 1e-1
    radius_nn = 200

    normalized_neighbours = Pipeline([("select tracks", FunctionTransformer(func=lambda x: x[:, tracks_to_keep].reshape(x.shape[0], -1))),
                                      ("Normalize", Normalizer(norm='max')),
                                      ("NN", NN(n_neighbors=20, radius_nn=radius_nn, n_jobs=10))])
    unnormalized_neighbours = Pipeline([("select tracks", FunctionTransformer(func=lambda x: x[:, tracks_to_keep].reshape(x.shape[0], -1))),
                                        ("NN", NN(radius_nn=radius_nn, n_neighbors=40, n_jobs=10))])
    dist_db = []
    for _ in range(6):
        for batch_data, _, _ in dist_dataloader:
            dist_db.append(batch_data.numpy())
    dist_db = np.concatenate(dist_db)
    print("Database done")
    normalized_neighbours = normalized_neighbours.fit(dist_db)
    print("Nearest neighbors constructed")

    out_dist_db = []
    for batch_data, _, _ in out_dist_dataloader:
        out_dist_db.append(batch_data.numpy())
    out_dist_db = np.concatenate(out_dist_db)
    print("Out of dist database done")
    min_post_log_prob = 0
    for i, ecgs in enumerate(out_dist_db):
        posterior = get_posterior(obs_full=ecgs, dataset=dist_db, dataset_nn=normalized_neighbours,
                                  indexes_observed=tracks_to_keep, prior_std=prior_std, observation_std=measurement_std)
        post_logprob = posterior.log_prob(ecgs.flatten())
        print(i, post_logprob)
        draws = posterior.sample(key=PRNGKey(0), sample_shape=(100,)).reshape(100, *ecgs.shape)
        fig, ax = plt.subplots(1, 1, figsize=(5, 10))
        fig.subplots_adjust(left=.12, right=.99, top=.99, bottom=.05)
        ax = plot_outlier_ecg(outlier_ecg=ecgs / np.abs(ecgs).max(axis=1).clip(1e-6, 10)[:, None],
                              normal_ecgs=draws / np.abs(draws).max(axis=2).clip(1e-6, 10)[..., None], ax=ax)
        fig.show()
        if min_post_log_prob > post_logprob:
            min_post_log_prob = post_logprob
            fig.savefig(f'images/{label}_{tracks_type}_example.pdf')
        plt.close(fig)