from benchopt import BaseSolver, safe_import_context
from abc import ABC, abstractmethod

with safe_import_context() as import_ctx:
    import torch
    from tqdm import tqdm


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data)


class Solver(object):
    def __init__(self):
        self.name = 'Python-DPS'  # proximal gradient, optionally accelerated

        # Any parameter defined here is accessible as an attribute of the solver.
        self.parameters = {"ddim_params": [(10000, .1)], "zeta":[10]}
        self.ddim_params = (10000, .1)
        self.zeta = 10

    # Store the information to compute the objective. The parameters of this
    # function are the keys of the dictionary obtained when calling
    # ``Objective.get_objective``.
    def set_objective(self,
                      observation,
                      # bservation_noise,
                      forward_operator_l,
                      forward_operator_t,
                      score_network,
                      alphas_cumprod,
                      n_samples):
        self.dim_y_l, self.dim_x_l = forward_operator_l.shape
        self.dim_y_t, self.dim_x_t = forward_operator_t.shape
        self.n_samples = n_samples
        n_steps, self.eta = self.ddim_params
        self.timesteps = torch.linspace(0, 999, n_steps).long().tolist()
        self.alphas_cumprod = alphas_cumprod
        self.score_network = score_network
        def dps_grad_fun(sample, t, alpha_t):
            diffusion_steps = (t * torch.ones((1, 1)).cuda())
            score = score_network((sample.unsqueeze(0), diffusion_steps,), mel_spec=None)[0]
            pred_x0 = (1 / (alpha_t**.5))*(sample +
                                           (1 - alpha_t) * score)
            residue = torch.linalg.norm(forward_operator_l @ pred_x0 @ forward_operator_t - observation)
            return residue**2, pred_x0

        self.dps_fn = torch.func.vmap(torch.func.grad_and_value(dps_grad_fun, has_aux=True), in_dims=(0, None, None))


    def run(self, n_iter):
        n_steps, eta = self.ddim_params
        samples = torch.randn(size=(self.n_samples, self.dim_x_l, self.dim_x_t)).cuda()
        for i, (t, t_prev) in tqdm(enumerate(zip(self.timesteps[1:][::-1], self.timesteps[:-1][::-1])), total=len(self.timesteps)-1):
            # print(samples.shape)
            alpha_t, alpha_t_prev = self.alphas_cumprod[t], self.alphas_cumprod[t_prev]
            grad_residue, (residue_sq, pred_x0) = self.dps_fn(samples, t, alpha_t)
            z = torch.randn_like(samples)
            coeff_sample = (alpha_t ** .5) * (1 - alpha_t_prev) / (1 - alpha_t)
            coeff_pred = (alpha_t ** .5) * (1 - alpha_t / alpha_t_prev) / (1 - alpha_t)
            coeff_lik_score = self.zeta / residue_sq**.5
            noise_std = eta * (((1 - alpha_t_prev) / (1 - alpha_t)) * (1 - alpha_t / alpha_t_prev)) ** .5

            samples = coeff_sample * samples + coeff_pred * pred_x0 - coeff_lik_score[:, None,  None]*grad_residue + noise_std * z
        self.samples = samples

    # Return the solution estimate computed.
    def get_result(self):
        return {'samples': self.samples}

