import numpy as np
import torch


class DiffusionProcess():
    def __init__(self, args, diffusion_fn, device, shape):
        '''
        beta_1        : beta_1 of diffusion process
        beta_T        : beta_T of diffusion process
        T             : step of diffusion process
        diffusion_fn  : trained diffusion network
        shape         : data shape
        '''
        self.args = args
        self.betas = torch.linspace(start=args.beta1, end=args.betaT, steps=args.diffusion_steps)
        self.shape = shape
        self.deterministic = args.deterministic
        self.karras = 'karras' in args.model
        self.model = args.model
        self.diffusion_fn = diffusion_fn
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.S_churn = 0
        self.S_min = 0
        self.S_max = float('inf')
        self.S_noise = 1
        self.T = args.diffusion_steps
        self.device = device


    def _karras_one_diffusion_step(self, sample, a):
        '''
        x   : perturbated data
        '''
        step_indices = torch.arange(self.T, dtype=torch.float64, device=sample.device)
        t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (self.T - 1) * (
                    self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])])
        sample = sample * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            gamma = min(self.S_churn / self.T, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            x_hat = sample + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(sample)

            # Euler step.
            deionised = self.diffusion_fn(x_hat, t_hat, a).view(sample.shape).to(torch.float64)
            d_cur = (x_hat - deionised) / t_hat
            sample = x_hat + (t_next - t_hat) * d_cur
            sample = sample.to(torch.float32)

            # Apply 2nd order correction.
            if i < self.T - 1:
                deionised = self.diffusion_fn(sample.to(torch.float32), t_next, a).view(sample.shape).to(torch.float64)
                d_prime = (sample - deionised) / t_next
                sample = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                sample = sample.to(torch.float32)

            yield sample

    @torch.no_grad()
    def reverse_sampling(self, x0, a=None):
        sample = x0
        for sample in self._karras_one_reverse_diffusion_step(sample, a=a):
            final = sample
        final = final / self.sigma_max

        return final

    @torch.no_grad()
    def sampling(self, sampling_number=16, xT=None, a=None):
        assert a is not None
        if xT is None:
            if self.args.shared_noise:
                xT = torch.randn([sampling_number, 1, *self.shape[1:]]).to(device=self.device)
                xT = xT.repeat(1, self.shape[0], 1, 1, 1)
            else:
                xT = torch.randn([sampling_number, *self.shape]).to(device=self.device)
        sample = xT
        for sample in self._karras_one_diffusion_step(sample=sample, a=a):
            final = sample
        return final

    def _karras_one_reverse_diffusion_step(self, x, a):
        '''
        x   : perturbated data
        '''
        step_indices = torch.arange(self.T, dtype=torch.float64, device=x.device)
        t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (self.T - 1) * (
                self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])])
        t_steps = t_steps.flip(0)
        t_steps[0] = 1e-5

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            gamma = min((self.S_churn) / self.T, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            x_hat = x #+ (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x)

            # Euler step.
            deionised = self.diffusion_fn(x_hat, t_hat, a).view(x.shape).to(torch.float64)
            d_cur = (x_hat - deionised) / t_hat
            x = x_hat + (t_next - t_hat) * d_cur
            x = x.to(torch.float32)

            # Apply 2nd order correction.
            if i < self.T - 1:
                deionised = self.diffusion_fn(x.to(torch.float32), t_next, a).view(x.shape).to(torch.float64)
                d_prime = (x - deionised) / t_next
                x = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                x = x.to(torch.float32)

            yield x

class LatentDiffusionProcess():
    def __init__(self, args, diffusion_fn, device):
        '''
        beta_1        : beta_1 of diffusion process
        beta_T        : beta_T of diffusion process
        T             : step of diffusion process
        diffusion_fn  : trained diffusion network
        '''
        if args.latent_const:
            args.beta1 = args.betaT = 0.008
        self.betas = torch.linspace(start=args.beta1, end=args.betaT, steps=args.diffusion_steps_latent)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start=args.beta1, end=args.betaT, steps=args.diffusion_steps_latent), dim=0).to(device=device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])
        self.deterministic = args.deterministic
        self.a_dim = args.s_dim + args.d_dim * args.video_length
        self.model = args.model
        self.mode = args.mode
        self.diffusion_fn = diffusion_fn.to(device=device)
        self.device = device

    def _ddpm_one_diffusion_step(self, x):
        '''
        x   : perturbated data
        '''
        for idx in reversed(range(len(self.alpha_bars))):

            noise = torch.zeros_like(x) if idx == 0 else torch.randn_like(x)
            sqrt_tilde_beta = torch.sqrt((1 - self.alpha_prev_bars[idx]) / (1 - self.alpha_bars[idx]) * self.betas[idx])
            predict_epsilon = self.diffusion_fn(x, idx)
            mu_theta_xt = torch.sqrt(1 / self.alphas[idx]) * (x - self.betas[idx] / torch.sqrt(1 - self.alpha_bars[idx]) * predict_epsilon)

            x = mu_theta_xt + sqrt_tilde_beta * noise

            yield x

    def _ddim_one_diffusion_step(self, x):
        '''
        x   : perturbated data
        '''
        eta = 0.01
        for idx in reversed(range(len(self.alpha_bars))):
            predict_epsilon = self.diffusion_fn(x, idx)
            x_0 = (x - torch.sqrt(1 - self.alpha_prev_bars[idx]) * predict_epsilon) / torch.sqrt(self.alpha_prev_bars[idx])
            if idx == 0:
                x = x_0
            else:
                noise = torch.randn_like(x)
                sigma = eta * torch.sqrt((1 - self.alpha_prev_bars[idx-1]) / (1 - self.alpha_bars[idx-1])) * torch.sqrt(self.betas[idx-1])
                x = torch.sqrt(self.alpha_prev_bars[idx-1]) * x_0 + torch.sqrt(1 - self.alpha_prev_bars[idx-1] - sigma**2) * predict_epsilon
                x += sigma * noise
            yield x

    def _ddim_one_reverse_diffusion_step(self, x):
        for idx in range(len(self.alpha_bars)-1):
            if idx == 0:
                yield x
            else:
                predict_epsilon = self.diffusion_fn(x, idx)
                x_0 = (x - torch.sqrt(1 - self.alpha_prev_bars[idx]) * predict_epsilon) / torch.sqrt(self.alpha_prev_bars[idx])
                x = torch.sqrt(self.alpha_prev_bars[idx+1]) * x_0 + torch.sqrt(1 - self.alpha_prev_bars[idx+1]) * predict_epsilon
                yield x

    def _one_diffusion_step(self, sample, deterministic=True):
        if deterministic:
            return self._ddim_one_diffusion_step(sample)
        else:
            return self._ddpm_one_diffusion_step(sample)

    @torch.no_grad()
    def reverse_sampling(self, x0):
        sample = x0
        for sample in self._ddim_one_reverse_diffusion_step(sample):
            final = sample

        return final

    @torch.no_grad()
    def sampling(self, sampling_number=16, xT=None):
        if xT is None:
            xT = torch.randn([sampling_number, self.a_dim]).to(device=self.device)

        sample = xT
        for sample in self._one_diffusion_step(sample=sample, deterministic=self.deterministic):
            final = sample

        return final