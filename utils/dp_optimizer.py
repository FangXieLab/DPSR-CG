import numpy as np
import torch
from torch.optim import Optimizer
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions.normal import Normal
from torch.optim import SGD, Adam, Adagrad, RMSprop
from math import exp


def make_optimizer_class(cls):
    class DPOptimizerClass(cls):
        def __init__(self, l2_norm_clip, noise_multiplier, minibatch_size, microbatch_size,soft_c,max_error, *args, **kwargs):

            super(DPOptimizerClass, self).__init__(*args, **kwargs)

            self.l2_norm_clip = l2_norm_clip
            self.error_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier
            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size
            # MNIST
            # self.soft_c = 8
            # self.max_error = 11
            #FMNIST
            # self.soft_c = 6
            # self.max_error = 11
            # CIFAR
            # self.soft_c = 3
            # self.max_error = 11
            # IMDB
            self.soft_c = soft_c
            self.max_error = max_error
            
            self.b_t = 0
            self.sigma2 = 10 * noise_multiplier


            for id, group in enumerate(self.param_groups):
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in
                                        group['params']]

        def zero_microbatch_grad(self):
            super(DPOptimizerClass, self).zero_grad()

        def microbatch_step(self):
            total_norm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2.

            total_norm = total_norm ** .5
            clip_coef = min(self.l2_norm_clip / (total_norm + 1e-6), 1.)

            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul(clip_coef))

            return total_norm

        def microbatch_step_GA(self):
            total_norm = 0.0
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2.

            total_norm = total_norm ** 0.5

            if total_norm <= self.soft_c:
                clip_coef = self.l2_norm_clip / self.soft_c
            else:
                clip_coef = self.soft_c / (total_norm + 1e-6)
                self.b_t += 1

            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul(clip_coef))

            return total_norm

        def microbatch_step_SEGA(self):
            total_norm = 0.0
            num_e = 0.0
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2.

            total_norm = (total_norm) ** 0.5

            if total_norm <= self.soft_c:
                clip_coef = self.l2_norm_clip / self.soft_c
                total_error = 0
            else:
                clip_coef = self.soft_c / (total_norm + 1e-6)
                num_e += 1
                self.b_t += 1
                if total_norm <= self.max_error:
                    total_error = total_norm * self.l2_norm_clip / self.max_error
                else:
                    total_error = total_norm * self.l2_norm_clip / total_norm

            
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul(clip_coef))

            return total_norm, total_error, num_e


        def step_soft_c(self):
            self.b_t = self.b_t + self.sigma2 * np.random.randn()
            self.soft_c = self.soft_c * exp(-0.1 + (1 / self.minibatch_size) * self.b_t)
            self.b_t = 0

        def zero_accum_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()

        def step_dp(self, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()
                        param.grad.data.add_(
                            self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))
                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)
            super(DPOptimizerClass, self).step(*args, **kwargs)

        def step_dp_ser(self, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()
                        param.grad.data.add_(
                            self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))
                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)
            super(DPOptimizerClass, self).step(*args, **kwargs)

        def init_matrix_mechanism(self, W_weights):
            device = self.param_groups[0]['params'][0].device
            self.W_weights = W_weights.to(device)
            self.noise_history = []

        def step_dp_matrix(self, current_step, *args, **kwargs):
            flat_z_t_list = []
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        z = torch.randn_like(param.grad.data) * self.l2_norm_clip * self.noise_multiplier
                        flat_z_t_list.append(z.view(-1))

            flat_z_t = torch.cat(flat_z_t_list)
            self.noise_history.append(flat_z_t)  # 存入历史缓存

            w_slice = self.W_weights[:current_step + 1].flip(0).unsqueeze(1)
            history_tensor = torch.stack(self.noise_history)  # shape: (t+1, total_params)


            correlated_noise_flat = torch.sum(history_tensor * w_slice, dim=0)

            offset = 0
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        numel = param.numel()
                        noise_chunk = correlated_noise_flat[offset:offset + numel].view_as(param.grad.data)
                        offset += numel

                        param.grad.data = accum_grad.clone()
                        param.grad.data.add_(noise_chunk)
                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)

            super(DPOptimizerClass, self).step(*args, **kwargs)
    return DPOptimizerClass


DPAdam_Optimizer = make_optimizer_class(Adam)
DPAdagrad_Optimizer = make_optimizer_class(Adagrad)
DPSGD_Optimizer = make_optimizer_class(SGD)
DPRMSprop_Optimizer = make_optimizer_class(RMSprop)


def get_dp_optimizer(dataset_name, algortithm, lr, momentum, C_t, sigma, batch_size, model):
    if dataset_name == 'IMDB' and algortithm != 'DPAGD':
        optimizer = DPAdam_Optimizer(
            l2_norm_clip=C_t,
            noise_multiplier=sigma,
            minibatch_size=batch_size,
            microbatch_size=1,
            soft_c=8,
            max_error=11,
            params=model.parameters(),
            lr=lr,
        )
    else:
        optimizer = DPSGD_Optimizer(
            l2_norm_clip=C_t,
            noise_multiplier=sigma,
            minibatch_size=batch_size,
            microbatch_size=1,
            soft_c=8,
            max_error=11,
            params=model.parameters(),
            lr=lr,
            momentum=momentum
        )
    return optimizer