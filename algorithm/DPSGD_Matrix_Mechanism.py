import torch
import math
import numpy as np
from torch.utils.data import DataLoader
import copy
import os

from data.util.get_data import get_scatter_transform, get_scattered_dataset, get_scattered_loader, \
    get_scatter_transform_l
from train_and_validation.train_with_dp import train_with_dp_matrix
from train_and_validation.validation import validation

from privacy_analysis.Matrix_Mechanism_RDP.Find_optimal_sigma_C import optimize_matrix_mechanism

from privacy_analysis.dp_utils import scatter_normalization
from utils.dp_optimizer import DPSGD_Optimizer, DPAdam_Optimizer
from model.CNN import CIFAR10_CNN_Tanh, MNIST_CNN_Tanh

CNNS = {
    "CIFAR-10": CIFAR10_CNN_Tanh,
    "FMNIST": MNIST_CNN_Tanh,
    "MNIST": MNIST_CNN_Tanh,
}


# Balls-in-bins Batch Sampler (用于完美的矩阵机制无偏采样)
class BallsInBinsBatchSampler:
    def __init__(self, dataset_size, bins, target_batch_size):
        self.dataset_size = dataset_size
        self.bins = bins
        self.B = target_batch_size
        bin_assignments = np.random.randint(0, self.bins, size=self.dataset_size)
        self.batches = []
        for b_idx in range(self.bins):
            indices = np.where(bin_assignments == b_idx)[0].tolist()
            if len(indices) < self.B:
                padding = np.random.choice(indices, self.B - len(indices)).tolist() if len(indices) > 0 else []
                indices.extend(padding)
            elif len(indices) > self.B:
                indices = indices[:self.B]
            self.batches.append(indices)

    def __iter__(self):
        for batch_indices in self.batches:
            yield batch_indices

    def __len__(self):
        return self.bins


def get_inverse_toeplitz_weights(c_col):
    """求解 C^{-1} 的第一列权重 W"""
    n = len(c_col)
    device = c_col.device
    col_indices = torch.arange(n, device=device).unsqueeze(0)
    row_indices = torch.arange(n, device=device).unsqueeze(1)
    mask = row_indices >= col_indices
    C = torch.where(mask, c_col[row_indices - col_indices], torch.tensor(0.0, device=device))

    e_0 = torch.zeros(n, 1, device=device, dtype=c_col.dtype)
    e_0[0, 0] = 1.0
    W = torch.linalg.solve_triangular(C, e_0, upper=False).squeeze()
    return W


def DPSGD_Matrix(dataset_name, train_dataset, test_data, model, optimizer, batch_size,
                             epsilon_budget, delta, device, target_epochs, beta, sigma_th, C_t, args):
    b = len(train_dataset) // batch_size
    target_n = target_epochs * b
    sampler = BallsInBinsBatchSampler(len(train_dataset), b, batch_size)

    test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    use_scattering = args.use_scattering
    input_norm = args.input_norm
    bn_noise_multiplier = args.bn_noise_multiplier
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
    num_groups = args.num_groups
    sigma_t = args.sigma_t
    lr = args.lr
    momentum = args.momentum

    if dataset_name != 'IMDB':

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

        if use_scattering:
            if dataset_name == 'FMNIST':
                scattering, K, _ = get_scatter_transform_l(dataset_name)
                scattering.to(device)
            else:
                scattering, K, _ = get_scatter_transform(dataset_name)
                scattering.to(device)
        else:
            scattering = None
            K = 3 if len(train_dataset.data.shape) == 4 else 1

        if input_norm == "BN":
            save_dir = f"bn_stats/{dataset_name}"
            os.makedirs(save_dir, exist_ok=True)
            bn_stats, rdp_norm = scatter_normalization(train_loader,
                                                       scattering,
                                                       K,
                                                       device,
                                                       len(train_dataset),
                                                       len(train_dataset),
                                                       noise_multiplier=bn_noise_multiplier,
                                                       orders=orders,
                                                       save_dir=save_dir)

            model = CNNS[dataset_name](K, input_norm="BN", bn_stats=bn_stats, size=None)


        else:
            model = CNNS[dataset_name](K, input_norm=input_norm, num_groups=num_groups, size=None)

        model.to(device)
        train_dataset = get_scattered_dataset(train_loader, scattering, device, len(train_dataset))
        test_dl = get_scattered_loader(test_dl, scattering, device)

        optimizer = DPSGD_Optimizer(
            l2_norm_clip=C_t,
            noise_multiplier=sigma_t,
            minibatch_size=batch_size,
            microbatch_size=1,
            soft_c=args.soft_c,
            max_error=args.max_error,
            params=model.parameters(),
            lr=lr,
            momentum=momentum
        )
    else:
        optimizer = DPAdam_Optimizer(
            l2_norm_clip=C_t,
            noise_multiplier=sigma_t,
            minibatch_size=batch_size,
            microbatch_size=1,
            soft_c=args.soft_c,
            max_error=args.max_error,
            params=model.parameters(),
            lr=lr)

    train_dl = DataLoader(train_dataset, batch_sampler=sampler)

    print(f"Target Epsilon: {epsilon_budget}, Delta: {delta}")
    print(f"update K: {target_n}")

    #bins=64, epoch=4, epsilon=1.0, target_delta=1e-5, num_iterations=100
    optimal_c_col, sigma_opt = optimize_matrix_mechanism(
        bins=b,
        epoch=target_epochs,
        epsilon=epsilon_budget,
        target_delta=delta,
        num_iterations=40
    )

    print(f"矩阵机制分配的最优结构噪声 Sigma_MM: {sigma_opt:.4f}")

    W_weights = get_inverse_toeplitz_weights(optimal_c_col)

    optimizer.noise_multiplier = sigma_opt
    optimizer.init_matrix_mechanism(W_weights)

    best_test_acc = 0.
    best_iter = 0
    test_loss_list = []
    epsilon_list = []

    global_step = 0
    last_E_tracker = 10000.0  # 初始极大的伪截断误差

    current_epoch = 1


    while True:
        train_loss, train_acc, _, global_step = train_with_dp_matrix(
            model, train_dl, optimizer, device, global_step_start=global_step, test_dl = test_dl
        )

        test_loss, test_accuracy = validation(model, test_dl, device)

        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_iter = current_epoch

        test_loss_list.append(test_loss)
        epsilon_list.append(torch.tensor(epsilon_budget))

        print(f'Epoch:{current_epoch} |  '
              f'process: {global_step}/{target_n} | Test loss: {test_loss:.4f}, | Test Acc: {test_accuracy:.2f}%')

        if global_step >= target_n:
            print(f"\n{target_n}/{target_n} 矩阵容量已满，训练结束。")
            break
        if target_n - global_step <= 10:
            print(f"\n仅剩 {target_n - global_step} 步 (<=10)，停止延展。")
            break

        current_epoch += 1

    print("------ Finished ------")
    return test_accuracy, current_epoch, best_test_acc, best_iter, model, [epsilon_list, test_loss_list]