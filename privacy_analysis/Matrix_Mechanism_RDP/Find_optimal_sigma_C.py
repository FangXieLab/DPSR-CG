import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math


def compute_m(c_col, b, E):

    n = b * E
    m = torch.zeros((b, n), dtype=c_col.dtype, device=c_col.device)
    for k in range(b):
        for j in range(E):
            col_idx = j * b + k
            if col_idx < n:
                m[k, col_idx:] += c_col[:n - col_idx]
    return m


def compute_delta_hat(c_col, sigma, epsilon, b, E, Z, i_idx):

    n = b * E
    M = Z.shape[0]

    m = compute_m(c_col, b, E)

    m_i_sampled = m[i_idx]

    X = m_i_sampled + sigma * Z


    X_dot_m = torch.matmul(X, m.T)
    m_sq = torch.sum(m ** 2, dim=1)
    exponent = (2 * X_dot_m - m_sq.unsqueeze(0)) / (2 * sigma ** 2)


    Y = torch.logsumexp(exponent, dim=1) - math.log(b)

    loss_term = 1.0 - torch.exp(epsilon - Y)
    delta_hat = torch.mean(F.relu(loss_term))
    return delta_hat


class BisectionSigmaSolver(torch.autograd.Function):

    @staticmethod
    def forward(ctx, c_col, epsilon, target_delta, b, E, Z, i_idx):

        with torch.no_grad():
            low = 1e-4
            high = 50.0

            for _ in range(15):
                d_high = compute_delta_hat(c_col, high, epsilon, b, E, Z, i_idx)
                if d_high <= target_delta:
                    break
                high *= 2.0

            for _ in range(60):
                mid = (low + high) / 2.0
                d_mid = compute_delta_hat(c_col, mid, epsilon, b, E, Z, i_idx)
                if d_mid > target_delta:
                    low = mid
                else:
                    high = mid
            sigma_star = torch.tensor((low + high) / 2.0, device=c_col.device, dtype=c_col.dtype)

        ctx.save_for_backward(c_col, sigma_star, Z, i_idx)
        ctx.epsilon = epsilon
        ctx.target_delta = target_delta
        ctx.b = b
        ctx.E = E
        return sigma_star

    @staticmethod
    def backward(ctx, grad_output):
        c_col, sigma_star, Z, i_idx = ctx.saved_tensors
        epsilon = ctx.epsilon
        b, E = ctx.b, ctx.E

        with torch.enable_grad():
            c_col_copy = c_col.detach().requires_grad_(True)
            sigma_copy = sigma_star.detach().requires_grad_(True)

            delta_hat = compute_delta_hat(c_col_copy, sigma_copy, epsilon, b, E, Z, i_idx)

            g_c, g_s = torch.autograd.grad(delta_hat, (c_col_copy, sigma_copy), retain_graph=False)

        grad_c = -g_c / (g_s + 1e-12) * grad_output
        return grad_c, None, None, None, None, None, None


def compute_rmse_true(c_col, sigma, n):
    device = c_col.device

    col_indices = torch.arange(n, device=device).unsqueeze(0)
    row_indices = torch.arange(n, device=device).unsqueeze(1)
    mask = row_indices >= col_indices
    C = torch.where(mask, c_col[row_indices - col_indices], torch.tensor(0.0, device=device))

    A = torch.tril(torch.ones(n, n, device=device, dtype=c_col.dtype))

    AC_inv = torch.linalg.solve_triangular(C.T, A.T, upper=True).T

    norm_AC_inv = torch.norm(AC_inv, p='fro') / math.sqrt(n)
    return sigma * norm_AC_inv, norm_AC_inv


def optimize_matrix_mechanism(bins=64, epoch=4, epsilon=1.0, target_delta=1e-5, num_iterations=100):
    b, E = bins, epoch
    n = b * E
    M = 50000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    train_proxy_delta = 1e-3

    initial_gamma = 0.9
    init_c_rest = torch.tensor([initial_gamma ** i for i in range(1, n)], dtype=torch.float32, device=device)
    init_c_raw_rest = torch.log(torch.exp(init_c_rest) - 1.0)

    c_raw_rest = nn.Parameter(init_c_raw_rest)
    optimizer = optim.Adam([c_raw_rest], lr=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)

    for step in range(num_iterations):
        optimizer.zero_grad()

        c_0 = torch.tensor([1.0], dtype=torch.float32, device=device)
        c_rest = F.softplus(c_raw_rest)
        c_col = torch.cat([c_0, c_rest])

        Z = torch.randn(M, n, device=device)
        i_idx = torch.randint(0, b, (M,), device=device)

        sigma_train = BisectionSigmaSolver.apply(c_col, epsilon, train_proxy_delta, b, E, Z, i_idx)
        rmse_train,rmse_struct = compute_rmse_true(c_col, sigma_train, n)

        rmse_train.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % 20 == 0:
            print(
                f"Step {step + 1:03d} | Train RMSE (Proxy): {rmse_train.item():.4f} | Structure RMSE: {rmse_struct.item(): .4f} | Proxy Sigma: {sigma_train.item():.4f}")


    optimal_c_col = torch.cat([
        torch.tensor([1.0], dtype=torch.float32, device=device),
        F.softplus(c_raw_rest)
    ]).detach()


    M_eval = 200000
    Z_eval = torch.randn(M_eval, n, device=device)
    i_idx_eval = torch.randint(0, b, (M_eval,), device=device)

    with torch.no_grad():
        final_sigma = BisectionSigmaSolver.apply(optimal_c_col, epsilon, target_delta, b, E, Z_eval, i_idx_eval)

    return optimal_c_col, final_sigma.item()




if __name__ == "__main__":
    c_opt, sigma_opt = optimize_matrix_mechanism(bins=64, epoch=4, epsilon=3.0, target_delta=1e-5)