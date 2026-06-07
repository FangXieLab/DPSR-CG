# DPSR-CG

*DPSR-CG: Differentially Private SGD with Selective Release Based On Clipped Gradients*

This repository is the official implementation of the paper:

**Revisiting Privacy Amplification by Subsampling in Selective Release DPSGD**

*Under review at ICML 2026*

## Citation
If you find this repository or our work useful, please consider citing our paper:

```bibtex
@article{huang2026revisiting,
  title={Revisiting Privacy Amplification by Subsampling in Selective Release DPSGD},
  author={Huang, Xiaobo and Xie, Fang},
  journal={arXiv preprint},
  year={2026}
}

```

## Acknowledgements & Base Work

This codebase is built upon the official implementation of DPSUR (VLDB 2024). We express our gratitude to the authors for open-sourcing their code. If you find this repository useful, please consider citing both our work and the original DPSUR paper:

```bibtex
@article{fu2024dpsur,
  title={DPSUR: Accelerating Differentially Private Stochastic Gradient Descent Using Selective Update and Release},
  author={Fu, Jie and Ye, Qingqing and Hu, Haibo and Chen, Zhili and Wang, Lulu and Wang, Kuncan and Ran, Xun},
  journal={Proceedings of the VLDB Endowment},
  volume={17},
  number={6},
  pages={1200--1213},
  year={2024},
  publisher={VLDB Endowment}
}

```

## Main Results & Execution Commands

Below are the tables of our main experimental results and their corresponding execution commands.

*(Note: All experiments were run using a single NVIDIA RTX 5090 with the random seeds fixed internally in `main.py`.)*

### 1. DPSR-CG (Proposed)

| Dataset | Privacy Budget ($\epsilon$) | Test Accuracy | Exact Command to Reproduce |
| --- | --- | --- | --- |
| **MNIST** | 1.0 | 99.06% | `nohup python -u main.py --algorithm DPSR_CG --dataset_name MNIST --sigma_t 2.5 --lr 0.1 --batch_size 1024 --C_t=1 --beta=3.0 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --soft_c=8 --max_error=11 --eps=1.0 > DPSR_CG_MNIST_eps1.log 2>&1 &` |
|  | 2.0 | 99.22% | `nohup python -u main.py --algorithm DPSR_CG --dataset_name MNIST --sigma_t 2.0 --lr 0.1 --batch_size 1024 --C_t=1 --beta=3.0 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --soft_c=8 --max_error=11 --eps=2.0 > DPSR_CG_MNIST_eps2.log 2>&1 &` |
|  | 3.0 | 99.24% | `nohup python -u main.py --algorithm DPSR_CG --dataset_name MNIST --sigma_t 1.8 --lr 0.1 --batch_size 1024 --C_t=1 --beta=3.0 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --soft_c=8 --max_error=11 --eps=3.0 > DPSR_CG_MNIST_eps3.log 2>&1 &` |
| **FMNIST** | 1.0 | 90.31% | `nohup python -u main.py --algorithm DPSR_CG --dataset_name FMNIST --sigma_t 4.0 --lr 0.1 --batch_size 2048 --C_t=1 --beta=3 --input_norm=GroupNorm --num_groups=27 --use_scattering --soft_c=6 --max_error=11 --eps=1.0 > DPSR_CG_FMNIST_eps1.log 2>&1 &` |
|  | 2.0 | 90.87% | `nohup python -u main.py --algorithm DPSR_CG --dataset_name FMNIST --sigma_t 3.0 --lr 0.1 --batch_size 2048 --C_t=1 --beta=3 --input_norm=GroupNorm --num_groups=27 --use_scattering --soft_c=6 --max_error=11 --eps=2.0 > DPSR_CG_FMNIST_eps2.log 2>&1 &` |
|  | 3.0 | 90.98% | `nohup python -u main.py --algorithm DPSR_CG --dataset_name FMNIST --sigma_t 2.15 --lr 0.1 --batch_size 2048 --C_t=1 --beta=3 --input_norm=GroupNorm --num_groups=27 --use_scattering --soft_c=6 --max_error=11 --eps=3.0 > DPSR_CG_FMNIST_eps3.log 2>&1 &` |
| **CIFAR-10** | 1.0 | 69.94% | `nohup python -u main.py --algorithm DPSR_CG --dataset_name CIFAR-10 --sigma_t 5.67 --lr 0.1 --batch_size 2048 --C_t=1.0 --beta=3 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --soft_c=3 --max_error=11 --eps=1.0 > DPSR_CG_CIFAR_eps1.log 2>&1 &` |
|  | 2.0 | 72.25% | `nohup python -u main.py --algorithm DPSR_CG --dataset_name CIFAR-10 --sigma_t 3.5 --lr 0.1 --batch_size 2048 --C_t=1.0 --beta=3 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --soft_c=3 --max_error=11 --eps=2.0 > DPSR_CG_CIFAR_eps2.log 2>&1 &` |
|  | 3.0 | 72.37% | `nohup python -u main.py --algorithm DPSR_CG --dataset_name CIFAR-10 --sigma_t 3.5 --lr 0.1 --batch_size 2048 --C_t=1.0 --beta=3 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --soft_c=3 --max_error=11 --eps=3.0 > DPSR_CG_CIFAR_eps3.log 2>&1 &` |
| **IMDB** | 1.0 | 68.43% | `nohup python -u main.py --algorithm DPSR_CG --dataset_name IMDB --sigma_t 2.5 --lr 0.02 --batch_size 1024 --C_t=1.0 --beta=3 --soft_c=1 --max_error=11 --eps=1.0 > DPSR_CG_IMDB_eps1.log 2>&1 &` |
|  | 2.0 | 71.44% | `nohup python -u main.py --algorithm DPSR_CG --dataset_name IMDB --sigma_t 1.8 --lr 0.02 --batch_size 1024 --C_t=1.0 --beta=3 --soft_c=1 --max_error=11 --eps=2.0 > DPSR_CG_IMDB_eps2.log 2>&1 &` |
|  | 3.0 | 73.01% | `nohup python -u main.py --algorithm DPSR_CG --dataset_name IMDB --sigma_t 1.5 --lr 0.02 --batch_size 1024 --C_t=1.0 --beta=3 --soft_c=1 --max_error=11 --eps=3.0 > DPSR_CG_IMDB_eps3.log 2>&1 &` |

---

### 2. DPSUR (Base Work)

| Dataset | Privacy Budget ($\epsilon$) | Test Accuracy | Exact Command to Reproduce |
| --- | --- | --- | --- |
| **MNIST** | 1.0 | 96.32% | `nohup python -u main.py --algorithm DPSUR --dataset_name MNIST --sigma_t 2.0 --lr 2.0 --batch_size 1024 --C_v=0.001 --sigma_v=1.3 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=1.0 > DPSUR_MNIST_eps1.log 2>&1 &` |
|  | 2.0 | 97.51% | `nohup python -u main.py --algorithm DPSUR --dataset_name MNIST --sigma_t 1.5 --lr 2.0 --batch_size 1024 --C_v=0.001 --sigma_v=1.0 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=2.0 > DPSUR_MNIST_eps2.log 2>&1 &` |
|  | 3.0 | 98.12% | `nohup python -u main.py --algorithm DPSUR --dataset_name MNIST --sigma_t 1.35 --lr 2.0 --batch_size 1024 --C_v=0.001 --sigma_v=0.9 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=3.0 > DPSUR_MNIST_eps3.log 2>&1 &` |
| **FMNIST** | 1.0 | 86.40% | `nohup python -u main.py --algorithm DPSUR --dataset_name FMNIST --sigma_t 4.0 --lr 4.0 --batch_size 2048 --C_v=0.001 --sigma_v=1.3 --bs_valid=256 --beta=-1 --input_norm=GroupNorm --num_groups=27 --use_scattering --eps=1.0 > DPSUR_FMNIST_eps1.log 2>&1 &` |
|  | 2.0 | 86.46% | `nohup python -u main.py --algorithm DPSUR --dataset_name FMNIST --sigma_t 2.15 --lr 4.0 --batch_size 2048 --C_v=0.001 --sigma_v=1.3 --bs_valid=256 --beta=-1 --input_norm=GroupNorm --num_groups=27 --use_scattering --eps=2.0 > DPSUR_FMNIST_eps2.log 2>&1 &` |
|  | 3.0 | 86.60% | `nohup python -u main.py --algorithm DPSUR --dataset_name FMNIST --sigma_t 2.15 --lr 4.0 --batch_size 2048 --C_v=0.001 --sigma_v=0.8 --bs_valid=256 --beta=-1 --input_norm=GroupNorm --num_groups=27 --use_scattering --eps=3.0 > DPSUR_FMNIST_eps3.log 2>&1 &` |
| **CIFAR-10** | 1.0 | 59.49% | `nohup python -u main.py --algorithm DPSUR --dataset_name CIFAR-10 --sigma_t 5.67 --lr 4.0 --batch_size 2048 --C_v=0.001 --sigma_v=1.3 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=1.0 > DPSUR_CIFAR_eps1.log 2>&1 &` |
|  | 2.0 | 66.25% | `nohup python -u main.py --algorithm DPSUR --dataset_name CIFAR-10 --sigma_t 3.5 --lr 4.0 --batch_size 2048 --C_v=0.001 --sigma_v=1.3 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=2.0 > DPSUR_CIFAR_eps2.log 2>&1 &` |
|  | 3.0 | 67.48% | `nohup python -u main.py --algorithm DPSUR --dataset_name CIFAR-10 --sigma_t 3.5 --lr 4.0 --batch_size 2048 --C_v=0.001 --sigma_v=1.1 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=3.0 > DPSUR_CIFAR_eps3.log 2>&1 &` |
| **IMDB** | 1.0 | 60.56% | `nohup python -u main.py --algorithm DPSUR --dataset_name IMDB --sigma_t 2.0 --lr 0.02 --batch_size 512 --C_v=0.001 --sigma_v=1.3 --bs_valid=256 --beta=-1 --eps=1.0 > DPSUR_IMDB_eps1.log 2>&1 &` |
|  | 2.0 | 67.56% | `nohup python -u main.py --algorithm DPSUR --dataset_name IMDB --sigma_t 1.8 --lr 0.02 --batch_size 512 --C_v=0.001 --sigma_v=1.2 --bs_valid=256 --beta=-1 --eps=2.0 > DPSUR_IMDB_eps2.log 2>&1 &` |
|  | 3.0 | 67.59% | `nohup python -u main.py --algorithm DPSUR --dataset_name IMDB --sigma_t 1.35 --lr 0.02 --batch_size 512 --C_v=0.001 --sigma_v=1.0 --bs_valid=256 --beta=-1 --eps=3.0 > DPSUR_IMDB_eps3.log 2>&1 &` |

---

### 3. DPSUR_GC

| Dataset | Privacy Budget ($\epsilon$) | Test Accuracy | Exact Command to Reproduce |
| --- | --- | --- | --- |
| **CIFAR-10** | 1.0 | 65.54% | `nohup python -u main.py --algorithm DPSUR_GC --dataset_name CIFAR-10 --sigma_t 5.67 --lr 0.1 --batch_size 2048 --C_v=0.001 --sigma_v=1.3 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --soft_c=3 --max_error=11 --eps=1.0 > DPSUR_GC_CIFAR_eps1.log 2>&1 &` |
|  | 2.0 | 69.81% | `nohup python -u main.py --algorithm DPSUR_GC --dataset_name CIFAR-10 --sigma_t 3.5 --lr 0.1 --batch_size 2048 --C_v=0.001 --sigma_v=1.3 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --soft_c=3 --max_error=11 --eps=2.0 > DPSUR_GC_CIFAR_eps2.log 2>&1 &` |
|  | 3.0 | 71.78% | `nohup python -u main.py --algorithm DPSUR_GC --dataset_name CIFAR-10 --sigma_t 3.5 --lr 0.1 --batch_size 2048 --C_v=0.001 --sigma_v=1.1 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --soft_c=3 --max_error=11 --eps=3.0 > DPSUR_GC_CIFAR_eps3.log 2>&1 &` |
| **IMDB** | 1.0 | 64.43% | `nohup python -u main.py --algorithm DPSUR_GC --dataset_name IMDB --sigma_t 2.0 --lr 0.02 --batch_size 512 --C_v=0.001 --sigma_v=1.3 --bs_valid=256 --beta=-1 --soft_c=1 --max_error=11 --eps=1.0 > DPSUR_GC_IMDB_eps1.log 2>&1 &` |
|  | 2.0 | 68.49% | `nohup python -u main.py --algorithm DPSUR_GC --dataset_name IMDB --sigma_t 1.8 --lr 0.02 --batch_size 512 --C_v=0.001 --sigma_v=1.2 --bs_valid=256 --beta=-1 --soft_c=1 --max_error=11 --eps=2.0 > DPSUR_GC_IMDB_eps2.log 2>&1 &` |
|  | 3.0 | 69.16% | `nohup python -u main.py --algorithm DPSUR_GC --dataset_name IMDB --sigma_t 1.35 --lr 0.02 --batch_size 512 --C_v=0.001 --sigma_v=1.0 --bs_valid=256 --beta=-1 --soft_c=1 --max_error=11 --eps=3.0 > DPSUR_GC_IMDB_eps3.log 2>&1 &` |

---

### 4. DPSR-CG-woSR

| Dataset | Privacy Budget ($\epsilon$) | Test Accuracy | Exact Command to Reproduce |
| --- | --- | --- | --- |
| **MNIST** | 1.0 | N/A | `nohup python -u main.py --algorithm DPSR-CG-woSR --dataset_name MNIST --sigma_t 2.5 --lr 0.1 --batch_size 1024 --C_t=1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --soft_c=8 --max_error=11 --eps=1.0 > DPSR-CG-woSR_MNIST_eps1.log 2>&1 &` |
|  | 2.0 | N/A | `nohup python -u main.py --algorithm DPSR-CG-woSR --dataset_name MNIST --sigma_t 1.5 --lr 0.1 --batch_size 1024 --C_t=1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --soft_c=8 --max_error=11 --eps=3.0 > DPSR-CG-woSR_MNIST_eps2.log 2>&1 &` |
|  | 3.0 | N/A | `nohup python -u main.py --algorithm DPSR-CG-woSR --dataset_name MNIST --sigma_t 1.35 --lr 0.1 --batch_size 1024 --C_t=1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --soft_c=8 --max_error=11 --eps=3.0 > DPSR-CG-woSR_MNIST_eps3.log 2>&1 &` |
| **FMNIST** | 1.0 | N/A | `nohup python -u main.py --algorithm DPSR-CG-woSR --dataset_name FMNIST --sigma_t 4.0 --lr 0.1 --batch_size 2048 --C_t=1 --input_norm=GroupNorm --num_groups=27 --use_scattering --soft_c=6 --max_error=11 --eps=3.0 > DPSR-CG-woSR_FMNIST_eps1.log 2>&1 &` |
|  | 2.0 | N/A | `nohup python -u main.py --algorithm DPSR-CG-woSR --dataset_name FMNIST --sigma_t 2.15 --lr 0.1 --batch_size 2048 --C_t=1 --input_norm=GroupNorm --num_groups=27 --use_scattering --soft_c=6 --max_error=11 --eps=3.0 > DPSR-CG-woSR_FMNIST_eps2.log 2>&1 &` |
|  | 3.0 | N/A | `nohup python -u main.py --algorithm DPSR-CG-woSR --dataset_name FMNIST --sigma_t 2.15 --lr 0.1 --batch_size 2048 --C_t=1 --input_norm=GroupNorm --num_groups=27 --use_scattering --soft_c=6 --max_error=11 --eps=3.0 > DPSR-CG-woSR_FMNIST_eps3.log 2>&1 &` |
| **CIFAR-10** | 1.0 | 68.41% | `nohup python -u main.py --algorithm DPSR-CG-woSR --dataset_name CIFAR-10 --sigma_t 5.67 --lr 0.1 --batch_size 2048 --C_t=1.0 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --soft_c=3 --max_error=11 --eps=3.0 > DPSR-CG-woSR_CIFAR_eps1.log 2>&1 &` |
|  | 2.0 | 71.58% | `nohup python -u main.py --algorithm DPSR-CG-woSR --dataset_name CIFAR-10 --sigma_t 3.5 --lr 0.1 --batch_size 2048 --C_t=1.0 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --soft_c=3 --max_error=11 --eps=3.0 > DPSR-CG-woSR_CIFAR_eps2.log 2>&1 &` |
|  | 3.0 | 71.85% | `nohup python -u main.py --algorithm DPSR-CG-woSR --dataset_name CIFAR-10 --sigma_t 3.5 --lr 0.1 --batch_size 2048 --C_t=1.0 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --soft_c=3 --max_error=11 --eps=3.0 > DPSR-CG-woSR_CIFAR_eps3.log 2>&1 &` |
| **IMDB** | 1.0 | 67.26% | `nohup python -u main.py --algorithm DPSR-CG-woSR --dataset_name IMDB --sigma_t 2.5 --lr 0.02 --batch_size 1024 --C_t=1.0 --soft_c=1 --max_error=11 --eps=1.0 > DPSR-CG-woSR_IMDB_eps1.log 2>&1 &` |
|  | 2.0 | 69.57% | `nohup python -u main.py --algorithm DPSR-CG-woSR --dataset_name IMDB --sigma_t 1.8 --lr 0.02 --batch_size 1024 --C_t=1.0 --soft_c=1 --max_error=11 --eps=2.0 > DPSR-CG-woSR_IMDB_eps2.log 2>&1 &` |
|  | 3.0 | 70.43% | `nohup python -u main.py --algorithm DPSR-CG-woSR --dataset_name IMDB --sigma_t 1.35 --lr 0.02 --batch_size 1024 --C_t=1.0 --soft_c=1 --max_error=11 --eps=3.0 > DPSR-CG-woSR_IMDB_eps3.log 2>&1 &` |

---

### 5. DPSGD_Matrix mechanism

| Dataset | Privacy Budget ($\epsilon$) | Test Accuracy | Exact Command to Reproduce |
| --- | --- | --- | --- |
| **MNIST** | 1.0 | 98.39% | ``nohup python -u main.py --algorithm DPSGD_Matrix --dataset_name MNIST --lr=0.1 --batch_size 1024 --C_t=1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --target_epoch=20 --epsilon=1.0 > DPSGD_Matrix_MNIST_eps1.log 2>&1 &` <br> `tail -f DPSGD_Matrix_MNIST_eps1.log`` |
|  | 2.0 | 98.66% | ``nohup python -u main.py --algorithm DPSGD_Matrix --dataset_name MNIST --lr=0.1 --batch_size 1024 --C_t=1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --target_epoch=25 --epsilon=2.0 > DPSGD_Matrix_MNIST_eps2.log 2>&1 &` <br> `tail -f DPSGD_Matrix_MNIST_eps2.log`` |
|  | 3.0 | 98.79% | ``nohup python -u main.py --algorithm DPSGD_Matrix --dataset_name MNIST --lr=0.1 --batch_size 1024 --C_t=1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --target_epoch=30 --epsilon=3.0 > DPSGD_Matrix_MNIST_eps3.log 2>&1 &` <br> `tail -f DPSGD_Matrix_MNIST_eps3.log`` |
| **FMNIST** | 1.0 | 87.16% | ``nohup python -u main.py --algorithm DPSGD_Matrix --dataset_name FMNIST --lr=0.1 --batch_size 2048 --C_t=1 --input_norm=GroupNorm --num_groups=27 --use_scattering --target_epoch=40 --epsilon=1.0 > DPSGD_Matrix_FMNIST_eps1.log 2>&1 &` <br> `tail -f DPSGD_Matrix_FMNIST_eps1.log`` |
|  | 2.0 | 87.84% | ``nohup python -u main.py --algorithm DPSGD_Matrix --dataset_name FMNIST --lr=0.1 --batch_size 2048 --C_t=1 --input_norm=GroupNorm --num_groups=27 --use_scattering --target_epoch=50 --epsilon=2.0 > DPSGD_Matrix_FMNIST_eps2.log 2>&1 &` <br> `tail -f DPSGD_Matrix_FMNIST_eps2.log`` |
|  | 3.0 | 88.32% | ``nohup python -u main.py --algorithm DPSGD_Matrix --dataset_name FMNIST --lr=0.1 --batch_size 2048 --C_t=1 --input_norm=GroupNorm --num_groups=27 --use_scattering --target_epoch=60 --epsilon=3.0 > DPSGD_Matrix_FMNIST_eps3.log 2>&1 &` <br> `tail -f DPSGD_Matrix_FMNIST_eps3.log`` |
| **CIFAR-10** | 1.0 | 59.53% | ``nohup python -u main.py --algorithm DPSGD_Matrix --dataset_name CIFAR-10 --lr=0.1 --batch_size 1024 --C_t=1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --target_epoch=40 --epsilon=1.0 > DPSGD_Matrix_CIFAR_eps1.log 2>&1 &` <br> `tail -f DPSGD_Matrix_CIFAR_eps1.log`` |
|  | 2.0 | 66.40% | ``nohup python -u main.py --algorithm DPSGD_Matrix --dataset_name CIFAR-10 --lr=0.1 --batch_size 1024 --C_t=1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --target_epoch=50 --epsilon=2.0 > DPSGD_Matrix_CIFAR_eps2.log 2>&1 &` <br> `tail -f DPSGD_Matrix_CIFAR_eps2.log`` |
|  | 3.0 | 69.21% | ``nohup python -u main.py --algorithm DPSGD_Matrix --dataset_name CIFAR-10 --lr=0.1 --batch_size 1024 --C_t=1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --target_epoch=60 --epsilon=3.0 > DPSGD_Matrix_CIFAR_eps3.log 2>&1 &` <br> `tail -f DPSGD_Matrix_CIFAR_eps3.log`` |
| **IMDB** | 1.0 | 67.51% | ``nohup python -u main.py --algorithm DPSGD_Matrix --dataset_name IMDB --lr=0.1 --batch_size 1024 --C_t=1 --target_epoch=5 --epsilon=1.0 > DPSGD_Matrix_IMDB_eps1.log 2>&1 &` <br> `tail -f DPSGD_Matrix_IMDB_eps1.log`` |
|  | 2.0 | 71.87% | ``nohup python -u main.py --algorithm DPSGD_Matrix --dataset_name IMDB --lr=0.1 --batch_size 1024 --C_t=1 --target_epoch=10 --epsilon=2.0 > DPSGD_Matrix_IMDB_eps2.log 2>&1 &` <br> `tail -f DPSGD_Matrix_IMDB_eps2.log`` |
|  | 3.0 | 72.53% | ``nohup python -u main.py --algorithm DPSGD_Matrix --dataset_name IMDB --lr=0.1 --batch_size 1024 --C_t=1 --target_epoch=15 --epsilon=3.0 > DPSGD_Matrix_IMDB_eps3.log 2>&1 &` <br> `tail -f DPSGD_Matrix_IMDB_eps3.log`` |

---

## Comparison Algorithms (DPAGD)

You can run other comparison algorithms by simply modifying the `--algorithm=[algorithm name]` parameter.

| Dataset | Privacy Budget ($\epsilon$) | Test Accuracy | Exact Command to Reproduce |
| --- | --- | --- | --- |
| **MNIST** | 1.0 | 95.91% | `python main.py --algorithm DPAGD --dataset_name MNIST --sigma_t 2.0 --lr 2.0 --batch_size 1024 --C_v=3.0 --sigma_v=1.5 --eps=1.0` |
|  | 2.0 | 97.30% | `python main.py --algorithm DPAGD --dataset_name MNIST --sigma_t 1.5 --lr 2.0 --batch_size 1024 --C_v=3.0 --sigma_v=1.5 --eps=2.0` |
|  | 3.0 | 97.52% | `python main.py --algorithm DPAGD --dataset_name MNIST --sigma_t 1.35 --lr 2.0 --batch_size 1024 --C_v=3.0 --sigma_v=1.5 --eps=3.0` |
| **FMNIST** | 1.0 | 81.26% | `python main.py --algorithm DPAGD --dataset_name FMNIST --sigma_t 4.0 --lr 4.0 --batch_size 2048 --C_v=3.0 --sigma_v=2.0 --eps=1.0` |
|  | 2.0 | 84.50% | `python main.py --algorithm DPAGD --dataset_name FMNIST --sigma_t 2.15 --lr 4.0 --batch_size 2048 --C_v=3.0 --sigma_v=2.0 --eps=2.0` |
|  | 3.0 | 86.04% | `python main.py --algorithm DPAGD --dataset_name FMNIST --sigma_t 2.15 --lr 4.0 --batch_size 2048 --C_v=3.0 --sigma_v=2.0 --eps=3.0` |
| **CIFAR-10** | 1.0 | 45.78% | `python main.py --algorithm DPAGD --dataset_name CIFAR-10 --sigma_t 11.0 --lr 4.0 --batch_size 8192 --C_v=3.0 --sigma_v=15.0 --eps=1.0` |
|  | 2.0 | 53.30% | `python main.py --algorithm DPAGD --dataset_name CIFAR-10 --sigma_t 9.0 --lr 4.0 --batch_size 8192 --C_v=3.0 --sigma_v=15.0 --eps=2.0` |
|  | 3.0 | 56.21% | `python main.py --algorithm DPAGD --dataset_name CIFAR-10 --sigma_t 5.67 --lr 4.0 --batch_size 8192 --C_v=3.0 --sigma_v=15.0 --eps=3.0` |
| **IMDB** | 1.0 | 58.72% | `python main.py --algorithm DPAGD --dataset_name IMDB --sigma_t 2.0 --lr 4.0 --batch_size 1024 --C_v=3.0 --sigma_v=5.0 --eps=1.0` |
|  | 2.0 | 63.48% | `python main.py --algorithm DPAGD --dataset_name IMDB --sigma_t 1.8 --lr 4.0 --batch_size 1024 --C_v=3.0 --sigma_v=5.0 --eps=2.0` |
|  | 3.0 | 64.59% | `python main.py --algorithm DPAGD --dataset_name IMDB --sigma_t 1.35 --lr 4.0 --batch_size 1024 --C_v=3.0 --sigma_v=5.0 --eps=3.0` |

---

## Member Inference Attacks

In Member Inference Attacks setting, we do not support scattering networks.
And for each dataset, we randomly split it into four subsets: the target training dataset, target testing dataset, shadow training dataset, and shadow testing dataset.
The ratio of the sample sizes in each subset is 2:1:2:1.

We adopt two membership inference attacks, Black-Box/Shadow ([ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models](https://arxiv.org/abs/1806.01246)) and White-Box/Partial ([Comprehensive Privacy Analysis of Deep Learning: Passive and Active White-box Inference Attacks against Centralized and Federated Learning](https://arxiv.org/abs/1812.00910)), which are the SOTA methods in membership inference attack to our knowledge.

Our target model and training parameters are consistent with those described above.
We can run MIA through adding the following to the above settings:

```bash
-- MIA=True

```
