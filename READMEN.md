
# DPSUR & DPSR-CB

*DPSUR: Accelerating Differentially Private Stochastic Gradient Descent Using Selective Update and Release*</br>

This repository is the official implementation of the paper:

[DPSUR: Accelerating Differentially Private Stochastic Gradient Descent Using Selective Update and Release](https://dl.acm.org/doi/10.14778/3648160.3648164)

Accepted at VLDB 2024

## Citations
The details of this pipeline are described in the following paper. If you use this code in your work, please kindly cite it. Thanks.
```bash
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


## Results

This table presents the main results from our paper. For each dataset, we target the privacy budget `epsilon={1, 2, 3, 4}` and fixed `delta=1e-5`.
For all experiments, we report the average test acc of `5` independent trials.

| Dataset | epsilon=1 | epsilon=2 | epsilon=3 | epsilon=4 |
| --- | --- | --- | --- | --- |
| MNIST | 97.93% | 98.70% | 98.88% | 98.95% |
| Fashion-MNIST | 88.38% | 89.34% | 89.71% | 90.18% |
| CIFAR-10 | 64.41% | 69.40% | 70.83% | 71.45% |
| IMDB | 66.50% | 71.02% | 72.16% | 74.14% |

## Execution Commands

### 1. DPSR-CB (New)

The following commands run the DPSR-CB algorithm across different datasets. 

#### MNIST

```bash
python -u main.py --algorithm DPSR_CB --dataset_name MNIST  --sigma_t 2.0 --lr 0.1 --batch_size 1024 --C_t=1 --beta=2.0 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --s=8 --max_error=11 --eps=1.0
python -u main.py --algorithm DPSR_CB --dataset_name MNIST  --sigma_t 1.5 --lr 0.1 --batch_size 1024 --C_t=1 --beta=2.0 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --s=8 --max_error=11 --eps=2.0
python -u main.py --algorithm DPSR_CB --dataset_name MNIST  --sigma_t 1.35 --lr 0.1 --batch_size 1024 --C_t=1 --beta=2.0 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --s=8 --max_error=11 --eps=3.0

```

#### FMNIST

```bash
python -u main.py --algorithm DPSR_CB --dataset_name FMNIST  --sigma_t 4.0 --lr 0.1 --batch_size 2048 --C_t=1 --beta=2 --input_norm=GroupNorm --num_groups=27 --use_scattering --s=8 --max_error=11 --eps=1.0
python -u main.py --algorithm DPSR_CB --dataset_name FMNIST  --sigma_t 2.15 --lr 0.1 --batch_size 2048 --C_t=1 --beta=2 --input_norm=GroupNorm --num_groups=27 --use_scattering --s=8 --max_error=11 --eps=2.0
python -u main.py --algorithm DPSR_CB --dataset_name FMNIST  --sigma_t 2.15 --lr 0.1 --batch_size 2048 --C_t=1 --beta=2 --input_norm=GroupNorm --num_groups=27 --use_scattering --s=8 --max_error=11 --eps=3.0

```



#### CIFAR-10

```bash
python main.py --algorithm DPSR_CB --dataset_name CIFAR-10 --sigma_t 3.5 --lr 0.1 --batch_size 2048 --C_t=1.0 --beta=2 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --s=3 --max_error=11 --eps=2.0
python main.py --algorithm DPSR_CB --dataset_name CIFAR-10 --sigma_t 3.5 --lr 0.1 --batch_size 2048 --C_t=1.0 --beta=2 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --s=3 --max_error=11 --eps=3.0

```



#### IMDB

```bash
python -u main.py --algorithm DPSR_CB --dataset_name IMDB  --sigma_t 2.0 --lr 0.02  --batch_size 1024 --C_t=1.0 --beta=2 --s=1 --max_error=11 --eps=1.0
python -u main.py --algorithm DPSR_CB --dataset_name IMDB  --sigma_t 1.8 --lr 0.02  --batch_size 1024 --C_t=1.0 --beta=2 --s=1 --max_error=11 --eps=2.0
python -u main.py --algorithm DPSR_CB --dataset_name IMDB  --sigma_t 1.35 --lr 0.02 --batch_size 1024 --C_t=1.0 --beta=2 --s=1 --max_error=11 --eps=3.0

```

---

### 2. DPSUR-GC (New)

#### MNIST

```bash
python -u main.py --algorithm DPSUR_GC --dataset_name MNIST  --sigma_t 2.0 --lr 0.1 --batch_size 1024 --C_t=1 --C_v=0.001 --sigma_v=1.3 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --s=8 --max_error=11 --eps=1.0
python -u main.py --algorithm DPSUR_GC --dataset_name MNIST  --sigma_t 1.5 --lr 0.1 --batch_size 1024 --C_t=1 --C_v=0.001 --sigma_v=1.0 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --s=8 --max_error=11 --eps=2.0
python -u main.py --algorithm DPSUR_GC --dataset_name MNIST  --sigma_t 1.35 --lr 0.1 --batch_size 1024 --C_t=1 --C_v=0.001 --sigma_v=0.9 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --s=8 --max_error=11 --eps=3.0

```



#### IMDB

```bash
python -u main.py --algorithm DPSUR_GC --dataset_name IMDB  --sigma_t 2.5 --lr 0.02  --batch_size 1024 --C_t=1.0 --C_v=0.001 --sigma_v=1.4 --bs_valid=256 --beta=-1 --eps=1.0
python -u main.py --algorithm DPSUR_GC --dataset_name IMDB  --sigma_t 1.8 --lr 0.02  --batch_size 1024 --C_t=1.0 --C_v=0.001 --sigma_v=1.2 --bs_valid=256 --beta=-1 --eps=2.0
python -u main.py --algorithm DPSUR_GC --dataset_name IMDB  --sigma_t 1.35 --lr 0.02 --batch_size 1024 --C_t=1.0 --C_v=0.001 --sigma_v=1.0 --bs_valid=256 --beta=-1 --eps=3.0

```



---

### 3. DPIS-GC (DPIS_GA) (New)

#### MNIST

```bash
python -u main.py --algorithm DPIS_GA --dataset_name MNIST  --sigma_t 2.0 --lr 0.1 --batch_size 1024 --C_t=1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --s=8 --max_error=11 --eps=1.0
python -u main.py --algorithm DPIS_GA --dataset_name MNIST  --sigma_t 1.5 --lr 0.1 --batch_size 1024 --C_t=1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --s=8 --max_error=11 --eps=2.0
python -u main.py --algorithm DPIS_GA --dataset_name MNIST  --sigma_t 1.35 --lr 0.1 --batch_size 1024 --C_t=1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --s=8 --max_error=11 --eps=3.0

```



#### IMDB

```bash
python -u main.py --algorithm DPIS_GA --dataset_name IMDB  --sigma_t 2.5 --lr 0.02  --batch_size 1024 --C_t=1.0 --s=1 --max_error=11 --eps=1.0
python -u main.py --algorithm DPIS_GA --dataset_name IMDB  --sigma_t 1.8 --lr 0.02  --batch_size 1024 --C_t=1.0 --s=1 --max_error=11 --eps=2.0
python -u main.py --algorithm DPIS_GA --dataset_name IMDB  --sigma_t 1.35 --lr 0.02 --batch_size 1024 --C_t=1.0 --s=1 --max_error=11 --eps=3.0

```



---

### 4. DPSUR (Original)

To reproduce the results for linear ScatterNet models with the original DPSUR algorithm:

#### MNIST

```bash
python main.py --algorithm DPSUR --dataset_name MNIST  --sigma_t 2.0 --lr 2.0 --batch_size 1024  --C_v=0.001 --sigma_v=1.3 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=1.0
python main.py --algorithm DPSUR --dataset_name MNIST  --sigma_t 1.5 --lr 2.0 --batch_size 1024  --C_v=0.001 --sigma_v=1.0 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=2.0
python main.py --algorithm DPSUR --dataset_name MNIST  --sigma_t 1.35 --lr 2.0 --batch_size 1024 --C_v=0.001 --sigma_v=0.9 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=3.0
python main.py --algorithm DPSUR --dataset_name MNIST  --sigma_t 1.35 --lr 2.0 --batch_size 1024 --C_v=0.001 --sigma_v=0.8 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=4.0

```

#### FMNIST

```bash
python main.py --algorithm DPSUR --dataset_name FMNIST  --sigma 4.0 --lr 4.0  --batch_size 2048 --C_v=0.001 --sigma_v=1.3 --bs_valid=256 --beta=-1 --input_norm=GroupNorm --num_groups=27 --use_scattering --eps=1.0
python main.py --algorithm DPSUR --dataset_name FMNIST  --sigma 2.15 --lr 4.0 --batch_size 2048 --C_v=0.001 --sigma_v=1.3 --bs_valid=256 --beta=-1 --input_norm=GroupNorm --num_groups=27 --use_scattering --eps=2.0
python main.py --algorithm DPSUR --dataset_name FMNIST  --sigma 2.15 --lr 4.0 --batch_size 2048 --C_v=0.001 --sigma_v=0.8 --bs_valid=256 --beta=-1 --input_norm=GroupNorm --num_groups=27 --use_scattering --eps=3.0
python main.py --algorithm DPSUR --dataset_name FMNIST  --sigma 2.15 --lr 4.0 --batch_size 2048 --C_v=0.001 --sigma_v=0.8 --bs_valid=256 --beta=-1 --input_norm=GroupNorm --num_groups=27 --use_scattering --eps=4.0

```

#### CIFAR-10

```bash
python main.py --algorithm DPSUR --dataset_name CIFAR-10 --sigma 11.0 --lr 4.0 --batch_size 8192 --C_v=0.001 --sigma_v=1.3 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=1.0
python main.py --algorithm DPSUR --dataset_name CIFAR-10 --sigma 9.0  --lr 4.0 --batch_size 8192 --C_v=0.001 --sigma_v=1.3 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=2.0
python main.py --algorithm DPSUR --dataset_name CIFAR-10 --sigma 5.67 --lr 4.0 --batch_size 8192 --C_v=0.001 --sigma_v=1.1 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=3.0
python main.py --algorithm DPSUR --dataset_name CIFAR-10 --sigma 5.67 --lr 4.0 --batch_size 8192 --C_v=0.001 --sigma_v=1.1 --bs_valid=256 --beta=-1 --input_norm=BN --bn_noise_multiplier=8 --use_scattering --eps=4.0

```

#### IMDB

IMDB does not support ScatterNet models

```bash
python main.py --algorithm DPSUR --dataset_name IMDB  --sigma 2.0 --lr 0.02  --batch_size 1024 --C_v=0.001 --sigma_v=1.3 --bs_valid=256 --beta=-1 --eps=1.0
python main.py --algorithm DPSUR --dataset_name IMDB  --sigma 1.8 --lr 0.02  --batch_size 1024 --C_v=0.001 --sigma_v=1.2 --bs_valid=256 --beta=-1 --eps=2.0
python main.py --algorithm DPSUR --dataset_name IMDB  --sigma 1.35 --lr 0.02 --batch_size 1024 --C_v=0.001 --sigma_v=1.0 --bs_valid=256 --beta=-1 --eps=3.0
python main.py --algorithm DPSUR --dataset_name IMDB  --sigma 1.23 --lr 0.02 --batch_size 1024 --C_v=0.001 --sigma_v=0.9 --bs_valid=256 --beta=-1 --eps=4.0

```

---

## Comparison Algorithms

You can run other comparison algorithms by simply modifying the '--algorithm=[algorithm name]' parameter.

### DPAGD

#### MNIST

```bash
python main.py --algorithm DPAGD --dataset_name MNIST  --sigma_t 2.0 --lr 2.0 --batch_size 1024 --C_v=3.0 --sigma_v=1.5  --eps=1.0
python main.py --algorithm DPAGD --dataset_name MNIST  --sigma_t 1.5 --lr 2.0 --batch_size 1024 --C_v=3.0 --sigma_v=1.5 --eps=2.0
python main.py --algorithm DPAGD --dataset_name MNIST  --sigma_t 1.35 --lr 2.0 --batch_size 1024 --C_v=3.0 --sigma_v=1.5  --eps=3.0
python main.py --algorithm DPAGD --dataset_name MNIST  --sigma_t 1.35 --lr 2.0 --batch_size 1024 --C_v=3.0 --sigma_v=1.5 --eps=4.0

```

#### FMNIST

```bash
python main.py --algorithm DPAGD --dataset_name FMNIST  --sigma_t 4.0 --lr 4.0  --batch_size 2048 --C_v=3.0 --sigma_v=2.0   --eps=1.0
python main.py --algorithm DPAGD --dataset_name FMNIST  --sigma_t 2.15 --lr 4.0 --batch_size 2048 --C_v=3.0 --sigma_v=2.0  --eps=2.0
python main.py --algorithm DPAGD --dataset_name FMNIST  --sigma_t 2.15 --lr 4.0 --batch_size 2048 --C_v=3.0 --sigma_v=2.0   --eps=3.0
python main.py --algorithm DPAGD --dataset_name FMNIST  --sigma_t 2.15 --lr 4.0 --batch_size 2048 --C_v=3.0 --sigma_v=2.0  --eps=4.0

```

#### CIFAR-10

```bash
python main.py --algorithm DPAGD --dataset_name CIFAR-10 --sigma_t 11.0 --lr 4.0 --batch_size 8192 --C_v=3.0 --sigma_v=15.0 --eps=1.0
python main.py --algorithm DPAGD --dataset_name CIFAR-10 --sigma_t 9.0  --lr 4.0 --batch_size 8192 --C_v=3.0 --sigma_v=15.0 --eps=2.0
python main.py --algorithm DPAGD --dataset_name CIFAR-10 --sigma_t 5.67 --lr 4.0 --batch_size 8192 --C_v=3.0 --sigma_v=15.0  --eps=3.0
python main.py --algorithm DPAGD --dataset_name CIFAR-10 --sigma_t 5.67 --lr 4.0 --batch_size 8192 --C_v=3.0 --sigma_v=15.0  --eps=4.0

```

#### IMDB

IMDB does not support ScatterNet models

```bash
python main.py --algorithm DPAGD --dataset_name IMDB  --sigma_t 2.0  --lr 4.0  --batch_size 1024 --C_v=3.0 --sigma_v=5.0  --eps=1.0
python main.py --algorithm DPAGD --dataset_name IMDB  --sigma_t 1.8 --lr 4.0  --batch_size 1024 --C_v=3.0 --sigma_v=5.0  --eps=2.0
python main.py --algorithm DPAGD --dataset_name IMDB  --sigma_t 1.35 --lr 4.0 --batch_size 1024 --C_v=3.0 --sigma_v=5.0  --eps=3.0
python main.py --algorithm DPAGD --dataset_name IMDB  --sigma_t 1.23 --lr 4.0 --batch_size 1024 --C_v=3.0 --sigma_v=5.0  --eps=4.0

```

## Member Inference Attacks

In Member Inference Attacks setting, we do not support scattering networks.
And for each dataset, we randomly split it into four subsets: the target training dataset, target testing dataset, shadow training dataset, and shadow testing dataset.
The ratio of the sample sizes in each subset is 2:1:2:1.

We adopt two membership inference attacks, Black-Box/Shadow ([ML-Leaks: Model and Data Independent
Membership Inference Attacks and Defenses on Machine Learning Models](https://arxiv.org/abs/1806.01246)
and White-Box/Partial ([Comprehensive Privacy Analysis of Deep Learning: Passive and Active White-box Inference Attacks against Centralized and Federated Learning
](https://arxiv.org/abs/1812.00910)) , which are the SOTA methods in membership inference attack to our knowledge.

Our target model and training parameters are consistent with those described above.
We can run MIA through adding the following to the above settings:

```bash
-- MIA=True

```
