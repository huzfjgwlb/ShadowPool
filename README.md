# SHAPOOL

Source code for ***"Inference Attacks Made Faster: Boosting Shadow Model Construction via Mixture of Experts"***.

---

## 📂 Dataset

For **CIFAR10** and **CIFAR100**, PyTorch provides built-in dataset utilities, making them easy to use.


---

## ⚡ Baseline Approach

We take **ResNet18** and **CIFAR100** as an example.

### 🔹 Step 1: Train Shadow Models from Scratch
Run the training script:
```bash
bash scripts/ori_cifar100_res18.sh
```

### 🔹 Step 2: Conduct Inference Attacks (LiRA)
Run the attack script:
```bash
bash scripts/attack_cifar100_res18.sh
```

### 📊 Step 3: Check Results
The attack results will be saved in *ori_res18-ori_cifar100.csv*.

---

## 🚀 Our Method: **SHAPOOL**

### 🔹 Step 1: Randomly Select Member Samples
To randomly select a subset from the dataset as member samples, run the following Python code:
```python
import numpy as np
random_numbers = np.random.randint(0, 50000, size=5000)
np.savez('saved_member_idx/cifar100-5000.npz', member_idx=random_numbers)
```

### 🔹 Step 2-4: Train and Attack with Shared Models
We use the following scripts for each step:
- **Train MoE-Based Shared Models**: `train_moe.py`
- **Align Trained Pathways**: `train_moe_ft.py`
- **Conduct Inference Attacks (LiRA)**: `attack_moe_ft.py`

### 📌 Alternative: Run All Steps in One Script
Instead of running Steps 2–4 separately, you can execute the following script to complete all steps in one go:
```bash
bash scripts/moe_cifar100_res18.sh
```

### 📊 Step 5: Check Results
The results will be saved in *ft_res18-moe2_cifar100.csv*.


---

## 🔗 References

1. [Canary in a Coalmine: Better Membership Inference with Ensembled Adversarial Queries](https://github.com/YuxinWenRick/canary-in-a-coalmine)
2. [Robust Mixture-of-Expert Training for Convolutional Neural Networks](https://github.com/OPTML-Group/Robust-MoE-CNN)
