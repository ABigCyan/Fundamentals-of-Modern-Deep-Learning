![[matrix-media-1781094114309-a793681b.png]]
这图有点问题（点点个数和位置变了），不过无伤大雅，凑合看吧～
## 1. 问题

你手上有 N 个观测点 $X = \{x_1, ..., x_N\}$。**假设**这些点是从 K 个高斯分布**混合**生成的，但**你不知道**：

- 哪些点来自哪个高斯（隐变量 z）
- 每个高斯的形状（参数 μ、Σ、π）

**目标**：估出参数 θ = (μ, Σ, π)，让模型能解释这堆点。

## 2. 参数

| 符号                    | 性质      | 含义                  |
| --------------------- | ------- | ------------------- |
| $x_i$                 | 观测变量    | 第 i 个点的坐标           |
| $z_i \in \{1,...,K\}$ | **隐变量** | 第 i 个点来自第几个高斯（看不到）  |
| $\pi_k$               | 参数      | 第 k 个高斯被选中的概率（混合系数） |
| $\mu_k$               | 参数      | 第 k 个高斯的中心          |
| $\Sigma_k$            | 参数      | 第 k 个高斯的协方差         |

## 3. 建模（联合分布）

**先验**（z 的分布）：
$$p(z_i = k) = \pi_k$$

**似然**（给定 z 后 x 怎么生成）：
$$p(x_i | z_i = k) = \mathcal{N}(x_i | \mu_k, \Sigma_k)$$

**联合**（最关键的一行）：
$$p(x_i, z_i = k) = \pi_k \cdot \mathcal{N}(x_i | \mu_k, \Sigma_k)$$

**边缘**（不管 z）：
$$p(x_i) = \sum_{k=1}^K \pi_k \, \mathcal{N}(x_i | \mu_k, \Sigma_k)$$

## 4. 目标

最大化对数似然：
$$\log p(X|\theta) = \sum_i \log p(x_i|\theta) = \sum_i \log \sum_k \pi_k \, \mathcal{N}(x_i|\mu_k,\Sigma_k)$$

**难点**：$\log$ 里面有 $\sum_k$（log-sum-exp），**没法直接求导**——因为不知道 z 是哪个，求和拆不开。

## 5. 引入 q(z)，构造 ELBO

引入任意分布 $q(z)$，对每个 $x_i$ 都有：

$$\log p(x_i) = \log \sum_{z_i} p(x_i, z_i) = \log \sum_{z_i} q(z_i) \cdot \frac{p(x_i, z_i)}{q(z_i)}$$

$$= \log \mathbb{E}_{q(z_i)}\!\left[\frac{p(x_i, z_i)}{q(z_i)}\right]$$

**Jensen 不等式**（log 是凹函数 → log E ≥ E log）：

$$\geq \mathbb{E}_{q(z_i)}\!\left[\log \frac{p(x_i, z_i)}{q(z_i)}\right] = \mathbb{E}_{q}[\log p(x_i, z_i)] - \mathbb{E}_{q}[\log q(z_i)]$$

**ELBO 定义**：
$$\mathcal{L}(q, \theta) = \sum_i \left[\mathbb{E}_{q(z_i)}[\log p(x_i, z_i|\theta)] - \mathbb{E}_{q(z_i)}[\log q(z_i)]\right]$$

**恒等式**（用 KL 散度写成等式）：
$$\log p(X|\theta) = \mathcal{L}(q, \theta) + \sum_i KL(q(z_i) \| p(z_i | x_i, \theta))$$

因为 KL ≥ 0，所以 ELBO 是 log p 的**下界**。

## 6. 求解：交替优化 ELBO

**核心思想**：固定一个、优化另一个，交替进行 → ELBO 不断抬高 → log p 也跟着抬高（因为 KL 在 E 步能到 0）。

### E 步：固定 θ，优化 q

**最优 q** 就是真后验：
$$q^*(z_i = k) = p(z_i = k | x_i, \theta) \triangleq \gamma_{ik}$$

此时 $KL(q \| p(z|x,\theta)) = 0$，ELBO 顶到 log p。

写成公式（贝叶斯）：
$$\gamma_{ik} = \frac{\pi_k \, \mathcal{N}(x_i|\mu_k,\Sigma_k)}{\sum_j \pi_j \, \mathcal{N}(x_i|\mu_j,\Sigma_j)}$$

**意义**：在当前参数下，给每个点算"属于各高斯的软概率"。

### M 步：固定 q，优化 θ

把 ELBO 关于 θ 求偏导=0（用闭式解）：

$$\mu_k^{\text{new}} = \frac{\sum_i \gamma_{ik} \, x_i}{\sum_i \gamma_{ik}}$$

$$\pi_k^{\text{new}} = \frac{\sum_i \gamma_{ik}}{N}$$

$$\Sigma_k^{\text{new}} = \frac{\sum_i \gamma_{ik} \, (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_i \gamma_{ik}}$$

**意义**：
- $\mu_k$ = "软归属于高斯 k 的点"的加权平均
- $\pi_k$ = 软归属到高斯 k 的点的占比
- $\Sigma_k$ = 同上的加权协方差

### 迭代

```
循环：
  E 步：用当前 θ 算 γ_{ik}
  M 步：用 γ_{ik} 算新 θ
  直到 θ 不再变
```

## 7. 为什么这样能 work

| 步 | 发生了什么 | ELBO 变化 | log p 变化 |
|---|---|---|---|
| **E 步** | q → 真后验，KL→0 | 顶到 log p | 不变 |
| **M 步** | 优化 θ | 抬升 | 跟着抬升（因为 KL=0）|

**ELBO 单调不降 → log p 单调不降 → 收敛**。
