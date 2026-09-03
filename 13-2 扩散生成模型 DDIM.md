## DDIM（把 DDPM 的采样公式推广到了任意时间步，并证明这种推广是合法的）

### 1. DDPM 的瓶颈：采样速度

DDPM 的生成过程严格遵循马尔可夫链：

$$x_T \to x_{T-1} \to \cdots \to x_0$$

- **步数限制**：为保证"近似高斯"假设，$T$ 常设为 $1000$。
- **耗时**：生成一张图需前向推理约 $1000$ 次网络，远慢于 GAN。
- **僵化**：训练用 $T = 1000$，采样时也几乎必须走满 $1000$ 步，难以跳步。

**问题**：能否在不重新训练的情况下，用更少的步数完成采样？

> [!note]
> 马尔可夫链（Markov Chain）就是一串连续变化的状态，并且每一步都满足马尔可夫性质。

#### 关键洞察

重新审视 DDPM 的训练目标：

$$L_{\text{simple}} = \mathbb{E}_{x_0, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta\!\left( \underbrace{\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon}_{x_t \sim q(x_t \mid x_0)},\ t \right) \right\|^2 \right]$$

- 该损失**只依赖边缘分布** $q(x_t \mid x_0)$；
- **不需要**前向联合分布 $q(x_{1:T} \mid x_0)$ 一定是"逐步马尔可夫"形式。

#### DDIM 的核心假设

**只要构造一个新的前向过程，使其边缘分布 $q(x_t \mid x_0)$ 与 DDPM 一致，就能复用同一个 $\epsilon_\theta$，但在采样时选择不同的（可跳步、非马尔可夫）路径。**

### 2. 重新定义后验分布

#### 打破马尔可夫假设

我们不再强制要求 $x_{t-1}$ 只依赖于 $x_t$。相反，我们构建一个更通用的**非马尔可夫推断分布** $q_\sigma(x_{t-1} \mid x_t, x_0)$。

**构造目标**：该分布必须满足一个硬性约束：**边缘分布一致性**。

$$\int q_\sigma(x_{t-1} \mid x_t, x_0)\, q(x_t \mid x_0)\, dx_t = q(x_{t-1} \mid x_0)$$

只要满足这一点，原本训练好的 $MSE\ Loss$ 依然有效。

#### 构造形式（Implicit Model）

DDIM 提出了一族满足上述条件的高斯分布：

$$q_\sigma(x_{t-1} \mid x_t, x_0) = \mathcal{N}\!\left( x_{t-1};\ \underbrace{\sqrt{\bar{\alpha}_{t-1}}\, x_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \frac{x_t - \sqrt{\bar{\alpha}_t}\, x_0}{\sqrt{1 - \bar{\alpha}_t}}}_{\text{均值 (Mean)}},\ \sigma_t^2\, I \right)$$

**解读**：我们将 $x_{t-1}$ 分解为 "$x_0$ 的贡献" + "指向 $x_t$ 的方向" + "随机噪声 $\sigma_t$"。

### 3. 广义采样公式推导

在推理阶段，我们无法预知真实的 $x_0$。**策略**：用神经网络的预测值 $\hat{x}_0$ 来替代公式中的 $x_0$。

#### 1. 估计 $x_0$（Denoised Observation）

根据前向公式 $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$，反解出 $x_0$：

$$\hat{x}_0(x_t) = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\, \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

#### 2. 代入均值公式，得到通用更新方程

$$x_{t-1} = \underbrace{\sqrt{\bar{\alpha}_{t-1}}\, \hat{x}_0(x_t)}_{\text{指向 } x_0 \text{ 的分量}} + \underbrace{\sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(x_t, t)}_{\text{指向 } x_t \text{ 的方向分量}} + \underbrace{\sigma_t\, \epsilon_t}_{\text{随机噪声}}$$

这个公式涵盖了从完全随机到完全确定的所有可能性，取决于 $\sigma_t$ 的取值。

> [!note]
> 因为这个公式不仅适用于 t−1，实际上适用于任意 s<t。论文为了推导方便先写成了 t−1，真正采样时才把它推广到跳步。
>
> S步时情况：
> $x_s = \sqrt{\bar\alpha_s}\hat x_0 + \sqrt{1-\bar\alpha_s\sigma_s^2},\epsilon_\theta(x_t,t)  + \sigma_s\epsilon.$
#### $\sigma$ 的选择与两种特例

上面的公式中，$\sigma_t$ 是一个我们可以自由控制的超参数。

#### Case 1：回归 DDPM

**设定**：

$$\sigma_t = \sqrt{\tilde{\beta}_t} = \sqrt{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}} \sqrt{1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}$$

**结果**：此时公式完全退化为 DDPM 的采样过程。

- 过程是**随机的**（Stochastic）。
- 必须遵循马尔可夫链。

#### Case 2：确定性采样（DDIM）

**设定**：

$$\sigma_t = 0$$

**结果**：随机噪声项消失，过程变为**完全确定**（Deterministic）。

- 后验分布退化为点质量分布（Point Mass）。
- 此时模型近似于求解常微分方程（ODE）。

#### 核心优势与性质

当 $\sigma_t = 0$ 时（即标准 DDIM），我们获得了极其重要的特性：

#### 1. 加速采样（Acceleration）

由于去除了随机游走带来的震荡，轨迹更加平滑。
- 我们可以安全地进行**跳步**（Skip Steps）。
- 例如：$1000 \to 980 \to \cdots \to 0$。
- **效果**：$10\text{-}50$ 步即可生成高质量图片（DDPM 需 $1000$ 步）。

#### 2. 一致性（Consistency）

给定相同的初始噪声 $x_T$，必定生成**同一张图片** $x_0$。不再像 DDPM 那样每次生成都不一样。

### 4. DDIM采样过程可视化理解

可以看王峰老师的这篇知乎文章：[零推导理解Diffusion和Flow Matching](https://zhuanlan.zhihu.com/p/11228697012)

> [!note]
> 有一点要注意下，文章里把 $\bar\alpha_t$简写成 $\alpha_t$ ，别搞混了。
> 


