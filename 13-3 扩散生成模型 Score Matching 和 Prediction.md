## Score Matching
### Score Matching 在做什么？

#### 生成模型的困境

- 生成模型的目标：根据真实样本 $X$ 建模真实分布 $p(x)$。
- 直接基于极大似然估计优化通常过于困难。

#### 转换思路：刻画"变化"而非"分布"

- 不去刻画分布 $p(x)$ 本身，而是刻画它在每个位置的**变化程度**。
- **直观假设**：若两个分布在任意位置的梯度一致，它们的形状也会十分接近。
- **梯度**：指向概率密度增长最快的方向。

#### 基于梯度的采样策略

如果我们拥有了空间内任意位置对于分布梯度的刻画，采样过程就变得十分简单：

#### 迭代过程：

1. 在空间中任意采样一个初始样本（随机噪声）。
2. 将其沿**梯度方向**移动一段距离（向高密度区域靠近）。
3. 添加一个**随机扰动**（注入噪声）。
4. 经过多次迭代，样本逐渐收敛至真实数据分布附近。

#### 优化目标的选择

出于**训练稳定性**和**数值尺度**的考量，我们通常不去直接学习分布的密度梯度 $\nabla_x p(x)$。

#### Score Function 的定义

相反，我们选择学习**对数密度的梯度**，并将其定义为 **Score Function**：

$$s(x) \triangleq \nabla_x \log p(x) = \frac{\nabla_x p(x)}{p(x)}$$

- 对数操作拉平了数量级，使数值范围更适合神经网络处理。
- 我们的目标即是训练模型 $s_\theta(x)$ 逼近 $\nabla_x \log p(x)$。

> [!note]
> 最优化里加 log，是为了方便优化一个关于参数 θ的目标；Score Matching 里的 log，则是概率论中定义 Score 的方式，然后研究这个函数关于随机变量 xx 的空间梯度。
#### 基于 Score 理解 Diffusion Model

#### 数据流形的困境

基于**流形假设**，高维空间中绝大多数位置的概率密度 $p(x)$ 都接近于 $0$。

- 在这些低密度区域，Score Function 的数值十分不稳定或无定义。
- **后果**：模型无法在这些区域获得有效的梯度指引，不知道如何移动回数据流形。

#### 引入 Diffusion：噪声平滑

我们不直接学习原分布，而是学习被**高斯噪声**扰动后的分布 $p_\sigma(x)$：

- **直观理解**：噪声将真实分布"糊开"到了整个空间。
- **数学性质**：使得空间内任意位置概率非 $0$（$p_\sigma(x) > 0$）。
- **意义**：Score 在全空间处处有定义且可学习，保证了采样路径的存在性。

#### 引入时间条件

- **问题**：不同噪声强度的分布 $p_\sigma$ 形状不同，单一模型无法刻画所有梯度。
- **解决**：引入参数 $t$ 指示当前噪声强度，学习条件梯度

$$s_\theta(x, t) \approx \nabla_x \log p_{\sigma_t}(x)$$
#### 退火 朗之万 采样

1. **初始化**：从高斯分布中采样初始噪声 $x_T$。
2. **迭代去噪**：
   - 利用 $s_\theta(x_t, t)$ 预测当前分布梯度方向。
   - 沿梯度更新得到 $x_{t-1}$，同时降低噪声等级。
1. **结果**：随 $t \to 0$，样本从纯噪声平滑转移至真实数据流形。

### SDE 视角下的前向加噪

#### 前向过程回顾

在 DDPM 中，离散时间的前向加噪公式为：

$$x_t = \sqrt{1 - \beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_{t-1}$$

#### 连续时间极限：SDE 形式

如果我们将时间 $t$ 视为连续变量，上述离散过程可以推广为如下形式的 **SDE (Stochastic Differential Equation)**：

$$dx = -\frac{1}{2}\beta(t)\, x\, dt + \sqrt{\beta(t)}\, dw$$

其中：

- $dx$：样本 $x$ 在微小时间内的变化量。
- $-\frac{1}{2}\beta(t)\, x\, dt$：**漂移项**，使数据逐渐向原点收缩。（方差收敛） 
- $\sqrt{\beta(t)}\, dw$：**扩散项**，$dw$ 是标准布朗运动增量（随机噪声）。

> [!note]
> 过程我猜测是两边减去xt-1，对右边第一项进行泰勒展开得来的

### 反向 SDE——反向去噪

#### 逆向过程的数学保证（Anderson, 1982）

- 对于任意满足特定条件的 SDE，都存在一个**反向 SDE**。
- 该反向 SDE 描述了时间倒回去的扩散过程。

#### 反向 SDE 公式

对应于前向过程，反向 SDE 的形式为：

$$dx = \left[ -\frac{1}{2}\beta(t)\, x - \beta(t)\, \nabla_x \log p_t(x) \right] dt + \sqrt{\beta(t)}\, d\bar{w}$$

- $dt$：时间反向流逝。
- $d\bar{w}$：反向时间下的标准布朗运动。
- $\nabla_x \log p_t(x)$：**唯一未知项**，即 **Score Function**。

#### 结论

模型的任务就是通过神经网络去逼近这个 Score，从而能够求解反向 SDE 进行生成。

### 训练目标

#### 困境：边缘分布 Score 难以计算

我们的理想目标是学习边缘分布的 Score $\nabla_{x_t} \log p_t(x_t)$。

- 然而，$p_t(x_t) = \int q(x_t \mid x_0)\, p_{\text{data}}(x_0)\, dx_0$ 涉及对整个数据分布的积分。
- 这使得直接计算并优化 $\nabla_{x_t} \log p_t(x_t)$ 在计算上是不可行的。

#### 转机：Denoising Score Matching (DSM)

Vincent (2011) 证明了一个关键结论：而在**期望意义**下，匹配条件分布的 Score 等价于匹配边缘分布的 Score。我们将优化目标转换为：

$$\mathcal{L} = \mathbb{E}_{x_0, x_t} \left[ \left\| s_\theta(x_t, t) - \nabla_{x_t} \log q(x_t \mid x_0) \right\|^2 \right]$$

**优势**：每一个 $q(x_t \mid x_0)$ 都是我们可以显式写出的高斯分布，因此其梯度容易计算。

#### 推导：条件 Score 的解析形式

已知 $q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\ \sqrt{\bar{\alpha}_t}\, x_0,\ (1 - \bar{\alpha}_t) \mathbf{I}\right)$，其对数梯度为：

$$\nabla_{x_t} \log q(x_t \mid x_0) = \nabla_{x_t} \left( -\frac{\|x_t - \sqrt{\bar{\alpha}_t}\, x_0\|^2}{2(1 - \bar{\alpha}_t)} \right)$$

$$= -\frac{x_t - \sqrt{\bar{\alpha}_t}\, x_0}{1 - \bar{\alpha}_t}$$

利用重参数化 $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$，代入得：

$$\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{\sqrt{1 - \bar{\alpha}_t}\, \epsilon}{1 - \bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$$

#### 结论：Score 即噪声

Score Function 与加入的噪声 $\epsilon$ 之间呈现**线性关系**。为此，我们将网络参数化为预测噪声 $\epsilon_\theta$：

$$s_\theta(x_t, t) \approx -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

#### 物理意义统一：

- 预测 Score Function **等价于**预测当前图像上的噪声 $\epsilon$。
- 这在数学上证明了 DDPM 预测噪声目标的合理性。

### SDE 视角下 DDPM 的采样过程

#### SDE 数值解法

对于标准的 DDPM，其采样过程本质上是求解反向 SDE：

$$dx = \left[ -\frac{1}{2}\beta(t)\, x - \beta(t)\, s_\theta(x, t) \right] dt + \sqrt{\beta(t)}\, d\bar{w}$$

使用 Euler-Maruyama 等数值积分方法离散化得：

$$x_{t-1} \approx x_t - \left[ -\frac{1}{2}\beta_t\, x_t - \beta_t\, s_\theta(x_t, t) \right] \Delta t + \sqrt{\beta_t}\, \Delta \bar{w}$$

如果我们将 $s_\theta$ 替换为 $-\dfrac{\epsilon_\theta}{\sqrt{1 - \bar{\alpha}_t}}$ 并整理系数，会发现它与 DDPM 的采样公式**形式一致**，仅存在系数定义的细微差别。

至此我们证明了 **求解反向 SDE** 和 **DDPM 采样**之间的等价性。这为理解扩散模型提供了更广阔的随机过程视角。

### ODE 视角下 DDIM 的采样过程

#### 关键结论：概率流 ODE

研究发现，对于前述的 SDE，存在一个对应的常微分方程（ODE）：

$$dx = \left[ -\frac{1}{2}\beta(t)\, x - \frac{1}{2}\beta(t) \underbrace{\nabla_x \log p_t(x)}_{\text{Score Function}} \right] dt$$

- **分布一致性**：该 ODE 定义的演化路径，其边缘分布 $p_t(x)$ 与原来的 SDE 完全相同。
- **确定性**：方程中**消去了随机噪声项** $d\bar{w}$，意味着从噪声到数据的映射是确定性的。

#### 从 ODE 到 DDIM

- **推导**：通过一些特定的数学变换及离散化处理，我们可以从上述 ODE 推导出 DDIM 的采样公式。
- **等价性**：由此证明了 **求解反向 ODE** 和 **DDIM 采样**之间的等价性。

## Prediction
### $\epsilon$-pred vs. $x$-pred

#### 两种预测范式

- **$\epsilon$-prediction**：DDPM 和 DDIM 的标准做法。即模型直接预测噪声 $\epsilon$，并且将优化目标设为预测噪声与实际噪声的 MSE。
- **$x_0$-prediction**：让模型输出 $x_0$，即直接从加噪后的样本中预测干净样本。

#### 数学本质：线性等价

根据前向加噪公式，$\epsilon$ 和 $x_0$ 在给定 $x_t$ 的情况下呈**线性关系**：

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon \quad \Rightarrow \quad x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\, \epsilon}{\sqrt{\bar{\alpha}_t}}$$

**结论**：理论上预测 $\epsilon$ 和预测 $x_0$ 是完全等价的任务。但在**实际工程**中，这两种参数化方式的训练稳定性存在显著差异。

#### $\epsilon$-pred：预测噪声

#### 高噪声区域（$t \to T$）

- **数理特征**：此时 $x_t \approx \sqrt{1 - \bar{\alpha}_t}\, \epsilon$，信噪比极低。
- **优势**：输入主要是噪声，预测目标也是噪声。噪声特征十分**显性**，模型容易捕捉并学习分布的大致结构。

#### 低噪声区域（$t \to 0$）

- **数理特征**：噪声项系数 $\sqrt{1 - \bar{\alpha}_t}$ 趋近于 $0$。
- **劣势**：模型需要从强烈的图像信号中分离出极微弱的噪声。这在数值上较为困难，导致**训练信号很弱**，可能影响最终生成图像的细节纹理。

#### $x$-pred：预测原图

#### 高噪声区域（$t \to T$）

- **现象**：样本 $x_t$ 几乎为纯随机高斯噪声。
- **劣势**：模型需从毫无信息的噪声中凭空"猜"出具体的原图 $x_0$。由于不确定性极高，模型通常只能输出数据集的平均值，导致训练**极其不稳定**。

#### 低噪声区域（$t \to 0$）

- **现象**：$x_t \approx x_0$，输入与预测目标高度相似。
- **优势**：预测任务退化为近似恒等映射。这对神经网络非常简单，模型容易学到**高频样本细节**。 

#### $v$-pred：结合两者

#### 定义与动机

为融合两种预测方式的优点，Salimans (2022) 提出了 $v$-prediction。定义目标 $v_t$ 为：

$$v_t \equiv \sqrt{\bar{\alpha}_t}\, \epsilon - \sqrt{1 - \bar{\alpha}_t}\, x_0$$

$\epsilon$ 和 $x_0$ 均可通过 $v_t$ 进行线性恢复，数学上与前述方式完全等价。

| | (a) $x$-pred | (b) $\epsilon$-pred | (c) $v$-pred |
|---|---|---|---|
| 网络输出 | $x_\theta := \mathrm{net}_\theta(z_t, t)$ | $\epsilon_\theta := \mathrm{net}_\theta(z_t, t)$ | $v_\theta := \mathrm{net}_\theta(z_t, t)$ |
| (1) $x$-loss | $\mathbb{E}\left\|x_0 - x\right\|^2$，$x_\theta$ | $x_\theta = (z_t - (1-t)\epsilon_\theta)/t$ | $x_\theta = (1-t)v_\theta + z_t$ |
| (2) $\epsilon$-loss | $\mathbb{E}\left\|\epsilon_\theta - \epsilon\right\|^2$，$\epsilon_\theta = (z_t - t x_\theta)/(1-t)$ | $\epsilon_\theta$ | $\epsilon_\theta = z_t - t v_\theta$ |
| (3) $v$-loss | $\mathbb{E}\left\|v_\theta - v\right\|^2$，$v_\theta = (x_\theta - z_t)/(1-t)$ | $v_\theta = (z_t - \epsilon_\theta)/t$ | $v_\theta$ |

**图：不同预测目标形式总结**

#### 自适应特性

$v$-pred 通过训练权重自动平衡关注点：

- $t \to T$（高噪）：$v_t \approx \epsilon$，$v$-pred 退化为 $\epsilon$-pred，学习容易。
- $t \to 0$（低噪）：$v_t \approx -x_0$，$v$-pred 退化为 $x$-pred，细节恢复好。

#### 结论

兼具了训练稳定性和生成质量。

### 高维空间下的拟合挑战
#### Back to Basics (He et al., 2024)

该论文详细对比了不同 Parameterization 和 Loss 的组合。

#### 高维空间下的低维流形实验

**设置**：固定模型参数量，统一使用 $v$-loss 训练。

**观察**：随着空间维度 $D$ 的提高：

- $\epsilon$-pred / $v$-pred：模型无法有效学习，能力逐渐退化直至**完全失效**。
- $x$-pred：即使在高维空间中，依然能正确恢复出数据流形。

这暗示在高维复杂分布建模中，$x$-prediction 可能具有本质优势。

> [!note]
> 当数据的**环境维度 D** 远大于数据真正的**内在维度 d**，并且网络存在信息瓶颈时，x-prediction 可以显著优于 v-prediction。
