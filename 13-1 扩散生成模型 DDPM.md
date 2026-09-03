## 一、引言
### 1.1 什么是 Generative Model？

#### 核心直觉

生成模型的本质是学习数据的**概率分布**。

- **目标**：从有限样本 $X$ 中还原真实分布 $U$。
- **举例**：$X_{\text{Cat}}$（你的数据集） $\to$ $U_{\text{Cat}}$（所有可能的猫）。

#### 形式化定义

给定样本集合 $X$ 及标签 $Y$，学习：

- **联合分布** $P(X, Y)$（有监督）
- **数据分布** $P(X)$（无监督 / 生成任务）

![[Pasted image 20260722205124.png]]
### 1.2 物理灵感：扩散现象

#### 自然界的熵增

Diffusion（扩散）源自非平衡热力学。

- **现象**：粒子因布朗运动，自发从高浓度区域 $\to$ 低浓度区域。
- **结果**：任何分布最终都会被"模糊"、"抹平"。
- **极限**：在无限空间中，最终趋向于**高斯分布**（Gaussian Distribution）。

"直觉上，这是一个信息丢失、变得无序的过程。"

### 1.3 Diffusion Model 的核心思想

#### 一个天才的问题

如果物理扩散是不断叠加高斯噪声直到图片不可辨认……如果我们能学会这个过程的"逆过程"会怎样？

#### The Forward Process（前向）

$$x_0 \to x_1 \to \cdots \to x_T$$

图片 $\xrightarrow{+\text{Noise}} \cdots \xrightarrow{+\text{Noise}}$ 纯噪声

#### The Generative Backward Process（逆向）

$$x_0 \leftarrow x_1 \leftarrow \cdots \leftarrow x_T$$

纯噪声 $\xrightarrow{\text{Denoise}} \cdots \xrightarrow{\text{Denoise}}$ 图片

#### Core Idea

从"噪声"中逐渐恢复出"数据"。


## 二、为什么需要 Diffusion？
### 2.1 高维建模的难点

#### 维度爆炸与有效数据

以 $255 \times 255$ RGB 图像为例：

- 维度极高（$D \approx 200{,}000$）。
- 绝大部分区域充满**无意义的噪声**。

#### 流形假设（Manifold Hypothesis）

**假设**：数据分布在嵌入于高维空间中的**低维、非线性流形**上（$d \ll D$）。

- **直觉**：如同三维空间中"卷曲"的二维平面。
- **观察视角**：点在三维空间中。
- **流形视角**：点在二维平面上。

![[Pasted image 20260722210645.png|596]]
### 传统模型（VAE/GAN）做了什么？
#### 核心机制

试图学习从低维隐空间 $Z$（流形）到真实样本空间 $X$ 的**单步直接映射**。
#### 优势：快与线性 

- 生成速度快（一次采样）。
- 流形空间具有良好的**线性性质**。
- 支持**线性插值**来平滑改变样本特征。
#### 缺点：学习过于困难 

- **VAE**：引入 KL 散度约束 $\to$ 损失细节 $\to$ **图像模糊**。
- **GAN**：对抗训练不稳定 $\to$ 难以学习全部映射 $\to$ **模式坍塌**（Mode Collapse）。

> [!note]
> 模式坍塌 = Generator 找到了一个能骗过 Discriminator 的"捷径"，于是不断生成同一种（或少数几种）样本，导致生成结果缺乏多样性。

### 2.3 一步生成 vs 逐步修正

#### 全新的建模思路

不再直接学习 $Z \to X$ 的映射，而是把生成改写为："从简单噪声分布出发的多步逐渐修正"。

- **本质**：在样本空间上学习一条**逐步逼近**目标分布的变换路径。
- **过程**：标准高斯噪声 $\to$ 逐步去噪 $\to$ **复杂结构**数据。
- **优势**：兼具细节与多样性，避免了模式坍塌。

## 三、深入 Diffusion：

### 3.1 算法宏观概览

#### Diffusion 的两阶段

- **前向过程（Forward Process） $q$**：模拟物理扩散，将图像逐渐混入噪声，直到变为纯高斯噪声。
- **反向过程（Reverse Process） $p$**：学习去噪网络，从噪声中逐步恢复图像结构。

#### 主要目标

训练一个神经网络来模拟反向过程 $p_\theta$，即"学会如何去噪"。

### 3.2 前向加噪：定义与设计初衷

#### 形式化定义 $q(x_t \mid x_{t-1})$

前向过程是一个**固定的马尔可夫链**，其转移概率设计为高斯分布：

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1 - \beta_t}\, x_{t-1},\ \beta_t I\right)$$

#### 重参数化采样（[[11-3 高斯分布-对数密度是二次型的分布]]的线性变换）

$$x_t = \sqrt{1 - \beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_t$$

注：$\beta_t$ 是预设的噪声系数（Noise Schedule），通常随 $t$ 递增。

#### 为什么要用 $\sqrt{1 - \beta_t}$？

为什么不直接 $x_t = x_{t-1} + \beta_t \epsilon_t$？

**目标：方差恒定（Variance Preservation）**

- 假设输入 $x_{t-1}$ 的方差为 $1$。
- 我们希望加噪后 $x_t$ 的方差**仍保持为 $1$**，防止多步后数值爆炸。
- 验证：$\left(\sqrt{1 - \beta_t}\right)^2 + \left(\sqrt{\beta_t}\right)^2 = 1$。

### 3.3 前向加噪：任意步采样推导

#### 记号定义

令 $\alpha_t = 1 - \beta_t$，并定义累乘系数 $\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$。

利用高斯分布的可加性 $\mathcal{N}(0, \sigma_1^2) + \mathcal{N}(0, \sigma_2^2) \sim \mathcal{N}(0, \sigma_1^2 + \sigma_2^2)$，我们可以逐步展开递归式：

$$x_t = \sqrt{\alpha_t}\, x_{t-1} + \sqrt{1 - \alpha_t}\, \epsilon_t \quad (\epsilon_t \sim \mathcal{N}(0, I))$$

$$= \sqrt{\alpha_t} \Big( \underbrace{\sqrt{\alpha_{t-1}}\, x_{t-2} + \sqrt{1 - \alpha_{t-1}}\, \epsilon_{t-1}}_{x_{t-1}} \Big) + \sqrt{1 - \alpha_t}\, \epsilon_t$$

$$= \sqrt{\alpha_t \alpha_{t-1}}\, x_{t-2} + \underbrace{\sqrt{\alpha_t (1 - \alpha_{t-1})}\, \epsilon_{t-1} + \sqrt{1 - \alpha_t}\, \epsilon_t}_{\text{两个独立高斯噪声的线性组合}}$$

#### 高斯合并技巧（关键一步）

上述噪声项的方差合并为：

$$\alpha_t (1 - \alpha_{t-1}) + (1 - \alpha_t) = \alpha_t - \alpha_t \alpha_{t-1} + 1 - \alpha_t = 1 - \alpha_t \alpha_{t-1}$$

因此，两项合并为一个新的标准高斯噪声 $\bar{\epsilon}$：

$$\longrightarrow \sqrt{1 - \alpha_t \alpha_{t-1}}\, \bar{\epsilon}$$

#### 继续递归至 $t = 0$：

$$\boxed{x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon} \quad \text{where } \epsilon \sim \mathcal{N}(0, I)$$

### 3.4 反向去噪：目标与直觉

#### 核心目标

前向过程是将结构化信息逐步破坏为噪声。**反向过程（Reverse Process）** 则是要学习一个参数化的概率模型 $p_\theta$，去近似前向的逆过程 $q(x_{t-1} \mid x_t, x_0)$。

#### 困难的任务

**直接预测 $p(x_0 \mid x_t)$**

从一个很大的噪声中直接变回清晰图像。（跨度太大，几乎不可能）

$\downarrow$
#### 容易的任务

**逐步去噪 $p(x_{t-1} \mid x_t)$**

去除由于单步扩散引入的微小噪声。（化整为零，简单可行）

### 3.5 反向去噪：高斯分布建模

#### 关键理论依据

由于前向过程是高斯分布，且每一步噪声均为独立同分布（i.i.d.），因此其后验分布 $q(x_{t-1} \mid x_t, x_0)$ 也是高斯分布。

- **结论**：反向过程的真实分布形式已知，为高斯分布。
- **意义**：我们只需训练神经网络来预测高斯的**均值**和**方差**即可。

#### 参数化形式

基于上述理论，我们将神经网络 $p_\theta$ 显式地定义为高斯分布：

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left(x_{t-1};\ \underbrace{\mu_\theta(x_t, t)}_{\text{待学习的均值}},\ \underbrace{\Sigma_\theta(x_t, t)}_{\text{方差}}\right)$$

- **均值 $\mu_\theta$**：这是神经网络的核心输出。模型需要观察 $x_t$，推测出原本的 $x_{t-1}$ 的中心位置。
- **方差 $\Sigma_\theta$**：通常被简化处理。
  - 方案 A：固定为 $\beta_t I$（DDPM 原文）。
  - 方案 B：固定为后验方差 $\tilde{\beta}_t$。

## 四、训练目标

### 4.1 最终的 Loss Function

Diffusion Model 的训练目标出人意料地简单。它本质上只是一个**均方误差（MSE）**：

$$\boxed{L_{\text{simple}} = \left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2}$$

#### 公式拆解

- $\epsilon$：**真实噪声**（Ground Truth）。前向过程中采样得到的标准高斯噪声。
- $\epsilon_\theta(x_t, t)$：**预测噪声**。模型看着带噪图片 $x_t$，猜测加了**什么噪声**。

#### 核心直觉

**Q：为什么预测噪声 $\epsilon$ 而不是原图 $x_0$？**

- **难度**：从混乱的 $x_t$ 直接还原 $x_0$ 太难了。
- **类比**：这类似 ResNet 的残差学习。从结构化图像中剥离出无规律的噪声，比直接重构图像要容易得多、稳定得多。

#### 极大似然估计（MLE）

我们的目标是最大化观测数据 $x_0$ 的对数似然：

$$\mathcal{J} = \max_\theta \log p_\theta(x_0)$$
#### 推演：Diffusion（多隐变量）

将整个轨迹 $x_{1:T}$ 视为隐变量：

$$\log p_\theta(x_0) = \log \int p_\theta(x_{0:T})\, dx_{1:T}$$

#### 关键技巧：引入变分分布

为了计算积分，我们需要引入一个已知的分布。这里我们利用**前向过程** $q(x_{1:T} \mid x_0)$ 作为我们的变分分布。

#### Step 1：恒等变换（Multiply and Divide）

$$\log p_\theta(x_0) = \log \int p_\theta(x_{0:T})\, dx_{1:T}$$

$$= \log \int q(x_{1:T} \mid x_0) \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\, dx_{1:T}$$

#### Step 2：转换为期望形式（Expectation）

根据期望定义 $\mathbb{E}_{x \sim q}[f(x)] = \int q(x) f(x)\, dx$，上式等价于：

$$= \log \mathbb{E}_{q(x_{1:T} \mid x_0)} \left[ \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)} \right]$$

#### Step 3：导出下界（ELBO）

$$\log p_\theta(x_0) = \log \mathbb{E}_{q(x_{1:T} \mid x_0)} \left[ \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)} \right]$$

$$\geq \mathbb{E}_{q(x_{1:T} \mid x_0)} \left[ \log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)} \right]$$
#### 1.利用马尔可夫性质展开

- **分母（前向过程）**：$q(x_{1:T} \mid x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1})$
- **分子（反向过程）**：$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t)$

> [!note]
> 
>$P(A,B,C,D)= P(A)P(B|A)P(C|A,B)P(D|A,B,C)$ 
> 
> 马尔可夫性质（Markov Property）其实一句话就能说清：
> 
> > **未来只和现在有关，与过去无关。**
> 
> 数学上写成：
> 
> $$
> P(x_t\mid x_{t-1},x_{t-2},\ldots,x_0) = P(x_t\mid x_{t-1})  
> $$
#### 2. 代入 ELBO 并拆分项

$$\mathcal{L} = \mathbb{E}_q \left[ \log \frac{p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t)}{\prod_{t=1}^{T} q(x_t \mid x_{t-1})} \right]$$

$$= \mathbb{E}_q \left[ \log p(x_T) + \sum_{t=1}^{T} \log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_t \mid x_{t-1})} \right]$$

现在的形式中，分母 $q(x_t \mid x_{t-1})$ 和分子 $p_\theta(x_{t-1} \mid x_t)$ 的方向是反的，无法直接比较。

#### 关键技巧：Conditioning on $x_0$

为了让分母的方向转过来（从 $x_t \to x_{t-1}$），我们利用贝叶斯公式将 $q(x_t \mid x_{t-1})$ 改写：

$$q(x_t \mid x_{t-1}) = q(x_t \mid x_{t-1}, x_0) = \frac{q(x_{t-1} \mid x_t, x_0)\, q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)}$$

将上式代入原方程，经过一系列复杂的代数消元（中间项互相抵消），我们得到整理后的形式：

$$\mathcal{L} = \mathbb{E}_q \left[ \underbrace{\log \frac{p(x_T)}{q(x_T \mid x_0)}}_{\mathcal{L}_T} + \sum_{t=2}^{T} \underbrace{\log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_{t-1} \mid x_t, x_0)}}_{\mathcal{L}_{t-1}} + \underbrace{\log p_\theta(x_0 \mid x_1)}_{\mathcal{L}_0} \right]$$

**Magic Happens**：现在的对比项变成了 $p_\theta(x_{t-1} \mid x_t)$ vs $q(x_{t-1} \mid x_t, x_0)$。

（两个都是 $x_t \to x_{t-1}$ 的方向！）

我们将损失函数拆解为三部分：

$$\mathcal{L} = \underbrace{L_T}_{\text{常数项}} + \underbrace{L_0}_{\text{重构项}} + \sum_{t=2}^{T} \underbrace{L_{t-1}}_{\text{去噪匹配项}}$$

#### $L_T$（忽略）

$$D_{KL}\big(q(x_T \mid x_0) \,\|\, p(x_T)\big)$$

前向最后一步得到的噪声分布 vs 标准高斯。由于 $q$ 和 $p$ 均无参数，梯度为 $0$。

#### $L_{t-1}$（核心）

$$D_{KL}\big(q \,\|\, p_\theta\big)$$

衡量模型预测的去噪分布 $p_\theta$ 与真实后验分布 $q$ 的差异。**这是我们要优化的主战场。**

#### $L_0$（忽略）

$$-\log p_\theta(x_0 \mid x_1)$$

最后一步的重构误差。由于 $x_1$ 已非常接近 $x_0$，通常也忽略或合并。

### 我们现在的目标是最小化 $L_{t-1}$：

$$L_{t-1} = D_{KL}\big(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t)\big)$$

#### 1. 真实后验 $q$

通过贝叶斯公式推导，它是一个高斯分布：

$$\mathcal{N}\!\left(x_{t-1};\ \tilde{\mu}_t(x_t, x_0),\ \tilde{\beta}_t I\right)$$

其均值为（这是由公式推出来的定值）：

$$\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1 - \bar{\alpha}_t}\, x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\, x_t$$

#### 2. 模型预测 $p_\theta$

我们也将其建模为高斯分布：

$$\mathcal{N}\!\left(x_{t-1};\ \mu_\theta(x_t, t),\ \tilde{\beta}_t I\right)$$

这里假设方差相同，**唯一的变量是均值 $\mu_\theta$**。

#### 结论

两个**同方差**高斯分布的 KL 散度，等价于衡量它们**均值的平方差**。

$$L_{t-1} \propto \big\| \tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t) \big\|^2$$

#### 问题

真实均值 $\tilde{\mu}_t$ 的公式里包含了 $x_0$。但模型推断时只有 $x_t$，不知道 $x_0$。我们需要把 $x_0$ 换掉。

#### Step 1：利用前向公式表示 $x_0$

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon \quad \Longrightarrow \quad \boxed{x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\, \epsilon}{\sqrt{\bar{\alpha}_t}}}$$

#### Step 2：代入 $\tilde{\mu}_t$ 表达式

将 $x_0$ 代入上一页的 $\tilde{\mu}_t$ 公式，经过繁琐的代数化简，我们得到一个非常漂亮的式子：

$$\tilde{\mu}_t(x_t, \epsilon) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\, \epsilon \right)$$

#### 关键发现

真实的后验均值 $\tilde{\mu}_t$ **完全由当前图像 $x_t$ 和加入的噪声 $\epsilon$ 决定！**

> [!note]
> > **训练时，公式里的 ϵ 是已知的；推理时，公式里的 ϵ 是未知的。神经网络存在的意义，就是估计这个未知的 ϵ。**

为了让模型预测的均值 $\mu_\theta$ 能够逼近真实的均值 $\tilde{\mu}_t$，我们让模型模仿 $\tilde{\mu}_t$ 的结构。

#### 参数化设计

将真实噪声 $\epsilon$ 替换为神经网络的预测噪声 $\epsilon_\theta$：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\, \epsilon_\theta(x_t, t) \right)$$

代入 Loss 函数进行计算：

$$L_{t-1} \propto \left\| \tilde{\mu}_t - \mu_\theta \right\|^2$$

$$= \left\| \frac{1}{\sqrt{\alpha_t}} \left( x_t - \ldots \epsilon \right) - \frac{1}{\sqrt{\alpha_t}} \left( x_t - \ldots \epsilon_\theta \right) \right\|^2$$

$$= \frac{(1 - \alpha_t)^2}{\alpha_t (1 - \bar{\alpha}_t)} \left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2$$

忽略前面的常数系数，训练目标最终简化为预测噪声的 MSE：

$$L_{\text{simple}} = \left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2$$

#### 算法流程：训练

#### 核心任务

让神经网络 $\epsilon_\theta$ 学会"识别噪声"。即：给定 $x_t$ 和 $t$，预测出叠加的 $\epsilon$。

#### Training Loop

1. **Repeat until converged:**
2. 采样真实数据：$x_0 \sim q(x_0)$
3. 采样时间步：$t \sim \mathrm{Uniform}(\{1, \dots, T\})$
4. 采样高斯噪声：$\epsilon \sim \mathcal{N}(0, I)$
5. 执行梯度下降：

$$\nabla_\theta \left\| \epsilon - \epsilon_\theta\!\left( \underbrace{\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon}_{\text{构造 } x_t},\ t \right) \right\|^2$$

6. **End Repeat**

#### 算法流程：推理

#### 核心任务

从纯噪声出发，利用学好的 $\epsilon_\theta$ 逐步去噪，还原图像。

#### Sampling Loop

1. 初始噪声：$x_T \sim \mathcal{N}(0, I)$
2. **For $t = T, \dots, 1$ do:**
3. 采样随机扰动：$z \sim \mathcal{N}(0, I)$（若 $t = 1$ 则 $z = 0$）
4. 计算上一时刻图像 $x_{t-1}$：

$$x_{t-1} = \underbrace{\frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\, \epsilon_\theta(x_t, t) \right)}_{\text{确定性去噪 (Mean)}} + \underbrace{\sigma_t\, z}_{\text{随机扰动 (Variance)}}$$

5. **End For**
6. **Return $x_0$**


