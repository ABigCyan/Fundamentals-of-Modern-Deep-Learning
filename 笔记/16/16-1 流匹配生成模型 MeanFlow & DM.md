## MeanFlow

### 从瞬时 $v$ 到平均 $u$

**核心思想**

不强行让路径变直（Reflow），而是让模型**使用平均速度**。

- **旧方法（FM）**：预测切线 $v_t$（瞬时速度）。
- **新方法（MeanFlow）**：预测向量 $u_t$，它真正把当前位置 $z_t$ 与目标 $z_r$ 连起来。
![[16-1-fig-1.png|377]]

弯曲时：$v \neq u$

直线时：$v \approx u$

> 它其实就是想要直截了当地预测

### MeanFlow 恒等式

我们如何从瞬时速度 $v$ 找到平均速度 $u$？

**修正公式（泰勒展开视角）**

$$ x(t) = x_r + (t - r) \, u(x_r, r, t) $$

$$ v = u + (t - r) \cdot \underbrace{\frac{du}{dt}}_{\text{Acceleration}} $$
 ![[16-1-fig-2.png]]

> [!note]
> **Figure **：**平均速度 $u(z, r, t)$ 的场。** **最左图**：虽然*瞬时速度* $v$ 决定路径的切线方向，由平均速度 $u(z, r, t)$ 一般**与 $v$ 不对齐**。平均速度与*位移*对齐，而位移是 $(t - r)u(z, r, t)$。**右侧三个子图**：场 $u(z, r, t)$ 同时以 $r$ 和 $t$ 为条件，这里分别展示 $t = 0.5,\ 0.7,\ 1.0$ 的情况。

我们使用 **Jacobian-Vector Product (JVP)** 来计算方向导数 $\frac{du}{dt}$。

```python
# PyTorch 风格的伪代码
# fn: 给定 (z, r, t) 输出 v 的模型函数
t, r = sample_time_steps()
x = sample_data()     # 采样一批数据点
e = sample_noise()    # 采样噪声

# 获取速度
z = (1-t) * x + t * e    # 线性插值
v = e - x    # 瞬时速度（切线）

# 2. 用 JVP 计算 dv/dt
u, dudt = jvp(fn, (z, r, t), (v, 0, 1))

# 3. 构造修正后的目标
u_target = v - (t - r) * dudt

# 4. 损失
loss = MSE(u, stop_gradient(u_target))
```

**Classifier-Free Guidance (CFG)** 是一种在不需要额外分类器的前提下增强条件生成的技术：训练 *conditional* 与 *unconditional* 两个模型，然后在推理时把它们的预测线性组合起来，以控制条件强度。

$$ \hat{u}_\omega = u_{\text{uncond}} + \omega \left( u_{\text{cond}} - u_{\text{uncond}} \right) $$

其中 $\omega$ 是引导强度。$\omega = 0$ 退化为无条件生成；$\omega > 1$ 提升条件对齐度，但可能降低多样性。

过去是使用分类器来判断图像并修改梯度，这种方法无法处理未见过的情形。

虽然原始 MF 建立了用于一步生成的框架，但它面临若干根本性的局限：

- **网络依赖的目标**：训练目标 $u_{tgt}$ 不仅依赖于真实场，还依赖于**网络自身的预测**（JVP 项里的 $u_\theta$），使其成为非标准的回归问题。

- **训练不稳定**：因为目标会随模型而漂移，训练过程呈现高方差。

- **固定的引导尺度**：CFG 尺度（$\omega$）在训练时是固定的，这牺牲了推理时的灵活性，也无法搜索最优的引导强度。

## Improved MF：把 MeanFlow 改写为 $v$-loss

为了让训练目标与模型参数**解耦**，iMF 重新表达恒等式：

$$ v(z_t) = u(z_t) + (t - r) \frac{d}{dt} u(z_t) $$

- **目标重定义**：目标变成对**瞬时速度 $v$** 的损失（标准回归），由一个预测平均速度 $u$ 的网络进行重新参数化。

- **合法输入**：与 MF（用 JVP 中的**条件速度** $(e - x)$）不同，iMF 使用网络预测的**边缘速度** $v_\theta(z_t, t) = u(z_t, t, t)$，从而保证输入**只依赖于 $z_t$**。

- **影响**：这一重写显著稳定了训练，并得到更标准的优化景观。


```python
# fn(z, r, t): 预测平均速度 u 的网络
# x: 训练批次

t, r = sample_t_r()
e = randn_like(x)

# 时刻 t 处的带噪样本
z = (1 - t) * x + t * e

# 边界条件版本：
# v_theta(z, t) = u_theta(z, t, t)
v = fn(z, t, t)

# JVP 同时返回函数输出与 Jacobian-vector product
u, dudt = jvp(fn, (z, r, t), (v, 0, 1))

# 复合预测量 V_theta
V = u + (t - r) * stopgrad(dudt)

# Flow-Matching 风格的回归目标
error = V - (e - x)
loss = metric(error)
```

iMF 把引导尺度视为**动态参数**而非训练时的常数：

- **显式条件化**：CFG 尺度 $\omega$ 被建模为一个**输入条件变量**，与时间步 $t$、$r$ 类似。

- **训练策略**：训练时从一个分布中随机采样 $\omega$，让模型学到不同引导强度下的轨迹。

- **额外控制**：支持把 **CFG 区间** $(t_{\min}, t_{\max})$ 作为条件，从而在推理时实现有效的质量-多样性权衡。

## Pixel MeanFlow

**MeanFlow 回顾**：平均速度场 ⇒ **一步（1-NFE）采样**。

**下一个问题**：能否**直接在像素空间做一步生成**？
- **消除潜空间瓶颈**：避免自编码器对清晰度/细节的限制。
- **把速度推到极致**：无需潜空间管线的一步生成。
- **新挑战**：像素空间的 patch token 维度很高；朴素的 $\epsilon$-/v- 输出会偏离流形并变得不稳定。

**向 JiT 过渡**：在像素 Transformer 中，**输出参数化**很关键（$x$-pred 稳健；$\epsilon$/v-pred 可能灾难性失败）。

![[16-1-fig-3.png|378]]

**主要论断（流形视角）**

- 干净图像 $x_0$ 位于一个**低维流形附近**。
- 高噪状态 $x_t$（或 $\epsilon, v$）**远离流形**。
- 因此，**网络的输出空间**很关键：

  $x$-pred（在/近流形上） 对比 $\epsilon$-pred / $v$-pred（远离流形）。

**为什么用 $x$-prediction：**

- 高维像素 token 放大了 $\epsilon$/v 输出的离流形困难。
- 只有 $x$-prediction 在 $D$ 增大时仍能产生合理的结果。
- 在像素 Transformer 中，**输出什么** 比 **把损失放在哪里** 更重要。

**实验设置**

- 直接在**像素空间**运行扩散。
- 把图像切成 token（ViT 风格）：token 维度 $\approx p^2 \times 3$。
- 使用普通的 Transformer 去噪器（DiT），但**不使用 VAE 潜空间**。

**要点**：JiT 提出 "**让去噪模型去做去噪**"：在像素 ViT 中应输出**类似 $x$ 的量**，而不是类似噪声的量。

**JiT**：$x$-prediction 优于离流形的 $v$/$\epsilon$-prediction。

**MeanFlow**：速度约束对**一步采样**至关重要。

**pMF**：在流形上的 $x$ 处输出，用动力学空间的 $v$ 进行监督。

- **网络输出**：$x_\theta(z_t, r, t)$（图像-like / 在流形上）
- **训练目标**：速度约束（v-/u- 空间）

**为什么这有帮助**

- $\epsilon$/$v$ 是动力学的*工具*，并不一定就是最好的*输出空间*。
- 避免让模型去生成类噪声的、远离流形的目标。
- 把 $x$-pred 的稳定性与基于速度的训练的效率结合在一起。
![[16-1-fig-4.png]]
模型输出一个图像式的预测

$$ x_\theta = f_\theta(z_t, r, t) $$

转换为速度 / 平均速度参数化

$$ x(z_t, r, t) = z_t - t \, u(z_t, r, t) $$

当 $t = r$ 时

$$ x(z_t, t, t) = z_t - t \, v(z_t, t) $$

重参数化：$x \to u \to v$

$$ x(z_t, r, t) = z_t - t \, u(z_t, r, t) \;\Rightarrow\; u(z_t, r, t) = \frac{z_t - x(z_t, r, t)}{t} $$

优化目标：速度约束

$$ \mathcal{L} = \mathbb{E}_{t, r, x, \epsilon}\left[ \left\| V_\theta - v \right\|^2 \right] $$

其中

$$ V_\theta = u_\theta + (t - r) \, \text{JVP}_{sg} $$

**推理仍是 1 步**

$$ \hat{x} = z_0 = z_1 - u_\theta(z_1, 0, 1) $$

> [!note]
> 这里可以和[[13-3 扩散生成模型 Score Matching 和 Prediction]]的Prediction部分做关联

## Drifting Generating Models

**旧方法：**

- 从噪声出发
- 跑很多步迭代细化
- 分布在**推理时**演化

    noise → step 1 → step 2 → ... → image

**新方法（Drifting）：**

- 让生成器本身保持**一步**
- 让优化过程**逐步推移其输出分布**
- 分布在**训练时**演化

**直观理解**

Diffusion / Flow Matching 把计算花在采样上。

Drifting 把演化过程放在训练阶段，因此采样保持极快。

    f_θ⁽⁰⁾ → f_θ⁽¹⁾ → f_θ⁽²⁾ → ... → f_θ* （one forward pass）

#### "Drift" 是什么意思？

在**任意训练步**，当前的生成器从一个不完美的分布中采样。

对每一个生成的样本：

- 把它拉向附近的**真实**样本
- 把它推离附近的**生成**样本

这给出一个局部更新方向：

$$ x \longrightarrow x + V(x) $$

$$ \mathbf{V}_{p,q}(\mathbf{x}) = \frac{1}{Z_p Z_q} \, \mathbb{E}_{p,q} \Big[ k(\mathbf{x}, \mathbf{y}^+) \, k(\mathbf{x}, \mathbf{y}^-) \, (\mathbf{y}^+ - \mathbf{y}^-) \Big]. $$

其中，相似度核定义为

$$ k(\mathbf{x}, \mathbf{y}) = \exp\left( -\frac{1}{\tau} \|\mathbf{x} - \mathbf{y}\| \right). $$

- $\mathbf{y}^+ \sim p$：正样本 / 真实样本
- $\mathbf{y}^- \sim q$：负样本 / 生成样本
- $k(\mathbf{x}, \mathbf{y}^+) k(\mathbf{x}, \mathbf{y}^-)$：联合相似度权重
- $(\mathbf{y}^+ - \mathbf{y}^-)$：局部漂移方向

**关键性质**

该形式是反对称的：

$$ \mathbf{V}_{p,q} = -\mathbf{V}_{q,p}, \quad \mathbf{V}_{p,p} = \mathbf{0}. $$$$ \mathcal{L}_{\text{drift}}(\theta) = \mathbb{E}_{e \sim p(e)} \Big[ \big\| f_\theta(e) - \text{sg}\big( f_\theta(e) + V_{p, q_\theta}(f_\theta(e)) \big) \big\|_2^2 \Big] $$

等价地，令 $x = f_\theta(e)$，$x_{\text{drifted}} = \text{sg}(x + V)$，则小批量损失就是 $\| x - x_{\text{drifted}} \|_2^2$。

```python
# f: 生成器
# y_pos: [N_pos, D], 数据样本

e = randn([N, C])              # 噪声
x = f(e)                       # [N, D], 生成样本
y_neg = x                      # 复用 x 作为负样本

V = compute_V(x, y_pos, y_neg)
x_drifted = stopgrad(x + V)

loss = mse_loss(x - x_drifted)
```

注：为简洁起见，负样本 $y_{\text{neg}}$ 取自同一批生成数据，但它们也可以来自其他负样本来源。

> [!note]
> 
> [Generative Modeling via Drifting](https://lambertae.github.io/projects/drifting/)
> 
> ![[16-1-fig-5.png]]
> 
> ![[16-1-fig-6.png|483]]
> ##### Drift vectors：
> - 相近的粒子的趋势是相近的，指向真实目标分布
> - 相近的粒子也具有排斥力以防坍缩到同一个中心点
> 
> 这个 v 可以理解成：如果我现在手里有一些粒子（样本），我想让这些粒子一步一步往真实数据分布靠近，那么每一步应该怎么更新。网络 fit 的是我从 noise 出发去先预测的一个不准的 t，去看它怎么去更新，把这个 $Δv$加在预测不好的结果上去做 drifting。
> 

## 总结

**Flow Matching**：学习*速度场*，而不是显式的 flow map。

- **为什么重要**：与其预测 flow 结果的均值（这未必准确），FM 直接学习**速度**，更加灵活且更具泛化性，从而允许多条合法轨迹到达目标。

**MeanFlow**：把场取*平均*，从而实现**一步**生成。

- **为什么重要**：把多步 diffusion/flow 的质量带到**实时**生成。

**Pixel MeanFlow**：预测 $x$（在流形上），并用 MeanFlow 的速度约束（动力学）进行训练。

- **为什么重要**：把 $x$-pred 的稳定性与基于速度的训练的效率结合起来，把一步生成**推到像素空间**，无需潜空间瓶颈。

**Drifting**：在训练过程中演化*生成分布*，同时让推理保持**一步**。

- **为什么重要**：不再需要设计时间相关的传输路径并在采样时求解它；Drifting 让模型分布在优化过程中**逐步向真实数据漂移**，从而得到一个**原生的一步生成器**，而不是蒸馏出来的多步求解器。

