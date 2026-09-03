## **scaling law 做的事情就是在算力受限下的数据/参数/训练步数的最优配比。**

变量：
	- 模型本身：参数量（M）N、数据量（Tokens）D、训练算力（Flops）C
	- 训练过程：Batch size 、Step
> [!note]
> 其中D = B · S ， C ≈ 6N · B · S

单约束下：$$\mathcal{L}(\mathbf{X}) = \left(\frac{\mathbf{X}_c}{\mathbf{X}}\right)^{\alpha_X}, \quad \mathbf{X} \in \{\mathcal{N}, \mathcal{D}, \mathcal{C}\}$$
双约束：
- 第一种: s.t. N,D:$$\mathcal{L}(N, D) = \left[ \left( \frac{N_c}{N} \right)^{\frac{\alpha_N}{\alpha_D}} + \left( \frac{D_c}{D} \right) \right]^{\alpha_D}$$
- 第二种: s.t. N,C:
	- **C是可以看作B和S共同决定的B增大S就要相应减小，反之亦然，当B scaling达到一定阈值时会出现Ineffective scaling的情况**。
	- 数据质量高可以用大Batch size（Loss中 Gradient Noise ≈ 数据质量，Noise 小 ，需要更少Step ，C给定，Batch就可以很大）。
	- **Step-Data 权衡关系：**$$ \left( \frac{S}{S_{\min}} - 1 \right) \left( \frac{E}{E_{\min}} - 1 \right) = 1 $$
	- **Critical Batch Size 定义与经验公式：**$$ B_{\text{crit}}(L) = \frac{E_{\min}}{S_{\min}} \propto L^{-1/\alpha_B} $$
	综上我们可以得到一个修正过后的S<sub>min</sub>的一个优化问题：$$ \min_{N, S_{\text{min}}} \quad \mathcal{L}(N, S_{\text{min}}) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{S_c}{S_{\text{min}}}\right)^{\alpha_S} \quad \text{s.t.} \quad C_{\text{min}} \approx 6 N B_{\text{crit}}(L) S_{\text{min}} $$这个问题可以得到一个结论，在算力增加的情况下提升参数量的收益大于提升Batch size大于提升Steps。

| 模型参数      | 数据        | 算力         | Batch            | 公式                                                                                                    | 含义                 |
| --------- | --------- | ---------- | ---------------- | ----------------------------------------------------------------------------------------------------- | ------------------ |
| N         | ∞         | ∞          | 固定               | $L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}$                                                        | 模型容量受限             |
| ∞         | D         | Early Stop | 固定               | $L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}$                                                        | 数据规模受限             |
| Optimal   | ∞         | C          | 固定               | $L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}$ (naive)                                                | 固定 batch 下的算力瓶颈    |
| $N_{opt}$ | $D_{opt}$ | $C_{min}$  | $B \ll B_{crit}$ | $L(C_{min}) = \left(\frac{C_c^{min}}{C_{min}}\right)^{\alpha_c^{min}}$                                | Compute-optimal 训练 |
| N         | D         | Early Stop | 固定               | $L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N / \alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}$  | 容量-数据双瓶颈           |
| N         | ∞         | S 步        | B                | $L(N, S) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{S_c}{S_{min}(S, B)}\right)^{\alpha_S}$ | 训练动态               |

> **计算高效训练应该停在收敛损失以上约 10% 的位置**

即 L(N,Smin​)=(1+αS​αN​​)⋅L(N,∞)≈1.1⋅L(N,∞)

这个 10% 是从 αN​/αS​=0.076/0.76≈0.1 来的，是最优停止的**解析解**

## Chinchilla Law

### 1. LR schedule 修正，不同模型使用不同的LR schedule 
### 2. 最优化训练设置：

$$N_{\text{opt}}(C), D_{\text{opt}}(C) = \arg\min_{N,D} L(N, D) \quad \text{s.t.} \quad C = 6ND$$

假设最优解存在形式：

$$N_{\text{opt}}(C) \propto C^a \qquad D_{\text{opt}}(C) \propto C^b$$
> [!note]
> 
> - 使用不同方法来估计 $a, b$ 的值
> - 不再强调训练步数 ($S$) 和 batch size 的影响 ($B$)，更关心总算力使用。

### 3.三种方法确定ab的值
**方法1：固定 N，变 D，取最优 (N, D) → 拟合幂律**

- 训练 70M ~ 10B 参数，每种规模 4 个不同 D
- 对每个 FLOPs 预算 C，取所有 run 中 loss 最低的点
- 拟合：Nopt​∝C0.50, Dopt​∝C0.50

**方法2：固定 C，变 N，取最优 (N, D) → 拟合幂律**

- 固定 9 个 FLOPs 预算（如 10²⁰），每个下训练不同大小的 N
- 画 loss-vs-N 曲线，抛物线拟合找谷底
- 9 条 IsoFLOP 曲线，拟合：a=0.49, b=0.51

**方法3：直接拟合三段式损失函数**

$$\hat{L}(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

- E：自然语言熵（loss 理论下限）
- A/N^α：模型容量不足带来的损失
- B/D^β：训练不充分带来的损失
- L-BFGS + Huber loss 在 400+ 模型上拟合，得 a≈0.51, b≈0.49

还有一个结论就是数据量起码要是参数量的20倍以上。（现在肯定不止20倍了）


 