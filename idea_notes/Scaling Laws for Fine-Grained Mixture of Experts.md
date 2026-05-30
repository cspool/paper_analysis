## Scaling Laws for Fine-Grained Mixture of Experts

- baseline方法是什么？
  Baseline方法：**标准MoE（Vanilla MoE, G=1）**——每个expert的hidden dimension固定为d_ff（标准FFN层的4×d_model），expansion rate E控制总expert数量（N_expert = E），每个token经由router选择top-k个（通常k=1或k=2）expert处理。训练采用固定时长（如Clark et al. 2022使用130B tokens固定数据集），不调整训练token数与模型size的配比，也不调整expert粒度。Scaling law沿用Clark et al. (2022)的形式 L(N,E) = (10^{d/a}/N)^a × (1/E)^{b+c·log N}，仅适用于固定数据集大小，无法指导compute-optimal训练配置。
  全栈执行例子（Baseline: G=1 MoE, E=64, N_act=64×25M, D=130B tokens, A100 GPU）：
  ```
  训练一个Fine-Grained MoE layer（Transformer decoder第L层）：
  ├─ [算法Pipeline] MoE Layer定义：
  │   d_model=512, d_ff=2048, G=1, E=64
  │   d_expert = d_ff/G = 2048  ← 每个expert与标准FFN相同大小
  │   N_expert = G×E = 64个expert
  │   每token路由到k=1个expert
  │   问题：expert粒度粗，每个expert处理高度混合的token模式
  │
  ├─ [GPU Kernel] Router forward:
  │   W_router [512, 64] → router_logits [T, 64]
  │   softmax over expert dim, top-1 selection
  │   Router FLOPs ≈ d_model × E × 14 (c_r) per token per layer
  │
  ├─ [GPU Kernel] Expert Computation:
  │   每个expert: W1 [512, 2048], W2 [2048, 512]
  │   Per-token active FLOPs ≈ 8d_model^2 = 2.1M
  │   总active参数: 2 × d_model × d_ff = 2.1M per token
  │   问题：G=1时所有64个expert竞争token，expert specialization有限
  │
  ├─ [算法Pipeline] Training:
  │   D=130B tokens固定（Clark et al. 2022的设置）
  │   模型参数量N增加但D不相应增加 → undertraining
  │   问题：N增大时模型未能充分训练，dense逐渐追平MoE
  │
  └─ Scaling失效：
      Clark et al. (2022)结论：N>1T时dense超越MoE
      原因：固定D=130B使大模型undertrained
  ```
  Baseline缺陷：(1) **Expert粒度固定为G=1**：d_expert固定为d_ff（=4×d_model），expert层与标准FFN等价大，缺乏灵活度来更精细地匹配token-expert映射，限制了MoE的潜力；(2) **Scaling law未包含训练时长变量**：Clark et al. (2022)的公式仅适用于固定数据集大小，无法预测compute-optimal训练配置；(3) **固定训练时长导致错误结论**：训练时间不随模型size增长时，大模型undertrained，造成"dense在大模型时超越MoE"的假象；(4) **缺乏细粒度路由的建模**：没有理论工具来量化细粒度expert（d_expert < d_ff）的收益，实际中无法系统性选择最优G值。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：**Fine-Grained MoE Scaling Laws**——引入granularity G作为新超参数（G = d_ff / d_expert），将expert大小从固定d_ff解放为可调整变量，并在Scaling Law公式中显式建模G的影响。核心机制：
  (1) **Granularity参数化与Power-Law假设** → 解决缺陷(1)：定义G使d_expert = d_ff / G，对任意G，每个token被路由到G个细粒度expert（而非1个粗粒度expert），保持激活参数量不变（d_model × 2d_ff = d_model × 2G·(d_ff/G)）。更细粒度的expert提供更灵活的token-to-expert映射，实验验证power-law关系 L(G) = g/G^γ + h。
  (2) **Joint Scaling Law L(N,D,G)** → 解决缺陷(2)(4)：推导包含G的联合Scaling Law——L(N,D,G) = c + (g/G^γ + a)/N^α + b/D^β，基于以下观察：(a) c不依赖于架构（数据固有熵），(b) D和G无交互（图3c：不同G下训练更多token的收益相同，故b/D^β项不含G），(c) a_G = g/G^γ + a（G对模型size scaling的修正项，a确保无限G不会超越密集模型）。用Huber loss + BFGS拟合100+实验数据，RMSE=0.015。
  (3) **Compute-Optimal配置求解** → 解决缺陷(3)：在含routing overhead的FLOPs约束下（F = (12d_model^2 c_f + d_model·E·G·c_r) × D × n_blocks），使用Brent's method求解min_{N,D,G} L(N,D,G)，找到给定计算预算F下的最优N、D、G组合。FLOPs建模包含routing overhead（c_f=6, c_r=14），使G的选择受实际计算成本约束。
  全栈执行例子（论文方法: G=8, E=64, N_act=64×25M, D=66B tokens, A100 GPU）：
  ```
  训练一个Fine-Grained MoE layer（Transformer decoder第L层）：
  ├─ [算法Pipeline] Fine-Grained MoE Layer定义：
  │   d_model=512, d_ff=2048, G=8, E=64
  │   d_expert = d_ff/G = 256  ← 每个expert缩小8倍
  │   N_expert = G×E = 512个expert
  │   每token路由到k=G=8个细粒度expert（而非1个）
  │   激活参数量不变: 8 × 2 × 512 × 256 = 2.1M = 2 × 512 × 2048 ✓
  │
  ├─ [GPU Kernel] Router forward:
  │   W_router [512, 512] → router_logits [T, 512]  ← router变大8×
  │   Expert Choice Routing: softmax over expert dim
  │   每group 256 tokens，每个expert选择top-k个token
  │   负载天然均衡（无需auxiliary loss），G越大expert越多routing越关键
  │   Router FLOPs ≈ d_model × E × G × c_r per token per layer
  │
  ├─ [GPU Kernel] Fine-Grained Expert Computation:
  │   每个细粒度expert: W1 [512, 256], W2 [256, 512]
  │   Per-token计算: 分配给G=8个expert
  │   每个expert计算量: 2 × d_model × d_expert = 2 × 512 × 256 = 262K FLOPs
  │   总每token: 8 × 262K = 2.1M FLOPs（与G=1时相同）
  │   G=8优势: 512个expert vs 64个expert，token-to-expert映射精细8倍
  │
  ├─ [GPU Kernel] Extra LayerNorm after MoE:
  │   critical for G>1: 稳定G个expert输出的加和
  │
  ├─ [算法Pipeline] Compute-Optimal Training:
  │   F = (12×512²×6 + 512×64×8×14) × D × n_blocks
  │     = (18.9M + 3.7M) × D × 8  ← routing占~16%
  │   D = 66B tokens（compute-optimal for N=64×25M, G=8）
  │   对比G=1: 同样D下loss从3.12降至2.95（图3a）
  │
  └─ Scaling效果：
      G=4时最优: N=64×25M, D=66B, loss更低
      G=8时进一步: wall-clock time最优（A100实测，图5b）
      G=16时: 更大routing开销开始主导（router参数512×1024）
      更大模型（N=64×7B）最优G=32→64，验证"compute budget越大→G越大"
  ```

  解决 Baseline 缺陷的方式总结：
  1. **针对"Expert粒度固定G=1"**：引入granularity G解耦expert大小与d_ff，更多细粒度expert（G=8时512个 vs G=1时64个）提供更精细的token-expert mapping，固定N、D下降低loss（L ∝ 1/G^γ with γ=0.58），在几乎所有FLOPs预算下G>1都优于G=1。
  2. **针对"Scaling law不含训练时长变量D"**：借鉴Chinchilla方法将D显式纳入公式 L(N,D,G) = c + (g/G^γ + a)/N^α + b/D^β，使Scaling Law能预测不同N、D、G组合下的loss，覆盖未训练过的region（validation RMSE=0.019）。
  3. **针对"固定训练时长导致错误结论"**：当N、D、G都选为compute-optimal时，MoE始终优于dense且差距随计算预算扩大（10^20 FLOPs时节省20×，10^25 FLOPs时节省>40×），推翻Clark et al. (2022)的"大模型时dense超越MoE"结论——其错误根源在于未调整训练token数D。
  4. **针对"缺乏细粒度路由的成本建模"**：在FLOPs约束中显式建模routing overhead（c_r=14），包含router计算的正反向传播、token dispatch/combine等7组operations。Brent's method优化时routing cost会限制G的选择，使G随模型增大而增长但不会无限增长（图5b：G=16时routing overhead超过granularity收益）。
