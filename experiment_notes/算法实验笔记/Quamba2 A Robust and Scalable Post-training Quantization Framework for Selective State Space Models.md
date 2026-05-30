## Quamba2 A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Quamba2 是一个针对 Selective State Space Models（Mamba1/Mamba2）的后训练量化（PTQ）框架，支持 W8A8、W4A8、W4A16 三种 bit-width 配置，以及 W4A{8/16}-mixed 混合精度。核心算法 pipeline：(1) **Sort-and-cluster**：利用 SSM 的 channel order preserving 属性，offline 校准各 channel 的最大值后对 head 内 channel 排序，对 head 和 channel 分别聚类（m=4 组 head, n=4 组 channel），为每个 head×channel 组计算独立 scaling factor 量化 x_t 到 8-bit；(2) **Per-state-group quantization**：利用 SSM 的 state persistence 属性（B 和 C 中激活的 state group 在时间步和样本间保持一致），对 B_t 和 C_t 按 state group 分别量化（每组一个 scaling factor）；(3) **Cluster-aware weight reordering**：根据 sort-and-cluster 的排序/聚类索引，offline 重排 input projection、causal convolution、normalization 和 output projection 的权重，保证 SSD 计算保持 channel order 从而输出不变；(4) **Offline Hadamard matrix fusion**：将 Hadamard 矩阵 offline 融合到 input/output projection 权重中（W_out^H = H_n W_out H_n^T, W_in^H = W_in H_n^T），配合 online Hadamard transform 实现 compute-invariance；(5) **Head-to-toe quantization**：从 embedding 层到 SSM blocks 到 lm_head 全量化，embedding 用 per-token quantization，lm_head weight 用 per-group quantization；(6) **W4AX-mixed**：进化搜索（population=40, generations=5）自动识别敏感 block 分配 W4A16，其余用 W4A8。伪代码：calibration set（Pile 512句）→ calibrate x channel max → sort channels → cluster heads(m) → cluster channels per head(n) → quantize x_t with m×n scales → quantize B_t/C_t per state group → reorder weights offline → fuse Hadamard offline → GPTQ on 4-bit projection weights → W4AX evolutionary search。
  - 实验比较：(a) Quamba2 vs Quamba vs MambaQuant 在 Mamba1 1.4B/2.8B 和 Mamba2 1.3B/2.7B/8B 上的零样本准确率（6 任务平均）；(b) W8A8/W4A8/W4A16 latency 对比（TPOT/TTFT, A5000 + Orin Nano 8G）；(c) MMLU 5-shot 评估 W4A8 vs W4A16 vs W4AX-mixed；(d) 消融：sort-and-cluster、per-state-group、Hadamard、GPTQ 各组件贡献；(e) embedding/lm_head 量化消融；(f) mixed-precision handcrafted vs auto search 对比；(g) batch size scaling（b=1/32/64/128/256）TPOT 对比；(h) 能效分析（J/req, tokens/GW）；(i) Pareto front: accuracy vs latency vs memory 与 QuaRot/Llama2/Llama3-QServe 对比。

- 硬件平台是什么，配置是什么。
  - Cloud: NVIDIA A5000 GPU 24GB。Edge: NVIDIA Orin Nano 8G。Latency profiling：warm-up iterations + 100 iterations 平均。CUDA kernel 基于 CUTLASS 实现。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Mamba1（1.4B, 2.8B），Mamba2（130M, 370M, 1.3B, 2.7B, 8B）。Calibration: Pile dataset 随机 512 句（fixed seed）。
  - Benchmarks：LM-EVAL 框架。Zero-shot: LAMBADA, HellaSwag, PIQA, ARC-easy, ARC-challenge, WinoGrande（5 次平均）。MMLU 5-shot（57 学科）。Generation: Natural Questions (exact match), SquadV2 (F1)。
  - Baseline：Quamba (Chiang et al. 2025), MambaQuant (Xu et al. 2025)。对比 Transformer 量化：QuaRot (Llama2), QServe (Llama3 W4A8KV4)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/enyac-group/Quamba（论文声明 will be released）
  - 算法 pipeline 张量计算流程（以 Mamba2 W4A8 为例）：
    1. **Calibration**：从 Pile 采样 512 句 → 前向传播 → 记录每层 x 各 channel 的 max(|x_c|) → 按 max 降序排列 channel → 对 head 聚类（m=4 组）→ 对每组 head 内 channel 聚类（n=4 组）→ 记录 sort/cluster indices。同时记录 B/C 的 state group 激活模式。
    2. **Weight offline processing**：(a) 根据 cluster indices 重排 input proj weights W_in 的列和 causal conv1d weights 的 channel；(b) 重排 norm weights；(c) 重排 output proj weights W_out 的行；(d) Hadamard fusion: W_in^H = W_in @ H_n^T, W_out^H = H_n @ W_out @ H_n^T；(e) GPTQ 优化 4-bit 量化权重。
    3. **Inference（单 token 前向）**：
       - u_t ∈ R^D → W4A16/W4A8 input projection: x_t, B_t, C_t, Δ_t = (W_in^H)^T @ u_t（权重 4-bit, 激活 16-bit 或 4-bit weight×8-bit act）
       - Online Hadamard: x_t^H = H_n @ x_t
       - Sort-and-cluster: 按 sort/cluster indices 重排 x_t^H → 分组 → 每组内 quantize: x̄_t^s = clamp(round(x_t^H / s_{m,n}), -127, 127)（8-bit）
       - Per-state-group 量化 B_t/C_t: B̄_t^g = clamp(round(B_t / s_g), -127, 127)（每组 state group 一个 scale）
       - Causal conv1d: y_conv = conv1d_8bit(x̄_t^s, W_conv_8bit)
       - SSD scan: h_t = A_t @ h_{t-1} + B̄_t^g @ x̄_t^s, y_ssd = C̄_t^g @ h_t（8-bit states）
       - Online Hadamard: y^H = H_n @ (y_ssd ⊙ SiLU(z_t))
       - Output projection: y_out = (W_out^H)^T @ ȳ^H（权重 4-bit, 激活 8-bit）
    4. **W4AX 混合精度搜索**：evolutionary search (pop=40, gen=5) → 每代保留 top 50% → 10 mutation + 10 crossover → 最终每层选 W4A8 或 W4A16。
