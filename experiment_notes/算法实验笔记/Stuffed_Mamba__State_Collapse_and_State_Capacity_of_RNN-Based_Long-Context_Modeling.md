## Stuffed_Mamba__State_Collapse_and_State_Capacity_of_RNN-Based_Long-Context_Modeling

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：论文分析了 Mamba-2 等 RNN 架构在长上下文泛化失败的根本原因——"无法遗忘（inability to forget）"，并提出两种无需训练的遗忘诱导方法。核心发现：(1) Mamba-2 的某些 head 对首 token 的记忆保留强度 α_{1:t} 在整个训练长度窗口内始终 > 0.997，导致状态过参数化（state overparameterization）；(2) 状态均值/方差在超过训练长度后发生爆炸（variance explosion），主要由少数 outlier channel 驱动；(3) 遗忘阈值 T_forget 与状态大小 N_S 成线性关系：T_forget = 5.172 · N_S − 4.469 (R² > 0.999)；(4) 最大召回上下文长度 T_recall 与状态大小成指数关系：T_recall = 4.756 · (1.365^{N_S} − 1) − 0.742 (R² > 0.999)。两种遗忘诱导方法：(a) RRI（Reduced Memory Retention and Insertion）：将记忆保留强度 α_t 缩放 0.9999 倍，记忆插入强度 B_t 缩放 0.75 倍（超参由 32K 验证集选择）；(b) Sliding Window：利用状态加权和性质，通过 h_t^{(r)} = h_t − α_{t−r+1:t} · h_{t−r} 精确模拟滑动窗口，维护 Δ_{t−r:t} 避免浮点不稳定。
  - 实验比较：(1) LM loss vs token position：Mamba-2 130M/370M/780M 在 8K 训练长度下，context > 8K 后 perplexity 急剧爆炸（Figure 1, 10）；(2) Passkey Retrieval：Mamba-2 130M/370M/780M 在 1K-256K context，近乎完美准确率仅在 ≤8K，>16K 后几乎为零（Figure 2）；(3) Inducing forgetting：RRI 和 Sliding Window 方法在 32K context 下将 LM loss 从 ~15 降至 ~8-10，LongMamba 也有类似改善但牺牲短上下文性能（Figure 4）；(4) 训练长度 vs 状态大小：sweep 6 种模型规模（36M/47M/85M/130M/370M/780M）和多种训练长度（最高 256K），验证 T_forget 线性关系和 T_recall 指数关系（Figure 9, 11）；(5) 更多训练 ⇒ 更少遗忘：Mamba-2 370M 从零训练过程中，passkey retrieval 精度在短训练长度内随数据量增加反而下降（Figure 8），呈现过拟合行为；(6) 其他架构比较：RWKV-5、RWKV-6、Mamba-1、HGRN-2 在 passkey retrieval 和 "newlines" prompt 上的对比（Figure 13-15, Appendix H）。

- 硬件平台是什么，配置是什么。
  - 训练：NVIDIA A800 80GB GPU，部分实验多节点、部分单节点多 GPU
  - 训练精度：BF16 为主，部分激活值用 FP32（与官方 Mamba-2 实现一致）
  - 推理评估：FP32 精度（确保精度误差不引入噪声），greedy decoding
  - 优化器：AdamW，weight decay=0.1，gradient clipping=1.0
  - LR scheduler：WSD（warmup-stable-decay），10% decay steps，1000 步 linear warmup，50K 步 linear decay
  - Batch size：0.5M tokens/step
  - 学习率 sweep：{1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3}，passkey retrieval 验证选择最优

- 模型是什么。数据集和bench分别是什么。
  - 模型：Mamba-2 官方 checkpoint（130M: L24/D768/H24, state size 4.8M；370M: L48/D1024/H32, state size 12.9M；780M: L48/D1536/H48, state size 19.3M）+ 从零训练的 checkpoint（36.4M: L6/D512/H16, state size 0.8M；47.0M: L12/D512/H16, state size 1.6M；84.6M: L12/D768/H24, state size 2.4M）。Mamba-2 配置：P=64（head dim），N=128（state dim），H=2d/P（head 数），expansion factor=2，conv kernel=4。还评估了 RWKV-5、RWKV-6、Mamba-1、HGRN-2 作为对比。
  - 训练数据集：RedPajama-V2（30T tokens，过滤短于 4K tokens 的文档——过滤掉约 97.6% 数据），使用 Truncated BPTT（12 序列拼接，等价于 concatenation + 截断梯度）
  - Benchmark：(1) Language Modeling：RedPajama 验证集，perplexity/loss 随 token position 变化；(2) Passkey Retrieval：5-digit passkey，均匀分布 needle position，context 1K-256K，greedy decoding；(3) "newlines" prompt：纯换行符序列，用于检测状态分布稳定性（mean/variance explosion）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 官方 Mamba-2 checkpoint：https://huggingface.co/state-spaces/mamba2-130m / mamba2-370m / mamba2-780m
  - 论文未明确给出独立代码仓库，但方法和分析均基于公开检查点
  - 算法 pipeline（RRI + Sliding Window 的推理时遗忘诱导）：
    ```
    # ===== Mamba-2 原始推理（一个 head，t 时刻） =====
    # 输入: u_t ∈ R^d
    # 参数: W_B, W_C, W_x, W_Δ, b_Δ, A ∈ R
    x_t = SiLU(Conv(u_t @ W_x))^T      # [1, P]
    B_t = σ(Conv(u_t @ W_B))            # [N, 1]
    C_t = σ(Conv(u_t @ W_C))^T          # [1, N]
    Δ_t = Softplus(u_t @ W_Δ + b_Δ)     # scalar
    α_t = exp(-Δ_t * exp(A))            # decay multiplier ∈ (0,1)
    h_t = h_{t-1} * α_t + Δ_t * B_t * x_t  # [N, P] state update
    y_t = C_t @ h_t + D ⊙ x_t           # [1, P]

    # ===== RRI (Reduced Memory Retention and Insertion) =====
    # 干预 α_t 和 B_t，无需重新训练
    α_t' = α_t ** 0.9999       # scale retention closer to 0
    B_t' = B_t * 0.75           # weaken insertion strength
    h_t = h_{t-1} * α_t' + Δ_t * B_t' * x_t

    # ===== Sliding Window (精确窗口状态) =====
    # 窗口大小 w
    # 维护三个量: h_{t-1}, h_{t-w-1}, Δ_sum = Σ_{i=t-w}^t Δ_i
    # 在 t 时刻:
    h_t = h_{t-1} * α_t + Δ_t * B_t * x_t        # 正常更新
    Δ_sum = prev_Δ_sum * (1 − reset_flag) + Δ_t  # 维护 Δ 累积和
    α_window = exp(-Δ_sum * exp(A))              # 窗口衰减因子
    h_t^{(w)} = h_t - α_window * h_{t-w}          # 精确窗口状态

    # 推理时使用 h_t^{(w)} 替代 h_t 进行 query
    y_t = C_t @ h_t^{(w)} + D ⊙ x_t
    ```
    关键洞察：(1) Mamba-2 的 state h_t 是加权和形式（h_t = Σ α_{i:t} B̄_i x_i），因此 Sliding Window 可精确计算为两个状态的差，无需重新处理窗口内所有 token；(2) 直接计算 α_{t−r:t} 可能因浮点精度不稳定，改为维护 Δ 累积和并每步重新计算 α_window；(3) 方法适用于所有可表为加权和的 RNN（GLA、RWKV、RetNet 等）。

  - 训练时 Truncated BPTT 实现：
    ```
    # 等价于序列拼接 + 梯度截断
    # 12 序列拼接，总长 ≈ 12 * T_train
    for batch in dataloader:
        h_0 = zeros(N, P)          # 初始化为零
        for seq in batch:          # batch 内 12 个序列
            for t in range(len(seq)):
                h_t = update(h_{t-1}, seq[t])
                loss += CE(linear_head(h_t), seq[t+1])
            h_0 = h_t.detach()     # 截断梯度，继续用当前 state
            # 下一个序列从 h_t 开始（状态延续）
    ```
