## MoE-ERAS: Expert Residency Aware Selection

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoE-ERAS 提出两种 residency-aware 的 expert 选择算法，在 MoE 推理时修改 gating network 的输出，使路由器偏向选择已驻留在 HBM（fast memory）上的 expert，减少 host-to-device 的 expert 加载：
    1. **Thresholding（阈值法）**：对已驻留在 fast memory（HBM）中的 expert，在其 softmax 概率上添加用户定义的超参数 α（Weights_i += α），人工提升 on-chip expert 的激活概率，使得"足够好"的 on-chip expert 能够击败略微更好的 off-chip expert。
    2. **Biasing（偏置法）**：在 softmax 之前，对 off-chip expert 的 logits 施加惩罚 β(1 - freq(E_i))，其中 freq(E_i) 是 expert 在 profiling 阶段收集的归一化激活频率。频率越低的 off-chip expert 惩罚越重——因为冷门 expert 被加载到 HBM 后大概率很快被换出，导致两次 swap。相比 thresholding，biasing 额外考虑了 expert 的热度信息。
  - 实验比较：(1) Top-K routing baseline（含 quantization + LRU caching）vs Thresholding（α=0.05, 0.15, 0.25）vs Biasing（β=1）的解码延迟和 expert swap 次数；(2) 不同 offload per layer 设置下的 speedup；(3) WikiText2-PPL、C4-PPL、MMLU-Acc 的 quality-speedup trade-off。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA H100（用于图 2 的 CPU vs GPU expert read time 对比）
  - 主机内存：CPU DRAM 用于 offload expert 参数
  - Baseline 框架（dvmazur/mixtral-offloading）可在 Tesla T4 16GB 上运行 Mixtral-8x7B
  - 计算精度：论文未明确说明具体精度（baseline 使用 quantization）

- 模型是什么。数据集和bench分别是什么。
  - 模型：
    - Mixtral-8x7B：32 hidden layers，每层 8 experts，Top-K=2（主要实验模型）
    - Switch Transformer-32E：6 hidden layers，每层 32 experts（仅 profiling）
  - 数据集：
    - CNN DailyMail（profiling expert activation patterns，139k tokens for Mixtral，500k tokens for Switch Transformer）
    - WikiText2（test set，perplexity 评估）
    - C4（validation set，perplexity 评估）
  - Benchmark：
    - MMLU（5-shot accuracy，完整数据集）
    - 解码延迟（wall clock time）、throughput（tokens/sec）、expert swaps saved

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文自身代码未开源。Baseline 基于开源项目 `dvmazur/mixtral-offloading`（https://github.com/dvmazur/mixtral-offloading），该 baseline 已包含 quantization 和 LRU caching。
  - 算法 pipeline 伪代码：
    ```
    # 标准 MoE Gating（Baseline Top-K）
    # H_i: 第 i 层的 self-attention 输出
    Logits = H_i @ W_exp                    # (seq_len, num_experts)
    Weights = Softmax(Logits)               # (seq_len, num_experts)
    Activated = SelectTopK(Weights, k=2)    # 选择 Top-2 experts

    # === MoE-ERAS Thresholding ===
    # residency[e]: True 表示 expert e 当前在 HBM 上
    Weights = Softmax(Logits)
    for e in range(num_experts):
        if residency[e]:  # expert 在 fast mem (HBM)
            Weights[:, e] += alpha           # 添加阈值偏置 α
    Activated = SelectTopK(Weights, k=2)

    # === MoE-ERAS Biasing ===
    # freq[e]: 从 profiling 收集的归一化激活频率
    Logits = H_i @ W_exp
    for e in range(num_experts):
        if not residency[e]:  # expert 在 slow mem (CPU)
            Logits[:, e] -= beta * (1 - freq[e])  # 频率越低惩罚越大
    Weights = Softmax(Logits)
    Activated = SelectTopK(Weights, k=2)
    ```
    关键张量维度：H_i ∈ R^{seq_len × hidden_dim}, W_exp ∈ R^{hidden_dim × num_experts}。MoE-ERAS 在 softmax 前后修改 logits/weights，不改变模型参数，仅在推理时生效。
