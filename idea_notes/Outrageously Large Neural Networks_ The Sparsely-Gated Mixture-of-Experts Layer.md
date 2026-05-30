## Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer

- baseline方法是什么？
  - Baseline 方法包括：
    - **Dense LSTM 模型**：Jozefowicz et al. (2016) 的 stacked LSTM（LSTM-512-512, LSTM-1024-512, LSTM-2048-512, 2xLSTM-8192-1024），所有参数对每样本均激活，模型容量受限于计算资源。参数从 2M 到 151M 不等。
    - **GNMT (Wu et al., 2016)**：9 层 LSTM encoder + 8 层 LSTM decoder 的机器翻译模型（278M 参数，214M ops/timestep）。所有参数对每样本均激活。
    - **计算匹配的密集 baseline（本文内部）**：MoE-1-Wide（单 FFN 4096 hidden）、MoE-1-Deep（4 层 1024 FFN）、4xLSTM-512、MoE-4（无稀疏，4 个 expert 始终激活）。
    - **GNMT-Multi (Johnson et al., 2016)**：单一 GNMT 模型处理 12 个语言对的翻译，但因容量有限，结果不如 12 个单独的单语言对模型。
  - 全栈执行例子（以 LSTM-2048-512 Baseline 为例，1 Billion Word LM Benchmark）：
    - **算法层**：输入句子 `x[1:T]` → Word Embedding (d_model=512) → 经过 2048 单元 LSTM 层（全激活）→ 输出投影到 512 → Softmax 层（重要性采样）预测下一个词。所有计算对所有 token 和所有时间步全激活。容量 = 参数数（~151M），计算开销 ≈ O(d_model × n_params) 随参数数线性增长。
    - **系统框架层**：数据并行：每个 GPU 处理独立的 batch，参数服务器同步梯度。Batch size 受激活值存储限制。GPU 间通信主要传输梯度。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：TensorFlow 框架 (Abadi et al., 2016) 在 Tesla K40 GPU 上执行标准 LSTM kernel（matmul, sigmoid, tanh），计算效率约 1.07-1.29 TFLOPS/GPU。
    - **硬件架构层**：Tesla K40 GPU 集群（16-32 卡），内存限制每 GPU 能容纳的参数规模。带宽需求与参数数成正比。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：
    - **Sparsely-Gated MoE Layer**：引入一个可训练的门控网络 G(x)，对每个输入选择稀疏的 k 个 expert 子集进行条件计算（而非全部）。输出 y = Σ_i G(x)_i · E_i(x)，其中 G(x) 是稀疏 n 维向量。每个 expert 是独立参数的 FFN。
    - **Noisy Top-K Gating**：在 Softmax 前加入可调高斯噪声（噪声幅度由 W_noise 控制），然后 KeepTopK 保留最大的 k 个值。噪声项使负载均衡 loss 可微（通过 Φ(CDF) 估计 P(x,i)）。
    - **两级辅助损失函数**：L_importance = w·CV(Importance)²（防止门控塌缩到少数 expert）和 L_load = w·CV(Load)²（防止 expert 负载不均，Load 为平滑估计器）。
    - **混合数据并行与模型并行**：标准层和门控网络用数据并行，各 expert 只保留一份共享副本（模型并行）。同一设备既做数据并行副本又做模型并行分片。所有数据并行 input batch 中的相关样本组合后送给每个 expert，使 expert batch size 放大 d 倍（d=设备数），解决 shrinking batch 问题。
    - **卷积式 MoE 应用**：等前一层对所有时间步完成后，将 MoE 卷积式应用于所有时间步，将 seq_len 折叠入 batch dim，进一步增大 expert batch size 至 b × T × kd / n。
    - **Hierarchical MoE**：主门控选择次级组，次级门控在组内选择 expert。第一级分支因子 = GPU 数，第二级在单 GPU 内执行，消除跨设备通信。
  - 对比 Baseline 的全栈执行改进（以 MoE-4096-h, k=4, 8M ops/timestep 为例）：
    - **算法层**：输入 x → Word Embedding (512) → LSTM (512) → 门控网络 G(x) 计算稀疏 top-k=4（从 4096 个 expert 中选 4 个，稀疏度 99.9%）→ 仅 4 个 expert 执行 FFN(1024 ReLU → 512) → 加权 sum → LSTM (512) → Softmax。模型参数从 ~151M 增至 ~4.3B（28×），但计算量仅 ~8M ops/timestep（baseline ~151M 的 5.3%）。Perplexity 从 34.7 降至 34.1（仅需 6% 的计算）。
    - **系统框架层**：16 GPU 同步训练。标准层 + 门控网络在每 GPU 上全复制（数据并行），4096 个 expert 分布到 16 GPU（每 GPU 256 个 expert，模型并行）。每个 expert 接收来自所有 16 GPU 的 input batch 中选中该 expert 的样本。branching factor=16（第一级 16 个次级组，每 GPU 一个），第二级 256 选 2。总 batch size ~300K words，expert batch ≈ kb×d/n ≈ 4×300K×16/4096 ≈ 4690 words/expert。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：TensorFlow 在 16-32 K40 GPU 上执行。Expert 计算占 37%-46% 的总浮点运算。计算效率 0.74-0.90 TFLOPS/GPU（低计算模型），最高 1.56 TFLOPS/GPU（高计算模型 MoE-143M），均为 Tesla K40 理论峰值 4.29 TFLOPS/GPU 的显著比例。关键瓶颈从"计算所有参数"变为"网络带宽传输 expert 输入/输出"——通过增大 expert hidden layer（1024/2048/8192）提高 compute-to-IO ratio。
    - **硬件架构层**：Tesla K40 GPU 集群。每 GPU 内存需求恒定（不随 expert 总数增加），因为每个 expert 参数固定为 ~1M，每 GPU 只需存储其托管的 256 个 expert。模型容量（#expert × params/expert）随设备数线性扩展。内存优化：不存储 expert hidden layer 激活（reverse 时重算），Adam 二阶矩使用分解近似（row-wise × col-wise outer product ÷ mean），降低优化器内存从 3× 至 ~2×。
