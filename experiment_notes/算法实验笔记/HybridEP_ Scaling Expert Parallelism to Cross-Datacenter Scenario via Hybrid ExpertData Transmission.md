## HybridEP: Scaling Expert Parallelism to Cross-Datacenter Scenario via Hybrid ExpertData Transmission

- 属于算法pipeline的实现是什么？实验比较什么？
  - HybridEP 的 SR-Based Expert Compression（共享-残差专家压缩）是一种算法级参数压缩技术，用于减少跨 DC 的专家传输流量。核心设计：
    1. **Shared Expert（共享专家）**：将所有 expert 参数取平均得到共享专家，由所有 GPU 共享。共享专家学习各专家间的冗余/共性知识，通过 backward 阶段的 All-Reduce 同步梯度。
    2. **Residual Expert（残差专家）**：每个 expert 的残差 = expert - shared_expert。残差分布更集中、更稀疏（Figure 9a 中的 "res" 分布），因为不同 expert 的主要差异集中在少量参数上。
    3. **Top-k Sparse Compression**：对残差 expert 应用 Top-k 压缩（保留绝对值最大的 k 个参数），以稀疏 value-index 格式传输。压缩比 (CR) 最高 50× 时仍不损失模型精度（Figure 14）。
    4. **SREncode/SRDecode 流水线**：编码阶段计算残差 → Top-k → value-index 格式存储；解码阶段从稀疏格式恢复残差 → 与 shared expert 相加恢复完整 expert。SRDecode 中将恢复和加法操作 fused 以减少 overhead（Figure 15b，与 expert computation 融合可减少 45% overhead）。
  - 实验比较：
    - **HybridEP w/ S**（有共享专家）vs **HybridEP w/o S**（直接 Top-k 压缩，无共享专家）vs Tutel/FasterMoE/SmartMoE baseline
    - Loss 曲线分析（Figure 14）：HybridEP w/S 的 loss 与 baseline 几乎一致（50× CR），HybridEP w/o S 的 loss 显著偏高
    - 时间分解分析（Figure 15）：不同 expert 大小下 SREncode/SRDecode 的 overhead 及 fusion 效果（SREncode+optimizer step 融合减 30%，SRDecode+expert computation 融合减 45%）

- 硬件平台是什么，配置是什么。
  - 4 种模型配置在 GPU 集群上训练评估，压缩效果验证使用与 Serving 调度实验相同的集群（Cluster-S/M/L: NVIDIA A800 GPUs）
  - 具体模型与数据集见 Table II：Llama-Tiny (PennTreebank)、Mistral-Small (WikiText2)、GPT-Medium (OpenWebText-10k)、GPT-Large (WikiText103)
  - 所有模型 expert 数 E=32，激活 expert 数 K∈{1,2,4}

- 模型是什么。数据集和bench分别是什么。
  - **模型**（Table II）：
    - **Llama-Tiny**：E=32, H=512, P_E=2.1M, #Layers=12, Dataset=PennTreebank
    - **Mistral-Small**：E=32, H=768, P_E=4.7M, #Layers=12, Dataset=WikiText2
    - **GPT-Medium**：E=32, H=1024, P_E=8.4M, #Layers=12, Dataset=OpenWebText-10k
    - **GPT-Large**：E=32, H=1024, P_E=8.4M, #Layers=16, Dataset=WikiText103
  - 数据集均为语言建模标准 benchmark
  - 压缩效果评估使用 loss 曲线（训练 loss 对比），通过 loss 值判断压缩对精度的影响

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码未公开开源。论文未提供开源链接。SR 压缩算法描述详见论文 §IV-B。
  - SR-Based Expert Compression 伪代码：
    ```
    # === 初始化 ===
    # 所有 experts 参数: experts[0..E-1], 每个 expert 大小为 P_E
    # 共享专家: shared_expert = mean(experts[0..E-1])
    # CR: 压缩比 (e.g., 50x)

    # === SREncode (编码阶段, 与 optimizer step 融合) ===
    def SREncode(expert, shared_expert, CR):
        # 1. 计算残差
        residual = expert - shared_expert  # size: P_E
        
        # 2. Top-k 压缩: 保留绝对值最大的 k 个参数
        k = P_E // CR
        values, indices = topk(abs(residual), k)
        # values: 保留的 k 个残差值(含符号)
        # indices: 对应在 P_E 中的位置
        
        return (values, indices)  # 稀疏 value-index 格式
    
    # === SRDecode (解码阶段, 与 expert computation 融合) ===
    def SRDecode(values, indices, shared_expert):
        # 1. 从稀疏格式恢复残差
        residual_recovered = scatter(indices, values, size=P_E)
        # 未保留位置填 0
        
        # 2. 恢复完整 expert (fused with addition)
        expert_recovered = shared_expert + residual_recovered
        
        return expert_recovered
    
    # === 训练 iteration 中的使用 ===
    # -- 前一步: Initialization 阶段 (与 optimizer.step() 融合) --
    for each expert in local_experts:
        compressed = SREncode(expert, shared_expert, CR)
        send_queue.push(compressed)  # 存入异步发送队列
    
    # -- 当前步: Asyn-comm 阶段 --
    # GPU i 从 send_queue 取出压缩 expert 残差
    for compressed in send_queue:
        # NCCL All-Gather: 域内所有 GPU 收集彼此的压缩 expert
        all_compressed = all_gather(compressed, group=domain_group)
    
    for compressed in all_compressed:
        expert = SRDecode(compressed.values, compressed.indices, shared_expert)
        recv_queue.push(expert)  # 供 expert FFN 计算使用
    
    # -- Expert FFN 计算 --
    for expert, tokens in zip(recv_queue, token_batches):
        output = expert_ffn(expert, tokens)  # 标准 MoE expert 前向
    ```
  - **张量计算示例**（Mistral-Small, E=32, H=768, P_E=4.7M, CR=50, 单个 expert）：
    - 输入：expert 参数 W ∈ R^{4.7M}（gate/up/down 矩阵展平后的总参数量）
    - Shared expert：W_shared = mean(W_0, W_1, ..., W_31) ∈ R^{4.7M}
    - Residual：R = W - W_shared ∈ R^{4.7M}
    - Top-k (k = 4.7M/50 ≈ 94k)：保留 |R| 最大的 94k 个元素
    - 输出：values[94k] + indices[94k]，压缩后数据量 = 94k × (FP16 + INT32) ≈ 0.56 MB
    - 对比原始 P_E=4.7M × FP16 ≈ 9.4 MB，压缩比 ≈ 16.8×（与带宽相关）
    - 论文用 50× CR：P_E=0.094 MB per expert (Table IV, AG-only 配置)，与 P_E=4.7 MB 相比

  - **关键结论**：SR-Based Expert Compression 的核心洞察是 experts 间存在知识冗余（residual 分布比原始 weight 更集中，Figure 9a），通过 shared + residual 分解可以安全地以高压缩比（50×）压缩专家参数，在几乎不损失模型精度的情况下大幅减少跨 DC 传输流量。无 shared expert 的 naive Top-k 压缩会导致显著精度损失（Figure 14），证明 shared expert 对维护精度至关重要。
