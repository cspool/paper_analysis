## LocMoE: A Low-overhead MoE for Large Language Model Training

- 属于算法pipeline的实现是什么？实验比较什么？
  - LocMoE 提出三种算法 pipeline 优化用于 MoE 大语言模型训练：
    1. **GrAP（Grouped Average Pooling）正交门控权重层**：用分组平均池化替代传统 Dense 层计算门控值 G_{m,E_i}。权重矩阵 ω_i 为固定正交矩阵，元素为 0 或 1，按分组聚集：ω_{i,j} = 1{i·d/n ≤ j < (i+1)·d/n}。正交性使不相关 token 倾向被路由到不同领域的 expert，利于收敛和精度，同时避免 Dense 层的 FLOPs 开销。
    2. **局部性专家正则化（Locality Loss）**：在辅助负载均衡 loss L_aux = α·n·Σ f_i·P_i（α=0.01）基础上，增加 KL 散度正则化项 L_loc = μ·KL(D_c || D_l)，其中 D_c 为当前 expert 分配分布，D_l 为完全局部化的理想分布。该 loss 促使同一节点的 token 优先路由到本地 expert，将跨节点 All-to-All 通信转化为节点内高带宽通信。总 loss 为 L_task = L_aux + L_loc + L_cross。
    3. **专家容量下界理论（Expert Capacity Lower Bound）**：首次在 NLP 领域证明了 MoE expert capacity 存在临界值。基于高维球面均匀分布假设，推导出 ec_min ≥ 1/(n·erfc(√(δ²d/(2-δ²))))，其中 δ 为 token 与 gating weight 夹角余弦的最小阈值。实验测得 δ≈0.03，可据此下界安全降低 expert capacity 而不损失精度。
  - 系统层面：使用 MindSpore 内置的 Group-wise All-to-All 将通信拆分到 TP 域高速带宽 + EP 域，并实现 FFN 计算与 All-to-All 通信的切片重叠（slice-and-overlap），进一步隐藏通信延迟。
  - 实验比较：
    - LocMoE vs HashMoE（基于哈希函数的绝对均衡路由）vs SwitchMoE（Top-1 gating with auxiliary loss）
    - 每 epoch 训练时间减少：64N 下 12.68%~22.24%，128N 下也有显著加速
    - Expert 分配均衡性、收敛速度（valid perplexity）、多 NLP 任务推理精度
    - All-to-All 通信时间下降 5.13%（64N/128N）
    - Ablation：计算/通信/重叠/空闲时间占比分析，在不同规模（64N, 128N, 256N）下对比
  - 结果：64N 下相对 HashMoE 加速 1.15x，相对 SwitchMoE 加速 1.29x；LocMoE 在 256N 下不如 HashMoE（因部分节点无 expert，locality 失效）

- 硬件平台是什么，配置是什么。
  - **Ascend 910A NPU 集群**：3 组配置
    - 64N：8 节点 × 8 Ascend 910A（64 NPUs）
    - 128N：16 节点 × 8 Ascend 910A（128 NPUs）
    - 256N：32 节点 × 8 Ascend 910A（256 NPUs）
  - 每 Ascend 910A：32 AI Cores，最大内存 2TB，最大内存带宽 1.07TB/s，FP16 算力 320 TFLOPS，INT8 算力 640 TOPS
  - 服务器型号：Atlas 800 9000，每 8 个 NPU 通过 HCCS（Huawei Cache Coherence System）互联，节点间通过两级 Fat-tree 网络 + RoCE 互联
  - 软件栈：CANN 5.1.RC2.1 (toolkit 1.84, driver 23.0.rc2)，MindSpore 2.0.0
  - 通信库：HCCL（Huawei Collective Communication Library），支持 ring/mesh/HD/ring+HD/mesh+HD 算法

- 模型是什么。数据集和bench分别是什么。
  - **模型**：PanGu-Σ，1.085T 参数稀疏 MoE 模型（从密集模型 PanGu-α 扩展而来），包含 Dense + Sparse Transformer Encoder layers + Decoder layers + Query layer。稀疏层含 RRE（Random Routing with Expert selection）两级路由：第一级按领域分组，第二级（原为随机哈希路由 → 被 LocMoE 替换）路由到组内具体 expert。配置：16 experts，8 MoE layers，40 attention heads，batch size 32，expert parallel=16。
  - **数据集**：华为内部移动网络运营商服务文档语料
    - iCase（技术案例）：591,972 文档，387M tokens
    - Wiki（内部知识管理平台）：1,146,755 文档，1,162M tokens
    - 核心网/MML：223,898 文档，137M tokens
    - 配置翻译（Huawei/Cisco 产品文档）：1,460,680 文档，560M tokens
    - 特性文档（4G/5G FAQ, fault tree 等）：86,913 文档
    - 语料为中/英/双语，格式包括 Word/PDF/HDX/HTML
  - **Benchmark**：自定义 NLP 任务评估集（从 10 个业务角度提取），包括故障树节点识别（L1-L3 难度分级）、方案类、ICT 认证考试、标题改写等。每个任务 30~80 条 Q&A 作为评估集，人工评分。
  - 预训练目标：自回归语言模型（cross-entropy loss），验证指标为 valid perplexity

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：论文未提供开源代码。LocMoE 基于 Huawei 内部 MindSpore 框架和 PanGu-Σ 模型实现，运行于 Ascend NPU 集群，均为商用/内部系统。MindSpore 开源（https://github.com/mindspore-ai/mindspore），但 LocMoE 特定修改未公开。
  - **算法 pipeline 伪代码**：
  ```python
  # LocMoE 前向传播核心流程
  # Input: token_embeddings x_m of shape [T, d]
  # Experts: {E_0, ..., E_{n-1}} on EP domain, n=16
  # Hyperparams: alpha=0.01, mu (locality weight)

  # Step 1: GrAP Gating (替代 Dense Gating)
  # 分组平均池化: 将 d 维 token 均分为 n 组，每组取均值作为门控值
  # x_m: [T, d] -> reshape [T, n, d//n] -> mean(dim=2) -> [T, n]
  # 等价于正交权重矩阵 omega 与 x_m 的内积
  # omega_i 满足: omega_{i,j} = 1 if i*d/n <= j < (i+1)*d/n else 0
  gate_logits = x_m.reshape(T, n, d // n).mean(dim=2)  # [T, n]

  # Step 2: Top-1 Routing with Softmax
  gate_probs = softmax(gate_logits, dim=1)  # [T, n]
  expert_idx = argmax(gate_probs, dim=1)    # [T]

  # Step 3: Locality-Aware Token Dispatch
  # Group-wise All-to-All: 将 All-to-All 按 TP 域拆分
  # 每个 device 在 EP 域内负责部分传输，All-Gather 在 TP 域同步
  tokens_dispatched = groupwise_all_to_all(x_m, expert_idx)

  # Step 4: Expert FFN 计算 (切片与 All-to-All 重叠)
  # 每个 expert 执行: W_out · GeLU(W_in · x)
  for expert_i in range(n):
      mask_i = (expert_idx == expert_i)
      tokens_i = tokens_dispatched[mask_i]
      if len(tokens_i) > 0:
          output_i = W_out[expert_i] @ GeLU(W_in[expert_i] @ tokens_i)

  # Step 5: Combined Loss
  # Auxiliary loss (Switch Transformer 风格负载均衡)
  f_i = sum(expert_idx == i) / T  # expert i 的 token 比例
  P_i = mean(gate_probs[:, i])    # router 选择 expert i 的平均概率
  L_aux = alpha * n * sum(f_i * P_i for i in range(n))

  # Locality loss (KL 散度促使局部路由)
  # D_c: 当前 batch 中 (节点, expert) 的 token 分配分布
  # D_l: 完全局部化的理想分布（token 只在本地 expert）
  L_loc = mu * KL_divergence(D_c, D_l)

  # Task loss
  L_task = L_aux + L_loc + L_cross
  ```
  - **Expert Capacity 下界推导**：
    1. GrAP 层的正交门控权重满足 Lemma 2（各 expert 被等概率选择：P{i_j = i'} = 1/n）
    2. 基于高维球面几何（Lemma 3），当 d 很大且 δ = Θ(1/√d) 时，token 应分配给某 expert 的概率 p_δ ≈ 0.3
    3. 当 δ 增大（即夹角变小，token 与 expert 更匹配），p_δ 快速衰减至 0（仅少量 token 为 class-discriminative）
    4. 由此得到 ec_min = 1/(n·[1 - I_{δ²}(1/2, (d-1)/2)])，在大 d 下退化为 ec_min ≥ 1/(n·erfc(√(δ²d/(2-δ²))))
    5. 论文实验测得 δ≈0.03，可据此计算安全的 expert capacity 下界，避免超量分配
