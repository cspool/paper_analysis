## On-device Semantic Selection Made Low Latency and Memory Efficient with Monolithic Forwarding

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：**Progressive Cluster Pruning**——训练无关（training-free）的候选文档渐进式聚类剪枝算法。核心思路：利用 cross-encoder reranker 各层输出的候选文档分数渐进分散为统计显著的聚类（sequence-level sparsity），通过聚类分析将候选文档三路路由（selected / deferred / dropped），仅对边界聚类中的 uncertain candidates 继续前向计算。具体流程：
    1. 每层执行前，用模型原始 classifier 计算所有候选文档的当前 relevance score
    2. 计算分值的变异系数（CV = |std(scores) / mean(scores)|），判断是否超过预设 dispersion threshold
    3. 若 CV > threshold，触发 K-Means 聚类（在 CPU 上执行，约 ~1 ms）
    4. 识别包含第 K 个候选文档的 boundary cluster，以此为界：
       - 高于 boundary cluster 的簇：**selected**（已确定进入 top-K，停止计算）
       - 低于 boundary cluster 的簇：**dropped**（已确定无缘 top-K，剪枝）
       - boundary cluster 内部：**deferred**（继续下一层计算）
    5. 当 deferred 候选数 = 剩余 top-K 名额时，立即终止推理
  - Dispersion threshold 提供 precision-latency 可调 trade-off：低 threshold → 激进剪枝→更快速但可能损失精度；高 threshold → 保守→精度更高。系统支持自动校准：对采样请求定期重跑完整推理做 ground truth，若精度低于目标则提高 threshold，反之降低。
  - 实验比较：
    - Baseline：HF（HuggingFace Transformers 标准 in-memory 推理）、HF Offload（HuggingFace Accelerate disk offloading）、HF Quant（GPTQ W4A16 量化）、PRISM Quant（PRISM + 量化，验证正交性）
    - 指标：Latency（ms）、Precision@K（K=1/5/10）、Peak/Average Memory（MB）
    - Microbenchmark 18 数据集（BEIR 15 datasets + LoTTE + Wikipedia + CodeRAG），5 模型（0.6B–8B），2 平台（NVIDIA RTX 5070 Laptop, Apple M2）
    - Real-world 3 场景：RAG、Agent Memory、LLM Long Context Selection

- 硬件平台是什么，配置是什么。
  - **NVIDIA Platform**：笔记本 + Intel Ultra9-275HX + 32 GiB RAM + RTX 5070 Laptop GPU（8 GiB VRAM）+ 1 TiB PCIe 4.0 SSD
  - **Apple Platform**：Mac Mini + Apple M2 SoC + 16 GiB unified memory + 256 GiB PCIe 4.0 SSD
  - **额外 GPU**（仅用于 HF baseline OOM 测量）：NVIDIA A800

- 模型是什么。数据集和bench分别是什么。
  - 模型（5 个，encoder-only 和 decoder-only，0.6B–8B）：
    - Qwen3-Reranker-0.6B（decoder-only）、Qwen3-Reranker-4B（decoder-only）、Qwen3-Reranker-8B（decoder-only）
    - Bge-Reranker-v2-MiniCPM（decoder-only, 2B）、Bge-Reranker-v2-M3（encoder-only, 0.6B）
  - 数据集与 Benchmark：
    - Microbenchmark：BEIR 15 datasets + LoTTE + Wikipedia + CodeRAG（18 个总计）
    - Real-world：RAG（Milvus + Qwen3-Embedding-0.6B + Qwen3-32B server）、Agent Memory（MobiAgent video/community scenarios + MobiMind-Decider-7B VLM）、LLM Long Context Selection（LongBench2 + 量化 Qwen3-4B-Instruct）
  - 指标：Precision@K（K=1/5/10）、Latency（ms）、Peak/Average Memory（MB）

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源状态：**已开源**。源码：https://ipads.se.sjtu.edu.cn:1312/opensource/monolithic_forwarding，存档 DOI：10.5281/zenodo.18809731。EUROSYS '26 Artifact Evaluation。基于 HuggingFace Transformers v4.52.4 + Accelerate v1.6.0。~5K lines Python + ~1.7K lines C。
  - 算法 Pipeline 全过程（以 20 候选选 top-10 为例）：

    ```
    === Monolithic Forwarding 初始化 ===
    输入：query q，候选文档集 D = {d_1, ..., d_20}（各 ≤512 tokens）
    目标：选出 top-K=10
    1. 将所有候选与 query 拼接为单一 monolithic batch：
       input = [CLS] q [SEP] d_i [SEP]  for i=1..20
       shape = [20, L]（L ≤ 512）
    2. Embedding：查 embedding table cache，仅加载激活 token embedding
    3. Chunked Execution：将 20-candidate batch 分区为 chunks（如 chunk_size=2 → 10 chunks）

    === 逐层前向 + Progressive Cluster Pruning ===
    对 layer i = 0, 1, ..., L-1：
      
      # 1a. 逐 chunk 执行 layer i 前向
      for chunk in remaining_candidates:
          h_chunk = TransformerLayer_i(h_chunk)  # self-attn + FFN, [chunk_size, L, D]
      
      # 1b. Progressive Cluster Pruning（layer i 执行后）
      scores = classifier_head(h[:, last_token, :])  # [num_remaining]
      cv = |std(scores) / mean(scores)|
      
      IF cv > dispersion_threshold:  # 排名稳定，触发剪枝
          clusters = KMeans(scores.reshape(-1,1))  # CPU 上 ~1ms
          sorted_clusters = sort_by_mean_score(clusters)
          boundary_idx = find cluster containing K-th ranked candidate
          
          selected  = sum(clusters[:boundary_idx])       # 确定进入 top-K
          deferred  = clusters[boundary_idx]              # 不确定，继续
          dropped   = sum(clusters[boundary_idx+1:])     # 确定无缘 top-K
          
          final_topK.extend(selected)
          remaining_candidates = deferred  # 仅边界簇继续
          
          IF len(final_topK) + len(deferred) == K:
              全部 deferred 加入 final_topK，**终止推理**，BREAK
      
      ELSE: 继续下层，不剪枝

    === 输出 ===
    return final_topK  # K 个最相关候选

    === 关键张量形状（Qwen3-Reranker-0.6B single layer）===
    h ∈ R^{B × L × D}  B=remaining candidates, L≤512, D=hidden_dim
    Q/K/V = h @ W_{Q/K/V}  [B,L,D]
    attn = causal_softmax(QK^T / sqrt(d))  [B,L,L]（decoder-only）
    h = h + FFN(attn @ V)   [B,L,D]
    score = classifier_head(h[:, last_token, :])  [B]
    ```

## IntAttention Fully Integer Attention Pipeline for Edge LLM Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：IndexSoftmax —— 一个训练无关（training-free）的全整数 softmax 替代算子，作为 INT8 注意力 pipeline 的 drop-in replacement。包含三个核心组件：
    1. **Sparsity-aware Clipping**：利用 softmax 中远离最大值的 logit 贡献近乎为零的性质，对 logits 做 max-subtraction 后裁剪至 [0, c] 范围（c=6.6），跳过可忽略的指数计算。
    2. **32-entry LUT Exponential**：利用裁剪后指数域有界（[0, c=6.6]）的特性，用 32 条目的紧凑单调查找表替代 FP32 exp 计算。每个条目仅 1 字节（UINT8），总 LUT 仅 32 bytes，可放入寄存器。
    3. **Integer Normalization（UINT8）**：将 softmax 概率 P 矩阵输出为 UINT8（而非 INT8），在相同 32B 表预算下获得 4× 更细粒度（256 vs 128 个值），保证行归一化精度。CosSim vs FP32 = 0.999081（vs INT8 = 0.996612）。
  - 整体 pipeline 数据流：`S8 × S8 → S32`（QK^T 整数累积）→ `S32 → U8`（IndexSoftmax 概率路径）→ `U8 × S8 → S32`（PV 混合）。全程无浮点转换，消除 dequantize → softmax → requantize 的瓶颈。
  - 实验比较：
    - Baseline：FP16（完整浮点注意力）、EXAQ（INT2/INT3 量化，带动态裁剪的 LUT softmax）、IndexSoftmax（孤立算子级评估）、Quant-Only（INT8 GEMM 但 softmax 仍为 FP32 浮点——即仅量化矩阵乘、不量化 softmax 路径）。
    - 指标：
      - 速度：attention latency（μs），RK3588S2 与 Apple M2
      - 能量：attention energy（mJ），RK3588S2 USB 功率计
      - 精度：WikiText PPL↓、HellaSwag/LAMBADA/PIQA/WinoGrande/ARC-C/ARC-E Accuracy↑、C4 PPL↓、HumanEval/MBPP/GSM8K/IFEval Accuracy↑、ImageNet-1K Top-1/Top-5↑
      - 消融：裁剪参数 c 与 LUT 位数 b 的 sweep（c ∈ [4,8]，b ∈ [4,8]，发现 c=6.6, b=5 为稳定区域）

- 硬件平台是什么，配置是什么。
  - RK3588S2 嵌入式开发板：ARM Cortex-A76 + A55 CPU，Armv8.2-A 架构
  - Apple M2 笔记本电脑：ARM-based SoC，Apple Silicon
  - 软件依赖：Arm Compute Library (ACL)，内置 INT8 GEMM kernel
  - 编译：clang++，scons 构建 ACL

- 模型是什么。数据集和bench分别是什么。
  - 语言模型：
    - LLaMA-3.2-1B（主要实验模型）
    - LLaMA-3.2-1B-Instruct（指令微调版本，用于推理/代码/数学评估）
    - OPT-1.3B、Qwen3-1.7B（扩展评估，论文 Figure/Table 中有提及）
  - 视觉模型：
    - DeiT-B-224（Vision Transformer，ImageNet-1K 预训练）
    - ViT-L-P16-384
    - CaiT-L-M48-448
  - 数据集与 Benchmark：
    - 语言 perplexity：WikiText-2、C4、OWT-10k、RedPajama
    - 语言理解：HellaSwag、LAMBADA、PIQA、WinoGrande、ARC-C（Challenge）、ARC-E（Easy）
    - 推理/代码/数学（仅 Instruct 模型）：HumanEval（代码生成）、MBPP（代码生成）、GSM8K（数学推理）、IFEval（指令遵循）
    - 视觉：ImageNet-1K（Top-1/Top-5 Accuracy）

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源状态：**已开源**，MLSys 2026 官方代码库 https://github.com/WanliZhong/IntAttention。Python 69.2% + C++ 30.8%。License 未在 README 中明确说明。
  - 算法 Pipeline 全过程（以 LLaMA-3.2-1B 单层 attention 为例）：

    ```
    === 注意力层输入 ===
    输入：INT8 量化后的 Q, K, V ∈ Z^{L × d}（S8 格式，即 signed int8）
    其中 L=序列长度，d=head_dim

    === 阶段 1：QK^T 整数累积（S8 × S8 → S32） ===
    1. S = Q × K^T     # ACL INT8 GEMM，输出 S32 累加器
       # S ∈ Z^{L × L}，每个元素为 int32
       # 与传统 attention 相同，但全程在整数域

    === 阶段 2：IndexSoftmax（S32 → U8） ===
    输入：S ∈ Z^{L × L}（int32 logits），每行独立处理
    对每一行 s ∈ Z^L：
      2a. Clipping（稀疏感知裁剪）：
          s_max = max(s)                     # 行内最大值（int32）
          s_shifted = s - s_max              # 减去最大值，范围 ≤ 0
          s_clipped = clamp(s_shifted, -∞, 0)  # 只保留 > -c 的元素
          # c = 6.6（固定超参），裁剪后 s_clipped ∈ [-c·q_scale, 0]
          # 实际实现中 c 转换为整数阈值：c_int = c × (Q_scale × K_scale)
          mask = (s_shifted > -c_int)        # boolean mask
          s_valid = s_shifted[mask]          # 仅保留有效 logits

      2b. LUT Exponential（32-entry 查表）：
          # LUT[i] = round(exp(-i / 2^b) × 255)，i = 0..31
          # b = 5，即每步 1/32，覆盖 [0, 1] → [exp(0), exp(-~1)]
          # 实际 LUT 实现需将整数 logit 值映射至 LUT 索引
          idx = quantize_to_lut_index(s_valid / q_scale)  # 量化为 0..31
          prob = LUT[idx]                                  # UINT8 概率值 ∈ [0, 255]

      2c. Integer Normalization（UINT8 归一化）：
          sum_prob = sum(prob)                # 行内概率和（uint32）
          # 对未裁剪元素：prob_norm[i] = prob[i] / sum_prob
          # 使用整数除法 + 移位近似：
          # prob_norm[i] = (prob[i] << 8) / sum_prob  # 定点归一化
          # 对已裁剪元素：prob_norm[j] = 0
          P_row = prob_norm                   # UINT8，和为 255（近似）

    输出：P ∈ [0, 255]^{L × L}（UINT8），每行近似满足 ΣP_i ≈ 255

    === 阶段 3：PV 整数混合（U8 × S8 → S32） ===
    3. O = P × V     # ACL INT8 GEMM（P 为 U8→S8 reinterpret，V 为 S8）
       # O ∈ Z^{L × d}（int32 累加器）
       # 与传统 attention 输出同格式，后续可 requantize 回 S8

    === 端到端对比 ===
    传统 INT8 Pipeline（Quant-Only Baseline）：
      S8×S8→S32 → dequant(S32→FP32) → FP32 softmax → quant(FP32→S8) → S8×S8→S32
                                    ↑_______ 瓶颈（≤65% latency）_______↑
    IntAttention Pipeline：
      S8×S8→S32 → IndexSoftmax(S32→U8) → U8×S8→S32
                  ↑ 全整数，无类型转换 ↑
    ```

  - PyTorch 模拟代码（`pysimulation/`，用于 GPU 精度验证）：`acc_llm.py --model-name llama-3.2-1b --method int_attention` 加载 HuggingFace 模型并替换 attention 的 softmax 为 IndexSoftmax 的 PyTorch 模拟实现（使用高精度算术模拟整数行为），然后通过 lm_eval 评估各 benchmark。
  - C++ 延迟 Benchmark（`bench_speed.cpp`，用于 ARM CPU 实测）：`bench_speed --pipe 3 --L 1024 --d 128 --warmup 10 --runs 100`，比较 pipe 0/1/2/3 分别对应 FP32/FP16/INT8 Quant-Only/IntAttention 四种 pipeline。

## Federated Fine-Tuning of Sparsely-Activated Large Language Models on Resource-Constrained Devices

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：FLUX 系统（~3K 行 Python/PyTorch），包含三个核心模块：
    1. **Quantization-based Local Profiling（§4）**：使用量化 MoE 模型（如 INT4）在本地估计 expert activation frequency，避免运行全精度模型的开销。通过 stale profiling 机制将 profiling 与 parameter aggregation 并行执行，隐藏时间开销。
    2. **Adaptive Layer-Aware Expert Merging（§5）**：非调优 expert 不丢弃而是合并。基于 (a) 各层 expert activation 分布的 variance 和 (b) 各层 error accumulation 特性（浅层误差传播更大），自适应分配每层的 merge budget（公式 1：B_i^non(l) = floor(b_i^l / Σb_i^l × B_i^non)，b_i^l = (L-l+1)/v_i^l）。合并时通过 PCA 降维 + K-Means 聚类（跨层融合聚类，40× 加速），再按 attention score × activation frequency 加权合并（公式 2：W_merged = Σ(α_e/Σα_e')W_e，α_e = f_e · ā_e）。
    3. **Dynamic Expert Role Assignment（§6）**：定义 expert utility u_i^e = |D_i^e|√(1/|D_i^e| Σ∇g_k)，基于 gradient magnitude 和 data utilization。用 exploration-exploitation 策略（动态 ε）：ε 比例选高 utility expert 做 exploitation（backprop 计算梯度），1-ε 比例随机探索（forward-only gradient estimation，加小扰动估计梯度，省去 backprop）。优化问题 max Σx_i^e u_i^e s.t. Σx_i^e ≤ B_i^tune（公式 4）在 parameter server 上求解，下发 expert role assignment 到各 participant。
  - 实验比较：
    - Baseline：FMD（expert offloading CPU↔GPU）、FMQ（INT4 量化全模型 fine-tuning）、FMES（按 activation frequency 选 expert subset fine-tuning，类似 FedMoE）
    - 指标：time-to-accuracy（ROUGE-L 0.5 for Dolly, Accuracy 0.62 for GSM8K, 0.75 for MMLU, 0.8 for PIQA）
    - 收敛：FLUX vs baselines 在 LLaMA-MoE 和 DeepSeek-MoE 上的 ROUGE/Accuracy 曲线
    - 可扩展性：participant 数量 10→30 的 time-to-accuracy 变化
    - Ablation：stale profiling 误差 vs 时间节省、adaptive layer size vs uniform/single、expert clustering 跨层融合加速比、merging 策略对比（Avg. vs Freq.-weighted vs Att.+Freq.-weighted）、gradient estimation 精度、动态 ε vs 固定 ε=0.3/0.7、FLUX 额外开销占比（~5%）

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA L20，48GB GPU memory，PCIe 互联
  - OS：Ubuntu 20.04，Linux kernel 5.15.0
  - 驱动/CUDA：NVIDIA driver 550.67，CUDA 11.8
  - 架构：parameter-server-based federated learning，一个 central server 协调 N 个 participant，FedAvg 聚合
  - 论文未明确说明 server 端硬件配置；participant 端为 consumer-grade GPU（L20 48GB）

- 模型是什么。数据集和bench分别是什么。
  - 模型：
    - LLaMA-MoE（6.7B params，32 layers，16 experts/layer，13.48GB）
    - DeepSeek-MoE（16.4B params，28 layers，64 experts/layer，32.77GB）
    - 论文 Table 1 还列出 DeepSeek-v2-lite (15.7B)、Mixtral-8x7B (46.7B)、Qwen2-MoE (57.4B) 作为参考但未用于实验
  - 数据集（均按 FedNLP benchmark 分割为 non-IID）：
    - Dolly（15K+ records，instruction-tuning）
    - GSM8K（8.5K，grade-school math）
    - MMLU（multi-task，multiple-choice）
    - PIQA（commonsense reasoning）
  - Benchmark 指标：ROUGE-L (Dolly)、Accuracy (GSM8K/MMLU/PIQA)
  - 数据划分：80% fine-tuning / 20% testing

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源状态：**论文未提供开源代码链接**。arXiv 页面 (arxiv.org/abs/2508.19078) 及论文正文均未提及 GitHub repository 或 artifact。EUROSYS '26 论文。
  - 算法 Pipeline 全过程（以 LLaMA-MoE 在 participant i 上一轮 fine-tuning 为例）：

    ```
    === 阶段 0：Stale Profiling（与上一轮 parameter aggregation 并行） ===
    输入：上一轮下载的全局模型 w^{r-1}
    1. 对 w^{r-1} 做 INT4 量化，得到 quantized MoE model
    2. 用量化模型对本地数据 D_i 做 forward pass
    3. 记录每个 expert e 的激活情况：
       - f_e = 激活 expert e 的 token 数 / 总 token 数  （activation frequency）
       - D_i^e = 流经 expert e 的 token 子集
       - ā_e = 流经 expert e 的 token 的平均 attention score
    输出：expert activation profile {f_e, D_i^e, ā_e}

    === 阶段 1：Expert Role Assignment（parameter server 执行） ===
    输入：所有 participant 上报的 expert utility u_i^e
    4. 求解优化问题（公式 4）：
       max Σ_i Σ_e x_i^e u_i^e
       s.t. Σ_e x_i^e ≤ B_i^{tune}, ∀i ∈ N
       得到候选 expert 集合 E_i
    5. Exploitation：从 E_i 中选 ε|E_i| 个最高 utility 的 expert → E_i^{exp}
    6. Exploration：随机选 (1-ε)|E_i| 个 expert → E_i^{exl}
    7. 发送 E_i^{exp} ∪ E_i^{exl} 到 participant i

    === 阶段 2：Non-Tuning Expert Merging（participant 本地执行） ===
    输入：当前 MoE 模型权重 w^r，expert activation profile
    8. 对 layer l = 1..L：
       a. 计算 variance v_i^l = Var(各 expert 的 activation frequency)
       b. 计算 merge budget B_i^{non}(l)（公式 1）：
          b_i^l = (L-l+1) / v_i^l
          B_i^{non}(l) = floor(b_i^l / Σ_k b_i^k × B_i^{non})
    9. 跨层融合聚类：
       a. PCA 降维 expert 参数 → low-dim feature vectors
       b. 初始化 Σ_l B_i^{non}(l) 个 centroid，标注 layer 标签
       c. 计算所有 expert-centroid cosine distances（矩阵运算）
       d. layer 间距离设 0（保证层内聚类）
       e. K-Means 迭代至收敛
    10. 每 cluster c 内合并（公式 2）：
        W_merged = Σ_{e∈E_c} (α_e / Σ_{e'∈E_c} α_e') · W_e
        其中 α_e = f_e × ā_e（activation frequency × average attention score）
    11. Gate re-routing：更新 gating network，将原 expert index 重映射到 merged expert

    === 阶段 3：Local Fine-Tuning ===
    输入：customized MoE（tuning experts + merged non-tuning experts）
    12. Tuning experts E_i^{exp}：
        - 加载完整 FP32 参数，参与 backprop
        - 用 D_i^e 中对应 expert e 的数据进行训练
    13. Exploration experts E_i^{exl}：
        - 加小扰动 ξ ~ N(0,σ²) 到 expert 参数
        - Forward pass 计算 loss L(W+ξ) 和 L(W)
        - 梯度估计：∇̂ = (L(W+ξ) - L(W)) / ξ（forward-only，省 backprop）
    14. Non-tuning experts：frozen，仅参与 forward pass
    15. 本地 fine-tuning iteration × 1，mini-batch=16，lr=1e-5

    === 阶段 4：Upload & Aggregate ===
    16. Upload：仅上传 tuning experts E_i^{exp} 的更新参数（ΔW）
    17. Aggregate：parameter server 用 FedAvg 聚合各 participant 的 expert updates
    18. 并发执行：量化 + profiling 下一轮模型 w^{r+1}（stale profiling）

    === 张量计算示意：MoE Layer Forward ===
    对于 layer l，输入 x ∈ R^{seq_len × d_model}：
      - Gate: g = softmax(W_g · x)  → 选 top-k expert indices
      - 若 expert e 是 tuning expert：
          y_e = W_e^{full} · x  （FP32 GEMM, 需 backprop）
      - 若 expert e 是 merged non-tuning expert：
          y_e = W_e^{merged} · x  （FP32 GEMM, frozen, 无 grad）
      - 输出: y = Σ_e g_e · y_e  （weighted sum over top-k experts）
    ```

## OpenJarvis

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：**LLM-guided spec search**（算法1）——一种本地-云端协作的 spec 优化算法。核心思路是将个人AI系统的五个原语（Intelligence/Engine/Agents/Tools & Memory/Learning）暴露为类型化 spec 中的独立可编辑字段，云端前沿模型在搜索时诊断失败模式并提出跨原语协调编辑，由 held-out gate 仅接受非退化编辑，推理时 spec 完全在本地执行。算法流程：
    1. **Diagnose**（诊断）：教师 LM 读取 trace corpus（benchmark traces、合成 traces 或用户批准的脱敏 traces），按失败模式分组为 *failure clusters*，每个 cluster 标注 student vs teacher 成功率及技能差距的自然语言描述（如"student 在需要日历查找的多跳问题上失败，因为它不调用 calendar tool"）。
    2. **Propose**（提议编辑）：教师提出跨四个可编辑原语的协调编辑——Intelligence edits（模型选择、量化格式、LoRA/GRPO 训练触发）、Engine edits（backend 切换、batch size、KV-cache）、Agent edits（prompt 重写、few-shot exemplars、agent type 切换、turn limits）、Tools & Memory edits（添加/移除工具、修改 tool description、切换 memory backend、cloud-as-tool routing）。
    3. **Execute**（执行门控验证）：候选编辑在 held-out gate 上评估——gate 由合成标注 traces、大规模 agentic datasets（GeneralThoughtArchive, ToolScale）和标准 benchmark splits（MMLU-Pro, GAIA, τ-bench）组成。编辑仅当目标 failure cluster 改进且非目标 cluster 退化 ≤ ε（默认 1%）时接受。
    4. **Repeat**（迭代）：接受编辑 → 更新 spec → 重复 diagnose→propose→gate，直到 gate score 在 k 个 session（默认 k=5）内停滞或预算耗尽。
  - Intelligence 编辑触发训练时，使用复合 reward：R(q,y) = α·R_acc − β·Ê − γ·L̂ − δ·Ĉ，默认权重 (0.5, 0.1, 0.1, 0.3)，在 GRPO 训练中平衡精度、能耗、延迟和成本。
  - 实验比较：
    - **4 个学生模型 × 3 个教师模型 × 8 benchmarks**（Table 9）：Nemotron-Nano-4B, Gemma4-E4B, Qwen3.5-4B, Qwen3.5-9B 作为学生；Claude Opus 4.6, GPT 5.4, Gemini 3.1 Pro 作为教师。平均收益：+31.5 pp (Nemotron-Nano-4B), +13.1 pp (Gemma4-E4B), +22.9 pp (Qwen3.5-4B), +14.9 pp (Qwen3.5-9B)。
    - **单原语 baseline 对比**（Figure 7）：prompt-only (DSPy/SIMBA, GEPA) 仅 +4.1–5.2 pp；weight-only (SFT, LoRA) 是最强单原语 baseline；LLM-guided spec search 比 LoRA 高 1.1–8.8 pp，比 GEPA 高 5.0–18.8 pp，且优化成本低 7.1–10.9×。
    - **Proposer 消融**（Figure 8）：固定四原语 move space，LLM proposer 比 evolutionary spec search 高 +10.0 pp（均值），比 template-random proposer 高 +14.0 pp。
    - **Move-space 消融**（Figure 8）：固定 LLM proposer，从 1-of-4 扩展到 4-of-4 可编辑原语增加 +5.5–16.5 pp 精度和 2.65–3.45× 延迟加速。合并原语对（如 Agent+Tool 合并）回退 2.8–9.7 pp 精度。
    - **编辑类型分布**（Table 10, Figure 10-11）：Intelligence 编辑占 16–44%（code 任务主导），Agent 编辑占 24–45%（agentic/customer-service 任务主导），Tool 编辑占 14–47%（tool-calling/research 任务主导），Engine 编辑占 12–14%（效率优化）。按 failure cluster 类别：retrieval failure→Tool edits (65%), reasoning failure→Intelligence edits (52%), control-flow failure→Agent edits (51%), efficiency-bounded→Engine edits (58%)。
    - **鲁棒性**（Table 11-13）：reward weights 扰动（4.6 pp 带）、search seed 方差（均值 1.4 pp std）、random restarts（Best-of-5 仅 +1.2 pp over default），均远小于 13–32 pp 的搜索收益。

- 硬件平台是什么，配置是什么。
  - 同 Serving 调度条目：7 个硬件平台——Mac Mini M4, Radeon RX 9070 XT, Arc Pro B70, Ryzen AI Max+ 395, Mac Studio M4 Max, RTX 6000 Pro, DGX Spark（详见 Table 6）。
  - 搜索时使用云端 API（OpenAI/Anthropic/Google）作为教师模型；学生训练（LoRA/GRPO）在本地硬件执行；推理在本地硬件执行。

- 模型是什么。数据集和bench分别是什么。
  - 模型：
    - **本地模型（学生）**：Qwen3.5 家族（4B/9B/27B/35B-MoE/122B-MoE）、Nemotron 家族（Nano-4B Hybrid/Super-120B Hybrid MoE）、Gemma4 家族（E4B-Dense-PLE/26B-MoE）、Granite 家族（3.3-8B/4.0 H-Small Hybrid MoE）——共 11 个本地模型、4 个家族。
    - **云端模型（教师/baseline）**：Claude Opus 4.6（Anthropic）、GPT 5.4（OpenAI）、Gemini 3.1 Pro（Google）。
    - **LLM judge**：GPT-5-mini（论文注明可能存在 judge bias）。
  - 数据集与 Benchmark（8 个，508 任务，Table 3）：
    - ToolCall-15（TC15）：15 个工具调用场景，自动评分
    - PinchBench（PB）：23 个端到端 agent 任务，自动+LLM judge
    - LiveCodeBench v6（LCB）：100 题 competitive programming，自动测试用例执行
    - τ-Bench V2（TauB）：100 个多轮客服对话，数据库状态比对
    - τ²-Bench Telecom（TBTel）：40 个双控电信客服任务，数据库状态比对
    - GAIA：50 个通用 AI 助手问题，exact match + 2h timeout/task
    - LiveResearchBench v4（LRB）：100 个 deep research 任务，DeepEval checklist 评分
    - DeepResearchBench（DRB）：80 个 PhD 级研究任务（22 领域），RACE + FACT 评分
  - 训练数据来源（Learning 原语）：GeneralThoughtArchive（431K reasoning traces）、ToolScale（大规模 agentic dataset）、合成 teacher-generated traces、用户批准的脱敏交互 traces。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源状态：论文声明开源（https://github.com/openjarvis/openjarvis），链接当前 404。网站 https://open-jarvis.github.io/OpenJarvis/ 论文未明确说明可访问性。
  - 算法伪代码（LLM-guided spec search，算法1）：
    ```
    Require: Spec S_0, teacher T, gate G, regression tolerance ε, budget B
    1: S ← S_0
    2: while not converged and cost < B do
    3:   C ← T.diagnose(traces(S))
    4:   e ← T.propose(S, C)         ▷ 可同时编辑 Intelligence, Engine, Agents, Tools & Memory
    5:   S' ← apply(S, e)
    6:   if GateOK(S', S, C, ε) then  ▷ G_c(S') > G_c(S) 且 ∀c'≠c: G_c'(S') ≥ G_c'(S)-ε
    7:     S ← S'                     ▷ greedy accept
    8:   end if
    9: end while
    10: return S
    ```
  - 张量计算层面（以 Intelligence edit 触发 GRPO 训练为例）：
    - 学生模型（如 Qwen3.5-9B）在 teacher-generated SFT pairs 或 GRPO 协议下进行 LoRA 微调
    - 复合 reward 函数：R(q,y) = 0.5·R_acc(q,y) − 0.1·Ê(q,y) − 0.1·L̂(q,y) − 0.3·Ĉ(q,y)，其中 Ê/L̂/Ĉ 为 z-score 归一化后的能耗/延迟/成本偏差
    - LoRA adapter 更新仅修改 Intelligence weights，不影响 Agent prompts/Tool descriptions/Engine settings
    - Gate 评估整个 spec（包含更新后的权重 + 其余原语配置）的端到端精度，而非仅评估模型困惑度
  - 关键设计要点：
    - **Failure cluster 驱动的编辑分配**：教师将 failure mode 映射到对应原语——retrieval failure → Tool edit (65%)、reasoning failure → Intelligence edit (52%)、control-flow failure → Agent edit (51%)、efficiency-bounded → Engine edit (58%)
    - **跨原语协调编辑**：单次 proposal 可同时改写 tool description + 调整 prompt + 切换 Engine backend + 更改模型量化格式，实现单原语优化器无法表达的联合优化
    - **非退化门控**：GateOK 保证目标 cluster 改进且非目标 cluster 退化 ≤ 1%，防止过拟合
    - **搜索-推理分离**：云端教师仅在搜索时使用（median \$15.6/benchmark），推理时 spec 零云端调用。100 queries/day 部署 6 个月后摊销教师成本 <\$0.001/query
