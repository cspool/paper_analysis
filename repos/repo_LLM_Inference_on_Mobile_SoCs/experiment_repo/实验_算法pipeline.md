## Agent-X: Full Pipeline Acceleration of On-device AI Agents

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Agent-X 包含两个核心组件——PromptWeaver 和 ExSpec。PromptWeaver 通过动态重组输入 prompt 来最大化 prefix caching 的 KV cache 复用，减少 prefill 阶段的在线计算量。具体做法：(1) 将早期动态的 tool descriptions/guidelines 替换为包含所有工具的静态全量描述；(2) 基于 tool co-activation locality 使用 NMF 进行工具聚类；(3) 按 theme 固定 cluster 排序以最大化 prefix cache 命中；(4) 使用贪心算法选择 cluster combination 的 KV cache 预计算并存入 SSD；(5) 追加单工具 example 和 top-K (K=1) 动态 example 以保证准确率。ExSpec 使用基于输入 prompt 中 few-shot examples 和 user query 构建的 n-gram (trigram, n=3) lookup table (LUT) 作为 speculative decoding 的 draft model，避免使用额外的 draft LLM 带来的内存和延迟开销，并通过 selective decoding 在 LUT 无法命中时回退到自回归生成以避免 multi-token tax。
  - 实验比较：(1) PromptWeaver vs. baseline（无 prefix caching）和 Static（仅缓存静态 token）的 prefill 延迟和 speedup；(2) ExSpec vs. baseline 和基于 Llama-3.2-1B-Instruct 的 speculative decoding (SpecDec) 的 decode 延迟和 speedup；(3) PromptWeaver + ExSpec (Agent-X) 端到端延迟和 speedup；(4) PromptWeaver 在不同 K（动态 tool-use example 数量）下的 Planner 准确率变化；(5) KV cache budget 对 tool-use example coverage 和存储开销的影响；(6) n-gram 的 n 值（bigram/trigram/quadgram）对 draft token accuracy 和 decode 延迟的影响；(7) LUT 提取区域（全输入 vs. few-shot + user query）对 decode speedup 的影响。

- 硬件平台是什么，配置是什么。
  - Apple Mac mini，M4 Pro 芯片，64 GB 内存，512 GB SSD 存储，12 CPU 核心，16 GPU 核心
  - macOS Sequoia 15.5
  - 对比参考硬件（Table 1）：NVIDIA H100/H200/B200、AMD MI325X、Google TPU v6e（服务端）；Apple M4 Max、Qualcomm Snapdragon X Elite、AMD Ryzen AI+ PRO 395（端侧）

- 模型是什么。数据集和bench分别是什么。
  - 模型：TinyAgent-7B（backend LLM），基于 WizardLM-2-7B 针对 agentic 任务微调；额外评估 TinyAgent-1.1B
  - 数据集：TinyAgent fine-tuning dataset [68]，训练集用于 PromptWeaver 的工具聚类和 combination selection，测试集（1,022 个样例，最多涉及 16 种工具）用于评估
  - Benchmark 指标：端到端任务延迟、prefill 阶段延迟和 speedup、decode 阶段延迟和 speedup、Planner 准确率（通过比较生成 DAG 与 ground truth DAG）、draft token accuracy、tool-use example coverage、KV cache 存储开销

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：TinyAgent 框架开源 (https://github.com/SqueezeAILab/TinyAgent)，TinyAgent-7B 模型开源 (https://huggingface.co/squeeze-ai-lab/TinyAgent-7B)，TinyAgent 数据集开源 (https://huggingface.co/datasets/squeeze-ai-lab/TinyAgent-dataset)，ToolRAG 开源 (https://huggingface.co/squeeze-ai-lab/TinyAgent-ToolRAG)，MLX-LM 开源 (https://github.com/ml-explore/mlx-lm)，MLX-engine 开源 (https://github.com/lmstudio-ai/mlx-engine)。Agent-X 论文未明确说明自身代码是否已开源（通过 URL 搜索确认）。
  
  - 算法 Pipeline 执行例子（PromptWeaver + ExSpec 全流程）：
    
    **阶段 1：离线预处理 — PromptWeaver KV Cache 构建**
    ```
    输入: 训练数据集 D, KV cache budget N, 可用工具集合 T = {t1, t2, ..., t16}
    输出: 预计算的 KV cache 集合 C 存入 SSD
    
    1. 构建 tool co-activation matrix M (|T|×|T|):
       对 D 中每个样本的 ground truth label:
         M[i][j] += 1  若 ti 和 tj 在同一任务中被调用
    
    2. 对 M 应用 NMF 聚类，产生 C1..C8 共 8 个 cluster
       每个 cluster 包含 2-6 个工具
    
    3. Theme-based cluster ordering:
       对每个 cluster，确定主导 theme (email/contacts/maps/notes/...)
       同 theme 的 cluster 相邻排列，固定全局顺序 seq = [C_email_A, C_email_B, C_contacts_A, ...]
    
    4. Cluster combination selection (Algorithm 1):
       C = ∅
       对 D 中所有样本的激活 cluster 序列，提取所有前缀
       for i = 1 to N:
         对每个候选前缀 p (len(p)==1 或 p[:-1] ∈ C):
           计算 coverage(D, C ∪ {p}) - coverage(D, C)
         选择增量最大的 p 加入 C
    
    5. 对 C 中每个 cluster combination:
       用 MLX-LM 预计算对应 prompt 的 KV cache → 存入 SSD
       (N=15 时约 5.87 GB)
    ```
    
    **阶段 2：在线推理 — PromptWeaver 动态 Prompt 构造**
    ```
    输入: 用户查询 query
    输出: 重组的 Planner prompt
    
    1. 构造静态前缀:
       prompt_prefix = "[System Prompt] + [所有 16 个工具的 descriptions + guidelines]"
       → 从 SSD 加载该静态前缀的 KV cache (预计算)
    
    2. ToolRAG 工具检索:
       classifier(query) → P(tool_i | query) for i=1..T
       activated_tools = {ti | P(tool_i|query) > τ}
    
    3. Cluster 激活:
       activated_clusters = {Cj | Cj ∩ activated_tools ≠ ∅}
       按固定 theme 顺序排列 activated_clusters
       → 从 SSD 加载最长匹配前缀的 KV cache
    
    4. 追加单工具 examples + Top-1 动态 example:
       for each tool in activated_tools:
         prompt += single_tool_example[tool]
       prompt += ToolRAG_top1_example(query)
       → 这些 ~519 tokens 需在线 prefill 计算
    
    5. 拼接 Planner prompt → 送入 LLM prefill + decode
    ```
    
    **阶段 3：在线推理 — ExSpec 解码加速**
    ```
    输入: 当前 prompt (已 prefill 完成), 目标模型 M_target (TinyAgent-7B)
    输出: 生成的 token 序列
    
    1. 构建 n-gram LUT (trigram, n=3):
       extraction_stream = concat(few_shot_examples, user_query)
       for i from 0 to len(extraction_stream) - 3:
         key = (token[i], token[i+1])     # 2-token context
         value_counts[key][token[i+2]] += 1
       LUT[key] = argmax(value_counts[key])   # 最频繁的后继 token
    
    2. Speculative decoding loop (每次迭代):
       # Step A: 检查 LUT 是否有当前上下文
       current_context = (token[-2], token[-1])
       if current_context not in LUT:
         # Selective fallback: 直接自回归生成 1 token, continue
         next_token = M_target.autoregressive(current_context)
         output.append(next_token)
         continue
       
       # Step B: 生成 draft tokens (N=4)
       drafts = []
       ctx = current_context
       for j = 1 to 4:
         draft_token = LUT[ctx]
         drafts.append(draft_token)
         ctx = (ctx[1], draft_token)  # 滑动窗口
       
       # Step C: 目标模型验证
       # 输入: [last_real_token, draft_1, draft_2, draft_3, draft_4]
       M_target_logits = M_target.forward([last_real_token] + drafts)
       
       # Step D: Token acceptance
       for j = 1 to 4:
         if M_target_logits[j].argmax() == drafts[j-1]:
           output.append(drafts[j-1])  # accept
         else:
           output.append(M_target_logits[j].argmax())  # first mismatch
           break  # 丢弃后续所有 draft tokens
     
     注: MLX-LM 中单 token 自回归推理 131ms/token，2-token 并行验证 244ms
         (multi-token tax = 1.86× slowdown per token group)
         Selective decoding 使 Planner 平均 17 次/query、Arbiter 37 次/query 回退到自回归
    ```
    
    **端到端流程示例**（查询 "Schedule a meeting with John tomorrow at 5pm"）：
    ```
    User Query: "Schedule a meeting with John tomorrow at 5pm"
    ↓
    [ToolRAG: 检索工具集 → {get_email_address, create_calendar_event}]
    ↓
    [PromptWeaver: 重组 prompt]
      ├── 静态全量 tool descriptions: 从 SSD 加载预计算 KV cache (命中)
      ├── Clustered examples (激活 cluster): 从 SSD 加载 KV cache (命中)
      └── 单工具 + Top-1 动态 example: 在线 prefill ~519 tokens
    ↓
    [Planner LLM Prefill: 仅计算 ~519 uncacheable tokens]
      加速比: 1.57× (vs. baseline 1,711 uncacheable tokens)
    ↓
    [Planner LLM Decode: ExSpec trigram LUT]
      ├── 从 few-shot examples + user query 构建 LUT
      ├── draft token accuracy: 0.25 (selective mode)
      └── 加速比: 1.73× (vs. baseline autoregressive)
    ↓
    Planner 输出: [{tool: "get_email_address", args: {name: "John"}}, 
                   {tool: "create_calendar_event", args: {..., depends: "$1"}}]
    ↓
    [Execution Unit: 串行/并行执行工具调用]
    ↓
    [Arbiter LLM: PromptWeaver (prefix caching, 4.35× prefill speedup) 
                 + ExSpec (1.73× decode speedup)]
    ↓
    Arbiter 输出: "task_complete" → 结束
    ↓
    端到端加速比: 1.61× (PromptWeaver + ExSpec 联合)
    总延迟: baseline 35.4s → Agent-X ~22.0s
    ```

## Fast On-device LLM Inference with NPUs

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：llm.npu 提出 **shadow outlier execution（影子异常值执行）**，一种 NPU 友好的 per-tensor W8A8 量化推理方法。核心思想是将激活张量中的异常值通道（outlier channels）在运行时提取为紧凑子张量，在 CPU/GPU 上以 FP16 精度并行执行，然后将结果合并回 NPU 上 per-tensor INT8 MatMul 的输出。公式为：`(x/s) ⊙ w = clamp(x/s, -127, 128) ⊙ w (NPU INT8) + extract(⌊(x/s)/128⌋×128) ⊙ w (CPU FP16)`。关键优化包括：(1) **热通道权重驻留**——基于异常值高度集中于少数通道（<3% 通道贡献 >80% 异常值），仅将"热通道"的 MatMul 权重副本保留在 CPU 内存中，其余从磁盘按需加载（可与 NPU 执行重叠），减少 34.3% 内存开销；(2) **逐层异常值剪枝**——离线用大规模语料分析每层异常值重要性（最大异常值与量化尺度的比值），剪枝 85% 最不重要层的异常值，消除 CPU-NPU 同步开销。该算法与任何 per-tensor 量化方法兼容，当前原型基于 max-min 对称量化。
  - 实验比较：(1) llm.npu 量化精度 vs. FP16、SmoothQuant（per-tensor）、K-Quant（per-group，llama.cpp 使用）、LLM.Int8()（浮点异常值处理），在 LAMBADA、HellaSwag、WinoGrande、OpenBookQA、MMLU 五个 benchmark 上评估，llm.npu 平均精度损失仅 1%（vs FP16），优于 SmoothQuant 最高 32.9%、优于 K-Quant 最高 70.9%；(2) 不同异常值剪枝率（0%–100%）下的 prefill 速度与生成精度权衡，在 HellaSwag 和 LAMBADA 上评估 Qwen1.5-1.8B 和 Gemma-2B；(3) 消融实验中 shadow outlier execution 对 prefill 延迟的贡献（3.91–8.68× 加速）。

- 硬件平台是什么，配置是什么。
  - 主测试平台 1：Redmi K70 Pro（Qualcomm Snapdragon 8 Gen 3 SoC，Hexagon NPU，24GB 内存，Adreno 750 GPU），Android 13
  - 主测试平台 2：Redmi K60 Pro（Qualcomm Snapdragon 8 Gen 2 SoC，Hexagon NPU，16GB 内存），Android 13
  - NPU 规格：Hexagon NPU 支持 1024-bit INT8 SIMD 向量运算，最高 73 TOPS（INT8），时钟频率 500–750 MHz
  - 架构：统一内存（CPU/GPU/NPU 共享物理内存），无需数据拷贝

- 模型是什么。数据集和bench分别是什么。
  - 模型：Qwen1.5-1.8B、Gemma-2B、Phi2-2.7B、LLaMA-2-Chat-7B、Mistral-7B（均为 decoder-only Transformer 架构，Hugging Face 格式导出）
  - 量化精度 Benchmark：LAMBADA、HellaSwag、WinoGrande、OpenBookQA、MMLU
  - Prefill 性能数据集：LongBench（2wikimqa、TriviaQA，模拟 context-aware 邮件回复，prompt 1451–1787 tokens）、DroidTask（模拟 UI 自动化，prompt 505–827 tokens）、Persona-Chat（模拟聊天摘要，prompt 488–584 tokens）
  - 指标：prefill speed（tokens/sec）、prefill 能耗（通过 /sys/class/power_supply 每 100ms 采样）、prefill 内存消耗、端到端延迟（prefill + decode）

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：代码开源在 https://github.com/UbiquitousLearning/mllm，MIT 许可证，约 10K 行 C/C++ 和汇编代码。Artifact 存档在 https://doi.org/10.5281/zenodo.14392760。构建于 MLLM 和 QNN (Qualcomm Neural Processing SDK) 之上。
  
  - **Shadow Outlier Execution 算法 Pipeline 执行例子（以 Qwen1.5-1.8B 单层 FFN 为例）**：

    ```
    # ========== 阶段 0：离线准备 ==========
    # 输入: FP16 模型权重, 校准语料 (wikitext), 目标剪枝率 85%
    # 输出: INT8 量化权重, 热通道权重副本, 逐层异常值阈值 s_l

    1. Per-tensor 对称量化 (max-min):
       对每层 MatMul 权重 W_FP16 [M, N]:
         s_w = max(|W|) / 127
         W_INT8 = round(W_FP16 / s_w), clip to [-127, 127]
       激活量化尺度 s_l 通过校准语料离线确定。

    2. 异常值重要性分析:
       用校准语料前向推理，对每层 l 的每个激活通道 c:
         统计最大异常值 outlier_max[c] 和量化尺度 s_l
         importance_l = max_c(outlier_max[c] / s_l)  # 比值越大→分布越分散→越重要
       按 importance_l 对所有层排序，标记 top 85% 低重要性层待剪枝。

    3. 热通道识别:
       统计校准语料推理中每个通道出现异常值的频率。
       保留频率最高的 "热通道"（<3% 通道数覆盖 >80% 异常值出现），
       将其对应的权重列 W[:, hot_channels] 复制到 CPU 内存空间作为 FP16 副本。
       其余通道权重按需从磁盘加载。

    # ========== 阶段 1：在线推理 — 单次 Shadow MatMul ==========
    # 输入: FP16 激活 x [seq_len, hidden_dim]
    #       INT8 权重 W_INT8 [hidden_dim, out_dim]
    #       量化尺度 s
    # 输出: FP16 结果 y [seq_len, out_dim]

    # Step A: NPU 主路径 (INT8 per-tensor MatMul)
    x_quant = round(clamp(x / s, -127, 128))         # 量化 + 截断到 INT8 范围
    y_npu_int = MatMul_INT8(x_quant, W_INT8)          # NPU 上 INT8 MatMul, ~2ms
    y_npu = y_npu_int * s                             # 反量化到 FP16

    # Step B: 判断该层是否需要影子执行
    if layer_index in pruned_layers:                  # 85% 层被剪枝
        return y_npu                                  # 直接返回 NPU 结果

    # Step C: 识别 + 提取异常值通道
    outlier_mask = (abs(x / s) > 128)                 # 布尔掩码, [seq_len, hidden_dim]
    # 典型稀疏度: 仅 5–15 个通道有异常值, ~0.1–0.3% 总通道数
    outlier_ch = unique(outlier_mask.nonzero()[:, 1]) # 异常值通道索引, ~10 个
    x_outlier = extract((x / s) // 128 * 128, outlier_ch)  # 紧凑提取超出范围部分
    # x_outlier shape: [seq_len, ~10]

    # Step D: 获取影子权重 (热通道优先)
    hot_ch = outlier_ch ∩ hot_channels_set            # 在 CPU 内存中的热通道
    cold_ch = outlier_ch - hot_ch                     # 需从磁盘加载
    W_cpu = W_FP16[:, hot_ch]                         # 从 CPU 内存直接获取
    for ch in cold_ch:
        W_cpu.append(load_weight_from_disk(ch))       # 磁盘 I/O 可与 NPU MatMul 重叠
    # W_cpu shape: [~10, out_dim]

    # Step E: CPU/GPU 影子 MatMul (FP16)
    y_shadow = MatMul_FP16(x_outlier, W_cpu)          # [seq_len, ~10] × [~10, out_dim], ≪1ms

    # Step F: 合并 (FP16 addition)
    y = y_npu + y_shadow                              # ~μs 级向量加法
    return y

    # ========== 阶段 2：全模型 Prefill Flow ==========
    for each decoder layer l = 0 to L-1:
        # Q/K/V 投影 — shadow_matmul (NPU INT8 + CPU FP16 补偿)
        x_q = shadow_matmul(layer_input, W_Q, s_Q)
        x_k = shadow_matmul(layer_input, W_K, s_K) 
        x_v = shadow_matmul(layer_input, W_V, s_V)

        # RoPE + Attention Score + Softmax — CPU/GPU FP16
        x_q, x_k = RoPE_FP16(x_q, x_k)
        attn = Softmax_FP16(x_q @ x_k^T / sqrt(d_k))
        attn_out = attn @ x_v                         # 论文中可能部分在此用 shadow

        # O 投影 — shadow_matmul
        attn_proj = shadow_matmul(attn_out, W_O, s_O)
        x = LayerNorm_FP16(layer_input + attn_proj)   # CPU FP16

        # FFN Up/Gate — shadow_matmul
        gate = SiLU(shadow_matmul(x, W_gate, s_gate)) # SiLU 在 CPU
        up = shadow_matmul(x, W_up, s_up)

        # FFN Down — shadow_matmul
        ffn_out = shadow_matmul(gate * up, W_down, s_down)
        x = LayerNorm_FP16(x + ffn_out)               # CPU FP16
    ```

  - **关键数值（Qwen1.5-1.8B, prompt_len=1024, Redmi K70 Pro）**：
    - NPU INT8 per-tensor MatMul: ~2.0ms（[64,2048]×[2048,11008] 规模）
    - 异常值通道数: 5–15 通道（0.1%–0.3% 总通道）
    - CPU shadow FP16 MatMul: ≪0.1ms（可被 NPU 完全隐藏）
    - 0% 剪枝: 最高精度 (~1% loss), prefill ~156 tok/s
    - 85% 剪枝 (默认): <1% loss vs FP16, prefill ~544 tok/s
    - 100% 剪枝: prefill ~596 tok/s, 精度降至 8.1%
    - 热通道优化: 减少 34.3% shadow 内存开销
    - 相比 CPU baselines: 7.3–18.4× prefill 加速, 1.9–59.5× 能耗降低

## LLM as a System Service on Mobile Devices

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Tolerance-Aware Compression（容忍度感知压缩），一种基于 attention score 的 chunk-wise KV cache 差异化压缩方法。核心思想是利用不同 token/chunk 对 LLM 推理的贡献不均等——某些 chunk（如高频信息词）比另一些（如填充词）更重要，因此可以承受更激进的压缩。具体实现：(1) 从 attention score matrix 计算每个 token 的"信息密度"——被其他 token 关注越多的 token 信息量越大；(2) 将 token-level 信息密度聚合到 chunk level（跨 head、layer、token）；(3) 在全局平均压缩比 constraint 下，通过优化问题最大化整体 context 信息强度，为每个 chunk 分配不同压缩级别（INT8/INT4/INT2）。该方法在现有 KV cache 量化方法（如 LMDeploy INT8）之上叠加，通过 channel-wise 线性量化实现更低位宽。
  - 实验比较：(1) LLMS tolerance-aware compression vs. 静态均匀量化（LMDeploy INT8/INT4）的 perplexity-compression ratio trade-off，在 WikiText-2 数据集上使用 Llama2-7B 评估；(2) 各数据集（Table 3 中 6 个数据集）下的平均 accuracy loss vs. compression ratio 对比。

- 硬件平台是什么，配置是什么。
  - 实验使用与主 evaluation 相同的硬件平台（Jetson Orin NX/TX2/MI14），但 tolerance-aware compression 的微实验在 MI14 上进行。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama2-7B（主要）、OPT-6.7B
  - 压缩精度评估数据集：WikiText-2（language modeling perplexity）
  - 端到端精度验证：AGnews、Xsum、Samsum、CNN/DailyMail、WMT17-de-en、SST-2（共 6 个数据集，覆盖分类、摘要、翻译、理解任务）
  - 指标：Perplexity（压缩精度）、Compression Ratio（压缩率）、Accuracy（端到端任务准确率）

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文未明确说明 LLMS 自身代码是否已开源。所用 baseline 量化方法 LMDeploy (https://github.com/InternLM/lmdeploy) 和 SmoothQuant (https://github.com/mit-han-lab/smoothquant) 均为开源。

  - **Tolerance-Aware Compression 算法 Pipeline 执行例子（以 Llama2-7B, 4K context, ratio_global=50%, {ratio_w}={8/8, 4/8, 2/8}）**：

    ```
    # ========== 输入 ==========
    # KV cache: 256 chunks, 每 chunk=16 tokens × 32 layers × 32 heads × 128 dim
    #   shape per chunk: [16, 32, 32, 128] for K, same for V
    #   原始精度: FP16 → 先由 baseline 方法量化为 INT8
    # Attention scores: 已在前向推理中计算

    # ========== Step 1: 计算 Attention Score Matrix ==========
    # 对每层 l (0..31)、每个 head h (0..31):
    #   Q_lh = X * W_Q_lh  # [seq_len, head_dim]
    #   K_lh = X * W_K_lh  # [seq_len, head_dim]
    #   A_lh = softmax(mask(Q_lh @ K_lh^T / sqrt(d_k)))  # [seq_len, seq_len], 下三角矩阵
    #
    # A_lh[row, col] 含义: token_row 对 token_col 的注意力权重
    #   行归一化: 每行 sum = 1.0

    # ========== Step 2: 计算 Token-level 信息密度 ==========
    # 对每个 token position col:
    #   token_density[col] = mean over l, h of:
    #     mean over all rows > col of A_lh[row, col]
    #
    # 伪代码:
    for col in 0..seq_len-1:
        density_sum = 0
        for l in 0..L-1:
            for h in 0..H-1:
                col_sum = 0
                for row in col+1..seq_len-1:  # 只看后续 token 对该 token 的关注
                    col_sum += A_lh[row, col]
                density_sum += col_sum / (seq_len - col - 1)
        token_density[col] = density_sum / (L * H)

    # ========== Step 3: 聚合到 Chunk-level 信息密度 (Equation 1) ==========
    # chunk_size = 16
    for chunk_i in 0..255:
        p = chunk_i * 16
        q = (chunk_i + 1) * 16 - 1
        D_i = mean(token_density[p:q+1])

    # 示例结果（归一化后）:
    #   chunk  0 (prompt 开头 "You are a helpful..."):  D ≈ 0.85 高密度
    #   chunk 100 ("and so on the other hand..."):       D ≈ 0.42 低密度
    #   chunk 200 ("the summary of the document is..."): D ≈ 0.73 中密度

    # ========== Step 4: 排序并分配压缩级别 (Equation 2 & 3) ==========
    # Rank_i = percentile(D_i)  # 0%~100%

    # 优化问题 (Equation 3):
    #   maximize: ctxInfo = Σ_w (1/ratio_w) * Σ_{chunk in [σ_{w+1}, σ_w]} D_i
    #   s.t.:     Σ_w ratio_w * (σ_w - σ_{w+1}) = ratio_global
    #
    #   ratio_w ∈ {1.0, 0.5, 0.25} 对应 {8/8, 4/8, 2/8}
    #   ratio_global = 0.50

    # 求解（三种压缩级别时简化为确定两个分割点）:
    #   σ_{4/8} (分割 INT8 和 INT4) = ?
    #   σ_{2/8} (分割 INT4 和 INT2) = ?
    #   满足: 1.0*σ_{4/8} + 0.5*(σ_{4/8}-σ_{2/8}) + 0.25*(1-σ_{4/8}) = 0.50
    #
    #   解得: σ_{4/8} ≈ 0.30, σ_{2/8} ≈ 0.70
    #   → top 30% chunk 保持 INT8, middle 40% 压到 INT4, bottom 30% 压到 INT2

    # ========== Step 5: 执行压缩 ==========
    # 对每个 chunk_i:
    compress_chunk(chunk_i):
        rank = Rank_i
        if rank > 0.70:       # top 30%
            # 保持 LMDeploy INT8 量化结果
            chunk_compressed = chunk_int8
        elif rank > 0.30:     # middle 40%
            # 在 INT8 基础上再量化到 INT4 (channel-wise linear)
            for channel in chunk:
                s_ch = max(abs(channel)) / 7   # INT4 范围 [-7, 7]
                chunk_compressed[channel] = round(chunk_int8[channel] / s_ch)
                chunk_compressed = clip(chunk_compressed, -7, 7)
            # bit-pack: 2 个 INT4 → 1 个 INT8
        else:                  # bottom 30%
            # 在 INT8 基础上再量化到 INT2 (channel-wise linear)
            for channel in chunk:
                s_ch = max(abs(channel)) / 1   # INT2 范围 [-1, 1]
                chunk_compressed[channel] = round(chunk_int8[channel] / s_ch)
                chunk_compressed = clip(chunk_compressed, -1, 1)
            # bit-pack: 4 个 INT2 → 1 个 INT8

    # ========== Step 6: 数据打包（适配推理框架） ==========
    # LLM 推理框架仅支持 INT8，sub-byte 数据需打包:
    #   INT4 packing: parallel bit-shift on GPU/NPU
    #     packed[i] = (chunk_4bit[2i] & 0x0F) | ((chunk_4bit[2i+1] & 0x0F) << 4)
    #   INT2 packing: 
    #     packed[i] = (c[4i]&0x03) | ((c[4i+1]&0x03)<<2) | ((c[4i+2]&0x03)<<4) | ((c[4i+3]&0x03)<<6)

    # 解压时: unpack 回 INT8 → 乘以各 channel scale → 用于 attention 计算
    ```

  - **关键数值**：
    - ratio_global=50% 时，LLMS 方法 perplexity ≈ 与 INT8 baseline 持平，memory ≈ INT4 baseline
    - 4-bit 全量静态量化: accuracy loss up to 59%（极端 outlier 导致）
    - 2-bit 全量静态量化: accuracy loss up to 99%（不可用）
    - LLMS 三级别混合压缩: accuracy loss negligible，compression ratio 2× vs 静态 INT8

  - 注：该论文主要贡献在 Serving 调度层（LLMS 系统设计），算法 pipeline 层的 tolerance-aware compression 是其中的一个关键技术组件。若需完整的算法 pipeline 分析，请同时参考 Serving 调度层条目。

## Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：论文提出两个算法pipeline层面的关键技术：(1) **Hardware-aware Fine-grained Tile Quantization Scheme**——将权重在量化前按照 HMX tile layout（32×32 tile, 每两行permute）重新排列，然后在新的内存顺序上进行 group-wise 量化（group_size=32，即 2×16 tile 单位），使量化后的权重布局与 NPU 矩阵单元的内存访问模式对齐。量化后将 8 个 group coalesce 为 super-group，使 INT4 量化值填满 128-byte HVX 向量寄存器。(2) **LUT-Based Efficient Dequantization**——使用 HVX 的 vlut16 指令，将 4-bit 量化值直接通过查表转换为 FP16 值（16 元素表），避免传统的 mask-unpack-convert 指令序列。同时使用 vlut16 实现 4 组 scales 的广播，替代标量广播+寄存器拼接。支持 Q4_0 对称量化（4.5 BPW），对 FFN down 矩阵使用 Q8_0（8.5 BPW）以减少量化误差。(3) **LUT-Based Fast Softmax**——在 FlashAttention 中，使用 HVX 的 vgather 指令实现预计算 LUT 的指数函数（exp），LUT 占 64 KiB TCM。利用 safe softmax 确保 exp 输入非正，仅存储 32768 个 FP16 条目。输入 FP16 值忽略符号位并左移一位作为 vgather 的字节偏移。
  - 实验比较：(1) Tile quantization vs. conventional group quantization 的精度对比（WinoGrande/MMLU/Wiki PPL），tile group 无显著精度损失；(2) LUT-based FP16 FlashAttention vs. F32 Attention 的精度对比，无明显差异；(3) QNN per-channel W4A16 vs. AutoAWQ per-group W4A16 的精度对比（MATH500/GSM8K/Wiki PPL），QNN 量化严重损害推理能力；(4) Best-of-N 和 Beam Search test-time scaling 的 accuracy-latency Pareto frontier，小型模型+scaling 可匹敌甚至超越大模型；(5) 不同 batch size 下 decoding throughput 的扩展特性。

- 硬件平台是什么，配置是什么。
  - OnePlus Ace3：Qualcomm Snapdragon 8 Gen 2 SoC，Hexagon NPU V73 架构
  - OnePlus 12：Qualcomm Snapdragon 8 Gen 3 SoC，Hexagon NPU V75 架构
  - OnePlus Ace5 Pro：Qualcomm Snapdragon 8 Elite SoC，Hexagon NPU V79 架构
  - NPU 架构：HVX (Hexagon Vector eXtension) 向量单元（4-6 个，每单元 32×1024-bit 向量寄存器）+ HMX (Hexagon Matrix eXtension) 矩阵单元（1-2 个，FP16 32×32 tile 为基本计算单元）。TCM 8 MiB，L2 cache 1 MiB。
  - 精度对比使用 NVIDIA RTX3090 GPU server 作为参考。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Qwen 2.5 系列（1.5B, 3B, 7B Instruct）、Llama 3.2 系列（1B, 3B Instruct）
  - Test-time scaling 任务数据集：MATH500（pass@1 accuracy）、GSM8K（pass@1 accuracy），统一使用 0-shot CoT prompt
  - 其他精度测量：WinoGrande（accuracy）、MMLU（accuracy）、Wikitext-2（perplexity，通过 llama-perplexity 工具）
  - Test-time scaling 方法：Best-of-N search（使用 Skywork-1.5B-PRM 作为 outcome-reward scorer）、Step-level Beam Search（使用 Skywork-1.5B-PRM 作为 process-reward scorer）
  - 量化方案：Q4_0（4.5 BPW）用于大多数矩阵；Q8_0（8.5 BPW）用于 FFN down 矩阵

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：主仓库 https://github.com/haozixu/llama.cpp-npu（MIT 许可证，~7K 行 C/C++ + inline assembly）；算子库 https://github.com/haozixu/htp-ops-lib。无 QNN 依赖，使用 Hexagon SDK 6.0.0.2 的 LLVM toolchain。

  - **Tile-Group Quantization 算法 Pipeline**：
    ```
    # ========== 离线阶段：权重预量化变换 ==========
    # 输入: FP16 权重矩阵 W [M, N]（column-major）
    # 输出: Q4_0 量化权重，按 HMX tile layout 排列

    # Step 1: Pre-quantization weight permutation
    # 将 W 重新排列为 HMX tile layout（图 4）：
    #   - 外层：tile 按 column-major 排列（匹配 HMX tile-level inner product）
    #   - 内层：每个 32×32 tile 内每两行 permute（同 transposed 2×32 sub-matrix）
    for tile_row in range(0, M, 32):
        for tile_col in range(0, N, 32):
            tile = W[tile_row:tile_row+32, tile_col:tile_col+32]
            # 每两行 shuffle（cross-lane, 对应 HVX 指令）
            for i in range(0, 32, 2):
                tile_permuted[i], tile_permuted[i+1] = shuffle_pair(tile[i], tile[i+1])
            # 写入 permuted 权重 buffer

    # Step 2: Tile-group quantization（group_size=32, 即 2×16 tile）
    # 在 permuted 内存顺序上执行 group-wise 量化
    for group_idx in range(0, total_elements, 32):
        group = permuted_weights[group_idx:group_idx+32]  # 对应 2×16 tile
        scale = max(abs(group)) / 7.0                      # Q4_0 symmetric
        group_quant = round(group / scale), clip to [-7, 7]
        # 打包：每 2 个 INT4 → 1 个 INT8 byte
        for i in range(0, 32, 2):
            packed[i//2] = (group_quant[i] & 0x0F) | ((group_quant[i+1] & 0x0F) << 4)
        存储 packed[16 bytes] + scale[2 bytes FP16]

    # Step 3: Post-quantization super-group coalesce（图 7）
    # 将 8 个 group（256 个 INT4）合并为 super-group
    # 使 INT4 值恰好填满 1 个 128-byte HVX 向量寄存器
    for super_idx in range(0, num_groups, 8):
        # 提取 8 groups × 16 bytes INT4 = 128 bytes
        super_block_int4 = concat([groups[super_idx+i].packed for i in range(8)])
        # 提取 8 groups × 2 bytes FP16 scale = 16 bytes  
        super_block_scales = concat([groups[super_idx+i].scale for i in range(8)])
        # 存储：先 128 bytes INT4，再 16 bytes scales（AoS 布局，对齐 NPU prefetch）
    ```

  - **LUT-Based Dequantization GEMM 运行时流程**：
    ```
    # 输入: FP16 激活 A [batch, hidden_dim]，Q4_0 量化权重（super-group 格式）
    # 输出: FP16 结果 C [batch, proj_dim]

    for each weight super-group (256 INT4 weights + 8 FP16 scales):
        # Step A: LUT-based INT4 → FP16 转换（图 9）
        # vlut16 指令：对每个 8-bit index，在 16-entry LUT 中查表 → 16-bit 输出
        lut_table = [FP16(-7), FP16(-6), ..., FP16(0), ..., FP16(7)]  # 16 entries
        packed_int4 = load_128bytes(super_group_addr)                   # HVX vector load
        fp16_vals_low  = vlut16(packed_int4, lut_table)                # 低 4-bit → FP16
        fp16_vals_high = vlut16(packed_int4 >> 4, lut_table)           # 高 4-bit → FP16
        # 注意: 对于 V79 之前的 NPU，vlut16 直接输出 IEEE-754 FP16
        #       避免了传统 qfloat ↔ IEEE-754 转换开销

        # Step B: LUT-based scales broadcast
        # 使用 vlut16 将 4 组 scales 广播到整个向量寄存器
        scale_lut = [s0, s1, s2, s3, ...]  # 填充到 16 entries
        const_indices = [0,0,0,..., 1,1,1,..., 2,2,2,..., 3,3,3,...]  # 预定义
        scales_broadcast = vlut16(const_indices, scale_lut)

        # Step C: FP16 乘加
        # 对每对 (low, high) 乘以对应 scale 后累加到结果
        dequant_fp16 = fp16_vals_low * scales_broadcast[0:32] 
                     + fp16_vals_high * scales_broadcast[32:64]
        
        # Step D: 写入 TCM，HMX 执行 tile MatMul
        write_tcm(dequant_fp16, tile_layout)
        # HMX 执行：activation_tile [batch, 32] × weight_tile [32, 32]
        # 内部使用 FP32 累加，输出 FP16

    # 最终 HMX accumulator → TCM → DDR
    ```

  - **LUT-Based FlashAttention Softmax 流程**（Algorithm 1）：
    ```
    # 使用 FP16 FlashAttention，LUT 实现 exp
    # LUT 预计算（系统初始化时）：
    lut_exp = zeros(32768, FP16)  # 64 KiB, 存储在 TCM
    for i in range(32768):
        x = -i / 256.0            # 非正值（safe softmax），步长 1/256
        lut_exp[i] = FP16(exp(x))

    # 在线 Softmax（每个 Attention tile）：
    S = MatMul(Q, K^T, AccumType=FP32)  # [B_q, B_kv] FP16
    m_new = max(m_old, rowmax(S))        # [B_q] FP16
    # LUT-based exp:
    S_shifted = S - m_new                # [B_q, B_kv] FP16, 所有元素 ≤ 0
    for each element s in S_shifted:
        s_abs = s & 0x7FFF              # 忽略符号位（已知非正）
        byte_offset = s_abs << 1        # 左移 1 位 → 2-byte 对齐
        P = vgather(lut_exp, byte_offset)  # 一次 vgather 收集 64 个 FP16
    # 后续：rowsum(P) FP32, rescale, MatMul(P, V) — 同标准 FlashAttention
    ```

  - **关键数值**：
    - Tile quantization vs. conventional：精度差异远小于量化本身损失（Wiki PPL: tile 10.206 vs. common 10.190 vs. F16 9.798）
    - LUT-based GEMM dequantization：相比 baseline scatter-based 方法加速 9.65–19.04×，仅比 "no dequantization" 上界慢 27%
    - LUT-based Softmax：相比 F32 exp 加速 1.26–2.19×，相比 FP16 polynomial exp 加速 up to 1.60×
    - Test-time scaling Pareto：Qwen2.5-1.5B + Best-of-N 超越 Qwen2.5-3B baseline；Qwen2.5-3B + Best-of-N 超越 Qwen2.5-7B baseline
