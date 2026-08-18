## Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现为训练无关（training-free）的 self-speculative decoding 格式变换 pipeline：把目标模型（BF16）的权重与 KV cache 拆成"speculation data + verification data"两份数据，草稿模型不独立存储。(1) 非结构化值剪枝：权重采用 Wanda 的 activation-aware 剪枝（用校准数据计算激活 L2 norm，与权重逐元素相乘得 importance score，按 top-k 保留）；KV cache 采用 per-token 幅度剪枝（Key cache 存在 channel-wise outlier，Value cache 的 per-token 剪枝在注意力机制内功能等价于 output-aware 剪枝）。(2) 尾数截断（mantissa truncation）：不用量化而是直接截断低位 mantissa 比特——不改变数值表示、重建开销低，草稿模型是目标模型的严格比特子集。(3) 指数压缩：BF16 的 8-bit 指数占位宽 50%，是压缩率瓶颈；提供两种配置——Cassandra-1 用 unary 编码（1/01/001/…，以 1 结尾的码字边界可直接并行解析）做无损熵压缩（权重/KV 指数 Shannon 熵约 2.6/2.7 bits，实际平均压到约 2.85 bits）；Cassandra-2 用 MX 格式共享指数（压缩率更高、轻微精度损失）。draft 推理只加载 speculation data，zero-padding 重建为标准浮点格式后在标准 FP 单元上执行（降带宽不减算术）；target 推理加载 speculation+verification 全部数据完全重建原始模型做并行验证，无独立草稿模型 → 零额外显存。draft 长度 γ 在 3–5 内取最优。实际算法顺序：指数压缩 → mantissa 截断（图 4）。
  - 实验比较：zero-shot 精度 vs BF16 baseline 与损失压缩方法（SmoothQuant W8A8（vLLM 官方 INT8）、QoQ、SqueezeLLM、DuQuant、Wanda）；性能 vs BF16、INT8/FP8 量化与 speculative decoding（Draft&Verify、MagicDec、Lookahead Decoding、EAGLE-3）；内存容量 vs Llama3-based speculative decoding 与 Eagle-3。
- 硬件平台是什么，配置是什么。
  - 软件模拟器：PyTorch + 自定义 CUDA kernels（仅功能建模，不带来 GPU 性能收益）；超参搜索在 NVIDIA A100（8B 模型约 5 分钟）。性能：Accel-Sim 模拟的 Nvidia RTX 4090（用 Ampere 架构 trace + 近似 RTX 4090 的配置，因 Accel-Sim 不支持 Ada-Lovelace）与 Nvidia Jetson AGX Orin；NPU 为扩展 Scale-Sim + LPU simulator 的 cycle-level 模拟器（64 TFLOPS MAC、273 GB/s 带宽、128GB LPDDR5X、9MB scratchpad）。面积功耗：Synopsys Design Compiler 28nm + Samsung 28nm SRAM Compiler。
- 模型是什么。数据集和bench分别是什么。
  - 模型：Deepseek-R1-Distillated-Llama3-8B、Qwen3-8B-Thinking、Qwen3-4B-Thinking-2507。benchmark：GPQA-Diamond、Math-500、AIME 2025（zero-shot 精度；GPQA-Diamond 与 Math-500 随机抽 100 题，AIME2025 全题）；LiveCodeBench、GPQA-Diamond、Longbench(QMSum)、Math-500（性能与接受率场景，按 benchmark 取平均输入/输出长度）。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - Cassandra 自身代码：论文未提供开源链接，未检索到公开仓库（无法确认）；arXiv:2605.26558 仅论文正文。baseline 均为官方开源实现（Draft&Verify、MagicDec、Lookahead Decoding、EAGLE-3 官方实现，SmoothQuant 用 vLLM 官方 INT8 量化实现）。pipeline 伪代码（以权重 W 为例）：
    1. importance = |W| ⊙ norm2(X_calib)（Wanda，激活 L2 norm 按输出通道广播）；mask = topk(importance, 1−w_p)；spec_W = W[mask]，ver_W = W[¬mask]（KV cache 按 per-token 幅度 top-k 同法）。
    2. 指数压缩：spec_W 的 BF16 8-bit 指数按频率建 unary codebook（高频→短码，码字以 '1' 结尾），拼接为 bitstream（Cassandra-1）；或同 block 元素共享指数（Cassandra-2，MX）。
    3. 尾数截断：spec_W mantissa 保留高 w_t 位（默认 4-bit），低位丢弃。
    4. draft 前向：spec_W 与 KV 的 speculation 部分 → decoder 重建（unary 解码 = parallel zero counter 数连续 0 + LUT；bitmap de-sparsification 补 0）→ 标准 FP GEMM；target 前向：加载 ver_W 补全后完整 BF16 GEMM，对 γ 个候选 token 并行验证（greedy 或 rejection sampling，式 (1)：n ← min({i−1 | r_i > p_i/q_i} ∪ {γ})）。
    5. 超参数目标函数 J = α / (S_w(1−w_p)(B−w_t) + S_kv(1−kv_p)(B−kv_t))：grid search 剪枝率 30–60%（步长 10%）、截断 0–5 bits（步长 1 bit），dev set 8 样本；默认配置 40% 剪枝 + 4-bit 截断（权重与 KV cache 相同），可直接迁移到其他模型。
  - 效果：性能 1.78×–2.41×（vs BF16；INT8 量化仅 1.3×）；Cassandra-1 精度与 BF16 完全一致（Deepseek-R1-Distillated-Llama3-8B：GPQA 49.0 / Math-500 87.0 / AIME 26.7），Cassandra-2 与 SmoothQuant 相当；接受率 Cassandra-1(γ=5) 0.74–0.88、Cassandra-2(γ=3) 0.74–0.91；固定内存预算下比 Llama3-based speculative decoding 多生成 11.59× token、比 Eagle-3 多 1.81×。
