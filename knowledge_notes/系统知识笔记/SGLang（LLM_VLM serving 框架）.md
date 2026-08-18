## SGLang（LLM/VLM serving 框架）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SGLang 是 LMSYS/UC Berkeley 主导（现加入 PyTorch 生态）的 LLM/VLM 高速 serving 框架（github.com/sgl-project/sglang），核心机制：RadixAttention（用基数树自动缓存/复用共享 prompt prefix 的 KV-cache，降 TTFT、省重复 prefill）、continuous batching（新请求随时加入运行中 batch）、zero-overhead CPU scheduler、token/paged attention、chunked prefill、TP/EP/DP 并行、speculative decoding、FP8/INT4/AWQ/GPTQ 量化。生产部署：xAI Grok 3、Azure DeepSeek R1（AMD GPU）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
本论文把 SGLang 作为集成载体（Section VI-C）：把 dense 矩阵乘（投影层）的默认 CUTLASS GEMM 后端替换为 ANS-enabled 融合内核，权重以压缩 bitstream 加载。运转流程：请求 → SGLang 调度器按剩余显存决定 batch 上限并组批 → 投影层调用融合解压 GEMM（权重从 27.5→18.1 GB，Qwen-14B）→ 释放的显存进入 KV-cache 预算 → 调度器可容纳更多并发请求 → 吞吐上升。论文不改调度算法本身，只改变调度器的资源约束（weight/KV 显存分解），Mixtral-176B 以 EP 部署 4×A100。结果：Qwen-14B 吞吐 1.1–1.2×（batch 47→60/75）、Mixtral-176B 1.6×（batch 20→95）；median TPOT 略增（解压开销 71→81 ms）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
使用：pip 安装 sglang，`python -m sglang.launch_server --model-path <model> --tp N`，关键旋钮 --mem-fraction-static、--max-running-requests、--chunked-prefill-size。集成定制 kernel 的方式：替换投影算子后端（如本论文的 plugin 式 GEMM wrapper）或加自定义 attention/quantization 后端。适用：吞吐优先的在线 serving、共享前缀负载（RadixAttention 收益大）。本论文显示"框架调度不变、只换权重表示"也能获得 1.2–1.6× 吞吐——压缩与调度优化正交可叠加。

MoE 专家放置控制补充视角（ISCA'26，Patterns behind Chaos，Case Study 2）：SGLang 用作 MoE 专家放置实验的载体——(1) 通过 SGLang 的 init_expert_location 接口设置专家在 GPU 上的放置（论文用 Remap/Dup 算法算出的每层专家→GPU 分配结果加载权重，见系统架构层 Prefill-Guided Expert Placement 条目）；(2) 在 SGLang 中插入 cuda. Event timer 构建分布式 profiler，独立测量每 GPU 的 attention、top-k、all-to-all、MoE 各操作耗时；(3) MoE 后端用 DeepEP（ep_dispatch_algorithm="dynamic" 使复制专家 token 均分）+ DeepGEMM。评估 Qwen3-235B（94 MoE 层、128 专家/层、top-8）on 8×H100 NVLink，MMLU/Global-MMLU，batch 64-16384，指标为 MoE 计算时间（三个专家线性层）。这说明 SGLang 的放置/后端接口可以支撑 serving 层负载均衡策略的落地：默认连续放置（0-15/16-31…）被重排（Remap）或复制（Dup）后，MoE 计算提速最高 1.25x。


- Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs
RESONATOR 补充视角（ISCA'26，MLLM 下把 encoder 变为一等公民动态负载）：RESONATOR 基于 SGLang-0.4.7 实现：LLM backbone 沿用 SGLang chunked-prefill（TP=4/8），在框架上新增三条运行时机制——(1) 增强 chunked-prefill 的 LLM chunk 特征（c=(n_p,n_d,L_c)）+ Performance Atlas 离线 profiler 与查询接口；(2) Intra-GPU Sharing Engine（wide/narrow CUDA 流 + SM 配额，green-ctx/libsmctrl 绑 SM）；(3) Inter-GPU Parallelism Engine（encoder 请求批形成 + PRISM DP 调度 + logical sharding 零开销 TP/DP 切换）。baseline 中 SGLang（encoder 挂在 LLM GPU）是主要对照：RESONATOR 同 GPU 预算下 mean E2E 最高 4.9×（Kimi-VL-16B）、TTFT 最高 5.1×、吞吐最高 3.4×（Qwen2-VL-7B 876 vs 462 tokens/s）。
涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression
- Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference
