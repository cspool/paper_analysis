## Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：基于 Insight 1（prefill-decode 阶段的 expert selection 相关性），提出两种 **prefill-aware expert placement 算法**来指导 decode 阶段的 expert 分布：(1) **Remap-based placement**——保持每 GPU 的 expert 数量不变，按 roofline cost 降序排列 expert，贪心分配给负载最小的 GPU（每 GPU 容量上限 E/G）；(2) **Duplication-based placement**——保留默认连续布局（experts 0-15 on GPU 0, etc.），利用 prefill traces 复制热门 expert 到额外槽位（每 GPU 预留 R 个额外槽位），每次贪心选择能最大减少瓶颈负载 max_g load_g 的 (expert, GPU) 对。
  - 实验比较：(1) Remap 和 Dup vs Default（SGLang/Qwen 标准连续布局，experts 0-15 on GPU-0, 16-31 on GPU-1 等）、Best（oracle decode-stage 选择的最优 placement）、Worst（oracle 最差 placement）；(2) 不同 batch size（64-16,384）下的 MoE 计算时间加速比。

- 硬件平台是什么，配置是什么。
  - 8×NVIDIA H100 80GB GPU，NVLink 互联。
  - 网络：NVLink + 节点内互联。
  - 使用 SGLang 部署 Qwen3-235B（94 MoE layers, 128 experts per layer, top-8 selection）。

- 开源Serving框架是什么。修改了什么。
  - 开源 Serving 框架：**SGLang** (https://github.com/sgl-project/sglang)。
  - 修改内容：
    1. **Expert placement 接口**：通过 SGLang 的 `init_expert_location` 接口操纵 expert 在各 GPU 上的分布。
    2. **MoE backend**：使用 **DeepEP** 作为 MoE 后端，`ep_dispatch_algorithm` 设为 "dynamic"，使 tokens 均匀分布到复制 expert 的各副本上。
    3. **分布式 profiler**：在 SGLang 中插入 `cuda.Event` timers 独立测量每个 GPU 上的 attention、top-k、all-to-all 和 MoE 操作时间。
  - 开源链接：Case Study 2 代码仓库 https://github.com/zhongkaiyu/moe_exp_placement，DOI: 10.5281/zenodo.19617695。expert selection traces 开源在 https://huggingface.co/datasets/core12345/MoE_expert_selection_trace。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源**：SGLang（Apache-2.0），DeepEP（开源），Case Study 2 代码 + traces 已开源。
  - **全流程（以 Qwen3-235B 在 8×H100 上使用 Duplication-based placement 为例）**：
    1. **输入阶段**：用户通过 HTTP/gRPC 发送文本请求到 SGLang server。请求先经过 prefill 阶段——所有 input tokens 一起处理，Gate 网络记录每个 token 在每层的 expert 选择（prefill traces）。
    2. **Placement 计算**：基于收集的 prefill traces，运行 Duplication-based 算法——(a) 从 traces 计算每层每个 expert 的频率 f_{l,e}；(b) 生成默认连续布局（GPU-0: experts 0-15, ..., GPU-7: experts 112-127）；(c) 每 GPU 预留 R=1 个额外槽位，总计 128+8=136 experts per layer；(d) 贪心迭代：每次选择能最大减少 max_g load_g 的 (expert, GPU) 对，直到额外槽位用完。使用 roofline-based cost model 估算每个 GPU 的负载。
    3. **Expert 重分布**：通过 SGLang 的 `init_expert_location` 接口将新布局加载到各 GPU。DeepEP backend 使用 "dynamic" dispatch 算法确保 tokens 均匀分配到复制 expert 的各副本。
    4. **Decode 执行**：每个 decode step：(a) Attention 计算（各 GPU 处理自己的 KV cache 分片）；(b) Gate/Top-k 路由——选择每个 token 的 top-8 experts；(c) DeepEP all-to-all 通信——将 tokens 发送到目标 expert 所在 GPU；(d) MoE 计算——各 GPU 执行本地 expert 的 3 个 GEMM 操作（gate_proj, up_proj, down_proj）；(e) 第二个 all-to-all——将结果返回原 GPU；(f) 下一层继续。
    5. **性能测量**：通过插入的 `cuda.Event` timers 测量 MoE computation time（3 个 expert linear layers + all-to-all + top-k），排除 attention 时间。Remap 和 Dup 分别实现 15.5% 和 12.5% 的加速（vs Default），均在 Best（oracle）的 10% 以内。
