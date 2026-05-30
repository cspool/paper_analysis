## MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：MoE-SpeQ 的 Serving 调度核心包含三大组件：
    1. **Speculative Governor（自适应控制面）**：基于 Amortization Roofline Model 动态优化 speculative draft length k。将性能建模为两 Roof：Compute Roof（水平线，I/O 完美隐藏时的最大吞吐，高度依赖 k 影响草稿+验证成本）和 I/O Roof（斜率=有效 PCIe 带宽 B_PCIe 的斜线）。在线求解 argmax_k Θ(k) = k_accept(k) / T_cycle(k)，其中 k_accept(k) = Σ∏p_j（EMA 更新的条件接受概率），T_cycle(k) = max(T_draft(k), T_pcie,init) + T_pcie,new(k) + T_verify(k+1)。离线 SLO 约束确定 k_max 上限（如 TTFT < 500ms），在线受限搜索 [k_min, k_SLO]。
    2. **Expert Scheduler（数据编排器）**：基于 Expert Lookahead Buffer (ELB) 的 lookahead 驱动三阶段预取流水线——Phase I: 利用 cache 做 locality-aware cache priming；Phase II: ELB 中部选择性预取高置信度 experts；Phase III: 饱和 VRAM cache 消除 verify 阶段 I/O stall。ELB 为 k×L 结构（每个草稿 token×每层包含 (expert_id, confidence_score) 元组），在 CPU 端无阻塞构建。lookahead-aware eviction 策略替换最不可能在后续阶段使用的 experts。
    3. **Execution Engine（执行引擎）**：CUDA multi-stream 调度管理四维并发——多阶段预取、预取与按需加载协调、计算通信 overlap、双向 host-device 传输。pipeline-based 异步加载机制使用 pinned memory + non-blocking CUDA memcpy。verification 阶段 computation reordering 重排 batch tokens 使同 expert 计算连续，最大化 L1/L2 cache 利用率。static shared memory 配置避免运行时分配同步。batched expert selection pattern 处理最小化 D2H 传输。
  - 实验比较：（1）MoE-SpeQ vs HuggingFace Transformers (device_map) vs Mixtral-Offloading-SC vs Mixtral-Offloading-SM，end-to-end TPOT 对比；（2）不同 cache 策略命中率：speculative prefetching vs LRU vs LRU(scaled) vs Single Prefetch(sooner/later)，在 16/24/32GB VRAM 预算下；（3）消融：Full vs without async prefetch vs without fused kernel。

- 硬件平台是什么，配置是什么。
  - 单卡 NVIDIA A100-40GB GPU（PCIe 4.0 x16，理论双向 32GB/s 聚合带宽）。
  - CPU：24-core Intel Xeon Silver 4310，256GB RAM。
  - 实验变量：GPU 内存约束（low-memory / high-memory 两种配置），expert cache 容量（6/16/22/32/48 槽位对应不同 VRAM 预算）。

- 开源Serving框架是什么。修改了什么。
  - 基于 Hugging Face Transformers 框架构建，利用其通用模型接口。未使用 vLLM、SGLang 或 llama.cpp（论文明确指出不同底层框架间不可直接对比）。
  - 修改/新增内容包括：
    (1) **Expert Scheduler**：新增 Expert Lookahead Buffer 数据结构、三阶段预取流水线、lookahead-aware eviction 逻辑，替换原生 Transformers 的粗粒度 device_map offloading。
    (2) **Speculative Governor**：新增 Amortization Roofline Model 建模与在线优化逻辑，动态调整 k。
    (3) **Execution Engine**：新增 CUDA multi-stream 调度的异步执行管理（多阶段 prefetch、计算-通信 overlap、双向传输协调），static shared memory 预配置避免运行时分配同步，pinned memory + non-blocking CUDA memcpy 流水线化异步加载，batched D2H expert selection transfer。
    (4) **Speculative Decoding Loop**：新增 draft→prefetch→verify 三步循环替代原生单步 autoregressive decode。
    (5) **Parameter/KV Cache Sharing**：修改 draft/target 模型管理逻辑，使两者共享 non-expert 参数和 KV cache，每步 verify 后同步 KV cache。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文未提供开源代码仓库链接。
  - Serving 框架全流程（基于论文 §3 和 §4.1.2）：
    1. **输入**：用户 prompt token 序列 → GPU 端 prefill 阶段计算 attention + 加载首层 experts（通过 Transformers model.forward）。
    2. **解码循环**（每步一个 cycle）：
       - **Step 1 - 初始专家加载**：GPU host-to-device 异步传输首组所需 experts（T_pcie,init），同时 CPU 构建初始 ELB。
       - **Step 2 - Draft 生成**：量化的 INT4 draft model 在 GPU 上自回归生成 k 个候选 token，使用 fuseMoE kernel 加速细粒度 MoE 计算。每 token 生成后 CPU 非阻塞解析 router logits → 追加 ELB 条目 (expert_id, confidence_score)。
       - **Step 3 - 预取调度**：Expert Scheduler 读 ELB，Phase I 利用现有 cache hits → Phase II 选择性预取 ELB 中部高置信度 experts（non-blocking H2D transfer） → Phase III 饱和所有缺失 experts。Speculative Governor 根据 Amortization Roofline Model 实时计算最优 k，若 token 接受率下降则动态缩短 k。
       - **Step 4 - Verify**：Target FP16 model 对 [original_prompt + k draft tokens] 单次并行 forward。执行前做 computation reordering（按 expert id 重排 token 计算顺序以最大化 L2 cache 命中）。逐 token 比对 target logits vs draft tokens：接受匹配前缀，从 target 分布采样分歧处新 token，回滚 KV cache/sequence states。
       - **Step 5 - KV Cache 同步**：将 target 的高精度 KV cache 复写到 shared KV cache，供下一步 draft 使用。
    3. **输出**：每个 cycle 输出 ≥1 个有效 token，循环直到 EOS 或 max_len。
    4. **CUDA 多流调度细节**：4 条 CUDA stream 分别管理——(a) draft 模型前向计算流、(b) H2D expert 预取传输流、(c) target 模型 verify 计算流、(d) D2H router logits 回流。通过 CUDA events 管理跨流同步与互斥，CUDA streams 间通过 cudaStreamWaitEvent 确保数据依赖。
