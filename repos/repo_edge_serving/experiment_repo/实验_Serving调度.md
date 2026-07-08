## TZ-LLM

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：**TZ-LLM**——基于 Arm TrustZone TEE 保护端侧 LLM 模型机密性的安全推理系统。在 OpenHarmony OS 上将 llama.cpp 推理框架运行在 TEE 中作为 Trusted Application (TA)，实现两个 Serving 层面的核心优化：
    1. **Pipelined Parameter Restoration（流水线参数恢复，§4.1）**：利用 LLM 推理的 DAG 确定性内存访问模式，在推理执行时并行地进行安全内存分配（CMA allocation）、加密参数加载（flash I/O）和解密。将这三个恢复操作作为独立的 restoration operators 插入到 LLM 计算图中，与计算 operators 并发执行。使用基于优先级的抢占式流水线调度策略：当多个恢复/计算 operators 竞争 CPU 时，优先调度最紧急的任务以避免关键路径上的流水线气泡（pipeline bubble）。支持 partial parameter caching：推理结束后按计算图逆拓扑序惰性释放安全内存，保留早期 operators 的参数以消除下次推理的初始气泡。
    2. **Co-driver NPU Time-Sharing（协同驱动 NPU 分时复用，§4.3）**：将 NPU 驱动分离为 REE 中的控制平面（调度、电源管理、频率控制）和 TEE 中的微型数据平面（~1K LoC，仅负责安全 NPU job 的启动和完成处理）。两个驱动通过 smc 协作：REE 驱动维护统一的安全/非安全 NPU job 调度队列，当安全 job 被调度时通过 shadow job 机制移交 NPU 控制权给 TEE 驱动，TEE 驱动配置 TZPC/TZASC/GIC 切换 NPU 到安全模式后执行 job。
    3. **多线程 TA 支持（§3.2）**：通过 shadow thread 机制（TA 线程与 CA 中的 shadow thread 配对，shadow thread 通过 smc 启动/恢复 TA 线程）使 TA 可使用多 CPU 核心加速推理。
    4. **框架状态检查点（§3.2）**：保存推理框架初始化状态（框架、模型元数据、tokenizer）到 flash，每次推理请求从检查点恢复，消除冷启动的框架初始化开销。
  - 实验比较：
    - Baseline：（1）**Strawman**：在 TEE 中执行冷启动 + CPU-only 推理，无流水线恢复和 NPU 支持（提供安全性但无性能优化）；（2）**REE-LLM-Memory**：llama.cpp 在 REE 中运行，模型参数全部预加载在内存中（理论最优性能，无保护，内存低效）；（3）**REE-LLM-Flash**：llama.cpp 在 REE 中运行，使用流水线恢复从 flash 加载参数（无加密，无保护）
    - 指标：TTFT（Time-to-First-Token）、Decoding Speed（token/s）、NPU 分时复用吞吐
    - Benchmarks：UltraChat（多轮对话）、PersonaChat（聊天摘要）、DroidTask（UI 自动化）
    - 关键结果：vs Strawman：TTFT 降低 76.1%∼90.9%，Decoding speed 提升 0.9%∼23.2%；vs REE-LLM-Flash：TTFT 开销 5.2%∼28.3%，Decoding 开销 1.3%∼4.9%

- 硬件平台是什么，配置是什么。
  - **Orange Pi 5 Plus 开发板**：Rockchip RK3588 SoC
    - CPU：4× Cortex-A76 @ 2.4GHz + 4× Cortex-A55 @ 1.8GHz
    - NPU：3 核，最高 6 TOPS（INT8）
    - 内存：16 GB LPDDR4X
    - 存储：1 TB NVMe SSD（PCIe 3.0 x4）
  - TrustZone 硬件：Arm TrustZone（TZASC、TZPC、GIC 扩展）
  - 构建机：Ubuntu 22.04.3，需要 OpenHarmony Device Connector、Docker

- 开源Serving框架是什么。修改了什么。
  - 开源框架：**llama.cpp**（ggml-org/llama.cpp）——流行的 C/C++ 端侧 LLM 推理框架
  - 修改内容：
    1. **Pipelined restoration（+1.2K LoC）**：在 llama.cpp 计算图中插入三类 restoration operators（CMA allocation、flash I/O、AES 解密），实现优先级抢占式流水线调度器，支持 partial parameter caching 的逆拓扑序内存释放策略。计算图通过 llama.cpp 内部接口直接提取。
    2. **NPU 数据平面驱动集成（+1K LoC）**：将 Rockchip NPU 驱动的数据平面（job setup、MMIO launch、completion interrupt handling）集成到 llama.cpp TA 中，支持安全 NPU job 的发起和完成处理。
    3. **OpenSSL 解密集成**：使用 OpenSSL 库对加密模型参数进行 AES 解密。
    4. **TEE OS 修改（+112 LoC）**：CMA page 内存映射管理（62 LoC）+ 动态 TZASC/TZPC 配置（50 LoC）。
    5. **REE Linux 内核修改（+364 LoC）**：NPU 驱动 shadow job 调度（167 LoC）+ TZ 驱动 CMA allocation/deallocation（197 LoC）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源状态：**已开源**。Artifact DOI：https://doi.org/10.5281/zenodo.17213486。包含 TZ-LLM 原型系统源码和论文实验脚本。需 Orange Pi 5 Plus（RK3588）硬件。
  - 全过程（以 Llama-3-8B 8-bit 量化、512-token prompt 推理为例，在 Orange Pi 5 Plus 上）：
    1. **REE CA 发起请求**：用户在 REE Android/Linux 应用中发起 LLM 推理请求。CA 通过 TZ driver 向 TEE OS 发送请求启动 LLM TA。
    2. **TA 初始化（检查点恢复）**：LLM TA 从 flash 中恢复预保存的框架初始化状态检查点（包含框架、模型元数据、tokenizer 已初始化状态），跳过冷启动的框架初始化（2.3s → 0）。
    3. **流水线参数恢复 + 推理并行启动**：TA 按计算图拓扑序遍历 operators：
       - **Stage 1（首参数恢复）**：若启用 partial caching，operator 0∼1 的参数已在安全内存中 → 立即开始 CPU/NPU 计算。若无缓存，先调用 `extend_allocated` → TEE OS → TZ driver → Linux CMA 分配物理连续内存 → `extend_protected` → TZASC 配置保护该区域 → REE 文件系统通过 aio 直接 DMA 加载加密参数到已分配但未保护的内存（避免 bounce buffer） → `extend_protected` 保护新内存 → CPU 解密参数。
       - **Stage 2（流水线展开）**：scheduler 维护一个 ready operators 优先级队列。优先级规则：ready CPU computation operator > 最早 computation operator 的 restoration operator。CPU allocation/decryption operators 被划分为 micro-operators 并插入抢占点。当 operator 0 的 CPU computation 在 Cortex-A76 上执行时，operator 1 的参数通过 NVMe SSD → CMA 分配 → TZASC 保护 → AES 解密流水线并行进行。当 operator 0 的矩阵乘法提交到 NPU 时，operator 2 的 I/O + allocation + decryption 并行进行。
       - **Stage 3（NPU job 执行）**：LLM TA 中的 TEE NPU 驱动（数据平面）初始化安全 NPU job 的执行上下文（I/O page table、register commands、input/output buffers 均在安全内存中）→ 向 REE NPU 驱动发出 shadow job → REE 调度器将 shadow job 排入统一调度队列 → 当 shadow job 被调度时，REE 驱动通过 smc 通知 TEE 驱动接管 NPU → TEE 驱动配置 TZPC（隔离 NPU MMIO 从 REE）→ 等待非安全 job 完成 → 配置 TZASC（授权 NPU 访问安全内存）→ 配置 GIC（NPU 中断路由到 TEE）→ MMIO 写入 NPU 寄存器启动 job → NPU 三核并行执行矩阵乘法 → 完成后 NPU 触发安全中断 → TEE 驱动处理中断 → 归还 NPU 到非安全模式 → 通知 REE 驱动 shadow job 完成。
    4. **解码阶段**：prefill 完成后进入 decode loop。每步生成一个 token，使用 GPU/NPU 执行单 batch 矩阵乘法。KV cache 在安全内存中增长。解码速度开销主要来自 TEE-REE NPU 驱动的 smc 通信（1.3%∼4.9% vs REE baseline）。
    5. **推理结束**：TA 按逆拓扑序调用 `shrink` 释放安全内存 → TEE OS 清除敏感数据 → 归还内存到 CMA。若 REE 内存压力低，部分早期参数保留在安全内存中供下次推理使用（partial parameter caching）。


## On-device Semantic Selection Made Low Latency and Memory Efficient with Monolithic Forwarding

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：**PRISM 推理系统**——基于 **monolithic forwarding**（单一化前向）范式的训练无关 cross-encoder reranker 推理服务系统。将传统按候选文档独立分批（isolated batches）的方式改为将所有候选文档合并为一个 monolithic batch 统一前向，在此基础上实现三个 Serving 层面的优化：
    1. **Overlapped Layer Streaming（§4.2）**：将模型权重从 SSD 按层流式加载，仅保持当前层和下一层两层权重在内存中。当前层计算时并发异步 prefetch 下一层权重，计算完后释放当前层 buffer 并回收用于 prefetch 再下一层。通过 monolithic batch 提供的大计算窗口（compute-heavy prefill-only 特性）完全隐藏 I/O 延迟。使用独立 I/O 进程（Libuv 异步 I/O）绕过 GIL，CUDA MPS 实现 GPU 共享。
    2. **Chunked Execution（§4.3）**：将 monolithic batch 拆分为 chunks 逐 chunk 顺序执行，仅保留一个 chunk 的中间张量在内存中。chunk size 根据设备算力、模型大小、序列长度动态确定下界以充分利用硬件。支持动态 hidden states offload（当前 chunk 计算时，并发 offload 已完成 chunk 的 hidden states + prefetch 下一 chunk 的 hidden states，最多 3 chunks 在内存）。
    3. **Embedding Table Caching（§4.4）**：基于 token 分布稀疏性（20 文档 × 512 tokens ≤ 10,240 unique tokens，仅为 151,669 词汇表的 6.75%），使用小型 LRU cache（10% 词汇表大小）缓存活跃 embedding 权重。cache miss 时同步从 SSD 读取。利用自然语言 Zipf 分布维持高命中率。
  - 实现栈：~5K lines Python + ~1.7K lines C，基于 HuggingFace Transformers v4.52.4 + Accelerate v1.6.0。I/O 进程使用 Libuv 异步 I/O + PyTorch Multiprocessing shared memory。GPU 共享通过 CUDA MPS。
  - 实验比较：
    - Baseline：HF（Transformers 标准 in-memory）、HF Offload（Accelerate disk offloading，所有层 offload 到 SSD，执行前加载）、HF Quant（GPTQ W4A16）、PRISM Quant（PRISM + 量化正交叠加）
    - Microbenchmark：18 datasets × 5 models × 2 platforms
    - Real-world：RAG pipeline、Agent Memory、LLM Long Context Selection

- 硬件平台是什么，配置是什么。
  - **NVIDIA Platform**：笔记本 Intel Ultra9-275HX + 32 GiB RAM + RTX 5070 Laptop GPU（8 GiB VRAM）+ 1 TiB PCIe 4.0 SSD
  - **Apple Platform**：Mac Mini M2 SoC + 16 GiB unified memory + 256 GiB PCIe 4.0 SSD
  - 额外 A800 GPU 仅用于测量 Qwen3-4B/8B 在 HF baseline 下的 OOM 参考值

- 开源Serving框架是什么。修改了什么。
  - 开源框架：**HuggingFace Transformers v4.52.4** + **HuggingFace Accelerate v1.6.0**
  - 修改内容：
    1. **Monolithic batching**：将 query-candidate pairs 从独立 batch 执行改为统一 single batch 前向，维护全局候选视图
    2. **Overlapped Layer Streaming**：替换 Accelerate 的全模型 offload 方案（执行前加载整个模型），改为仅保留两层权重的流式加载。实现双进程架构：computation process + I/O process，通过 PyTorch Multiprocessing shared memory buffer 通信，I/O process 使用 Libuv 饱和 SSD 带宽
    3. **Chunked Execution**：在每层前向中拆分 batch 为 chunks 顺序执行，替代 HF 的整 batch 并行执行。动态 chunk size 计算考虑硬件 FLOPS、内存带宽、序列长度。支持 hidden states dynamic offloading
    4. **Embedding Table Caching**：替换 HF 的全量 embedding 加载为 LRU cache-based 按需加载，利用 token 分布稀疏性
    5. **CUDA MPS**：启用 NVIDIA Multi-Process Service 实现 computation 与 I/O 进程间的低开销 GPU context 切换

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源状态：**已开源**。源码：https://ipads.se.sjtu.edu.cn:1312/opensource/monolithic_forwarding，存档 DOI：10.5281/zenodo.18809731。EUROSYS '26 Artifact Evaluation 通过。
  - 全过程（以 RAG pipeline 中 reranking 阶段为例，20 候选选 top-10，Qwen3-Reranker-0.6B on NVIDIA RTX 5070）：

    ```
    === Step 1：环境准备（安装） ===
    1. git clone https://ipads.se.sjtu.edu.cn:1312/opensource/monolithic_forwarding
    2. bash install_dependencies.sh  # 创建 Conda 环境 + 安装依赖 + 编译 C extension
    3. bash download_models.sh       # 从 HuggingFace 下载 reranker checkpoints
    
    === Step 2：输入准备 ===
    4. RAG pipeline 的 hybrid search（dense vector + sparse keyword）返回 top-20 候选文档
    5. 将 query + 20 个候选文档拼接为输入：
       inputs = tokenizer([q+d_1, q+d_2, ..., q+d_20], 
                          padding=True, truncation=True, max_length=512)
       # inputs shape: [20, L]（L=平均 500 tokens）
    
    === Step 3：PRISM 推理执行 ===
    6. python -m prism.rerank \
         --model Qwen3-Reranker-0.6B \
         --candidates 20 --topk 10 \
         --input_file candidates.json
    
    ==== PRISM 内部执行流程 ====
    
    6a. Embedding Layer（embedding table cache）：
        - 收集所有 input_ids 中的 unique tokens（≤10,240 个，vocab 的 6.75%）
        - LRU cache lookup：命中 → 直接使用 cached embedding weights
        - Cache miss → libuv 同步从 SSD 读取对应 embedding rows → 插入 cache
        - 执行 embedding lookup：hidden_states = embedding_table[input_ids]
          shape: [20, 500, D]（D=hidden_dim）
    
    6b. Chunked execution partition：
        - 计算 optimal chunk_size：考虑 GPU FLOPS、PCIe BW、L、D
        - chunk_size = max(min_chunk_for_gpu_saturation, 
                           min_chunk_for_io_overlap)
        - 本例：20 candidates → chunks of 2 → 10 chunks/layer
    
    6c. Overlapped Layer Streaming（Transformer layers 0..27）：
        初始化：buffer_A ← layer_0 weights（SSD→RAM），buffer_B ← layer_1 weights（prefetching）
        
        FOR layer_i = 0 to 27:
          // === 6c-1. Progressive Cluster Pruning（layer_i 前） ===
          IF layer_i > 0:
              scores = classifier_head(all_hidden_states[:, last_token, :])
              cv = |std(scores) / mean(scores)|
              IF cv > threshold:
                  clusters = KMeans(scores)  # CPU ~1ms
                  路由 selected/dropped/deferred
                  IF 提前终止条件满足 → 跳到 Step 4
    
          // === 6c-2. Chunked forward pass（layer_i 计算） ===
          FOR chunk_j IN remaining_candidates_chunks:
              # GPU 执行：
              h_chunk = TransformerLayer_i(h_chunk)
              # h_chunk = h + FFN(causal_attn(LayerNorm(h)))
              # GEMM: cuBLAS on RTX 5070 SM
              # 中间张量仅保留当前 chunk，峰值内存 ∝ chunk_size × L × D
    
          // === 6c-3. 并发 I/O（I/O process 并行执行） ===
          I/O process (independent, bypasses GIL):
              # Prefetch layer_{i+1} weights from SSD → buffer_B
              libuv_fs_read(layer_{i+1}_weights_path, buffer_B)
              # 利用 monolithic batch 创建的充足计算窗口隐藏 I/O
    
          // === 6c-4. Buffer swap ===
          释放 buffer_A（layer_i weights）
          buffer_A ← buffer_B（layer_{i+1} now ready）
          buffer_B ← start prefetch layer_{i+2}
    
    6d. 输出：返回 top-10 候选文档 scores 和 indices
    
    === Step 4：结果传递给 LLM ===
    7. PRISM 输出的 top-10 文档 + 原始 query → 拼接为 prompt
    8. prompt → Qwen3-32B（server side, 2× A800）→ 生成最终回答

    === 关键硬件数据路径（NVIDIA RTX 5070 Laptop） ===
    SSD (PCIe 4.0, ~7 GB/s read)
      │ libuv async read (I/O process)
      ▼
    CPU RAM (32 GiB)
      │ shared memory buffer (PyTorch Multiprocessing)
      ├─► GPU VRAM (8 GiB) ──► CUDA SM ──► Tensor Core (FP16 GEMM)
      │        │                                      │
      │        └─ current layer weights (buffer_A)    └─ h_chunk output
      └─ prefetch next layer weights (buffer_B)
    ```

## Efficient, VRAM-Constrained Cross-Lingual Model Inference on Client Devices

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Pipelined Sharding（Profile → Plan → Infer 三阶段流水线调度），通过在 llma.cpp 中实现三种调度计划（GPU-only / Static / Dynamic），按 token tier 自动选择最优 CPU-GPU 混合执行策略。VLMOpt 针对 VLM 的 vision encoder 做 tensor offload + Tiled FlashAttention + serialized teardown。
  - 实验比较：
    - 单用户 LLM：TTFT、TPS、E2EL 三项指标，在不同 VRAM budget（2G–32G）和 context length（1K–64K）下对比 baseline（llama.cpp b6097，-ngl 手动调优）。
    - 多用户 batched：batch size 1–64 下的 TPS 和 scaling speedup。
    - VLM：Cosmos-Reason1 和 vnemo4b 在不同分辨率（480p–1440p）和 VRAM budget（4G–14.5G）下的 E2EL speedup。

- 硬件平台是什么，配置是什么。
  - cli1: RTX 3500 Laptop (12GB VRAM), Intel Ultra 7 (16 cores), 64GB RAM, 119.5 GBps mem BW, PCIe 13 GBps (16 peak)
  - cli2: RTX 5070 Ti Desktop (16GB VRAM), Ryzen 7 (8 cores), 128GB RAM, 57.6 GBps mem BW, PCIe 50 GBps (64 peak)
  - cli3: RTX 5090 Workstation (32GB VRAM), EPYC (16 cores), 256GB RAM, 153.6 GBps mem BW, PCIe 50 GBps (64 peak)

- 开源Serving框架是什么。修改了什么。
  - 开源框架：llama.cpp（GGML 生态）。Baseline 使用 llma.cpp b6097，通过 -ngl 参数手动指定 GPU 层数。
  - 修改内容：
    1. 实现 Profile 阶段：在安装时 benchmark CPU/GPU kernel（不同量化、op shape、线程数），构建 kernel profile database。
    2. 实现 Plan 阶段：按 token tier（1, 4, 16, ..., 16K）生成 GPU-only / Static / Dynamic 三种 plan，通过 roofline + profile 选最优，写入 schedule lookup table。
    3. 实现 Infer 阶段：运行时根据当前 batch new-token count 查表，O(1) 调度。支持 plan 切换、streamed weight transfer、CPU/GPU 间 KV/RS cache 同步、split scheduling callback 和 async copy backend 实现 compute/copy overlap。
    4. VLMOpt 实现：vision tensor offload（CLIP/vision-encoder 权重 pin 在 sysRAM 上按需 stream）、Tiled FlashAttention（tile Q 限制峰值显存）、serialized vision teardown（vision GPU buffer 释放后再初始化 language context）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源状态：MLSys 2026 Artifact Evaluation 三枚全获。代码上游 PR：https://github.com/ggml-org/llama.cpp/pull/22692（open，审核中，作者正在精简 patch）。
  - 全过程（以推理一个 LLM prompt 为例）：
    1. **Install-time Profile**：在用户安装时，对当前机器的 CPU/GPU 执行 kernel benchmark，测量不同量化格式（f16/q4/q2_k）、不同 op shape、不同线程数下的 latency。同时在 PCIe contention 条件下 profile CPU kernel。结果存储为 kernel profile database。
    2. **Model-load-time Plan**：加载模型后，planner 读取 profile database。对每个 token tier（1, 4, 16, ..., 16K 个 new tokens），分别生成三种 plan：
       - GPU-only：所有权重 stream 到 VRAM double-buffer scratch，全部 GPU 执行。
       - Static：按优先级分配 VRAM（attention > KV cache > FFN > outputs），仅中间输出过 PCIe。
       - Dynamic：Static + extra GPU work via dynamic streaming，CPU compute 与 GPU weight streaming overlap。
       通过 roofline model + profile 选每个 tier 的最优 plan，写入 schedule lookup table。
    3. **Inference-time Infer**：用户发送 prompt → tokenizer 分词 → prefill 阶段产生 KV cache。Decode 每步：统计当前 batch 的 new-token 数 → 查 schedule table 选 plan → 按 plan 分配 layer 到 CPU/GPU → 执行（CPU kernel 和 GPU kernel 异步 overlap）→ 输出 token。KV cache 按 plan 分布于 CPU/GPU 间，切换 plan 时同步。
    4. **VLM 额外流程**：图像输入 → vision encoder（vision weights 从 sysRAM stream 到 GPU）→ Tiled FlashAttention 处理高分辨率 → vision teardown 释放 GPU buffer → language model 初始化 → decode loop 同上。

## Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：基于 llama.cpp 构建的端到端 LLM 推理系统，专门为移动 NPU 上的 test-time scaling 工作负载优化。核心 Serving 层面的设计包括：
    1. **Test-Time Scaling 支持**：在 llama.cpp 的 NPU backend 中实现 batch decode（batch_size > 1），支持 Best-of-N（并行采样后按 reward model 选最优）和 Beam Search（按 PRM score 动态剪枝低质量路径）。系统通过 rpcmem shared memory 实现 CPU 与 NPU 间零拷贝数据传输，CPU 侧负责 sampler、tokenizer、lm_head vocab projection、PRM/ORM scoring，NPU 侧负责 transformer layers 的 GEMM + Attention 计算。
    2. **CPU-NPU 混合推理调度**：并非所有算子都放在 NPU 上。lm_head（vocab projection matrix，因词汇表大导致对数张量占用空间大）和采样逻辑在 CPU 端执行，transformer body（attention + FFN）在 NPU 端执行。两者通过 shared memory buffer 交换中间激活。由于 Hexagon NPU 仅支持 32-bit 虚拟地址空间，将完整 logits 张量放 NPU 会导致部分模型无法运行，因此 lm_head 保守地保留在 CPU 侧。
    3. **FastRPC Shared Memory 通信**：替代默认的 RPC 实现，使用 rpcmem（内核 dmabuf 封装）在 CPU 和 NPU 间共享物理内存，消除 inter-processor data copy。CPU 写数据后需手动 invalidate NPU cache（因 Snapdragon SoC 上 CPU→NPU 仅单向缓存一致性）。NPU 侧线程持续轮询共享内存区域接收计算请求。
    4. **不依赖 QNN**：整个推理系统通过逆向工程未公开 HMX 指令实现 NPU 编程，避免 QNN 的静态固定形状计算图限制，可以动态调整 batch size 以满足 test-time scaling 需求。关键障碍：QNN 仅支持 per-tensor/per-channel 量化（导致严重精度下降——Table 1 显示 QNN W4A16 下 MATH500 仅 2.1% vs AutoAWQ 的 15.9%）。
  - 实验比较：
    - End-to-End Decoding Throughput：不同 batch size（1/2/4/8/16）、不同模型（Qwen2.5-1.5B/3B, Llama3.2-1B/3B）、不同设备（Snapdragon 8 Gen 2/3/Elite）的 decode tokens/s。
    - Accuracy-Latency Pareto：Best-of-N 和 Beam Search 在不同 generation budget 下的 MATH500/GSM8K 精度 vs 平均 decode 延迟。与 baseline（单次常规采样）和更大模型（Qwen2.5-3B/7B）对比。
    - Power & Energy：OnePlus 12 上 sysfs 接口测量的解码功耗（<5W），归一化能量消耗与 batch size 的关系。1.5B batch=8 的解码能量低于 3B batch=1，同时精度相当。
    - Prefill Throughput：vs GPU-based llama.cpp OpenCL backend 和 QNN FP16（reference）。
    - CPU/Memory Overhead：top/pmap 测量的 CPU 利用率（≤4 cores）和内存消耗（1.3 GiB for 1.5B, 2.4 GiB for 3B，含 dmabuf）。
    - Sensitivity：prompt length (512–4096 tokens) 对 decode throughput 的影响。

- 硬件平台是什么，配置是什么。
  - 三款 Android 设备，覆盖三代 Snapdragon NPU 架构：
    | 设备 | SoC | NPU 架构 | 关键约束 |
    |------|-----|----------|----------|
    | OnePlus Ace3 | Snapdragon 8 Gen 2 | V73 | 2 GiB NPU 虚拟地址空间限制（无法运行 ≥3B 模型） |
    | OnePlus 12 | Snapdragon 8 Gen 3 | V75 | 主要评估平台 |
    | OnePlus Ace5 Pro | Snapdragon 8 Elite | V79 | 最新 NPU 架构 |
  - SoC 内部：CPU（ARM Cortex）+ GPU（Adreno）+ NPU（Hexagon V73/V75/V79，含 HVX 向量单元 + HMX 矩阵单元 + 8 MiB TCM + 1 MiB L2 cache）
  - 额外 GPU（仅精度实验）：NVIDIA RTX3090

- 开源Serving框架是什么。修改了什么。
  - 开源框架：**llama.cpp**（GGML 生态）。论文基于 llama.cpp 实现 NPU backend，基准对比使用 llama.cpp OpenCL backend (commit 1caae7f) 作为 GPU baseline。
  - 修改内容（~7K lines C/C++ + inline assembly）：
    1. **GGML-HTP Backend**：新增 `ggml-htp` backend，注册为 llama.cpp 的一个计算后端。`-DGGML_HTP=ON` 启用。不兼容 OpenMP 线程（`-DGGML_OPENMP=OFF`）。
    2. **HTP-Ops-Lib**：独立算子库，编译为 Hexagon DSP 共享对象（Stub `libhtp_ops.so` + Skeleton `libhtp_ops_skel.so`）。Stub 在 CPU 侧（AArch64），Skeleton 在 NPU 侧（Hexagon DSP）。FastRPC 在 Stub-Skeleton 间建立调用。
    3. **FastRPC + rpcmem 通信层**：CPU 侧通过 `libcdsprpc.so`（Android vendor library）分配、映射、释放 rpcmem shared memory。NPU 侧通过 polling 接收计算请求（比默认 RPC 更低延迟）。手动 cache maintenance（CPU 写入后 invalidate NPU cache）。
    4. **NPU 算子实现**：FP16 FlashAttention（HMX GEMM + HVX LUT Softmax）、Dequantization GEMM（Q4_0/IQ4_NL/Q8_0）、LayerNorm、RoPE、激活函数等。
    5. **权重布局转换**：`convert_hf_to_gguf_htp.py` 将 HuggingFace 权重转换为 HMX tile 布局的 GGUF 格式。`REPACK_FOR_HVX=1` 量化时触发 HVX super-group coalesce。
    6. **Test-Time Scaling 集成**：支持 Best-of-N（CPU 侧 ORM scorer, Skywork-1.5B-PRM）和 Beam Search（CPU 侧 PRM scorer）的批处理 decode。
    7. **Power Management**：NPU 侧含电源管理模块，推理时管理 NPU 频率/电压状态。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源状态：**已开源**。主仓库 https://github.com/haozixu/llama.cpp-npu（MIT License）+ 算子库 https://github.com/haozixu/htp-ops-lib。EUROSYS '26。
  - 全过程（以 OnePlus 12 上 Best-of-N test-time scaling，Qwen2.5-1.5B IQ4_NL+Q8_0，batch=8, MATH500 推理为例）：

    ```
    === Step 1：环境准备 ===
    1. 构建 htp-ops-lib（参见 kernel调度 条目 Step 1）
    2. 构建 llama.cpp-npu（参见 kernel调度 条目 Step 2）
    3. 模型转换 + 量化部署（参见 kernel调度 条目 Step 3-4）

    === Step 2：运行 Best-of-N Test-Time Scaling ===
    4. adb shell
    5. export LD_LIBRARY_PATH=/data/local/tmp/llama.cpp:/vendor/lib64:/system/lib64
    6. export DSP_LIBRARY_PATH=/data/local/tmp/llama.cpp
    7. ./llama-cli -t 4 -fa \
         -m qwen2.5-1.5b.iq4_nl+q8_0-hmx.gguf \
         -f math_prompt.txt \           # MATH500 0-shot CoT prompt
         -n 512 --batch-size 8          # Best-of-N: batch=8 条并行采样路径

    === 系统内部执行流程（单个 decode step） ===

    ┌─────────────────────────────────────────────────────────┐
    │                     CPU 侧 (AArch64)                      │
    │                                                           │
    │  1. Tokenizer: 将 8 条路径各自的已生成 token 序列 →       │
    │     8 个 input_ids (batch=8)                             │
    │                                                           │
    │  2. Embedding Lookup:                                     │
    │     hidden_states = embedding_table[input_ids]            │
    │     shape: [8, hidden_dim]  (FP16)                       │
    │                                                           │
    │  3. 写入 shared memory (rpcmem):                          │
    │     memcpy(shm_act_buffer, hidden_states, ...)            │
    │     手动 invalidate NPU cache:                            │
    │       rpcmem_flush_cache(shm_act_buffer)                  │
    │                                                           │
    │  4. 写入 computation request 到 shared memory             │
    │     (op_type=TRANSFORMER, layers=0..27)                  │
    │                                                           │
    │  5. NPU 侧轮询检测到 request → 开始执行                   │
    │     (CPU 在此期间可做其他工作，如上一轮 sampler)          │
    └───────────────────┬───────────────────────────────────────┘
                        │ shared memory (rpcmem/dmabuf)
                        ▼
    ┌─────────────────────────────────────────────────────────┐
    │               NPU 侧 (Hexagon DSP, V75)                   │
    │                                                           │
    │  6. 轮询线程检测到 computation request                    │
    │                                                           │
    │  7. DMA prefetch:                                         │
    │     DDR → L2 cache: FP16 activations [8, hidden_dim]      │
    │     DDR → TCM: INT4 weights (预加载当前层)                │
    │                                                           │
    │  8. For layer_i = 0 to 27:                               │
    │                                                           │
    │     === Attention (NPU FlashAttention) ===                │
    │     8a. Q/K/V projection (Dequantization GEMM):          │
    │         HVX vlut16: INT4→FP16 dequantize weights         │
    │         HMX mxmem: 加载 activation tile + weight tile     │
    │         HMX: FP16 GEMM → Q/K/V [8, head_dim]             │
    │                                                           │
    │     8b. FP16 FlashAttention (Algorithm 1):                │
    │         对每个 KV tile:                                   │
    │           HMX: S = Q × K^T (FP16, FP32 accum)            │
    │           HVX: rowmax(S)                                 │
    │           HVX vgather: P = LUT[S-m] (64 KiB LUT in TCM)  │
    │           HVX: rowsum(P) (FP32 accum)                    │
    │           HMX: O += P × V (FP16, FP32 accum)            │
    │                                                           │
    │     === FFN (Dequantization GEMM) ===                     │
    │     8c. gate_proj: HVX dequant + HMX GEMM                │
    │     8d. up_proj: HVX dequant + HMX GEMM                  │
    │     8e. SiLU activation (HVX LUT-based)                  │
    │     8f. down_proj: HVX dequant (Q8_0, higher precision)  │
    │         + HMX GEMM                                       │
    │                                                           │
    │     8g. Residual Add + LayerNorm (HVX)                   │
    │                                                           │
    │  9. DMA writeback:                                        │
    │     TCM → DDR: output hidden_states [8, hidden_dim]      │
    │     + 手动 invalidate CPU cache (如需要)                  │
    │                                                           │
    │  10. 更新 shared memory request status → DONE            │
    └───────────────────┬───────────────────────────────────────┘
                        │ shared memory
                        ▼
    ┌─────────────────────────────────────────────────────────┐
    │                     CPU 侧 (继续)                         │
    │                                                           │
    │  11. CPU 轮询检测到 DONE → 读取 output hidden_states     │
    │                                                           │
    │  12. lm_head (CPU 侧, 因 NPU 地址空间限制):               │
    │      logits = hidden_states × lm_head_weight^T            │
    │      shape: [8, vocab_size]                              │
    │      (batch=16 时 lm_head 耗时占比接近/超过 50%)         │
    │                                                           │
    │  13. Sampler (CPU 侧):                                    │
    │      temperature sampling → 8 个 next_tokens             │
    │      追加到各路径的生成序列                               │
    │                                                           │
    │  14. 检查终止条件 (EOS token 或 max_tokens)              │
    │      若未终止 → 回到 Step 1（下一 decode step）          │
    └─────────────────────────────────────────────────────────┘

    === Step 3：Best-of-N Scoring ===
    15. 8 条路径全部生成完成后（各 ≤512 tokens）：
        使用 Skywork-1.5B-PRM (CPU 侧或云端) 对每条路径评分
        选最高分路径作为最终输出

    16. 评估：compare 最终答案 vs MATH500 ground truth → pass@1

    === 关键系统特性 ===
    - Batch decode 使得 HMX tile [32,32] 利用率大幅提升：
      batch=1: activation [1, hidden_dim] → tile [1,32] 仅 1/32 有效行
      batch=8: activation [8, hidden_dim] → tile [8,32] 8/32 有效行
      但是因为 GEMM latency 中 HMX 部分受 activation tile 有效行数影响小，
      所以 batch 增加时 decode 吞吐显著提升。
    - lm_head 在 CPU 侧：batch=16 时 CPU lm_head 计算占比 ≈50%，
      论文指出解决 NPU 地址空间限制后将大幅改善 scaling 特性。
    - 总内存：1.5B 模型 ~1.3 GiB（含 1056 MiB dmabuf），3B 模型 ~2.4 GiB（含 2090 MiB dmabuf）。
    - CPU 利用率：≤4 cores，随 batch size 增长（lm_head 计算增加）。
    - 解码功耗：<5W 整机（1.5B），~4.3W（3B）。
    ```

## OpenJarvis

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：**OPENJARVIS 五原语架构**中的 **Engine、Agents、Tools & Memory** 三个可编辑原语（primitive）的联合优化。将个人AI系统分解为五个类型化接口（Intelligence/Engine/Agents/Tools & Memory/Learning），通过 *spec*（类型化配置对象）暴露 Engine 后端选择（Ollama/vLLM）、Agents 推理循环（ReAct/CodeAct/single-turn）、Tools 描述与 Memory 后端（SQLite-FTS5/FAISS/ColBERTv2/BM25/hybrid RRF）等作为独立可优化自由度。LLM-guided spec search 使用云端前沿模型在搜索时诊断失败痕迹、提出跨原语的协调编辑，由 held-out gate 仅接受非退化编辑，推理时完全在本地执行。
  - 实验比较：
    - **可移植性三角验证**（Table 1）：在 OpenClaw 和 Hermes Agent 两个生产框架上，将云端模型替换为 Qwen3.5-9B 并保持其余配置固定（条件b），与使用 OPENJARVIS spec 重新配置 Engine/Agent/Tools 但 Intelligence 固定为 Qwen3.5-9B（条件c）对比。spec 恢复了 PinchBench 上 77% 的下降、GAIA 上 56-57% 的下降。
    - **本地-云端精度前沿**（Table 5, Figure 5）：11 个本地模型 × 4 个模型家族 × 7 个硬件平台 vs 3 个云端 baseline（Claude Opus 4.6, GPT 5.4, Gemini 3.1 Pro），8 个 benchmark（508 任务）。最佳单机本地 spec Qwen3.5-122B 达到 80.3% 平均精度，距最佳云端 3.2 pp。
    - **LLM-guided spec search**（Figure 6-8, Table 9）：4 个学生模型 × 3 个教师模型 × 8 benchmarks。与 prompt-only（DSPy/SIMBA, GEPA）和 weight-only（SFT, LoRA）基线对比。与 proposal 消融（template-random, evolutionary spec search）和 move-space 消融对比。
    - **成本-延迟-能耗 Pareto 前沿**（Figure 5, Table 8）：本地 spec 边际 API 成本 ~800× 低于云端，端到端延迟 ~4× 低。

- 硬件平台是什么，配置是什么。
  - 7 个硬件平台，覆盖 4 个厂商（Table 6）：
    - **Consumer**：Mac Mini M4（16-32 GB unified memory, \$999）、AMD Radeon RX 9070 XT（16 GB GDDR6, \$599）
    - **AI-focused Discrete**：Intel Arc Pro B70（32 GB GDDR6, 367 INT8 TOPS, \$949）
    - **Prosumer/Workstation**：AMD Ryzen AI Max+ 395（128 GB LPDDR5X, \$1,999）、Mac Studio M4 Max（36-128 GB unified, \$3,499）、NVIDIA RTX 6000 Pro（96 GB GDDR7, 24,064 CUDA cores, \$8,900）
    - **AI Workstation**：NVIDIA DGX Spark（Grace-Blackwell Superchip, 128 GB LPDDR5X, 1 PFLOPS, \$4,699）
  - 能耗测量使用厂商 API：NVML（NVIDIA）、ROCm SMI（AMD）、xpu-smi（Intel）、powermetrics（Apple）。

- 开源Serving框架是什么。修改了什么。
  - 开源框架：**OPENJARVIS 自身**。论文声明开源（https://github.com/openjarvis/openjarvis），但链接当前返回 404（可能尚未公开或已变更）。网站：https://open-jarvis.github.io/OpenJarvis/。
  - 基于 vLLM 和 Ollama 作为 Engine 后端；Engine 原语抽象了后端切换、batch size、KV-cache 配置。
  - 修改/设计内容：
    1. **Engine 原语**：硬件检测模块自动识别加速器类型并推荐 Engine。支持 backend 切换（Ollama ↔ vLLM）、batch size、KV-cache 设置、量化 kernel 选择。
    2. **Agents 原语**：离散 Agent（single-turn chat, multi-turn orchestration with function calling, ReAct loop, CodeAct-style code execution）+ 连续 Agent（持久运行、跨会话状态维护）。Agent type、system prompt、few-shot exemplars、turn limits、verification steps、tool-calling strategy 均可编辑。
    3. **Tools & Memory 原语**：内置 7 类工具（reasoning/math/code execution/web search/file I/O/memory/inference delegation）。25+ Connectors（Gmail, Calendar, Apple Health, iMessage, Notion, Slack 等），32+ Channels（WhatsApp, Telegram, Discord 等）。Memory 后端可互换（SQLite/FTS5, FAISS, ColBERTv2, BM25, hybrid RRF）。
    4. **Learning 原语**：LLM-guided spec search 作为优化器，云端教师诊断失败→提出跨原语编辑→held-out gate 验证→仅接受非退化编辑。支持 LoRA/GRPO 权重更新、DSPy prompt 优化、structured editing。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源状态：论文声明开源，代码仓库 https://github.com/openjarvis/openjarvis 当前 404；网站 https://open-jarvis.github.io/OpenJarvis/ 论文未明确说明是否可访问。
  - 全过程（以 Qwen3.5-9B 本地 spec 执行 PinchBench agent 任务为例，Mac Mini M4 24 GB）：
    1. **Spec 加载**：从 TOML 配置文件加载 spec——Intelligence=Qwen3.5-9B FP16、Engine=Ollama、Agent=ReAct loop（max 10 turns, tools=think/calc/web_search）、Tools=SQLite FTS5 memory、Learning=disabled（推理时）。五个原语通过线程安全的 publish-subscribe EventBus 通信。
    2. **用户查询输入**：用户通过 Channel（如 WhatsApp/Telegram/Discord）发送自然语言查询，经安全层扫描（PII/credential 检测、prompt injection 检测）后进入 Agent loop。
    3. **Agent 推理循环**：ReAct agent 解析用户意图，决定是否调用工具。若需 web_search，通过 Tools 原语调用 Google Custom Search / Tavily / Brave API（\~\$0.005–0.01/query）。若需 memory_search，通过 Memory 原语（SQLite FTS5）检索相关历史对话。若需代码执行，通过 code_interpreter 在隔离 Docker 容器中运行。
    4. **Engine 推理执行**：Agent 构造的 prompt 发送至 Engine 原语。Engine 根据 spec 配置选择 Ollama backend，以 batch size 1（交互式 workload）、FP16 精度在 Mac Mini M4 的 GPU（10-core）和 Neural Engine（16-core）上执行 Qwen3.5-9B 推理。prefill ~89.9K tokens/s, decode ~9.5 ms/tok, 每 query 能耗 ~27.2 KJ。
    5. **结果返回与遥测记录**：模型输出经安全层扫描后返回用户。instrumented wrapper 自动记录 accuracy、energy（J）、latency（s）、power（W）、dollar cost（\$0 marginal API cost for local inference）。
    6. **（搜索时）LLM-guided spec search**：若 Learning 启用，云端教师（如 Claude Opus 4.6）读取脱敏后的 trace 数据→diagnose 失败模式→propose 跨原语编辑（如修改 tool description + 调整 prompt + 切换 Engine backend）→held-out gate 验证→仅接受无退化编辑→更新 spec。搜索结束后推理完全在本地执行，零云端调用。

## Rethinking DVFS for Mobile LLMs: Unified Energy-Aware Scheduling with CORE

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：**CORE 统一能量感知 DVFS 调度器**——在 llama.cpp 推理框架上实现的统一 CPU/GPU/内存频率协调调度器（~2K 行 Python 代码）。核心设计包含三个组件：
    1. **离线 Profiling-based 频率搜索**：安装时对每个模型进行一次性的频率组合搜索。将 prefill 长度划分为 5 个 range，加上 1 个 decode 长度设定，对每个设定在候选频率组合下运行推理，测量延迟与能耗。搜索空间从穷举的 2808 种组合（CPU 18 级 × GPU 12 级 × Memory 13 级）经两阶段启发式缩减至平均 14.5–30.8 次推理/模型（减少 374×）。
    2. **"GPU First, CPU Next" 两步启发式搜索**：基于两点发现设计——(a) GPU 频率是影响推理延迟与能耗的主导因素；(b) 默认 CPU/GPU governor 级联式互相压低频率。Step 1：固定 GPU 在候选频率（从高到低搜索），找到满足延迟/能耗预算的最优 GPU 频率；Step 2：在 Step 1 选定的 GPU 频率（最多 2 个）下微调 CPU 频率。Memory governor 保留默认（实验表明 memory governor 接近最优）。
    3. **运行时频率钉扎**：推理时根据当前 prefill 长度和 decode 阶段查表获取预计算的最优频率组合，通过 Android sysfs 接口 `echo <freq> > /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq` 等钉扎 CPU/GPU/内存频率。在 prefill 和 decode 阶段切换时切换频率组合（prefill 和 decode 使用不同最优频率）。
  - 实验比较：
    - **Baseline**：默认 DVFS governor（Pixel 7 默认：CPU=sched_pelt/EAS、GPU=quickstep、Memory=interactive）。对比 pinned 最优频率组合（穷举搜索的上界）。
    - **指标**：TTFT (Time-To-First-Token)、TPOT (Time-Per-Output-Token)、E2E Latency、Energy per Token、Energy-Delay Product (EDP)
    - **Workload**：ShareGPT 200 条随机请求（prefill ≤512 tokens, decode ≤256 tokens），平均 prompt 232.4 tokens，平均 decode 70.0 tokens
    - **模型**：TinyLlama 1.1B、DeepSeek-R1-Distill-Qwen 1.5B、Refact-fim 1.6B、StableLM-Zephyr 2.7B、Phi-2 2.7B、Llama-2 6.7B（全部 4-bit 量化）
    - **消融研究**：Finding 1（默认 governor 远非最优：23.0–40.4% 延迟更长 / 5.0–16.6% 能耗更高）、Finding 2（各 governor 独立选择过低频率）、Finding 3（CPU/GPU governor 拮抗交互——decode 低利用率触发"向下螺旋"）、CORE vs. 默认 governor 的 TTFT/TPOT/Energy 对比

- 硬件平台是什么，配置是什么。
  - **手机平台**：Google Pixel 7 / Pixel 7 Pro
  - **SoC**：Google Tensor G2（Samsung 5nm 工艺）
  - **CPU**：2× Cortex-X1 (2.85 GHz) + 2× Cortex-A78 (2.35 GHz) + 4× Cortex-A55 (1.80 GHz)，共 18 级频率（500–2850 MHz）
  - **GPU**：ARM Mali-G710 MP7，共 12 级频率（151–848 MHz）
  - **Memory**：LPDDR5，共 13 级频率（421–3172 MHz），总组合数 = 18×12×13 = 2808
  - **电源测量**：电池旁路，通过 **Monsoon Power Monitor** 供电并每 0.2 ms 记录细粒度功耗（电压+电流），屏幕关闭
  - **OS**：Android，adb 不可用期间使用 profiling daemon 运行 benchmark 脚本
  - **推理后端**：llama.cpp (tag b2202) + OpenCL（GPU compute via OpenCL）

- 开源Serving框架是什么。修改了什么。
  - 开源框架：**llama.cpp** (tag b2202)，基于 GGML 的轻量级 C++ LLM 推理框架，OpenCL 后端实现移动端 GPU 推理。
  - 修改内容（~2K 行 Python 扩展）：
    1. **Profiling 模块**：实现自动化频率组合搜索脚本。对每个模型遍历 5 个 prefill length range + 1 个 decode length setting，在每个 setting 下使用 sysfs 切换 CPU/GPU 频率，运行推理并记录 Monsoon power monitor 的功耗数据 + 推理延迟。
    2. **CORE Governor 运行时**：在 prefill/decode 阶段开始/结束时，从预计算的频率 lookup table 中查表获取最优 CPU/GPU 频率组合，通过 sysfs 钉扎频率（scaling_max_freq）。prefill 和 decode 阶段使用不同的最优频率——prefill 偏好更高 GPU 频率（compute-bound 大矩阵乘），decode 偏好适中 CPU/GPU 频率（memory-bound，高频率浪费能量）。
    3. **Hook 注入**：在 llama.cpp 推理循环中注入阶段通知 hook（prefill 开始/结束、decode 每步开始/结束），触发频率切换。论文未修改 llama.cpp 的 C++ 核心代码，外部 Python 脚本通过文件/信号与 llama.cpp 进程通信。
    4. **Monsoon 集成**：Python 脚本与 Monsoon power monitor USB 通信，同步推理阶段时间戳与功耗采样（0.2 ms 粒度），计算每阶段平均功耗和 Energy per token。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源状态：论文声明已开源（"released FUSE/CORE as an extension to the llama.cpp framework"），但论文正文和 MLSys 页面**未提供具体 GitHub URL**。arXiv 预印本 (arxiv.org/abs/2507.02135) 也未提供代码链接。可通过作者 Zongpu Zhang (Purdue) 的 GitHub 主页查找。
  - 全过程（以 Pixel 7 Pro + Tensor G2 推理 TinyLlama 1.1B Q4，ShareGPT 一条请求，prefill=200 tokens, decode=100 tokens 为例）：

    ```
    === Step 0：安装时 Profiling（一次性，per model） ===
    0a. 硬件准备：
        - Pixel 7 Pro 电池物理旁路，通过 Monsoon Power Monitor 供电
        - USB 连接 Monsoon → 主机（Python 脚本控制）
        - 屏幕关闭，后台服务最小化
    
    0b. 频率搜索空间定义：
        - CPU: 18 levels [500, 574, ..., 2850] MHz (scaling_available_frequencies)
        - GPU: 12 levels [151, 200, ..., 848] MHz
        - Memory: 13 levels [421, 547, ..., 3172] MHz
        - 穷举空间: 18×12×13 = 2808 combinations
    
    0c. Profiling 执行（CORE 两阶段搜索）：
        prefill_ranges = [[1,32], [33,64], [65,128], [129,256], [257,512]]  # 5 buckets
        decode_setting = 1  # single decode setting
        
        for each prefill_range in prefill_ranges:
            for each decode_setting:
                # Step 1: GPU First
                for gpu_freq in [848, 806, ..., 151]:  # high → low
                    pin_frequencies(gpu=gpu_freq, cpu=max, mem=default)
                    run_inference(model, prefill_range, decode_setting)
                    latency, power = record_monsoon()
                    if meets_budget(latency, power):
                        candidate_gpu = gpu_freq
                        break
                
                # Step 2: CPU Next (pin GPU at candidate, search CPU)
                for cpu_freq in [2850, 2650, ..., 500]:  # high → low
                    pin_frequencies(gpu=candidate_gpu, cpu=cpu_freq, mem=default)
                    run_inference(model, prefill_range, decode_setting)
                    latency, power = record_monsoon()
                    if pareto_optimal(latency, power):
                        store_to_lut(prefill_range, decode_setting, gpu_freq, cpu_freq)
        
        总搜索量：2.4+5.1 = 7.5 次推理/setting × 6 settings = ~45 次推理/model
        （vs 穷举 2808 次，减少 ~62×）
    
    === Step 1：运行时推理（CORE Governor 激活） ===
    1. 用户输入 prompt（200 tokens）
    
    2. llama.cpp 加载模型（TinyLlama 1.1B Q4_K_M, ~700 MB）
       - GGUF 格式，OpenCL backend
       - 模型权重加载至 unified memory (LPDDR5)
    
    3. Python CORE daemon 启动，监听推理阶段信号：
    
    ==== Phase A: Prefill ====
    3a. llama.cpp 发送 "prefill_start" 信号 → CORE daemon
    3b. CORE daemon 查表（prefill_len=200 → bucket [129,256]）：
        optimal_gpu = 806 MHz   # prefill 需高 GPU 频率（大矩阵乘）
        optimal_cpu = 2100 MHz  # CPU 需足够频率喂 GPU 工作队列
    3c. sysfs 钉扎：
        echo 806000 > /sys/class/kgsl/kgsl-3d0/devfreq/max_freq    # GPU
        echo 2100000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq  # CPU
    3d. llama.cpp 执行 prefill forward pass：
        - GPU (Mali-G710, OpenCL): 执行 attention + FFN GEMM（INT4 matmul kernel）
        - CPU (Cortex-X1/A78): tokenization + 喂 GPU 命令队列 + 管理 KV cache
        - Memory (LPDDR5 @ default governor): 模型权重读取 + KV cache 读写
    3e. Monsoon 记录 prefill 阶段功耗（每 0.2 ms 采样）→ 计算 TTFT + Energy
    
    ==== Phase B: Decode (逐 token, 100 tokens) ====
    4. llama.cpp 发送 "decode_start" 信号 → CORE daemon
    5. CORE daemon 查表（decode）：
        optimal_gpu = 506 MHz   # decode 为 memory-bound, 低 GPU 足够
        optimal_cpu = 1400 MHz  # decode 低利用率, 中频 CPU 避免浪费
    6. sysfs 重新钉扎：
        echo 506000 > /sys/class/kgsl/kgsl-3d0/devfreq/max_freq
        echo 1400000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
    
    7. For token_i = 1 to 100:
       7a. llama.cpp decode step：
           - GPU: 1-token attention (memory-bound, O(L×d) 而非 O(L²×d))
           - CPU: sampling + KV cache 管理
       7b. Monsoon 记录每步功耗
       7c. 输出 token
    
    === Step 2：结果汇总 ===
    8. 指标计算：
       TTFT = t_prefill_end - t_prompt_arrival
       TPOT = mean(t_decode_token_i) for i=1..100
       E2E Latency = TTFT + Σ TPOT_i
       Energy_per_prefill_token = P_prefill × TTFT / N_prefill
       Energy_per_decode_token = mean(P_decode_i × TPOT_i)
    
    9. 对比 Baseline（默认 governor）：
       默认 governor 行为（无 CORE）：
       - CPU EAS governor 检测 GPU 执行时 CPU 低利用率 → 降频至 ~800 MHz
       - GPU quickstep governor 检测 CPU 慢喂队列 → 降频至 ~300 MHz
       - 两者互相压低 → "向下螺旋"：CPU 800 MHz + GPU 300 MHz
       - 结果：TTFT 显著增加（GPU compute 变慢），TPOT 大幅延长
       
       CORE 行为：
       - Prefill：GPU 806 MHz + CPU 2100 MHz → TTFT 减少 8.5-17.7%
       - Decode：GPU 506 MHz + CPU 1400 MHz → TPOT 减少 27.8-39.6%
       - Energy per token 无增加（解码阶段低频率补偿了 prefill 高频率）
    
    === 关键硬件数据路径（Pixel 7 Pro Tensor G2） ===
    LPDDR5 (unified memory, ~51.2 GB/s)
      │ model weights (Q4 GGUF, ~700 MB for TinyLlama 1.1B)
      ├─► ARM Mali-G710 MP7 GPU (OpenCL)
      │     │ INT4 matmul kernel (attention QK^T + FFN GEMM)
      │     │ 848 MHz max → 506 MHz (decode, memory-bound 不需要高频)
      │     └─► Output activations → back to LPDDR5
      ├─► CPU (Cortex-X1/A78)
      │     │ Tokenization, KV cache management, sampling
      │     │ 2850 MHz max → 1400-2100 MHz (CORE selected)
      │     └─► OpenCL command queue → GPU
      └─► Monsoon Power Monitor (external, USB)
            │ 每 0.2 ms 采样 V + I
            └─► Python daemon 同步时间戳 + 计算 Energy
    ```
