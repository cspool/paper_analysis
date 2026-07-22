## LLM as a System Service on Mobile Devices

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：LLMS，一个面向移动设备的 LLM-as-a-System-Service (LLMaaS) 原型系统，核心理念是将 LLM 作为系统级服务暴露给所有 app 共享使用（类似 Android 的 Location Service），而非每个 app 各自加载 LLM。LLMS 的核心是实现高效的 LLM context 内存管理，解耦 app 内存与 LLM context 内存的管理，通过细粒度、chunk-wise、全局优化的 KV cache 压缩与交换来最小化 context switching 延迟。LLMS 包含三个关键技术：(1) Tolerance-Aware Compression（容忍度感知压缩）——利用 attention scores 计算每个 chunk 的信息密度来决定压缩率，在保证全局平均压缩比的前提下最大化整体信息强度；(2) Swapping-Recompute Pipeline（交换-重计算流水线）——将部分 chunk 的重计算与其余 chunk 的磁盘 I/O 以流水线方式重叠执行，修改 LLM 的 position encoding 和 causal mask 以支持不连续 chunk 的重计算；(3) Chunk Lifecycle Management（chunk 生命周期管理）——使用 LCTRU (Least Compression-Tolerable and Recently-Used) 队列决定 eviction 优先级，采用 ahead-of-time (AoT) swapping-out 在 callLLM() 返回阶段提前换出已修改 chunk 以隐藏回收延迟。
  - 实验比较：(1) 端到端 context switching latency 对比 LLMS vs. LMK（Android low-memory killer）、Swapping（整上下文交换）、VLLM-S（chunk-wise KV cache 管理无压缩）、VLLM-SQ（chunk-wise + 统一 INT8 量化），在 72 小时合成 trace 上评估 2/4/6/8/12/16 个 active contexts 的切换延迟；(2) 不同 memory budget（1GB/2GB/3GB）下 max number of active contexts 对比；(3) 不同 maximal context length（256-4096 tokens）下 active contexts 数量对比；(4) 三种 context switching pattern（Random/Markov/Gaussian）下的性能一致性；(5) 压缩效率对比：LLMS tolerance-aware compression vs. 静态均匀量化（4-bit/2-bit）的 accuracy-compression ratio trade-off；(6) 消融实验：逐步移除 tolerance-aware compression / swapping-recompute pipeline / chunk lifecycle management 的影响；(7) chunk size 选择实验（不同 token/chunk 数下 switching latency 变化）；(8) LLM 推理性能稳定性分析（LLMS 对正常推理的影响）；(9) Service calling frequency 敏感性分析。

- 硬件平台是什么，配置是什么。
  - Jetson Orin NX：8 GB RAM，NVMe SSD，1024-core Ampere™ GPU
  - Jetson TX2：8 GB RAM，SATA HDD（磁盘带宽较低），256-core Pascal™ GPU
  - MI14 Smartphone：8 GB RAM，UFS 4.0 存储，Hexagon™ 8Gen3 NPU（Qualcomm Snapdragon 8 Gen 3）
  - 注：所有设备均为 8 GB RAM，LLMS 在 TX2 上因 SATA HDD 低带宽导致整体 switching latency 更长，但仍显著优于 baseline

- 开源Serving框架是什么。修改了什么。
  - LLMS 不是基于现有 serving 框架修改，而是自建的 LLM Service 原型（3.5k LoC Python/C++），构建在 Huggingface Transformers [68] 和 mllm [53] 之上（分别用于 Jetson 设备和 MI14 智能手机）。LLMS 作为独立进程运行，通过 socket IPC 接收来自客户端进程的推理请求。
  - LLMS 修改/新增的核心模块：
    (1) Context Memory Management Module：嵌入 LLM Service 内部，实现 claim/reclaim/load/fault 四个内存操作原语——Claim 直接分配空闲内存给 chunk；Reclaim 在内存压力下将 chunk 换出到磁盘；Load 在调用 callLLM() 前将缺失 chunk 从磁盘加载到内存；Fault 在每次 LLM 推理迭代时按需加载缺失 chunk（保留用于异常处理如系统崩溃）。
    (2) Tolerance-Aware Compression：在 LLM 推理框架现有 KV cache 量化（如 LMDeploy 的 INT8）之上，对低信息密度 chunk 进一步执行 channel-wise 线性量化（4-bit/2-bit），使用并行 bit-shift 操作将 sub-byte 数据打包为 INT8 格式。
    (3) Swapping-Recompute Pipeline：多线程实现——独立 I/O 线程从磁盘加载 chunk 到内存，计算线程在当前层 I/O 完成后才进入下一层（流水线同步），修改 position encoding（全局编码不连续 token）和 causal mask 以支持不连续 chunk 的正确重计算。
    (4) Chunk Lifecycle Management：LCTRU 队列由多个按压缩率分组的 LRU 子队列串联组成，AoT swapping-out 在 callLLM() 返回阶段将已修改 chunk 换出。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文未明确说明 LLMS 自身代码是否已开源。论文声明将公开实验使用的 context switching traces。LLMS 依赖的开源组件：Huggingface Transformers (https://github.com/huggingface/transformers)、mllm (https://github.com/UbiquitousLearning/mllm)、GPTQ (https://github.com/IST-DASLab/gptq)、Pickle (Python/C++)、Llama2-7B (https://huggingface.co/meta-llama/Llama-2-7b)、OPT-6.7B (https://huggingface.co/facebook/opt-6.7b)。

  - **LLMS Serving 框架输入到硬件执行全过程（以 MI14 上 Llama2-7B 的 context switching 为例）**：

    **阶段 0：LLM Service 初始化与 LLMS API**
    ```
    # LLMS 定义四类 API（兼容 Android Service 模式）

    Class LLMService:              # 系统级 LLM 服务类，类似 Android Service
    Class LLMCtx:                  # LLM 上下文类，封装状态
    Method bindLLMService(app):    # App 绑定 LLM Service
    Method newLLMCtx(sysPrompt):   # 创建新 LLM 上下文（返回 LLMCtxStub）
    Method callLLM(ctx, prompt):   # 调用 LLM（输入上下文 + 新 prompt，返回输出 + 更新上下文）
    Method delLLMCtx(ctx):         # 删除上下文

    # 配置参数（OS 可配置）：
    # - K: 每个 app 最大 active context 数
    # - max_context_length: 每个 context 最大 token 数（Llama2=4K, OPT=2K）
    # - ratio_global: 全局平均 KV cache 压缩比（默认 50%）
    # - chunk_size: 每个 chunk 包含的 token 数（默认 16）
    # - {ratio_w}: 可用压缩级别 {8/8, 4/8, 2/8}（即 INT8/INT4/INT2）
    ```

    **阶段 1：App 调用 LLM Service（以 Chatbot 为例）**
    ```
    输入: App 发送 socket IPC 请求到 LLM Service 独立进程
          ctx = "历史对话 KV cache", newPrompt = "What's the weather today?"

    1. LLMS 接收 callLLM(CtxStub, newPrompt):
       - 解析 CtxStub → 定位磁盘上的 chunk 状态
       - 触发 Load primitive: 将缺失 chunk 从磁盘加载到内存

    2. 内存布局（LLMS Memory Model）:
       Context 分为两部分:
         Memory-resident fragment: prompt/output text（不可换出）
         Swappable fragment: KV cache chunks（可压缩、可换出）
       KV cache layout: token 维度增长, 每 chunk = 16 tokens × 所有层 × 所有 head
       e.g. Llama2-7B, 4K tokens: 256 chunks per context
            每个 chunk 原始大小 ≈ 2GB / 256 ≈ 8 MB（INT8 后 ~8 MB, INT4 后 ~4 MB）
    ```

    **阶段 2：Context Switching — Load 阶段（三种技术协同工作）**

    ```
    # === 子阶段 2A: Tolerance-Aware Compression 决定 chunk 压缩率 ===
    # （在 callLLM() 之前的 token generation 阶段已完成，此处说明其原理）

    对 context 中每个 chunk_i（16 tokens）:
      1. 计算 attention score matrix A [R×C]:
         A = softmax(mask(Q·K^T / sqrt(d_k)))
         对每列 col, 对每个 head h 和 layer l:
           计算 token-level 信息密度 = avg(attention scores paid to this token)
       
      2. 计算 chunk-level 信息密度 D_i (Equation 1):
         D_i = 1/(q-p) · Σ_col Σ_layer Σ_head avg(A[*, col] in head h, layer l)
         含义: 被更多其他 token "关注" 的 chunk 信息量更大, 压缩容忍度更低
       
      3. 排序: Rank_i = D_i 在所有 chunk 中的百分位排名
       
      4. 确定压缩阈值 {σ_ratio} (Equation 3):
         在全局平均压缩比 ratio_global=50% 约束下
         maximize ctxInfo = Σ_w (1/ratio_w) · Σ_{chunk in [σ_{w+1}, σ_w]} D_i
         → σ_8/8 (top 排名 chunk: 保持 INT8, 不额外压缩)
         → σ_4/8 (中间排名: 从 INT8 再压到 INT4)
         → σ_2/8 (低排名: 从 INT8 再压到 INT2)

    结果: e.g. 256 chunks 中: top ~30% 保持 INT8, middle ~40% 压到 INT4, bottom ~30% 压到 INT2
          总内存: 256×8MB×(0.3×1 + 0.4×0.5 + 0.3×0.25) = 256×8×0.575 ≈ 1178 MB
          vs 不压缩 2048 MB → 节省 ~43% 内存, 无明显精度损失
    ```

    ```
    # === 子阶段 2B: Swapping-Recompute Pipeline 加载缺失 chunk ===
    # 假设 callLLM() 触发时, 目标 context 有 60% chunk 已在内存, 40% 在磁盘

    1. Profiling (安装时 offline 一次性):
       对当前设备测量:
         T_re(x, f, e): 重计算 x 个 chunk 的延迟（函数: chunk 数, CPU/GPU 频率, 能耗模式）
         T_IO(m): 从磁盘加载 m MB 的延迟
       实践中用线性函数近似: 离散测试点拟合

    2. Planning (Equation 4, 线性规划求解):
       minimize pipelineDelay = max(T_re(Σx_re), T_IO(m_onload - Σratio_w · x_re))
       s.t. x_re^{ratio_w} < x^{ratio_w}  (每种压缩率的重计算 chunk 数不可超过该率实际缺失 chunk 数)
       
       给定: m_onload = 40% × 1178 MB ≈ 471 MB（需加载的总内存大小）
             {x^{8/8}=20, x^{4/8}=50, x^{2/8}=30}（各压缩率下缺失的 chunk 数）
       
       求解: 决定重计算哪些 chunk 来最大化 I/O 与 recompute 的重叠
             优先重计算压缩率低的 chunk（更少数据需要从磁盘读取）

    3. Pipeline 执行 (多线程):
       I/O Thread:           |==Load L0 K/V==|==Load L1 K/V==|==Load L2 K/V==| ...
       Compute Thread:  |==Recompute L0==|==Recompute L1==| ...
       同步条件: Compute thread 在当前层 I/O 完成后才进入下一层

    4. Chunk Recompute Procedure (处理不连续 chunk):
       例如 context 文本 "a b c d e f" 中 c 和 e 的 KV cache 被换出:
         → Embed "c" 和 "e" 为 token → 全局 position encoding (pos_c=3, pos_e=5)
         → 重计算 Q, K, V → 插入到已有 K/V 的对应位置
         → 应用 causal mask: c 只能看到 a b c, e 只能看到 a b c d e
         → 每层重计算后进入下一层 pipeline
    ```

    ```
    # === 子阶段 2C: Chunk Lifecycle Management 决定 eviction ===
    # （在 callLLM() 返回阶段和后续内存压力时触发）

    1. AoT (Ahead-of-Time) Swapping-out:
       callLLM() 的 returning stage:
         识别所有在本次推理中被修改的 chunk → 立即写入磁盘
         即使当前无内存压力也执行（writeback 而非 eviction）
         延迟对 caller 不可感知（发生在 token 生成完成后到返回前）

    2. LCTRU Queue Eviction (当 Reclaim primitive 被触发):
       LCTRU queue 结构:
         Q_{LCTRU} = [Q_{8/8}] → [Q_{4/8}] → [Q_{2/8}]
         每个子队列按最近访问时间排序（LRU）
         头部的子队列（低压缩率 → heavy chunk）优先被 evict
       
       Eviction 决策（需要释放 M bytes 时）:
         从 Q_{8/8} 头部开始 pop → 换出到磁盘
         若 Q_{8/8} 空 → pop Q_{4/8}
         若 Q_{4/8} 空 → pop Q_{2/8}
       
       设计原理:
         - Heavy chunk first: 重计算 pipeline 中, 更少的 chunk 数 → 更低的重计算延迟
         - LRU within same ratio: 利用 context 访问的时间局部性
       
    3. Working Set Lock:
       callLLM() 执行期间, LLMS 锁定当前 context 的所有 chunk
       → 禁止 Reclaim 回收自己的 chunk（避免 thrashing）
       → Fault primitive 保留用于异常处理（如系统崩溃后恢复）
    ```

    **阶段 3：LLM 推理执行与 Token Generation**
    ```
    所有 chunk 就绪后, LLM 正常执行自回归推理:
      Prefill phase: 新 prompt tokens → 生成新 KV cache
      Decode phase:  逐 token 生成
      LLMS 不干预 LLM 推理过程本身, 仅负责推理前的 context 准备
    
    LLMS 使用的推理配置:
      - 权重量化: GPTQ W4A16 (4-bit INT)
      - KV cache 默认: INT8 (SmoothQuant 类方法)
      - LLMS 额外压缩: chunk-wise 4-bit/2-bit (基于 tolerance-aware)
      - 滑动窗口 attention: streaming LLM [71]
      - 框架: Jetson → HuggingFace Transformers + PyTorch; MI14 → mllm
    ```

    **端到端 Context Switching 示例（MI14, Llama2-7B, 8 active contexts, Markov pattern）**：
    ```
    App 发送 callLLM(ctx_id=5, "Summarize the previous emails about Q3 budget")
    ↓
    [LLMS Load Phase]:
      - 识别 ctx_5 当前状态: 256 chunks 中 150 在内存, 106 在磁盘
      - LCTRU queue: 决策是否需先 evict 其他 context 的 chunk（若内存不足）
      - AoT: 前一次 callLLM() 返回时 ctx_5 的修改 chunk 已在磁盘（无需额外 I/O）
    ↓
    [LLMS Swapping-Recompute Pipeline]:
      - Planning: T_re vs T_IO 分析 → 决定重计算 40 个 INT4/INT2 chunk + I/O 加载 66 个
      - Pipeline 执行: I/O 和 recompute 重叠 → ~0.27s (vs. Swapping baseline ~27s)
    ↓
    [LLM Inference]:
      - Prefill: new prompt tokens → 生成新 KV cache chunk
      - Decode: 逐 token 生成 "The Q3 budget emails discuss..."
      - 新生成的 KV cache chunk 按 tolerance-aware compression 压缩
    ↓
    [LLMS AoT Writeback]:
      - callLLM() 返回前: ctx_5 新修改 chunk → 写入磁盘
      - LCTRU queue 更新: ctx_5 的 chunk 移到各自子队列末尾
    ↓
    返回: 生成的 tokens + 更新后的 LLMCtxStub 给 App
    总 context switching 延迟: ~0.27s（LLMS）vs. LMK recompute ~22.92s (85× reduction)
    ```

  - **关键性能数据**（72h trace, 平均 switching latency）：
    - LLMS vs LMK: up to 2 orders of magnitude reduction
    - LLMS vs Swapping: 1-2 orders of magnitude reduction
    - LLMS vs VLLM-SQ (chunk + INT8): up to 20×, average 9.7× reduction
    - 消融: 全部技术 → 0.27s; 去掉 lifecycle mgmt → 0.62s; 去掉 tolerance compression → 0.42s; 去掉 recompute pipeline → 1.62s
