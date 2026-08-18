## Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现是 Tetris，一个面向在线长上下文 LLM Serving 的调度系统，核心是 CDSP（Chunkwise Dynamic Sequence Parallelism，逐块动态序列并行）：把每个请求的 prompt 拆成多个 chunk，为每个 chunk 分配不同的 SP 大小（前段小 SP 用空闲资源碎片提前开跑、后段大 SP 满足长序列计算需求，类似俄罗斯方块填空隙），替代既有"整请求统一 SP 大小"的粗粒度分配。Tetris 由三个组件构成：①**CDSP 推理引擎**——在 prefill-decoding 解耦集群上把全部 prefill 实例连成统一 SP 组（小 TP），decoding 实例用大 TP 的 DP 部署；扩展 zigzag ring attention 支持跨 chunk 的负载均衡（历史 KV cache 在 chunk 实例组间均匀重分布，并复用 ring 通信器做跨层 cache balancing 重叠）；引入 handshake 式 backend 分配管理 prefill→decoding 的 KV cache 传输。②**CDSP 调度器**（C++，递归算法 Algorithm 1/2/3）——基于 FLOPs 延迟模型 T_s(R)=a_s+b_s·L+c_s·(C·L)+d_s·L²（最小二乘离线拟合）递归搜索最优 chunk 划分与实例分配；用 improvement rate（改进率阈值）按实时负载调控 SP 扩张程度，避免过度 SP 扩张。③**simulator 式 improvement rate profiler**（~2.1K 行 Python）——离线按请求长度分布 + Poisson 到达率模拟不同负载，为每个到达率选出最优 improvement rate，在线每 30 秒按观测到达率自刷新。
  - 实验比较：①压力测试（把真实请求 trace 的时间戳缩放模拟不同负载，归一化到 25× light-load 延迟）对比 LoongServe、LoongServe Disaggregated、Fixed-SP Scheduling（SP=8/16），比 TTFT/TBT 的 P50/P99 与最大可持续负载；②TTFT 分布分析（临界请求率下的累计 TTFT CDF）；③吞吐分析（TTFT 约束下的吞吐）；④消融：improvement rate 分析（不同固定 rate vs 动态 rate）、chunking 分析（CDSP vs single-chunk 调度）；⑤调度器开销分析（Algorithm 1 在 SP≤128 时平均/最大延迟 ≤86.8µs，端到端 ≤93.79µs/32.90µs for 8B/70B）；⑥cache transfer 开销（CDSP balancing ≤1.8% 额外开销，CDSP handshake 0.6%-11.8%）；⑦simulator 精度（性能模型误差 ≤7.64%/6.35%，模拟器误差平均 6.9%/2.5%）。主要结果：相对 SOTA baseline，TTFT 最高降低 4.35×（P99，LLaMA3-70B），median TBT 最高降低 40.1%，最大请求容量提升最高 45%（20%-45%），吞吐提升 1.24-3.38×/1.15-1.81×（8B/70B），P50 TTFT 降低 1.64-2.78×/2.86-4.17×。
- 硬件平台是什么，配置是什么。
  - A100 GPU 集群：每个节点 8× NVIDIA-A100-SXM4-80GB（NVLink 互联）、128 CPU 核、2TB 主机内存、8× 200Gbps InfiniBand NIC。LLaMA3-8B 部署在 4 节点（32 GPU），LLaMA3-70B 部署在 8 节点（64 GPU）。P/D 比例为 1:1；8B 模型 prefill TP=1、decoding TP=8，70B 模型全 TP=4。
- 开源Serving框架是什么。修改了什么。
  - 基于开源 Serving 框架 **vLLM**（控制面）与 **PyTorch + Triton-distributed**（推理后端），并复用部分 vLLM 组件；前端用 FastAPI；跨进程通信用 Ray。总实现 ~17.5K 行 C++/Python。修改/新增：①扩展 vLLM 调度器三个接口——initialize_schedule（初始化延迟模型与 improvement rate）、update_schedule（HTTP POST {service_url}/update 更新调度元数据：improvement_rate_mapping、sp_size_candidates、improvement_rate_update_period）、cdsp_schedule（对到达的 prefill 请求调用 Algorithm 1 生成 CDSP 执行计划，构造 per-instance 元数据转发给 local manager）；CDSP 调度器本体用 C++ 写（消除调度延迟），global manager 用 Python + Ray。②初始化分布式集群时（initialize_model_parallel）显式配置 prefill 统一实例池的 SP 大小建立 ring attention communicator，decoding 实例指定 DP 大小。③prefill 计算：扩展 Flash Attention 支持 zigzag ring attention（历史 token），用 NVSHMEM 降低 ring 通信开销；decoding 计算：Flash Decoding + CUDAGraph（消除 kernel launch 开销）。④KV cache balancing 与 prefill-decoding cache 传输用 NCCL（v2.26+ 支持并发 communicator 执行），预留专用 buffer 与 CUDA stream 提升带宽利用率。⑤decoding 调度器扩展 Llumnix 的 "virtual usage"：把正在 cache transfer 的请求的 KV cache 槽视为虚拟占用，新请求路由到 freeness rate（可用槽/活跃 batch 大小）最高的实例。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文未提供 Tetris 的开源代码或链接。其 arXiv 预印本（"Optimizing Long-context LLM Serving via Fine-grained Sequence Parallelism"，arXiv:2511.06247）已被撤回以符合所属机构内部发表政策，暂无公开 GitHub 仓库（截至检索时无法确认；注意与同名 ZhaoxuanWu/Tetris 的投机解码仓库区分）。可复用组件均开源：vLLM（https://github.com/vllm-project/vllm）、LoongServe（https://github.com/LoongServe/LoongServe，SOSP 2024 的 ESP baseline）、NCCL（https://github.com/NVIDIA/nccl）、Triton-distributed、Ray。vLLM 使用方式：在线 Serving 框架，接收 HTTP 请求→scheduler 决定执行计划→执行器在 GPU 上跑 prefill/decode；Tetris 在其 scheduler 上叠加 CDSP 执行计划求解。
  - 使用例子（一个 128k 长 prompt 请求到达 4 节点 A100 集群，vLLM/Tetris 框架输入到硬件执行全过程）：
    ```
    # 输入：HTTP 请求（FastAPI 前端收，127k token prompt + 已知长度分布），Poisson 流到达
    # 1) 前端（FastAPI）：POST 请求 → 解析为 prefill 请求 → 交给 CDSP scheduler
    # 2) CDSP scheduler（C++，控制面 global manager）：
    #    - 输入 L=127k、A=∅、S={1,2,4,8}、P=32 个 prefill 实例（各带排队延迟 T_i）
    #    - Algorithm 2：按当前 improvement_rate（30s 刷新，来自离线 simulator 的 rate→最优 rate 映射）
    #      用延迟模型 T_s=a_s+b_s·L+c_s·(C·L)+d_s·L² 估计各 SP 的 TTFT=T_queue+max{T_i}，
    #      只在 TTFT 增益超过阈值时才扩 SP（防止过度扩张）
    #    - Algorithm 1 递归：枚举 (s_current,s_next) 对，Algorithm 3 用排队延迟差设 chunk 延迟预算、
    #      数值求解当前 chunk 长度（如 chunk0 用 SP=2 填 p2,p3 的空隙、chunk1 用 SP=4、chunk2 用 SP=8 跑主计算）
    #    - 递归比较各计划的估计 TTFT，选出最优 chunk 计划 → 构造 per-instance 元数据 → Ray 转发给 local managers
    # 3) 推理后端（PyTorch + Triton-distributed，prefill 实例）：
    #    - 每个 chunk 的 token 按 zigzag 交错到该 chunk 的实例组（扩展的 Flash Attention 做 ring attention）
    #    - 计算新 chunk 前，用 NCCL 把前序 chunk 的 KV cache 均匀重分布到当前实例组（cache balancing，
    #      复用 ring communicator 与下一层 prefill 计算跨层重叠，隐藏传输开销）
    #    - send manager 向 decoding 侧 receive manager 发 handshake 预留传输 backend（防 starvation），
    #      确认后用 NCCL/NVSHMEM 把各 chunk 的 KV cache 流式传到目标 decoding 实例
    # 4) decoding 实例（DP，TP=8，Flash Decoding + CUDAGraph）：
    #    - receive manager 收齐全部 chunk 的 KV cache 后通知 local scheduler
    #    - 用 iteration-level/continuous batching 把请求加入 decoding batch，
    #      每个实例作为一部分请求的 master 跑多请求分布式 decoding，逐 token 输出
    # 5) 输出：TTFT（首 token 延迟）与 TBT（token 间延迟）日志，P50/P99 指标
    ```
    作用：以 chunk 级细粒度 SP 分配同时优化 TTFT 与资源利用率——长请求用后段大 SP 压 prefill 延迟，前段小 SP 利用资源碎片提前开跑，配合负载感知的 improvement rate 控制 SP 扩张，避免 LoongServe 式"整请求大 SP"造成的实例空转，在 prefill 与 decoding 异构并行需求（小 TP prefill / 大 TP decode）下最大化集群吞吐并满足 SLO。
