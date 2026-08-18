## Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism

- baseline方法是什么？
  - Baseline 有两类：①**Fixed-SP 系统**——集群按固定 SP 大小静态分区（论文实验用 SP=8/16，请求路由到排队延迟最低的 SP 组）；②**LoongServe（ESP，弹性序列并行，SOSP 2024）**——把所有实例组成统一 SP 池（共享相同 TP 大小），在**请求批粒度**上动态调整 SP 分配（请求级弹性 SP），以及它的 prefill-decoding 解耦变体 LoongServe Disaggregated。LoongServe 的调度是贪心静态 batching：DP 为整批请求选 SP 实例（赋最大 SP 穷尽最小化 per-batch prefill 延迟），整批 prefill 完成后集体进入 decoding，batch 固定不变。
  - baseline 全栈执行例子（一个 128k 长请求到达 16 实例 A100 集群）：
    ```
    # 算法层：transformer prefill 整请求统一 SP=16（LoongServe 为最小化该请求 prefill 延迟而最大化 SP）
    # 系统框架层：所有实例统一 TP=1（ESP 不能为 prefill 与 decoding 设不同 TP）；
    #   请求级调度：先到先服务/贪心 batch，整批 prefill 完才进 decoding（静态 batching）
    # 编译框架层：论文未明确说明（沿用 vLLM/现有 serving 的算子库）
    # kernel 调度层：ring attention（zigzag/带状交错）——16 实例同时开始，每实例仅拿到 8k token 的极小计算量，
    #   计算无法掩盖 ring 通信，短请求/小 chunk 的 GPU 利用率极低；整批跨实例同步等待
    # 硬件架构层：A100-SXM4-80GB + NVLink + InfiniBand；大 SP 把 16 实例绑成同步环，
    #   任何实例排队就绪晚（如已被前序请求占用 t1）时，其余实例全部空转 t1
    # 输出：TTFT = max 排队延迟 + SP=16 prefill 延迟；过度 SP 扩张导致后到请求排队更久，全局 TTFT 分布恶化
    ```
    痛点：①prefill 受益于小 TP（SP 分配灵活）、decoding 受益于大 TP（压低计算延迟），ESP 统一 TP 两头不讨好；②贪心为整批请求赋最大 SP 缺乏全局负载感知，过度 SP 扩张恶化全局 TTFT 分布；③请求级 SP 分配 + ring attention 跨实例同步，SP 大小变化带来的资源碎片（实例排队延迟差）导致实例空转。
- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：**CDSP（Chunkwise Dynamic Sequence Parallelism）**——把每个请求的 prompt 拆成多个 chunk，为每个 chunk 分配不同的 SP 大小（前段小 SP、后段大 SP，SP 沿 chunk 递增）；构建 **Tetris** 服务系统：prefill-decoding 解耦集群（prefill 统一 SP 池小 TP、decoding 大 TP DP）、负载感知 improvement rate 控制 SP 扩张、递归 CDSP 调度算法 + FLOPs 延迟模型、handshake cache 传输管理。对应解决：①**异构并行需求**（Cluster Architecture 缺陷）→ 解耦集群中 prefill 用小 TP 的 SP 池（TP=1 for 8B），decoding 用大 TP（TP=8）的 DP，各自最优化 TTFT/TBT；②**过度 SP 扩张**（Batching Strategy 缺陷）→ Algorithm 2 用 improvement rate 阈值：只有 TTFT 增益超过阈值（阈值随实时到达率经离线 simulator 预计算的最优 rate 映射动态刷新，每 30s）才扩大 SP，避免全局排队延迟恶化；③**资源碎片/实例空转**（SP Allocation Granularity 缺陷）→ chunk 级分配：前段 chunk 用小 SP 填空闲实例的排队空隙（如 Fig.4 中实例 2、3 的空档），后段 chunk 用大 SP 满足长序列计算，同时优化 TTFT 与利用率；配套推理引擎（跨 chunk 的 KV cache 均匀重分布 + 层间重叠、handshake backend 分配）让 CDSP 计算/传输不成为瓶颈。
  - 论文方法全栈执行例子（同一个 128k 长请求 + 前序请求占用部分实例，4 节点 A100 集群，32 prefill + decoding 实例）：
    ```
    # 算法层：CDSP 把 prompt 拆成多 chunk，如 (C0=20k, SP=2)→(C1=40k, SP=4)→(C2=68k, SP=8)，
    #   chunk 间注意力经全注意力掩码，历史 KV cache 均匀重分布保持负载均衡
    # 系统框架层（vLLM 修改）：FastAPI 收请求 → C++ CDSP scheduler 递归求解 chunk 计划
    #   （Algorithm 1/2/3，延迟模型 T_s=a_s+b_s·L+c_s·(C·L)+d_s·L² 数值求解 chunk 长度，
    #   improvement rate 按 30s 窗口观测到达率查离线 simulator 的最优映射）→ Ray 转发 per-instance 计划
    #   → prefill 实例（TP=1 SP 池）执行 CDSP prefill，decoding 实例（TP=8）continuous batching
    # 编译框架层：PyTorch + Triton-distributed 运行时；论文未明确说明自定义编译 pass
    # kernel 调度层：扩展 Flash Attention 的 zigzag ring attention（chunk 内交错、跨 chunk 重分布 cache），
    #   NVSHMEM 做 ring 通信、Flash Decoding + CUDAGraph 做 decoding、NCCL（v2.26 并发 communicator）
    #   做 cache balancing 与 P/D 传输，专用 buffer/stream + handshake 与 prefill 计算跨层重叠隐藏开销
    # 硬件架构层：A100-SXM4-80GB×32/64（NVLink、200Gbps InfiniBand）；小 SP 前段 chunk 利用空闲实例
    #   提前计算，实例不再因同步 ring 而空转；大 SP 后段 chunk 压低长请求 prefill 延迟
    # 输出：TTFT 降低（最高 4.35×）、median TBT 降低 40.1%、最大请求容量提升 45%、吞吐 1.24-3.38×
    ```
