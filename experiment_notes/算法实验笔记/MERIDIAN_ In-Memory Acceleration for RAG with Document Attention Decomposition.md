## MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition

- 属于算法pipeline的实现是什么？实验比较什么？
  - 属于算法pipeline的实现：**document attention decomposition（文档注意力分解）**——面向 KV 预计算（KV-precomputed）RAG 推理的注意力分解算法。把标准 softmax 注意力模块拆成两个独立分支：DocumentAttention 分支（在持有文档 KV 分片的 PIM 设备上本地执行，对本地 K/V shard 计算注意力）与 QueryResponseAttention 分支（处理用户 query 与已生成 token 的 KV），各自只产出紧凑的局部摘要（未归一化输出 o、局部最大值 m、归一化因子 l），再通过数值稳定的 softmax 全局聚合（基于共享基线 m=max(m_d,m_c)，公式 l=e^{m_d-m}l_d+e^{m_c-m}l_c，o=(e^{m_d-m}o_d+e^{m_c-m}o_c)/l，即 FlashAttention 的 online-softmax 思想 [10]）合并。算法 1（In-Layer Document Attention Decomposition）给出逐 token 流程：QKVProjection → DocumentAttention(q,state_doc) → QueryResponseAttention(q,k,v,state_ctx) → Fusion → LN1 → FFN → LN2。后续所有 transformer 层（LN、FFN、残差）保持不变，因此数值语义与标准 softmax attention 完全一致，无需重训即可保持精度。
  - 通信量对比（本算法的核心收益）：集中式需传输整份文档 KV：V_ce = #Document tokens × 2 × d_model × 2 (bytes)（FP16）；去中心化只需传 query 向量并回收局部摘要：V_de ≈ (#Query tokens + #Response tokens) × 2 × d_model × 2 (bytes)。由于检索文档平均比 query+response 长 ~380×，通信量降低两个数量级以上；文档 K/V 按 attention head 维度分片到 N 个设备后每设备只传 d_model/N 大小的输出切片，跨设备总流量基本恒定。
  - 实验比较什么：端到端吞吐（tokens/s，batch size 2/4/8/16，图 9）、每请求延迟（图 10）、通信/prefill/decode 三阶段延迟分解（图 11）、能量效率（图 12）、准确率（图 13，与 CPU-GPU baseline 差 <0.4pp）、与 HeterRAG 的端到端吞吐/延迟（图 14/15）、组件消融（M-pim/M-non/M-ad/M-ad+ire/M，图 16）、规模扩展（OPT-66B，4→32 设备，图 17）、长 query 通用性（4×/16×/64× 原始 query 长度，图 18）。平均吞吐提升 5.36×/6.64×/3.98×/3.32×/3.91×、延迟降低 4.30×/5.34×/3.31×/2.73×/2.79×（对 TurboRAG/BlockAttention/CENT/PAPI/HeterRAG）。
- 硬件平台是什么，配置是什么。
  - MERIDIAN 为仿真评估：32 个 PIM 设备（默认 16 个 Document Attention Cluster DAC + 16 个 Context Execution Cluster CEC），每个设备 512 GB 容量、32 TFLOPS 峰值；CXL 3.0 over PCIe Gen5 ×16（每链路 128 GB/s），端到端 CXL 访问延迟 165ns（25ns 端口 + 10ns retimer + 70ns switch + 60ns 内存控制器/DRAM）。内存：LPDDR5X，64 GB/package，8.5 Gb/s/pin，×128 channels，t_RC=60/t_RAS=40/t_CL=23/t_RP=20/t_RCDRD=17/t_RCDWR=8；外部带宽 1.1 TB/s、内部带宽 16 TB/s。PIM Unit：16 FP16 比较器 + 16 FP16 乘法器 + 16 FP16 加法器 + 4 KB 双缓冲 buffer；Controller-Side Unit：1 个加法单元（16 FP16 加法器）+ 1 个软max单元 + 8 个 BOOMv2 RISC-V 核。每 DRAM bank 配 16-lane PU 运行于 1 GHz，32 设备共提供 16 TB 容量。
  - CPU-GPU baseline 平台：Intel Xeon Gold 6454S + 1 TB DDR5 + 4× NVIDIA H100（每卡 80 GB HBM2e），GPU 能耗用 nvprof 测量。
- 模型是什么。数据集和bench分别是什么。
  - 模型（两个微调 RAG 模型）：① Qwen-TB，基于 Qwen2-7B [53] 用 TurboRAG [44] 微调，28 层、28 heads、hidden 3584；② Tulu3-Block-FT，基于 Llama-3.1-Tulu-3-8B-SFT [3] 用 BlockAttention [46] 微调，32 层、32 heads、hidden 4096。可扩展性实验另用 OPT-66B（64 层、72 heads、hidden 9216）。默认 batch size 8。
  - 数据集：2WikiMultiHopQA (2Wiki) [18]、HotpotQA (HQA) [63]、Natural Questions (NQ) [30]、TriviaQA (TQA) [26]。平均 token 长度：2Wiki Doc 856.76/Q 17.60/Resp 3.03；HQA 1341.04/20.41/3.97；NQ 14630.04/10.28/4.36；TQA 14748.69/18.57/4.59。与 HeterRAG 对比时采用 HeterRAG 的检索管线：AccelDIMM 加速器 + HNSW 索引 + 完整 Wikipedia 语料，模型用 Tulu3-Block-FT。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：MERIDIAN 未开源（截至 2026-08 联网搜索未找到公开仓库，ISCA 2026 论文，后续可能随 artifact 发布）。论文采用 PIM-SYCL 式异构编程模型，与 CUDA 类似，高层面 API 暴露 GEMV、GeLU 等操作与设备初始化、并行策略选择；编译为底层指令（PIM 计算命令 PIM_MAC/PIM_CMP/PIM_EW_MULT/PIM_EW_ADD 与数据移动命令 PIM_ACT/PIM_WR_PB/PIM_RD_PB，标准 load/store 经 CXL.mem）派发到设备控制器，控制器广播 PIM 指令到相关 channel 和 PU。
  - 算法 pipeline 伪代码（单设备持有文档 KV (K_d,V_d)、另一设备持有上下文 KV (K_c,V_c)，给定 query 向量 q）：
    ```
    # 1) 两个分支独立计算注意力 logits（GEMV）：
    s_d = q @ K_d^T ;   s_c = q @ K_c^T
    # 2) 各分支本地归一化基线（max）：
    m_d = max(s_d) ;    m_c = max(s_c)
    # 3) 各分支形成未归一化输出与归一化因子（累加，GEMV-like）：
    o_d = Σ_j exp(s_d[j]-m_d) * V_d[j] ;  l_d = Σ_j exp(s_d[j]-m_d)
    o_c = Σ_j exp(s_c[j]-m_c) * V_c[j] ;  l_c = Σ_j exp(s_c[j]-m_c)
    # 4) 全局融合（共享基线 m=max(m_d,m_c)，数值稳定）：
    l = exp(m_d-m)*l_d + exp(m_c-m)*l_c
    o = ( exp(m_d-m)*o_d + exp(m_c-m)*o_c ) / l
    # 5) 下游：x ← LN1(x + o)；f ← FFN(x)；y ← LN2(x + f)
    ```
  - 张量计算例子（一次解码 step，d_model=3584、Qwen-TB、单头场景）：只传输当前 query 向量 q（d_model×FP16 ≈ 7KB）进内存，文档侧 K_d（如 1.4 万 token × 3584，FP16 ≈ 96MB）保持静止在 PIM 设备内；设备本地算 s_d = q·K_d^T（1×14630 GEMM 退化为 GEMV），在线 softmax 输出 o_d、m_d、l_d 三个紧凑量（o_d 为 d_model 向量 + 2 个标量），仅此经 CXL 返回 host/聚合层。对比集中式需把 96MB 文档 KV 从 host 搬到设备再算注意力。批量/多头时 K/V 按 head 分片到 N 个设备，各设备只返回 d_model/N 的输出切片，跨设备聚合流量与设备数无关。
