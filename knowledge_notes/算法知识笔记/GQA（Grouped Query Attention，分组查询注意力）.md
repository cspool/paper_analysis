## GQA（Grouped Query Attention，分组查询注意力）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GQA 让多组 Q head 共享同一组 K/V head（如 Llama2-70B 的 8 个 Q head 共享 1 个 K/V head），在接近 MHA 质量的同时大幅减小 KV Cache 与 K/V 投影计算量（Llama3 同样采用）。对 PIM 系统的意义（CompAir）：K^T/V 权重被多个 Q head 复用，等价于给 K^T/V 引入 batch 级复用——普通 MLA/MHA 下 K^T/V 输入相关、每次推理都变，只适合 DRAM-PIM；GQA 下 K/V 共享使 SRAM-PIM 加速 attention 成为可能。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CompAir 的 GQA 映射：TP 沿 seqlen 切 K^T/V → SRAM-PIM 的 batch 维 = 序列长度段、输出维 = GQA group size（Llama2-70B 为 8）、输入维 = hidden size（QK^T）或 seqlen（SV）。
```
# Llama2-70B GQA：8 Q heads 共享 1 组 K/V
for q_head in group:
    score = Q_q @ K_shared^T      # K_shared 被复用 8 次 → 有复用
    out_q  = softmax(score) @ V_shared
```
权衡：长序列必然带来更多 cross-die 传输与更高能耗；QK^T 是否用 SRAM-PIM 取决于 TP 与 seqlen 组合，SV 恒用 DRAM-PIM（能量优势）。结论规则：GQA 的共享结构把 K/V 从"每次推理变"变成"可复用"，据此决定硬件分派。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：模型参数 num_key_value_heads < num_attention_heads，KV Cache 按 KV head 数存储、attention 时 repeat_kv 展开。使用方式：降低长上下文 KV Cache 内存与 K/V 计算量（对比 MLA：MLA 用低秩压缩、GQA 用 head 共享）；硬件侧按"是否有 head 级复用"决定映射（CompAir：复用→SRAM-PIM 候选、无复用→DRAM-PIM）。CHIME 的 GQA-8 使用（QWEN-72B）：bank PU 的计算访存比按 group 放大（N_cmr=8 才能满带宽利用），跨 chip 传输放大 N_gqa 倍使 bubble-free 条件更紧，head 映射只能取 N_hc=1（每 head 单 chip）；调度器对 GQA 的 decoding 请求加批步长取 N=16（head 少、rank 负载均衡难），且 GQA 更难达到 attention 瓶颈（需更大容量请求）。

PLENA 补充视角（ISCA'26）：GQA 从硬件利用率角度产生"per-head fat GEMM"问题——head_dim 小（LLaMA-3-70B 为 128）且一个 K 头被多个 Q 头同时相乘（GQA 组内复用），在方形大脉动阵列上 per-head GEMM 的计算维度小、利用率低。PLENA 的解法：FlashAttention 阶段把扁平化脉动阵列切分成多个小 flattened core，每个 core 执行 (BLEN,HLEN)×(HLEN,BLEN) 的 per-head GEMM、并行覆盖 MLEN//HLEN 个 Q 头（head 预加载），使 attention 计算与有效 batch 解耦——decode 长上下文（有效 batch 小）下仍保持高利用率；FFN 阶段同一阵列以 (BLEN,MLEN) 形态跑 fat GEMM（BLEN≈batch）。即同一阵列通过 BLEN/头级分解同时适配 FFN 与 GQA attention 两类 GEMM 形状。

ConServe 补充视角（ISCA'26）：评估模型均为 GQA——Yi-6B-200K（Hq=32、Hkv=4、L=32）、Llama-3-8B-262K（Hq=32、Hkv=8、L=32）、Yi-34B-200K（Hq=56、Hkv=8、L=60）。KV footprint 按 KV 头数计算：B_tok=2·L_shard·H_shard·d_head·b（H_shard=TP 分片上的 KV 头数，Llama-3-8B BF16 约 4 KB/token/layer）；GQA 的小 KV 头数直接决定 slice 每层段尺寸（B_layer=2×H_shard×d_head×b）与 resize 触发频率，K/V 共享结构不影响 ConServe 的连续布局（K、V 都落进该层段）。

Raptor 补充视角（ISCA'26）：GQA 直接决定 KV cache 足迹与带宽需求——Llama-3.1-70B（80 层、8 KV 头、head dim 128）每 token 产生 2×8×128×q B 的 KV 状态（q=1B 时 8-bit 2KB、q=2B 时 FP16 4KB）；KV 读带宽下界 D_KV ≥ S·(2LH_KV·d_head·q)+2LH_KV·d_head·q 中的 H_KV 即 GQA 的 KV 头数（比 MHA 小一个"Q 头/KV 头"组比），因此 GQA 在容量与带宽两个维度同时降低 decode 压力。Raptor 以 KV 为中心设计 3D-DRAM：单层 4K 上下文 16MB KV cache 切成 1024 个 16KB stream-blocked tile 摊到 16 channel（每 channel 3 bank、128B flit），KV tile 粒度（16KB）与 paged-attention ≥4KB page 对齐；GQA 的小 H_KV 使 per-token KV 字节少，单卡 32GB 3D-DRAM 即可容纳 Llama-70B 权重+KV（TP=1），这是其"低并行度、低 collective、网络不敏感"部署的结构前提。

P3-LLM 补充视角（ISCA'26，NPU-PIM 边缘 LLM 推理）：GQA 使 KV cache 容量按 group 数 G 缩小、算术强度 >1（区别于早期 MHA 的算术强度=1），因此低 batch 解码下 PIM 不再天然占优——roofline 显示 HBM-PIM 在 batch≥4 或 GQA 场景相对 NPU 优势消失，这驱动 P3-LLM 设计吞吐增强 PCU（TEP：时间维输入复用使同一 KV/权重切片在 tCCD_S 窗口内服务两个输入）并保持 attention 全模块在 PIM 执行（4-bit KV + 8-bit 注意力分数）。Llama-3.1-8B（G=4）与 Llama-3.2-3B（G=3）在 batch 2-64 下 attention 仍占主导，P3-LLM 借高内部 PIM 带宽 + TEP 优于 Ecco。

QiMeng-Tensify 补充视角（ISCA'26）：GQA 作为图级编译优化 benchmark 子图（Table VII，Arch. 列标注 LLaMA3-70B）——代表"多算子 + 复杂数据依赖"的 attention 类子图（Q 投影、K/V 投影、QK^T、softmax、SV、输出投影的图）。QiMeng-Tensify 把它用于 LLM prior 消融（Fig.8b，GQA 与 GatedMLP 并列：LLM 先验比统计先验高 20%-30%）与可移植性分析（L0/L1/L2，GQA 为三个代表子图之一）。同一 benchmark 还含 QKNorm（Chameleon-7B 的 GQA+QK 归一化）与 SelfAtten 等 attention 类子图；QKNorm 上 QiMeng-Tensify 超出 FlashAttention 1.66×、TensorRT 1.40×——表明自动图优化能超越注意力专用手写优化。图级视角：GQA 的 K/V head 共享（8 Q head 共享 1 组 K/V）在编译层面表现为可被 compute_at/tiling 利用的数据复用结构，但本论文关注点在"多算子图调度"而非 PIM 映射。

  - SHyLA 补充：GQA 减少 KV head 数 → KVCache 写事务相对 Weight/KVCache 读进一步被抑制（图 4b 红虚线），KVCache 压力下降 → 微批可更大 → DSE 中 NVM 偏好从"容量"转向"带宽"（GQA 模型如 Mixtral 8×22B 偏好更高 NVM 带宽）。SHyLA 对 GQA 的映射：decode 请求按 attention-group/request 级并行（每组在一个 tile 内处理，消除组间 KVCache 共享的跨 tile 传输）；同一 group 内多个 Q head 复用同一 K/V 对，KVCache 重载次数由 tile weight buffer 容量决定（装得下则只载一次）；并行化策略取决于 attention group 数 g 与张量并行度 pt（g≥pt 按组分配，g<pt 用 sequence parallelism 子切分）。
Understanding Inference Scaling 补充视角（ISCA'26，reasoning 负载下 GQA 的 KV 足迹量化）：GQA 的 8 KV head 配置（Llama 系蒸馏模型）把 KV footprint 相比 MHA 降 3×–8×，但内存成本仍随层数线性增长——32B 模型（≈64 层）FP16 下 ≈262 KB/token，70B 模型（≈80 层）≈328 KB/token，Llama-3.1-405B（≈126 层）≈1.05 MB/token；128 请求 × 10k reasoning token 的 batch 仅 KV cache 就超 1.3 TB（远超单 H200 的 141 GB）。论文用该数字论证"KV 容量是推理第一瓶颈"：GQA 只缓解、不消除 decode 容量压力，容量墙出现于 prefill 与 decode 交界（batch 4K/5K 时 KV 在 prefill 阶段即耗尽）。
涉及论文标题：
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation
- Understanding Inference Scaling for LLMs Bottlenecks, Trade-offs, and Performance Principles
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM
- Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference
- P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats
- ConServe: Contiguity-Preserving Memory Management for Multi-Turn LLM Serving
- Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity
