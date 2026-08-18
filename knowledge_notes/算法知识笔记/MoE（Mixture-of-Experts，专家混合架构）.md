## MoE（Mixture-of-Experts，专家混合架构）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MoE 是稀疏激活的 Transformer 变体：把稠密 FFN 层替换为 N 个并行的"专家"（小型 FFN）+ 一个 gate（路由）网络。每个 token 经 gate 打分选出 topk 个专家（DeepSeek-V3 topk=8，带 bias 项调节负载），只在这些专家上执行 FFN 计算（GEMM-1：hidden→moe_hidden；GEMM-2：moe_hidden→hidden），topk 个输出按 gate 权重加权求和，激活计算量约为稠密模型的 k/N，从而在参数规模大幅扩张的同时控制算力开销。代表模型：DeepSeek-V3（256 路由专家 + 1 shared，MoE hidden 2048）、Mixtral 8x7B、GPT-OSS-120B、Qwen3-235B。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 每层 MoE 前向（token 粒度）
scores = gate(x)                       # (S, N) 路由打分
topk_idx, topk_w = topk(scores, k)     # 每 token 选 k 个专家与权重
y = 0
for (e, w) in zip(topk_idx, topk_w):   # 各 token 的专家子集动态不同
    x_e = dispatch(x, e)               # EP 下把 token 发往专家 e 所在 GPU
    h_e = W1[e] @ x_e                  # GEMM-1
    o_e = W2[e] @ act(h_e)             # GEMM-2（激活函数视模型而定）
    y += w * combine(o_e)              # 加权聚合回 token 所在 GPU
```
Annotations：S=序列长度，N=专家数，k=topk。Dispatch/Combine 在单卡是局部 gather/scatter，在专家并行（EP）下变成跨 GPU 通信算子（见 kernel 调度层条目）。本论文在算法层的唯一改动：把加权求和 w×o 提前到 GEMM-2 的 epilogue（输出写回前先乘 gate 权重），使 Combine 可以退化为交换机内的无权重归约，算法语义保持不变（最终 y = Σ_i w_i·o_i 等价）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
训练/推理框架（Megatron-Core、DeepSpeed-MoE、HuggingFace Transformers 等）实现 gate、专家 FFN 与负载均衡（aux loss 或 bias 调节）。分布式下专家按 EP 切分到多 GPU，通信库（DeepEP、Tutel、FasterMoE）实现 Dispatch/Combine。token 分布：训练近似正态（本论文取 std=0.032，源自 ByteDance 对典型训练任务的观测；COMET 同设置），推理近似 power-law（α≈1.5，本论文灵敏度分析取 0.5-2.5）。本论文评估配置：DeepSeek-V3 Large（hidden 7168、MoE hidden 2048、128 注意力头、256 专家、topk 8）+ Small/Medium 扩展 + GPT-OSS-120B / Qwen3-235B。

DIAMoND 的边缘部署补充视角（ISCA'26）：边缘跑 MoE 的三重挑战——(1) 全量专家权重超边缘 DRAM 容量（Mixtral-8x7B INT8 需 47GB，全模型 3.7~7.5× 于激活模型），须驻留 SSD（密度 20.27~29.18 Gb/mm²、1Tb 容量）；(2) 批量 1 解码带宽敏感：以 Apple A18+LPDDR5X+NVMe 为例，单专家单层从 DRAM 到 NPU 的加载延迟是单 token 解码计算的 1.55~3.84×，SSD→DRAM 再加 15×；(3) 路由动态选专家（前向传播中才确定），权重无法预载、按需加载加剧带宽瓶颈。路由器实现细节：router = linear 层对 self-attention 输出打分 + top-k 选择（Mixtral top-2/8 专家；专家 FFN = Up/Gate/Down 三投影，隐维 4096、专家中间维 14336）；GRIN-MoE 的 gate 结构特殊——把 top-k 之外多数专家的路由权重置零（因此 DIAMoND 的 expert similarity 指标不适用于 GRIN）。DIAMoND 对路由的算法级改动（冲突感知的专家替换）见 Adaptive Expert Selection 条目。

Lit Silicon 补充视角（ISCA'26，MoE 训练的通信特征）：MoE 用专家并行（expert parallelism）把不同专家分配到不同 GPU，引入 all-to-all 通信把 token 路由到对应专家。与 AG/RS 不同，MoE 的 all-to-all 通常不与计算重叠（每层同步），因此每次同步点都重置 lead、产生比 dense 训练更小的 lead 值，但 token 分布不均导致偶发高延迟通信 spike（极大 lead 值），更难以分类 leader/straggler。论文用 AMD 推荐平台 Primus + torchtitan 后端训练 DeepSeek V3 16B（8 路专家并行，padding GEMM 使 MoE 权重计算均衡），比较 dense（Llama 3 8B b2s4）与 MoE 训练的功率/频率特征——两者相似，功率重分配算法仍收敛到稳定功率分布并取得与 dense 相当的节电。

- M100 补充视角（ISCA'26，车规自动驾驶 MoE 推理）：MindVLA 是理想汽车（Li Auto）下一代端到端 AD 模型，其 LLM 组件采用 MoE 策略（8 个专家）以提升模型容量与推理效率，评估使用 431M 参数配置。在 M100 NPU（12/14 cluster）上 MindVLA LLM 组件 decode 0.1ms vs Thor-U 0.3ms（3×）、prefill 0.84ms vs 1.74ms（2.1×）——数据流架构以 tensor 级指令 + TPB 功能单元数据流执行 MoE 的专家前向与加权聚合（论文未详述 MoE 的 dispatch/combine 在 M100 上的 kernel 级映射）。
MoE-Hub 补充视角（ISCA'26，MoE 通信的硬件化视角）：MoE 算法层的五步结构（routing → All-to-All dispatch → 专家计算（两次 GEMM + 中间激活）→ All-to-All combine → 加权缩放）在 EP 下每步都可能成为瓶颈：routing 结果动态变化使 dispatch 的目标集/消息大小运行时才确定，与 GPU address-centric 通信模型冲突，迫使软件先做地址解析再通信。MoE-Hub 不改算法语义（Top-K 路由、专家 GEMM、加权聚合均不变），只把 dispatch/combine 的地址解析与数据流编排下沉到 GPU hub 硬件：算法层每层仍是五步，但 dispatch 变成"路由结果一出来就发 st.rowsp"，combine 时源信息作为 expert 输入张量的一列随 token 一起传输（.nop 优先级），专家 GEMM 末尾读源元数据用常规 store 发起 combine，缩放照常加权求和。评估用三个代表模型：Mixtral 8x7B（32 层、Hidden 4096、FFN Hidden 14336、TopK/Experts 2/8）、Qwen2-MoE-2.7B（24 层、2048、1408、4/64）、Phi-3.5-MoE（32 层、4096、6400、2/16），total token = SeqLength×NGPU（128-32768）、token 分布 std 0-0.05（典型训练 std≈0.032）；结果 MoE 层 1.40×-3.08×、端到端 1.21×-1.98× 相对 SOTA，达理想 MoE 层 96.8%，且与负载均衡/TP+EP 等计算侧优化正交可叠加。

专家选择模式补充视角（ISCA'26，Patterns behind Chaos，首个 200B-1000B 四模型数据移动 profiling）：对 DeepSeek-V3（671B）、Llama4-Maverick-128E（402B）、Qwen3-235B、Kimi K2（1000B）用 SGLang（8×H100 DGX + 8×H200 AWS）采集 >24,000 请求、150GB+ JSON 专家选择 trace（开源 https://huggingface.co/datasets/core12345/MoE_expert_selection_trace）。时序模式（temporal）：(1) layer 级——相邻层专家共激活相关（top 20% 下一层候选覆盖 50%/65%/77%/56% 条件概率质量，Llama4 最强 DeepSeek 最弱；Qwen 热图比 DeepSeek 亮，相关性更强）；(2) token 级——同层相邻 token 倾向于选同一专家（高层的对角线模式），top 20% 下一 token 候选覆盖 47%/62%/80%/53%；(3) prefill-decode 级——两阶段专家对热图与专家频率分布高度相似（Spearman ρ≥0.7 强相关，top-5/10/20 重叠 60%/75%/90%）。空间模式（spatial）：(4) 单专家激活偏斜——部分专家激活频率超均值 16 倍；(5) 专家对共激活 affinity——DeepSeek/Qwen 的 top 10% 专家对贡献 60-80% 激活（Llama 每层只选 1 专家无共激活；DeepSeek 因路由限制只路由到相邻节点，共激活形成方块簇）；(6) 任务/语言影响——57 个 MMLU 学科的热门专家横线重叠但差异显著，中英相同题目下热门专家仅 2 个重叠（Insight 6：任务感知 serving 可提前迁移/复制专家）。这些模式证明"专家选择看似随机实则可预测"：时序可预测性支撑单单元策略（预取/缓存/复制，Insight 1/2），空间可预测性支撑多单元策略（放置/去中心化/分离，Insight 3/4/5）。

RoCC 补充视角（ISCA'26，专家并行 AllToAll 的 ROP 卸载）：RoCC 论文把 MoE 专家并行的 AllToAll（token dispatch/combine）纳入 CC 评估（A2A workload：每个专家与所有其它专家交换 token 的压力测试），并把 AllToAll 分解为最简 primitive（send→recv）在 ROP 上执行。区别于前述"大模型需 tensor 并行频繁同步"的动机，MoE 的 AllToAll 通信量动态变化（每轮 token 路由不同）、通常无法与计算重叠；RoCC 因 ROP 近内存、SM 全容量算专家 GEMM 且 warp 级细粒度重叠，对 AllToAll 大消息 CC-only 延迟达 25% 加速。
STEP 补充视角（ISCA'26，空间/时间局部性驱动的静态-动态混合 MoE 推理优化）：STEP 对 MoE 的算法级理解是"专家选择同时具有空间不均与时间连续"——(1) 空间：层内 top-k 低排名专家（如 top-4 的第 3/4 名）平均路由权重常 ≤0.05，贡献极小却照常进入加载与计算（Insight 1，Fig.3）；(2) 时间：长序列生成中一小撮专家被连续 step 反复选中（Insight 2，Fig.4），且跨任务/层/序列的激活模式差异大（Insight 3，Fig.5）。据此把每层结构从固定 j shared + k routed 改造为"离线层内剪枝（k_l 按归一化路由权重阈值 θ 下降）+ 在线窗口投票选举 top-c 高频专家为临时 shared（j+c shared + k−c routed）"，并用 token 感知自适应窗口（th_s=75%/th_f=40%/τ=3-4）动态调投票与预取跨度。加载时间模型 T_load = S·Σ_l(k_l − p_l·R_l)·t_expert（Eq.1，S=解码步数、k_l=层 l 激活 routed 数、p_l=预取数、R_l=预取命中率、t_expert=单专家加载时间）。效果：Mixtral 8x7B 平均专家数 2→1.75 时 MMLU 77.3→77.0（几乎无损）；prefill/decode 相对 llama.cpp 最高 3.12×/2.22×；与 MoE-I2 压缩、APTMoE 卸载正交可叠加（Table V/VI）。

- STAGE 补充视角（ISCA'26）：STAGE 把 MoE 作为一类模块模板建模（Gshard/Switch Transformer 的 MoE、DeepSeek-MoE 的 MoE+Shared Experts），在符号张量图中用专家激活直方图描述每层各专家被 token 激活的概率分布（默认均匀，可覆盖为自定义统计）；EP 下 token 经 AllToAll 路由到专家所在设备（dispatch/combine 两轮）。验证覆盖 Mixtral 8x7、DeepSeek-MoE 8E/144E 等：算子时间误差 3.0%~15.0%、通信量误差 0.945%~2.755%；指出真实训练 micro-batch=1 时部分专家不激活、与默认"全专家激活"假设产生偏差。

Understanding Inference Scaling 补充视角（ISCA'26，MoE 的推理特性与并行选择）：DeepSeek-R1-671B 作为稀疏 MoE 推理代表——总参数 671B 但每 token 只激活 ≈37B（激活参数比 ≈1/18），配合 MLA 压缩 KV。论文的核心推理侧发现：(1) 低激活参数量使计算-通信比远低于 405B 密集模型，TP=8 下高频 all-reduce 同步成为瓶颈（GPU 两次同步间计算时间短、同步开销无法摊销），故纯 TP=8（2047s）劣于 PP=4+TP=2 混合（1663s）；(2) MoE 的 routing 与 pipeline 同步延迟使 HBM 带宽利用率仅 ≈50–60%（对比 8B 密集 ≈85%）——frontier MoE 受同步与路由延迟约束而非原始带宽；(3) MLA 压缩的 KV 让 PP 每 stage 可容纳更多 micro-batch、填满 pipeline bubble——稀疏架构与 PP 天然协同。
涉及论文标题：
- Accelerating MoE with Dynamic In-Switch Computing on Multi-GPUs
- Understanding Inference Scaling for LLMs Bottlenecks, Trade-offs, and Performance Principles
- Scalable Synthesis of Distributed LLM Workloads Through Symbolic Tensor Graphs
- STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference
- DIAMoND Dynamic Inference for Adaptive Edge MoE with Heterogeneous In-NAND and Near-DRAM Compute Architecture
- IroKnight: Ownership-Preserving Neural Acceleration for Inference Serving
- Lit Silicon: A Case Where Thermal Imbalance Couples Concurrent Execution in Multiple GPUs
- MoE-Hub Taming Software Complexity for Seamless MoE Overlap with Hardware-Accelerated Communication on Multi-GPU Systems
- Rearchitecting the Datacenter Lifecycle for AI

IroKnight 补充视角（ISCA'26，MoE 层的细粒度规则访问作为加密范式的基础）：IroKnight 用 MoE transformer 层作全文运行示例——一个 MoE 层含三个子块：(a) self-attention = 两个 MatMul（Q·K^T、softmax 输出·V）+ softmax；(b) expert router = MatMul + softmax + top-k 选择；(c) expert FFN = 同一批 tiled/vectored 算子（MatMul、sigmoid 等）。关键观察：这些算子全部是细粒度 tiled/vectored 计算、地址呈仿射 walk（SIMD stride=1 扫描、vector reduction 的 stride=SIMD 宽度），因此每个算子都能由编译器给出基址/stride/offset 并提前预计算加密 pad——这是 Fully-State Encrypted Acceleration（所有存储保持密文、明文只在 ALU 组合逻辑瞬态）成立的前提。评估中 MoE 模型 GPT-OSS-120B（16 NPU）与 Llama4-Scout（16 NPU）经 Microsoft DeepSpeed 的 tensor-parallel + expert-parallel 跨 NPU 分发（芯片间经 RSA 私密通道交换 AES-GCM key 交换加密中间数据）；加密变体端到端运行时 0.2%、认证变体 3.3%（图 6/7 输入输出 token 256→4096 扫描），token pruning 0-80% 下加密 0.3%-0.7%、认证 3.3%-3.7%。
- M100: An Orchestrated Dataflow Architecture Powering General AI Computing
- Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference
- RoCC Harnessing Raster Operations Pipeline for Efficient Tensor Collective Communication
