## TailorLLM: Collaborative End-Cloud Inference of Large and Small Language Models Based on Low-Rank Adaptation

- baseline方法是什么？
  Baseline是cloud-only LLM推理服务（Llama3-70B纯云端部署），用户所有请求都发送到云端LLM执行decoder-based autoregressive inference。这是当前LLM推理服务的主流部署方案，提供稳定高精度体验，但面临cloud computing costs急剧上升的问题（推理服务成本已接近或超过预训练成本）。

  全栈执行例子（以Llama3-70B cloud-only text summarization请求为例）：
  - 算法层：单个用户请求→Llama3-70B autoregressive decode，每token经全部decoder layers（RMSNorm→Grouped Query Attention→FeedForward Network with SiLU gate），无任何端侧参与。全精度推理。
  - 系统框架/Serving层：请求到达云端服务器（4×RTX 3090）→排队等待GPU→prefill处理prompt→逐token decode→返回response。单请求无法利用GPU并行计算能力（sequential token generation成为瓶颈），无请求间batching时GPU利用率低。
  - 编译框架层：论文未明确说明（使用默认PyTorch/CUDA推理路径）。
  - kernel调度层：论文未明确说明（使用默认GPU kernel，无定制kernel）。
  - 硬件架构层：NVIDIA RTX 3090 GPU (24GB)。cloud-only下所有计算集中在云端GPU，成本按API定价累积（输入$2.50/1M token + 输出$10.00/1M token）。

  论文同时将token-level end-cloud collaboration（HSL, speculative decoding）和model partitioning（Petals）作为对比baseline。HSL方法：端侧SLM生成draft tokens→云端LLM逐token验证（每5 token验证一次），频繁端云交互导致累计延迟（单次QA可能有15+次通信），token验证频率高时接近cloud-only延迟。Petals方法：将Llama3-70B按5:65拆分到端云两侧，受限于端侧显存。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **方法概述**：TailorLLM提出task-level（而非token-level）端云协同推理框架。核心理念：观察用户请求在task-level具有高度集中性（少数任务覆盖>70%请求）和时间周期性，用LoRA微调SLM使其在high-frequency任务上达到接近LLM的精度→端侧自主完成这些任务→减少云端调用频率和端云通信轮次。两个关键算法：RFLoRA（离线低秩微调，冻结共享A矩阵，仅传任务特定B+magnitude，减少~50%传输）和AdapterMgr（在线imitation learning管理端侧LoRA cache，用Mamba时序模型+Belady最优策略学习做近最优替换决策）。

  **缺陷→方法映射**：

  **缺陷1：Cloud-only方案成本过高，GPU并行能力在single-request autoregressive decode中未充分利用**
  → 方法：Task-level任务分流。高频简单任务（text translation/sentiment analysis/summarization等）在端侧经LoRA增强的SLM完成，复杂/稀有任务（数学推理/新类别）发往云端LLM。实验：约70%请求端侧处理，cloud cost减少69.8%，end-to-end latency减少62%（vs cloud-only）。Llama3-1B推理速度22.6ms/token vs Llama3-70B 5.3ms/token，但由于端侧无网络RTT，总体延迟更低。

  **缺陷2：Token-level collaboration（HSL）频繁端云通信（单QA 15+次），在弱网环境下累计延迟严重，且cloud validation部分抵消成本节省**
  → 方法：Task-level替换token-level。端侧SLM经LoRA增强后在特定任务上独立完成推理（而非仅生成draft tokens等待cloud验证），单次请求最多1次端云通信（仅在卸载任务时）。实验：RTT 20→200ms时TPOT仅1%退化（HSL 22%退化、Petals 46%退化），TTFT始终保持低位。TailorLLM端侧hit rate决定cloud cost节省程度（AdapterMgr hit rate接近Belady最优）。

  **缺陷3：标准LoRA adapter传输开销大（每个adapter ~22MB for Llama3-1B），在无线网络下更新端侧LoRA library成本高，限制可存储的adapter数量**
  → 方法：RFLoRA参数解耦。发现A矩阵跨任务收敛（domain-invariant encoder），B矩阵任务特异（domain-specific transformation），且权重可分解为direction+magnitude。设计：冻结共享A矩阵（所有任务共用一份），仅传输B矩阵+magnitude参数m（~11.56MB vs 22MB，减少~50%）。解耦还使magnitude分量不依赖frozen A就能独立优化方向分量。实验：RFLoRA达到81.6% accuracy (3.4M trainable params, 0.273%) vs LoRA 81.2% (0.454%)、DoRA 82.1% (0.484%)。Trainable params约为DoRA的56%，但精度仅差0.5pt。11.56MB adapter在端侧可存储更多任务（同等存储空间下数量翻倍）。

  **缺陷4：静态LoRA部署无法适应用户需求的动态变化（任务周期性+新任务出现），LRU简单策略预测精度低**
  → 方法：AdapterMgr基于imitation learning的动态cache管理。用Mamba (SSM)捕捉用户历史访问序列的长程时序依赖（支持训练时并行、推理时recurrent），融合端侧cache state双模态信息（projection fusion），以Belady最优替换策略（evict longest reuse distance）为训练标签，BCE loss引导模型不仅学习最优action还学习区分"正确/错误"替换决策。滑动窗口H=100，cache capacity w=5，embedding dim d=128。实验：AdapterMgr在所有数据集上hit rate最接近Belady上界，在用户请求越动态（cycle 200 vs cycle 30）时相对LRU优势越明显。

  **论文方法全栈执行例子（以Llama3-1B + RFLoRA + AdapterMgr text summarization请求为例）**：
  - 算法层：RFLoRA离线训练：云端对text summarization数据集fine-tune Llama3-1B→权重W分解为magnitude m + direction W/||W||_c→direction分量施加LoRA ΔW=BA (A共享冻结,B可训练)→仅B(11.56MB)+m传输到端侧。端侧预存一份共享A矩阵。推理时加载B→合并W' = m·(W_0+BA)/||W_0+BA||_c→SLM执行autoregressive decode (22.6ms/token)。
  - 系统框架/Serving层：用户text summarization query到达→Task Classifier (Contriever semantic encoding→UMAP降维→HDBSCAN聚类)判定为已知summarization类别→Allocator查表确认精度达标+local cache命中→加载对应LoRA→端侧独立完成推理（0次云端通信）。AdapterMgr后台：维护历史访问序列H=100→Mamba提取时序特征→融合当前cache state→MLP输出替换概率→若B矩阵已在cache则hit（不替换），若miss且cache满则evict max eviction-probability slot并下载新B。
  - 编译框架层：论文未明确说明（使用默认PyTorch/CUDA路径，端侧Tesla T4 GPU推理）。
  - kernel调度层：论文未明确说明（使用默认GPU kernel，无定制kernel）。
  - 硬件架构层：Cloud-side 4×RTX 3090 (24GB)，End-side Tesla T4 (16GB→10GB limited)。端侧Llama3-1B推理占用~2.8GB显存，LoRA switching <1ms，task classification 0.45-1.53ms。端侧energy消耗经llama.cpp在手机上评估约等于轻量2D游戏功耗。

  **关键trade-off**：依赖LoRA adapter的wireless传输（每个~11.56MB），在弱网/低带宽下仍有传输延迟。SLM模型更新时所有已训练LoRA需重新训练。对新任务类型需积累足够样本才能形成新类别（HDBSCAN基于密度聚类），冷启动阶段依赖云端。仅评估1B/70B模型组合，未验证其他SLM/LLM组合。
