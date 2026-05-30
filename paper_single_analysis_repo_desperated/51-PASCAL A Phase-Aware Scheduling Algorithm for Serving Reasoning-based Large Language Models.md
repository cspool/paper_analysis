论文标题：PASCAL: A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Models

reasoning模型中，TTFT包含Prefill全部和Decode中reasoning token decoding，不包含Decode中answering token decoding，因此调度需要优先reasoning token decoding。

开源仓库确认：
    - 状态：未找到明确官方开源仓库
    - 链接：N/A
    - 说明：本地 PDF、arXiv 页面、HPCA 2026 会议页面和 KAIST 论文页均确认论文与摘要信息，但没有给出官方代码仓库。arXiv 的 Code/Data/Media 区域仅列出外部代码检索入口；CatalyzeX 页面显示 “Request Code” 而非仓库链接。面向 GitHub 的标题、arXiv 编号和关键术语检索也未发现作者官方实现。因此只能确认论文使用 vLLM profiling 和自建模拟器评估，不能确认 PASCAL 已开源。
    - 证据来源：
        - 本地全文：paper_2026/51-PASCAL A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Models.pdf
        - arXiv：https://arxiv.org/abs/2602.11530
        - HPCA 2026：https://2026.hpca-conf.org/details/hpca-2026-main-conference/19/PASCAL-A-Phase-Aware-Scheduling-Algorithm-for-Serving-Reasoning-based-Large-Language
        - KAIST Pure：https://pure.kaist.ac.kr/en/publications/pascal-a-phase-aware-scheduling-algorithm-for-serving-reasoning-b/
        - CatalyzeX：https://www.catalyzex.com/paper/pascal-a-phase-aware-scheduling-algorithm-for

1、论文工作：
    - 论文要解决的核心问题：reasoning-based LLM 在输出用户可见答案前，会先生成大量用户不可见的 reasoning tokens。传统 serving 系统把 TTFT 主要对应到 prefill，把 TPOT 对应到 decode；但在 reasoning 模型中，TTFT 实际包含 prefill、reasoning token decoding、以及从 reasoning 结束到第一个 answering token 的延迟。现有 LLM serving framework 没有区分 decoding 内部的 reasoning phase 和 answering phase，在 GPU KV cache 内存受限时会用同一套阻塞或抢占策略处理两类阶段，导致 TTFT 尾延迟恶化或 answering SLO 被破坏。
    - 论文的主要贡献：PASCAL 提出 phase-aware scheduling，把 decoding 阶段显式拆成 reasoning phase 和 answering phase，并按二者对中断的敏感性分配不同调度策略。它由 instance-level scheduler 和 intra-instance scheduler 两级组成：前者决定请求应进入哪个 GPU serving instance，并在 reasoning-to-answering 边界做动态迁移；后者在单个 instance 内维护 high-priority reasoning queue 和 low-priority answering queue，用优先级、RR、token pacing 和有条件降级来兼顾 TTFT 与 answering QoE。
    - 论文所处背景：LLM serving 通常受 TTFT 和 TPOT 两个用户体验指标约束。传统模型中 prefill 计算密集、decode 内存带宽受限，因此已有系统可以围绕 prefill/decode 做 disaggregation、batching 或 SLO-aware scheduling。reasoning 模型改变了这个映射：reasoning tokens 虽然属于 decode，但会阻塞第一个用户可见答案，因此 reasoning latency 直接进入 TTFT；answering tokens 才是用户流式看到的内容，其目标不是绝对越快越好，而是达到足够稳定的输出速率。

2、相对 Baseline 解决的问题与设计方法：
    - Baseline 的具体问题：论文对比的 FCFS 是 vLLM 默认策略，请求按到达顺序服务，KV cache 超过 GPU 内存时会 spill/offload 到 CPU 或暂停新请求接纳。FCFS 的缺陷是 head-of-line blocking：长 reasoning 请求占着 KV cache，短请求必须排队，短请求 TTFT 在 characterization 中最高可比 oracle 增加 5.14 倍。RR baseline 通过固定 token quantum 抢占请求，能缓解阻塞，但对 reasoning phase 不友好；长 reasoning 请求会反复被抢占、等待和恢复，2048 reasoning tokens 时 latency 最高比不中断 oracle 增加 1.75 倍。两者共同问题是都不理解 reasoning 和 answering 的不同 SLO 语义。
    - 瓶颈来源：主要瓶颈来自 GPU KV cache 内存容量和调度引发的内存驻留/迁移/抢占行为，而不是单个 attention kernel 的算子优化。GPU 内存不足时，系统必须在 blocking、preemption、CPU offload、KV cache migration 之间取舍；这些操作对 reasoning phase 会直接拉长 TTFT，对 answering phase 则主要影响 TTFAT 和 TPOT/QoE。
    - 论文的设计方法：PASCAL 的核心是 “reasoning-first, answering-paced”。reasoning 请求进入 high-priority queue，优先获得 GPU memory 和执行机会，以减少 TTFT 中最敏感的一段；answering 请求进入 low-priority queue，用 RR 加 token pacer 控制输出速率，只要满足 QoE 阈值即可。instance-level scheduler 对 reasoning 和 answering 使用不同选择算法：reasoning 选 SLO 正常且总 KV footprint 最小的 instance；answering 在 phase transition 时选 reasoning 请求最少的 instance，必要时考虑未耗尽首个 quantum 的 fresh answering 请求数量。
    - 方法如何对冲 Baseline 缺陷：对 FCFS，PASCAL 避免让 answering 或长请求长期堵住短 reasoning 请求，使 reasoning phase 能跨所有请求保持全局高优先级。对 RR，PASCAL 避免无差别公平抢占，把抢占主要留给对中断更能容忍的 answering phase，并用 token pacer 平滑用户可见输出。对多 instance 场景，PASCAL 在 reasoning 结束检测到特殊 token（如 DeepSeek-R1 的 end-of-thinking token）后，重新选择 answering instance，降低 reasoning 和 answering 在同一 instance 上争抢 KV cache 的概率。
    - 关键 trade-off：PASCAL 接受 answering 请求被 reasoning 请求抢占，因此少数请求的 TTFT 或 answering latency 可能轻微退化；论文报告 AlpacaEval2.0 中最坏个例相对 FCFS 增加 13.30 秒、相对 RR 增加 8.25 秒。跨 instance migration 还会带来 KV cache 传输开销，但论文测得高负载下 P99 迁移延迟为 0.14 秒（AlpacaEval2.0）和 0.25 秒（Arena-Hard），相对数秒到数百秒级 TTFT 较小。另一个 trade-off 是策略依赖 phase boundary 检测，无法像 prefill/decode disaggregation 那样提前重叠迁移。

3、论文实现：
    - Baseline 如何实现：论文 characterization 在一台 Intel Xeon Platinum 8558 CPU、256 GB DDR5、NVIDIA H100 96 GB HBM、PCIe 5.0 的服务器上使用 vLLM v0.6.1。FCFS 使用 vLLM 默认语义；RR 使用固定 token quantum 的抢占式循环调度。为了制造 GPU memory pressure，实验把可用于 KV cache 的 GPU 内存限制为 oracle 容量的 50%，迫使 FCFS blocking 或 RR preemption/offload 发生。
    - 新设计如何实现：论文没有给出可下载系统代码，而是实现了 cluster-level simulator 来评估 PASCAL 的设计点。PASCAL 在模拟器中建模为八个 serving instances，每个 instance 内有高/低优先级队列、RR dispatch、token pacer、phase transition monitor、KV cache migration 和 adaptive migration 逻辑。单 instance 行为基于 vLLM profiling data，模拟器用真实系统测量延迟校准。
    - 实验 / 实现平台：cluster-level simulator 建模 8 个 server nodes/instances，实例之间通过 100 Gbps fabric 连接，每个 instance 配一块 NVIDIA H100 96 GB GPU。单实例验证平台为 Intel Xeon Platinum 8558 CPU + H100 GPU（PCIe 5.0），软件栈包括 vLLM 0.6.1、PyTorch 2.4.0、CUDA 12.1。模拟器相对实测的 MAPE 为 end-to-end latency 1.62%、mean TTFT 12.6%、TPOT 6.49%。
    - 模型与 workload：主评估模型是 DeepSeek-R1-Distill-Qwen-32B。serving traces 来自 AlpacaEval2.0 和 Arena-Hard prompts，并通过 OpenAI o4-mini API 得到 reasoning/answering token counts。额外讨论中还构造了 50% Arena-Hard + MATH-500/GPQA/LiveCodeBench reasoning-heavy 请求的混合场景。
    - 关键实验设置与指标：RR 和 PASCAL 每个 queue 的 token quantum 都设为 500；PASCAL 中 reasoning tokens 超过 5000 的请求会降级到 low-priority queue。TTFT 定义为从请求提交到第一个 answering token 的延迟。answering SLO 用 QoE 评估，QoE 从第一个 answering token 后的 TPOT 计算，QoE < 0.95 记为 violation；characterization 中 TTFAT target 为 0.25 秒，TPOT target 为 100 ms。
    - 主要结果：相比 FCFS，PASCAL 在 AlpacaEval2.0 上最高降低 tail TTFT 61%，在 Arena-Hard 上最高降低 72%，绝对降低分别为 43.22 秒和 64.21 秒。相比 RR，PASCAL 最高降低 tail TTFT 89.91 秒和 72.73 秒，对应 33% 和 29%。吞吐量与 baseline 差异不超过 3%。禁用迁移的 PASCAL(NoMigration) 会让 phase transition 后的 P99 blocking latency 最高达到 27.39 秒，而完整 PASCAL 将其保持在接近 0。关闭 adaptive migration 的 PASCAL(NonAdaptive) 在高到达率下 SLO violation 达 7.45%，完整 PASCAL 为 0.69%，且 NonAdaptive 的 median end-to-end latency 和 tail latency 分别恶化 20.1% 和 9.7%。
    - 实验限制与不确定性：评估主体是 profile-based simulator，而不是公开的端到端生产 serving 系统。论文验证了模拟器与单机实测的一致性，但没有公开代码或 artifact 页面，因此复现实验需要自行实现调度器和模拟框架。PASCAL 的收益也与 workload 中 reasoning/answering 长度比例相关；在 answering 很短、reasoning 很长的问题求解 workload 上，phase-aware 隔离的边际收益会下降，虽然仍能在部分 token range 降低 tail TTFT。

4、pipeline/kernel 解析：
    - 新 pipeline/kernel 是什么：论文没有提出新的 GPU kernel 或 attention operator。它提出的是面向 reasoning-based LLM serving 的 phase-aware request scheduling pipeline：跨 instance 的 phase-aware placement/migration pipeline + instance 内 high/low priority queue execution pipeline。其执行对象是 request 及其 KV cache，而不是 tensor tile 或单个 CUDA kernel。
    - 新 pipeline/kernel 的执行流例子：一个请求到达后，系统先把它视为 reasoning request。instance-level scheduler 读取各 instance 的 token pacer 状态和 KV footprint；若某 instance 的 answering 请求已经低于 SLO，则优先排除，否则在候选中选择 KV cache 占用最小的 instance。请求进入该 instance 的 high-priority queue，执行 prefill 和 reasoning-token decoding。instance monitor 持续检查模型输出，一旦检测到 reasoning 结束特殊 token，请求进入 phase transition。此时 scheduler 运行 answering placement：排除 answering SLO 已失败的 instance，在候选中选 high-priority reasoning 请求最少的 instance；若没有 SLO 正常候选，则选 ri + ai 最小的 instance，其中 ai 表示 low-priority queue 中尚未耗尽首个 time quantum 的 answering 请求数。
    - phase transition 与 KV cache 流动：如果 answering 目标 instance 与当前 instance 不同，请求的 KV cache 需要从源 instance 传到目标 instance；但 adaptive migration 会检查当前 instance 与目标 instance 的 GPU memory 可用性。若当前 instance 有足够空间而目标 instance 没有，PASCAL 会覆盖 Algorithm 2 的迁移决定，让请求留在当前 instance，避免不必要 offload、等待和 KV cache transfer。若迁移确实发生，目标 instance 收到 KV cache 后将请求放入 low-priority queue。
    - answering 执行流：low-priority answering queue 使用 RR 调度，利用剩余 GPU memory 执行 answering token generation。token pacer 负责按用户期望速率释放/缓冲 tokens，使即使后台执行被抢占，用户看到的 token stream 仍尽量稳定。若 high-priority reasoning queue 需要内存，answering 请求可被抢占或延后；但由于 QoE 是阈值型指标，只要 TTFAT 和 TPOT 仍在目标范围内，用户体验不会显著下降。
    - 一个具体请求的简化轨迹：Request X 到达，Algorithm 1 把它放到 Instance 0，因为 Instance 0 的 answering SLO 正常且 KV footprint 最小。X 在 high-priority queue 中生成 reasoning tokens；期间若 GPU memory 紧张，PASCAL 仍尽量优先保留 reasoning 执行机会，以降低 X 的 TTFT。X 输出 end-of-thinking token 后，Algorithm 2 发现 Instance 2 的 reasoning 请求最少，但 adaptive migration 发现 Instance 2 GPU memory 已满、Instance 0 仍有空位，于是 X 不迁移，直接在 Instance 0 的 low-priority queue 中开始 answering。随后 token pacer 控制 X 的用户可见 token stream，使输出速度达到 100 ms/token 左右的体验目标。

5、读后判断：
    - 这篇论文的关键价值不在于提出更快的 decode kernel，而在于重新定义 reasoning LLM serving 的调度目标：reasoning tokens 虽然用户不可见，但决定 TTFT；answering tokens 用户可见，但只需要满足稳定流式输出阈值。这个观察让调度策略从“所有 decode token 同质”变成“同一 decode stage 内按语义 phase 区分优先级”。
    - 对系统实现者最有用的点是 Algorithm 1/2 与 adaptive migration 的组合。单纯把 reasoning 放高优先级会造成 answering 饥饿；单纯迁移 answering 会造成无谓 KV cache transfer 和目标 instance memory contention。PASCAL 把 token pacer 的 SLO 状态、KV footprint、reasoning queue occupancy、fresh answering count 和当前/目标 instance memory availability 结合起来，形成了一个相对轻量但有明确系统语义的策略。
    - 主要风险是部署依赖真实 serving framework 能否低开销暴露 phase boundary、token pacer 状态、KV cache residency、CPU/GPU KV transfer 和跨 instance migration。论文中这些能力主要在模拟器中建模；若落到 vLLM/SGLang 等实际系统，还需要处理调度器并发、KV block ownership、迁移一致性、batch rebuild 和网络传输调度等工程细节。
