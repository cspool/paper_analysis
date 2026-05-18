## Module Multiplexing（模块多路复用）

术语是什么？

Module Multiplexing 是 EEVEE 提出的一种多模态模型 serving 范式，将多模态模型中的 modality-specific modules（如 visual encoder、text encoder、text decoder、LLM backbone）作为独立可调度单元，各自拥有独立的 batch size 和 GPU SM allocation，在同一 GPU 上通过 NVIDIA MPS 实现空间多路复用并发执行。与传统的 monolithic serving（将模型所有层/模块视为单一计算图、统一 batch size 顺序执行）不同，Module Multiplexing 将 serving 粒度从"模型"下沉到"模块"，通过 module-level scheduling、stage-level parallelism 和 request-aware modal cache reuse 三项机制提升 intra-GPU utilization 和 system throughput。

从系统架构角度拆解术语：

Module Multiplexing 在 EEVEE 中的运转流程：

1. **Offline Profiling & Strategy Generation**：对给定的（模型结构、任务 SLO、GPU 硬件、workload pattern）组合，scheduler 使用 synergistic greedy search 算法生成最优 multiplexing strategy。策略定义为 s = {R_O}_O∈M，其中 R_O = {(M_O^P, N_O^P)}_P∈M 表示在每个模块 O 的阶段中，哪些模块 P 并发执行、各自的 batch size M 和 SM 百分比 N。约束包括：(1) per-stage SM 总和 = 100%、(2) 各模块 batch size 为 egress module batch 的整数倍（由模型结构和用户输入模式决定）、(3) end-to-end latency ≤ SLO。

2. **Module-Level Scheduling**：在线部署时，为多模态模型的每个模块启动独立的 vLLM process，通过 CUDA 环境变量（MPS_ACTIVE_THREAD_PERCENTAGE）在 CUDA 初始化前配置各自的 SM allocation。不同模块 process 通过 NVIDIA MPS 在同一 GPU 上并发执行——例如 visual encoder 分配 70% SM 并以较小 batch（如 B=2）处理图像，text decoder 分配 30% SM 并以较大 batch 处理文本生成。模块间通过 GPU shared memory + CUDA IPC handler 传递中间结果。

3. **Stage-Level Parallelism**：将每个模块进一步拆分为 preprocessing 和 inference 两个可独立调度阶段。如 visual encoder 的图像预处理（resize/normalize/H2D transfer，可达 inference 时间的 40%）可与 text decoder 的 inference 重叠执行，利用 pipeline 并行度避免 GPU 等待 CPU 完成预处理。

4. **Request-Aware Reuse via Modal Cache**：通过 modal cache 缓存 visual encoder 输出的 visual tokens 或对应的 KV cache。当同一图像对应多个问题时（VQA2 中单图至少 3 个问题，最多 246 个），后续问题直接加载缓存，跳过 visual encoder 重计算。Cache loading 通过 CUDA streams 与新请求的 computation kernel 重叠执行。

5. **Synergistic Scheduling Algorithm**：离线 greedy search 从 monolithic 顺序执行的可行策略初始化，逐步增加各模块 batch size，每次将增量放在对 batch latency 负面影响最小的 stage。SM reallocation 使用 balanced SM allocation heuristic——从最短 latency 模块转移 SM 到最长 latency 模块直至 stage 内各模块 latency 趋近平衡。算法复杂度 O(|M|² · max_batch · 100)，远小于全搜索空间的暴力枚举。

6. **Dynamic Strategy Switching**：当请求负载变化超过阈值，通过 active-standby 切换机制（模型权重已驻留 GPU memory，仅更新 scheduler metadata）在毫秒级切换到预计算的其他策略。若无线接近的策略匹配，回退到 conservative default strategy。

关键约束：CUDA MPS 的 SM allocation 必须在 CUDA 初始化前设定且运行期间不可更改，因此 EEVEE 的 strategy 是"部署时固定、按需切换"而非真正的"运行时动态调节每模块 SM 份额"。

术语一般如何实现？如何使用？

EEVEE 原型约 5,000 行 Python 实现，基于 vLLM 作为后端 inference engine + NVIDIA MPS 实现 GPU spatial sharing。MPS 通过设置环境变量 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE 在 CUDA 初始化前配置各 process 的 SM 份额。多 GPU 场景下，大模块（如 LLM backbone）使用 intra-operator model parallelism 跨 GPU 拆分，小模块（如 encoder）使用 data parallel replication。模块间通信：GPU shared memory + CUDA IPC handler（单机内），论文显示 shared memory 传输显著降低 inter-module transfer latency。论文在 CLIP、BLIP、BLIP-2、LLaVA-1.5 和 InternVL2.5-8B 上验证，相对于 Triton（monolithic serving）在 RTX 3090/A100 上提升 max capacity 平均 157%，高负载 LLaVA 平均 latency 降低约 90%，BLIP GPU active SM 达到接近 90%。未找到官方开源仓库。与 Gpulet 的对比：Gpulet 支持多模型 spatio-temporal GPU sharing 但将各模块视为独立竞争模型，无模块间协同调度和 modal cache，论文实验显示 EEVEE 显著优于 Gpulet。

涉及论文标题：
- Efficient Multimodal Serving via Module Multiplexing

---
