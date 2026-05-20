## Towards Resource-Efficient Serverless LLM Inference with SLINFER

- 属于Serving调度的实现是什么？实验比较什么？
  提出SLINFER，一个面向small- to mid-sized LLM的resource-efficient serverless inference方案，实现异构硬件（CPU+GPU）上的弹性、按需资源共享。核心实现包含三大子系统：(1) Headroom-Driven Compute Subsystem：基于request headroom (公式1: headroom=ST+TTFTSLO+TPOTSLO·O-CT) 的token-level调度，每次调度cycle选最短headroom的instance执行一个iteration，通过shadow validation虚拟添加新请求并模拟future compute流程来避免SLO violation，使用linear interpolation做prefill time quantification、2D linear interpolation (batch size×token length) 做decode time quantification；(2) Hazard-Aware Memory Subsystem：watermark-based KV-cache scaling（early scale-up + lazy scale-down，watermark w=25%），optimistic budgeting + pessimistic scheduling的inter-instance orchestration机制避免OOM，intra-instance scaling compromise当full scale-up不可行时降级为Mrequire而非Mrecommend，极端memory不足时evict最长headroom请求；(3) Efficiency-Oriented Consolidator：proactive consolidation with preemption（允许大batch instance preempt小batch邻居来scale-up）+ reactive consolidation with bin-packing（新请求优先路由到大batch instance，加速小batch instance回收）。实验在4×32-core Intel Xeon 6462C CPU + 4×A100-80GB GPU上，使用Llama-3.2-3B/Llama-2-7B/Llama-2-13B模型，Azure LLM Conversation Trace + Azure Serverless Trace workload，SLO为TTFT ≤ min(max(0.5, L/512), 8)s、TPOT ≤ 0.25s。对比baselines：ServerlessLLM (sllm, GPU-only)、sllm+c (加CPU support)、sllm+c+s (加CPU+time-sharing)。主要指标：Nodes Used、Decode Speed、SLO-met Req、TTFT CDF、memory utilization CDF、batch size CDF。SLINFER在serving 128 models时SLO-met requests比sllm提升86%-154%，比sllm+c提升47%-62%，比sllm+c+s提升18%-70%。

- 硬件平台是什么，配置是什么。
  4×32-core Intel Xeon 6462C @3.3 GHz CPU（第4代Xeon，支持AMX）+ 4×NVIDIA A100-80GB GPU，逻辑上分离为两个物理节点各2 GPU。CPU用于独立inference（通过OpenVINO 2024.6.0）或辅助GPU offloading。Ubuntu 22.04, CUDA 12.4, Conda。

- 开源Serving框架是什么。修改了什么。
  基于ServerlessLLM [26] (OSDI'24) 和 vLLM 0.5.2 [37] (SOSP'23) 修改。主要修改：(1) 新增Headroom-Driven Compute Subsystem：实现token-level调度器替代vLLM默认continuous batching，每cycle选最短headroom instance执行一个iteration，shadow validation仿真prefill/decode timeline检查SLO violation风险；(2) 新增Hazard-Aware Memory Subsystem：在paged-attention基础上增加watermark-based KV-cache scaling逻辑、optimistic budget + pessimistic reservation station机制协调多instance并发memory操作；(3) 新增Efficiency-Oriented Consolidator：preemption-based proactive consolidation和bin-packing-based reactive consolidation；(4) CPU backend集成OpenVINO 2024.6.0替代vLLM GPU backend用于CPU inference；(5) 统一硬件抽象层：将CPU/GPU节点统一为resource pool，proxy层路由请求优先到CPU instance，CPU无法满足SLO时fallback到GPU。源码约需200GB磁盘空间，完整实验约26小时。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  已开源：GitHub https://github.com/BarrinXu/SLINFER，Zenodo DOI: 10.5281/zenodo.17846442。使用流程（以3B模型实验为例）：
  1. 环境准备：每台GPU机器安装Conda环境SLINFER-GPU (Python 3.11)，安装ServerlessLLM model loader (`sllm-store`)、modified vLLM (`pip install -e .`)、transformers 4.46.3等依赖。每台CPU机器安装Conda环境SLINFER-CPU，安装OpenVINO版modified vLLM。
  2. 模型下载：从HuggingFace下载Llama-3.2-3B-Instruct/Llama-2-7b-chat-hf/Llama-2-13b-chat-hf到`$PROJECT_BASE/huggingface_models/`。
  3. 启动系统：GPU机器上启用NVIDIA MPS (`nvidia-cuda-mps-control -d`)，启动4个GPU instance wrapper（每GPU一个，如`python vllm_batch_starter.py --model llama-3.2-3b --device gpu --worker_num 8 --port 8000 --gpu 0`），启动ServerlessLLM model loader (`sllm-store-server`)，启动root gateway (`python gateway.py`)。CPU机器上各启动CPU instance wrapper (`--device aliyun --cpu_kv_gb 16`) 和dist gateway。
  4. 运行实验：`python test_3B_extreme_lite.py` (26分钟快速测试) 或 `python test_3B_full.py` (396分钟完整测试)，生成JSON结果后用`python draw.py`生成PDF图表。
  5. SLINFER作用：event-driven请求到达时，compute subsystem通过shadow validation选择合适instance（优先CPU），headroom-based token-level调度在instance间轮转执行iteration；memory subsystem在请求加入/完成时动态scale KV-cache（early scale-up预留watermark空间，lazy scale-down减少ping-pong）；当instance无法scale-up容纳新请求时，consolidator尝试preempt小邻居或创建新instance并bin-packing路由后续请求。

- 属于Serving调度的实现是什么？实验比较什么？
  提出TetriServe，一个面向DiT (Diffusion Transformer) serving的deadline-aware round-based调度系统。核心实现包含：(1) Deadline-Aware Round-Based Scheduler：将连续时间离散化为固定时长round，每个round内通过offline profiling的cost model确定每个请求在最小GPU hour消耗下满足deadline所需的最少GPU数量（step-level sequence parallelism），然后用动态规划(DP)进行request packing以最小化下一round将超时的请求数；(2) GPU Placement Preservation：跨round保持请求在同一GPU集合上执行，消除状态迁移延迟；(3) Work-Conserving Elastic Scale-Up：利用placement后空闲的GPU给有余量steps的请求增加并行度；(4) Selective Continuous Batching：仅对相同小分辨率请求进行step级batching，不牺牲SLO；(5) VAE Decoder Sequential Execution：顺序执行VAE decoder以避免高分辨率下大batch的activation memory峰值。实验比较SLO Attainment Ratio (SAR)、end-to-end latency CDF、平均并行度、对arrival rate/step granularity/resolution的sensitivity，对比baselines：xDiT (SP=1/2/4/8固定并行度)和RSSP (Resolution-Specific SP, 对每种分辨率选最优固定SP)。

- 硬件平台是什么，配置是什么。
  两个GPU集群：（1）8×NVIDIA H100-80GB HBM3, NVLink 4.0 (900 GB/s inter-GPU bandwidth)；（2）4×NVIDIA A40-48GB, GPU对之间NVLink连接, host通过PCIe 4.0连接。软件环境：NVIDIA NGC container, CUDA 12.5, NCCL 2.22.3, PyTorch 2.4.0。

- 开源Serving框架是什么。修改了什么。
  基于xDiT (git-hash 8f4b9d30) 的sequence parallelism engine，复用vLLM的async logic和MuxServe/SGLang的process launcher。TetriServe在xDiT之上新增5,033行Python和C++代码实现：C++编写的scheduler核心决策循环（达到毫秒级控制面延迟）、Round-based DP调度算法、Future-like latent transfer抽象（异步非阻塞传输中间latent）、NCCL communication process groups预warmup策略、以及selective continuous batching逻辑。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源地址：https://github.com/DiT-Serving/TetriServe。以FLUX.1-dev serving为例：300个prompt从DiffusionDB采样，以Poisson过程按12 req/min到达。请求包含四种分辨率(256×256, 512×512, 1024×1024, 2048×2048)和对应SLO deadline（1.5s/2.0s/3.0s/5.0s）。TetriServe在8×H100上运行：Scheduler在每个round开始时，根据offline profiled的cost model查表获取每种(分辨率, GPU数)组合的单步耗时，为每个请求计算满足deadline的最小GPU分配，再通过DP pack requests入round。Execution Engine的8个GPU worker执行分配的diffusion steps，Latent Manager处理中间latent。请求完成后VAE decoder顺序执行解码。对比xDiT固定SP=1/2/4/8：在Uniform workload下TetriServe SAR平均高出10%（tight SLO 1.1×时高出28%），Skewed workload下平均高出15%（1.2× SLO时高出32%）。

- 属于Serving调度的实现是什么？实验比较什么？
  提出Bullet，一个面向LLM serving的intra-GPU prefill-decode spatial-temporal orchestration系统。核心实现包含四个模块：(1) Performance estimator：基于SM-scaling roofline model (SRM) 建模compute/memory/network bandwidth随SM数量变化的性能上界，用少量offline concurrent sample校准prefill/decode interference，在线统计持续修正；(2) SLO-aware task scheduler：周期性读取全局状态，预测TTFT/TPOT，重排waiting requests，为下一批prefill layer或decode step搜索合适SM分区；(3) Resource manager：通过libsmctrl_set_stream_mask修改CUDA stream metadata，使后续kernel限制在指定SM子集执行；(4) Concurrent execution engine：prefill和decode放在独立进程/worker中，共享CPU metadata buffer、统一GPU memory pool (CUDA IPC)、ZeroMQ metadata传递来协同执行。prefill以layer为粒度运行并同步回CPU决策，decode使用CUDA Graph发射一个step的小kernel。实验比较TTFT、TPOT、throughput、P90 SLO attainment、normalized input/generation latency以及Nsight Systems采集的SM/Tensor Core/memory bandwidth utilization，对比vLLM v0.8.5 (1024 chunk)、SGLang v0.4.6+FlashInfer v0.2.7 (1024/2048 chunk)、NanoFlow (1024 chunk)和基于SGLang+Mooncake的xP-yD disaggregated-prefill配置。

- 硬件平台是什么，配置是什么。
  三类服务器：(1) 8×A100-80GB GPU（108 SM/GPU，NVLink 600 GB/s）；(2) 8×H100 GPU（132 SM/GPU，600 GB/s）；(3) 8×H20 GPU（78 SM/GPU，intra-node 400 GB/s，inter-node 200 GB/s）。CUDA 12.4。Artifact Appendix的复现实验聚焦单张NVIDIA A100 80GB、Debian 5.10、CUDA 12.4、Python 3.12.9、PyTorch 2.6.0、CMake 3.17、GCC 10.2.1、约20GB磁盘。

- 开源Serving框架是什么。修改了什么。
  基于SGLang v0.4.6 + PyTorch 2.6.0修改，约4100行Python代码，集成修改版libsmctrl做GPU resource allocation。主要修改：(1) prefill/decode engine拆为两个SGLang worker进程，MPS用于spatial sharing；(2) GPU memory由初始化进程预先分配模型权重和KV cache，通过CUDA IPC在两个engine间共享；(3) CPU侧用OS-managed shared memory保存全局状态与metadata；(4) 请求metadata用ZeroMQ异步传递；(5) scheduler下发repartition command后，resource manager立即修改相应CUDA stream的SM mask；(6) prefill以若干layer为粒度运行并同步回CPU决策，decode使用CUDA Graph发射step。对比的disaggregated baseline基于SGLang+Mooncake构建xP-yD配置（如1P1D、3P1D、6P1D、30P6D）。开源地址：https://github.com/zejia-lin/BulletServe，Zenodo DOI: https://doi.org/10.5281/zenodo.17937105。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  已开源，GitHub仓库（zejia-lin/BulletServe）为研究原型。使用步骤：
  1. 环境准备：Debian 5.10，CUDA 12.4，Python 3.12.9，PyTorch 2.6.0，CMake 3.17，GCC 10.2.1
  2. 克隆仓库：`git clone https://github.com/zejia-lin/BulletServe`
  3. 构建libsmctrl等依赖
  4. 启动MPS服务：`nvidia-cuda-mps-control -d`
  5. 启动Bullet engine：通过`--enable-bullet-engine`和MPS启动参数运行
  6. 复现实验：运行`artifact_evaluation/run_all.sh`生成日志、JSON结果和图
  Bullet通过在同一GPU内将prefill和decode分配给不同SM子集并发执行，从机制上减少chunked prefill的重复KV reload和lock-step等待。例如在A100+Llama3.1-8B上，ShareGPT 20 req/s下mean TTFT 0.16s（相比SGLang-1024好54.9x），SLO compliance提升1.49x；SM active cycles 86.2%（比SGLang高11.2%），Tensor Core utilization高11.8%，memory-bandwidth utilization高19.3%。

