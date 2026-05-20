## Efficient Multimodal Serving via Module Multiplexing

- 属于Serving调度的实现是什么？实验比较什么？
  论文实现EEVEE，一个基于module multiplexing的多模态模型serving系统（约5,000行Python，基于vLLM + NVIDIA MPS）。核心Serving调度实现包含：(1) Module-level Scheduling：将多模态模型（如BLIP的visual encoder、text encoder、text decoder）拆为独立进程，每个模块配置独立的batch size和SM allocation，不再使用统一batch顺序执行；(2) Stage-level Parallelism：将预处理和推理作为可独立调度的阶段，visual preprocessing与text inference可重叠执行，减少GPU等待CPU的空闲；(3) Request-aware Reuse via Modal Cache：对encoder-decoder模型（BLIP）缓存cross-attention消费的视觉token KV pairs，对decoder-only MLLM（LLaVA）缓存decoder消费的视觉token KV pairs；支持modal cache compression（按attention score剪除低重要性token）；(4) Synergistic Scheduling Algorithm：离线greedy search算法从monolithic顺序执行初始化，逐步增加各模块batch size，将增量放在对batch latency负面影响最小的stage，用balanced SM allocation将SM从短latency模块转移到长latency模块；(5) Controller：每GPU上的controller进程管理GPU shared memory、模块间通信（CUDA IPC）、参数read-only copy共享、cache LRU eviction和host memory spill；(6) 多GPU支持：大模块使用intra-operator model parallelism、小模块使用data parallel replication。实验比较Triton（monolithic graph、统一batch顺序执行）和Gpulet（spatio-temporal GPU sharing，但将各模块视为独立竞争模型，无协同调度和cache reuse）。五种模型：CLIP、BLIP、BLIP-2、LLaVA-1.5、InternVL2.5-8B。三种任务：VQA (VQA2)、NLVR (NVLR2)、image-text matching (COCO)。工作负载：Poisson distribution + Azure trace采样。指标：max capacity (queries/sec under SLO)、average/P99 latency、SM active ratio。消融：stage-level parallelism / module-level scheduling / request-aware reuse分别提升capacity 146%/152%/116%。调度算法对比simulated annealing、genetic algorithm、random greedy，相同搜索步数下提升最高30.10%。

- 硬件平台是什么，配置是什么。
  高端服务器：Intel Xeon Gold 6230R CPU (104 cores)、503GB内存、8×NVIDIA A100 40GB GPU。低端服务器：Intel Xeon E5-2698 v4 CPU (80 cores)、377GB内存、8×NVIDIA GeForce RTX 3090 GPU。所有GPU通过PCIe 3.0连接CPU，峰值带宽32GB/s。软件环境：Ubuntu 20.04.5、Docker 23.0.1、CUDA 11.8、Python 3.8、PyTorch 2.0.1。GPU利用率使用NVIDIA Nsight的SMs Active指标测量。

- 开源Serving框架是什么。修改了什么。
  基于vLLM作为后端inference engine。论文未修改vLLM源码，而是在vLLM之上构建module multiplexing serving layer：(1) 为多模态模型的每个模块启动独立的vLLM process，通过CUDA环境变量（MPS active thread percentage）在CUDA初始化前配置不同的SM allocation；(2) 使用NVIDIA MPS实现模块间GPU资源并发共享（而非MIG，因MIG仅提供刚性分区且仅限最新GPU）；(3) 单机内模块间通信使用shared memory + CUDA IPC handler；(4) 每GPU上运行controller进程管理共享GPU memory、模块输出缓存（64-bit hash索引、per-module hashmap、global LRU eviction、host memory spill）。Gpulet baseline同样集成到vLLM上，但将每个模块当做独立模型并强制共享固定batch size，无module-level协同调度。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源地址：论文未提供开源仓库URL，正文和参考文献中未找到EEVEE官方GitHub、artifact appendix或复现包。相关公开页面：https://2026.eurosys.org/papers.html（EuroSys 2026 accepted papers）；https://zicongs-homepage.webflow.io/publications（作者主页）；https://doi.org/10.1145/3767295.3769389（ACM DOI）。论文仅说明原型约5,000行Python、基于vLLM和NVIDIA MPS实现。截至分析时无法确认EEVEE已公开官方实现。
  以BLIP VQA serving为例（基于论文描述和vLLM+MPS文档重建）：
  1. 部署：启动vLLM backend→为visual encoder、text encoder、text decoder各启动一个vLLM process→配置CUDA_MPS_ACTIVE_THREAD_PERCENTAGE为各模块分配不同SM份额→启动controller进程管理GPU shared memory→scheduler加载离线生成的multiplexing strategy。
  2. 请求到达：用户提交图像+问题Q1→front-end将请求送入scheduler→scheduler按离线策略决定：visual encoder以batch size较小的配置处理图像（例如2请求/batch，70% SM），同时text decoder以较大batch size继续处理上一批请求的文本解码。
  3. Visual encoder处理图像生成visual tokens→controller将输出存入GPU shared memory，以64-bit hash索引→标记为可复用modal cache→超额时按LRU evict到host memory。
  4. Text encoder/decoder消费visual tokens和问题文本→生成答案A1。此时同一GPU上visual encoder和text decoder通过MPS并发执行，SM按离线策略动态分配。
  5. 同一图像的后续问题Q2到达→controller根据hash查找modal cache命中→从GPU或host memory加载缓存的visual tokens/KV cache→跳过visual encoder重计算→cache loading可与新图像的encoder compute overlap。
  6. 若GPU memory紧张→使用compressed critical modal cache（按attention score剪除30%低重要性token）→优先放入GPU memory以支持快速访问→full modal cache保留在host memory备用。
  7. 效果：相比Triton（monolithic serving）在3090 server Azure workload上平均max capacity提升157%，高负载LLaVA平均latency降低约90%，BLIP VQA GPU active SM接近90%。

