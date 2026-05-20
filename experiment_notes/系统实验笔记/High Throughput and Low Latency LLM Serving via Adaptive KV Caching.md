## High Throughput and Low Latency LLM Serving via Adaptive KV Caching

- 属于Serving调度的实现是什么？实验比较什么？
  论文提出eLLM，一个面向高吞吐和低延迟LLM serving的adaptive KV caching系统，基于vLLM扩展实现（约3,500行Python代码和1,700行CUDA kernel-level优化代码）。核心Serving调度实现包含：(1) request-level optimizer：scheduler启动时收集实时request metadata（sequence length、队列大小、最大等待时间），用SciPy SLSQP求解受GPU显存和TPOT SLO约束的优化问题，输出近似最优batch size b与uncached token ratio r；(2) token-wise/layer-wise KV cache管理：将vLLM的KV block从所有layer/多token放在一个物理block重塑为按F个连续layer划分的更小单元，维护seq_id/token_id/layer_id/logical_block_id/physical_block_id/#filled等map table元数据，每个SequenceGroup维护自己的uncached-token ratio r，较新的(1-r) token保留GPU KV cache，较早的r token释放显存（或swap到host memory）；(3) layer-wise execution pipeline：使用两个CUDA stream实现host-GPU KV传输与GPU重算并行，将下一层未缓存token的KV recomputation K1与当前层当前token的decode K2融合为fused kernel；(4) closed-loop adaptation：layer-level计算overlap/fusion额外显存开销Mo后反馈给request-level optimizer，联合优化缓存比例、batch size和fused kernel线程分配。实验比较：(a) baseline对比：vLLM-Recompute（显存满时丢弃整请求KV cache后整体重算）、vLLM-Swap（preempted request KV cache整体swap到host memory再通过PCIe恢复）、HCache（缓存hidden states加速状态恢复），均集成到vLLM中复现；(b) ShareGPT数据集上Llama2-13B单卡throughput分别提升2.64×/2.61×/1.91×，Llama2-70B 4卡TP分别提升2.0×/2.0×/1.6×；(c) TTFT最多降低2.63×，TPOT SLO attainment达97.3%-98.6%；(d) L-Eval长上下文上平均只缓存约53% prefix length，memory saving超47%，最高throughput提升3.03×，TTFT降低1.79×；(e) ablation显示禁用Comm-Com Overlapping或Kernel Fusion均退化，但保留token-wise caching后仍优于多数baseline；(f) 系统overhead：request-level optimizer 90%耗时<10ms，map table layer lookup平均<0.015ms。

- 硬件平台是什么，配置是什么。
  主实验服务器：4×NVIDIA A100-80GB GPU通过PCIe 4.0 x16连接（无NVLink）；CPU为96-core Intel Xeon Gold 6342 @ 2.80GHz，host memory 256GB。软件栈：Docker环境，CUDA 12.4，NVIDIA Driver 550.107.02。Llama2-13B使用1张A100，Llama2-70B使用4张A100以tensor parallel运行。host memory为Llama2-13B分配40GB，为Llama2-70B分配160GB存放swapped KV caches。

- 开源Serving框架是什么。修改了什么。
  基于vLLM开源Serving框架和PagedAttention机制。修改内容包括：(1) KV block结构重塑：从原有所有layer/多token共享物理block改为按F个连续layer划分的layer-granular block（默认F=4），降低粗粒度block内部碎片；(2) 新增map table元数据结构：维护seq_id、token_id、layer_id、logical_block_id、physical_block_id、#filled等，支持部分token/部分layer的缓存、swap和recompute精确定位；(3) request-level optimizer组件：引入SciPy SLSQP在线求解batch size b与uncached token ratio r的约束优化；(4) layer-level异步执行：引入torch.cuda.stream双CUDA stream管理，预编译多组CUDA shared libraries（32到1024线程、步长32）；(5) 保留vLLM核心API以兼容第三方模型。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源状态：论文正文提及基于vLLM扩展实现，作者实验室页面（https://cds-macau.github.io/publication/conference-paper/ellm/）确认论文发表于EuroSys 2026，但截至本次分析未找到eLLM官方GitHub仓库或artifact appendix。使用例子（以Llama2-13B在A100上处理ShareGPT请求为例）：
  1. 请求到达后进入vLLM等待队列，eLLM scheduler启动request-level optimizer：收集queue中所有请求的sequence length、队列大小、最大等待时间以及当前GPU显存占用和FLOPS信息。
  2. SLSQP求解约束优化（Eq.5）：maximize throughput = b / (compute_time(b,r) + overhead)，subject to GPU memory ≤ M_GPU + M_saved(r) - M_overhead(b,r)，且predicted TPOT ≤ SLO阈值。输出本轮batch size b和uncached token ratio r。
  3. 对batch中每个请求，若r=0.4，则其较早的40%历史token不再长期保留KV cache（可swap到host或释放），较新的60% token保留在GPU layer-granular block中。map table追踪每个token在每个layer的缓存状态。
  4. 进入transformer layer i时：layer-level scheduler通过map table定位cached token的物理block，对uncached token根据预算确定在本层是recompute还是从host swap。Stream A传输cached KV到SM，Stream B执行fused kernel K1（为layer i+1旧uncached token生成KV）+ K2（用layer i完整历史KV对新token decode attention）。完成后临时重算的旧token KV立即释放，新token KV写入对应layer-granular block。
  5. layer-level完成后计算实际overlap/fusion额外显存Mo，反馈给request-level optimizer，下一轮可能进一步增加b或降低r，形成closed-loop adaptation。

