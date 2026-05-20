## TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  论文实现Hierarchical KV Cache Manager，一个基于多CUDA stream并行pipeline的KV cache运行时管理子系统，用于支持请求的抢占和恢复。核心kernel调度实现包含：(1) 三并行CUDA stream pipeline：compute stream（GPU推理计算）、write/load stream（KV cache在GPU显存与CPU memory间的传输）、evict stream（GPU显存释放）；(2) Write-through KV cache策略：每次decode iteration后将新生成的KV chunk放入write buffer，在下一轮计算前根据compute duration预估选择合适大小的chunk，通过write stream同步到host memory；(3) Synchronous chunked writing：动态chunk sizing和batched transfer，用CUDA events协调compute、write、load、evict四类操作的非阻塞执行；(4) Load-evict overlap：preempted请求已同步的chunk直接释放，未同步的剩余chunk与load操作重叠传输，减少上下文切换延迟。实验比较：(a) 消融实验：完整系统 vs w/o offload（127.28s vs 66.00s完成时间）、w/o write-through、w/o evict-load overlap；(b) 系统端到端性能：burst/Poisson/real trace场景下的effective throughput和TTFT。

- 后端平台是什么，配置是什么。
  NVIDIA H200、NVIDIA RTX 4090、NVIDIA A6000 GPU。micro experiment中报告Huawei Ascend 910B支持。H200设置mem-frac=0.3。GPU显存作为CPU memory上大容量KV cache的高速cache。

- 评估性能的软件/脚本是什么。修改了什么。
  基于SGLang框架扩展，约3000行Python代码。Hierarchical KV Cache Manager使用Python multithreading + CUDA streams（PyTorch CUDA stream API），动态管理三类stream：compute stream（LLM推理forward pass）、write/load stream（KV cache chunk的device↔host传输）、evict stream（GPU显存block释放）。通过CUDA events在stream间建立同步点，实现非阻塞overlap。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  未开源（arXiv https://arxiv.org/abs/2510.02758 和 EuroSys 2026 DOI https://doi.org/10.1145/3767295.3769328 均未提供代码仓库）。KV cache数据流运行流程（以H200 + Llama3-8B为例）：
  1. GPU显存作为CPU memory上大容量KV cache的高速cache。不同于普通write-back策略（仅在真正抢占时写回），TokenFlow使用write-through。
  2. Decode iteration后：新生成的KV chunk放入write buffer → 下一轮compute前预估compute duration → 选择大小合适的chunk通过write stream同步到host memory → compute stream和write stream通过CUDA event并发执行（overlap compute和I/O）。
  3. Request preemption：scheduler决定抢占请求 → 已write-through同步的chunk可立即释放显存block → 未同步的剩余chunk与load stream加载新请求KV chunk的操作通过evict-load overlap重叠执行 → 显存block在evict stream中释放。
  4. Request resume：load stream从CPU memory加载请求的KV chunk回GPU → 恢复decode。后台write-through保证大部分KV cache已在host同步，resume时只需加载最近未同步的增量chunk。
  5. 消融效果：去掉offload时完成时间从66.00s恶化到127.28s（恶化93%），说明分层KV cache管理是TokenFlow性能收益的核心来源。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：论文未开源RTL和simulator。Dataflow运行流程（以256×256 VAR, N=4096, K=1936为例）：
  1. Attention Core dataflow切换：浅层Learning Region→SA模式：Snooper配置PE Cell按OP dataflow接收packet ID→Fat Tree分发→local window内token经Small Attention聚合为representative token；随后→BA模式：PE Cluster按attention head分配→Row dataflow执行Big Attention全局建模
  2. Radix Sort Core TopK dataflow（PD/DB阶段）：confidence array (N=4096) 经TP串行化→CountBin按radix digit将4096个元素分到bins→PrefixSum计算每bin前缀和确定第1936大元素所在bin→SelectBin精确定位含K-th元素的candidate bin→Filter筛选TopK 1936元素的indices→Locality-aware Scheduling：history table标记已解码区域→PE分组优先处理高置信空间区域
  3. DB阶段TopK：类似流程但处理per-token importance scores→选TopK进入完整Transformer，其余bypass
  4. Divide-and-Conquer FP accumulation：Row和OP MAC共享FP accumulator，Fluid Zone Detection动态调整累加精度边界降低功耗

