## Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  Bullet的kernel调度/运行时计算实现是动态SM分区下的prefill/decode kernel并发调度：通过libsmctrl_set_stream_mask修改CUDA stream metadata，将prefill和decode的kernel流绑定到GPU上不同的SM子集执行，实现同一GPU内的空间分区。prefill engine在分配的SM子集上逐layer发射QKV/O-proj/MLP/attention等kernel，decode engine在另一SM子集上以CUDA Graph step发射decode kernel。Resource manager在收到scheduler的repartition command后微秒级（平均4.1us）修改CUDA stream的SM mask，使后续kernel立即在新SM子集运行。实验对比Nsight Systems采集的SM active cycles、Tensor Core utilization和memory-bandwidth utilization，以及不同SM partitioning策略下的throughput、TTFT、TPOT。

- 后端平台是什么，配置是什么。
  NVIDIA A100-80GB (108 SM/GPU, NVLink 600 GB/s)、NVIDIA H100 (132 SM/GPU, 600 GB/s)、NVIDIA H20 (78 SM/GPU, intra-node 400 GB/s)，CUDA 12.4。

- 评估性能的软件/脚本是什么。修改了什么。
  Nsight Systems采集SM/Tensor Core/memory bandwidth utilization。libsmctrl修改：使用libsmctrl_set_stream_mask()在运行时修改CUDA stream的SM mask（GPC配置掩码），使后续kernel launch限制在指定SM子集。CUDA MPS用于spatial sharing支持。CUDA Graph用于decode step的低开销一次性发射（减少kernel launch overhead）。实验中prefill和decode engine各自持有独立的CUDA stream，resource manager在scheduler下发repartition command后立即修改stream mask。SLO-aware scheduler周期性（每个prefill layer group或decode step后）读取全局状态并搜索新SM分区方案。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源仓库（https://github.com/zejia-lin/BulletServe）包含修改版libsmctrl。SM分区使用流程：
  1. 通过libsmctrl_get_gpc_info()获取GPU GPC/TPC拓扑
  2. 构建SM mask（以16 SM为粒度分组），论文在A100上定义6种SM配置，H100上7种
  3. 调用libsmctrl_set_stream_mask()将stream绑定到目标SM mask
  4. 后续kernel launch（通过PyTorch CUDA stream）自动限制在mask指定的SM子集执行
  5. scheduler根据实时队列状态和SLO压力下发repartition command，resource manager更新stream mask
  例如：prefill队列堆积时将prefill stream mask扩展到接近全部SM，decode SLO紧张时缩小prefill mask以释放更多SM给decode。SM mask更新开销平均4.1us，metadata传递平均0.21ms，performance prediction平均10.2us。

