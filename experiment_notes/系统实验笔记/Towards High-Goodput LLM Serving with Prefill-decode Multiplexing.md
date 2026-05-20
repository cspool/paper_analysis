## Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

- 属于Serving调度的实现是什么？实验比较什么？
  MuxWise提出intra-GPU prefill-decode (PD) multiplexing，在同一GPU内的不同SM上空间复用prefill和decode阶段。实现包含三个模块：(1) bubble-less multiplex engine：将prefill按layer切分执行，通过query-based同步消除GPU气泡；(2) contention-tolerant estimator：利用solo-run predictor + contention guard提供worst-case延时估计以保障SLO；(3) SLO-aware dispatcher：为decode分配best-fit SMs来满足TBT SLO，剩余SMs分配给prefill最大化goodput。实验比较goodput（SLO约束下的峰值吞吐）、99%-ile TTFT和TBT，对比chunked-prefill (SGLang)、NanoFlow、LoongServe和SGLang-PD四种baseline。

- 硬件平台是什么，配置是什么。
  主测试平台：8×A100-80GB GPU（NVLink 600 GB/s）。附加测试：8×H100-SMX5-80GB GPU，8×H200-SMX5-141GB GPU。NVIDIA driver 570.124.06，CUDA 12.8。单GPU测试也评估了Llama-8B在单块A100上的表现。

- 开源Serving框架是什么。修改了什么。
  基于SGLang v0.4.10post2修改，PyTorch 2.6.0。主要修改：(1) 集成GreenContext实现intra-process SM空间分区，支持运行时低开销（微秒级）重配分partition ratio；(2) 将prefill阶段改为layer-wise执行（piecewise CUDA graph），每层可独立调度；(3) 实现query-based同步机制：定期轮询CUDA events，异步发射decode batch和prefill layer，event完成后立即合并；(4) 添加contention guard离线profiling和在线estimator模块；(5) SLO-aware dispatcher根据worst-case estimation动态选择multiplexing plan（SM配置和prefill layer数量）。开源地址：https://github.com/ykcombat/sglang.git (branch: slo_config)，Zenodo: https://zenodo.org/records/18062118。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  已开源，代码在GitHub（ykcombat/sglang, slo_config分支），提供Docker镜像。使用步骤：
  1. 克隆仓库并切换到slo_config分支：`git clone https://github.com/ykcombat/sglang.git && cd sglang && git checkout slo_config`
  2. 构建SGLang：`pip install -e "python"`
  3. 启动MuxWise服务：`./start_pdmux.sh`（启动带PD multiplexing的SGLang server）
  4. 启动chunked-prefill对比：`./start_chunk.sh`（通过设置`$CHUNK_SIZE`环境变量控制token budget）
  5. 运行benchmark：`./bench_pdmux.sh`（测试ShareGPT和LooGLE workload）
  MuxWise通过intra-GPU PD multiplexing，将decode阶段的SMs保留用于满足TBT SLO（如100ms），剩余SMs分配给prefill。例如对Llama-70B在8×A100上，decode分配约60% SMs，prefill分配约40% SMs，避免chunked-prefill的token budget dilemma（需要4K tokens才能打满GPU，但SLO只允许256）。Layer-wise prefill避免了chunked-prefill因重复读KV cache导致TBT膨胀的问题，并支持长请求的抢占调度。

