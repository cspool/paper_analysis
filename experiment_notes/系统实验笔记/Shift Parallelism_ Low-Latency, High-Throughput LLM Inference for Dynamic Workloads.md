## Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads

- 属于Serving调度的实现是什么？实验比较什么？
  提出Shift Parallelism，将训练用Ulysses Sequence Parallelism (SP) 改造为inference可用的并行方式，并与Tensor Parallelism (TP) 在同一vLLM部署中按batch token数阈值动态切换。核心实现：(1) SP for Inference：为SP添加GQA支持（通过all-to-all中KV cache replication处理Q head数与KV head数不匹配的场景）、small batch load balancing（padding到SP degree倍数）和任意(SP, TP)组合forward pass（Algorithm 1）；(2) KV Cache Invariance：约束base config与shift config使用相同attention head layout和ordering，使请求在SP与TP之间切换时无需搬移KV cache；(3) Dual Configurations：base configuration使用SP或mixed (SP, TP)处理大batch优化TTFT和throughput，shift configuration固定为(SP=1, TP=P)的full TP处理小batch优化TPOT，运行时按batch token数是否超过shift threshold选择执行路径（Algorithm 2）；(4) Dual Model Loading：base model和shift model分别加载权重并独立CUDA graph capture，共享KV cache，shift model内存开销约1/SP。实验比较TTFT、TPOT、completion time、combined throughput (tokens/sec)、不同arrival rate下的latency-throughput曲线、不同input sequence length的峰值吞吐和最低延迟，对比vLLM内置TP和DP、独立SP (Ulysses)，以及在production trace中对比SGLang v0.4.6、TensorRT-LLM v0.18.2。

- 硬件平台是什么，配置是什么。
  AWS p5en.48xlarge单节点8×H200 GPU，每卡141GB HBM、4.8TB/s带宽、FP8 tensor core峰值1,979 TFLOPS，GPU间通过NVSwitch互联，标称900GB/s。Artifact Appendix注明通用NVIDIA DGX-H200节点即可复现。

- 开源Serving框架是什么。修改了什么。
  基于vLLM v0.9.2（Artifact Appendix使用vLLM v0.10.1，ArcticInference插件commit `5e08f0f`），通过ArcticInference插件系统集成。主要修改：(1) 实现SP for inference forward path：支持GQA（QKV投影中Q head与KV head数不一致时通过all-to-all send/receive buffer完成KV cache replication）、fused all-to-all（Q/K/V通信融合到单次all-to-all）、padding-based load balancing（小batch时padding到SP degree倍数）；(2) 实现combined (SP, TP) forward pass：SP按sequence维度切分输入，attention前后通过all-to-all在sequence parallel与head parallel layout间转换，MLP路径使用TP all-reduce；(3) 实现Shift Parallelism runtime：base model和shift model分别编译和CUDA graph capture，运行时根据batch size选择执行，通过SP_TP group确保shift model按base config的SP group order加载权重；(4) vLLM插件系统：编译并capture base和shift两套CUDA graphs，初始化时注册，运行时按threshold选择replay。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  已开源，GitHub仓库（snowflakedb/ArcticInference），Apache-2许可证，vLLM插件实现，Zenodo可复现实验包（DOI: 10.5281/zenodo.18240909）。使用步骤：
  1. 安装vLLM v0.10.1：`pip install vllm==0.10.1`
  2. 克隆ArcticInference：`git clone https://github.com/snowflakedb/ArcticInference`
  3. 安装插件依赖
  4. 下载模型（如RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic）
  5. 启动vLLM server带Shift Parallelism：`--enable-shift-parallel --shift-parallel-threshold <N>`
  6. 复现实验：运行`ArcticInference/benchmark/reproducibility`中脚本
  Shift Parallelism通过在一个部署中保留base (SP-biased)和shift (full TP)两套配置，运行时根据每轮batch token数是否超过阈值动态选择。例如在8×H200上，base config设为(SP=4, TP=2)，当batch token多时选base用SP降低TTFT和提高throughput；当batch token少（如decode阶段仅有少量活跃请求）时选shift (SP=1, TP=8)用full TP降低TPOT。切换不搬移KV cache因为两者attention head layout一致。在Llama-70B-FP8、4k input/250 output场景下，Shift Parallelism的median TTFT为148ms、median TPOT为51ms、peak throughput为69,147 tok/s，对比vLLM throughput-opt DP (1,355ms/83ms/75,535 tok/s) 和vLLM latency-opt TP (3,930ms/85ms/51,162 tok/s)，以17% throughput牺牲换来9.15x TTFT降低和1.63x TPOT降低。

