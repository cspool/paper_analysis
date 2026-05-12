## Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

- 属于Serving调度的实现是什么？实验比较什么？
  论文实现 MuxWise，一个基于 SGLang 的 intra-GPU prefill-decode multiplexing serving 框架。它把同一 GPU 内的 SM 动态划分给 prefill 和 decode，使两阶段空间并发执行，同时共享同一进程内的模型权重、显存空间和 KV cache pool。系统包含 bubble-less multiplex engine、contention-tolerant estimator 和 SLO-aware dispatcher：prefill 被拆成 transformer layer 粒度执行，decode 优先按 CUDA graph / graph-level 路径发射；调度器用 worst-case decode latency 估计选择满足 TBT SLO 的 best-fit SM 数，其余 SM 推进 prefill，并允许短 prefill 非递归抢占长 prefill。实验比较 MuxWise 与 SGLang chunked-prefill / SARATHI-Serve 风格 token budget、NanoFlow、LoongServe、SGLang-PD 静态分离式 serving；核心指标是 P99 / 99%-ile TTFT、TBT、SLO attainment、goodput、token throughput 和 GPU utilization。
- 硬件平台是什么，配置是什么。
  主实验为 8 张 NVIDIA A100-80GB GPU，NVLink 带宽 600 GB/s；泛化实验还使用 8 张 H100-SMX5-80GB GPU 和 8 张 H200-SMX5-141GB GPU。软件环境为 PyTorch 2.6.0、SGLang 0.4.10post2、NVIDIA driver 570.124.06、CUDA 12.8。模型包括 Llama-8B、Llama-70B 和 Qwen3-235B MoE（22B activated）。工作负载包括 Conversation、Tool&Agent、ShareGPT、LooGLE、OpenThoughts；artifact 附录复现实验使用 CodeLlama-34B-Instruct-hf、ShareGPT、LooGLE 和 H200 NVL（140GB、132 SMs）。
- 开源Serving框架是什么。修改了什么。
  开源 Serving 框架是 SGLang 0.4.10post2。论文在其上扩展 PD multiplexing：使用 NVIDIA GreenContext 将 prefill / decode CUDA streams 绑定到不同 SM 集合，在同一进程内共享 KV cache pool；将 prefill 改为 layer-wise execution 和 query-based synchronization；为 decode 构建 solo-run predictor + contention guard 的 worst-case latency estimator；增加 SLO-aware dispatcher 在线选择 SM 分区、prefill layer 发射数量和 preemption。Artifact Appendix 给出源码入口 https://zenodo.org/records/18062118 以及 GitHub 分支 https://github.com/ykcombat/sglang/tree/slo_config。
- 分层说明：完全匹配。论文核心贡献是修改 LLM serving framework 和在线调度策略，以在 TBT SLO 下提高 goodput；kernel / runtime 层机制服务于这一调度目标。
