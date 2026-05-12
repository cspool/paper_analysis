## Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  论文没有提出新的注意力或 FFN 数学 kernel，而是在 GPU runtime 层实现 PD multiplexing：用 GreenContext 做 intra-process SM spatial partition，将 prefill 与 decode 的 CUDA streams 绑定到不同 SM 集合；decode 侧优先发射低 launch overhead 的 graph-level execution，prefill 侧拆成 transformer layer units 以缩短可调度粒度；通过 CUDA event polling 做 query-based synchronization，减少 prefill 完成并入 decode batch 时的 bubble；调度器根据 estimator 计算要发射的 prefill layer 数，使 prefill layers 与 decode iteration 在时间上对齐。实验比较 MuxWise 与 chunked-prefill、NanoFlow、LoongServe、SGLang-PD，在真实和合成负载上看 P99 TTFT / TBT、goodput、token throughput、GPU utilization、SLO attainment。
- 后端平台是什么，配置是什么。
  后端平台主要是 NVIDIA GPU：8x A100-80GB（NVLink 600 GB/s）、8x H100-SMX5-80GB、8x H200-SMX5-141GB。软件栈为 PyTorch 2.6.0、CUDA 12.8、NVIDIA driver 570.124.06、SGLang 0.4.10post2；artifact 附录要求 H200 NVL 140GB / 132 SMs、driver 580.65.06 或高于 570。模型为 Llama-8B、Llama-70B、Qwen3-235B MoE；artifact 复现用 CodeLlama-34B-Instruct-hf。
- 评估性能的软件/脚本是什么。修改了什么。
  论文正文使用基于 SGLang 的 MuxWise 实现和 Nsight Systems 聚合 GPU utilization；artifact 附录提供 `start_pdmux.sh`、`start_chunk.sh`、`bench_pdmux.sh`、`bench_chunk.sh` 和 `plot.ipynb`，用于比较 MuxWise 与 chunked-prefill 并输出 jsonl 指标。修改内容包括 GreenContext-based stream/SM partition、layer-wise prefill launch、decode-priority launch order、CUDA event polling synchronization、offline profiling estimator、runtime contention guard update 和 SLO-aware SM partition selection。
- 分层说明：部分匹配。论文重点不是新 kernel 算子，而是 GPU stream、SM 分区、kernel launch 顺序和 layer 粒度运行时调度；因此归入 kernel调度/运行时计算层作为辅助实验层。
