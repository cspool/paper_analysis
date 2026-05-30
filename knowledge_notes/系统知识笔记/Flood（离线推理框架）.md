## Flood（离线推理框架）

术语解释
Flood 是蚂蚁集团 Ling 团队开发的高效离线推理框架（开源 https://github.com/alipay/PainlessInferenceAcceleration），采用全流水线并行架构和 Segment Cache 机制，专为缺少 NVLINK 等高速互连的低规格硬件设计。

术语是什么？
Flood 的核心设计是放弃 Tensor Parallelism (TP)，纯用 Pipeline Parallelism (PP) 进行推理。传统框架（vLLM）依赖 TP 在节点间切分张量需 NVLINK 高带宽互联；无 NVLINK 时 TP 通信开销可占总时间 50% 以上。Flood 通过 PP 避免张量切分和 AllReduce 通信。两个关键设计：(1) 多对一进程映射——单加速器部署多进程各自绑定独立 CUDA stream，零 CPU 开销；(2) Segment Cache——替代 PageAttention，连续内存分配 KV cache 使用更大 block 提升计算效率。

从系统架构角度拆解术语：
```
=== 初始化（8 加速器） ===
启动 9 进程 (>GPU 数以消除空闲)
stage_0→Acc0, ..., stage_7→Acc7，各绑独立 stream

=== Token 生成 ===
for each batch:
    hidden = stage_i(hidden)     # 仅 PP，无 TP AllReduce
    kv_cache[pos] = (k, v)       # Segment Cache 连续写入
```

实验：Ling-Plus(FP8) 8×Device E: Flood 6569 vs vLLM 2742 tokens/s (2.40×)。

术语一般如何实现？如何使用？
- 开源：github.com/alipay/PainlessInferenceAcceleration
- 适合低互联带宽异构硬件
- 支持 FP8 和 prefix caching

涉及论文标题：
- Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs

---
