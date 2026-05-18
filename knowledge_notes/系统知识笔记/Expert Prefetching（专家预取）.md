## Expert Prefetching（专家预取）

术语是什么？
Expert Prefetching是在expert被gate network实际选中之前，提前将expert权重从CPU memory异步加载到GPU cache的技术。目的是隐藏CPU→GPU数据传输延迟（PCIe 4.0 32GB/s为主要瓶颈）。FineMoE使用两种搜索模式：前d层semantic-based search（用prompt semantic embedding搜索）、d+1层起trajectory-based search（用已过层gate probability trajectory搜索）。Prefetch priority = p/(l - l_now)：probability越高、距离当前layer越近的expert优先预取。

从系统架构角度拆解术语：
```
# FineMoE异步预取流程（Publisher-Subscriber模式）：
Context Collector (Publisher):
    收集semantic embedding (前d层) 或 expert trajectory (d层之后)
    → 发布到Expert Map Searcher

Expert Map Searcher (Subscriber):
    接收context → 在Expert Map Store做cosine similarity
    → 返回top-k相似历史map → Similarity-Aware Selection
    → 发布预取expert列表到Prefetch Publisher

Prefetch Publisher:
    按p/(l-l_now)排序预取任务
    → GPU task pool调度 → CUDA async copy CPU→GPU
    → Cache Subscriber更新GPU expert cache状态

# 关键：此异步路径与inference forward解耦
# forward不受search/prefetch延迟影响
# 仅在cache miss时暂停普通prefetch执行on-demand loading
```

术语一般如何实现？
在FineMoE中用C++和CUDA Runtime APIs实现。GPU侧维护task pool和异步线程调度。与MoE-Infinity同步prefetch的关键区别：FineMoE异步路径将context collection、map search、prefetch和map update与forward解耦，不阻塞推理。默认prefetch distance为d层，需平衡隐藏search/prefetch开销（太小无法隐藏）和预测hit rate（太大则预测不准）。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading
