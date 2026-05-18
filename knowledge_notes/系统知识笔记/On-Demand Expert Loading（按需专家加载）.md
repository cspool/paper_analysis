## On-Demand Expert Loading（按需专家加载）

术语是什么？
On-Demand Expert Loading是当gate network选择的expert不在GPU cache中时，立即从CPU memory加载该expert权重到GPU的操作。它是expert offloading系统的fallback机制。在FineMoE中，on-demand loading暂停普通prefetch任务、以最高优先级执行CPU→GPU传输，使forward可以继续执行。加载延迟主要由PCIe带宽和expert权重size决定。

从系统架构角度拆解术语：
```
# FineMoE On-Demand Loading流程（在MoE layer forward中触发）：
for expert_idx in gate_selected_experts:
    if expert_idx in gpu_cache:
        output += gate_prob * gpu_cache[expert_idx](h)  # cache hit
    else:
        # cache miss → on-demand loading
        pause_normal_prefetch_tasks()                    # 暂停普通prefetch
        expert_weights = cpu_to_gpu_transfer(expert_idx) # PCIe 4.0, 32GB/s
        gpu_cache[expert_idx] = expert_weights           # 加载到GPU cache
        output += gate_prob * expert_weights(h)          # 继续计算
        resume_normal_prefetch_tasks()                   # 恢复普通prefetch
```
On-Demand Loading延迟是FineMoE优化的核心目标：通过提高expert hit rate减少其发生频率，通过异步prefetch降低其对forward的阻塞影响。

术语一般如何实现？
用CUDA Runtime APIs（cudaMemcpyAsync）实现CPU→GPU数据传输。FineMoE通过高优先级task清空普通prefetch queue、独占PCIe带宽执行on-demand load。Expert size（Mixtral-8x7B每个expert约1.6GB/8≈200MB per expert）决定单次加载延迟下限（200MB/32GB/s≈6.25ms per expert，top-2 worst-case≈12.5ms）。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading
