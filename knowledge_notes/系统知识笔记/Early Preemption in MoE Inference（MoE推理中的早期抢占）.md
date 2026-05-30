## Early Preemption in MoE Inference（MoE推理中的早期抢占）

术语是什么？
Early Preemption 是 ProMoE 提出的 prefetch 任务调度优化：在 MoE gate function 完成后立即获取当前层所需的精确 expert 列表，清除同层中仍处于 LOW priority 的 speculative prefetch 任务，为缺失的 experts 发起 HIGH priority precise prefetch 任务。传统 reactive caching 中 cache miss 在 expert 实际被访问时才检测，Early Preemption 将缺失检测提前到 gate function 完成时刻，为缺失 experts 的 prefetch 争取更多时间窗口（与当前层其他 cached experts 的计算重叠）。

从系统架构角度拆解术语：
Traditional: gate → expert1(cached) → expert2(MISS! cudaMemcpy blocks) → ...
Early Preemption: gate → hook 获取所有 required experts → 清除同层 LOW 任务 → 缺失 experts 作为 HIGH 入队 → expert1(cached 立即执行) → expert2(HIGH 任务在 expert1 计算期间已开始传输) → wait_chunks_ready → compute

术语一般如何实现？如何使用：
实现为 gate function 末尾的 hook（~100 行 C++），调用 PushPreciseExperts API。与 reordered inference 同时触发。在 ProMoE ablation 中，early preemption 在 prefill 阶段贡献 1.27× speedup。

涉及论文标题：
- ProMoE: Fast MoE-based LLM Serving using Proactive Caching
