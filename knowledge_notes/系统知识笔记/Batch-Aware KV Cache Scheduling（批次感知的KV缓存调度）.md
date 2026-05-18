## Batch-Aware KV Cache Scheduling（批次感知的KV缓存调度）

术语是什么？通过联网搜索让回答具体和精准。
Batch-Aware KV Cache Scheduling 是 PiLLM 的单GPU内内存管理策略，核心思想：将 KV cache 从 per-request 固定预留改为 batch 级别的共享内存池。传统策略（如 vLLM PagedAttention）为每个请求独立管理 KV cache block，加入 batch 时预留 worst-case 剩余输出长度的 KV 空间，导致两项：激进预留提升显存利用率但 eviction 高（vLLM 可高达 68.39%），保守预留压低 eviction 但显存利用率和 batch size 偏低。Batch-aware 策略利用批级负载预测结果——batch 内所有请求的预测剩余输出 token 总量作为共享内存预算——不替单个请求预留最大空间。实际执行中，部分请求生成少于预测值释放余量，其他请求可使用更多 token slot，只要 batch 总量在 error bound 内即可。利用大数定律：单个请求输出长度难以预测，但 batch 越大平均值越稳定，因此可在维持低 eviction （0.01%–0.53%）的同时允许更高显存 overcommit（显存利用率 78.93%–96.05%）。

从系统架构角度拆解术语：
1. **预算设定**：每管理窗口开始，根据批级输出长度预测值计算该 batch 的预期 KV cache 需求上限。
2. **共享池分配**：Intra-GPU manager 将预测 KV cache 总量设为 batch memory pool 上界，而非 per-request 独立上限。
3. **运行时分配**：KV cache 组织为 token slot linked list。请求生成新 token 时从共享池分配 slot；请求完成或 evict 时释放 slot 回池。
4. **溢出控制**：当 batch 实际 KV cache 使用接近预测上界时触发 eviction（选择 slack 最少请求），eviction rate 控制在极低水平。
5. **Prefill→Decode 转移**：prefill 完成后 KV 状态从 prefill instance 逐层转移到 decode instance，利用逐层转移减少跨 GPU 传输峰值带宽。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
基于 LightLLM 的 token-level KV cache 管理实现。LightLLM 原生使用 token slot linked list 而非固定 block table，天然适合共享池化的分配。PiLLM 在每个 decode instance 内维护 running batch 和 waiting queue，running batch 内请求共享 batch memory pool。需注意：批级策略在 batch size 小时方差下降不显著，收益会降低；论文实验表明在 medium-to-large batch 下收益最明显。

涉及论文标题：
- PiLLM: Resource-Efficient LLM Inference Using Workload Prediction
