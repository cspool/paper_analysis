## High Throughput and Low Latency LLM Serving via Adaptive KV Caching

- baseline方法是什么？
  Baseline是三种vLLM中已有的或复现的KV cache管理策略：(1) vLLM-Recompute：GPU显存满时丢弃preempted requests的全部KV cache，等请求重新获得显存后整体重算所有历史token的KV，问题是长序列下重算开销高且粒度为整请求；(2) vLLM-Swap：GPU显存满时将preempted requests的全部KV cache swap到host memory，再通过PCIe恢复，问题是host-GPU带宽（PCIe 4.0 x16约32GB/s）成为瓶颈，长序列swap延迟高；(3) HCache：缓存hidden states加速状态恢复，但恢复仍发生在request-level或layer subset粒度，decode阶段仍倾向于保留当前并发请求的完整KV cache。共同缺陷：all-or-nothing恢复策略未利用"旧token可部分重算、SM有空闲、显存更稀缺"的资源不平衡——论文实测Llama2-13B在A100上显存常接近饱和而GPU SM utilization处于很低区间。

  baseline全栈执行例子（以Llama2-13B + vLLM-Recompute + ShareGPT单请求在A100上decode为例）：
  - 算法层：Llama2-13B MHA decoder模型不变，每层40 heads × 128 head_dim，每token每层产生约40×128×2个float16 KV元素。
  - 系统框架/Serving层：vLLM continuous batching + PagedAttention。请求进入decode后KV cache按PagedAttention block分配填充。当GPU显存满时scheduler选择一批请求preempt，丢弃其全部KV cache blocks。请求重新被调度时从token 0开始逐token recompute所有layer的KV（O(L×T)重算开销），然后继续decode新token。batch size受限于GPU显存必须容纳所有running request的全部KV cache + 模型权重。
  - 编译框架层：PyTorch torch.compile默认路径，无特殊kernel编译优化。
  - kernel调度层：PagedAttention默认GEMM kernel + FlashInfer/PagedAttention kernel，无融合。KV swap路径：cudaMemcpy host↔GPU同步传输（一条stream阻塞）。
  - 硬件架构层：A100-80GB GPU，108个SM，PCIe 4.0 x16连接host。GPU显存容纳模型权重(~26GB for Llama2-13B FP16) + batch内所有请求的完整KV cache。SM在重算期间利用率短暂上升但大部分时间因显存瓶颈batch size受限而idle。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出eLLM adaptive KV caching系统，核心设计：(1) token-wise/layer-wise KV cache管理：从整请求粒度细化为token和layer粒度，每个请求缓存较新的(1-r) token KV、释放较早的r token KV（或swap到host），用map table追踪每个(token, layer)的缓存/swap/recompute状态，layer-granular block按F=4层划分减少内存碎片；(2) cross-request batched recomputation：将多个请求的uncached token聚合成batch统一重算KV，提高GEMM/attention效率——避免小batch单独重算效率差；(3) request-level SLSQP在线优化：将batch size b和uncached token ratio r作为优化变量，在GPU显存和TPOT SLO约束下用SciPy SLSQP求解器最大化throughput；(4) layer-wise communication-computation overlapping：双CUDA stream，Stream A异步传输cached KV（host↔GPU），Stream B执行recomputation+decode；(5) kernel fusion：将compute-intensive K1（旧token KV recompute）和memory-intensive K2（新token decode）融合，减少launch overhead并动态分配线程；(6) closed-loop adaptation：layer-level实际额外显存Mo反馈给request-level，形成b和r的闭环调整。

  论文方法全栈执行例子（以Llama2-13B + eLLM + ShareGPT单请求在A100上decode为例）：
  - 算法层：Llama2-13B MHA decoder模型不变。eLLM不修改模型权重或attention公式，仅在KV cache存储策略和kernel执行层面优化。
  - 系统框架/Serving层：基于vLLM扩展。request-level optimizer用SLSQP求解b和r（如r=0.4）。历史序列前40% token的KV被释放或swap，后60% token的KV保留在layer-granular block中（F=4层一组block）。进入transformer layer i时layer-level scheduler通过map table精确定位cached token物理block，确定uncached token在本层是recompute还是swap。本轮可容纳更大batch size（因每请求KV residency降低约40%）。
  - 编译框架层：PyTorch + 预编译CUDA shared libraries。对K1+K2 fused kernel预生成32组线程variant（32-1024 step 32），运行时按K1/K2计算量比选择.so。
  - kernel调度层：Stream A异步cudaMemcpyAsync传输layer i+1的swapped KV，Stream B执行fused kernel K1+K2——K1为layer i+1的uncached旧token重算KV（利用cross-request batching提高GEMM效率），K2用layer i完整历史KV对current token decode。线程按K1/K2 FLOP比例分配（如70:30），取整到32的倍数。K1临时KV使用后立即释放，额外workspace仅约1 layer KV。Ttoken completion后新token KV写入对应layer-granular block，layer-level计算Mo反馈给request-level。
  - 硬件架构层：A100-80GB GPU，108 SM。eLLM用compute换memory——SM在recompute K1时利用原本idle的计算资源，换出显存容纳更多并发请求。PCIe 4.0 x16的swap传输与computation并行避免成为单一瓶颈。

  **缺陷对应关系**：
  - vLLM-Recompute的整请求全量重算 → token-wise partial recomputation只重算已释放的token/layer，且cross-request batching提高重算效率
  - vLLM-Swap的整请求host↔GPU全量传输 → 按uncached ratio部分传输+cached token保持GPU resident，减少PCIe传输量
  - 三者共享的"decode阶段每个活跃请求必须保留完整KV cache"假设 → 直接降低每个active request的KV residency，从源头提高并发容量（用空闲SM compute换显存）
  - 粗粒度KV block → layer-granular block（F=4）减少碎片
  - 重算与decode串行 → fused kernel + dual stream overlap消除idle bubble
