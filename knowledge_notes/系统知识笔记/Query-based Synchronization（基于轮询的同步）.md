## Query-based Synchronization（基于轮询的同步）

术语是什么？

Query-based Synchronization是MuxWise中协调Prefill和Decode异步执行的同步机制，通过**周期性轮询CUDA events**来检测prefill layer完成事件，而非阻塞等待。该机制允许在prefill执行期间持续发射decode iteration，一旦检测到prefill完成的CUDA event，立即将该prefill请求合并入当前decode batch（inflight batching），避免长时间阻塞decode流。

从系统架构角度拆解术语：

同步流程：

1. **CUDA Event记录**：每个prefill batch的最后一个PL完成后，在prefill stream上记录一个CUDA event。
2. **异步轮询**：dispatcher在每轮decode iteration发射前/后，调用`cudaEventQuery()`轮询所有pending的prefill完成event。该调用是非阻塞的（返回cudaSuccess表示完成，cudaErrorNotReady表示仍在执行）。
3. **即时合并**：当event就绪，dispatcher立即将对应prefill请求的KV cache信息合并入decode batch，下一次decode iteration即包含新请求的token生成。
4. **无Bubble保证**：由于轮询发生在decode launch间隙而非阻塞等待prefill完成，即使prefill与decode的完成时刻不对齐，也不会在GPU上产生明显气泡。论文实测总开销在1.5%内。

与Naive同步的区别：Naive方案是在发射decode iteration前阻塞等待prefill完成（`cudaStreamSynchronize`），这导致：(a) prefill launch时间长时GPU空闲；(b) decode提前终止时正在执行的prefill无法被中断回收SM；(c) prefill完成后decode才重新开始，产生无请求时的GPU bubble。

术语一般如何实现？如何使用？

实现为host端CUDA event polling循环，与CUDA graph launch交替执行。在SGLang中集成于multiplex engine的调度循环。该机制与layer-wise prefill配合：由于prefill被拆为小粒度PL，即使单个PL未完成也不会长时间阻塞decode；query-based sync确保了各个PL完成后即可被感知。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing
- Laser: Unlocking Layer-Level Scheduling for Efficient Multi-SLO LLM Serving

Laser的Global Controller实现了一种不同的SLO-aware dispatching：(1) **Prefill侧**：controller持续监控各prefill instance的调度决策和请求统计，选择在EDF+layer-level chunked prefill下可接纳新请求且TTFT slack最大的instance；若都不安全则选prefill token最少的instance做best-effort。(2) **Decode侧**：采用group-based decode assignment——将decode instance按TBT target分组为SLO-homogeneous groups，新请求优先分配到SLO最接近的group中TBT increment最小的instance，允许跨group分配以平衡负载。与MuxWise的decode优先best-fit SM思路不同，Laser的dispatcher关注请求粒度的SLO-aware路由而非SM资源分配。

---
