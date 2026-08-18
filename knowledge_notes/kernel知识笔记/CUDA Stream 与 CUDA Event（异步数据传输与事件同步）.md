## CUDA Stream 与 CUDA Event（异步数据传输与事件同步）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CUDA Stream 是 GPU 上串行执行 kernel/传输操作的独立命令队列，不同 stream 之间可并发执行（无依赖时），是 CUDA 实现"传输与计算重叠"的基本手段；CUDA Event 是流内/跨流的同步原语，标记某流中已入队操作完成的时刻。STEP 用这两个机制实现专家预取与计算的流水线重叠：预取 kernel（异步 H2D 拷贝）在独立 stream 上先于专家计算 kernel 入队，二者并发执行；在每个解码步序列的最后一个预取 kernel 后记录 cudaEvent，CPU 侧用 cudaEventQuery 非阻塞查询其完成状态——查询返回"未完成"时 CPU 可继续做其他工作，避免 cudaStreamSynchronize 式阻塞同步，从而在不阻塞主线程的前提下保证"计算开始前所需专家已就绪"。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// STEP 式预取-计算重叠（伪代码）
cudaStream_t s_prefetch, s_compute;   // 两条流
cudaStreamCreate(&s_prefetch); cudaStreamCreate(&s_compute);
cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, s_prefetch); // 预取入队
expert_gemm<<<grid, block, 0, s_compute>>>(...);                       // 计算入队（并发）
cudaEventRecord(ev_done, s_prefetch);  // 记录预取完成点（非阻塞）
while (cudaEventQuery(ev_done) == cudaErrorNotReady) { /* CPU 可做其他事 */ }
// 数据可用后继续下一层
```
Annotations：s_prefetch=预取流、s_compute=计算流、cudaMemcpyAsync=异步 H2D 拷贝、cudaEventRecord=在流中插入事件标记、cudaEventQuery=非阻塞查询（返回 cudaErrorNotReady 表示未完成）、cudaStreamSynchronize 是应避免的阻塞版本。CUDA 的非抢占式 kernel 执行保证同 SM 上 kernel 完整运行，多流并发主要依赖资源余量（拷贝引擎与 SM 计算并行、或不同 kernel 占用不同资源）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：CUDA Runtime/Driver API 的标准组件（cudaStreamCreate/cudaMemcpyAsync/cudaEventRecord/cudaEventQuery）；STEP 在 Transformers 推理路径上按"预取 kernel 先入队、计算 kernel 后入队、事件标记同步点"的模式组织每层执行。使用场景：一切"数据搬运在关键路径上"的 GPU 系统——MoE 专家权重 H2D 预取、KV cache 搬运、pipeline 阶段间传输等；与 CUDA Graph 结合可减少 launch 开销。注意事项：异步传输要求 pinned memory；同一流内保持顺序、跨流靠 event/依赖保证顺序；过度并发会争抢带宽（STEP 用命中率反馈控制预取激进性，命中率 <40% 时减窗停用）。


- Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs
RESONATOR 补充视角（ISCA'26，stream 到 SM 子集的绑定）：CUDA Stream 除"异步并发/事件同步"外还可作为 SM 配额载体——RESONATOR 用 green-ctx/libsmctrl 把 CUDA 流绑定到 SM 子集（GPC 配置掩码），预创建每任务两条流：wide 流（SMCTRL.SetQuota 1.0，全部 SM）与 narrow 流（SetQuota q_narrow，窄 SM 子集），contending 模式下每 kernel 查 profile 表路由到 wide/narrow 流（compute-bound→wide、memory-bound/低占用→narrow），实现 kernel 级 time-space sharing；complementary 模式下 decode 流固定绑 SM_dec 切片（SM 分区，兼容 CUDA Graph 重放）。与 STEP 的"预取/计算两流并发"不同，RESONATOR 用流既做并发也做 SM 隔离/配额。
涉及论文标题：
- STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference
