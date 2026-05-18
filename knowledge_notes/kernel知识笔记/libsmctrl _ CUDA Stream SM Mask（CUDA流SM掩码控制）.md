## libsmctrl / CUDA Stream SM Mask（CUDA流SM掩码控制）

术语是什么？

libsmctrl是一个底层CUDA库，用于在运行时修改CUDA Stream的SM（Streaming Multiprocessor）执行掩码。通过调用`libsmctrl_set_stream_mask()`，可以将一个CUDA stream上的后续kernel launch限制在GPU特定的SM子集上执行，实现进程内（intra-process）的GPU SM空间分区。与NVIDIA GreenContext相比，libsmctrl通过直接修改CUDA stream metadata（GPC配置掩码）绕过context切换开销，实现微秒级（~4us）的SM重分区。

从kernel调度角度拆解术语：

libsmctrl修改stream mask的kernel执行流程：

```
// 1. 获取GPU GPC/TPC拓扑信息
gpc_info = libsmctrl_get_gpc_info()  // 返回GPC数量和每个GPC的TPC数量

// 2. 构建SM mask（A100: 108 SM, 以16 SM为粒度, 6种配置; H100: 132 SM, 7种配置）
// mask是一个GPC配置掩码，指定哪些GPC中的哪些TPC可用于后续kernel执行
sm_mask_prefill = build_sm_mask(sm_ids=[0..71])   // 72个SM给prefill
sm_mask_decode  = build_sm_mask(sm_ids=[72..107]) // 36个SM给decode

// 3. 将CUDA stream绑定到SM mask
cudaStream_t stream_prefill, stream_decode;
cudaStreamCreate(&stream_prefill);
cudaStreamCreate(&stream_decode);
libsmctrl_set_stream_mask(stream_prefill, sm_mask_prefill);
libsmctrl_set_stream_mask(stream_decode, sm_mask_decode);

// 4. 后续kernel launch自动限制在mask指定的SM子集
// Prefill engine: 在其stream上发射layer-wise kernel
for layer in transformer_layers:
    launch_qkv_kernel(stream_prefill, ...)    // 仅在SM 0-71上执行
    launch_attention_kernel(stream_prefill, ...)
    launch_mlp_kernel(stream_prefill, ...)

// Decode engine: 在其stream上发射CUDA Graph decode step
cudaGraphLaunch(decode_graph, stream_decode)  // 仅在SM 72-107上执行

// 5. Scheduler动态下发repartition command（平均4.1us延迟）
// 当prefill队列堆积或decode接近SLO边界时，更新stream mask
new_sm_mask_prefill = build_sm_mask(sm_ids=[0..89])  // 扩展prefill SM
libsmctrl_set_stream_mask(stream_prefill, new_sm_mask_prefill)
// 后续kernel立即在新SM子集上运行
```

术语一般如何实现？如何使用？

libsmctrl通过修改CUDA stream的内部metadata（具体是GPC配置掩码）工作。当stream上发射一个grid时，CUDA的GigaThread Engine根据stream的SM mask决定将thread block分发到哪些SM。修改mask后，已排队的后续kernel自动遵从新mask，无需重新创建stream或context。

Bullet使用libsmctrl而非GreenContext的原因：libsmctrl的mask更新开销仅~4us，而GreenContext的context切换需要重新初始化CUDA Graph等资源。Bullet的resource manager收到scheduler的repartition command后，直接调用libsmctrl_set_stream_mask()更新相应stream的mask。

依赖与限制：libsmctrl是用户空间库，利用CUDA driver未文档化的内部API修改stream metadata，依赖特定NVIDIA driver版本兼容性。Bullet论文使用CUDA 12.4 + NVIDIA driver。GitHub仓库包含修改版libsmctrl（https://github.com/zejia-lin/BulletServe）。

涉及论文标题：
- Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration

---

