## Pingpong Scheduling（乒乓调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Pingpong Scheduling 是 FlashAttention-3 中利用 warp-specialization 实现 inter-warpgroup 级别的 GEMM 和 softmax 重叠技术。CTA 内有两个 consumer warpgroups（各2 warps），通过 bar.sync 强制 warpgroup 1 的 GEMMs（WGMMA QK^T of iter j + WGMMA PV of iter j-1）在 warpgroup 2 的 GEMMs 之前调度。即：当 warpgroup 1 执行 softmax 时，warpgroup 2 的 Tensor Core 正在执行 GEMM；然后角色互换——warpgroup 2 执行 softmax 时，warpgroup 1 的 Tensor Core 执行 GEMM。Tensor Core 在 pingpong 交替中始终被占用。该技术解决了 attention forward pass 中的关键瓶颈：FP16 head_dim=128 时 matmul FLOPs:exponential FLOPs = 512:1，但 exponential throughput 仅 ~1/256 of matmul throughput，导致 exponential 占用 ~50% cycle time。Pingpong 将 softmax（包含 exponential）完全重叠到另一个 warpgroup 的 GEMM 执行期间。

从kernel调度角度拆解术语：
Pingpong timeline（Gantt chart 形式，每个 warpgroup 2 warps，time →）：
```
Warpgroup 1: | QK^T(j) | PV(j-1) | softmax(j) | QK^T(j+1) | PV(j) | softmax(j+1) |
Warpgroup 2: | softmax(j-1) | QK^T(j) | PV(j-1) | softmax(j) | QK^T(j+1) | PV(j) |
Tensor Core: |████ WG1 ████|████ WG2 ████|████ WG1 ████|████ WG2 ████|
CUDA Core:   |████ WG2 ████|████ WG1 ████|████ WG2 ████|████ WG1 ████|
```
关键同步：bar.sync 保证 warpgroup 2 的 WGMMA 在 warpgroup 1 的 WGMMA 之后发射，从而 warpgroup 1 的 softmax 自动与 warpgroup 2 的 GEMM 在时间上对齐（NVCC compiler 在 warp 级别独立调度各 warpgroup 的指令流）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Pingpong scheduling 需要：(1) CTA 内至少 2 个 consumer warpgroups（各2 warps），加上 1 个 producer warpgroup（1 warp for TMA），总共 5 warps per CTA（Hopper支持max 8 warps/CTA）；(2) bar.sync 指令显式同步 warpgroup 间的发射顺序；(3) 每个 warpgroup 独立等待自己的 pipeline barriers（K/V TMA load completion）。FlashAttention-3 中 pingpong scheduling 将 FP16 forward 从570 TFLOPS提升至620-640 TFLOPS（~12% gain）。该技术也可用于其他存在 non-matmul bottleneck 的 fused kernel（如 normalization + GEMM）。

涉及论文标题：
- FlashAttention-3 Fast and Accurate Attention with Asynchrony and Low-precision
