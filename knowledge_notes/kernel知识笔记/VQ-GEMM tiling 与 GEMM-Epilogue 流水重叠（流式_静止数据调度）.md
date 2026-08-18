## VQ-GEMM tiling 与 GEMM-Epilogue 流水重叠（流式/静止数据调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
EVA 的 tiling 与调度策略：VQ-GEMM 涉及的张量（input、WI、WC、output）预先在 DRAM 布置，其中仅 input 与 WI 分块流式（streaming）载入——input 按 v×d（v=32, d=8）tile 载入片上 buffer 参与一轮计算，WI 按 v×N 流式流入（v×N 大，靠流式平衡吞吐与 buffer 占用）；WC 与 output 在层内静止（stationary）于片上 SRAM——WC 只读全程复用，output tile 逐轮更新、全部部分和累加后写回 DRAM。片上 buffer 分两个区：GEMM 计算区（input+WC）与 epilogue 区（WI+output）。调度上 GEMM 与 Epilogue 重叠：VQ-GEMM（256 cycles）产出 OC 直接片内送 EU（4096 cycles, N=4096），无片外往返、无带宽争用，EU 为关键路径、近峰值利用率；多 batch 时不同请求复用同一 weight tile 降带宽。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# decode 一层 FC（N=4096）：
for tile in tiles:                          # tile 高 v=32
    load X_tile(32×8), stream WI_tile(32×N) # 流式（DRAM→片上）
    O_tile = X_tile @ B                     # 32×8 阵列 VQ-GEMM，256 cycles（WC 静止）
    y_tile = EU(O_tile, WI_tile)            # 查找+加法树，4096 cycles（与下一 tile GEMM 重叠）
    writeback y_tile                        # output 静止累加后写回
```
作用：把 decode 从"带宽受限的 GEMV"变成"加法受限的 GEMM+Epilogue"；EU 为瓶颈时可加 EU 数（DSE：4 EU 匹配 64GB/s 带宽），GEMM 单元部分空闲但扩展代价极小。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：EVA 片上 buffer 528KB（16KB WC + 256KB weight + 32KB input + 192KB OC + 32KB output）；decode tiling m=M、k=4·v·d/M、n=N；prefill tiling m=1024、k=32、n=1024。使用方式：作为加速器设计参数做 DSE（Table III/Fig. 8）；PE:EU 比例 = 2^n:N 决定瓶颈——2^n<N 时 EU-bound（可加 EU），2^n>N 时 PE-bound 且 spurious 乘法增多；batch scaling 下 EU 阶段多请求复用 weight tile（Fig. 7c）。

涉及论文标题：
- EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture
