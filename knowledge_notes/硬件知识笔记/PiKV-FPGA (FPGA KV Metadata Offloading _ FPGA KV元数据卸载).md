## PiKV-FPGA (FPGA KV Metadata Offloading / FPGA KV元数据卸载)

术语是什么？
PiKV-FPGA 是 PiKV 提出的 GPU+FPGA 异构 KV cache 管理架构。核心思想：将 MoE KV cache 管理中的 metadata-heavy 阶段（routing、compression codec、scheduling scoring、page table lookup）从 GPU 卸载到 FPGA SmartNIC（AMD Alveo U55C / Intel Agilex），GPU 仅执行 encoding 和 attention 的核心计算。FPGA 通过 CXL Type-3 链接 disaggregated DDR 内存池——KV payload 存储在 CXL-attached DDR 中，FPGA 仅保留 metadata（page table、scores、codec weights）在 on-chip SRAM。系统拓扑：GPU → MMIO (AXI-Lite, 32B command queue) → PiKV-CTRL → {Routing/ScoreFuse/Codec/Scheduler} engines → CXL.mem DMA → DDR pool。GPU 每 decode step 接收 FPGA 打包好的 {(K̂,V̂,idx)}_{i∈P_t}，绕过 KV 管理的 metadata 关键路径。

从硬件架构角度拆解术语：
PiKV-FPGA 的硬件模块组织与 per-token 数据流：

```
# === FPGA On-Chip Modules (PiKV-CTRL) ===
# Table 5: Shared PiKV-FPGA modules
Module C (Cache):   Γ: (t,e)↦addr lookup + gather
                    BRAM_Γ = E·S·(32+48) bits
Module R (Routing): ScoreFuse + radix Top-k
                    {μ_e, m_e} per-expert stats in SRAM
Module C_cmp (Comp): Codec_ρ engine (LoRA/Pyramid/Chunk/FastV)
                    {W, σ} codec weights in URAM/BRAM
Module S (Sched):   u_i ≷ θ comparator
                    {(r_i, f_i)} per-page metadata in BRAM

# Per-token data path (from GPU query to attention):
# Step 1: GPU sends q_t via MMIO
q_t → [AXI-Lite MMIO, 32B] → PiKV-CTRL

# Step 2: Routing Engine (T_route = ⌈E/16⌉/f_fpga)
for each expert e (parallel 16-wide):
    score[e] = ScoreFuse(q_t, e)   # affinity + penalty terms
g_t = radix_topk(score, k)         # selects k experts

# Step 3: Page Table Lookup (T_Γ = 2/f_fpga per expert)
for e in g_t:
    for each shard s relevant to q_t:
        addr = Γ.lookup(t_hash, e)  # BRAM read
        fetch_req.append(addr)

# Step 4: DDR Fetch + Decompress (T_ddr = 2d'/B_mem, T_codec varies)
for addr in fetch_req:
    (K̂, V̂) = DMA_read(addr)         # CXL.mem read from DDR
    (K, V) = Codec_ρ.decode(K̂, V̂)   # LoRA matvec / Pyramid inverse / ...

# Step 5: Scheduling + Page Selection
for page in fetched_pages:
    u_i = Scheduler.score(page)     # H2O/AdaKV/Duo/...
    if u_i >= θ:
        output_pages.append((K, V, page.idx))

# Step 6: Return to GPU via PCIe/CXL
GPU ← [(K̂, V̂, idx) for valid pages]

# Step 7: GPU FlashAttention
y_t = FlashAttention(q_t, returned_KV_pages)
```

**Resource Budget**（tile: E=64, S=256, k=4, K=16, d=128）：
- `BRAM_Γ` ≈ E·S·80bit = 64·256·80 = 1,310,720 bit ≈ 164 KB
- `BRAM_meta` ≈ k·K·S·48bit = 4·16·256·48 = 786,432 bit ≈ 98 KB
- `URAM_W` = d·r = 128·8 = 1,024 bit ≈ 128 B (LoRA weights)
- Total ≈ 262 KB on-chip, 可装入单 U55C SLR（典型 BRAM/URAM > 1 MB/SLR）

**Latency per token**：
$$T_{\text{fpga}} = T_{\text{route}} + k\big(T_{\Gamma} + K(T_{\text{ddr}} + T_{\text{codec}})\big)$$
$$B_{\text{step}} \approx \frac{2kd'|\mathcal{P}_t|}{\rho_{\text{link}}} + k\log E$$
其中 T_route = ⌈E/16⌉/f_fpga, T_Γ = 2/f_fpga, T_ddr = 2d'/B_mem。

术语一般如何实现？如何使用？
- PiKV 实现：`core/fpga/` 包含完整 Vivado 工程——RTL (Verilog) + Tcl build scripts + XDC constraints + host C library (libpikv_fpga.so)。
- 构建流程：`build_fpga.sh` → Vivado read RTL → synthesis → place & route → generate bitstream → program FPGA。
- 硬件需求：AMD Alveo U55C (xcu55c-fsvh2892-2L-e)，CXL Type-3 支持的 DDR 内存池。
- GPU-FPGA 接口：MMIO over AXI-Lite (32B command queue)，KV payload over CXL.mem DMA。
- 相关系统：CXL-SpecKV (Liu & Yu, FPGA '26)——同作者在 FPGA 上实现 speculative KV prefetching + compression；KVCache-AI ecosystem (NVIDIA kvpress integration via PiKVpress)。

涉及论文标题：
- PiKV KV Cache Management System for Mixture of Experts
- SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations
