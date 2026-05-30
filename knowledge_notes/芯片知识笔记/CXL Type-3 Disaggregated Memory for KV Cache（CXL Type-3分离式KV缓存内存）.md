## CXL Type-3 Disaggregated Memory for KV Cache（CXL Type-3分离式KV缓存内存）

术语是什么？
CXL Type-3 Disaggregated Memory for KV Cache 是利用 Compute Express Link (CXL) Type-3 设备（内存扩展器）将 LLM 推理的 KV cache 从 GPU HBM 迁移到外部内存池的芯片级技术。CXL Type-3 设备仅支持 CXL.mem 协议（无 CXL.cache），作为纯内存扩展器出现——GPU/CPU 可通过 load/store 语义直接访问 CXL-attached DDR/持久内存，无需经过传统的 PCIe DMA 协议栈。在 PiKV-FPGA 架构中，FPGA SmartNIC (充当 CXL Type-3 controller) 将 KV payload 存储在 CXL-attached DDR pool 中，FPGA on-chip SRAM 仅保留 metadata（page table、scores、codec weights）。GPU 通过 FPGA 的 32B MMIO command queue (AXI-Lite) 发送 query，FPGA 内部 engines 处理后通过 CXL.mem DMA 将选定的 KV pages 传回 GPU。

从芯片设计角度拆解术语：
CXL Type-3 KV Cache 的物理组织和数据路径：

```
# === Physical Topology ===
# GPU (H100/H800) ←→ PCIe/CXL Switch ←→ FPGA SmartNIC (Alveo U55C)
#                                            ↓ CXL.mem
#                                       DDR Memory Pool
# (Multi-TB disaggregated KV storage)

# === Memory Mapping ===
# CXL Type-3 device exposes Host-managed Device Memory (HDM)
# through CXL.mem protocol:
# - HDM-H (Host-only coherent): GPU accesses via MMIO, no cache coherence
# - HDM-D (Device-managed): FPGA manages internal allocation

# PiKV-FPGA memory layout in CXL-attached DDR:
DDR_Pool:
├── Page_Table_Region     # Γ: (t,e)↦(addr, size, metadata_ptr)
├── KV_Payload_Region     # (K̂_t, V̂_t) pairs, per-shard circular buffers
├── Codec_Weight_Region   # LoRA/Pyramid/SVD pre-trained weights
└── Scheduling_Meta_Region # (r_i, f_i, a_i) per-page metadata backup

# On-FPGA SRAM (PiKV-CTRL):
├── BRAM_Γ (176 KB):      # Hot page table entries (E·S entries)
├── BRAM_meta (48 KB):    # Active page metadata (k·K·S entries)
├── URAM_W:               # Codec weights (LoRA rank-r matrices)
└── ScoreFuse_Buffer:     # Routing intermediate results
```

**CXL vs PCIe for KV Cache**:
| 特性 | CXL.mem (Type-3) | PCIe DMA |
|------|-----------------|----------|
| 访问语义 | Load/Store (byte-addressable) | DMA descriptor (block transfer) |
| 延迟 | ~200ns (CXL 2.0) | ~1-3μs (DMA setup + transfer) |
| 带宽 | 64 GB/s (CXL x16) | 64 GB/s (PCIe 5.0 x16) |
| 一致性 | Host-managed coherence | No coherence |
| CPU 开销 | 零（MMIO bypass） | DMA engine + interrupt handling |
| 适用场景 | 随机小粒度 KV page 访问 | 大块 sequential expert loading |

**Latency 模型**（PiKV-FPGA per token）:
$$T_{\text{fpga}} = T_{\text{route}} + k\big(T_{\Gamma} + K(T_{\text{ddr}} + T_{\text{codec}})\big)$$
其中 T_ddr = 2d'/B_mem（CXL.mem 带宽 B_mem），ρ_link 为 on-chip compression ratio。

$$B_{\text{step}} \approx \frac{2kd'|\mathcal{P}_t|}{\rho_{\text{link}}} + k\log E$$
其中 |P_t| ≤ kK 为 selected pages。当 B_step ≪ 2dL（全量 KV 传输），GPU 完全卸载 KV metadata 关键路径。

术语一般如何实现？如何使用？
- PiKV 实现：FPGA RTL (Verilog) 中 `pikv_cxl_dma.v` 处理 CXL.mem DMA 传输，`pikv_axi_lite_slave.v` 处理 MMIO command queue。Host 侧 C library `libpikv_fpga.so` 通过 FPGA driver 控制。
- CXL-SpecKV (Liu & Yu, FPGA '26)：同作者进一步将 KV 预取 (speculative prefetching) 和压缩集成到 CXL-FPGA 架构中，4-8× 内存容量扩展，3.2× 吞吐提升。
- 产业趋势：XConn + MemVerge (SC25) 演示 100 TiB CXL memory pool 用于 LLM KV cache offload；TraCT (arXiv 2025)、Beluga (arXiv 2025) 使用 CXL Type-3 shared memory 替代 RDMA 进行跨节点 KV 传输。
- CXL 3.0 展望：支持 multi-level switching 和 fabric management，实现 rack-scale disaggregated memory pool。CXL 3.0 带宽可达 128 GB/s (x16)。

涉及论文标题：
- PiKV KV Cache Management System for Mixture of Experts
