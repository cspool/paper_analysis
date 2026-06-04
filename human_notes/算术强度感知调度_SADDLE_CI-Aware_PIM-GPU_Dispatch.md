# 算术强度感知调度 —— SADDLE CI-Aware PIM-GPU Dispatch 学习笔记

- 导出时间：2026-06-04
- 来源：当前 Claude Code session 可用对话上下文（用户选中 `draft/review_draft.md` L68 行要求教学）
- 保存路径：human_notes/算术强度感知调度_SADDLE_CI-Aware_PIM-GPU_Dispatch.md
- 范围：仅用户输入与 Claude 最终输出

## 对话记录

### 001 User
（IDE 选中 `draft/review_draft.md` 第68行：`| 算术强度感知 | SADDLE CI-Aware PIM-GPU Dispatch* | 预标定PIM ridge (16.7 FLOP/Byte) vs GPU ridge (208 FLOP/Byte)→**运行时CI vs ridge比较动态operator-to-device映射**；CI<PIM_ridge→memory-bound→PIM高带宽；CI>GPU_ridge→compute-bound→GPU高算力 | Ridge point差距13×驱动异构调度决策 | Q2.4, vault: Arithmetic Intensity-Aware (333.5) |`）

教我

### 002 Claude
（详细讲解算术强度感知调度，包含以下章节：）

## 1. 先修知识：什么是算术强度（Arithmetic Intensity, CI）？

**定义**：

$$\text{CI} = \frac{\text{FLOPs}}{\text{Bytes of Memory Traffic}} \quad \text{单位：FLOP/Byte}$$

- **分子**：一个 operator/kernel 需要执行多少浮点运算
- **分母**：这个 operator/kernel 需要从内存读/写多少字节数据

**直觉**：每从内存搬 1 Byte 数据，能"顺便"做多少次浮点运算？这个比值越高，说明计算密度越高。

## 2. Roofline 模型与 Ridge Point（脊点）

Roofline 模型（Williams et al., 2009）是理解 CI-Aware Dispatch 的理论基础：

```
可达到性能 (FLOP/s)
  ↑
  │        ╱ memory-bound
  │       ╱  区域 (斜坡)
  │      ╱
  │     ╱ ← slope = bandwidth
  │    ╱
  │   ╱
  │  ╱              ridge point (脊点)
  │ ╱              │
  │╱               │    ←─────── compute-bound 区域 (天花板)
  └─────────────────┴─────────────────────→ 算术强度 CI (FLOP/Byte)
```

**Ridge Point 公式**：

$$\text{CI}_{\text{ridge}} = \frac{\text{Peak Compute (FLOP/s)}}{\text{Peak Memory Bandwidth (Byte/s)}}$$

**判定规则**：
- **CI < ridge** → operator 是 **memory-bound**（受限于带宽，算力闲置）
- **CI > ridge** → operator 是 **compute-bound**（受限于算力，带宽闲置）

## 3. 为什么需要 CI-Aware Dispatch？—— PIM+GPU 异构系统的挑战

SADDLE 的系统包含两种计算设备：

| 设备 | 特点 | Peak Compute | Peak Bandwidth | Ridge Point |
|------|------|-------------|----------------|-------------|
| **HBM-PIM** | DRAM 内嵌 PE array，算力弱但带宽极高 | ~2.4 TFLOPS | ~144 TB/s (内部) | **~16.7 FLOP/Byte** |
| **A100 GPU** | 传统加速器，算力强但 HBM 带宽相对低 | ~312 TFLOPS | ~1.5 TB/s (HBM2e) | **~208 FLOP/Byte** |

**两个 ridge point 差距 13×！** 这意味着：

- 同一个 operator，可能在 PIM 上是 compute-bound，但在 GPU 上是 memory-bound
- **静态映射（offline 决定谁跑什么）必然在某些 batch size / sequence length 下 suboptimal**

这就是 SADDLE 要解决的核心问题：**speculative decoding 中 draft length 和 effective batch size 动态变化 → operator CI 动态变化 → 最优设备也动态变化**。

## 4. SADDLE CI-Aware Dispatch 的核心机制

### 4.1 两步走：预标定 + 运行时决策

**Step 1: Offline 预标定（一次性）**

```python
# 测量每个设备的实际硬件参数
PIM_peak_compute = 2.4e12      # 2.4 TFLOPS (HBM-PIM PE array)
PIM_peak_bw      = 144e12      # 144 TB/s (PIM internal bandwidth)
PIM_ridge = PIM_peak_compute / PIM_peak_bw  # ≈ 16.7 FLOP/Byte

GPU_peak_compute = 312e12      # 312 TFLOPS (A100 Tensor Core FP16)
GPU_peak_bw      = 1.5e12      # 1.5 TB/s (HBM2e)
GPU_ridge = GPU_peak_compute / GPU_peak_bw  # ≈ 208 FLOP/Byte
```

**Step 2: Runtime Per-Iteration CI 估算 + 设备选择**

```python
def schedule_DLM_FC(active_requests, d_model):
    """
    DLM (Draft Language Model) 的 Fully-Connected 层调度
    
    矩阵乘法: [eff_bs, d_model] × [d_model, d_model]
    """
    eff_bs = len(active_requests)  # 当前还在 draft 的请求数
    
    # 估算 FLOPs 和 Bytes
    FLOPs = 2 * eff_bs * d_model**2       # GEMM: 2MNK
    Bytes = (eff_bs * d_model + d_model**2) * 2  # FP16: 2 bytes/element
    
    CI = FLOPs / Bytes  
    # 近似: 当 eff_bs << d_model 时，CI ≈ eff_bs
    
    if CI < PIM_ridge:      # CI < 16.7 → memory-bound on both
        return "PIM"         # PIM 带宽优势（144 TB/s >> 1.5 TB/s）
    else:                    # CI > 16.7 → 在 PIM 上可能 compute-bound
        return "GPU"         # GPU 算力优势（312 TFLOPS >> 2.4 TFLOPS）


def schedule_TLM_attention(total_tokens, d_head):
    """
    TLM (Target Language Model) 的 Attention 层调度
    
    Attention: QK^T softmax × V
    """
    FLOPs = 4 * total_tokens * d_head**2     # QK^T + SV
    Bytes = 2 * total_tokens * d_head * 2     # KV reads (FP16)
    
    CI = FLOPs / Bytes  # 近似: ≈ 2 * d_head
    
    if CI > GPU_ridge:       # CI > 208 → compute-bound even on GPU
        return "GPU"          # GPU 算力充分利用
    else:                     # CI < 208 → memory-bound on GPU
        return "PIM"          # 去带宽更高的 PIM
```

### 4.2 初始固定映射 vs 动态 Remap

| Operator | 初始固定映射 | 原因 |
|----------|------------|------|
| **DLM Attention** | → PIM | 每 iteration 仅 1 token/request，CI 极低 |
| **TLM FC** | → GPU | Shared Pool 聚合 token 后变为 compute-intensive GEMM |
| **DLM FC** | 🔄 **动态 remap** | eff_bs 变化导致 CI 在 PIM_ridge 上下波动 |
| **TLM Attention** | 🔄 **动态 remap** | draft length 变化导致 CI 跨越 GPU_ridge |

## 5. 具体例子：一次 Speculative Iteration 的调度过程

假设系统有 12 个请求（OPT-66B TLM + OPT-1.3B DLM），d_model = 2048，d_head = 64：

```
Iteration N:
┌─────────────────────────────────────────────────────────────────┐
│ 1. DLM Prediction Phase                                         │
│    - 12 requests 各自生成 1 draft token                          │
│    - DLM Attention: CI ≈ d_head/2 ≈ 32 FLOP/Byte               │
│      32 > PIM_ridge(16.7) but < GPU_ridge(208)                  │
│      → PIM (memory-bound on GPU, bandwidth wins on PIM)         │
│                                                                  │
│    - DLM FC: eff_bs=12, CI ≈ 12 FLOP/Byte                       │
│      12 < PIM_ridge(16.7) → PIM                                 │
│                                                                  │
│ 2. Draft Tokens → Shared Pool (CAM-based, 1KB)                  │
└─────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Intermediate: 部分请求提前完成 drafting                        │
│    - 4 个请求 drafting 结束（短序列先完成）                        │
│    - eff_bs 从 12 → 8                                           │
│                                                                  │
│    DLM FC (下一轮): eff_bs=8, CI ≈ 8                             │
│    8 < PIM_ridge(16.7) → 仍然 PIM                               │
│                                                                  │
│    ⚠ 但当 eff_bs 降至 4: CI ≈ 4                                 │
│    4 << PIM_ridge → 更加 memory-bound → PIM 优势更大             │
└─────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. TLM Verification Phase                                       │
│    - Shared Pool 中共 36 tokens（不同请求有不同 draft length）     │
│                                                                  │
│    - TLM Attention: CI ≈ 2 × d_head ≈ 128 FLOP/Byte             │
│      128 < GPU_ridge(208) → GPU memory-bound                    │
│      → remap to PIM！                                           │
│                                                                  │
│    - TLM FC: pooled 36 tokens → CI ≈ 36                         │
│      36 << GPU_ridge(208) but > PIM_ridge(16.7)                 │
│      → GPU preferred (compute-bound 侧)                          │
└─────────────────────────────────────────────────────────────────┘
```

**关键动态**：

```
eff_batch_size ↓ → DLM FC 的 CI ↓ → optimal target: GPU → PIM
draft_length  ↑ → TLM Attention 的 CI ↑ → optimal target: PIM → GPU
```

## 6. 为什么 Ridge Point 差距 13× 如此重要？

画出交叉 Roofline 图就能直观理解：

```
Performance (TFLOPS) [log scale]
  ↑
312 ┤                                    ┌─── GPU ceiling
    │                                    │
    │                         GPU        │
    │                        roofline   │
    │                      ╱             │
    │                    ╱               │
    │                  ╱ ← slope=1.5TB/s │
    │                ╱                   │
 2.4┤            ┌── PIM ceiling        │
    │           ╱│                       │
    │         ╱  │  PIM roofline         │
    │       ╱    │ ← slope=144TB/s       │
    │     ╱      │                       │
    │   ╱        │                       │
    │ ╱          │                       │
    └┴───────────┼───────────────────────┴──→ CI (FLOP/Byte)
     0          16.7                    208
     │           │                       │
     │← PIM胜 →│← 共优区 →│← GPU胜 →│
```

**三区决策逻辑**：

| CI 范围 | 决策 | 原因 |
|---------|------|------|
| **CI < 16.7** | → **PIM** | memory-bound 于两设备，PIM 144TB/s 带宽碾压 GPU 1.5TB/s |
| **16.7 < CI < 208** | → **GPU** (默认) | PIM 已 compute-bound (PE 算力不够)，GPU 有充足算力 |  
| **CI > 208** | → **GPU** | compute-bound 于两设备，GPU 312TFLOPS 碾压 PIM 2.4TFLOPS |

## 7. 与同类方法的对比

| 方法 | 调度依据 | 映射时机 | 局限 |
|------|---------|---------|------|
| **SpecPIM (static)** | Offline genetic algorithm → 一次性 mapping | 推理前 | 无法适应 draft length/eff_bs 动态变化 |
| **PAPI (dynamic profiling)** | Runtime performance counter 采样 | 运行时 | Profiling overhead 大 |
| **SADDLE CI-Aware** | 轻量 CI 估算（仅需 O(1) 代数运算） | 每 iteration | 需预标定设备参数（一次性） |

**SADDLE 的效果**：
- 动态 scheduling 在 static mapping 基础上再提升 **1.13×**
- PIM ops 占比从 9.51% → 14.89%（更多 memory-bound op 正确路由到 PIM）
- GPU ops 占比从 90.49% → 85.11%（释放 GPU 去干真正 compute-intensive 的事）
- 整体吞吐提升 **1.21×**

## 8. 一句话总结

> **SADDLE 的 CI-Aware Dispatch 本质上是一个极简但精准的异构调度器：用一行除法（CI = FLOPs / Bytes）判断 operator 的"瓶颈性格"，再与两个设备的 ridge point 比较——低 CI（memory-bound）→ 去带宽高的 PIM，高 CI（compute-bound）→ 去算力强的 GPU。不需要 profiling、不需要 ML 模型、不需要离线 search，只靠预标定的 ridge point 和运行时 batch size 就能做出接近最优的设备选择。**
