# LiquidQuant W4A8 — LiquidGEMM/LiquidServe 量化技术详解

- 导出时间：2026-06-03
- 来源：当前 Claude Code session 可用对话上下文
- 保存路径：human_notes/LiquidQuant_W4A8_LiquidGEMM_量化技术详解.md
- 范围：仅用户输入与 Claude 最终输出
- 涉及论文：LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving (ByteDance)

## 对话记录

### 001 User
量化	LiquidQuant W4A8	两级量化 FP16→INT8→UINT4；dequant 仅需 IMAD+XOR 两条 32-bit 指令处理 4 元素；H800 接近 FP8 性能	LiquidServe (ByteDance)

### 002 Claude
你分享了一条量化技术表格行数据，但我需要确认你的意图。你是想让我：

1. **搜索 Obsidian 知识库** 中与 LiquidQuant / LiquidServe 相关的内容？
2. **详细解释** 这个技术（LiquidQuant W4A8 两级量化、IMAD+XOR dequant 指令等）？
3. **将此条写入** 某个笔记或分析文档？
4. **整理成表格**，与其他量化方法（如 AWQ、GPTQ、HQQ 等）对比？

请告诉我你希望我做什么。

### 003 User
教我

### 004 Claude
下面从**问题动机 → 数学原理 → 硬件实现 → 系统集成 → 实验结果**五个层次，完整讲解 LiquidQuant W4A8。

---

## 一、为什么需要 W4A8？它为什么难？

### 1.1 量化精度 vs 内存带宽的权衡

LLM serving 的 GEMM（矩阵乘法）有两种 bound 状态：

| 场景 | 瓶颈 | 示例 |
|------|------|------|
| **Memory-bound**（小 batch） | HBM 带宽不足以喂饱 Tensor Cores | batch=1~16 推理 |
| **Compute-bound**（大 batch） | Tensor Cores 算力成为限制 | batch≥128 推理 |

W4A8 的理论优势：
- **权重 4-bit** → 内存占用比 W8A8 减半 → memory-bound 场景带宽压力小
- **激活 8-bit** → 用 INT8 Tensor Core MMA（990 TFLOPS on H100），比 FP16 激活快
- **memory→compute 转折点 batch size 从 300 降到 150**（H100 上）

### 1.2 Roofline 理论 vs 现实的鸿沟

论文作者实测发现（Section 3.1）：**W4A8 在小 batch 和 W8A8 差不多，大 batch 反而慢近 2×，甚至不如 FP16/W4A16！**

原因：**dequantization（反量化）成为了隐藏的瓶颈。**

```
W8A8 Symmetric GEMM:
  Load W8→SMEM → ldmatrix→RF → WGMMA(W8, A8)  ← 全在 Tensor Core 数据路径

W4A8 Asymmetric GEMM:
  Load W4→SMEM → LDS→RF → Dequant(W4→W8, CUDA Cores) → WGMMA(W8, A8)
                           ^^^^^^^^^^^^^^^^^^^^^^^^
                           这个步骤是瓶颈！
```

**H100 硬件异构性**：CUDA Core FP32 ≈ 60 TFLOPS，Tensor Core INT8 ≈ 990 TFLOPS——**16.5× 差距**。如果 dequant 太重，CUDA Core 会卡住整个 pipeline，Tensor Core 闲置等待。

---

## 二、LiquidQuant 核心算法

### 2.1 两级量化架构

```
离线量化（两阶段）：
  FP16 weight
    ↓ per-channel quant（第一级）
  INT8  Q_i8 ∈ [-119, 119]   ← protective range，防溢出
    ↓ per-group quant（第二级，group_size=64）
  UINT4 Q_u4 ∈ [0, 15]
```

**关键洞察**：第一级 INT8→FP16 的 dequant 在 GEMM epilogue 完成，开销可忽略。瓶颈在第二级 UINT4→INT8，这个必须在 main-loop 里做。

### 2.2 Shift-to-Unsigned 技巧

传统的 QoQ（QServe 的方法）直接从 INT8 量化到 INT4，dequant 时面临 **two's complement 溢出问题**：INT8 的负数（如 -104 = `1001 1000`）做加法时会 wraparound。

LiquidQuant 的做法：
1. **先将 INT8 shift 到 UINT8 域**：`Q_u8 = Q_i8 - min(Q_i8)`，把 `[-119, 119]` 映射到 `[0, 238]`
2. **再量化到 UINT4**：`Q_u4 = round(Q_u8 / s_u8)`，其中 `s_u8 ≤ 16`
3. 现在所有值都是非负的，**乘法不会溢出**：`Q_u4 × s_u8 ≤ 15 × 16 = 240 < 256` ✓

### 2.3 XOR 0x80 的数学魔术——这是全文最精巧的设计

Dequant 需要还原到 INT8：
```
Q̂_i8 = Q_u4 × s_u8 + min(Q_i8)
```

问题：`min(Q_i8)` 是负数（如 -104），直接加会溢出。

**核心数学洞察**：利用模 2⁸ 同余性质——INT8 的 `i` 和 UINT8 的 `j` 如果满足 `i ≡ j (mod 256)`，则二进制表示相同。例如 `-3 ≡ 253 (mod 256)`，都是 `1111 1101`。

推导：
```
Q̂_i8 ≡ Q_u4 × s_u8 + min(Q_i8) + x × 256   (mod 256)
       = Q_u4 × s_u8 + (128 + min(Q_i8)) + (2x-1) × 128
       = Q_u4 × s_u8 + a + b
```

其中 `a = 128 + min(Q_i8)` 可离线预计算，`b = (2x-1) × 128`。

**控制 x 消除溢出**：
- 如果 `Q_u4 × s_u8 + a ≥ 128`，设 x=0，b = -128
- 如果 `Q_u4 × s_u8 + a < 128`，设 x=1，b = +128

数学上可以严格证明 `Q_u4 × s_u8 + a + b` 始终在 [0, 255] 内（论文 Equation 11）。

**关键简化**：加 b 等价于**翻转 MSB（最高位）**！+128 把 MSB 从 0 翻成 1，-128 把 MSB 从 1 翻成 0。而翻转 MSB = XOR 0x80！

所以最终 dequant 公式：
```
Q̂_i8 = (Q_u4 × s_u8 + a) ⊕ 0x80
```

**不需要分支判断，不需要条件执行，一条 IMAD + 一条 XOR 完成！**

---

## 三、硬件实现：IMAD + XOR 两条指令处理 4 个元素

### 3.1 指令级细节

```
输入：32-bit 寄存器，存储 4 个 packed UINT8 值（或 8 个 UINT4 unpack 后）

Step 1: Unpack
  8×UINT4 → 2×32-bit 寄存器（每个寄存器 4×UINT8）
  方法来自 QServe

Step 2: Dequantize（每条 32-bit 寄存器，4 个元素）
  reg = IMAD(reg, s_u8_bcast, a_bcast)   // 4 路并行 multiply-add
  reg = XOR(reg, 0x80808080)              // 翻转每个 byte 的 MSB

结果：寄存器中已经是 INT8 的二进制表示 → 直接喂给 WGMMA
```

- `IMAD`：Integer Multiply-Add，GPU 原生 32-bit 指令，一个周期完成 `dst = src0 × src1 + src2`
- `XOR 0x80808080`：对每个 byte 的 bit 7 取反，等价于条件性地 ±128
- `s_u8` 和 `a` 离线预计算，存入 constant memory

### 3.2 α 因子——为什么 0.875 是 magic number

每元素指令数 α 是衡量 dequant 开销的核心指标：

| 方法 | 指令数（8 元素） | α（指令/元素） |
|------|:---:|:---:|
| QServe QoQ | ~80+ | ≥10 |
| **LiquidQuant** | **7**（含 unpack） | **0.875** |

H100 上消除 bottleneck 的阈值：
- 要与 weight loading 重叠：α ≤ 5.07
- 要与 MMA 重叠：α ≤ 5.05

QServe 的 α≥10 远超阈值 → dequant 成瓶颈。LiquidQuant 的 α=0.875 远低于阈值 → **dequant 被 pipeline 有效隐藏**。

### 3.3 Implicit Fine-Grained Pipeline (ImFP)

LiquidGEMM 的第二个关键技术。传统做法（ExCP）是串行的：

```
ExCP:  Load tile_k → Dequant → MMA → Load tile_{k+1} → Dequant → MMA → ...
       └── Warp Group 0 ──┘   └── Warp Group 1 ──┘
       两组交替执行，但同一时刻只有一个硬件单元工作
```

ImFP 的做法：
```
       WG_0 (CUDA Core): Dequant tile_k    │ Dequant tile_{k+2} │ ...
       WG_1 (Tensor Core): MMA tile_{k-1}  │ MMA tile_{k+1}     │ ...
       WG_2 (TMA): Load tile_{k+1}         │ Load tile_{k+3}    │ ...
       ─────────────────────────────────────────────────────────────→ time
       三个 warp group 并发，三种硬件单元同时工作，无 barrier！
```

因为 LiquidQuant 的 α=0.875 足够小，CUDA Core 不会成为瓶颈，三种硬件单元可以**天然并发**。

---

## 四、系统集成：LiquidServe

LiquidGEMM 是 kernel 层，LiquidServe 是端到端 serving 系统：

```
LiquidServe 系统栈：
┌─────────────────────────────────────────┐
│  FlashAttention-2    │  attention 计算    │
│  PagedAttention      │  KV cache 管理     │
│  SmoothQuant         │  激活 INT8 量化    │
│  LiquidGEMM          │  W4A8 GEMM kernel  │  ← 核心
│  CUTLASS 3.x + CuTe  │  warp-specialized  │
│  CUDA 12.4 + PTX     │  TMA/WGMMA/mbarrier│
└─────────────────────────────────────────┘
硬件：NVIDIA H800 (80GB), Intel Xeon 8457C
```

关键设计选择：
- KV cache 用 **INT8**（非 4-bit），保证 batch size 可扩展
- 激活量化用 SmoothQuant per-token 动态量化，fuse 到前序 kernel
- 未开源，论文说明已部署为 **ByteDance Seed 生产环境**的 primary GEMM kernel

---

## 五、实验结果

### 5.1 系统级（LiquidServe vs QServe vs TensorRT-LLM，H800）

| Model | TRT-FP8 | QServe | LiquidServe | 加速比 |
|-------|---------|--------|-------------|--------|
| LLaMA2-7B | 5,913 | 5,402 | **6,721** (bs=194) | 1.14× |
| LLaMA2-13B | 3,402 | 3,311 | **4,105** (bs=119) | 1.21× |
| LLaMA2-70B | 948 | 871 | **3,695** (bs=184) | **3.16× vs TRT-W8A8** |
| LLaMA3-8B | **16,820** | 5,240 | 16,694 (bs=256) | 0.99× |
| Mistral-7B | **17,433** | 5,361 | 17,011 (bs=256) | 0.98× |
| Mixtral 8×7B | 8,296 | NA | **10,745** (bs=256) | 1.30× |

**大模型优势明显**：LLaMA2-70B 上 batch size 能跑到 184（QServe 只有 64），因为 4-bit 权重 + INT8 KV cache（QServe 用 4-bit KV cache）。

LLaMA3-8B/Mistral-7B 略低于 TRT-FP8，因为 TRT-FP8 针对 H800 FP8 attention kernel 做了优化（LiquidServe 用 INT8 activation 而非 FP8）。

### 5.2 Kernel 级消融

LiquidServe vs LiquidServe/wo（把 LiquidGEMM 替换为 QServe 的 W4A8 kernel）：**1.13×–1.98× 端到端加速**，证明 LiquidGEMM kernel 是系统加速的核心来源。

---

## 六、总结：一张图理解全貌

```
                          离线量化
  FP16 ──→ INT8 (per-channel, [-119,119]) ──→ UINT4 (per-group, g=64)
           ↑ 第一级，epilogue 处理              ↑ 第二级，LiquidQuant

                          在线推理 (main-loop)
  ┌──────────┐    ┌──────────────┐    ┌──────────────┐
  │ TMA Load │ →  │ CUDA Dequant │ →  │ Tensor MMA   │
  │ UINT4    │    │ IMAD + XOR   │    │ INT8 WGMMA   │
  │ GMEM→SMEM│    │ α=0.875/el   │    │ W8×A8 → INT32│
  └──────────┘    └──────────────┘    └──────────────┘
       ↑               ↑                    ↑
    WG_Load         WG_Compute          WG_Compute
       └───────────────┴────────────────────┘
              三个 warp group 并发 (ImFP)
```

**核心贡献一句话**：LiquidQuant 通过 shift-to-unsigned + XOR 0x80 的数学技巧，将 W4A8 dequant 压缩到每元素 0.875 条指令，使 CUDA Core 不再成为瓶颈，让 W4A8 从"理论美好、实践拉胯"变成了真正可用的量化方案，在 H800 上实现接近甚至超越 FP8 的 serving 性能。
