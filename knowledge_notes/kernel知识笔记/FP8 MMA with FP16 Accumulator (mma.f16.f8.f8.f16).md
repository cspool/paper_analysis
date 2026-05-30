## FP8 MMA with FP16 Accumulator (mma.f16.f8.f8.f16)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

FP8 MMA with FP16 Accumulator（PTX 指令 `mma.sync.aligned.m16n8k32.row.col.f16.f8.f8.f16`）是 NVIDIA Ada Lovelace (RTX 4090) 及更新架构 GPU 上的一种 Tensor Core 矩阵乘加指令变体。它将两个 FP8 E4M3 操作数矩阵相乘，在内积累加过程中使用 FP16（半精度浮点，表示范围 ±65504）作为累加器数据类型，而非默认的 FP32 累加器。该指令的 tile 形状为 M=16, N=8, K=32，即每条指令处理 16×32 的 A 矩阵与 32×8 的 B 矩阵的乘法，产生 16×8 的 FP16 累加结果。根据 NVIDIA Ada 架构白皮书，FP8 MMA with FP16 accumulator 相对 FP16 MMA 实现 4× 理论吞吐加速，而 FP8 MMA with FP32 accumulator（mma.f32.f8.f8.f32）仅实现 2× 加速。在 SageAttention2++ 中，该指令专门用于 attention 中 P×V（attention weight × value）矩阵乘法，替代 SageAttention2 中使用的 mma.f32.f8.f8.f32 指令。

从kernel调度角度拆解术语：

该指令在 attention kernel 的 P×V 阶段被调用，核心调度考虑是 FP16 累加器仅能安全表达 ±65504 范围内的值，而 FP8 MMA 的 mma.m16n8k32 指令在 K=32 维度上会累积 32 个 p×v 乘积项。为保证累加不溢出，需要满足约束：

$$|32 \times p_{\max} \times v_{\max}| \leq 65504$$

即 $p_{\max} \times v_{\max} \leq 2047$。

在 SageAttention2++ kernel 中，P 被量化为 FP8 范围为 [-224, 224]（$P_r=224$），V 被量化为 FP8 范围为 [-4.5, 4.5]（$V_r=4.5$），满足 $224 \times 4.5 = 1008 \leq 2047$。

Kernel 伪代码（P×V 部分）：
```
// 每个 SM 处理一个 P×V tile
δ_P = max(|P̃|) / 224        // 缩小的 per-block scale
δ_V = colmax(|V|) / 4.5     // 缩小的 per-channel scale

P̂ = cvt_fp8_e4m3(P̃ / δ_P)    // 量化 P 到 FP8, 范围 [-224,224]
V̂ = cvt_fp8_e4m3(V / δ_V)    // 量化 V 到 FP8, 范围 [-4.5, 4.5]

acc_fp16 = 0 (FP16)
for k_step in range(K_dim / 32):
    p_tile = load_fp8(P̂[k_step*16 : (k_step+1)*16][:])   // 16×32 FP8
    v_tile = load_fp8(V̂[k_step*32 : (k_step+1)*32][:])   // 32×8 FP8
    // PTX: mma.sync.aligned.m16n8k32.row.col.f16.f8.f8.f16
    acc_fp16 += mma_f16_f8_f8_f16(p_tile, v_tile)

// 反量化
O = cvt_fp16_to_fp32(acc_fp16) * δ_P * δ_V
```

Annotations:
- `δ_P, δ_V`：per-block/per-channel scale factors，约束乘积 ≤1023.5（含 delayed FP32 buffering 时）
- `acc_fp16`：FP16 累加器，32 次内积累加结果 ≤32256 < 65504
- `mma_f16_f8_f8_f16`：关键 PTX 指令，4× FP16 理论吞吐
- `K_dim/32`：沿 K 维度的 tile 循环次数，每步处理 32 个元素的内积
- 与 mma.f32.f8.f8.f32 的关键区别：累加器从 FP32 变为 FP16，理论吞吐翻倍，但要求量化范围更窄以保证数值安全

术语一般如何实现？如何使用？

该指令是 NVIDIA PTX ISA 的一部分，从 Ada Lovelace (SM 8.9) 架构开始支持，在 Blackwell (RTX 5090) 架构上同样可用。在 CUDA 中通过内联 PTX 汇编调用：

```cuda
// CUDA inline PTX
asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.f16.f8.f8.f16 "
    "{%0, %1, %2, %3}, "
    "{%4, %5, %6, %7}, "
    "{%8, %9}, "
    "{%10, %11, %12, %13};"
    : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1),
      "r"(c0), "r"(c1), "r"(c2), "r"(c3)
);
```

在 CUTLASS 等模板库中，该指令被抽象为 `warp::mma` 操作，开发者通过指定 OperandA/OperandB/Accumulator 类型来间接选择指令变体。SageAttention2++ 直接在 CUDA kernel 中使用该指令实现 P×V 的量化 Matmul。

前置条件：使用该指令前必须确保量化后的 FP8 张量值在 FP16 累加可安全表达的范围内，即通过缩小量化范围（narrowing quantization range）来控制操作数上界。这与传统的"最大范围量化"（max range quantization，如 E4M3 的 [-448, 448]）形成对比。

涉及论文标题：
- SageAttention2++: A More Efficient Implementation of SageAttention2
