## Fused Dequantization-Matmul Kernel（融合解量化-矩阵乘的内核）

术语是什么？通过联网搜索让回答具体和精准。
Fused dequantization-matmul kernel是Quantix的核心在线计算kernel，将non-uniform quantized weights的dequantization与Tensor Core matrix multiplication融合为单一CUDA kernel。该kernel以hardware-aligned bit-shuffled weights (W1', W2')、activations A和centroids C为输入，在单次kernel launch中完成prefetch→load→dequantize→matmul全流程。与分开执行的"先dequantize kernel→写memory→matmul kernel→读memory"常规做法相比，fused kernel消除了中间全局内存往返，使dequantization latency被Tensor Core computation隐藏。kernel通过两层double buffering实现三级流水线重叠（global memory prefetch / dequantization on CUDA cores / MMA on Tensor Cores）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Fused kernel执行伪代码（Algorithm 1 in paper）：

```
// Algorithm 1: Fused Kernel in Quantix
// Input: W1' (1-bit), W2' (2-bit), A (FP16 activations), C (FP16 centroids)
// Output: Y = A × Dequant(W1', W2', C)

for each thread block in Split-K slices: // parallel
    // === Initialization ===
    Smem[0] = cp.async(W1'[tile_0], W2'[tile_0], C[tile_0], A[tile_0])
    cp.async.wait()
    
    // Load first subtile to registers and dequantize
    Reg[0] = ld.shared(Smem[0], subtile_0)
    Reg[0].Wt = dequantize(Reg[0].W1', Reg[0].W2', Reg[0].C)
    // Reg[0] now holds FP16 reconstructed weights
    
    // === Main Loop with Hierarchical Pipeline ===
    for k in range(num_K_tiles):
        // --- Inter-tile level: overlap prefetch with compute ---
        // Prefetch next K-tile to alternate shared memory buffer
        Smem[(k+1) % 2] = cp.async(W1'[k+1], W2'[k+1], C[k+1], A[k+1])
        
        // --- Intra-tile level: overlap dequant with MMA ---
        for s in range(1, num_subtiles):
            // Load next subtile from shared memory
            Reg[s % 2] = ld.shared(Smem[k % 2], subtile_s)
            
            // Dequantize on CUDA Cores (current subtile)
            Reg[s % 2].Wt = dequantize(Reg[s % 2].W1', Reg[s % 2].W2', Reg[s % 2].C)
            
            // MMA on Tensor Cores (previous subtile)
            Y_partial = mma(Y_partial, Reg[(s-1) % 2].A_frag, Reg[(s-1) % 2].Wt_frag)
            // mma.m16n8k16 on Tensor Cores
        
        cp.async.wait() // ensure prefetch complete
    
    // Reduction: merge partial sums from Split-K
    Y = reduction_kernel(Y_partials)

// === Helper: in-register dequantization ===
function dequantize(W1, W2, C):
    R_3bit = 0 // 32-bit register for 8 3-bit indices
    for pair in range(4):
        bit1 = W1[pair]         // 1-bit
        bit2 = W2[pair]         // 2-bit
        idx = (bit1 << 2) | bit2
        R_3bit |= (idx << (3 * pair))
    
    Wt = [] // reconstructed FP16 weights
    for i in range(8):
        qi = (R_3bit >> (3 * i)) & 0x7
        row = i // 0-3 for row0, 4-7 for row8 (interleaved layout)
        Wt[i] = C[row][qi] // centroid lookup
    
    return Wt
```

```
// ===== 数据流timeline =====
// Inter-tile (K-tile granularity):
//   Smem 0: [cp.async tile 0] [mma/dequant tile 0 subtiles] 
//   Smem 1:                    [cp.async tile 1]              [mma/dequant tile 1 subtiles]
//
// Intra-tile (subtile granularity):
//   Reg 0: [load+dequant s0]    [mma s0]   [load+dequant s2]  [mma s2] ...
//   Reg 1:          [load+dequant s1] [mma s1]          [load+dequant s3] ...
//
// Tensor Cores:  [idle] [mma s0] [mma s1] [mma s2] [mma s3] ...
// CUDA Cores:    [dequant s0] [dequant s1] [dequant s2] ...
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Quantix fused kernel以CUDA/C++实现，约数千行代码。关键实现技术：
1. **Memory access**: 128-bit cp.async (UINT4 reinterpret) 实现vectorized global→shared传输；ld.shared实现shared→register加载；ldmatrix实现activation register准备（Tensor Core operand format）
2. **Tensor Core调用**: 通过PTX内联汇编调用mma.m16n8k16等指令，支持FP16输入/输出
3. **Double buffering**: 两个shared memory buffer + 两个register buffer实现三级流水线
4. **Split-K**: 沿K维度切分，多个thread block group并行计算partial sums，最后lightweight reduction kernel归并
5. **集成**: Kernel集成进HuggingFace Transformers替换SqueezeLLM默认backend，uniform baselines (GPTQ/Marlin)使用AutoGPTQ library
6. **Accuracy**: Kernel不改变量化模型精度——bit shuffling是lossless，dequantization是bit-exact重建

开源：https://github.com/yuang-chen/Quantix-PPoPP26

涉及论文标题：
- High-Throughput Non-Uniformly Quantized 3-bit LLM Inference

术语是什么？通过联网搜索让回答具体和精准。

INT2-to-FP16 Efficient Unpacking是JanusQuant在mixed-precision attention kernel中使用的一种低开销2-bit整数到FP16浮点数的转换方法。其核心insight来自FP16格式特性：在[1024, 2048)数值区间内，所有FP16值共享相同的exponent（2^10=1024），而mantissa的10位直接编码0-1023的整数偏移量。因此，对于[0, 1023]范围内的整数，可以通过直接将该整数放入FP16 mantissa位并设置exponent=1024来"合成"FP16表示，再减去1024得到最终值。这种方法避免了通用INT→FP转换指令的高延迟，用3条指令（lop3 bitwise extract, or设置exponent, sub减偏移）处理两个2-bit值。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

INT2-to-FP16 unpacking在attention kernel中的具体实现（PTX级别）：

```
// === PTX Register-Level INT2→FP16 Unpacking ===
// 输入:
//   R1 (32-bit): 打包的16个2-bit values（共32 bits）
//   R2 (32-bit): 工作寄存器
// 输出:
//   每轮处理2个值 → 2个FP16 values packed in R2

// Step 1: 用lop3提取第i个2-bit值到R2
// lop3是NVIDIA GPU的三输入bitwise逻辑指令，可单指令完成复杂bitwise提取
lop3.b32 R2, R1, 0x3, RZ, extract_pattern;  
// R2 = (R1 >> (i*2)) & 0x3  → 提取的2-bit值在R2低位

// Step 2: 移位到mantissa位置 + OR设置exponent
shl.b32 R2, R2, 10;           // 将2-bit值左移10位到mantissa位置(bit 10-19)
or.b32  R2, R2, 0x64006400;   // 设置两个FP16值的exponent字段
// 0x6400 = 0110 0100 0000 0000
//   bit 15: sign = 0
//   bits 14-10: exponent = 01100 = 24 (bias 15 → actual exponent = 24-15 = 9, 
//                wait - 论文说exponent = 1024. 让我修正:
//   实际上: exponent_field = 1024 (actual exponent) → with bias 15:
//   stored_exponent = 1024 + 15 = 1039 = 0b10000001111 → 
//   but in FP16, stored exponent is 5 bits for values >= 1.0
//   
//   更准确的解释: FP16对于整数1024+n的表示：
//   1024 = 2^10 → stored exponent = 10+15 = 25 = 11001
//   mantissa = 0 (since 1024 is exact power of 2)
//   1024+n 其中n∈[0,1023]: stored exponent = 10+15 = 25,
//   mantissa = n (n的10-bit binary)
//   所以 0x6400 = 0110 0100 0000 0000
//     bit 15: 0 (sign)
//     bits 14-10: 11001 = 25 (exponent for 2^10)
//     bits 9-0: 0000000000 (mantissa = 0 → 代表1024)

// 处理两个值: 低位FP16和高位FP16
// R2低16位: val_low in mantissa bits 9-0, exponent = 25 → FP16(1024 + val_low)
// R2高16位: val_high in mantissa bits 9-0, exponent = 25 → FP16(1024 + val_high)
// 所以0x64006400将两个16-bit half的exponent都设为25

// Step 3: 减去1024得到最终值 (sub指令)
// 1024.0 in FP16 = 0x6400
sub.f16 R2_low, R2_low, 1024.0;   // FP16(1024+val_low) - 1024 = FP16(val_low)
sub.f16 R2_high, R2_high, 1024.0; // FP16(1024+val_high) - 1024 = FP16(val_high)

// 对比naive实现：__int2float_rn()需要完整INT→FP转换流水线
// 至少4条指令/值（包括range check, leading zero count, shift, round）
// JanusQuant方法: 3条指令处理2个值 (lop3 + or + sub×2 = 平均1.5指令/值)
```

此方法的关键约束：① 仅适用于[0,1023]范围的整数（2-bit值永远在[0,3]内，天然满足）；② 依赖FP16格式中exponent=1024区间mantissa直接编码整数的特性；③ 需要R1和R2均为32-bit register，一次处理16个2-bit值。论文称这是"inspired by prior work on 8-bit and 4-bit dequantization [17]"的适配。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在JanusQuant attention kernel中以PTX inline assembly实现。CUDA kernel中每thread处理一段KV cache segment，从shared memory加载INT2 packed data到register（R1），循环调用lop3→shl→or→sub序列解包每个2-bit值。该实现需要CUDA compute capability ≥ 7.0（支持lop3指令）。在A100上验证：unpacking优化相比naive INT→FP转换使attention kernel平均加速1.99×（Figure 15b），且与parameter block layout combined后达到3.05× speedup。此技术可推广到其他低位量化（3-bit/4-bit等）的dequantization kernel，但需调整bit提取逻辑和mantissa移位偏移量。

涉及论文标题：
- JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference

