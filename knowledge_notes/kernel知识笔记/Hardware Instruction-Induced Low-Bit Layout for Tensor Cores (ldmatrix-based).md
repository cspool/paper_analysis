## Hardware Instruction-Induced Low-Bit Layout for Tensor Cores (ldmatrix-based)

术语是什么？通过联网搜索让回答具体和精准。

Hardware Instruction-Induced Low-Bit Layout 是 BitDecoding 提出的方法，利用 ldmatrix PTX 指令的 thread-to-register 映射自动为低比特（INT4/INT2）量化数据生成 Tensor Cores 兼容的 packed layout。核心观察：ldmatrix 从 shared memory 加载数据到 register 时，按 Tensor Cores fragment 的 interleaved pattern 将数据分配到 warp 内各 thread。如果在 ldmatrix 加载 FP16 数据后立即由每个 thread 在 register 内做量化+pack，则写出到 global memory 的 packed low-bit data 天然保持 Tensor Cores 期望的 interleaved layout——后续用相同 ldmatrix 配置 unpack 时，值已处于正确位置可直接参与 mma。这消除了 Ladder/Marlin 等方法所需的重weight offline layout transformation kernel（prefill 58.02ms→0.06ms, decode 0.41ms→0.008ms overhead）。

从 kernel 调度角度拆解术语：

```
// === Hardware Instruction-Induced Layout (Offline zero-cost) ===
// Residual Kernel: 生成 layout-compatible packed KV cache
__global__ void residual_kernel(FP16* KV, INT16* KV_packed, ...) {
    // Step 1: ldmatrix加载FP16 KV到register（自动获得TC interleaved layout）
    ldmatrix.sync.aligned.m16n8k16.shared.b16 [...], [smem_addr];
    
    // Step 2: 执行mma计算（QK^T 或 PV）
    mma.sync.aligned.m16n8k16 [...];  // Tensor Cores
    
    // Step 3: 每thread在register内量化+pack
    // 关键：register中数据已按TC fragment layout排布
    FP16 val = reg_data[thread_local_idx];
    INT4  q   = quantize(val, scale, zero);  // FP16→INT4
    pack_bits(local_packed, q, bit_offset);   // 打包到INT16
    
    // Step 4: 写出packed data到global memory
    // layout已隐式正确——因步骤1的ldmatrix决定了pack后的排列
    KV_packed[global_idx] = local_packed;
}

// Packing Kernel: 消费时layout自动正确
__global__ void packing_kernel(INT16* KV_packed, FP16* Q, ...) {
    // 用与Residual Kernel相同的ldmatrix变体加载
    // 自动恢复正确的TC fragment layout
    ldmatrix.sync.aligned.m16n8k16.shared.b16 [...], [KV_packed_smem];
    
    // dequant后值已在正确register位置 → 可直接mma
    mma.sync.aligned.m16n8k16 [...];  // Tensor Cores，无layout mismatch
}
```

关键条件：(i) Packing Kernel 必须 mirror Residual Kernel 的 ldmatrix variant 和 mma variant；(ii) warp-tiling 配置必须一致；(iii) residual block size Nr=Pn×Wn×R 确保每个 TC fragment 被完整填充。

术语一般如何实现？如何使用？

实现通过统一 instruction configuration 协调 Residual 和 Packing kernel：根据 GPU 架构确定 ldmatrix 和 mma variant → 根据 bit-width（β=4或2）计算 packing ratio R=ω/β → 根据 Wn 和 Pn 计算 Nr。对于不同 GPU 世代（Ampere/Hopper/Blackwell），ldmatrix/mma variant 不同但原理通用。开源于 https://github.com/OpenBitSys/BitDecoding。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

