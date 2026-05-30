## Fast Hadamard Transform (快速哈达玛变换 Kernel)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fast Hadamard Transform 是 Walsh-Hadamard Transform (WHT) 的高效 GPU kernel 实现。由于 WHT 矩阵元素仅为 ±1，其计算不需要浮点乘法，仅通过加法和减法即可完成。对于维度 N=2^n 的 WHT，直接矩阵-向量乘需要 O(N²) 次运算，而 Fast Hadamard Transform 利用 WHT 的递归结构（H_N = H_2 ⊗ H_{2^{n-1}}），通过类似 FFT 的蝶形运算（butterfly operations）将复杂度降至 O(N log N)。具体地，每层递归执行 N/2 对元素的加法和减法（a+b 和 a-b），共 log₂N 层，总计 N log₂N 次加减操作。Dao-AILab (2024) 提供了 CUDA 实现的 fast-hadamard-transform，通过 fused kernel 避免显式 WHT 矩阵构造，将计算融合为单次 GPU kernel launch。QWHA 论文利用该 kernel 实现适配器中的 H^{-1} X 计算：训练时用于前向传播的 ΔW=FH^{-1} 计算，推理时直接对激活 X 做 WHT 再与稀疏 F 做稀疏矩阵乘法。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Fast Hadamard Transform 的递归 butterfly kernel 执行流程（GPU kernel 伪代码）：

```
// Fast Hadamard Transform Kernel (in-place, N=2^n)
// Input:  x[0..N-1]  (向量或矩阵的每一行/列)
// Output: y[0..N-1] = H_N @ x  (H_N: WHT 矩阵)

__global__ void fast_hadamard_kernel(float* x, int N, int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Butterfly stages: log2(N) iterations
    for (int step = 1; step < N; step <<= 1) {
        // step = 1, 2, 4, 8, ... N/2
        int paired = tid ^ step;  // XOR for butterfly partner
        if (tid < paired) {
            float a = x[tid];
            float b = x[paired];
            x[tid]   = a + b;    // sum
            x[paired] = a - b;   // difference
        }
        __syncthreads();
    }
    // 可选: 归一化 x /= sqrt(N)
}

// 在 QWHA 中使用场景 —— 对激活矩阵 X 的每一行做 WHT:
// X ∈ R^{d_in × (b·s)}, H^{-1} = H^T (WHT 正交)
// for each column of X (batch×seq):
//     fast_hadamard_kernel<<<grid, block>>>(X_col, d_in, 1)

// 完整的 WHA 推理前向 (融合 WHT + 稀疏 MatMul):
// Step 1: X_transformed = fast_hadamard(X)     // O(d_in log d_in) per token
// Step 2: Y_adapt = F_sparse @ X_transformed    // O(p) per token, F 仅 p 非零元
// Step 3: Y = W_Q @ X + α * Y_adapt            // 量化权重矩阵乘法
```

**与 DCT/DHT kernel 的对比**：
- WHT: 每对 (a,b) → (a+b, a-b)，2 次加减，无乘法。递归结构直接在 GPU shared memory 中完成。
- DCT/DHT: 正弦/余弦函数计算（cos, sin, cas），涉及浮点乘法，无简单 butterfly 模式 → 需显式矩阵乘法或较慢的递归 FFT kernel → 训练时间为 WHT 的 3-10x。
- 单变换 vs 双变换：QWHA 的 WHA 仅使用 1D WHT (对 d_in 维度)，而 LoCA/SSH 使用 2D DCT/DHT (同时对 d_in 和 d_out 维度)。1D WHT 训练时间 batch=4 为 6.0h，2D WHT 为 8.0h，1D DCT/DHT 为 17.4h，2D DCT/DHT 为 26.1h。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Fast Hadamard Transform 的工程实现要点（基于 Dao-AILab 的开源实现和 QWHA 论文）：
1. **无矩阵构造**：H_N 不需要显式存储为 N×N 矩阵。预计算的 H_N 仅用于需要矩阵形式的场景（如 H^{-1}R 的预计算），跨同维度层共享缓存。
2. **Fused Kernel**：将 WHT 的 log₂N 层 butterfly 融合为单个 CUDA kernel，消除中间结果的 global memory 往返。每层使用 shared memory 交换数据，仅需 __syncthreads() 同步。
3. **非 2 的幂维度**：对于 d_in 不是 2 的幂的情况，使用 H_N = H_{2^n} ⊗ H_m 分解（H_m 为已知 Hadamard 矩阵），或通过 padding 到最近的 2 的幂次。
4. **推理效率**：QWHA 中 WHA 的推理吞吐为 184.6 tok/s，仅比 LoRA (188.1 tok/s) 低 1.9%，远优于 DCA/DHA (92.4 tok/s, 下降 50.9%)。这是因为 WHT kernel 的计算开销几乎可忽略（仅加减法），而 DCT/DHT 每次变换都需要三角函数和浮点乘法。
5. **显存开销**：fast Hadamard kernel 不产生额外显存分配（in-place 操作），推理峰值显存 QWHA 52.68GB vs CLoQ 59.53GB（减少 13.0%），因稀疏适配器的 scatter ops 无额外内存。

涉及论文标题：
- QWHA: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning
- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks
- RoSTE: An Efficient Quantization-Aware Supervised Fine-Tuning Approach for Large Language Models

在 RoSTE 中，fast Hadamard CUDA kernel 用于实现在线旋转矩阵 R_3, R_3^T, R_4 的矩阵乘法（in-block online rotations），作用于 Query/Key projection（消除 KV cache outlier）和 Down projection（MLP 内）。这些在线旋转在训练和推理时均需执行，但论文指出其开销可忽略——RoSTE 训练时间 2.8h vs 无旋转 STE 2.4h（+16.7%），主要因 Hadamard kernel 的 O(d log d) 复杂度远低于主要线性层的 O(d²) 计算。RoSTE 的 fast Hadamard kernel 继承自 QuaRot/QuIP# 的开源实现，不涉及自定义 kernel 修改。
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

Quamba2 中的 Fast Hadamard Transform 用于：(1) offline Hadamard matrix fusion——将 Hadamard 矩阵 offline 融合到 input/output projection 权重（$W_{in}^H = W_{in} H_n^T$, $W_{out}^H = H_n W_{out} H_n^T$），融合后的权重与量化后的激活仍保持 compute-invariance；(2) online FWHT kernel 内联 scaling factor $s_y$，执行 $\bar{y}^H = (1/s_y) H_n \bar{y}$ 避免额外量化步骤的延迟开销。

---
