## 四步 FFT（Four-Step FFT）与 Tensor Core 递归分解

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 四步 FFT 是 Cooley–Tukey 的 cache 友好变体（Bailey 4-step）：把 N=n1×n2 的一维 DFT 重排为 n1×n2 的二维数据布局，分四步执行——(1) n2 个 n1 点列向 FFT；(2) 逐元素乘 twiddle factor $W_N^{j_2k_1}$；(3) 矩阵转置；(4) n1 个 n2 点行向 FFT。由索引映射 $j=j_1n_2+j_2$、$k=k_1+k_2n_1$ 可得 $X[k_1+k_2n_1]=\sum_{j_2}\big[\big(\sum_{j_1}x[j_1n_2+j_2]W_{n_1}^{j_1k_1}\big)W_N^{j_2k_1}\big]W_{n_2}^{j_2k_2}$。关键性质：列向与行向子变换都是批处理 DFT，可表达为矩阵-矩阵乘——天然映射到 GPU Tensor Core；且分解递归（每个子 DFT 可再用四步法细分），形成分层 radix 分解。在 MNEMOS 中，"radix"指四步分解的基数（如 radix-8 即基为 8 的分解）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- MNEMOS 的分层策略：N>64 优先 radix-64 一步；8<N≤64 回退 radix-8；<8 点用 CUDA Core + warp shuffle。基例为 8 点 FFT（匹配 FP64 WMMA 8×8×4：n 点 DFT 是 n×n 矩阵-向量积，批量即矩阵乘）。以 N=512（TFHE 中 512 点？实际 Tangent FFT 用 N/2=256 点）说明 256 点四步 FFT：
```
# 256 = 32×8：列向 32 个 8 点 FFT + 转置 + 行向 8 个 32 点 FFT
# 32 点再拆 = 4×8：列向 4 个 8 点 FFT + 转置 + 行向 8 个 4 点 FFT（CUDA Core）
# 每个 8 点 FFT = 8×8 矩阵 × 8×1 向量 → WMMA 8×8×4（FP64，2 次 MMA/复数分解后 4 个实乘×2）
A = reshape(x, (32, 8))                  # 列优先
Y = FFT8_batched(A, axis=0)              # 32 个 8 点列向 FFT（WMMA）
Y *= twiddle[32, 8]                       # 逐元素 twiddle
Y = transpose(Y)                          # 共享内存转置（swizzle 防 bank 冲突）
Z = FFT32_batched(Y, axis=0)              # 行向 32 点 FFT（再递归）
out = reshape(Z, (256,))
```
- Annotations：64 点特化算法（图 8）把两级 8×8 分解融合为一级，利用 WMMA fragment 在 warp 内的布局隐式完成转置，省去一次共享内存往返与同步；Fourier 矩阵不预存共享内存而在寄存器内生成（8 点 DFT 元素来自 {0,±1,±sin(π/4)}），消除共享内存载入延迟、bank 冲突与布局管理。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：cuFFT 等库内部即用类四步分解；MNEMOS 自研 CUDA kernel 以 WMMA（FP64 mma.sync.m8n8k4）实现列/行向批量子变换，转置在共享内存完成并用宽数据类型 swizzle 消除 bank 冲突。使用场景：TFHE 的 Tangent FFT（N/2 点复数 FFT）与 NTT 类变换（HyperDrive 的 radix-64 Inner-NTT 同思路）；凡"小变换批量、数据规整"的变换都能用四步法 + Tensor Core 加速。注意点：转置步骤是共享内存带宽与延迟的关键（bank 冲突 + 中间往返），MNEMOS 用"fragment 布局隐式转置 + swizzle"两招消除。

涉及论文标题：
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization
