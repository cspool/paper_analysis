## WMMA（Warp Matrix Multiply-Accumulate）与 FP64 Tensor Core 上的 FFT 映射

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- WMMA（Warp-level Matrix Multiply-Accumulate）是 CUDA 中暴露 Tensor Core 的 warp 级 API：整个 warp（32 线程）协作执行 $D_{M\times N}=A_{M\times K}\times B_{K\times N}+C_{M\times N}$ 的小块矩阵乘加，tile 形状由精度决定——半精度 16×16×16、双精度仅 8×8×4（A100 原生 mma.sync.aligned.m8n8k4.f64）。Tensor Core 不原生支持复数算术，需把复数矩阵乘分解为实数运算：$\mathbf{AB}=(\mathbf{A}_r\mathbf{B}_r-\mathbf{A}_i\mathbf{B}_i)+i(\mathbf{A}_r\mathbf{B}_i+\mathbf{A}_i\mathbf{B}_r)$，四个实数乘各映射到 2 次 WMMA（数学恒等、无额外开销）。这是"把 FFT/NTT 变成矩阵乘"类工作（TensorFHE/WarpDrive/Neo 用 INT8 TCU 做 NTT、tcFFT 做 FP16 FFT）在 TFHE 场景的关键差异点：TFHE FFT 需要 FP64 精度（≥30 小数位），故 MNEMOS 直接使用 A100/H100 的 FP64 Tensor Core。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- MNEMOS 中一次 8 点 FFT 的 WMMA 计算（图 8）：8 点 DFT 矩阵 F 与 8×batch 数据矩阵 X 相乘（批量向量），复数场景下分解为 4 个实数矩阵乘：
```
# A = F (8×8 复数 DFT 矩阵，元素 ∈ {0, ±1, ±sin(pi/4)}，运行时生成)
# B = X (8×batch 复数数据矩阵，按实/虚部分开)
for each 实数矩阵乘 (如 Ar@Br):
    mma.sync.aligned.m8n8k4.f64(D_frag, A_frag, B_frag, C_frag)  # 8×8×4 FP64 WMMA
# 4 个实乘 × 每乘 2 次 WMMA（K=4 需 2 步覆盖 8 列）→ 共 8 次 m8n8k4
# 结果按 (ArBr−AiBi) 与 (ArBi+AiBr) 合成复数输出
```
- Annotations：WMMA fragment 在 warp 内按 lane 分布（每线程 Fragment A/B 各 1 元素、Fragment C 2 元素、结果 D 均分）——MNEMOS 利用该寄存器布局省去 64 点 FFT 的一次显式转置；FP64 Tensor Core 吞吐：A100 19.5 TFLOPS、H100 67 TFLOPS，数据中心 GPU FP64 为 FP32 的 1/2（消费级仅 1/64），故该映射对数据中心 GPU 有效。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：`nvcuda::wmma` 命名空间 API（load_matrix_sync/store_matrix_sync/mma_sync）或内联 PTX mma.sync；MNEMOS 以 CUDA C++ 编写、针对 A100 并对 H100 兼容（sm_90 起支持更大 m16n8k4 等）。使用要点：(1) 复数乘必须先分解为实数 MMA（无复数指令）；(2) fragment 布局是转置/数据复用优化的杠杆（寄存器内数据再排布免共享内存）；(3) FP64 WMMA 仅数据中心旗舰（A100/H100 及后续）可用，消费级 GPU FP64 吞吐被阉割。对照：CKKS 的 INT8/FP16 TCU NTT 映射（TensorFHE/WarpDrive）因精度不足不能直接用于 TFHE FFT。

涉及论文标题：
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization
