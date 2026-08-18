## cuStateVec（NVIDIA GPU 量子态矢量模拟 kernel 库）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- cuStateVec 是 NVIDIA cuQuantum SDK（Bayraktar et al., QCE 2023）的核心组件之一：GPU 上高性能态矢量（statevector）模拟的 C/C++ 库，提供稠密/稀疏矩阵作用、张量积、测量、采样等 kernel（如 cuStateVecApplyMatrix、cuStateVecMeasure、cuStateVecBatchApplyMatrix），是 Qiskit-Aer GPU 后端与 CUDA-Q statevector 后端的底层模拟引擎。本论文用 cuStateVec v1.12.0 作为 TUSQ 的 SVS kernel 后端（backend-agnostic，用户可换任意模拟 kernel）。
- 在 TUSQ 中的角色：ECM+DFTT 决定"何时对哪个态矢量乘哪个门"，实际矩阵向量乘由 cuStateVec 在 GPU 上执行；DFTT 的 compute（乘 U）与 uncompute（乘 U†）都映射到 cuStateVec 的 apply-matrix kernel；多 GPU 时按子向量切分态、跨 GPU 门搬移振幅到单 GPU 运算。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- TUSQ 调用 cuStateVec 的 kernel 调度过程（一次 DFTT 边遍历）：
  ```
  // DFTT 树遍历逻辑（CPU）驱动 GPU kernel（伪代码）
  for edge in dfs_order(tree):
      if 正向: cuStateVecApplyMatrix(handle, gate_matrix, state, ...)   // |ψ'> = U|ψ>
      else:    cuStateVecApplyMatrix(handle, gate_matrix_dagger, state, ...) // uncompute：U†
  // 叶子处：cuStateVecSample / cuStateVecMeasureBatch 按 |ψ[i]|^2 采样 s_i 次
  ```
- kernel 语义：单比特门作用 2^n 元素（每个元素独立更新，GPU 并行度 2^n）、双比特门作用 4·2^n 元素（本论文操作计数 1/4）；TUSQ 通过与 Qiskit/CUDA-Q 相同的 cuStateVec v1.12.0 保证对比公平——加速全部来自算法层（冗余消除与树遍历复用），而非 kernel 本身。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：cuQuantum SDK 经 `pip install cuquantum-python` 或 C API 集成（https://github.com/NVIDIA/cuQuantum）；TUSQ 在 NERSC Perlmutter（NVIDIA A100 40GB）上以 CUDA_VISIBLE_DEVICES=0 单 GPU 运行，后端 kernel 为 cuStateVec v1.12.0，30-qubit Adder ×10^6 shots 约 820s（baseline >10 小时）。对比：Qiskit 2.1.0 与 CUDA-Q 0.11.0 的 baseline 后端同样是 cuStateVec v1.12.0，TUSQ 平均 59.06×/13.38× 加速归因于算法。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation
