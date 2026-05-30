## Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  为 DES 的 coreset selection 开发自定义 fused GPU kernel，将原有的 12 个碎片化算子（softmax、top-k、reduction 等）融合为仅 2 个 kernel。实验比较 fused kernel vs PyTorch baseline 的 coreset selection 延迟，验证消除 kernel launch 和 HBM traffic 开销的效果。

- 后端平台是什么，配置是什么。
  NVIDIA B200 GPU，CUDA 13.1，Intel Xeon 6960P CPU。

- 评估性能的软件/脚本是什么。修改了什么。
  NVIDIA Nsight Systems 进行 kernel profiling。自定义 CUDA kernel 替换 PyTorch 的碎片化算子链。
  Kernel 设计：
  - **Primary kernel**：融合 per-token softmax + Top-K filtering + weighted expert accumulation，使用 register-level computation 和 atomic instructions 更新 global saliency scores。
  - **Second kernel**：基于 threshold-governed ranking 执行 final expert masking。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未提供开源代码。自定义 fused kernel 的原理：
  ```
  输入: Router logits I (N×M tensor), Top-K, M_core
  PyTorch baseline 流程（12 kernels）:
    1. Softmax(I) → gate weights
    2. TopK(gate weights, K) → local selections
    3. Mask 非 Top-K 权重
    4. Sum over N dim → votes
    5. TopK(votes, M_core) → coreset
    6-12. ... 后续路由和计算
  
  Fused kernel 流程（2 kernels）:
    Kernel 1 (per-token fused):
      for each token n in parallel:
        softmax_n ← softmax(I_n)                 // register-level
        topk_idx_n, topk_val_n ← topk(softmax_n, K)  // register-level
        masked_weights_n ← mask_non_topk(softmax_n, topk_idx_n)
        for each expert i in topk_idx_n:
          atomicAdd(V[i], masked_weights_n[i])   // atomic to global memory
    Kernel 2 (final ranking):
      C ← topk(V, M_core)                        // select coreset
      output_mask ← create_mask(C, M)            // generate mask
  ```
  Fused kernel 实现 **6× speedup** over PyTorch baseline，通过消除冗余 HBM traffic 和 operator dispatch overhead。
