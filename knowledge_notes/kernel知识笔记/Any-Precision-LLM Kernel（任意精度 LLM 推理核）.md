## Any-Precision-LLM Kernel（任意精度 LLM 推理核）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Any-Precision-LLM 是 SNU 开发的 GPU CUDA kernel，用于 non-uniform scalar quantized LLM 的高效推理（支持 2/3/4-bit 多精度混合部署）。核心流程：(1) Bit-transpose：将 p 个独立 1-bit packed weight planes 重组为 p-bit centroid index；(2) Table lookup：用 index 查 per-channel codebook 恢复 FP16 权重；(3) FP16 GEMM（via cuBLAS）。每个 output channel 维护独立 codebook，支持混合 bit-width 模型在单 GPU 上运行。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 输入：packed BP_0..BP_{p-1} (1-bit/plane), activation A (FP16), codebook C (2^p × FP16)
# 输出：O = A @ W_deq (FP16)

for each tile (m_tile, k_tile):
    # Step 1: Bit-transpose — 最大开销（35-58% kernel 延迟）
    idx[m,k] = Σ_{i=0}^{p-1} extract_bit(BP_i, m, k) << i
    
    # Step 2: Codebook lookup（9-17% kernel 延迟）
    w_deq[m,k] = C[idx[m,k]]
    
    # Step 3: cuBLAS GEMM
    O_tile += A_tile @ w_deq_tile
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：github.com/yeonhongpark/Any-Precision-LLM。GuidedQuant (ICML 2025) 在 end-to-end throughput benchmark 中使用此 kernel 测量 LNQ/LNQ+GuidedQuant 的非均匀标量推理吞吐（RTX 4090: 2-bit Llama-2-7B 347 tok/s）。局限性：bit-transpose 是 kernel 延迟主要瓶颈（AnyBCQ Table 7），后续工作 Quantix 通过 hardware-aligned bit shuffling 消除此开销。

涉及论文标题：
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
- AnyBCQ: Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs

---
