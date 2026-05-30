## LLM-Viewer

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

LLM-Viewer 是一个 LLM 推理性能分析工具，由 Zhihang Yuan 等人在 2024 年的论文 *LLM Inference Unveiled: Survey and Roofline Model Insights* (arXiv:2402.16363) 中提出。它基于 **roofline model** 方法，根据硬件规格（计算吞吐量、内存带宽）和模型配置（层数、hidden dimension、head 数、序列长度等）来模拟/预估 transformer 模型在不同硬件上的最优推理性能（预填充时延和解码时延），而无需实际部署运行。

其核心原理是：将 transformer 推理分解为一系列线性代数运算（矩阵乘法、attention、softmax、LayerNorm 等），对每个操作判断其是 compute-bound 还是 memory-bound（通过 operational intensity vs roofline 比值），然后计算在理想条件下的执行时间。对 attention 算子，分别计算 QK^T 的 compute-bound 部分和 softmax/attention output 的 memory-bound 部分。

LLM-Viewer 的输入包括：(a) 硬件规格：GPU 的峰值 FLOPS（如 H100 FP16: 989 TFLOPS）、内存带宽（如 H100 HBM: 3.35 TB/s）、SRAM 大小；(b) 模型配置：层数、hidden dim、FFN dim、head 数、head dim、vocab size；(c) 推理配置：序列长度、batch size、precision。

从系统架构角度拆解，给出 LLM-Viewer 运转流程的具体例子。

**LLM-Viewer 分析流程（以评估 GTA-1B vs GQA-1B 预填充时延为例）：**

```
输入:
  - Hardware: NVIDIA H100 80GB (989 TFLOPS FP16, 3.35 TB/s HBM bandwidth)
  - GTA-1B: 54 layers, H=1280, n_h=20, n_q=5, n_k=1, n_c=1, d_h=64, d_l=128
  - GQA-1B: 54 layers, H=1280, n_h=20, n_k=5, d_h=64
  - Seq len N=2048, batch=1

Step 1: 每层 prefill 操作分解
  GTA-1B per layer:
    - QKV Projection: Q=XW_Q (N×H × H×320), K=XW_K (N×H × H×64), C=XW_C (N×H × H×128)
    - Attention: S_g = Q_g @ K^T (5 groups × N×d_h × d_h×N),
      A_g = softmax(S_g) (5 × N×N),
      O_latent = A_g @ C (5 × N×N × N×d_l),
      Gate: sigmoid(x_t @ W_G) (N×H × H×64 × 20 heads)
    - Output project: O_i @ W_O (20 × N×64 × 64×H)

Step 2: 判断每个操作的计算强度（Operational Intensity = FLOPs / Bytes）
    - Matmul Q=XW_Q: O(N×H×320) FLOPs / O(H×320 + N×H) Bytes → Compute-bound (>roofline)
    - Attention QK^T: O(n_q × N×d_h×N) FLOPs / O(n_q×N×d_h + n_k×N×d_h) Bytes → typically Compute-bound
    - Softmax: O(n_q × N×N) FLOPs / O(n_q×N×N) Bytes → Memory-bound (<roofline)
    - Attention Output A_g@C: O(n_q × N×N×d_l) FLOPs / O(n_q×N×N + n_c×N×d_l) Bytes → Compute-bound

Step 3: 为 compute-bound ops 分配 FCC (FLOPS ceiling) 执行时间
      为 memory-bound ops 分配 bandwidth ceiling 执行时间
      Total time = sum(max(FLOPs/peak_FLOPS, Bytes/peak_BW) × utilization_factor)

Step 4: Output
    GTA-1B prefill @ 2048: ~50ms
    GQA-1B prefill @ 2048: ~100ms
    → GTA-1B 约 2× prefill 加速
```

术语一般如何实现？如何使用？

LLM-Viewer 开源工具的使用方式：

1. **配置输入**：在 YAML/JSON 中定义硬件规格和模型配置。硬件规格可从厂商 spec sheet 获取（NVIDIA whitepaper、Apple M2 文档等）；模型配置从模型 card 或训练配置获取。

2. **运行模拟**：工具自动将模型分解为逐层逐操作序列，应用 roofline model 计算每操作的 ideal time。支持 prefill 和 decode 两阶段的分离分析。

3. **输出**：预填充时延 vs 序列长度曲线、解码时延 vs 缓存长度曲线、各操作的 bound 类型标注。

4. **局限性**：(a) 仅给出理想条件下的性能上界（assumes perfect kernel implementation, no overhead from framework/Python）；(b) 不考虑 kernel launch overhead、PCIe 传输等系统级开销；(c) 不能替代实际推理测量。GTA 论文同时使用 LLM-Viewer 和 transformers 实际推理来提供互补证据。

5. **在 GTA 论文中的使用场景**：补充实际 transformers 推理结果，在更多硬件（A100 40GB/80GB、H100 PCIe）上预估 GTA 和 GQA 的性能差异，验证 roofline model 预测的效率增益与实际测量趋势一致。

涉及论文标题：
- GTA__Grouped-head_latenT_Attention
