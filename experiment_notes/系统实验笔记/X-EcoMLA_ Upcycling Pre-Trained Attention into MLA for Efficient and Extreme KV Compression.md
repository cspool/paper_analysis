## X-EcoMLA: Upcycling Pre-Trained Attention into MLA for Efficient and Extreme KV Compression

- 属于Serving调度的实现是什么？实验比较什么？
  评估 X-EcoMLA 转换后的 MLA 模型在推理部署时的系统级吞吐和内存表现。实现是将预训练 Llama 模型通过 SVD 初始化 + 知识蒸馏 + DPO upcycle 为 MLA 版本后，在 AMD MI300 GPU 上测量推理性能。论文未修改开源 Serving 框架的调度逻辑，而是在标准推理框架上比较 baseline（MHA/GQA）与 X-EcoMLA (MLA) 在不同 batch size 下的吞吐（sequences/sec）和峰值 GPU 显存（GB）。

  实验比较（Figure 2）：(1) **吞吐对比**：Llama3.1-8B baseline vs X-EcoMLA-8B（r_kv=128, 10.67× KV 压缩），在 8× AMD MI300 上 batch size 从 1 到 1024，X-EcoMLA 实现 1.7× 到 2× 的吞吐提升；(2) **峰值显存对比**：batch size=128 时 Llama3.1-8B 消耗 143 GB 显存且无法运行更大 batch，X-EcoMLA-8B 仅需 28 GB（5× 内存减少），可平滑扩展到 batch size 1024 而不 OOM。

- 硬件平台是什么，配置是什么。
  推理评估硬件：8× AMD MI300 GPU（系统级吞吐/内存测试），single AMD MI300（吞吐测试）。训练硬件：8× AMD MI300 GPU（训练耗时约 70-140 GPU hours）。

- 开源Serving框架是什么。修改了什么。
  论文未明确说明使用哪个特定 Serving 框架进行推理评估。系统级推理性能测试可能基于 HuggingFace Transformers / PyTorch 原生推理或 AMD ROCm 生态下的推理后端。论文未对 Serving 框架进行调度逻辑修改——其 Serving 层面的收益完全来自 MLA 架构对 KV cache 内存的压缩，KV cache 从 2·n_h·d_h·l 降至 (r_kv + d_r)·l，从而释放了大量 GPU 显存，使得在相同硬件上可以支持更大的 batch size 和更高的吞吐。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源：https://github.com/AMD-AGI/AMD-Hybrid-Models

  Serving 全流程（以 X-EcoMLA-8B on AMD MI300 推理为例）：
  ```
  1. 输入: 客户端提交 batch_size=128 的推理请求序列
  2. 模型加载: HuggingFace Transformers 加载 X-EcoMLA-8B 权重
     - 模型已通过 SVD 初始化 + 蒸馏 + DPO 转换为 MLA 架构
     - W_UK 已吸收进 W_Q, W_UV 已吸收进 W_O（推理时无显式 up-projection）
  3. Prefill 阶段 (prompt 处理):
     a. 对输入 prompt tokens 逐层计算 MLA:
        C_KV = H @ W_DKV (down-proj to r_kv=128)
        K_C, V_C = C_KV @ W_UK, C_KV @ W_UV (up-proj，可吸收)
        K_R = RoPE(H @ W_KR)  (共享 RoPE key, d_r=32 dims)
     b. KV Cache 写入: 仅存储 C_KV[r_kv=128] + K_R[d_r=32] = 160 dims/token
        vs baseline 存储 2·n_h·d_h = 2·32·128 = 8192 dims/token
        压缩比 = 8192/160 = 51.2× per-token KV（实际约 10.67× 总压缩）
  4. Decode 阶段 (逐 token 生成):
     a. 新 token 的 hidden state H_new → 计算 C_KV_new, K_R_new
     b. 追加到 KV cache（每个新 token 仅 160 dims vs baseline 8192 dims）
     c. 从 cache 读取全部历史 KV、重建 K_C, V_C、计算 attention
     d. 因 KV cache 大幅缩小，batch_size=128 时 peak memory 仅 28 GB (vs 143 GB)
  5. 输出: 生成文本返回客户端
  ```
  核心收益：MLA 通过低秩压缩将 KV cache 内存需求从 O(2·n_h·d_h·l) 降至 O((r_kv + d_r)·l)。在 batch_size=128、Llama3.1-8B 场景下，baseline 因 143 GB 显存耗尽无法运行更大 batch，而 X-EcoMLA 在 28 GB 下可扩展到 batch_size=1024，达成 1.7-2× 吞吐提升。
