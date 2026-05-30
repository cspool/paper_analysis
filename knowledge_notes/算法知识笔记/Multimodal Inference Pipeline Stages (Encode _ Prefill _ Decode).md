## Multimodal Inference Pipeline Stages (Encode / Prefill / Decode)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Multimodal Inference Pipeline 是 EPD-Serve 定义的 MLLM 推理三阶段划分：(1) **Encode 阶段**：多模态编码器 E（Vision Transformer / Audio Encoder 等）将原始多模态输入 I_m 转换为高维特征向量序列 V_m ∈ R^{n×d}，作为 Prefill 阶段的输入特征；(2) **Prefill 阶段**：文本提示 I_t 编码为 V_t，拼接多模态特征 V_m + V_t 输入 LLM Decoder，执行首次前向传播生成首 token O_1 并构建全层 KVCache KV1；(3) **Decode 阶段**：基于 KVCache 和上一 token，LLM 自回归迭代生成后续 token O_i+1，直至 <eos> 或 max_length。三阶段具有显著的计算异质性：Encode 为 compute-heavy（ViT 参数 0.7-6B）、Prefill 为 memory+compute 混合（KV Cache 构建）、Decode 为 memory-bound（逐 token GEMV）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

EPD-Serve MLLM 推理 pipeline 的数学形式：

```
Encode 阶段:
  V_m = E(I_m)                    // I_m: 图像/音频/视频
  V_m ∈ R^{n×d}                   // n: visual tokens, d: feature dim
                                  // 例 openPangu-7B-VL: ViT 0.7B

Prefill 阶段:
  V_t = TokenEmbed(I_t)           // I_t: 文本 prompt
  O_1, KV_1 = LLM(V_m, V_t)      // 首 token + 全层 KVCache
                                  // attention: softmax(QK^T/√d_k)

Decode 阶段 (自回归循环):
  for i = 1 to max_length:
    O_{i+1}, KV_{i+1} = LLM(O_i, KV_i)  // 基于历史 KVCache
    if O_{i+1} == <eos>: break
```

三阶段计算特征对比（openPangu-7B-VL）：

| 阶段 | 模块 | 参数量 | 计算特征 | 瓶颈 |
|------|------|--------|----------|------|
| Encode | ViT | 0.7B | Compute-heavy (ViT forward) | AI Core utilization |
| Prefill | LLM(7B) | 7B | Memory+Compute (KVCache build) | Seq length quadratic |
| Decode | LLM(7B) | 7B | Memory-bound (GEMV/token) | HBM bandwidth |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

三阶段划分是实现 EPD Disaggregation 的基础——每个阶段被映射为独立可调度的实例进程。实例间通过：(1) E-P 异步特征预取（仅传 hash，从 MM Store 检索特征向量）；(2) P-D 分层分组 KV Cache 传输（按 Transformer 层打包，延迟调度对齐通信与计算）。三阶段的并行策略可按需独立配置：Encode 偏好数据/序列并行、Prefill 可根据序列长度选择 pipelining、Decode 偏好张量并行降低延迟。EPD-Serve 在论文中使用的模型为 openPangu-7B-VL (ViT 0.7B + LLM 7B) 和 Qwen3-VL-8B (ViT 0.6B + LLM 8B)，表明 pipeline 阶段划分适用于典型 MLLM 架构。Encode 阶段因 Attention 复杂度随序列长度平方增长，在某些场景下编码延迟可超过 LLM Prefill 时间，是该阶段的根本性能瓶颈。

涉及论文标题：
- EPD-Serve A Flexible Multimodal EPD Disaggregation Inference Serving System On Ascend
