## Zigzag Computation Order (Zigzag Offloading Strategy / Zigzag计算顺序)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Zigzag Computation Order 是 FlexGen [Sheng et al., ICML 2023] 提出的面向 GPU 内存受限场景的 LLM 批量推理计算顺序。核心思想是将模型逐层处理：对每个 transformer layer，先将该层所有需要的 weights 从 CPU 加载到 GPU，然后对所有微批次执行该层的 attention + FFN 计算，最后卸载中间结果并处理下一层。这种方式通过在同一层内处理更多 token（大 batch）来 amortize weight transfer 的 I/O overhead。Zigzag 与传统的逐 batch 处理（先完成一个 batch 的所有层再处理下一个 batch）不同——它交换了 batch 维度和 layer 维度的遍历顺序：外层循环是 layer，内层循环是 batch。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Zigzag 执行流程（以 FlexGen on Mixtral 8x7B 为例）：
```
for i = 1 to num_layers:           // 外层：逐层
    Load weights of layer i CPU→GPU  // 一次性加载所有 expert weights
    for j = 1 to num_micro_batches: // 内层：逐微批次
        GPU Attention(QKV proj + softmax)  // S4 scheme: KV cache H2D
        GPU MoE FFN(gate + Top-2 expert GEMM)
    Offload intermediate results GPU→CPU
```
关键特征：(1) 所有微批次共享同一轮 weights 加载（amortize I/O cost）；(2) 逐层串行，无跨层 I/O 重叠；(3) 在 S4 模式下，KV cache H2D 与 weight transfer 竞争同向 PCIe 带宽。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- FlexGen 中实现为 Python 调度循环，PyTorch 执行。用户指定 batch size N 和 micro-batch size μ。
- MoE-Lightning 采用相同的 zigzag 遍历顺序作为基础框架（外层逐层），但在每层内部用 CGOPipe 替换了简单的串行执行，实现了 GPU/CPU/I/O 的细粒度重叠。
- 局限性：(1) 无跨层 pipeline——每层 weights 传输时 GPU idle；(2) weight transfer 与 KV cache transfer 竞争 PCIe 带宽；(3) 在 FlexGen 中需 padding 所有请求到最大 prompt length。

涉及论文标题：
- MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs
