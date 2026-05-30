## Huffman Encoding for Model Compression (霍夫曼编码模型压缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Huffman Encoding 是一种无损数据压缩算法，Deep Compression (Han et al., ICLR 2016) 首次将其引入神经网络模型压缩。在 CNN 压缩 pipeline 中，Huffman 编码作为最后一步，对量化后的权重进行无损编码——利用量化后权重值分布不均匀的特性（某些值出现频率远高于其他值），为高频值分配短码字、低频值分配长码字，进一步减少存储空间。FQ 论文的 FC pipeline 在 FQ 量化后应用 Huffman 编码，例如 ResNet-50 (5-bit FQ) 从 5.19 MB 变为最终压缩后大小（CR=18.08×）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Huffman 编码在 FC pipeline 中的位置和流程：

```
# After FQ quantization: weights are in low-precision format
# Example: quantized weight values and their frequencies
values = [0, +4, -2, +8, -8, 0, 0, +4, 0, -2, ...]
freq = {0: 82700000, +4: 5200000, -2: 3100000, +8: 1800000, -8: 1700000, ...}

# Step 1: Build Huffman tree
build_huffman_tree(freq)  # 贪心合并最低频率节点

# Step 2: Assign codes (shorter for higher freq)
codes = {0: "0", +4: "10", -2: "110", +8: "1110", -8: "1111"}

# Step 3: Encode quantized weights
for each w_hat in quantized_weights:
    bitstream += codes[w_hat]

# Step 4: Store codebook + bitstream
# Decompression: read codebook → decode bitstream → reconstruct weights

# Effective bits per weight:
# Original FQ: 5 bits/weight
# After Huffman: Σ freq_i * len(code_i) / total_weights
# Example: (82.7M*1 + 5.2M*2 + ...) / 100M ≈ 3.2 bits/weight avg
```

**Annotations**: Huffman 编码的压缩率取决于量化后权重值的分布熵——分布越集中（如大部分值为 0），压缩率越高。FQ 论文中压缩率从 5-bit fixed 进一步压缩到有效 ~3-4 bits/weight。解码需要存储码本（codebook），对模型尺寸略有增加但通常可忽略。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Huffman 编码作为 lossless 后处理步骤，在模型部署前离线执行。推理前需先解码权重。实现可使用 Python `heapq` 构建 Huffman tree，或用 zlib/gzip 等通用压缩库。在 FQ 论文的硬件评估中，Huffman 编码对逻辑门数影响极小（275.6M → 276.4M gates, +0.3%），因为解压逻辑简单。Huffman 编码特别适合量化后分布高度偏斜的场景。

涉及论文标题：
- Focused Quantization for Sparse CNNs
