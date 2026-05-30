## Block-wise Absmax Quantization（逐块绝对值最大量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block-wise Absmax Quantization 是一种数据无关（data-free）的权重量化方法，由 Dettmers et al. (2022) 在 8-bit Optimizers 中首次引入，后被 QLoRA（NF4）采用。其核心流程：(1) 将网络权重 W 按固定 block size I（如 64 或 128）分组；(2) 对每 block b 计算绝对值最大值 `w_b^max = max_i |w_{b,i}|` 作为归一化常数；(3) 每个权重除以该 block 的 absmax，归一化到 [-1, 1]；(4) 对归一化权重用固定的 scalar quantizer（如 NF4, AF4, BOF4）量化到 4-bit；(5) 存储 4-bit 索引 × |W| + 每 block 一个 FP16/BF16 量化常数。解码时：`Ŵ_{b,i} = w_b^max * x̂(index)`。该方法不需要校准数据、不需要计算激活值，量化过程极快且内存开销低，但精度略低于依赖校准数据的 PTQ 方法（如 GPTQ, AWQ）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Block-wise Absmax Quantization Pipeline
W = linear_layer.weight                # [d_out, d_in]
W_flat = W.reshape(-1)                  # flatten
I = 64                                  # block size
B = len(W_flat) // I                    # number of blocks

# Step 1: Block partitioning + absmax computation
for b in 1..B:
    block = W_flat[(b-1)*I : b*I]
    w_max[b] = max(abs(block))           # quantization constant (store in BF16)

# Step 2: Normalization
for b in 1..B:
    for i in 1..I:
        x[b,i] = W_flat[(b-1)*I + i] / w_max[b]   # normalized to [-1, 1]

# Step 3: Scalar quantization (e.g., BOF4-S MSE codebook)
codebook = [-0.8568, -0.6693, ..., 0.0, ..., 1.0]  # 16 levels
for b in 1..B:
    for i in 1..I:
        idx = argmin_ℓ |x[b,i] - codebook[ℓ]|
        W_quant_idx[(b-1)*I + i] = idx

# Step 4: Decoding (inference)
for b in 1..B:
    for i in 1..I:
        W_hat[(b-1)*I + i] = w_max[b] * codebook[W_quant_idx[(b-1)*I + i]]
```
关键设计决策：block size I 越小→量化常数越多→精度越高（因 outlier 影响局限在小范围）但内存开销越大。典型 I=64 时，量化常数开销约为 16/64=0.25 bit 每参数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
bitsandbytes 库实现了 block-wise absmax 量化为 NF4 格式。在 HuggingFace 中使用 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_block_size=64)`。BOF4 论文提供了开源实现：https://github.com/ifnspaml/bof4，支持 BOF4/BOF4-S 码本和 OPQ。主要应用：(1) QLoRA 微调；(2) 内存受限推理场景；(3) PTQ 方法对比 baseline。

涉及论文标题：
- Improving Block-Wise LLM Quantization by 4-bit Block-Wise Optimal Float (BOF4)

---
