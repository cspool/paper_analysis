## Mixed MoE Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Mixed MoE Quantization 是一种针对 MoE 模型的差异化量化策略：对不同组件使用不同的量化精度以在模型质量和内存占用之间取得最优权衡。核心发现是 MoE 模型的 expert 参数占总体参数的绝大多数（Mixtral-8x7B 中 expert 占 96.6%），但 attention 层对量化更敏感。因此最优策略是 attention 层保持较高精度（4-bit 或 FP16），expert 层可激进量化到 2-3 bit。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Mixtral-8x7B 的量化方案组合及 perplexity 对比（Table 1）：

| Attn quant | Expert quant | Model size | WikiText2 perplexity |
|------------|-------------|-----------|---------------------|
| FP16 | FP16 | 86.99 GB | 3.59 |
| FP16 | 4-bit | 25.82 GB | 3.67 |
| FP16 | 3-bit | 23.21 GB | 3.96 |
| FP16 | 2-bit | 19.33 GB | 4.52 |
| 4-bit | 4-bit | 23.99 GB | 3.76 |
| 4-bit | 3-bit | 21.37 GB | 4.05 |
| 4-bit | 2-bit | 17.54 GB | 4.61 |

论文选择的两种方案（绿色标注）：
- **4-bit attention + 3-bit experts**: 21.37 GB, Wiki2=4.05, MMLU=68.47%
- **4-bit attention + 2-bit experts**: 17.54 GB, Wiki2=4.61, MMLU=65.58%

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Embedding、logits head、MoE gate 和 normalization 层保持 FP16（参数少，对精度关键）
- 混合量化的内存计算：针对 12-16GB GPU + 8-16GB/s PCIe 带宽，模型必须压缩到可放入 host RAM 且加载延迟可接受
- 所有量化使用 HQQ 算法，但因策略与算法选择无关，可替换为 GPTQ/AWQ 等同效果

涉及论文标题：
- Fast Inference of Mixture-of-Experts Language Models with Offloading
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU
