## ZipNN（AI 模型无损压缩库）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
IBM Research（与 BU/Dartmouth/MIT/Tel Aviv 合作）开源的面向神经网络模型的无损压缩库（CLOUD 2025，arXiv:2411.05239，github.com/zipnn/zipnn）。关键发现：浮点权重的符号位与分数位看似随机，但指数高度偏斜——256 个可能指数值中仅约 12 个出现 99.9% 的时间。做法：分离指数并用 Huffman（zstd 内置）熵编码，另加 "byte grouping"（字节分组）把分数位分流找更多模式；基于 Zstd v1.5.6，约 2000 行 C + 4000 行 Python。效果：BF16 模型（Llama/Granite/Mistral）约 33% 体积缩减（比 Zstd 好 11%），部分"干净"模型超 50%；解压吞吐约 80 GB/s、压缩约 13 GB/s（多线程）；完全无损、可集成 Hugging Face Transformers（zipnn_hf()），支持 delta 压缩与检查点版本管理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# ZipNN pipeline（BF16 权重文件）
streams = split_bytes(w):  # 按字节位平面拆分：符号位流 + 指数字节流 + 尾数高位/低位流
exponent_stream → Huffman 熵编码          # 指数偏斜 → 大压缩率
fraction_streams → byte grouping 找模式后编码
拼接各流头部(元数据) → 压缩文件
```
Annotations：与 ENEC 的指数-尾数分离思路一致（同为"指数可压、尾数难压"观察），区别在编码器——ZipNN 用 Huffman/变长（CPU 友好）、ENEC 用定长+线性变换（Ascend SIMD 友好）。ENEC 论文把 ZipNN 列为 CPU 侧主要 baseline：BF16 压缩比 1.50-1.51（高于 ENEC 的 1.35-1.37，因变长编码更接近熵），但 CPU 吞吐仅 0.4 GB/s 级，ENEC 在 NPU 上压缩吞吐为其 987×（BF16）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：基于 Zstd 库的 Huffman + 字节分组，chunk 级并行；提供 pip install zipnn、zipnn_hf() 无缝接入 HuggingFace；支持 torch 权重文件与 checkpoint delta。使用：模型存储/下载/分发压缩（HuggingFace 存储后端已采用其 byte grouping 技术，报告约 20% 存储节省）、训练检查点管理；ENEC 将其作为 CPU 端压缩比/吞吐对比基线。局限：CPU 吞吐远低于 GPU/NPU 专用实现，且变长解码需要分支，不适合 Ascend 这类无分支 SIMD 加速器直接落地。

涉及论文标题：
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs
