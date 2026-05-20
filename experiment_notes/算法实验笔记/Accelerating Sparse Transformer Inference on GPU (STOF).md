## Accelerating Sparse Transformer Inference on GPU (STOF)

- 属于算法pipeline的实现是什么？实验比较什么？
  提出STOF框架，包含两个核心模块：(1) Unified MHA Module：实现row-wise和block-wise两种MHA kernel，使用两级稀疏存储格式（BSR + bitmap），通过OuterTile (OT) 和 InnerTile (IT) 分别表示全局跳过block和block内元素分布，支持causal、sliding window、Longformer、Bigbird四种masking pattern；(2) Operator Fusion Module：通过hash encoding将fusion scheme量化为binary array，numerical decoding映射到Triton/TileLang compilation template，two-stage search engine（fusion expansion + parameter sampling）确定最优fusion方案和kernel参数。实验比较MHA computation speedup（4种mask×6种seq_len×3种batch size）和end-to-end inference speedup（BERT-Base/Large、GPT2、LLaMA、T5、ViT，Bigbird mask），对比PyTorch Native、PyTorch Compile、FA2、FlexAttention、ByteTransformer、Bolt、MCFuser、SPLAT。STOF相对FlexAttention在RTX 4090上平均1.8×、A100上平均1.6×加速；端到端相对PyTorch Compile平均1.3×（RTX 4090）和1.4×（A100）。

- 硬件平台是什么，配置是什么。
  NVIDIA RTX 4090 (Ada Lovelace, 24GB)、NVIDIA A100 (Ampere, 80GB)、NVIDIA H20 (Hopper，preliminary test)。Ubuntu 22.04, CUDA v12.6, PyTorch 2.7.0。Docker容器化迁移。

- 模型是什么。数据集和bench分别是什么。
  模型：BERT-Base (BERT-B)、BERT-Large (BERT-L)、GPT2、LLaMA、T5、ViT。其中BERT和ViT为encoder-only，GPT2和LLaMA为decoder-only，T5含encoder+decoder。MHA实验遵循BERT-Base配置（hidden_size=768, head_num=12）。Masking patterns：causal、sliding window (band width=32)、Longformer (global width=32, band width=32)、Bigbird (global width=32, band width=32, filling rate=10%)。Sequence length: 128-4096 (2× stride)，batch size: 1/8/16。论文未使用NLP benchmark（如GLUE/SQuAD），仅评估性能speedup。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：Zenodo artifact DOI: 10.5281/zenodo.17705801。STOF算法pipeline：
  1. 模型解析：STOF将sparse Transformer model分为MHA structure和downstream operators两部分
  2. MHA处理：(a) Kernel Selector根据analytical model（公式1，基于valid OT ratio和seq_len计算threshold）选择row-wise或block-wise kernel；(b) row-wise kernel将Q按row sliced parallel，warp内shuffle消除warp间sync，适合小seq_len+高稀疏场景；(c) block-wise kernel以OT为粒度partition Q/K/V到SMEM，跳过无效OT、仅加载需要计算的块，使用async data copying (cp.async)重叠V加载与GEMM计算，Q保持在register中复用
  3. Downstream operators处理：(a) Fusion Scheme Converter用hash encoding（convolutional subgraph analysis + neural hashing）发现频繁子图→predefined rules提取初始fusion scheme→binary hash code表达；(b) Numerical decoding将binary code映射到Triton/TileLang compilation template；(c) Two-stage search：Stage 1 fusion expansion通过expand/seize/compete三条规则用DFS逐步扩大fusion边界，性能有gain则保留否则回退；Stage 2 parameter sampling用reward-based算法分配各segment采样数
  4. 例如BERT-Base在RTX 4090上以Bigbird mask、(batch=16, seq_len=4096)推理：MHA用block-wise kernel跳过~80.8%无效计算→downstream GEMM+Layernorm/GEMM+GEMM按search engine确定的最优fusion方案用compilation template执行→端到端相对PyTorch Compile加速1.5×
