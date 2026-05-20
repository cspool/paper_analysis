## Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  提出Multilevel Concentration算法，对VLM推理的视觉token做三层冗余消除：(1) Semantic Concentrator (SEC)：从attention SoftMax(QK^T)中提取text-to-image cross-modal attention block（T×M），对每个image token计算所有text token/heads中接收到的最大attention score s_j = max_{1≤i≤T,1≤k≤n} I_{i,j}^{(k)}，得到1×M importance vector，再用a-way streaming bubble sorter做top-k selection保留语义相关tokens，prune掉的token在后续P(i)×V和下游层不再加载；top-k保留比例逐层递减（layer 3/6/9/18/26分别保留40%/30%/20%/15%/10% image tokens）。(2) Similarity Concentrator (SIC) Similarity Gather：在FC/O projection/PV GEMM的每个m×n tile输出后（m=1024, n=32），将token embedding分成32维vectors，按2×2×2时空block（相邻两帧、8个vectors）做localized cosine similarity matching，选block中最高index vector为key与其余7个vectors比较（threshold=0.9），匹配则仅存代表vector index，不匹配写入compact output buffer；每个tile最终只将deduplicated vectors和1×m similarity map写回DRAM。(3) Similarity Scatter：在后续GEMM中对compact vectors执行计算，根据similarity map将partial sums复制/分发回原始token indices并在output-stationary buffer中累加，tile完成后调用Similarity Gather做下一轮压缩。实验比较accuracy和computation sparsity，对比Original (dense)、FrameFusion、AdapTiV pruning、CMC pruning。Focus平均accuracy degradation仅1.20%，平均sparsity 80.19%（vs FrameFusion 70%、AdapTiV 42.82%、CMC 51.75%）。

- 硬件平台是什么，配置是什么。
  算法评估：NVIDIA A100 GPU (80GB HBM)，FP16 precision，HuggingFace Transformers + lmms-eval框架。GPU对照实验还包括NVIDIA Jetson Orin Nano GPU with/without FrameFusion。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-Video-7B (Llava-Vid)、LLaVA-OneVision-7B (Llava-OV)、MiniCPM-V-2.6 (MiniCPM)。视频benchmarks：VideoMME (VMME)、MVBench (MVB)、MLVU。扩展到image VLM还使用LLaVA-OneVision、Qwen2.5-VL以及VQAv2、MME、MMBench。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：https://github.com/dubcyfor3/Focus（MIT License, HPCA 2026 Best Paper Candidate）。Zenodo DOI: https://doi.org/10.5281/zenodo.17851346。算法pipeline：
  1. Video输入→视觉编码器将每帧切分为patches并tokenize为visual embeddings→与text prompt拼接送入LLM
  2. Attention层：SoftMax(QK^T)矩阵包含image-to-image、image-to-text、text-to-image、text-to-text四块→SEC提取text-to-image (T×M) block作为cross-modal importance matrix I
  3. SEC importance analyzer：对每个image token j，取所有text token i和所有attention heads k中最大attention score作为importance s_j → a-way streaming bubble sorter选出top-k tokens → offset encoder记录保留tokens相对位置
  4. Pruned tokens在后续P(i)×V和下游FC层不再加载。保留比例逐层递减(40%→30%→20%→15%→10%)
  5. FC/PV/O projection层GEMM：以m=1024, n=32 tile为单位输出→convolution-style layouter按FHW坐标将vectors组织进2×2×2 block→SIC选key vector做cosine similarity matching (threshold=0.9)→匹配vectors记录代表index，不匹配写入output buffer→每个tile输出deduplicated vectors (p个, p<1024) + 1×m similarity map
  6. 下一层GEMM对p个concentrated vectors执行计算→Similarity Scatter根据similarity map将partial sums复制回原始token位置并累加→tile完成后Similarity Gather再次压缩
  7. 例如LLaVA-Video-7B在VideoMME上：原始6272 visual tokens→SEC prune后约~1800 tokens→SIC压缩后等效约~1100 tokens effective compute→总sparsity 82.82%，accuracy从64.15降至62.74 (-1.41)
