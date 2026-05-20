## V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval

- 属于算法pipeline的实现是什么？实验比较什么？
  提出ReSV（Real-time Streaming Video KV Cache Retrieval），一种training-free动态KV cache retrieval算法，专为streaming video LLM的iterative prefill stage设计。核心算法设计：(1) Hash-bit Key Clustering：利用视频相邻帧token的高时空相似性（cosine similarity热力图验证），通过随机hyperplane projection (Nhp=32) 将key矩阵降维+二值化为hash-bit（≤原始dimension的0.5%），用XOR+popcount计算Hamming distance做轻量聚类，避免高维cosine similarity昂贵计算。Hamming distance与cosine similarity相关性约0.8。聚类结果存入HC table（含cluster index, token index, KeyCluster, KeyCluster hash-bit, token count）。(2) WiCSum Thresholding：先计算Query×KeyCluster^T得ScoreCluster矩阵（仅对聚类representative key计算，远小于完整key cache），再按每行score×token count加权求和得weighted sum，最后从高分bucket开始累计，超过阈值Thr-wics（论文设0.3）后early-exit，动态决定每layer/head选择的token数，而非固定top-k。(3) Light Attention：执行阶段仅对选定cluster做attention，大幅降低memory和compute。实验比较：COIN benchmark五类任务（Step/Next/Task/Proc./Proc.+）的Top-1 accuracy和retrieval ratio，对比VideoLLM-Online baseline和InfiniGen/InfiniGenP/ReKV等fixed top-k retrieval方法。ReSV相对baseline accuracy仅下降0.8%，frame processing retrieval ratio平均32.7%（vs InfiniGenP 50.8%、ReKV 58.4%），text generation retrieval ratio平均2.5%（vs ReKV 31.2%），比ReKV平均少检索3.0× token。

- 硬件平台是什么，配置是什么。
  Edge: V-Rex8 (8 cores, 53.3 TFLOPS BF16, LPDDR5 204.8 GB/s, PCIe 3.0 x4 4 GB/s, KV cache offload to M.2 NVMe SSD) vs NVIDIA Jetson AGX Orin (54 TFLOPS FP16, 32 GB, ~40W)。Server: V-Rex48 (48 cores, 319.5 TFLOPS BF16, HBM2e 1935 GB/s, PCIe 4.0 x16 32 GB/s, KV cache offload to DDR4 CPU memory) vs NVIDIA A100 (312 TFLOPS FP16, 80 GB, ~300W)。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3 8B (LLM backbone) + SigLIP-ViT-L-384 (vision encoder)。数据集：COIN benchmark（comprehensive instructional video analysis）。KV cache sequence length sweep: 1K/5K/10K/20K/40K。准确率评估使用COIN五类任务。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：论文未提供官方开源仓库（HPCA 2026）。ReSV算法pipeline（以COIN benchmark streaming video LLM, Llama-3 8B + SigLIP为例）：
  1. 新视频帧到达→Vision Tower (SigLIP) + MLP Projector生成视觉embedding→进入LLM decoder layers
  2. Hash-bit Generation：在每层QKV generation后，对当前frame key做RoPE→通过Nhp=32个随机hyperplane降维（Key×Hyperplane^T）→二值化（≤0→0, >0→1）生成hash-bit
  3. Hash-bit Key Clustering：HCU计算current hash-bit与HC table中已有KeyCluster hash-bit的Hamming distance（XOR+popcount）→distance<Thhd（论文设7）则归入已有cluster→更新HC table（cluster id, token indices, KeyCluster, KeyCluster hash-bit, token count）
  4. Query×KeyCluster^T：用当前query与representative KeyCluster（而非完整key cache）做矩阵乘法→得ScoreCluster矩阵
  5. WiCSum Thresholding：对ScoreCluster每行按score×token count加权求和→从高分bucket开始排序和累计→累计weighted sum超过Thr-wics（0.3）则early-exit→输出selected cluster→通过HC table映射回原始token indices
  6. Light Attention：仅对selected token做attention→输出进入FFN→下一层decoder
  7. 下一层QKV generation同时，KV prediction和prefetch与当前层attention/FFN重叠执行
