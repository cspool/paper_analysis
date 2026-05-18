## Lossless Model Compression for LLM Inference（面向LLM推理的无损模型压缩）

术语是什么？通过联网搜索让回答具体和精准。
Lossless model compression for LLM inference指在保证bit-exact（解压后与原始权重完全一致，无精度损失）的前提下，压缩LLM模型权重以减少GPU memory footprint并加速推理的技术。与量化（GPTQ、AWQ）或剪枝（SparseGPT）等lossy方法不同，lossless compression不牺牲模型精度。ZipServ是第一个同时提供storage savings和推理加速的lossless compression系统。此前方法（DFloat11、DietGPU、nvCOMP）虽能压缩存储，但因decoupled pipeline和变长entropy codec的GPU不友好设计导致严重推理overhead（仅0.17–0.28× cuBLAS性能）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Lossless compression for LLM的算法流程（以ZipServ为例）：
1. 分析BF16权重的exponent分布 → 发现exponent entropy仅2.57–2.74 bits（远低于8-bit allocation），top-7 exponent覆盖>95%权重
2. 分离compressible component（exponent field）和incompressible component（sign + mantissa = 8 bits，接近最大熵）
3. 对exponent进行编码：TCA-TBE用3-bit fixed-length codeword，DFloat11用Huffman变长编码，DietGPU用ANS变长编码
4. 存储：高频元素存compact格式（bitmap index + sign/mantissa），低频outlier存full precision
5. 推理时解压：TCA-TBE用SIMT-friendly bitwise操作并行解压，而变长编码需要串行bit parsing
压缩率：平均每元素11.3 bits（vs BF16的16 bits），模型size reduction 25-30%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
多种实现方案：
- ZipServ (ASPLOS'26)：TCA-TBE fixed-length bitmap + fused ZipGEMM kernel，开源https://github.com/HPMLL/ZipServ_ASPLOS26.git
- DFloat11 (NeurIPS'25)：Dynamic-Length Float with Huffman coding，开源https://github.com/LeanModels/DFloat11
- DietGPU (2024)：GPU-native rANS codec，开源https://github.com/facebookresearch/dietgpu
- nvCOMP (NVIDIA)：通用GPU压缩库（rANS-based），https://github.com/NVIDIA/nvcomp
- Unweight (Cloudflare Research, 2026)：Huffman on Hopper GPUs，开源https://github.com/cloudflareresearch/unweight-kernels
- ZipNN (Intel)：Huffman-based model checkpoint compression
使用场景：资源受限部署（consumer GPU）、增大有效batch size/context length（释放memory给KV cache）、bit-exact推理（安全敏感应用）。

涉及论文标题：
- ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression

