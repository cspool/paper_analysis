## Non-Uniform Quantization (K-means Clustering-Based Weight Quantization / 非均匀量化)

术语是什么？通过联网搜索让回答具体和精准。
Non-uniform quantization是基于K-means clustering的LLM权重压缩方法。与uniform quantization（使用固定scale和zero-point将浮点值映射为均匀间隔的整数）不同，non-uniform quantization对每行权重独立运行K-means clustering，将权重值聚类为2^k个centroids（k为bit-width，如3-bit有8个centroids），每个权重被替换为一个k-bit index Wq指向其所属centroid。重建时W†=C[Wq]，即用index查表取回FP16 centroid。因为clustering能更好拟合不规则权重分布（特别是tail和outlier区域），non-uniform quantization在ultra-low bit-widths（如3-bit/2-bit）下显著优于uniform方案：论文引用SqueezeLLM在3-bit LLaMA-7B上perplexity 6.32 vs GPTQ (uniform) 7.55。使用该技术的代表工作包括SqueezeLLM、Any-Precision LLM、Bitsandbytes（支持uniform和non-uniform双模式）、SpQR等。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Non-uniform quantization pipeline分两阶段：

```
// ===== 离线量化阶段 =====
// 对weight matrix W∈R^{K×M}，每行独立执行

for each row r in W (共K行):
    // Step 1: K-means clustering
    centroids_r = kmeans(W[r,:], k=2^k)  // k=3时8个centroids
    // centroids_r ∈ FP16^{2^k}  例如 8个FP16值
    // 例如: centroids_r = [33.14, -48.24, 1.32, 0.90, -7.82, 53.13, 73.96, -27.63]
    
    // Step 2: 为每个weight分配index
    for each weight w_ij in W[r,:]:
        Wq[r,j] = argmin_d ||w_ij - centroids_r[d]||  // d∈[0,2^k-1]
        // Wq ∈ INT^{K×M}  每个元素为k-bit index

// 输出: quantized indices Wq (K×M, k-bit) + centroids C (K×2^k, FP16)
// 压缩比: (K×M×k + K×2^k×16) / (K×M×16) ≈ k/16 (忽略centroid overhead时)
```

```
// ===== 在线推理dequantization =====
// 对batch=1, decode阶段: W† = Dequant(Wq, C)
// 因为每个token的新activation只有1行

for each row r in weight matrix (参与当前计算的rows):
    wq_index = Wq[r, col]  // k-bit integer
    w_deq = C[r, wq_index] // centroid lookup: FP16
    // w_deq即为reconstructed weight
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
代表性开源实现：
- **SqueezeLLM** (ICML 2024 workshop): 开源 https://github.com/SqueezeAILab/SqueezeLLM，3-bit/4-bit non-uniform quantization with dense-and-sparse decomposition，K-means clustering + sensitivity-based optimization
- **Any-Precision LLM** (ICML 2024): 开源 https://github.com/SqueezeAILab/Any-Precision-LLM，扩展SqueezeLLM支持多bit-width
- **Bitsandbytes**: 开源 https://github.com/bitsandbytes-foundation/bitsandbytes，支持NF4数据格式(normal float 4-bit，一种non-uniform方案)和8-bit量化
- **SpQR** (ICLR 2024): 开源 https://github.com/Vahe1994/SpQR，结合sparse和non-uniform quantization实现near-lossless压缩

Non-uniform的核心trade-off：比uniform量化保留更高accuracy（尤其在low bit-width），但dequantization的centroid lookup是pointer-chasing的indirect memory access，降低cache locality，且3-bit等sub-byte width与GPU native INT类型不对齐，需要精心设计的kernel才能将内存节省转化为推理加速。

涉及论文标题：
- High-Throughput Non-Uniformly Quantized 3-bit LLM Inference

