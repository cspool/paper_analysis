
## Diffusion Transformer (DiT)

术语是什么？

Diffusion Transformer (DiT) 是以Transformer架构替代传统UNet作为denoising backbone的扩散模型。由Peebles和Xie在ICCV 2023提出，将扩散过程的每一步操作在完整的图像latent token序列上通过Transformer的self-attention机制进行去噪。DiT已取代UNet成为主流高质量图像生成模型的backbone，代表性模型包括Stable Diffusion 3 (SD3)和FLUX.1-dev (12B参数)。与LLM不同，DiT模型参数通常较小（最大开源DiT仅12B），能够fit在单张80GB H100 GPU上，因此DiT推理中的并行化主要目的是降低延迟而非解决显存容量问题。

从算法pipeline角度拆解术语：

DiT推理pipeline（以FLUX.1-dev 50步去噪为例）：

```
1: latent ← VAE.encode(input_image)        // 将图像压缩到latent space
2: latent_noisy ← add_noise(latent, t=T)    // 从纯噪声开始
3: for step t = T, T-1, ..., 1 do           // 50个denoising steps
4:     latent_noisy ← DiT_block(latent_noisy, t, text_conditioning)
5:        // 每个DiT block: self-attention over all latent tokens
6:        // latent tokens数取决于分辨率: 256→256 tokens, 512→1024, 1024→4096, 2048→16384
7: end for
8: output_image ← VAE.decode(latent_noisy)  // VAE decoder从latent恢复图像
```

关键特征：(a) **Stateless**：无KV cache，每步独立计算全部latent tokens；(b) **Compute-bound**：多步去噪在全量latent tokens上执行，计算量由分辨率决定（2048×2048约25 TFLOPs per step on FLUX）；(c) **Step数固定**：通常50步，每步耗时高度可预测（CV < 0.7%）；(d) **异构输入**：DiT serving workload由少量离散分辨率组成（256/512/1024/2048），但不同分辨率计算量差异巨大（256→556 TFLOPs vs 2048→24965 TFLOPs）。

术语一般如何实现？如何使用？

DiT模型由一系列Transformer block组成，每block包含multi-head self-attention + MLP。主流开源实现包括HuggingFace diffusers库中的DiTPipeline。Sequence Parallelism (SP)是DiT推理加速的主要并行方式：通过Ulysses attention沿token序列维度切分数据到多GPU，attention前通过all-to-all collective转换layout，attention后再通过all-to-all转回。DiT推理中SP的scaling efficiency是sublinear的——小分辨率（256×256）在SP=8时通信占比超30%导致效率极低，大分辨率（2048×2048）则受益于更多GPU。xDiT是目前主要的开源DiT推理框架，支持固定degree的SP。

MixFusion论文补充了U-Net与DiT在serving角度的关键差异：(1) U-Net包含Convolution算子（kernel size 1-3），后者需要跨patch context→需要Patch Edge Stitcher处理边界依赖；DiT仅含Transformer blocks（无Convolution）→patched inference时accuracy自然达到100%（PSNR inf/SSIM 1.0, Table 4）；(2) U-Net模型（SDXL）单denoising step含~7 blocks，DiT模型（SD3）含~24 blocks——更多blocks意味着更频繁的cache操作和更高的cache management overhead（Figure 17，SD3 cache overhead显著高于SDXL）；(3) SDXL不同分辨率间latency差距仅1.3×，SD3超2.4×——更大的variance为scheduling optimization留下更多空间（SD3上MixFusion scheduling相对baseline的收益更大）；(4) U-Net中Convolution引入non-linear complexity→SD3的throughput更易于MLP prediction（Table 5, MLP accuracy SD3 0.96 vs SDXL 0.81）。

涉及论文标题：
- TetriServe: Efficiently Serving Mixed DiT Workloads
- MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models
- Latent Wavelet Diffusion for Ultra-High-Resolution Image Synthesis

---

## Prefill Phase / Decode Phase（预填充阶段 / 解码阶段）

术语是什么？

LLM推理的两个核心阶段：
- **Prefill Phase（预填充阶段）**：处理输入prompt的所有token，一次性计算所有位置的Key和Value并存入KV Cache，同时生成第一个输出token。计算特征为**compute-bound（计算密集）**，涉及大规模矩阵乘法（prompt_length × hidden_dim）、并行注意力计算和FFN。
- **Decode Phase（解码阶段）**：每步生成一个新token，计算该token的Query并与KV Cache中的所有历史Key/Value进行注意力计算。计算特征为**memory-bound（内存密集）**，每步计算量极小（约2×参数量FLOPs），HBM访问是主要瓶颈。

从算法pipeline角度拆解术语：

完整LLM推理pipeline伪代码：

```
// ===== Prefill Phase =====
function prefill(input_tokens[0..L-1], model_weights, kv_cache):
    // Step 1: 逐层处理
    for layer in 0..num_layers-1:
        // Step 1a: QKV投影（矩阵乘法，compute-bound）
        Q[layer] = input_tokens[layer] @ W_Q[layer]    // [L, d] @ [d, d]
        K[layer] = input_tokens[layer] @ W_K[layer]
        V[layer] = input_tokens[layer] @ W_V[layer]
        
        // Step 1b: 将KV存入Cache
        kv_cache.store(layer, K[layer], V[layer])       // HBM写入
        
        // Step 1c: 自注意力（FlashAttention kernel）
        attn_out[layer] = flash_attention(Q[layer], K[layer], V[layer])
        // 计算复杂度: O(L^2 * d)  — L为prompt长度
        
        // Step 1d: FFN + 残差
        input_tokens[layer+1] = ffn(attn_out[layer]) + input_tokens[layer]
    
    // Step 2: 从最后一层采样第一个token
    first_token = sample(input_tokens[num_layers])
    return first_token
// Prefill阶段总计算量: O(L * d^2 + L^2 * d)
// GPU利用率特征: 高SM占用率(>80%)，高计算/访存比

// ===== Decode Phase =====
function decode(last_token, model_weights, kv_cache):
    // Step 1: 逐层处理
    for layer in 0..num_layers-1:
        // Step 1a: 仅计算新token的Query
        q[layer] = last_token[layer] @ W_Q[layer]       // [1, d] @ [d, d]
        
        // Step 1b: 从KV Cache加载所有历史K, V（HBM读取瓶颈）
        K_all = kv_cache.load(layer)                     // HBM读取 [L, d]
        V_all = kv_cache.load(layer)                     // HBM读取 [L, d]
        
        // Step 1c: 单token注意力
        attn_out[layer] = paged_attention(q[layer], K_all, V_all)
        // 计算复杂度: O(L * d) — 远小于Prefill
        
        // Step 1d: FFN + 残差
        last_token[layer+1] = ffn(attn_out[layer]) + last_token[layer]
    
    // Step 2: 采样下一个token
    next_token = sample(last_token[num_layers])
    return next_token
// Decode阶段每token计算量: O(d^2 + L * d) — L为累积序列长度
// GPU利用率特征: 低SM占用率(<10-30%)，低计算/访存比，HBM带宽瓶颈

// ===== LLM推理主循环 =====
kv_cache = init_kv_cache()
next_token = prefill(prompt_tokens, weights, kv_cache)
output_tokens = [next_token]

while next_token != EOS and len(output_tokens) < max_len:
    next_token = decode(next_token, weights, kv_cache)
    output_tokens.append(next_token)
```

两阶段的GPU利用率特征差异是PD Multiplexing的理论基础：Prefill高计算低访存，Decode低计算高访存，两者在同一GPU上并行可以互补。

术语一般如何实现？如何使用？

实际推理框架中：
- **Prefill**：通常使用FlashAttention/FlashInfer等高效kernel，将prompt tokens一次性经所有Transformer层处理。
- **Decode**：使用PagedAttention管理KV Cache分页，减少HBM碎片；每token逐层处理，利用KV Cache避免重复计算。
- **Chunked Prefill**：将长Prefill按token切成多块交替执行，平衡TTFT和ITL。
- **PD Multiplexing**：利用两阶段不同的GPU利用率特征，在同一GPU上SM空间分区并行执行。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

## KV Cache Replication for Sequence Parallelism（序列并行中的KV缓存复制）

术语是什么？

KV Cache Replication for SP是Shift Parallelism为Sequence Parallelism在inference场景下添加的机制：当使用GQA的模型中KV head数（h_kv）小于SP degree时，无法将KV head直接均匀分配到所有GPU（因为head数不够），通过all-to-all通信的send buffer中复制KV head数据来"虚拟地"扩展KV head数。这与TP的KV cache复制思路不同：TP可以通过在QKV projection weight中复制KV列来重计算KV cache（因为是weight-level partitioning），而SP每个rank只持有部分input sequence，无法通过重计算覆盖全部sequence位置，因此必须通过通信中的buffer复制实现。

典型场景：Qwen-30B-A3B有4个KV heads，但需要在8 GPU上使用SP=8进行推理。每个GPU只能原生持有0.5个KV head，无法工作。通过KV cache replication，在all-to-all send buffer中将每个KV head复制到2个target rank，实现8-way SP。

从算法pipeline角度拆解术语：

KV Cache Replication在SP inference forward pass中的位置和流程：

1. **QKV Projection**（Line 3, Algorithm 1）：输入embedding `[n/SP, d]`与QKV weight `[d, h + 2×h_kv]`乘，得`qkv_heads[n/SP, h + 2×h_kv]`。此处`h + 2×h_kv`替代了标准MHA的`3×h`。——这是与GQA的对接点。

2. **Send Buffer构建**（Line 4前）：为SP all-to-all准备send buffer。对于KV heads部分（`2×h_kv`个heads），如果`h_kv < SP×TP`（即KV heads不足以覆盖所有all-to-all target ranks），则在send buffer中复制KV数据。例如SP=8、h_kv=4时，每个KV head复制到2个target ranks。对于Q heads无需复制（Q head数通常充足）。

3. **Fused All-to-All**（Line 4）：单次all-to-all同时承载Q、K、V的head重分布和KV复制。接收端每个GPU获得完整序列的`(h + 2×h_kv)/(SP×TP)`个unique或replicated heads。

4. **Attention计算**（Line 5）：每个GPU对本地head shard执行attention，使用本地或复制的KV cache条目。由于KV cache replication保证了每个GPU都能访问所需KV heads，attention正确性不受SP degree超过#KV heads的影响。

5. **Decoder侧一致性**：在shift config（full TP）下，KV cache replicate的条目与TP weight replication产生的K/V一致——因为两者都是进行head-level replication，保证了KV Cache Invariance。

术语一般如何实现？如何使用？

实现在ArcticInference的GQA extension中。通过将QKV projection的head维度从3×h适配为`h + 2×h_kv`，并在fused all-to-all的send buffer构建阶段按需复制KV head数据。用户无需手动配置——系统根据模型config（#Q heads, #KV heads）和SP degree自动判断是否需要replication。这种机制使SP能够扩展到任意模型，不受#KV heads限制，是SP从training（通常MHA无此问题）适配到inference（大量GQA模型）的关键一步。

涉及论文标题：
- Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads

---

## gpt-oss / gpt-oss 120B

术语是什么？
gpt-oss是OpenAI发布的open-source大型语言模型系列（论文引用[65]: OpenAI 2025 "Introducing GPT-OSS"）。gpt-oss 120B是120B参数的Mixture-of-Experts (MoE)架构版本，基于Llama-style架构。模型使用FP4精度（4-bit浮点权重），128个experts（MoE routing，每次激活top-k个experts），hidden size为2,880，36层transformer blocks。该模型在HNLPU论文中被选为系统级评估的target model，是论文"极致专用化"论点的基础——一个主导性的预训练LLM作为通用认知基底，使hardwired实现变得合理。

从算法pipeline角度拆解：
gpt-oss 120B在HNLPU中的推理pipeline（Token-In-Token-Out，36层transformer每层）：
```
1. Input: token embedding (1, 2880) from HBM
2. For layer = 1 to 36:
   a. GQA Attention:
      - Q = X × Wq  (column-wise partitioned, all-reduce)
      - K = X × Wk  (column-wise partitioned, reduce→chip#)
      - V = X × Wv  (same as K)
      - Attention = Softmax(Q×K^T/√d) × V  (VEX FlashAttention)
      - Xo = Attention × Wo + X  (row-wise Wo, all-reduce + all-gather)
   b. MoE FFN:
      - Xnorm = RMSNorm(Xo)
      - Xrout = Xnorm × Wrout (router: Wrout (2880,128), top-k)
      - Xup = masked_X × Wup (8 experts/chip, 128 experts total, parallel)
      - Xgate = masked_X × Wgate (same partitioning)
      - Y = SwiGLU(Xgate) ⊙ Xup × Wdown + Xo (all-chip all-reduce)
3. Unembedding: Y × Wue → logits → Sampling → output token ID
```
模型配置：hidden size 2880, 64 query heads (GQA: 每8 query heads对应1 KV head), 128 experts (每chip 8 experts), FP4 weight精度。

术语一般如何实现？如何使用？
模型在GPU baseline（H100）上通过TensorRT-LLM部署；在HNLPU中，模型权重被物理固化到芯片金属导线中——不需要任何软件框架加载和运行。gpt-oss 120B的FP4量化是OpenAI原生支持的，论文未进行额外压缩。模型作为通用认知基底：用户通过prompt（自然语言token序列）而非ISA指令来编程hardwired处理器，利用in-context learning和zero-shot reasoning执行任意任务。

涉及论文标题：
- Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates

## Weight Hardwiring / Hardwired Weights（权重固化/硬连线权重）

术语是什么？
Weight Hardwiring是将神经网络的权重参数物理固化到芯片电路中（而非存储于SRAM/DRAM并动态加载）的实现方法。其核心理念来自1980年代的VLSI神经网络实现（Graf et al., 1988）和Hinton的"mortal computing"论点。在hardwired实现中，权重不再是"数据"而是"电路的一部分"——乘法器被优化为multiply-by-constant（CMAC），权重参数在芯片制造时确定且物理不可变。HNLPU将这一概念推至新高度：通过Metal-Embedding将权重编码在金属导线的3D拓扑中，而非2D硅器件grid。

从算法pipeline角度拆解：
Hardwired weight inference与传统weight-loading inference的算法层对比：
```
// 传统推理（GPU）:
for each token in autoregressive loop:
    for layer in 1..36:
        W_qkv = HBM_read(weights_addr[layer].qkv)  // 从HBM加载权重
        Q, K, V = X × W_qkv                          // 通用GEMM
        // ... attention, FFN with repeated weight loading ...

// Hardwired weight推理（HNLPU）:
for each token:
    for layer in 1..36:  // 36层pipeline并行
        // 无weight loading步骤——权重存在于金属连线中
        // 输入信号通过金属线自动路由到对应的POPCNT累加器
        Q, K, V = HN_Array_compute(X)  // 即电路本身
        // ... attention on VEX, FFN on HN Array ...
```
关键区别：hardwired方案消除了每个decoding step的weight fetch开销。gpt-oss 120B有约120B参数，在GPU上每次decoding需反复读取这些权重（占大部分系统功耗），在HNLPU中这些权重是零开销的电路连接。代价是权重不可更新（除非重新制造芯片），但Sea-of-Neurons架构将参数更新respins的NRE从$480M降至$37M。

术语一般如何实现？如何使用？
实现方式演进：(1) 早期VLSI/光学/printed flexible电路直接hardwire小型网络；(2) CMAC方案——用constant multiplier替代通用multiplier，但仍嵌入在硅器件cell中（$6B光罩成本）；(3) HNLPU的Metal-Embedding——将权重从硅器件提升到金属互联（15×密度提升，112×光罩成本降低），使120B级模型在经济上可行。适用于长期高容量部署（年均数百万token服务），与GPU短期模型开发形成互补。论文提出LoRA for post-deployment updates（~1% field-programmable HNs at side-channel容纳动态权重）作为未来方向。

涉及论文标题：
- Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates

## TCA-TBE (Tensor-Core-Aware Triple Bitmap Encoding)

术语是什么？通过联网搜索让回答具体和精准。
TCA-TBE是ZipServ提出的面向GPU Tensor Core的固定长度无损压缩编码格式。其核心思路是：利用LLM中BF16权重指数字段的高偏态分布（>95%权重仅使用top-7连续exponent），将每个8×8 weight tile编码为三个独立64-bit bitmap（每bit表示codeword的一个bit-plane）加两个紧凑value buffer（PackedSignMantissa和FullValue fallback），替代Huffman/ANS等变长entropy codec。3-bit codeword使平均每元素仅11.3 bits（理论下界10.6 bits），压缩率约1.41×。TCA-TBE是对同一团队此前SpInfer中TCA-BME（面向sparsity的bitmap encoding）的演进，将bitmap方法从稀疏模式扩展到无损压缩场景。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TCA-TBE算法pipeline分两阶段：
1) 离线压缩（Algorithm 1 in paper）:
   - 对每层weight matrix扫描exponent直方图 → 选top-7连续exponent值（覆盖>95%权重），记录base_exp = min(top_exponents) - 1
   - 对每个8×8 FragTile编码：遍历64个元素，若exponent∈top-7则计算3-bit codeword c = exponent - base_exp（c∈[1,7]），将c的三个bit分别写入三个bitmap，将sign+mantissa (8-bit)写入PackedSignMantissa buffer；若exponent∉top-7，将完整BF16写入FullValue fallback buffer
   - 输出：三个全局bitmap array + PackedSignMantissa array + FullValue array + Offset array
2) 在线解压（per-thread）:
   - 对每个元素，bitwise OR三个bitmap得到64-bit spatial indicator mask M
   - 若M[position]=1（高频）：读取PackedSignMantissa中的sign+mantissa，从bitmap恢复3-bit codeword c，exponent = base_exp + c（implicit lookup，无表查找），组装BF16
   - 若M[position]=0（fallback）：直接从FullValue读取完整BF16
解压全程仅需bitwise OR + POPC（population count）+ integer ADD，无分支、无shared memory table lookup。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TCA-TBE实现为CUDA/C++（约2.5K行），作为ZipServ的offline compressor模块。使用方式：
1. 加载BF16模型权重 → 逐层调用compressor分析exponent分布 → 生成TCA-TBE格式压缩文件
2. 压缩LLaMA3.1-8B约需2.5分钟（16核CPU），为一次性offline操作
3. 运行时由ZipGEMM kernel直接读取TCA-TBE格式进行fused decompression+GEMM
TCA-TBE的3层tiling设计（8×8 FragTile → 16×16 TensorCoreTile → 64×64 BlockTile）直接对齐NVIDIA Tensor Core的mma.m16n8k16 operand register layout，消除runtime坐标变换。开源地址：https://github.com/HPMLL/ZipServ_ASPLOS26.git

涉及论文标题：
- ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression

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

## Exponent Contiguity in BF16 Weights（BF16权重中的指数连续性）

术语是什么？通过联网搜索让回答具体和精准。
Exponent contiguity是指LLM的BF16权重中，最频繁出现的top-K个exponent值在数值上形成连续序列的现象。ZipServ论文通过分析3875个weight matrices（覆盖Gemma-3、Mistral、Qwen2.5、LLaMA3.1四个LLM家族）发现：99.6%的矩阵中top-7 exponent构成连续序列（e*, e*+1, ..., e*+6），平均覆盖97.1%的权重。论文在Appendix A中给出数学证明：假设权重服从零均值正态分布N(0,σ²)，exponent概率函数P(X=x)=erf(2^(x+1)/(σ√2))−erf(2^x/(σ√2))是单峰的(unimodal)，因此top-K集合必然是连续的。这个性质是TCA-TBE设计的基础——它使编码仅需记录一个base_exp而非7个独立exponent值，将解码简化为base_exp + codeword的算术操作。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在TCA-TBE中，exponent contiguity被利用的方式：
1. Profiling阶段：扫描weight matrix exponent直方图 → 排序找top-7频率的exponent值
2. Contiguity检查：验证top-7是否连续 → 对于99.6%矩阵为连续，此时选择contiguous window覆盖，设base_exp = min(top_exponents) - 1
3. 编码：codeword 001–111直接映射到base_exp+1 ~ base_exp+7
4. 解码：exponent = base_exp + codeword，一条IADD指令完成
若无contiguity性质，则需7-entry lookup table（额外的shared memory访问和地址计算），contiguity将lookup简化为算术，消除shared memory往返。对比：Huffman/ANS等方法使用通用frequency-sorted codebook，无contiguity利用，每个symbol解码需查表。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该性质在LLM中广泛存在（源自权重的高斯分布假设），可用于任何基于exponent压缩的方案。实现时：
1. 对目标模型做exponent profiling（轻量，仅需一次前向扫描）
2. 验证contiguity（top-7是否连续）
3. 若不连续（<0.4%矩阵），可fallback到通用lookup table编码
该性质的数学基础（单峰性+高斯权重）使得它在不同LLM family间可迁移，无需per-model调优。

涉及论文标题：
- ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression

## Speculative Decoding（投机解码）

术语是什么？通过联网搜索让回答具体和精准。
Speculative Decoding（投机解码）是一种加速LLM自回归推理的算法范式。核心思想是"先小模型起草，后大模型验证"：使用一个轻量级draft model快速生成γ个候选token，再由完整的target model通过一次并行forward pass验证所有候选，通过概率接受机制(acceptance probability α_i = min(1, p(t_i)/q(t_i)))保证输出分布与target model原生自回归解码完全一致。理论加速比Speedup = c·γ / ((1-ρ)·c·γ + c·ρ + 1)，其中c为target/draft速度比、ρ为平均接受率、γ为draft长度。当c≫1且ρ→1时，加速比逼近γ。DFVG是ASPLOS'26上首个将speculative decoding完整映射到FPGA+GPU异构硬件的系统。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Speculative Decoding的单次迭代pipeline：
```
给定: prefix X_1:j, draft model M_q, target model M_p, draft length γ

// Step 1: Draft Generation (自回归，draft model逐token)
for i = 1 to γ:
    x̃_{j+i} ~ M_q(· | X_1:j+i-1)  // draft model逐token生成候选
    q_i = q(x̃_{j+i} | X_1:j+i-1)  // 记录draft概率

// Step 2: Target Verification (并行，target model一次forward)
p_1:γ = M_p(X_1:j+γ)  // target model并行计算所有位置的真实概率

// Step 3: Probabilistic Acceptance (逐token验证)
for i = 1 to γ:
    α_i = min(1, p_i(x̃_{j+i}) / q_i(x̃_{j+i}))  // 重要性采样接受概率
    if random() < α_i:
        accept x̃_{j+i}  // 接受此token
    else:
        // 拒绝，从修正分布重采样
        p'_i = norm(max(0, p_i - q_i))
        x_{j+i} ~ p'_i
        break  // 后续所有候选token丢弃
```
Tree-based变体：draft model每步生成k个分支候选（而非单token），形成token tree。验证时target model并行计算tree中所有节点概率，按top-down（SpecInfer的OT方法）或bottom-up（Traversal Verification）选择最长有效前缀。DFVG的ADAPT算法动态决定每层分支数k_i和tree深度D。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
主流实现：
- **SpecInfer** (ASPLOS'24)：多GPU tree-based speculative decoding，静态预定义branch配置，optimal transport tree verification
- **EAGLE** (ICML'24)：feature-level speculative decoding，利用draft model的feature uncertainty替代token probability
- **Medusa** (ICML'24)：在target model上加多个extra decoding heads并行预测多token
- **DuoDec** (2025)：CPU+GPU异构，hardware-aware draft budgeting
- **DFVG** (ASPLOS'26)：FPGA+GPU异构，ADAPT动态tree构建+TreeSort-Verify高效验证，开源https://github.com/ShaoqiangLu/DFVG
- **AdaServe** (2026)：构建于FlexFlow Serve之上，SLO-customized speculative decoding，将multi-SLO serving形式化为budget-constrained token tree构造，使用beam search speculation + two-phase selection (SLO-customized + throughput-optimized) + tree-based verification，开源https://github.com/zikun-li/AdaServe-Artifact-Evaluation
适用场景：所有LLM自回归解码，2×-4×加速比，数学等价于原生解码（无质量损失）。

AdaServe 的 SLO-customized speculative decoding：将 multi-SLO serving 与 speculative decoding 结合，形式化为带 budget 约束的 token tree 构造问题。小 draft model 用 beam search 生成 candidate token tree，CPU scheduler 按 SLO 需求选节点（SLO-customized selection），剩余 budget 按全局 path probability 分配（throughput-optimized selection），target LLM 并行做 tree-based verification。同一 batch 中不同 SLO 请求可在同一次 verification 中前进不同 token 数。

涉及论文标题：
- DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU
- Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems
- AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding

## ADAPT (Adaptive Dynamic Allocation for Parallel Tree / 自适应动态并行树分配)

术语是什么？通过联网搜索让回答具体和精准。
ADAPT是DFVG提出的budget-constrained integer programming方法，用于在speculative decoding中动态构建token tree。与传统静态预定义branch配置（如SpecInfer每层固定k分支）不同，ADAPT根据draft model在每个位置的confidence（概率分布）和FPGA硬件并行度约束（总branch预算B、每层最大branch数k_max、最小深度D_min=⌈T_verify/T_draft⌉）动态决定tree结构。目标函数max Σ p_{i,j,l}·x_{i,j,l}最大化expected accepted tokens。由于整数规划NP-hard，ADAPT使用Temperature-Controlled Probabilistic Gumbel Sampling作为贪心近似，时间复杂度O(D·k_max·log k_max)满足实时推理要求。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ADAPT算法（BuildTreeADAPT，每iteration执行）：
```
Input: prefix S, draft model M_d, budget B, k_max, D_min, temperature T, threshold τ
Output: token tree T

// 1. Draft model forward → vocab-level probabilities
probs[1..V] = M_d(S)

// 2. 逐层构建tree
T = {root}
for depth d = 1 to D_max:
    N_d = {}  // candidate extension set
    for each node j in layer d-1:
        for token l where p_{d,j,l} > τ_d:
            // Path Cumulative Probability
            P_cum(d,j,l) = p_{d,j,l} · Π_{(k,a_k)∈path(d,j)} p_{k,par(a_k),a_k}
            N_d.add((d, j, l, P_cum))
    
    // Temperature-controlled Gumbel sampling
    for each candidate in N_d:
        P̃_cum = exp(P_cum / T) / Σ exp(P_cum / T)  // softmax
        G = -log(-log(U)) + log(P̃_cum)  // Gumbel, U~Uniform(0,1)
    
    k_i = min(k_max, |N_d|)
    selected = argmax_k(G, k_i)
    
    for each s in selected:
        T.add_node(child_of(s.parent), s.token)
    
    if total_nodes(T) >= B and d >= D_min: break

return T
```
关键：k_max设为FPGA PE array支持的并行数（如8/16/32）；D_min=⌈T_verify/T_draft⌉确保draft-verify可pipeline overlap；T控制exploration（T→0→deterministic top-k，T大→随机探索）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在DFVG中ADAPT实现在FPGA的Branch Management和Token Management硬件模块中。draft model forward后→Token Management计算confidence→Branch Management执行ADAPT→PE array并行生成多分支tokens。配置通过yaml文件（B, k_max, D_min, T, τ）。Qwen3-0.6B/8B pair上维持75%-85% acceptance rate，draft length呈long-tail分布自动适配不同token难度。开源：https://github.com/ShaoqiangLu/DFVG。

涉及论文标题：
- DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU

## Generative Recommenders (GRs / 生成式推荐)

术语是什么？通过联网搜索让回答具体和精准。
Generative Recommenders (GRs) 是一种新兴的推荐系统范式，使用Transformer-based生成模型（如LLM或HSTU）替代传统Deep Learning Recommendation Models (DLRMs)，将推荐排序任务建模为sequence-to-sequence生成问题。GR将user profile、candidate items和system instructions编码为token序列，经Transformer self-attention处理后输出每个item的relevance score，最终选出top-k推荐。相比DLRM的embedding table + small dense MLP架构，GR具有更强的表达能力、可捕获复杂高阶user-item交互，且可由Scaling Law驱动——更大模型和更多计算带来更好的推荐效果。Meta的HSTU已在生产环境中部署并实现12.4% topline提升。然而GR的推理计算量比传统DLRM大两个数量级，呈现与LLM prefill阶段相似的compute-bound特性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GR ranking推理pipeline（以LLM-based GR为例）：
```
给定: user profile token set U, N个candidate items I={I1,...,I_N}, instruction tokens Instr
    L层Transformer, 每层含self-attention + FFN

// Step 1: Token序列组装
x^0 = Embed([U, I1, ..., I_N, Instr])  // [T, d] where T = |U|+Σ|I_i|+|Instr|

// Step 2: 逐层Transformer处理
for l = 1 to L:
    // Multi-head self-attention
    q^l, k^l, v^l = Proj(x^{l-1})  // QKV投影
    attn^l = CausalAttention(q^l, k^l, v^l)  // causal mask, 屏蔽跨item attention
    // FFN
    x^l = FFN(attn^l) + x^{l-1}

// Step 3: 判别token → relevance score
z = W_out · x^L[T]  // 最后一层最后一个token投影到vocab logits [V]
for each item i in 1..N:
    s_i = exp(z[v_i]) / Σ_{j=1}^{N} exp(z[v_j])  // softmax normalization

// Step 4: TopK排序
return TopK({s_1, ..., s_N})  // 返回前k个最高分的item
```
关键细节：attention mask屏蔽不同item之间的cross-attention（遵循HSTU设计），保证各item的评分独立性。GR inference的特征是compute-bound（类似LLM prefill），因为长输入序列（up to 8K tokens）需密集矩阵乘法。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
主流GR实现：
- **HSTU** (Meta, ICML'24)：Hierarchical Sequential Transduction Units，使用item embedding table + causal attention，万亿参数级，已在Meta生产部署。将user-item交互建模为next-token prediction。
- **LLM-based GR**：直接使用预训练LLM（如Qwen2-1.5B、Llama3-1B）fine-tune为推荐排序模型。通过自然语言或结构化token表示user profile和item属性。BAT采用此路线。
- **OneRec** (快手, 2025)：统一检索和排序的生成式推荐框架。
- **GenRank** (淘宝, 2025)：面向大规模工业级生成式排序。
GR可用于推荐系统的排序阶段（输入100-200个candidate items，输出top-k最终推荐），是推荐pipeline中计算最密集的环节。

涉及论文标题：
- BAT: Efficient Generative Recommender Serving with Bipartite Attention

---

## Bipartite Attention（二分注意力机制）

术语是什么？通过联网搜索让回答具体和精准。
Bipartite Attention是BAT提出的面向生成式推荐(GR)的新型注意力机制。其核心基于关键洞察：推荐prompt中user token和item token的语义是排列不变的(permutation-invariant)——交换user和item的顺序不影响上下文语义。Bipartite Attention自适应选择User-as-prefix或Item-as-prefix两种attention模式，使item KV cache可跨用户共享，打破传统User-as-prefix attention中item cache依赖前置user context而无法跨用户复用的限制。该机制通过修改attention mask和position encoding实现，确保不损失推荐精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Bipartite Attention两种模式的计算流程：

```
// 公共定义
给定: user tokens U (T_u tokens), N items I_1..I_N (T_i tokens total), instr tokens Instr
判别token: 序列最后一个token，投影到vocab logits计算各item relevance score

// ===== Mode 1: User-as-prefix Attention =====
// 输入序列: [U, I1, ..., I_N, Instr]
// KV Cache: user prefix U的K,V预计算并缓存
// 实时计算: 仅item和instruction token的Q,K,V

q_{I,Instr}, k_{I,Instr}, v_{I,Instr} = Proj(x_{I,Instr})  // 新计算
k_U, v_U = LoadFromCache()                                   // 缓存读取
attn = Attention(q_{I,Instr}, k_{I,Instr}∪k_U, v_{I,Instr}∪v_U)
// item间attention被mask屏蔽

position encoding: 
  - user tokens: positions 0..T_u-1
  - item tokens: positions 从T_u开始, 每个item共享相同起始位置
  - 保证item KV cache独立于后续user/instruction token

// ===== Mode 2: Item-as-prefix Attention =====
// 输入序列: [I1, ..., I_N, U, Instr]
// KV Cache: item prefix I的K,V预计算并缓存
// 实时计算: 仅user和instruction token的Q,K,V

q_{U,Instr}, k_{U,Instr}, v_{U,Instr} = Proj(x_{U,Instr})  // 新计算
k_I, v_I = LoadFromCache()                                    // 缓存读取
attn = Attention(q_{U,Instr}, k_{U,Instr}∪k_I, v_{U,Instr}∪v_I)

position encoding:
  - item tokens: positions 重置为0(或可选标记后)，所有item共享相同起始位置
  - user tokens: positions 从T_i开始
  - 保证item KV cache对任意user context独立可用

// Attention Mask (两者共用)
Mask[i][j] = 0 if i和j属于不同item  // 屏蔽跨item attention
Mask[i][j] = 0 if j > i              // causal mask
```

核心设计要点：(1) Item-as-prefix中item的position encoding从0开始，使item cache完全独立于user context；(2) 跨item attention被mask屏蔽，遵循推荐系统中item独立评估的原则；(3) Item-as-prefix不牺牲精度——实验中IP(Item-as-prefix)与UP(User-as-prefix)在Recall@k/MRR@k/NDCG@k上性能相当或更优。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Bipartite Attention在Bat系统中实现为vLLM的自定义attention module，使用FlashInfer高性能backend。模型训练时hardcode新的position encoding（从0重置item位置），无需额外训练开销；推理时根据hotness-aware scheduler决策动态选择使用哪一模式。对instruction-tuned模型（如Llama3-Instruct）可能出现IP精度下降，可选择支持修改position encoding的base model（如Qwen2）或应用Position-Independent Caching算法（如CacheBlend）选择性重新计算关键token以缓解精度损失。Bipartite Attention已在Qwen2-1.5B/7B和Llama3-1B上验证有效，在不同数据集上IP与UP的推荐质量相当。

涉及论文标题：
- BAT: Efficient Generative Recommender Serving with Bipartite Attention

---

## Denoising Diffusion Probabilistic Model / Diffusion Model Denoising Pipeline（去噪扩散概率模型 / 扩散模型去噪流水线）

术语是什么？

扩散模型 (Diffusion Model) 通过迭代去噪从纯噪声生成文本条件图像。其pipeline包含两个过程：(1) Forward Process（前向过程）：逐步向真实图像添加高斯噪声，经过T步后逼近标准正态分布；(2) Reverse Process（反向/去噪过程）：从纯噪声开始，逐步去噪恢复图像，每步将当前噪声latent送入完整模型（UNet或DiT），共需T步（通常T=50）。扩散模型评价指标包括FID（Frechet Inception Distance，越低越好，衡量与真实图像的分布距离）、CLIPScore（越高越好，衡量图文对齐）、IS（Inception Score，越高越好，衡量质量与多样性）、PickScore（越高越好，基于人类偏好训练的评分）。

从算法pipeline角度拆解术语：

扩散模型去噪pipeline（以Stable Diffusion 50步为例）：

```
// Forward Process (training only): x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * epsilon

// Reverse Process (inference):
1: latent_T ~ N(0, I)                          // 初始化纯噪声
2: for t = T, T-1, ..., 1 do                    // 50 denoising steps
3:     epsilon_theta = model(latent_t, t, prompt_embedding)  // UNet/DiT预测噪声
4:     latent_{t-1} = 1/sqrt(alpha_t) * (latent_t - (1-alpha_t)/sqrt(1-bar_alpha_t) * epsilon_theta) + sigma_t * z
5: end for
6: image = VAE.decode(latent_0)                 // VAE decoder将latent恢复为图像
```

扩散动态的关键特性：early denoising steps决定图像结构（layout、object position），later steps关注细节（texture、color、fine details）。这使small model可以处理cache-hit请求的later refinement steps——因为cached图像已提供结构，仅需细节调整。

术语一般如何实现？如何使用？

主流开源实现包括HuggingFace diffusers库的StableDiffusionPipeline、StableDiffusionXLPipeline、FluxPipeline等。MoDM利用扩散动态特性：cache-hit时对检索图像加噪到timestep t_k（公式2），然后仅用small model执行剩余T-k步。MoDM使用的模型：SD3.5L (8B, BF16), FLUX.1-dev (12B, BF16), SDXL (3B, FP16), SANA (1.6B, BF16), SD3.5L-Turbo (10步蒸馏版)。所有模型生成1024x1024图像，T=50步（除Turbo用10步）。

涉及论文标题：
- MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models
- Difflow: A Data-Characteristic-Aware Serving System for Diffusion Models（基于图文相似度的缓存检索）
- MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models（patch级并行去噪、operator taxonomy for serving）

## Operator Taxonomy for Patch-Level Diffusion Inference

术语是什么？通过联网搜索让回答具体和精准。

Operator Taxonomy for Patch-Level Diffusion Inference（patch级扩散推理的算子分类）是MixFusion提出的将扩散模型中的算子按是否需要跨patch上下文(context)进行分类的体系，用于指导patch-level parallel serving的设计。扩散模型的denoising过程中，大多数算子操作在"pixel level"（局部）而非"image level"（全局）信息上，这些算子可被decompose为独立子算子在各patch上并行执行。算子分为两类：(a) Pixel-wise operators——Linear、FeedForward、Cross Attention等，仅依赖当前像素/patch内部信息，可直接在各patch上独立并行执行；(b) Context-dependent operators——Self-Attention和Convolution，需要跨patch上下文信息来保证输出一致性（Self-Attention需要all patches交互形成Cartesian product；Convolution需要相邻patch的边界像素）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Patch-level operator execution pipeline：

```
对每个denoising step的每个block：
1. // Pixel-wise operators: 直接对各patch独立批处理
2. for op in [Linear, FeedForward, CrossAttention]:
3.     // 所有N个patches形状相同→直接组成batch[N, ...]执行
4.     output = op(patches_batch)  // standard batched execution
5.
6. // Self-Attention: 按分辨率分组reconstruct为全图后批处理
7. for resolution_group in unique_resolutions:
8.     // 例如：resolution=1024的patches重组为完整feature map
9.     full_feature = reconstruct_from_patches(patches_of_this_resolution)
10.    attention_output = SelfAttention(full_feature)  // batched attention per resolution group
11.
12. // Convolution (仅U-Net, DiT无此操作): 需要跨patch边界
13. for each patch:
14.    // GroupNorm + boundary stitching fused in single kernel
15.    output = FusedGroupNormWithStitching(patch, neighbor_boundaries)
16.    // Padding with 0 when neighbor absent（如image边缘patch）
```

U-Net（SDXL）vs DiT（SD3）在operator taxonomy下的差异：
- U-Net含ResNet blocks（含Convolution）和Transformer blocks（含Self/Cross Attention）
- DiT仅含Transformer blocks（Self-Attention + Cross Attention + FFN）
- SD3无Convolution→patched inference自然100% accuracy
- SDXL Convolution kernel size 1-3（kernel>1时出现跨patch依赖）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MixFusion在PyTorch diffusion pipeline中实现此分类：(1) pixel-wise operators通过标准PyTorch batch execution处理——因为CSP格式确保所有patches相同shape，无额外修改；(2) Self-Attention利用CSP的ResolutionOffset按resolution分组——每个resolution group内patch重组为full feature map后通过xformers执行batched attention；(3) Convolution通过Patch Edge Stitcher（fused GroupNorm + boundary stitching CUDA kernel）处理跨patch边界。此operator taxonomy的设计使MixFusion能在保留generation quality的前提下实现patch-level parallelism——仅在必要时引入cross-patch context（Convolution的PES, Self-Attention的per-resolution reconstruction），其他计算完全并行。

涉及论文标题：
- MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models

术语是什么？

Text-to-Image Similarity-Based Cache Retrieval是MoDM提出的扩散模型缓存检索方法：用CLIP text encoder提取query prompt的text embedding，与cached image的CLIP image embedding做cosine similarity匹配，选取最高相似度的图像返回。与Nirvana的text-to-text similarity retrieval（比较prompt text embedding与cached prompt text embedding）相比，text-to-image retrieval直接度量图文之间的语义-视觉对齐，更好地匹配用户意图中的style、structure和content。

从算法pipeline角度拆解术语：

Text-to-Image Cache Retrieval算法流程：

```
// 离线: 为cache中每张图像预计算CLIP image embedding
for each cached image I_j:
    e_{I_j} = CLIP_ImageEncoder(I_j)           // 512维embedding

// 在线检索:
Function RetrieveCacheImage(query_prompt, cache, threshold_tau):
    q = CLIP_TextEncoder(query_prompt)          // 512维text embedding
    best_sim = -inf, best_image = None
    
    for each (I_j, e_{I_j}) in cache:          // GPU并行矩阵乘法
        sim = cosine_similarity(q, e_{I_j})     // q·e_{I_j} / (||q|| x ||e_{I_j}||)
        if sim > best_sim:
            best_sim = sim; best_image = I_j
    
    if best_sim >= threshold_tau:
        return (best_image, best_sim)           // cache hit
    else:
        return None                             // cache miss
```

检索性能：100K图像embedding存储仅0.29GB；GPU上cosine similarity计算0.05s/100K张；retrieval latency远小于denoising（>10s），不构成瓶颈。相似度阈值τ（0.25-0.30）虽低于Nirvana的text-to-text threshold（0.65-0.95），但因CLIP score本身捕捉图文semantic alignment，更低阈值仍能保证更好的视觉匹配（Fig.2验证：text-to-image mean CLIPScore 0.28 vs text-to-text 0.22）。

术语一般如何实现？如何使用？

MoDM使用OpenAI CLIP ViT-L/14模型提取embedding。image encoder部署在Request Scheduler侧，每张新生成图像异步提取embedding。检索过程在GPU上实现为单次矩阵乘法（Q·K^T），利用了GPU的并行计算能力。MoDM证明text-to-image retrieval在CLIPScore和PickScore上均优于text-to-text retrieval，且避免使用CLIP做bias（同时用PickScore交叉验证）。

涉及论文标题：
- MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models

---

## Dynamic Denoising Step Selection / k-Selection（动态去噪步数选择）

术语是什么？

Dynamic Denoising Step Selection (k-Selection) 是MoDM提出的自适应去噪步数跳过策略：基于query prompt与retrieved cached image之间的text-image similarity score，动态决定skip多少步denoising。相似度越高表示检索图像与目标越接近→skip更多步（k更大）；相似度越低→skip更少步（k更小），留更多步做refine。k从离散集合K={5,10,15,20,25,30}中选择，总去噪步数T=50，k cap在30以防止生成图像与缓存图像过度相似。

从算法pipeline角度拆解术语：

k-Selection算法（offline calibration + online inference）：

```
// === Offline Calibration (Fig.5a) ===
1. 用大模型生成10000张图像+prompts作为候选cache (100K cached images)
2. 对每个k in {5,10,15,20,25,30}:
    for each (query_prompt, query_image) in 10000 samples:
        cached_img = RetrieveBestMatch(query_prompt, cache)  // text-to-image retrieval
        sim = CLIPScore(query_prompt, cached_img)
        refined_img = AddNoise(cached_img, t_k) + Denoise(small_model, T-k steps)
        quality = CLIPScore(query_prompt, refined_img)
        记录(sim, k, quality) tuple
3. 对每个k: 找到满足quality >= alpha * Q_full_gen 的最小sim (alpha=0.95)

// === Online k-Decision Heuristic (Fig.5b) ===
Function k_decision(similarity):
    if similarity >= 0.30: return 30
    if similarity >= 0.29: return 25
    if similarity >= 0.28: return 15
    if similarity >= 0.27: return 10
    if similarity >= 0.25: return 5
    return None  // cache miss, similarity < tau_min

// === Quality Constraint (Eq.5) ===
Q_cache-hit(k) >= alpha * Q_full-gen  // alpha=0.95
// 确保cache-hit生成质量 >= 全量大模型质量的95%
```

heuristic在1000个独立test prompts上的表现：平均CLIPScore 28.50 vs full large-model pipeline 28.59（达到99.7% baseline quality），超过95% quality retention目标。

术语一般如何实现？如何使用？

MoDM在Request Scheduler中实现k-Selection heuristic。Offline calibration用DiffusionDB数据集+大模型生成图像执行一次（cost较高但仅需一次），建立(similarity threshold, k) lookup table。Online inference时直接查表O(1)决定k值。alpha=0.95作为quality degradation factor可由系统部署者调节以在quality和throughput之间trade off。cache hit threshold tau也随k变化（更大的k要求更高similarity）。

涉及论文标题：
- MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models

---

## Noise Re-Introduction for Cached Image Refinement（缓存图像噪声重注入优化）

术语是什么？

Noise Re-Introduction是MoDM提出的将cached final image重新引入扩散去噪流程的机制：对检索到的缓存图像I*按扩散模型的noise schedule在timestep t_k处加噪，生成中间状态I_tilde = sigma_{t_k} * epsilon + (1-sigma_{t_k}) * I*（公式2），然后使用small model执行剩余的T-k步去噪。这与SDEdit和标准image-to-image diffusion使用相同的公式，但MoDM将其应用于serving cache pipeline：cached image提供high-level structure，noise添加variation，后续denoising将图像refine到匹配新prompt。

从算法pipeline角度拆解术语：

Noise Re-Introduction算法：

```
// 输入: cached image I*, target timestep t_k, diffusion model noise schedule sigmas[]
// 输出: noised intermediate latent I_tilde

Function ReIntroduceNoise(I*, t_k, sigmas):
    sigma_tk = sigmas[t_k]                      // 从noise schedule查表
    epsilon ~ N(0, I)                           // 标准高斯噪声
    I_tilde = sigma_tk * epsilon + (1 - sigma_tk) * I*  // 线性插值(Eq.2)

// 之后执行denoising:
    latent_tk = VAE.encode(I_tilde)             // 或直接在pixel space（取决于模型设计）
    for step = t_k-1, t_k-2, ..., 0:
        latent_step = diffusion_denoise_step(latent_{step+1}, step, prompt, small_model)
    output_image = VAE.decode(latent_0)
```

核心特性：(1) sigma_{t_k}控制噪声量——越大则越接近纯噪声（更多variation），越小则越接近原图（更多preservation）；(2) noise schedule sigmas[]由diffusion model预定义（通常为线性或cosine schedule）；(3) 不同k值对应不同的sigma_{t_k}，higher k → larger sigma → more variation，但起始于更好的prior（cached image）。

术语一般如何实现？如何使用？

MoDM直接使用扩散模型内置的noise schedule（Stable Diffusion系列使用DDPM schedule），在Request Scheduler中调用模型的scheduler.sigmas[t_k]获得sigma值。加噪操作是简单的element-wise linear interpolation，计算成本极低（可忽略 vs denoising cost）。与从头生成（从纯噪声t=T开始）相比，从t_k加噪再refine的compute savings来自两方面：跳过k步denoising + 用小模型替代大模型执行剩余T-k步。

涉及论文标题：
- MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models

## 2-bit KV Cache Quantization（2比特KV缓存量化）

术语是什么？通过联网搜索让回答具体和精准。

2-bit KV Cache Quantization是将长上下文LLM推理中的键值（Key-Value）缓存激活值量化到2位精度（即每个值用2比特表示，共4个离散值）的技术。在transformer decoder的自回归推理中，KV cache随序列长度线性增长，成为memory和bandwidth瓶颈。2-bit量化将每个FP16值（16位）映射到2位整数（{0,1,2,3}），提供理论8×压缩比。然而，2位精度下可表示值极少，outlier channel的极值会显著放大同组内其他非outlier值的量化误差。JanusQuant论文中将2-bit分组量化group size设为g=32，配合per-token smoothing transformation控制outlier影响。与4-bit量化（16个离散值，更鲁棒但对超长序列仍显不足）和KV cache selection/eviction（丢弃token有精度损失）不同，2-bit量化追求"保留所有token但用极低精度存储"的路线。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

2-bit KV cache quantization在长上下文LLM推理算法pipeline中的执行流程（以JanusQuant解码阶段为例）：

```
// === 2-bit 量化参数计算（per-group） ===
// group_size g = 32, num_bits n = 2, num_levels = 2^n - 1 = 3
for each quantization group in KV_cache:
    // Step 1: 计算scale和zero_point（公式2）
    s = (max(group_values) - min(group_values)) / 3
    z = min(group_values)
    
    // Step 2: 量化到2-bit整数（公式3）
    for each value v in group:
        q = clamp(round((v - z) / s), 0, 3)  // 2-bit: 仅4个离散值{0,1,2,3}
        store q as INT2 (2 bits in packed format)

    // Step 3: 解量化恢复（公式4，仅在attention前执行）
    for each quantized value q:
        v_hat = q * s + z  // 近似的FP16值
```

2-bit量化的特殊性：每组仅4个表示值（0→z, 1→s+z, 2→2s+z, 3→3s+z），量化scale s = (max-min)/3 由组内极值完全决定。若组内存在outlier channel的值远超其他channel，s会异常大→非outlier值的量化粒度极粗→MSE激增。JanusQuant中：Atom-2bit直接per-token group quantization→K cache MSE 1.0352；QServe-2bit加入per-channel smoothing→MSE 0.5552；理想情况（移除最极端outlier channel后量化）→MSE 0.3734。这解释了为何2-bit需要配套smoothing/outlier管理。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实践中，2-bit KV cache量化需要：① 量化粒度选择——per-channel（对K cache，outlier沿channel集中）或per-token（对V cache，无明显outlier pattern）；② group size trade-off——更小的g提高精度但增加scale/zero_point参数存储（每g个值存一对FP16参数），JanusQuant选g=32使2-bit uniform quantization平均bit-width约3.008（含参数）；③ outlier管理策略——或分离存储outlier（KVQuant dense-and-sparse）、或预保留recent token为FP16（KIVI/SKVQ）、或smoothing transformation（JanusQuant RtSmooth）。2-bit inferencing需要custom kernel支持INT2 unpacking和fused dequantization-attention。典型pipeline：prefill阶段仍可用full-precision KV token做attention保accuracy，decoding阶段逐步将历史KV token量化到2-bit、recent token保持FP16、需要时通过fused kernel解量化参与attention。

涉及论文标题：
- JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference

## RtSmooth（Runtime Smoothing Quantization Algorithm / 运行时平滑量化算法）

术语是什么？通过联网搜索让回答具体和精准。

RtSmooth是JanusQuant提出的运行时per-token smoothing量化算法，核心思想是在量化前对K cache执行per-token平滑变换（smoothing transformation），动态缩小每个token内值的范围，降低2-bit分组量化的scale因子，从而减少outlier channel对同组其他值的量化误差放大。与SmoothQuant（离线校准per-channel smoothing factor并融入前层权重）不同，RtSmooth在推理时为每个请求、每个decoding step动态计算smoothing factor，适应不同输入和序列长度下KV cache分布的变化。K cache使用per-token smoothing + per-channel group quantization，V cache（无显著outlier）使用per-token group quantization。Decoding中每g=32步将FP16缓冲的recent tokens批量量化。平滑因子计算为max(|K_i|)^λ，λ=0.5由经验选定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

RtSmooth在2-bit KV cache量化pipeline中的执行流程：

```
// === JanusQuant Decoding Step（RtSmooth核心路径）===
// 输入：新生成的token t，其K_t, V_t ∈ R^{hidden_dim}
// 常量：lambda = 0.5, group_size g = 32, recent_buffer 容量 = n*g

// Step 1: 将新token写入ring buffer（FP16精度）
recent_buffer[write_ptr] = (K_t, V_t)
write_ptr = (write_ptr + 1) % buffer_size

// Step 2: 若当前segment满g个token，触发量化
if tokens_since_last_quantize == g:
    quantize_segment = recent_buffer[oldest_segment_start : oldest_segment_start + g]
    
    // Step 2a: 对K cache执行RtSmooth + per-channel group quantization
    for each token K_i in quantize_segment:
        candidate_channels = FAVP_channel_set[layer_id]  // <2% of total channels
        absmax_i = max(|K_i[ch]| for ch in candidate_channels)
        
        // RtSmooth smoothing factor
        gamma_i = absmax_i ^ lambda  // lambda = 0.5
        if gamma_i == 0: gamma_i = 1.0
        
        // Per-token smoothing transformation
        K_i_smooth = K_i / gamma_i  // 缩放整行 → 缩小组内值域

    // Step 2b: 对smoothed K执行per-channel group quantization
    for each channel_group in K_smooth:
        for each group of g consecutive tokens within channel_group:
            s = (max(group) - min(group)) / 3
            z = min(group)
            K_quantized[group] = clamp(round((group - z) / s), 0, 3)
            store(K_quantized, s, z, gamma_i)

    // Step 2c: 对V cache执行per-token group quantization（无smoothing）
    for each token_group of g tokens in V_segment:
        for each value_group of g consecutive values within token:
            s = (max(value_group) - min(value_group)) / 3
            z = min(value_group)
            V_quantized[value_group] = clamp(round((value_group - z) / s), 0, 3)
            store(V_quantized, s, z)

    // Step 2d: 追加量化后的segment到历史KV cache
    history_KV_cache.append(quantize_segment_quantized)
    oldest_segment_start = (oldest_segment_start + g) % buffer_size

// Step 3: Fused Mixed-Precision Attention
output_token = fused_mixed_precision_attention(Q_t, history_KV_cache, recent_buffer)
```

核心特性：(1) smoothing因子从absmax导出而非全量统计，FAVP将其扫描成本降至O(0.02×hidden_dim)；(2) per-channel group quantization使outlier effects沿channel局部化，不跨token传播；(3) smoothing scale缩小后，误差上界从s/2降至s_smooth/2（公式6）；(4) 量化频率为每g步一次，而非每步一次，amortize量化开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

RtSmooth实现约为3500行CUDA/C++中的fused quantization kernel。该kernel在单一CUDA kernel中完成：扫描FAVP channel→计算absmax→生成smoothing factor→per-token smoothing→scale/zero_point计算→INT2 packing→参数重排(unified parameter block)。Smoothing factor gamma_i与scale/zero_point一起存入统一参数块供后续mixed-precision attention kernel使用。与SmoothQuant的offline calibration融合权重不同，RtSmooth的smoothing在运行时执行，因此需dequantization时做逆smoothing（K_hat = K_quantized_deq × gamma_i），这一操作在fused attention kernel内完成。Decoding开始时新KV token不做smoothing直接存入FP16 ring buffer，待segment达到g后统一量化。

涉及论文标题：
- JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference

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

## Outlier Channel in KV Cache（KV缓存中的离群通道）

术语是什么？通过联网搜索让回答具体和精准。

在transformer模型KV cache中，outlier channel（离群通道）指K cache某些hidden dimension channel中值幅度显著大于其他channel的现象。论文Figure 2展示了Llama2-13B的K cache各通道值幅度热力图：少数channel呈现强outlier特征（值可达正常channel的6倍以上），而V cache无此模式。这种channel-wise outlier集中性来自attention层中K投影权重和输入hidden states的特定交互。在2-bit分组量化中，当per-token group quantization将不同channel的值混在同一group内时，outlier channel的极端max值会扩大整个group的量化scale s = (max-min)/3，使同组非outlier channel值被粗糙量化（仅4个量化bin）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Outlier channel在2-bit量化中的量化误差放大机制：

```
// 假设per-token group quantization: 每token内g=4个连续值为一组
// 组内包含3个正常channel值 + 1个outlier channel值

// 正常值范围: [0.5, 2.0]，可被2-bit精细量化
normal_values = [0.5, 1.2, 0.8]

// Outlier channel值
outlier = 15.0

// 混合后group = [0.5, 1.2, 0.8, 15.0]
group = normal_values + [outlier]

// 量化参数被outlier主导
s = (15.0 - 0.5) / 3 ≈ 4.833
z = 0.5

// 2-bit量化表示
quantized_levels = {0→0.5, 1→5.333, 2→10.167, 3→15.0}

// 正常值0.5→0 (精确=0.5)，1.2→0 (恢复=0.5, error=-0.7)，0.8→0 (恢复=0.5, error=-0.3)
// 所有正常值被映射到同一量化level 0，完全丢失区分度
// MSE ≈ (0² + 0.7² + 0.3² + 0²)/4 = 0.145 per element
```

对应三种缓解策略：① Per-channel group quantization——沿channel维分组，同一group内各值来自不同token的同一channel，避免outlier channel与非outlier channel混合（JanusQuant在RtSmooth后对K cache采用）；② Per-token smoothing——缩小每个token内的值域→outlier channel值被smoothing factor压缩（RtSmooth核心）；③ Dense-and-sparse分离——检测outlier值并存入稀疏high-precision buffer，剩余值量化（KVQuant方法）。JanusQuant采用①②组合，避免③的稀疏路径额外overhead。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实践中outlier channel的识别和处理策略选择取决于calibration分析和目标精度。JanusQuant的FAVP离线校准用128个WikiText2 8K样本分析每层的absmax channel分布，识别出每层最可能持有absmax的稀疏channel集（>90%层涉及<2% channels）。RtSmooth通过per-token smoothing间接处理outlier：不直接检测或抽取outlier值，而是用smoothing factor = max(|K_i|)^0.5缩放整个token→缩小所有channel值域→降低outlier对group scale的影响。这种方法比explicit outlier detection/extraction更快（消除memory-bound的outlier handling步骤），参考论文中SKVQ attention layer breakdown：outlier handling在prefill占~20% runtime、decode占~2% runtime。

涉及论文标题：
- JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference

## Sparse Attention Patterns（稀疏注意力模式）：Atomic & Compound

术语是什么？通过联网搜索让回答具体和精准。

Sparse Attention Patterns是Transformer模型中通过mask layer引入的注意力稀疏化模式，用于减少MHA (Multi-Head Attention) 中Q×K^T score matrix的计算量。STOF论文系统分类了4种Atomic（原子）pattern和2种Compound（复合）pattern。Atomic patterns是基础构建块：(a) Causal Attention（因果注意力）——token只能attend到之前的token，mask矩阵呈下三角，sparsity=50%；(b) Global Attention（全局注意力）——某些"global"节点接收所有token信息（对应行全有效）并发送给所有token（对应列全有效），sparsity取决于global节点占比；(c) Sliding Window Attention（滑动窗口注意力）——每个query仅关注窗口大小w内的邻近token，mask矩阵呈带状（banded pattern），w=32时sparsity=93.8%；(d) Random Attention（随机注意力）——query随机关联前后token，通过filling rate控制密度。Compound patterns由atomic patterns组合而成：(e) Longformer——sliding window + global attention组合，w=32时sparsity=88.8%；(f) Bigbird——sliding window + global + random三组合，w=32, filling rate=10%时sparsity=80.8%。不同pattern在element distribution上存在关键差异：causal/sliding window的元素分布在row和column上均连续（structured sparsity），Longformer元素discrete但column仍结构化，Bigbird由于random pattern引入unstructured sparsity。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Sparse attention pattern在Transformer推理MHA计算中的位置和计算流程：

```
// ===== MHA with Sparse Mask in Transformer Inference =====
// Input: Q, K, V ∈ R^{seq_len × head_dim}
// Mask M ∈ {0, -∞}^{seq_len × seq_len}  (sparse pattern defined)
// 
// Step 1: QK^T GEMM
S = Q × K^T          // S ∈ R^{seq_len × seq_len}, O(seq_len²) compute
//
// Step 2: Scale
S = S / sqrt(d_k)    // d_k = head_dim
//
// Step 3: Apply Sparse Mask  ← 关键步骤，不同pattern差异在此
if mask[i][j] == 1:   // valid attention
    S[i][j] = S[i][j]
else:                  // masked out
    S[i][j] = -inf     // after softmax: attention weight ≈ 0
//
// Step 4: Softmax (row-wise)
P = softmax(S)        // P[i][j] ≈ 0 for masked positions
//
// Step 5: PV GEMM
O = P × V             // O ∈ R^{seq_len × head_dim}
```

不同pattern的mask示例（seq_len=8）：
```
// Causal mask (sparsity 50%):
//   0 1 2 3 4 5 6 7
// 0 1 0 0 0 0 0 0 0
// 1 1 1 0 0 0 0 0 0
// ...

// Sliding window (band width w=2, sparsity ~75%):
//   0 1 2 3 4 5 6 7
// 0 1 1 0 0 0 0 0 0
// 1 1 1 1 0 0 0 0 0
// ...

// Bigbird (sliding w=2 + global col 0/row 0 + random):
//   0 1 2 3 4 5 6 7
// 0 1 1 1 1 1 1 1 1   ← global row
// 1 1 1 1 0 1 0 1 0   ← sliding + random
// ...
```

关键性质：mask引入的sparsity使大量S[i,j]计算可以跳过——这是STOF论文优化的核心opportunity。但不同pattern的sparsity structure（连续vs离散、structured vs unstructured）要求不同的数据结构和kernel实现策略。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实践中，sparse attention pattern通过以下方式实现：(1) Mask矩阵直接定义——PyTorch等框架中mask作为binary tensor传入attention函数，最通用但最低效（仍需完整QK^T GEMM）；(2) Fused kernel直接利用sparsity——FA2支持causal mask作为内置参数，FlashMask扩展支持column-continuous mask，STOF用two-level BSR+bitmap格式表示任意mask；(3) Compound pattern分解——Bigbird等可通过多个atomic kernel组合实现（sliding window kernel + global kernel + random kernel，输出scatter-add合并）。

涉及论文标题：
- Accelerating Sparse Transformer Inference on GPU

## Relevance Scoring and Aggregation（相关性评分与聚合 / 统一注意力抽象）

术语是什么？通过联网搜索让回答具体和精准。

Relevance Scoring and Aggregation 是 MetaAttention 提出的统一注意力机制抽象。它将所有 attention 变体分解为两个不可变的核心操作：(1) Relevance Scoring——计算输入 token 之间的成对相似度或相关性权重，通常通过 Q 与 K 的内积、或其他相似度度量实现；(2) Aggregation——利用 relevance scores 将上下文信息整合为每个 token 的输出表示，通常通过 scores 与 V 的矩阵乘法实现。这两个操作捕捉了所有 attention 机制的共同本质：计算 token 间关系（relevance scoring）并用该关系加权聚合信息（aggregation）。围绕这两个固定操作，MetaAttention 暴露可定制函数（Mod/RowNorm）和输入 shape 配置，以表达 softmax、sigmoid、linear、sparse、recurrent 等多种变体。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 MetaAttention 框架中，Relevance Scoring 和 Aggregation 根据 attention 属于 Parallel Pattern 还是 Recurrent Pattern 有不同的实例化：

**Parallel Pattern**（全局 K/V context）：
```
// relevance_scoring: Q 与整个 K 序列做矩阵乘法
relevance = matmul(Q[i], K)    // Q[i]: [head_dim_qk], K: [seq_len_kv, head_dim_qk]
                                 // 输出 relevance: [seq_len_kv]，表示 token i 对所有 KV token 的相关性

// aggregate: 用 relevance 加权聚合 V
state = matmul(relevance, V)   // relevance: [seq_len_kv], V: [seq_len_kv, head_dim_v]
                                 // 输出 state: [head_dim_v]
```
实例：标准 Softmax Attention 中，relevance = QK^T/√d_k，aggregation = softmax(relevance) × V。RetNet 中，relevance = QK^T，但 normalization 从 softmax 替换为基于 reduceAbsSum 的 RowNorm。

**Recurrent Pattern**（迭代维护压缩 hidden state）：
```
// relevance_scoring: Q 与当前压缩 state 做矩阵乘法
output = matmul(Q[i], H)       // Q[i]: [head_dim_qk], H: [head_dim_qk, head_dim_v]
                                 // 输出 output: [head_dim_v]

// aggregate: 用当前 K[i]、V[i] 更新压缩 state
H = H + matmul(K[i], V[i])    // K[i]: [head_dim_qk, 1], V[i]: [1, head_dim_v]
                                 // H 累积历史 KV 信息为固定大小矩阵
```
实例：Mamba2 SSM 中，H 对应 SSM 的 hidden state，relevance scoring 通过 Q 与 H 的乘积输出当前 token 表示，aggregation 通过 K 和 V 的外积更新 H。RetNet Recurrent 中类似，H 为压缩的 retention state。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 MetaAttention 实现（7.3k 行 C++/Python）中，Relevance Scoring 和 Aggregation 作为固定计算被硬编码在 attention template 中。用户无需实现这两个操作，而是：(1) 选择 Parallel 或 Recurrent Pattern 确定其计算方式；(2) 通过 customizable functions（Mod、RowNorm、RowNorm online）修改 relevance scores 的数值变换（如 mask、scaling、normalization）；(3) 声明 Q/K/V 的 shape 以支持非标准维度（如 dimqk ≠ dimv）。MetaAttention runtime 自动将这两个核心操作映射到硬件高效实现：Parallel Pattern 中 relevance scoring 使用 Tensor Cores MMA 计算 QK^T，Aggregation 同样使用 MMA 计算 scores × V；Recurrent Pattern 中 state 维护在 on-chip memory，通过 chunk parallelism 并行处理序列块。这种抽象确保框架在不牺牲性能的前提下覆盖 Softmax Attention、Sigmoid Attention、ReLU Attention、RetNet、Mamba2、MLA、Sparse GQA 等 10+ 种 attention 变体。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends

## Attention Parallel Pattern and Recurrent Pattern（注意力并行模式与递归模式）

术语是什么？通过联网搜索让回答具体和精准。

Attention Parallel Pattern 和 Recurrent Pattern 是 MetaAttention 将统一 attention 模板实例化得到的两种计算模式，覆盖所有主流 attention 变体。Parallel Pattern 对应需要全局 K/V context 的 attention，relevance scoring 和 aggregation 以并行矩阵乘法方式在整个 K/V 序列上执行。Recurrent Pattern 对应可以将 context 压缩到固定大小 hidden state 的 attention，relevance scoring 和 aggregation 以迭代方式逐 token 更新和查询 hidden state。两种 pattern 共享相同的编程接口（input shapes + customizable functions），但底层数据流和优化策略不同。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

**Parallel Pattern** 数据流：
```
输入: Q[batch, head, seq_len, dimqk], K[batch, head, seq_len_kv, dimqk], V[batch, head, seq_len_kv, dimv]
阶段1 (Relevance Scoring): scores = matmul(Q, K^T)          // [batch, head, seq_len, seq_len_kv]
阶段2 (Customizable Mod):    scores = scores_mod(scores)      // 如 causal mask, scaling
阶段3 (Customizable RowNorm): scores = scores_rownorm(scores) // 如 softmax, RetNet norm
阶段4 (Aggregation):         output = matmul(scores, V)       // [batch, head, seq_len, dimv]
阶段5 (Customizable Output Mod): output = output_mod(output)
```
覆盖：Softmax Attention、Sigmoid Attention、ReLU Attention、RetNet Parallel、Sparse GQA、Multi-head Latent Attention (MLA，query seqlen=1 解码场景)。

**Recurrent Pattern** 数据流：
```
初始化: H = zeros[head_dim_qk, head_dim_v]  // 压缩 hidden state
对于 i = 1 到 seq_len:
    // relevance scoring: 用 Q[i] 查询压缩 state
    output[i] = matmul(Q_mod(Q[i]), H)       // [head_dim_v]
    // aggregation: 用 K[i], V[i] 更新压缩 state
    H = H + matmul(K_mod(K[i]), V_mod(V[i]))
    H = h_mod(H)                              // 可选 state 变换
```
覆盖：Mamba2 SSM、RetNet Recurrent、YOCO Gated Retention、RFA-Big。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 MetaAttention 中，两种 pattern 在 attention runtime 中以不同 kernel template 实现。Parallel Pattern kernel 使用类似 FlashAttention 的 online tiling 策略：沿 KV sequence 维度分 tile，在每个 tile 内计算局部 relevance scores → 应用 customizable functions → 使用 RowNorm online 更新全局归一化状态 → 聚合 V → 移动到下一个 KV tile。Recurrent Pattern kernel 使用 chunk parallelism：将长序列沿 sequence 维度切为多个 chunk，每个 chunk 内维护本地 recurrent state，chunk 间传递 state，chunk 内 elementwise/reduction 逻辑融合到 single fused kernel。用户通过 programming interface 声明 attention pattern（`pattern: Parallel` 或 `pattern: Recurrent`），MetaAttention 自动选择对应 kernel template。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends

## Multi-head Latent Attention (MLA / 多头潜在注意力)

术语是什么？通过联网搜索让回答具体和精准。

Multi-head Latent Attention (MLA) 是 DeepSeek-V2/V3 引入的一种 attention 机制创新。其核心思想是将传统 Multi-Head Attention (MHA) 的 Key-Value (KV) cache 压缩为低维 latent vector，通过低秩分解在两个阶段工作：(1) Latent Space Encoding：将 K/V 投影到低维 latent 空间（如 d_model=5120 压缩到 latent_dim=512，约 10× 压缩），仅存储压缩后的 latent vector 而非每个 head 的完整 K/V；(2) Dynamic Decoding：注意力计算时，从 latent vector 动态上投影恢复各 head 的 K/V 表示。MLA 通过矩阵乘法结合律将解压矩阵与 Q 投影权重融合（"matrix fusion trick"），避免推理时额外计算开销。对于需要 Rotary Position Embedding (RoPE) 的部分维度，MLA 采用混合设计——部分维度带 RoPE（跨 head 共享）、部分不带 RoPE（允许 fusion trick），两部分分别计算后通过 dot product 组合。MLA 在保持接近 MHA 表达能力的同时，将 KV cache 减少 87-92%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

MLA forward pass 伪代码：
```
输入: hidden_states [batch, seq_len, d_model]

// Stage 1: Latent Compression (Down-Projection)
c_KV = W_down_KV × hidden_states        // [batch, seq_len, latent_dim], 压缩 KV 到 latent 空间
c_Q  = W_down_Q  × hidden_states        // [batch, seq_len, latent_dim_Q]

// Stage 2: Up-Projection for K and V
K = W_up_K × c_KV                        // [batch, head, seq_len, dimqk]
V = W_up_V × c_KV                        // [batch, head, seq_len, dimv]

// Stage 3: RoPE Handling (hybrid design)
K_rope = RoPE(K[:, :, :d_rope])          // 部分维度带 RoPE，跨 head 共享
K_nope = K[:, :, d_rope:]                // 其余维度不带 RoPE
K_final = concat(K_rope, K_nope)

// Stage 4: Q with Matrix Fusion Trick
// 将 W_up_Q 与 W_up_K 通过结合律预融合，避免显式 up-project K
Q = W_up_Q × c_Q                         // 融合后的等效 Q

// Stage 5: Standard Attention (in MetaAttention terms: Parallel Pattern)
scores = Q × K_final^T                   // relevance scoring
scores = softmax(scores / sqrt(dimqk))   // RowNorm
output = scores × V                      // aggregation
// dimqk ≠ dimv (如 DeepSeek-V3: dimqk=576, dimv=512)
```

MLA 的关键特征：(1) KV cache 仅存 c_KV（latent vector，如 512 维），每个 head 不再存独立 K/V；(2) dimqk 和 dimv 通常不相等（如 dimqk=576, dimv=512）；(3) head 数远大于 head_kv（如 head=128, head_kv=1）；(4) query seqlen=1 的解码场景（LLM inference）下，Q 只有一个 token，K/V 来自 KV cache。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MLA 的优化实现有多个版本：(1) FlashMLA（DeepSeek 官方）——约 1.7k 行 CUDA，专门针对 H100/H800 的 MLA 解码 kernel，使用 blockSize=64，利用 TMA 异步加载和 Tensor Cores MMA，支持 dimqk≠dimv 的非标准 shape；(2) MLA Triton kernel——Triton 实现的 MLA forward kernel，性能低于 FlashMLA；(3) MetaAttention 中的 MLA 支持——约 90 行代码，通过 Parallel Pattern + 自定义 Q/K/V shape（head=128, head_kv=1, dimqk=576, dimv=512, query seqlen=1）+ customizable functions 表达 MLA，性能接近 FlashMLA（within comparable），且自动受益于 MetaAttention 的跨后端支持（NVIDIA/AMD）。开源：FlashMLA https://github.com/deepseek-ai/FlashMLA。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends

---

## ControlNet（条件控制网络）

术语是什么？通过联网搜索让回答具体和精准。

ControlNet是由Lvmin Zhang等人在ICCV 2023提出的扩散模型条件控制扩展架构。它通过复制预训练扩散模型（如Stable Diffusion）的encoder层并注入额外条件输入（如Canny edge maps、pose skeletons、depth maps、segmentation masks等），实现对生成图像内容的精确空间控制，而无需从头训练或微调原始模型。ControlNet的核心设计是"zero convolution"（零初始化卷积层），在训练初期输出为零以保持原始模型行为不变，逐步学习条件控制信号。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

ControlNet在扩散pipeline中的运转流程：

```
// ControlNet block（添加到U-Net的每个encoder layer）
1: x ← UNet_encoder_block(latent, timestep_embedding)    // 原始U-Net encoder输出
2: c ← ControlNet_encoder_block(latent, timestep_embedding, control_input)
   // ControlNet复制原始encoder结构+额外control input通道
3: c_transformed ← zero_convolution(c)                     // 1×1 zero-initialized conv
4: output ← x + c_transformed                              // 残差加法注入控制信号
```

Diffusion pipeline中的ControlNet使用：
```
5. 输入：prompt ("a dog") + control_image (Canny edge map)
6. CLIP text encoder → text_embedding
7. ControlNet encoder → extract control features from control_image at multiple resolutions
8. VAE encoder → latent (if image-to-image)
9. for t in denoising_steps:
10.    U-Net(latent_t, t, text_embedding) + ControlNet(control_features) → predicted_noise
11.    latent_{t-1} = denoise_step(latent_t, predicted_noise, t)
12. VAE decoder → output image
```

ControlNet inputs属于Difflow论文分析的扩散pipeline多种输入之一。Difflow评估的edit应用同时使用ControlNet（Canny edge spatial control）+ LoRA（style adaptation）。Symbolic property analysis发现ControlNet inputs共享相同的优化条件（同时冗余或同时非冗余），因此可将多个ControlNet inputs在属性条件枚举时合并为一个，指数级减少dEngine数量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

ControlNet开源实现：https://github.com/lllyasviel/ControlNet（基于Stable Diffusion 1.5）和Diffusers库中的ControlNetModel类。使用时加载预训练的Stable Diffusion backbone + 对应control type的ControlNet权重（如canny、depth、pose、scribble等）。支持multi-ControlNet（同时使用多个control信号）。在Difflow中，启用ControlNet会使U-Net输入从4个激增至14个，通过symbolic analysis将ControlNet inputs视为共享优化条件的一个整体避免了2^14=16384 engines的组合爆炸。

涉及论文标题：
- Difflow: A Data-Characteristic-Aware Serving System for Diffusion Models

---

## LoRA (Low-Rank Adaptation)（低秩适配）

术语是什么？通过联网搜索让回答具体和精准。

LoRA (Low-Rank Adaptation) 由Hu等人(2021)提出，通过在预训练权重旁添加低秩分解矩阵（W' = W + ΔW = W + BA，其中B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k)）实现参数高效微调。核心思想：模型权重更新ΔW实际具有内在低秩属性，大模型的自适应变化有效自由度远低于全参数空间维度。训练时W₀冻结，仅更新A（Kaiming初始化）和B（zero初始化），ΔW = BA在训练起始为零。训练后可将ΔW与W₀合并（W' = W₀ + BA）使推理时无额外延迟。LoRA具有"plug-and-play"特性：LoRA模块可独立于base model存储和重用，支持灵活的任务间切换，切换开销极低（Llama3-1B上<1ms）。

在**LLM端云协同推理**（TailorLLM）中：LoRA使端侧SLM（Llama3-1B）通过加载task-specific低秩矩阵即可在特定高频任务上达到接近cloud LLM（Llama3-70B）的精度，减少云端调用。每个adapter约22MB（r=16），可选择性应用到特定网络模块（如Q/K attention matrices）。TailorLLM通过RFLoRA进一步压缩adapter到~11.56MB。

在**扩散模型**（Difflow）中：LoRA被广泛用于注入特定视觉风格、角色或概念到预训练扩散模型中——每个LoRA仅adapt attention layers的Q/K/V/O projection权重，一个LoRA通常仅几MB（r=4-32），与base model独立加载/卸载。Difflow评估的edit应用使用了16种不同的LoRA weights作为style variants。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

**LLM LoRA adapter 切换与推理**（TailorLLM）：
```
// LoRA-augmented linear layer: y = W₀x + BAx
// W₀ ∈ R^{d×k} frozen pretrained weight
// A ∈ R^{r×k}, B ∈ R^{d×r}, r << min(d,k)

// 训练阶段（云端RTX 3090）:
1: 对每个下游任务task_i的训练数据:
2:     A ~ Kaiming初始化   // 通常跨任务可共享
3:     B = zeros(d, r)     // zero初始化
4:     for each training step:
5:         ΔW = B @ A
6:         W' = W₀ + ΔW   // 仅A, B有梯度
7:         loss = CrossEntropy(model(x), y)
8:         B.grad, A.grad ← backward(loss)
9:         更新B, A

// 端侧推理切换（Tesla T4，<1ms）:
10: 加载task_i对应的B_i矩阵 → 与预存A合并
11: for each token in autoregressive decode:
12:     hidden = W₀ @ x + B_i @ (A @ x)
```

**扩散模型LoRA**（Difflow）：

LoRA在扩散模型attention层的权重修改：

```
// Original attention linear layer: y = Wx, W ∈ R^{d×k}
// LoRA-augmented: y = Wx + BAx
// B ∈ R^{d×r}, A ∈ R^{r×k}, rank r << min(d,k)

// 在U-Net attention层中:
1: Q = W_Q @ latent + B_Q @ (A_Q @ latent)   // LoRA adapted Query
2: K = W_K @ latent + B_K @ (A_K @ latent)   // LoRA adapted Key
3: V = W_V @ latent + B_V @ (A_V @ latent)   // LoRA adapted Value
4: attention_output = softmax(Q @ K^T / sqrt(d)) @ V
5: O = W_O @ attention + B_O @ (A_O @ attention)
```

在Difflow的edit应用中使用16 LoRA styles的pipeline流程：
```
1. 固定 control_image (Canny edge) + prompt → CLIP + ControlNet features
2. 固定 latent_noise (相同初始噪声保证结构一致)
3. for each LoRA_i in 16 styles:
4.     load LoRA_i weights (merge BA into attention projections)
5.     denoise latent with LoRA_i → styled output image_i
```

由于16个LoRA variants共享相同的latent_noise和conditioning inputs，Difflow将U-Net分解为input-dependent dGraphs（per-request unique）和input-independent dGraphs（16 styles共享），后者识别为loop-invariant并通过multi-value compile-time caching实现precomputation。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LoRA权重在LLM领域通过HuggingFace PEFT库（`peft.LoraConfig`）配置和训练，支持指定target_modules（如q_proj, k_proj, v_proj, o_proj）、rank r和alpha scaling。训练后adapter以.bin/.safetensors保存B和A权重。推理时通过`model.load_adapter()`加载，切换开销极低（Llama3-1B<1ms），支持多adapter热切换。LoRA变体包括：DoRA（动态rank分配）、AdaLoRA（自适应layer-wise rank）、QLoRA（4-bit量化+LoRA）、HydraLoRA（非对称架构）、RFLoRA（冻结共享A+方向-幅度解耦）。开源实现：https://github.com/huggingface/peft。

在扩散模型领域，LoRA权重通常以.safetensors格式分发（如Civitai上的模型），通过Diffusers的`load_lora_weights()`加载。开源工具如kohya-ss/sd-scripts用于训练自定义LoRA。LoRA的应用产生大量具有相同base inputs但不同fine-tuning weights的correlative requests，Difflow利用invariant tensor elimination将input-independent计算在compile-time precompute（16 multi-value cached outputs），显著减少运行时计算。

涉及论文标题：
- TailorLLM: Collaborative End-Cloud Inference of Large and Small Language Models Based on Low-Rank Adaptation
- Difflow: A Data-Characteristic-Aware Serving System for Diffusion Models

---

## SpMM (Sparse Matrix-Matrix Multiplication)

术语是什么？

SpMM (Sparse Matrix-Matrix Multiplication) 是稀疏矩阵 A（通常表示图邻接矩阵或稀疏特征交互矩阵）与稠密矩阵 B（通常表示节点/特征嵌入矩阵）的乘法操作 C = A × B。SpMM 是科学计算、图分析、GNN（图神经网络）推理和推荐系统中的核心算子。与 SpMV（Sparse Matrix-Vector Multiplication）不同，SpMM 的右侧操作数是多列稠密矩阵，这使得 SpMM 具有更高的算术强度和更多数据重用机会，但也要求格式和 kernel 设计必须平衡稀疏 pattern 的 irregular access 与 dense tile 的 locality 利用。

从算法 pipeline 角度拆解术语：

GNN 推理中 SpMM 的 pipeline 角色（以 GCN 为例）：
```
// GCN Layer 的 SpMM 调用
输入: adjacency_matrix_A[n×n sparse], feature_matrix_W[n×d dense]
输出: C = A × W  // [n×d] dense output

// GCN 推理 pipeline
1. X = input_features                     // [n, d_in]
2. H = Linear(X, W_linear)                // [n, d_hidden]
3. H = SpMM(adjacency_A, H)               // 图卷积聚合：邻接矩阵×特征 → [n, d_hidden]
4. H = ReLU(H)                             // 非线性
5. output = Linear(H, W_output)           // [n, d_out]

// 在多层 GNN 中，SpMM 在每一层重复执行
```

SpMM 的核心计算特征：
```
C[i][j] = Σ_{k: A[i][k] ≠ 0} A[i][k] × B[k][j]
// 每个输出元素 C[i][j] 仅依赖 A 第 i 行的非零元素
// 同一行 i 的所有 j 列共享 A 的非零访问模式 → dense tile B 的数据重用机会
```

与相关算子的区别：
- **SpMV**：B 只有 1 列 → 无 dense tile 重用，算术强度低，主要为 memory-bound
- **SpMM**：B 有多列（如 GNN 中 d_hidden=64/128/256） → 每个非零 A[i][k] 参与 d 次乘法，数据重用的权衡取决于 d 的大小
- **GEMM**：A 和 B 都是 dense → 规则访存模式，可用高度优化的 BLAS kernel
- **SDDMM**：Sampled Dense-Dense Matrix Multiplication → 输出是 sparse，与 SpMM 方向相反

术语一般如何实现？如何使用？

CPU 上的 SpMM 实现策略：
1. **CSR-based SpMM**（如 ArmPL、Eigen、Cholmod）：基于 CSR 格式，对每个非零 A[i][k] 执行 k 行的 vectorized saxpy 更新 C[i][:] ← A[i][k] × B[k][:]。优点：格式简单；缺点：C 的随机写访问导致 cache miss，SME ZA 的 outer-product 能力未利用。
2. **ASM-SpMM 的 SME outer-product SpMM**：通过 OP-MCF 格式将稀疏矩阵转为与 SME vector length 对齐的 row window，每个 window 内用 SME MOPA outer-product 指令计算 sparse_vec ⊗ B_tile 直接累加到 ZA tile。关键优势：利用 predicate mask 消除 zero padding，outer-product 语义天然匹配稀疏计算（sparse vector 作为外积的一个操作数）。
3. **LOOPS 的 hybrid CSR+BCSR**：将矩阵分为 CSR 部分（NEON vector 处理）和 BCSR 部分（SME outer-product 处理），自适应分配避免 zero-propagation 在 SME outer-product 中的算力浪费。

GPU 上的 SpMM 实现策略：
4. **cuSPARSE SpMM**：NVIDIA 官方稀疏库，CSR/COO 等通用格式
5. **Tensor Core SpMM**（TCF、ME-TCF、DTC-LSH、TC-GNN）：将稀疏矩阵切为 fixed-size dense block（2×2/4×4），block 内有足够非零的用 Tensor Core 计算，其余 fallback 到 CUDA core。

典型 benchmark：SuiteSparse Matrix Collection（涵盖不同规模/形状/稀疏度的真实稀疏矩阵），GNN 图数据集（TC-GNN/SNAP/OGB/DGL 的图邻接矩阵）。B 列数通常评估 64/128/256/512/1024。

6. **Swift SpMM**：基于 CSC 格式的 GPU SpMM，通过 sparsity-based column sorting + B row rearrangement + warp-size blocking 实现 dual-input coalesced memory access。将稀疏矩阵按 warpSize=32 分为 regular block（warp 内线程处理连续 32 列→coalesced B 访问）和 irregular block（长列拆分均衡负载），regular kernel 用 segment sum 在 shared memory 中局部归约减少 atomicAdd。在 2757 个 SuiteSparse 矩阵上相对 ASpT/cuSPARSE/RoDe/Sputnik 在 FP64/N=128 下几何平均加速 1.79×/27.02×/3.62×/6.53×。

涉及论文标题：
- ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration
- Swift: High-Performance Sparse-Dense Matrix Multiplication on GPUs
- Uni-STC: Unified Sparse Tensor Core

## SpMV (Sparse Matrix-Vector Multiplication)

术语是什么？

Sparse Matrix-Vector Multiplication (SpMV) 是科学计算、图分析和机器学习中最基础的操作之一，定义为 y = α·A·x + β·y（或简写 y = A·x + y），其中 A 是稀疏矩阵（大部分元素为零），x 和 y 是密集向量。SpMV 是 memory-bound kernel——计算强度极低（每个非零元素仅 2 FLOPs：一次乘法 + 一次加法），性能瓶颈在内存带宽而非计算能力。由于矩阵 A 的稀疏模式高度不规则（非零分布不均、行间 NNZ 方差大），SpMV 在 GPU 上的高效实现极富挑战性：需要克服不规则访存（indirect indexing through column IDs）、行间负载不均衡、以及低计算密度导致的 memory latency 支配。

从算法pipeline角度拆解术语：

SpMV 的 CSR 格式基本算法流程：

```
Input: A (CSR format): row_ptr[N+1], col_idx[NNZ], values[NNZ]
       x: dense vector of size K
Output: y: dense vector of size N

for i in range(N):                      // 每个输出行
    y_i = 0
    for j in range(row_ptr[i], row_ptr[i+1]):  // 该行的非零元素范围
        col = col_idx[j]                 // 非零元素列索引
        val = values[j]                  // 非零元素值
        y_i += val * x[col]             // 乘累加
    y[i] = y_i
```

GPU 上 CSR SpMV 的并行化：将稀疏矩阵的行分配给 thread/warp/block 并行处理。常见策略为 (1) 一行多线程：每行分配多个 thread，每 thread 处理部分非零元素，warp 内用 shuffle/atomic 归约；(2) 多行一线程：每个 thread 处理多行（对短行矩阵有效）；(3) 2D 分解：将非零元素在 2D grid 上分布（如 merge-based SpMV）。

Tensor Core SpMV（Drawloom 方式）：将稀疏矩阵切分为 V-width row strip→填充到 TC block 的 A 矩阵（按列压缩去除零值列）→向量 X 被加载到 TC block 的 B 矩阵→TC MMA 输出沿对角线产生有效 Y 元素。这避免了传统 CSR 的逐非零迭代，通过 TC 硬件批量计算 row strip 内的乘累加。

术语一般如何实现？如何使用？

GPU SpMV 典型实现栈：
1. **cuSPARSE (NVIDIA)**：vendor 优化库，支持 CSR/CSC/BSR/SELL 等多种格式，运行时选最优格式
2. **Tensor Core SpMV**：DASP（m8n8k4固定TC shape）、Drawloom（ArbitWeave自适应任意TC shape+SpTC加速短行）、Spaden（bitmap格式TC）
3. **SuiteSparse Matrix Collection**：标准 benchmark 数据集，涵盖 >4000 个不同领域/规模/稀疏模式的矩阵。22 个代表矩阵（如 pwtk/circuit5M/webbase-1M/in-2004等）常用于论文性能评估
4. **Performance metrics**：GFlops/s（基于 NNZ 的理论 FLOPs = 2×NNZ），speedup over cuSPARSE

SpMV 在 LLM/科学计算中的使用场景：图神经网络（邻接矩阵乘特征向量）、物理仿真（FEM 刚度矩阵乘位移向量）、推荐系统（embedding 交互稀疏矩阵乘向量）。预处理（格式转换）只需执行一次，overhead 可在多次 solver 迭代中摊销。

涉及论文标题：
- Exploiting Efficient Mapping and Pipelined Execution for Accelerating SpMV on Tensor Cores
- Uni-STC: Unified Sparse Tensor Core

## SpMSpV (Sparse Matrix-Sparse Vector Multiplication)

术语是什么？

SpMSpV (Sparse Matrix-Sparse Vector Multiplication) 是计算 y = A·x 的线性代数原语，其中矩阵 A 和输入向量 x 都是稀疏的（仅少量非零元素）。与 SpMV (Sparse Matrix-Vector multiplication, x 为稠密) 不同，SpMSpV 的计算量由输入向量 x 的非零元数量决定而非矩阵大小，因此当 x 极度稀疏时 SpMSpV 比 SpMV 更高效。SpMSpV 是图分析 (BFS、PageRank、Personalized PageRank) 的核心原语，也是 GraphBLAS、Gunrock、GraphBLAST、GraphMat 等图计算框架的代数基础。此外，SpMSpV 在脉冲神经网络 (SNN) 的 event-driven spike propagation 中也有应用——spike delivery 可自然表示为稀疏矩阵-稀疏向量乘法。

从算法pipeline角度拆解术语：

SpMSpV 有两种执行范式（以矩阵 A 为稀疏矩阵、x 为稀疏向量）：

**Row-major (matrix-driven / CSR / pull) 范式：**
```
Input: A in CSR (row_ptr, indices, values) with N rows;
       dense vector x with value array and bitmask bm
Output: result vector y
1: for all r ← 0 to N in parallel do
2:     start ← row_ptr[r], end ← row_ptr[r+1]
3:     res ← 0
4:     for j ← start to end do
5:         col ← indices[j]
6:         if bm[col] then               // bitmask检查x[col]是否非零
7:             res ← res + values[j] × x[col]
8:     y[r] ← res
```
遍历矩阵所有行，通过 bitmask 跳过 x 的非活跃列。缺点：无论 x 稀疏度如何都遍历所有行，不能充分利用向量稀疏性。

**Column-major (vector-driven / CSC / push) 范式：**
```
Input: A in CSC (col_ptr, indices, values); x in sparse format (idx, val)
Output: result vector y
1: for all active entries i in x in parallel do
2:     col ← idx[i], v_val ← val[i]
3:     start ← col_ptr[col], end ← col_ptr[col+1]
4:     for j ← start to end do
5:         row ← indices[j], mat_val ← values[j]
6:         partial ← mat_val × v_val
7:         write_back(y[row], partial)   // scatter accumulation
```
仅遍历 x 的非零元，每个非零元索引 A 的对应列，计算 partial products 并 scatter 到输出向量 y。缺点：write-back 阶段的多对一 scatter 导致 conflict 和低带宽利用。

**Weighted vs Unweighted SpMSpV：** Weighted SpMSpV 中矩阵 A 和向量 x 均含一般权重（浮点值），要求 multiply-accumulate (乘加)。Unweighted SpMSpV（如 BFS-style）中值为二值，可用 atomicOr 或 output masking 优化，代表性工作包括 TileSpMSpV 和 BerryBees。VDHA 聚焦 Weighted SpMSpV。

术语一般如何实现？如何使用？

GPU 上的 SpMSpV 实现：
- **cuSPARSE**: NVIDIA 官方稀疏库提供 CSR-based SpMV，可通过添加 bitmask value validation 扩展到 SpMSpV
- **Gunrock**: GPU 图分析框架，使用 atomic-based column-major SpMSpV 内核
- **FastSpMSpV**: 提出 sort-reduce 方法避免 atomics，通过全局排序消除 write conflicts
- **Adaptive SpMSpV**: 根据矩阵特征和向量稀疏度在 8 个候选 kernel（row/col-major × atomic/sort × 不同负载均衡策略 + row-major SpMV fallback）间选择
- **VDHA**: 使用 shared-memory hash table 做 local aggregation 减少 global write conflicts，结合 column decomposition 增强 locality
- **HAM-SpMSpV** (CPU): 多核 CPU 上的 masked SpMSpV，使用 pre-bucketing 和 hash-based 算法
- 矩阵存储格式：row-major 使用 CSR (Compressed Sparse Row)，column-major 使用 CSC (Compressed Sparse Column)
- 负载均衡策略：Direct-mapped (直接映射)、Block-mapped (block 内分组)、Global-mapped (全局 prefix-scan 均匀分配)

涉及论文标题：
- VDHA: Vector-Driven Hash Aggregation for Sparse Matrix-Sparse Vector Multiplication on GPUs
- Uni-STC: Unified Sparse Tensor Core

## Mixed Precision Quantization（混合精度量化）

术语是什么？通过联网搜索让回答具体和精准。

Mixed Precision Quantization（混合精度量化）是一种对模型tensor中不同部分使用不同bit-width精度进行量化的技术。与uniform precision quantization（所有tensor使用同一精度如全INT4或全INT8）不同，混合精度量化利用LLM activation中的outlier sparsity property——少数值显著大于多数值——对error-sensitive的outlier区域分配更高精度（如INT8或FP16），对大多数normal值使用更低精度（如INT4），从而在保持模型准确率的同时最大化计算效率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

混合精度量化在LLM推理pipeline中的三种粒度（以4-bit W4A4场景为例）：

**(1) Tensor-wise Mixed Precision（粗粒度）**：
不同模型模块使用不同精度。如MxMoE中expert层用INT4、shared层用INT8。计算在每个模块内为uniform precision，易实现但与Tensor Core兼容性最好。

**(2) Channel-wise Mixed Precision（通道级，如MixQ/LLM.int8()）**：
```
Input: activation X [tokens, channels], outlier budget k_o
// 离线或在线识别channel-wise outliers
for c in 0..channels:
    max_vals[c] = max(|X[:, c]|)
O = topk_indices(max_vals, k_o)  // top-k outlier channels

// 混合精度GEMM沿reduction dimension分解
// Normal channels: INT4 weight × INT4 activation
C_normal = INT4_GEMM(W_normal, X_normal)
// Outlier channels: INT8/FP16 weight × INT8/FP16 activation  
C_outlier = INT8_GEMM(W_outlier, X_outlier)
// 结果相加
C = C_normal + C_outlier
```
优势：outlier channels沿reduction dimension，可自然分解matrix multiplication为两个独立dense GEMM，与Tensor Core完全兼容。

**(3) Token-wise Mixed Precision（令牌级，如RoMeo的RTMPQ）**：
```
Input: rotated activation X' [tokens, channels], outlier budget k_o
// Hadamard rotation先将channel outliers迁移到token维度
X' = X · H  // H为Hadamard矩阵

// 在线per-token outlier检测
for t in 0..tokens:
    max_vals[t] = max(|X'[t, :]|)
O_A = topk_indices(max_vals, k_o)  // top-k outlier tokens

// Token-wise mixed precision quantization
X'_Q = zeros_like(X')
for t in 0..tokens:
    if t in O_A:
        X'_Q[t, :] = INT8_quantize(X'[t, :])  // outlier token: INT8
    else:
        X'_Q[t, :] = INT4_quantize(X'[t, :])  // normal token: INT4

// Cross-precision GEMM（非归约维度混合精度）
// 四种精度组合: W4A4, W4A8, W8A4, W8A8
```
特征：token-wise outliers沿non-reduction dimension→sparse computation pattern→需要permutation-free等系统优化才能高效映射到GPU。

混合精度量化对LLM推理的关键trade-off：outlier比例越高→准确率越好但计算效率越低。RoMeo显示5% outlier tokens用INT8即可在Qwen3-8B上实现10.97 PPL（vs QuaRot 11.53），理论加速比公式：S = 1/(P_INT4/4 + (1-P_INT4)/2)，其中P_INT4为纯INT4计算比例。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：
- **Channel-wise**: MixQ使用静态offline calibration dataset分析per-channel max activation→离线识别outlier channels→serving时直接使用预计算outlier mask。LLM.int8()使用online per-channel outlier detection。
- **Token-wise**: RoMeo的RTMPQ使用online per-token row-max + top-k selection→fused Triton kernel实现。token-wise的outlier detection必须在线完成（outlier来自input sequence的语言特征，非静态模型特性）。
- **Group-wise**: Atom使用finer group-wise granularity（group size=128）→每group独立scaling factor→更高准确率但更多dequantization overhead。
- 对于weight矩阵，mixed precision可在offline完成（weight不随输入变化）。但RoMeo指出Hadamard旋转后weight也需mixed precision（H^T预乘amplify weight non-uniformity）。

涉及论文标题：
- RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

## Channel-wise Outlier / Token-wise Outlier（通道/令牌维度异常值）

术语是什么？通过联网搜索让回答具体和精准。

在LLM activation tensor中，outlier指那些显著大于其他值的极少数activation值。根据outlier在activation tensor（shape: [tokens, channels]）中的集中维度，分为两类：

- **Channel-wise Outlier (CO)**：outlier集中在特定channel（embedding position），即某些channel的所有token普遍具有更大的activation值。源于模型内部结构（特定FFN神经元输出）。模式相对稳定、可离线分析。
- **Token-wise Outlier (TO)**：outlier集中在特定token（输入序列中的特定词/短语），即某些token的所有channel普遍具有更大的activation值。源于输入语言特征（罕见词、特殊标点等）。模式动态、不可预测，需在线检测。

RoMeo首次明确指出dual-dimensional outliers是4-bit量化性能下降的根本原因：channel-wise方法（MixQ）移除top-256 outlier channels后max activation从1272降至110（8-bit足够），但4-bit下残余token-wise outliers仍导致严重量化误差。Hadamard rotation可将peak从1272降至58.5（channel-wise outlier被平滑），再通过token-wise mixed precision将残余TO移除后peak进一步降至18.6。

从算法pipeline角度拆解术语：

```
Activation Tensor X [M tokens, K channels]

Channel-wise Outlier检测（如MixQ）:
  max_per_channel = reduce_max(|X|, axis=0)  // shape: [K]
  outlier_channels = topk(max_per_channel, k_co)

Token-wise Outlier检测（如RoMeo，在Hadamard rotation后）:
  X_rotated = Hadamard_rotate(X)  // [M, K]
  max_per_token = reduce_max(|X_rotated|, axis=1)  // shape: [M]
  outlier_tokens = topk(max_per_token, k_to)

关键差异：
- CO沿channel维度（activation columns）→位于GEMM的reduction dimension (K)
  → 可分解为两个独立dense GEMM (W4A4 + W8A8)
- TO沿token维度（activation rows）→位于GEMM的non-reduction dimension (M)
  → 无法简单分解，产生sparse computation pattern
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Channel-wise outlier检测：使用offline calibration dataset（如WikiText的若干样本）运行模型→收集各layer activation→统计per-channel max值→topk选择outlier channels→serving时固定使用。MixQ进一步提出用小型predictor模型在线预测outlier channels以提高适应性。
- Token-wise outlier检测：必须在serving时在线执行。RoMeo使用fused Triton kernel：per-token row-max reduction（parallel reduction over K dimension）→topk selection→生成outlier mask。动态性来自token-wise outliers由输入prompt的语言特征决定（如特定罕见词触发某些attention pattern），无法离线预判。
- 双维度outlier处理策略：Hadamard rotation将CO平滑并迁移到token维度→两个维度的outlier统一为TO→仅需处理一种outlier类型。

涉及论文标题：
- RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

## Hadamard Rotation for LLM Quantization（Hadamard旋转LLM量化）

术语是什么？通过联网搜索让回答具体和精准。

Hadamard Rotation是在LLM weight-activation量化中使用Hadamard矩阵对activation进行正交变换以平滑outlier的技术。Hadamard矩阵H是元素仅为+1和-1的正交矩阵（H·H^T = I）。当H乘到activation矩阵X上时，它将每个channel的值重新分布到所有channel的线性组合中，从而平滑集中在特定channel的极端值（channel-wise outliers）。为保持数学等价性，weight矩阵需在offline左乘H^T。该技术由QuaRot首次提出用于4-bit uniform量化，RoMeo将其扩展为rotation+mixed precision的两阶段方案。

从算法pipeline角度拆解术语：

Hadamard rotation在LLM量化pipeline中的计算流程（以RoMeo为例）：

```
// 原始Linear层: Y = X · W
// 插入Hadamard rotation后: Y = (X·H) · (H^T·W)

Offline阶段（weight预处理）:
  W_rotated = H^T · W  // 离线完成，固定cost

Online阶段（每个token推理）:
  // Step 1: Activation Hadamard Rotation
  X_rotated = Fast_Walsh_Hadamard_Transform(X)  // O(MK log K)
  // 效果: peak activation从1272降至58.5（RoMeo实测）
  
  // Step 2: Token-wise Mixed Precision Quantization
  // 旋转后outlier呈纯token-wise分布
  for t in 0..M:
      if is_outlier_token(t):
          X_Q[t,:] = INT8_quantize(X_rotated[t,:])
      else:
          X_Q[t,:] = INT4_quantize(X_rotated[t,:])
  
  // Step 3: Cross-precision GEMM
  Y_low = X_Q_int4 · W_rotated_Q_int4  // W4A4
  Y_high = X_Q_int8 · W_rotated_Q_int8  // W8A8 (等)
  Y = combine(Y_low, Y_high)
```

Hadamard矩阵的递归结构（以H_4为例）：
```
H_1 = [1]
H_2 = [1  1; 1 -1]
H_4 = [H_2  H_2; H_2 -H_2]
     = [1  1  1  1; 1 -1  1 -1; 1  1 -1 -1; 1 -1 -1  1]
```

乘法可通过Fast Walsh-Hadamard Transform (FWT)在O(MK log K)复杂度内实现（vs naive O(MK^2)），相比Linear层的O(MKN) GEMM可忽略。RoMEo使用HadaCore库实现FWT。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 实现库：HadaCore（Tensor Core accelerated Hadamard transform kernel，K. Agarwal et al., 2024），QuaRot提供了参考PyTorch实现。
- 在线FWT的开销：RoMeo实测Hadamard transformation仅占layer latency的~4%（batch=64时），显著低于GEMM主导成本。
- 注意事项：
  (1) Hadamard matrix要求维度为2的幂，非2的幂需padding或truncate。
  (2) 旋转在模型中不同位置的应用策略影响性能——QuaRot在Qwen3-14B上因在attention heads间插入旋转导致性能下降（40 heads→inefficient transformation），RoMeo在heads的hidden dimension上应用旋转避免此问题。
  (3) 旋转非无损——它将model weight的数值分布改变，可能影响某些精度敏感层的quality，但正交性保证数学等价（X·W = X·H·H^T·W）。
- 旋转可与其他quantization技术正交组合：如SpinQuant（learned rotations）、DuQuant（dual transformation）、FlatQuant等优化旋转矩阵的方法可与RTMPQ的mixed precision方案相结合。
- **CoRFiG (Coarse Rotation, Fine Grouping)**：GyRot提出将rotation scope R与group quantization粒度G解耦——R=2^g·G, R≤1024。全局rotation导致outlier dispersed across all channels → 与group quantization的localized scaling冲突（小G下RTN PPL从7.40升至30.12 at G=32）。CoRFiG限制rotation scope到R=1024，既保留distribution flattening benefit又保留group-level local variance→G=32+R1024 PPL=6.91。配合HAP（Harmonic-Aligned Permutation，利用Hadamard harmonic rows对齐outlier channels）进一步tighten per-group range→INT8 SF可达FP16 SF parity。

涉及论文标题：
- RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization
- GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

## RTMPQ（Rotated Token-wise Mixed Precision Quantization / 旋转令牌级混合精度量化）

术语是什么？通过联网搜索让回答具体和精准。

RTMPQ（Rotated Token-wise Mixed Precision Quantization）是RoMeo提出的面向LLM 4-bit weight-activation量化的算法，通过"先旋转后混合精度"的两阶段策略处理双维度（channel+token）outlier。核心思路：(1) 用Hadamard rotation将channel-wise outliers平滑并迁移到token维度；(2) 对纯token-wise outliers用token-wise mixed precision（outlier tokens→INT8, normal tokens→INT4）。该算法使4-bit量化的LLM准确率显著优于仅处理单维度outlier的baseline（QuaRot、MixQ）。

从算法pipeline角度拆解术语：

RTMPQ完整算法流程（以Qwen3-8B Linear层推理为例）：

```
Algorithm: RTMPQ Forward Pass
Input:  Activation X [M, K] (FP16)
        Weight W [K, N] (offline rotated: W' = H^T·W)
        Outlier budgets: k_a (token outliers), k_w (weight outliers)
        
// === 离线预处理（Serving前完成） ===
1. W_rot = H^T · W           // Weight Hadamard rotation
2. max_w = reduce_max(|W_rot|, axis=1)  // per-column (对应per-channel)
3. O_W = topk_indices(max_w, k_w)
4. W_rot_Q = mixed_precision_quantize(W_rot, O_W)  // INT4+INT8

// === 在线推理 ===
5. X_rot = FWT(X)            // Fast Walsh-Hadamard Transform: O(MK log K)
6. max_x = reduce_max(|X_rot|, axis=1)  // per-token max
7. O_A = topk_indices(max_x, k_a)  // top-k outlier tokens (k_a ≈ 5% M)
8. X_rot_Q_int4 = INT4_quantize(X_rot)  // 全矩阵INT4
9. X_rot_Q_int8 = INT8_quantize(copy_outlier_tokens(X_rot, O_A))  // outlier buffer

// === Cross-Precision Multiplication ===
10. C_W4A4 = INT4_GEMM(X_rot_Q_int4, W_rot_Q_int4)     // 主体计算
11. C_W4A8 = INT4x8_GEMM(X_rot_Q_int4_normal, W_rot_Q_int8_outlier)
12. C_W8A4 = INT8x4_GEMM(X_rot_Q_int8_outlier, W_rot_Q_int4_normal)
13. C_W8A8 = INT8_GEMM(X_rot_Q_int8_outlier, W_rot_Q_int8_outlier)
14. C = dequantize_and_combine(C_W4A4, C_W4A8, C_W8A4, C_W8A8)

15. Post-mul Overwrite: C[O_A, :] = C_W8A*/C_W*A8结果覆盖对应行
```

RTMPQ的理论加速比（以m=n=4096, k_a=k_w=256为例）：
- P_INT4 = (4096-256)² / 4096² = 88%（纯INT4计算比例）
- S = 1 / (0.88/4 + 0.12/2) = 3.57× (vs FP16 baseline)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 实现方式：PyTorch nn.Module封装（可直接替换原始Linear），HadaCore提供FWT，CUTLASS提供INT4/INT8 GEMM kernel，Triton提供fused outlier detection+quantization kernel。
- Outlier budget配置：RoMeo使用5%作为默认值（k_a=0.05M, k_w=0.05N），实验显示outlier从0%到1.6%时perplexity改善最显著（Qwen3-8B: 0.40 PPL reduction），后续边际效益递减。
- Weight mixed precision的必要性：Hadamard rotation后的weight矩阵因H^T预乘amplify non-uniformity，也需要mixed precision处理（与activation对称，per-column outlier detection）。
- 精度优势：RoMeo在Qwen3-8B上PPL 10.97优于QuaRot 11.53（uniform INT4+rotation），Qwen3-14B上70.82 avg zero-shot accuracy优于QuaRot 70.04。
- 开源：https://github.com/thu-pacman/RoMeo

涉及论文标题：
- RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

## AU-accelerated LLM Serving Inference

术语是什么？

AU-accelerated LLM Serving是指利用CPU内置加速器单元（如Intel AMX）来加速LLM推理的serving范式。与传统GPU serving不同，AU-accelerated serving运行在通用CPU上，通过AMX TMUL加速矩阵乘。LLM推理的两阶段在AU上表现截然不同：(1) Prefill phase：GEMM operations (8192×4096×22016, batch=16) 通过AMX达到40.57 TFLOPS，属于compute-bound (92% backend bound)，高AMX cycle ratio (14.4%)，高功耗导致core频率降至2.5 GHz；(2) Decode phase：GEMV operations (16×4096×22016) 通过AMX仅3.87 TFLOPS（小矩阵tile register配置overhead大），属于memory-bound (DRAM 59.9%)，低AMX cycle ratio (1.5%)，此时AVX比AMX更高效。工业界使用xft/xFasterTransformer或ktransformers等框架实现AU-accelerated serving，支持BF16精度和batch size 16。主要SLO指标：TTFT (Time-To-First-Token, prefill) 和 TPOT (Time-Per-Output-Token, decode)。AU-enabled CPU适合serving小模型（Phi-3 3.8B, Llama2 7B/13B, Qwen3-A3B 30B MoE）或作为GPU的补充。

从算法pipeline角度拆解术语：

AU-accelerated LLM serving的pipeline（以llama2-7B, batch=16为例）：
```
// === Prefill Phase (compute-bound, AMX) ===
Input: prompt tokens [0..L-1]

for each transformer layer:
    1. QKV Mapping: input @ W_QKV
       └─ matrix dim: (8192, 4096×22016), batch=16, input_len=512
       └─ AMX GEMM → 40.57 TFLOPS, AMX cycle ratio 14.4%
       └─ Backend bound: 92% (execution port + memory stalls)

    2. Multi-head Attention: Q @ K^T @ V
       └─ FlashAttention-like, compute-bound
       └─ Store KV to KV Cache

    3. Feed Forward: SiLU(W_gate @ x) * (W_up @ x) @ W_down
       └─ Each GEMM uses AMX for large matrix dims

Output: first token generated, KV Cache populated

// === Decode Phase (memory-bound, AVX preferred) ===
Input: last generated token [1, 4096]

for each transformer layer:
    1. QKV Mapping: last_token @ W_QKV
       └─ matrix dim: (16, 4096×22016), batch=16
       └─ AMX GEMV → 3.87 TFLOPS (AMX inefficent for small dim)
       └─ AVX preferred: lower tile register overhead
       └─ AMX cycle ratio: 1.5%, AMX uop ratio: 0.5%
       └─ DRAM bound: 59.9%

    2. Attention: q @ K^T (from KV Cache) → memory-bound
       └─ Load entire KV Cache from memory

    3. Feed Forward → next layer

Output: next token → append KV Cache → repeat

// SLO: TTFT = time from prompt to first token
// SLO: TPOT = average time per subsequent token
```

术语一般如何实现？如何使用？

AU-accelerated serving通过xft或ktransformers调用oneDNN GEMM→oneDNN根据矩阵维度自动选择AMX或AVX。工业界现状：CPU与GPU hybrid部署日益普遍（Alibaba Cloud ~50% CPU核心idle但带AU可利用），AU serving perf-per-dollar优于GPU但perf-per-watt不如GPU（A100比GenA高2.1×）。AUM进一步提出AU CPU与通用workload共享提升平台效率。AU适用于CPU-only serving小模型或CPU-GPU hybrid部署中的轻量AU workload。

涉及论文标题：
- AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving

## CoRFiG (Coarse Rotation, Fine Grouping / 粗粒度旋转细粒度分组)

术语是什么？

CoRFiG是GyRot提出的旋转与分组量化协同策略。它将Hadamard rotation的scope R限制在coarse粒度（如R=1024），同时保持group quantization的group size G在fine粒度（如G=32），满足R=2^g·G关系（g为正整数）。这解耦了rotation的outlier dispersion scope与group quantization的localized scaling granularity。

从算法pipeline角度拆解术语：

全局rotation将outlier分散到所有channel，与fine group quantization（G=32）的localized scaling冲突——小G下group quantization自身即可捕获local variance，全局rotation反而引入inter-group interference。CoRFiG的pipeline（以LLaMA-3-8B, R=1024, G=32, g=5为例）：

```
// CoRFiG量化pipeline
Offline 权重旋转:
  for each weight matrix W with Nch channels:
    // 分割为Nch/R个rotation blocks
    for each block b of size R=1024:
      H_block = Hadamard(R)  // R×R Hadamard matrix
      W_rot[b] = H_block^T · W[b]  // 局部旋转，仅影响R channels内
    // HAP permutation: outlier channels → harmonic rows (可fuse进weight)
    W_rot = HAP_permute(W_rot)
    // Group量化: G=32 per group
    for each group g of size G=32:
      W_Q[g] = INT4_quantize(W_rot[g], SW[g])  // SW: INT8 scale

Online 推理 (每token):
  // Step 1: Online rotation (仅在非线性层后)
  if layer_has_nonlinear_before:
    for each rotation block of size R:
      X_rot = FHT(X)  // Fast Hadamard Transform, O(R log R)
    // 量化: G=32 per group
    for each group g:
      zx = ceil(-min(X_rot[g]))  // 直接unscaled domain计算 (reformulated asym)
      sx = (max(X_rot[g]) + zx) / (2^b - 1)
      X_Q[g] = clip((X_rot[g] + zx) / sx, 0, 2^b-1)  // INT4

  // Step 2: INT4 GEMM + Integer Dequantization
  for each group g:
    partial_sum = X_Q[g] · W_Q[g]  // INT4 dot product
    y[g] = SW[g] * (SX[g] * partial_sum - ZX[g] * WSUM[g])  // 全整数dequant

// 关键对比:
// global rotation (Quarot): X_rot = H_global · X → outlier disperse Nch-wide → G=32下PPL=7.04
// CoRFiG (GyRot): X_rot_block = H_R · X_block → outlier disperse R-wide → G=32下PPL=6.91
// No rotation + G32: PPL=7.40 (RTN) → CoRFiG achieves synergy
```

术语一般如何实现？如何使用？

CoRFiG的关键参数选择：(1) R选择：GyRot的FHT硬件支持2的幂次rotation scope up to 1024，实验显示R=1024时PPL饱和（Table III: R=1024 PPL=6.91 vs R=512 PPL=6.89→接近收敛）。(2) G选择：G=32为GyRot PE的最小支持粒度（32-way INT4 dot product），G=32配合R=1024实现最佳accuracy-hardware tradeoff。(3) R=2^g·G约束：保证rotation block内group对齐，使HAP harmonic row alignment的benefit均匀覆盖所有group。GyRot对比LightRot (R=G=128) 的关键区别在于CoRFiG解耦允许更fine的G而不损失rotation benefit。

涉及论文标题：
- GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

## HAP (Harmonic-Aligned Permutation / 谐波对齐排列)

术语是什么？

HAP是GyRot提出的基于Hadamard矩阵harmonic行结构的outlier channel排列策略。Hadamard矩阵递归构造（Sylvester's method）产生"harmonic rows"——长度为2^k的全+1或全-1向量（k<n）。HAP将全局选出的high-magnitude outlier channel permute到这些harmonic rows上，使post-rotation后每个group内的outlier乘以一致符号（全+1或全-1），从而tightly bound per-group range。

从算法pipeline角度拆解术语：

```
// HAP算法流程 (G=8, R=32, g=2, 即R=4G)
Input: activation X of shape [batch, Nch], G=8, R=32

Step 1: 识别全局outlier channels
  // 按per-channel magnitude排序
  O = topk_outlier_channels(X, k)  // 选top-k高magnitude channel
  // 例: O = {O1, O2, O3, O4} (4个outlier channels for R=32=4G)

Step 2: 识别harmonic rows
  // Hadamard H_32有harmonic rows at positions: G, 2G, 3G, 4G (=8, 16, 24, 32)
  // Harmonic row at position k·G (k=1..2^g): 长度为G的重复+1/-1 pattern
  // 例: row 8 = [+1×8, -1×8] within each block, row 16 = [+1×16, -1×16] etc.

Step 3: Permute outlier channels → harmonic rows
  // Permute O1→row G, O2→row 2G, O3→row 3G, O4→row 4G
  X_perm = apply_permutation(X, O, harmonic_rows)

Step 4: Apply Hadamard rotation (CoRFiG, R=32)
  X_rot = FHT_32(X_perm)

// Post-rotation effect (以group 0为例):
// Without HAP: outlier O1混入group 0 with 随机符号 (mix of +1/-1)
//   → group range = [-O1-O2-O3-O4, +O1+O2+O3+O4] → wide spread
// With HAP: 每个outlier×同符号 within its group
//   例: row G→[+1,..,+1] → outlier O1所有元素在group 0内×(+1)
//   → group range = [O1+O2+O3+O4, ...] → shifted bias, tighter bound

// Effect on Scale Factor precision (Table IV):
// G32 without HAP: INT8 SF → PPL 364.17 (catastrophic)
// G32 with HAP: INT8 SF → PPL 6.80 (near FP16 6.80)
```

术语一般如何实现？如何使用？

HAP的关键特性：(1) Permutation可fuse进weight矩阵（permutation-invariant property: 非线性和element-wise ops对排列不变），无runtime overhead——只需在offline阶段permute weight output channels，activation自然被permuted。(2) Harmonic rows的选择：Hadamard矩阵大小为R时，有log₂(R)个harmonic row位置（2^k, k=1..log₂(R)），每个位置对应一个特定stride的全+1/-1 pattern。(3) HAP与CoRFiG配合：CoRFiG确保rotation scope R=2^g·G，使所有harmonic rows的stride与group size对齐，每个group恰好包含一个harmonic row的完整segment。

涉及论文标题：
- GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

## Reformulated Asymmetric Quantization with Ceiling Zero-Point Rounding（重公式化非对称量化与上取整零點舍入）

术语是什么？

GyRot提出的非对称量化公式改造：将传统先scale后bias的公式 x̂=⌊x/s_x+z_x⌉ 改为先bias后scale的 x̂=⌊(x+z_x)/s_x⌉，同时zero-point从scaled domain计算（z_x=−min(x_g)/s_x）改为unscaled domain计算（z_x=−min(x_g)），并使用ceiling ⌈·⌉替代round做ZP量化以避免underflow clipping。

从算法pipeline角度拆解术语：

传统非对称量化在高asymmetry（如HAP后）下的问题：
```
// 传统公式: x̂ = clip(round(x/s_x + z_x), qmin, qmax)
// z_x = -min(x_g)/s_x  (in scaled domain)
// 问题: s_x小 → z_x被放大 → long-tailed ZP distribution (Fig. 5)
// 原因: min(x_g)在HAP后可以很大（biased group distribution）
//       s_x = (max(x_g) - min(x_g))/(2^b-1)
//       当min(x_g)和max(x_g)都很大时, s_x可能很大,
//       但z_x = -min(x_g)/s_x ≈ -(2^b-1) * min/(max-min)
//       若asymmetry严重, min/(max-min) ratio → large → z_x尾部拉长
```

GyRot重公式化：
```
// GyRot公式:
// z_x = ceil(-min(x_g))  // 直接从unscaled domain计算, 无除法放大
// s_x = (max(x_g) + z_x) / (2^b - 1)  // 注意: +z_x而非-min (因为z_x已含符号)
// x̂ = clip((x + z_x) / s_x, qmin, qmax)

// Dequantization (内积):
// y ≈ Σ_g SW[g] * (SX[g] * Σ_i x̂_i·ŵ_i - ZX[g] * Σ_i ŵ_i)
//               ↑ 先乘SX         ↑ 再减ZX×WSUM  ↑ 再乘SW
// vs 传统: y ≈ Σ_g SX[g] * SW[g] * Σ_i (x̂_i - ZX[g])·ŵ_i

// Ceiling ZP rounding:
// ZX_quantized = ceil(zx)  // 保证 ZX_Q ≥ zx, 消除 underflow
// vs 传统round: ZX_quantized = round(zx) → 可能 ZX_Q < zx → clipping (Fig. 6)
```

效果（Table V, LLaMA-3-8B G=32 R=1024, CoRFiG+HAP）：
- 传统asym + Round ZP: FP16 ZP=6.81, INT8 ZP=7.93 (degradation)
- Reformulated asym + Round ZP: FP16 ZP=6.80, INT8 ZP=7.65 (改善)
- Reformulated asym + Ceiling ZP: FP16 ZP=6.81, INT8 ZP=6.91 (near parity)

术语一般如何实现？如何使用？

实现要点：(1) z_x用⌈−min(x_g)⌉而非⌊−min(x_g)/s_x⌉——避免除法放大效应。这对HAP后per-group范围biased的情况尤其重要。(2) s_x公式从(max−min)/(2^b−1)变为(max+z_x)/(2^b−1)，保证量化范围覆盖全部正值。(3) Ceiling rounding在硬件上等同于负数方向的floor：⌈x⌉ = -⌊−x⌋，可用标准整数rounding单元实现。(4) 与fully integer dequantization配合：重公式化使ZX范围可控 → INT8 ZX精度充足 → 全整数dequantization datapath可行。

涉及论文标题：
- GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

## Row-Reuse Mapping for GQA on PIM

术语是什么？通过联网搜索让回答具体和精准。

Row-Reuse Mapping是PIM上执行GQA（Grouped Query Attention）时的优化策略：利用GQA中多个query heads共享同一组K/V cache的特性，优先在当前已打开的DRAM row上处理所有共享该KV row的query heads，减少DRAM row activation（ACT）和precharge（PRE）开销。在DRAM-based PIM中，读取不同row需要先precharge当前row再activate新row（ACT/PRE开销显著），row-reuse通过在同一open row上连续服务多个query head来摊薄此开销。但row-reuse引入了额外的WR-INP压力——需要为不同query head反复写入不同的GBuf entry，在静态PIM scheduling下这些input transfer stalls可能抵消row-reuse的收益。PIMphony的DCS通过dual-port GBuf/OBuf和entry-level dependency tracking在MAC消费当前GBuf entry时预取下一批query，将row-reuse的KV复用转化为真实吞吐收益。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// GQA Row-Reuse Mapping on PIM（group_size=4, 4 query heads share 1 KV pair）:

// 无row-reuse（逐个head处理）:
for head h in 0..3:  // 4 query heads share same K/V
    ACT(row_K_h)     // 为每个head单独activate K row
    for t in 0..T-1:
        MAC(GBuf[q_h], K[t], OBuf[h][t])
    PRE(row_K_h)
// 总计: 4×ACT + 4×PRE overhead

// 有row-reuse（所有heads在同一open row上处理）:
ACT(row_K_shared)    // 一次activate, 4个head共享
WR_INP(GBuf[0], q_h0)  // 写入第一个head的query
for h in 0..3:
    if h > 0:
        WR_INP(GBuf[h], q_h)  // DCS可预取: GBuf[h]写入与GBuf[h-1]消费并行
    for t in 0..T-1:
        MAC(GBuf[h], K[t], OBuf[h][t])  // 所有h读同一open row
PRE(row_K_shared)
// 总计: 1×ACT + 1×PRE, 但4×WR-INP串行化（无DCS时）

// DCS优化: dual-port GBuf
//   port A: MAC读取GBuf[0] → 处理q_h0
//   port B: 同时WR-INP写入GBuf[1] → 准备q_h1
// → WR-INP latency被MAC重叠，消除input transfer stalls
```

PIMphony论文中，DCS配合row-reuse在GQA 128K模型上取得比non-GQA模型更大的收益（up to 11.3× speedup vs CENT），因为DCS将row-reuse减少的ACT/PRE overhead转化为真实吞吐而不会被WR-INP stalls抵消。对比ping-pong buffering baseline：ping-pong因需等两个region均idle才能切换（hand-off pipeline stalls），row-reuse的WR-INP压力导致更频繁的hand-off，DCS同buffer size下up to 1.4× higher compute-unit utilization。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Row-reuse mapping由compiler或runtime的attention scheduler实现：识别GQA group structure→将共享同一KV pair的query heads聚合到同一PIM channel→调度MAC序列使同一DRAM row上的连续MAC操作最大化。PIMphony的MLIR compiler在pattern-matching阶段识别GQA config（group size g），在code generation阶段为共享KV的heads生成row-reuse optimized MAC序列并嵌入DCS dependency annotations使WR-INP与MAC overlap。GQA group size越大（如g=8），row-reuse收益越大，但WR-INP压力也越大——DCS的overlap能力在此trade-off中起关键作用。

涉及论文标题：
- PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System

## Adaptive Draft Sequence Length（自适应起草序列长度）

术语是什么？通过联网搜索让回答具体和精准。
Adaptive Draft Sequence Length是SADDLE提出的一种运行时动态调整speculative decoding中每个请求draft token数量的机制。与传统的固定draft length（如d=8，所有请求统一）不同，SADDLE为每个请求独立动态决定draft长度：Controller在DLM每生成一个draft token时读取该token的采样概率p_t，维护累计接受概率H_t = ∏_{i=1}^{t} p_i，当H_t低于预设阈值τ时停止该请求继续drafting。阈值τ通过离线验证集校准——对每个请求运行完整prediction-verification pipeline，记录每draft step的H_j和验证结果，估算条件成功率曲线，选择20%区间内平均draft length最高且≥90%验证成功率的τ。运行时τ可动态调节：轻负载时降低τ允许更长drafts提升并行度。该方法不需要额外模型训练或分类器，仅基于DLM自身输出概率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
Adaptive Draft Length的pipeline（per request per speculative iteration）：
```
Input: prefix X, DLM M_d, threshold τ
Output: draft tokens {x_1, ..., x_k} where k is adaptive

// 初始化
H_0 = 1.0
draft_tokens = []
context = X

// 逐token Draft Generation with early stopping
for t = 1, 2, ...:
    // DLM forward → next token distribution
    P_t = M_d(context)  // probabilities over vocabulary
    
    // Sample token and get its probability
    x_t ~ P_t
    p_t = P_t[x_t]  // probability of sampled token
    
    // Update cumulative acceptance probability
    H_t = H_{t-1} * p_t
    
    // Adaptive stopping check
    if H_t < τ:
        break  // stop drafting for this request
    
    draft_tokens.append(x_t)
    context = context + [x_t]

// Return variable-length draft sequence
return draft_tokens  // length k ≤ max_draft_length
```
关键特征：(1) 每步仅需一次乘法（H_t更新）和一次比较（H_t < τ），开销极低（SADDLE中仅占0.83% end-to-end latency）；(2) 简单请求（高p_t token，如常见continuation）H_t下降慢→自动获得更长drafts；复杂请求（低p_t token，如creative writing）H_t下降快→更早停止，避免生成可能被TLM拒绝的token；(3) 决策依赖DLM自身概率，无需TLM反馈，可完全在prediction stage内完成。

术语一般如何实现？如何使用？
SADDLE在Manager的Controller中实现：Controller集成softmax unit（计算P_t）、multipliers（更新H_t）和comparators（比较H_t与τ），以专用硬件实现低延迟决策。离线校准：用验证集（如Dolly的subset）对每个model pair（如OPT-66B+OPT-1.3B）扫描τ值→选满足90%+验证成功率的最大draft length对应的τ。SADDLE的实验显示仅用自适应draft length（SADDLE-d）不加pipeline优化时吞吐反比PIM-SD低1.22×，说明自适应drafting需要配合Shared Pool和异步pipeline才能发挥收益。该技术与Disco（classifier-based stop decision）、OPT-Tree（greedy draft tree construction）等自适应draft方法形成对比——SADDLE的方法更轻量（无额外模型/树搜索），但threshold tuning依赖workload分布。

涉及论文标题：
- Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems

## Cumulative Acceptance Probability（累计接受概率）

术语是什么？通过联网搜索让回答具体和精准。
Cumulative Acceptance Probability (H_t) 是SADDLE用于自适应控制draft sequence length的核心指标。定义为H_t = ∏_{i=1}^{t} p_i，其中p_i是DLM在第i步生成的draft token x_i的采样概率P_{DLM}(x_i | x_{<i})。H_t的本质是对当前draft sequence中所有t个token都被TLM接受的概率的下界估计——由于speculative decoding的rejection sampling机制，一个token被接受的概率为min(1, p_T / p_D)，其中p_D = P_{DLM}(token)和p_T = P_{TLM}(token)。当p_D较低时，该token被拒绝的概率更高，H_t快速下降→触发early stopping。H_t单调递减（每步乘以p_i ∈ (0,1]），天然适合阈值决策。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
H_t的更新和决策机制：
```
// Per draft step t:
H_t = H_{t-1} * p_t  // single FP16 multiplication

// Stopping rule:
if H_t < τ:
    stop_drafting()
    
// 直观理解（τ = 0.3为例）：
// Step 1: p_1 = 0.9 → H_1 = 0.9   ≥ 0.3 → continue
// Step 2: p_2 = 0.8 → H_2 = 0.72  ≥ 0.3 → continue  
// Step 3: p_3 = 0.5 → H_3 = 0.36  ≥ 0.3 → continue
// Step 4: p_4 = 0.6 → H_4 = 0.216 < 0.3 → stop (draft length = 4)
```
H_t的理论基础：当DLM对某个token的信心（p_i）低时，该token更可能与TLM的distribution不一致→被拒绝的概率高→继续在此token之后drafting大概率产生无效计算。H_t作为累积度量同时惩罚连续低概率token（如连续3个0.5概率token使H_t降为0.125）和单个极低概率token（如一个0.1概率token直接触发停止）。

术语一般如何实现？如何使用？
SADDLE在Controller硬件中实现H_t更新：1KB SRAM存储每请求的当前H_t值，multipliers执行FP16乘法更新，comparators与τ比较。仅需轻量级on-chip memory操作，延迟可忽略（<1% of prediction latency）。H_t的threshold τ通过离线校准确定，不同model pair和dataset可有不同τ。论文观察到OPT-66B+OPT-1.3B在Dolly上τ≈0.3效果最优，但具体值需根据workload调整。

涉及论文标题：

## Grouped-Query Attention (GQA) and Multi-Query Attention (MQA)

术语是什么？通过联网搜索让回答具体和精准。

Grouped-Query Attention (GQA, Ainslie et al., 2023) 和 Multi-Query Attention (MQA, Shazeer, 2019) 是 Transformer decoder 中减少 KV cache 内存占用的注意力变体。标准 Multi-Head Attention (MHA) 中，每个 query head 有独立的 K/V head（hq = hkv）。MQA 极端化：所有 query heads 共享单组 K/V heads（hkv = 1, K/V 投影仅需 2 个矩阵而非 MHA 的 2×hq 个）。GQA 折中：将 query heads 分为若干组，每组共享一组 K/V head——共享度由 gq = hq/hkv 度量。gq=1 即 MHA, gq>1 为 GQA, hkv=1 (gq=hq) 为 MQA。GQA/MQA 降低 KV cache 容量至 MHA 的 1/gq，同时减少 decode 时 K/V head projection 的计算量。现代 LLM 广泛采用 GQA：LLaMA-3.1-8B/70B (hq=32, hkv=8, gq=4)、Qwen3 (hq/hkv=4:1)、DeepSeek-V3 使用 MLA（Multi-head Latent Attention, 进一步压缩的下一个演进）。关键 trade-off：MQA 最大化 memory saving 但约束 attention expressiveness→GQA 在 memory 和 quality 间取平衡，training from scratch 或 up-training (从 MHA checkpoint 转换) 均可。

从算法 pipeline 角度拆解术语：

```
// === GQA Attention Computation (decode step, gq=4, hq=32, hkv=8) ===
// 输入: Q ∈ R^{1×hq×d} (1 token × 32 query heads × head_dim)
//       K_cache, V_cache ∈ R^{L×hkv×d} (L tokens × 8 KV heads)

// Step 1: Query projection (所有32个Q heads并行)
Q = X × W_Q  // shape: [1, 32*d_head]

// Step 2: K/V projection — 仅需8个heads（vs MHA需要32个）
K_new = X × W_K  // shape: [1, 8*d_head]
V_new = X × W_V  // shape: [1, 8*d_head]

// Step 3: GQA attention computation
for each kv_head in range(8):  // 8个KV heads
    // 该KV head对应的4个query heads
    q_group_start = kv_head * 4;
    q_group = Q[q_group_start : q_group_start+4];  // shape: [4, d_head]
    
    // Step 3a: QK^T — 4个queries共享同一K
    // MHA: 每个Q head有独立K → 1×L GEMV × 32 heads
    // GQA: 4个Q × 同一K → 4×L GEMM × 8 groups → 更高arithmetic intensity
    scores = q_group × K_cache[:, kv_head, :]^T  // shape: [4, L]
    
    // Step 3b: Softmax (row-wise, 沿L维度)
    attn_weights = softmax(scores / sqrt(d_head))  // shape: [4, L]
    
    // Step 3c: PV — 4个attention weights共享同一V
    output[q_group_start:q_group_start+4] = attn_weights × V_cache[:, kv_head, :]
    // shape: [4, d_head]

// Step 4: Output projection
output = concat(all_outputs) × W_O  // [1, 32*d_head]
```

GQA 对系统性能的关键影响：从 MHA 到 GQA (gq=4)→K/V projection 计算量减至 1/4→K/V cache memory 减至 1/4→但 QK^T 的 M 维度从 1 增至 4（grouped queries 做联合 GEMM）→arithmetic intensity 提升→更利于 Tensor Cores。BitDecoding 在 RTX 4090 GQA 下 3× speedup vs QServe 仅 1.4×——QServe 的 CUDA Cores-only approach 无法利用 GQA 带来的 compute intensity 提升，而 BitDecoding 的 Tensor Cores 天然受益于更大的 M 维度。

术语一般如何实现？如何使用？

GQA/MQA 在 training 阶段通过修改 attention layer 的 K/V projection 实现——将独立的 hq 个 K/V projection 矩阵合并/复用为 hkv 个。从 MHA checkpoint 转换为 GQA 可通过 mean pooling 或 select-and-fine-tune 已有 K/V heads。Inference 使用不需要特殊改动（K/V cache 自动减少），但 kernel 实现可利用 grouping 优化（如 BitDecoding Query Transformation）。GQA/MQA 与 FlashAttention 等 fused attention kernels 兼容，且与 KV cache quantization、PagedAttention 等正交优化可叠加。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

## Channel-wise vs Tensor-wise KV Cache Quantization Scaling

术语是什么？通过联网搜索让回答具体和精准。

Channel-wise 和 Tensor-wise 是低比特 KV cache 量化中两种不同的 scaling granularity，决定 scale factor 和 zero-point 按什么维度计算和存储。Channel-wise quantization: 沿 Key/Value tensor 的 channel 维度（hidden dimension）分组计算 scale/zero——每组 channel 有独立的量化参数。例如 K ∈ R^{L×d}, 沿 hidden dim 维度 groupsize=128 → d/128 个 scale/zero per token → total params = L × d/128 × 2。Tensor-wise quantization: 沿 token（序列）维度分组——整个 tensor 的所有元素共享一组 scale/zero→params = 1 × 2 per tensor。Channel-wise 更精细（per-channel quantization handles channel-specific value range, 尤适合处理 KV cache 中的 outlier channel），但 metadata overhead 更大。Tensor-wise metadata overhead 极小但可能放大量化误差（outlier channel 的极端值拉大整个 tensor 的 scale）。主流量化算法倾向不同：KIVI 使用 per-channel quantization（group_size=128 for K）；KVQuant 使用 per-channel with dense-and-sparse decomposition；Atom/QServe 支持 tensor-wise；Gear 和 JanusQuant 使用 per-channel。BitDecoding 同时支持两种 scaling，通过 Residual Kernel 的 residual block 内按不同维度做 reduction 实现。

从算法 pipeline 角度拆解术语：

```
// === Channel-wise vs Tensor-wise Quantization Example ===
// K ∈ R^{L×d}, L=128K tokens, d=4096

// --- Channel-wise (KIVI-style, group_size=128沿hidden dim) ---
for each token t in [0..L-1]:
    for each group g in [0..d/128-1]:  // 32 groups
        group_vals = K[t, g*128 : (g+1)*128]
        s[g] = (max(group_vals) - min(group_vals)) / (2^β - 1)
        z[g] = min(group_vals)
        K_q[g*128:(g+1)*128] = quantize(group_vals, s[g], z[g])
// metadata size: L × 32 × 2 = 64L scalars (FP16 each = 128L bytes)
// 优点: outlier channel仅影响其所在group，不污染其他channels

// --- Tensor-wise (Atom/QServe style) ---
for each token t in [0..L-1]:
    all_vals = K[t, :]  // 整行4096元素
    s = (max(all_vals) - min(all_vals)) / (2^β - 1)
    z = min(all_vals)
    K_q[t, :] = quantize(all_vals, s, z)
// metadata size: L × 2 scalars (FP16/2 = 4L bytes)
// 缺点: 1个outlier channel的极值拉大s→所有channels量化粒度变粗

// --- BitDecoding Residual Block内统一执行两种scaling ---
// Residual block: K_res ∈ R^{Nr×d}
// Channel-wise: reduction沿seq_len维度 → per-channel scale across Nr tokens
for each channel c in [0..d-1]:
    s_c = (max(K_res[:, c]) - min(K_res[:, c])) / (2^β - 1)
// Tensor-wise: reduction沿hidden维度 → per-tensor scale across d channels
for each token t in [0..Nr-1]:
    s_t = (max(K_res[t, :]) - min(K_res[t, :])) / (2^β - 1)
```

术语一般如何实现？如何使用？

Channel-wise 和 tensor-wise 的选择需权衡 accuracy 和 metadata overhead。Channel-wise 通常用于 Key cache（outlier 沿 channel 集中），tensor-wise 可用于 Value cache（distribution 更均匀）。BitDecoding 的 Residual Kernel 用 warp-level __shfl_xor_sync reduction 计算 min/max：对于 channel-wise，每组沿 seq_len 方向对 Nr 个 token 做统计 → scale/zero 按 hidden dim 输出（Nr 个 scale, d 个 zero）；对于 tensor-wise，每 token 沿 hidden dim 方向对 d 个值做统计。Scale/zero 以 half2 格式存储（两个 FP16 pack 到一个 INT32）以最小化 memory traffic。与 weight quantization 的 per-channel scaling（offline 预计算）关键区别：KV cache scaling 在 runtime 在线计算，必须高效（BitDecoding decode overhead 仅 0.008ms）。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache
- Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems

## Product Quantization (PQ) for KV Cache Compression（乘积量化KV缓存压缩）

术语是什么？通过联网搜索让回答具体和精准。
Product Quantization (PQ) 是一种基于向量量化的压缩技术，由Jegou等人于2011年提出[H. Jegou et al., "Product Quantization for Nearest Neighbor Search," TPAMI 2011]，起源于近似最近邻搜索领域。PQ的核心操作是：(1) Vector Splitting：将d维高维向量分解为m个子向量（每组d/m维）；(2) 对每个subvector group独立运行K-means clustering，生成K个centroid的codebook；(3) 原始向量由m个index（每个∈[0,K-1]）表示，指向各subvector group的centroid。AQPIM首次将PQ引入PIM-based KV cache在线量化，利用PIM高内部带宽(>TB/s)支持在prefill阶段on-the-fly执行K-means clustering（4迭代收敛），将PQ从传统离线weight-only压缩扩展到在线activation压缩。与per-weight scalar quantization（如uniform INT4/KVQuant）不同，PQ通过subvector-level clustering捕获KV cache activation的context-dependent locality和similarity（UMAP visualization显示KV vector呈tight cluster分布），可达到80%+ memory reduction ratio。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
AQPIM中PQ-based KV cache量化pipeline（以Mistral-7B, m=32, K=512为例）：

```
// ===== Prefill阶段: Codebook Generation (GPU-PIM并行) =====
// 输入: K, V ∈ R^{N×d} (N tokens, d=4096 hidden dim)
// Parameter: m=32 subvectors, K=512 centroids per subvector

for each attention layer:
    // Step 1: Channel Pre-Sorting (offline, absorb to projection)
    //    W_q'=W_q·P_k, W_k'=W_k·P_k, W_v'=W_v·P_v, W_o=W_o·P_v^T
    //    P_k, P_v generated offline via cosine similarity grouping
    
    // Step 2: Vector Splitting
    for i in 1..m:  // m=32 subvectors
        K_sub[i] = K[:, (i-1)*d/m : i*d/m]  // [N, d/m]
        V_sub[i] = V[:, (i-1)*d/m : i*d/m]
    
    // Step 3: Importance-Weighted K-means (per subvector, 4 iterations)
    w = sum(S[-32:, :], axis=0)  // attention score weights, t=32
    for iter in 1..4:
        // E-step: Assign tokens to nearest centroid
        for n in 1..N:
            c[n] = argmin_k ||K_sub[i][n,:] - centroid[k]||²
        
        // M-step: Weighted centroid update
        for k in 1..K:
            centroid[k] = Σ_{n: c[n]=k} w[n] × K_sub[i][n,:] / Σ_{n: c[n]=k} w[n]
    
    // Output per subvector: centroids[i] ∈ R^{K×d/m}, indices[i] ∈ Z_K^N

// ===== Decode阶段: PQ-Based Attention =====
// 输入: q ∈ R^{1×d} (new token query)
// Key codebooks + indices, Value codebooks + indices

// Step 1: Query subvector splitting
q_sub[1..m] = split(q, m)  // m=32, each [1, d/m]

// Step 2: Inner Product Matrix (query × codebook)
for i in 1..m:
    IPM[i] = q_sub[i] × centroids_k[i]^T  // [1, d/m] × [d/m, K] = [1, K]

// Step 3: Lookup indices → sum → qK^T approximation
qKT_approx = zeros(1, N)
for i in 1..m:
    for n in 1..N:
        qKT_approx[n] += IPM[i][ indices_k[i][n] ]

// Step 4: Softmax + Value Reconstruction
attn = softmax(qKT_approx / sqrt(d_head))  // [1, N]
output = Σ_{i=1..m} Σ_{n=1..N} attn[n] × centroids_v[i][indices_v[i][n]]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PQ的经典开源实现包括Faiss [https://github.com/facebookresearch/faiss, Meta AI]的ProductQuantizer类（支持PQ/OPQ/IVFPQ等变体，用于ANN search）。LLM领域的PQ应用：PQCache [Zhang et al., 2024] 使用PQ identify important tokens for KV cache offloading，但保留full KV copy在CPU memory；Squeezed Attention [Hooper et al., 2024] 用PQ做sparse attention token selection。AQPIM首次directly使用PQ as KV source（无full KV copy），并in PIM执行online clustering和compressed attention computation。核心实现要点：(a) subvector count m和centroid count K的trade-off——m=32, K=512在LongBench上accuracy饱和；(b) 保留前8 sink tokens和最近32 sliding window tokens为full precision以维持accuracy；(c) clustering overhead通过PIM-GPU parallel prefill完全隐藏（codebook_gen latency < prefill_total，图4）；(d) OnlinePQ (progressive centroid update)未带来accuracy增益因此被省略。

涉及论文标题：
- AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization

## Importance-Weighted K-Means Clustering（重要性加权K-Means聚类）

术语是什么？通过联网搜索让回答具体和精准。
Importance-Weighted K-Means Clustering是AQPIM提出的K-means变体，将token的attention-based importance score作为权重融入clustering objective function。标准K-means最小化 sum||x_n - µ_{c(n)}||² 对所有token平等对待，但LLM attention中某些token（如sink tokens, high-attention tokens）始终接收高attention scores，quantization error对这些critical token的accuracy影响更大。AQPIM修改objective为 Σ w_n ||x_n - µ_{c(n)}||²，其中weight w_n = sum(S[-t:, n], axis=0) 是该token收到的最近t个token attention scores之和。M-step中centroid更新为weighted average: µ_k = Σ_{n∈C_k} w_n x_n / Σ_{n∈C_k} w_n，使高importance token对centroid位置影响更大，从而获得更小的quantization error。该技术与FlashAttention协同：attention scores S既用于attention计算又用于weight计算，额外overhead minimal。

从算法pipeline角度拆解术语：
```
// 输入: K_sub ∈ R^{N×d/m}, S ∈ R^{N×N} (attention score matrix)
// 参数: K=512 centroids, t=32 (weight window)

// Step 1: Compute importance weights from attention scores
w = zeros(N)
for j in max(0, N-t) .. N-1:  // last t tokens
    for i in 0 .. N-1:
        w[i] += S[j, i]

// Step 2: Weighted K-means iteration (4 rounds)
centroids = random_init(K, d/m)
for iter in 1..4:
    // E-step: Standard nearest-centroid assignment (no weighting)
    for n in 1..N:
        c[n] = argmin_k ||K_sub[n,:] - centroids[k,:]||²
    
    // M-step: Weighted centroid update
    for k in 1..K:
        weight_sum = Σ_{n: c[n]=k} w[n]
        centroids[k,:] = Σ_{n: c[n]=k} w[n] × K_sub[n,:] / weight_sum
```

术语一般如何实现？如何使用？
Weights w计算在GPU prefilling阶段完成——利用FlashAttention已计算的attention scores S，仅需额外sum操作（see Eq.1: w = sum(S[-t:, :], axis=0)）。Weighted M-step中的weighted sum在BankPE (FP16 MUL+SUM)和BufferPE (reciprocal计算→送回BankPE做final multiplication)上分布式执行。Ablation study显示(Table IV)：Standard PQ avg 44.29 vs w/o weighting 43.25 vs full AQPIM 50.00，importance weighting在aggressive compression (K=128)场景贡献约+0.81 avg points。t=32（sliding window大小）的选择平衡了weight stability和计算开销。

涉及论文标题：
- AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization

## Channel Pre-Sorting for PQ Vector Splitting（PQ向量分裂的通道预排序）

术语是什么？通过联网搜索让回答具体和精准。
Channel Pre-Sorting是AQPIM提出的PQ向量分裂预处理步骤，解决标准PQ split vectors不考虑inter-channel similarity导致高quantization error的问题。方法基于cosine similarity将高相关channel聚类到同一subvector中：随机选一reference channel→计算所有channel对其cosine similarity→greedily选择top-k most similar channels形成group→重复直至所有channel被分配。生成的sorting matrices P_k, P_v可以absorb到projection weights中：W_q'=W_q·P_k, W_k'=W_k·P_k, W_v'=W_v·P_v, W_o=W_o·P_v^T。矩阵离线生成（calibration dataset: Wikitext-2-v1），推理时零额外开销。与SKVQ的channel reorder方法类似但目标不同：SKVQ为minimize uniform quantization error within per-group scaling，AQPIM为maximize subvector内channel affinity for PQ clustering。

从算法pipeline角度拆解术语：
```
// 离线阶段: Channel Sorting Matrix Generation
// 输入: Calibration KV activation samples, d=4096 channels
// 输出: Permutation matrix P_k, P_v ∈ R^{d×d}

channels_remaining = {1, 2, ..., d}
groups = []  // m groups, each ~d/m channels
for g in 1..m:
    ref = random_choice(channels_remaining)
    similarities = []
    for ch in channels_remaining:
        similarities[ch] = cosine_similarity(K[:, ref], K[:, ch])
    top_k = argsort(similarities, descending)[:d/m]
    groups[g] = top_k
    channels_remaining -= top_k

// 构建permutation matrix: P_k[i,j]=1 if channel j belongs to group[i]
// 吸收到projection: W_k' = W_k @ P_k  // offline weight transformation
```

术语一般如何实现？如何使用？
Sorting matrices在calibration dataset (Wikitext-2-v1)上离线生成，作为static permutation absorb到model projection weights——在线inference完全透明。Ablation (Table IV): AQPIM w/o pre-sort avg 48.76 → full AQPIM avg 50.00 (+1.24 avg points)。Channel pre-sorting与importance-weighted clustering互补：pre-sorting提高subvector内cohesion (reduce intra-group quantization error)，weighting减少critical token的error。两者combined效果好于单独使用。

涉及论文标题：
- AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization

## Vision-Language Model (VLM)

术语是什么？通过联网搜索让回答具体和精准。
Vision-Language Model (VLM) 是一种多模态AI模型，能联合推理视觉和文本数据。现代VLM由vision encoder（如ViT）和Large Language Model (LLM) 组成：vision encoder将图像/视频帧切分为patches并tokenize为visual embeddings，经projection映射到LLM的word embedding space后与text tokens拼接，由LLM的Transformer自回归生成文本输出。代表性VLM包括LLaVA-OneVision、LLaVA-Video、MiniCPM-V、Qwen2.5-VL等。视频VLM中，视频被采样为帧，每帧独立tokenize；视觉tokens通常占输入的98-99%（如LLaVA-OneVision在VideoMME上平均6272 visual tokens vs 109 text tokens），LLM部分占99.35%参数和98.98%操作，因此VLM推理效率的核心瓶颈在LLM侧的视觉token处理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
VLM推理pipeline：`Video → Frame Sampling → Vision Encoder (per-frame patch tokenization) → Visual Embedding Projection → Concatenate with Text Tokens → LLM Transformer (Multi-Head Attention + FFN layers) → Text Generation`。以LLaVA-Video-7B为例：输入视频→采样为N帧→每帧被ViT编码为M_f个visual tokens→所有帧的visual tokens (M=N×M_f) 与text tokens (T) 拼接→送入LLM的32层Transformer→attention层计算QK^T softmax矩阵含四个block: image-to-image (M×M)、image-to-text、text-to-image (T×M)、text-to-text→FC/FFN层对拼接后的hidden states做GEMM→最终输出text tokens。VLM推理的计算量主要由M决定（M ≫ T），减少M是加速VLM的关键。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VLM通常通过HuggingFace Transformers加载预训练权重实现推理，使用lmms-eval等多模态benchmarking框架评估。部署场景包括云端GPU (如NVIDIA A100/H100)、边缘设备 (如Jetson Orin) 和专用加速器。由于visual tokens数量远大于text tokens，VLM推理优化重点在视觉冗余消除：token pruning（移除不重要的visual tokens）、token merging（合并相似tokens）、sparse attention等。Focus论文指出VLM的LLM部分占绝大多数参数和计算，因此优化LLM处理visual tokens的效率是加速VLM的核心。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

## Multilevel Concentration

术语是什么？通过联网搜索让回答具体和精准。
Multilevel Concentration是Focus论文提出的硬件导向的冗余消除范式，在三个粒度层次上层次化地压缩VLM的视觉-语言输入：(1) Semantic Level (token级)：基于cross-modal attention的语义引导token pruning，保留文本prompt相关的视觉区域；(2) Block Level (spatiotemporal block级)：在2×2×2时空窗口内做localized similarity comparison，限制匹配范围以保持streaming和高局部性；(3) Vector Level (sub-token vector级)：将token embedding切分为32维vectors做细粒度cosine similarity matching，捕获motion引起的partial alignment。三层concentration使Focus达到平均80.19% computation sparsity，仅1.20%平均accuracy degradation。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Multilevel Concentration在VLM推理pipeline中的执行流程：
```
# Level 1: Semantic Concentration (in attention layers)
for each attention layer at positions [3,6,9,18,26]:
    attn_scores = SoftMax(Q @ K^T)  # shape: (M+T) × (M+T)
    I = attn_scores[T:, :M]  # text-to-image block, T×M
    for each image token j in 1..M:
        s[j] = max_{i=1..T, k=1..n_heads} I[i,j]^{(k)}  # cross-modal importance
    top_k_indices = StreamingBubbleSort(s, k=M*retain_ratio)  # retain 40%→10%
    prune tokens not in top_k_indices

# Level 2 & 3: Block-level + Vector-level (in FC layers)
for each FC/PV/O_proj GEMM tile output (m=1024, n=32):
    vectors = output.reshape(m, 32)  # each row is a 32-dim vector
    for each 2×2×2 spatiotemporal block (8 vectors):
        key = vectors[block[-1]]  # highest-index vector as key
        for other_vec in block[:7]:
            cosine_sim = dot(key, other_vec) / (norm(key) * norm(other_vec))
            if cosine_sim > 0.9: mark as redundant
    store deduplicated vectors + similarity_map (1×m)
```
SEC pruning ratio逐层递减：layer 3保留40%、layer 6保留30%、layer 9保留20%、layer 18保留15%、layer 26保留10% image tokens。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Multilevel Concentration的算法实现在PyTorch中，硬件实现在SystemVerilog RTL中。算法层面使用HuggingFace Transformers集成，在attention和FC层之间插入concentration操作。硬件层面通过Focus Unit（含SEC和SIC两个子模块）嵌入systolic-array accelerator的memory interface，在GEMM tile产生后立即on-chip执行concentration。算法和硬件co-design是关键：算法sparsity（~80%）被转化为硬件友好的structured tile-local稀疏，而非留给GPU处理的irregular稀疏。开源：https://github.com/dubcyfor3/Focus。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

## Cross-modal Attention (Text-to-Image Attention for VLM Pruning)

术语是什么？通过联网搜索让回答具体和精准。
Cross-modal Attention在VLM中指text tokens与image tokens之间的attention交互。在Vision-Language Model的Transformer attention层中，QK^T SoftMax矩阵包含四个block：image-to-image (M×M)、image-to-text、text-to-image (T×M)、text-to-text。Text-to-Image block的每个元素I[i,j]表示第i个text token对第j个image token的attention score，反映了"语言查询"对"视觉内容"的关注程度。Focus利用这个cross-modal attention做prompt-aware token pruning：对每个image token j计算其从所有text tokens和attention heads中接收到的最大attention score作为importance指标，从而根据实际文本prompt动态选择相关视觉区域。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Cross-modal attention在VLM attention层的计算流程：
```
# Attention computation in VLM Transformer layer
Q = Linear_q(concat(image_tokens, text_tokens))  # (M+T) × d
K = Linear_k(concat(image_tokens, text_tokens))
V = Linear_v(concat(image_tokens, text_tokens))
S = Q @ K^T  # (M+T) × (M+T) attention scores
# S is partitioned into 4 blocks:
# S[0:M, 0:M]     → image-to-image
# S[0:M, M:M+T]   → image-to-text
# S[M:M+T, 0:M]   → text-to-image (cross-modal)
# S[M:M+T, M:M+T] → text-to-text

# SEC extracts text-to-image block for importance estimation
I = S[M:M+T, 0:M]  # T×M matrix
for each image token j:
    importance[j] = max_{i=1..T, k=1..n_heads} I^{(k)}[i,j]
# This captures how much ANY text token attends to image token j
# across ALL attention heads
```
当prompt问"What is the type of the dog?"时，attention集中在狗的位置；当问"What is the color of the flower?"时，attention转向花的位置。SEC利用这个prompt-dependent attention模式做动态token pruning。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Cross-modal attention是standard VLM Transformer attention的天然产物，无需额外计算。Focus的SEC通过从已计算的SoftMax(QK^T)中提取text-to-image block来获取cross-modal attention scores，不引入额外attention层或模块。在硬件实现中，SEC的importance analyzer从systolic array输出的attention SoftMax结果中直接stream text-to-image attention columns到parallel max units进行计算，完全on-chip、streaming。此方法可推广至任何使用cross-modal attention的VLM架构。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

## Token Pruning (Prompt-aware Token Pruning)

术语是什么？通过联网搜索让回答具体和精准。
Token Pruning是一种通过移除不重要tokens来减少VLM/LLM推理计算量的技术。传统token pruning方法（如Prumerge、FrameFusion）依赖静态heuristics如token magnitude、saliency或visual token间相似度，忽略文本prompt对token重要性的影响。Focus提出的prompt-aware token pruning利用cross-modal attention scores动态识别与当前prompt语义相关的visual tokens，保留semantically important tokens同时prune不相关tokens。Prune掉的tokens在后续P×V计算和下游层中不再加载，实现cumulative computation和memory access节省。保留比例逐层递减（40%→10%），深层layer prune更激进。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Prompt-aware token pruning流程：
```
# 在attention层中，SEC执行token pruning
attn_softmax = SoftMax(Q @ K^T / sqrt(d))  # (M+T)×(M+T)
cross_modal_I = attn_softmax[M:M+T, 0:M]  # text-to-image block
for j in 0..M-1:  # for each image token
    importance[j] = max(cross_modal_I[:, j])  # max over text tokens
    for head in 1..n_heads:
        importance[j] = max(importance[j], max(cross_modal_I_head[head][:, j]))

# streaming top-k selection
top_k_indices = streaming_bubble_sort(importance, k=retain_ratio*M)
retained_tokens = image_tokens[top_k_indices]

# pruned tokens excluded from downstream:
# P(i)×V only loads retained_tokens (not full M tokens)
# FC layers only process retained_tokens
```
Pruning ratio per layer: [3: 40%, 6: 30%, 9: 20%, 18: 15%, 26: 10%] of original M image tokens retained。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
算法实现：在PyTorch中hijack VLM的attention层，从SoftMax输出中提取text-to-image attention block，计算importance vector并做top-k mask，将prune后的token indices传到下游层。硬件实现：Focus的SEC模块将importance analyzer（parallel max units + 25KB buffer）、streaming bubble sorter和offset encoder集成到systolic-array accelerator的attention pipeline中。与静态pruning（如FrameFusion固定70% sparsity）相比，prompt-aware pruning使Focus在更高sparsity下（82.82%）保持更好accuracy（62.74 vs original 64.15, -1.41）。开源实现见https://github.com/dubcyfor3/Focus。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

## Vector-wise Similarity Matching

术语是什么？通过联网搜索让回答具体和精准。
Vector-wise Similarity Matching是Focus提出的细粒度冗余检测技术，将token embedding（典型维度如3584）切分为多个32维vectors，在small spatiotemporal block内对vectors做localized cosine similarity comparison。与token-wise matching（比较整个3584维token embedding）相比，vector-wise matching能揭示更多冗余：论文对LLaVA-OneVision的MLVU数据集分析显示，64%的8维vectors cosine similarity超过0.9，而仅18%的3584维full tokens超过0.9。这是因为motion导致的partial alignment（一个token可能部分匹配多个邻近token的不同部分）只能通过sub-token级别的比较捕获。Focus用2×2×2时空block限定比较范围，每个block内最高index vector作为key与其他7个vectors比较，匹配则记录代表vector的index。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Vector-wise similarity matching流程：
```
# After GEMM produces an m×n output tile (m=1024, n=32)
vectors = tile.reshape(1024, 32)  # 1024 vectors of 32 dimensions each
# Build 2×2×2 spatiotemporal blocks
for t in [frame_A, frame_B]:
    for h in 0..H-1 step 1:
        for w in 0..W-1 step 1:
            block = [vectors[t,h,w], vectors[t,h,w+1],
                     vectors[t,h+1,w], vectors[t,h+1,w+1],
                     vectors[other_frame_t,h,w], ...]  # 8 vectors
            key = block[-1]  # highest index vector
            for v in block[0:7]:
                dot_prod = sum(key[p] * v[p] for p in 0..31)
                cos_sim = dot_prod / (L2_norm[key] * L2_norm[v])
                if cos_sim > 0.9:
                    similarity_map[v_idx] = representative_idx(key)
                    # v is redundant, reuse key's index
deduplicated = vectors[unique_indices]  # p vectors, p < 1024
```
L2-norm per vector可precompute并存储在buffer中，使matching仅需1次dot-product和少量element-wise operation。

Granularity trade-off：更小vector（如8-dim）揭示更多冗余但增加comparison和metadata开销；更大vector（如128-dim）减少comparison但降低sparsity。Focus选vector length=32作为综合平衡点。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

---

## KV Cache Retrieval（KV缓存检索）

术语是什么？通过联网搜索让回答具体和精准。

KV Cache Retrieval是一种通过将完整KV cache offload到CPU memory或storage，并在推理时选择性fetch相关token来降低GPU memory占用的技术。与pruning、compression、quantization等破坏性方法不同，KV cache retrieval保留全部历史KV cache，仅在每层attention前动态选择最相关token子集取回GPU memory参与计算。典型代表包括FlexGen（offload to CPU/storage）、InfiniGen（generation-stage retrieval）、ReKV（frame-level selection）和V-Rex的ReSV（dynamic per-layer/head retrieval）。核心trade-off是PCIe带宽（4-32 GB/s）远低于GPU memory带宽（1-2 TB/s），因此必须通过高效选择算法最小化fetch volume。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

KV Cache Retrieval在streaming video LLM中的三阶段pipeline（以FlexGen baseline为例）：

```
// Stage 1: Offloading
for each new KV cache entry generated:
    if GPU_memory_used > threshold:
        offload oldest KV entries to CPU memory or storage
        // 完整KV cache永不丢弃，仅迁移到低速存储

// Stage 2: Selection (KV Prediction)
for each decoder layer L:
    Q, K_new, V_new = QKV_Gen(input_hidden)
    // 计算query与历史key的attention score
    scores = Q @ K_history^T           // 在GPU上执行
    top_k_indices = topk(scores, k)    // fixed k selection
    // 固定top-k：所有layer/head用相同k值

// Stage 3: Pre-fetching
    selected_K = fetch_from_cpu(top_k_indices)  // PCIe传输
    selected_V = fetch_from_cpu(top_k_indices)
    // 将选中的K/V prefetch到GPU memory
    attention_out = Attention(Q, selected_K, selected_V)
```

关键特征：(a) Preserve context integrity：完整KV cache始终保留于CPU/storage，未来query可访问任意历史token，支持multi-turn对话。(b) Selective computation：仅对选中的KV子集计算attention，减少计算量。(c) PCIe bottleneck：fetch受PCIe带宽限制（4-32 GB/s vs GPU memory 1-2 TB/s），retrieval latency在streaming video 40K cache length下可达85%总延迟。(d) Selection overhead：KV prediction computation本身也随sequence length增长。

固定top-k的缺陷：token importance在不同layer和head间分布高度不均（有的layer仅需4.2% token，有的需44.0%），固定k导致不重要位置over-fetch浪费PCIe带宽、关键位置under-fetch降低accuracy。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FlexGen (https://github.com/FMInference/FlexGen) 将KV cache offload到CPU memory和SSD的三级存储层次（GPU→CPU→SSD），使用线性规划优化offloading schedule。InfiniGen在generation stage做KV cache retrieval，异步prefetch隐藏fetch latency。V-Rex的ReSV通过hash-bit key clustering和WiCSum thresholding实现动态per-layer/head token selection，但未开源。使用流程：模型加载→配置offloading target→设置token budget或selection ratio→每层decoder前计算KV prediction→通过PCIe fetch selected KV→执行attention。

涉及论文标题：
- V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval

---

## Iterative Prefill for Streaming Video LLM（流式视频LLM的迭代预填充）

术语是什么？通过联网搜索让回答具体和精准。

Iterative Prefill是streaming video LLM特有的推理阶段，区别于传统LLM的one-shot prefill和offline video LLM的batch processing。在streaming video LLM中，视频帧实时到达且无法batch（帧按时序到达），每个frame依次经过Vision Tower→MLP Projector→所有LLM decoder layers，每层生成新KV cache并逐层累积。KV cache以O(N²T)复杂度增长（N²为spatial resolution，T为temporal duration），prefill在每个新frame到达时重复执行，成为端到端延迟主要贡献者（80K cache length时占83% end-to-end latency）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Iterative Prefill的pipeline（以VideoLLM-Online + Llama-3 8B为例）：

```
while video_stream_active:
    frame_t = capture_frame()                    // 实时捕获帧
    vision_embed = VisionTower(frame_t)          // SigLIP-ViT-L-384
    projected = MLP_Projector(vision_embed)

    // Iterative Prefill: 逐层处理当前帧
    for layer L = 0 to N-1:
        Q, K_new, V_new = QKV_Gen(projected)
        // 对完整历史KV cache (所有之前帧) self-attention
        attention_out = Attention(Q, [K_history | K_new],
                                    [V_history | V_new])
        K_history.append(K_new); V_history.append(V_new)
        hidden = FFN(attention_out)
        projected = hidden
    // KV cache增长: |K_history| += tokens_per_frame × layers
    // 10 FPS × Llama-3 8B → 数分钟内超过32GB edge GPU capacity

// 用户query到达时:
question_tokens = Tokenize(user_query)
for layer L = 0 to N-1:
    output = DecoderLayer(question_tokens, K_history, V_history)
generation = autoregressive_decode(output)       // Generation Stage
```

与标准LLM prefill的关键区别：(a) Prefill非一次性——每个新视频帧触发完整prefill pass。(b) KV cache增长无界——随视频时长线性增长。(c) Frame间不能batch——帧按时序到达无法批量处理。(d) 后续query依赖历史——用户可能询问早期视频内容，不能简单丢弃旧KV cache。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

VideoLLM-Online (https://github.com/OpenGVLab/VideoLLM-Online) 是streaming video LLM的开源实现，支持实时视频流输入和多轮对话。实现使用asyncio管理帧到达与模型推理的异步流水线。V-Rex在此基础上增加KV cache retrieval pipeline（ReSV + DRE），offload完整KV cache并按需retrieve。使用流程：部署streaming video LLM→配置vision tower和LLM backbone→设置frame sampling rate→启动视频流→每帧触发iterative prefill→KV cache管理策略决定retention/offloading/retrieval。

涉及论文标题：
- V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval

---

## Hash-bit Key Clustering（哈希比特键聚类）

术语是什么？通过联网搜索让回答具体和精准。

Hash-bit Key Clustering是V-Rex ReSV算法的核心组件，通过随机hyperplane projection将高维key向量降维并二值化为hash-bit（仅≤0.5%原始dimension），用XOR+popcount计算Hamming distance替代cosine similarity做token聚类。利用视频相邻帧key token的高时空相似性（cosine similarity热力图验证），将相似token归入同一cluster。Hamming distance与cosine similarity相关性约0.8。关键优势：(1) bit-wise操作为硬件友好，避免浮点乘加；(2) 保留原始token value供后续attention（区别于merge/replace方法）；(3) 聚类后仅对representative KeyCluster而非完整key cache做后续computation。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Step 1: Hash-bit Generation (每个新frame执行一次)
Hyperplanes = RandomMatrix(N_hp=32, key_dim)
for each key token t in current frame:
    Key_hp[t] = Key[t] @ Hyperplanes^T     // N_hp=32维投影
    Hash_bit[t][i] = (Key_hp[t][i] > 0) ? 1 : 0  // 二值化

// Step 2: Hamming Distance Clustering
HC_table = load_existing_clusters()  // {cluster_id, KeyCluster, hash, token_count}
for each current token t:
    curr_hash = Hash_bit[t]
    for each existing cluster c:
        dist = popcount(curr_hash XOR HC_table[c].hash)
        if dist < Th_hd (论文设7):
            assign t to cluster c
            HC_table[c].token_count++
            HC_table[c].KeyCluster = mean(tokens in cluster c)
    if no cluster matched:
        create new cluster with token t

// HC_table metadata overhead: avg 1.67% of full KV cache
// avg 32 tokens per cluster
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在V-Rex中，HCU（Hash-bit Cluster Unit）硬件加速clustering：current hash-bit memory + key cache hash-bit memory + NHCU_h个并行XOR accumulator（NHCU_w=16 inputs）。LEE的VPE生成hash-bit→HCU读取已有cluster hash-bit→XOR accumulator并行计算Hamming distance→与Th_hd比较→HC table updater更新metadata。通用软件实现可用NumPy bitwise XOR: `numpy.bitwise_xor(curr, cluster).sum(axis=-1)` 计算Hamming distance，配合threshold筛选cluster。论文未开源。

涉及论文标题：
- V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval

---

## WiCSum Thresholding（加权累积和阈值选择）

术语是什么？通过联网搜索让回答具体和精准。

WiCSum（Weighted Cumulative Sum）Thresholding是V-Rex提出的动态token选择算法，替代fixed top-k。对每layer/head独立计算Query×KeyCluster^T得ScoreCluster矩阵，按每行score×token_count加权求和得weighted sum，从高分bucket开始累积直到超过动态阈值（Th_wics = 0.3 × weighted sum）即停止。结果：每layer/head自适应选择不同数量token（论文Fig.20：layer间4.2%-44.0%，head间也有显著差异），平均比ReKV少检索3.0× token，accuracy仅下降0.8%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Input: ScoreCluster [num_queries, num_clusters]
//        HC_table: {token_count per cluster}
// Thr_wics = 0.3

// Step 1: Preprocess per row
for each row i:
    Sum_i = Σ_j (ScoreCluster[i][j] * HC_table[j].token_count)
    Th_wics_i = Sum_i * Thr_wics

// Step 2: Early-Exit Token Selection per row
sorted = sort_descending(ScoreCluster[i])
Acc_i = 0; selected = []
for t = 0 to num_clusters-1:
    Acc_i += sorted[t].score * HC_table[sorted[t].cluster].token_count
    selected.append(sorted[t].cluster)
    if Acc_i > Th_wics_i:
        break  // Early exit: avg only 16% of row processed

// Step 3: Aggregate
all_selected = unique(∪ selected_i for all rows)
selected_tokens = map_to_tokens(all_selected, HC_table)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在V-Rex中，WTU（WiCSum Threshold Unit）硬件加速thresholding：每core含score memory/token count memory/upper-lower bucket sorters/multipliers/adder tree/bucket range updater。Preprocess step预计算weighted sum和threshold→Token selection step从高分bucket做bucket sort→cumulative sum→与threshold比较→early exit。通用PyTorch实现核心~50行：`cumsum = (sorted_scores * token_counts).cumsum(dim=-1); mask = cumsum <= threshold; selected = sorted_indices[mask]`。论文未开源。

涉及论文标题：
- V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval

## AI Agent / Dynamic Reasoning（AI智能体/动态推理）

术语是什么？通过联网搜索让回答具体和精准。

AI Agent是基于LLM的推理时框架，通过多步推理、自适应决策和与外部环境的交互来扩展LLM的能力。与传统的单轮LLM推理（输入→静态prompt→输出）不同，AI Agent执行动态推理（Dynamic Reasoning）：在每个iteration中，agent可能生成中间推理结果、调用外部工具（如搜索引擎、计算器、代码解释器），并将工具返回结果纳入后续决策中。这个过程允许agent动态获取缺失信息、根据任务需求调整策略。AI Agent通常包含四个核心组件：(1) Agent Core：由LLM担任actor/planner/reflection等角色；(2) Memory：存储短期交互轨迹和长期知识；(3) Plan：将目标分解为子任务序列或DAG；(4) Tools：扩展能力的外部接口（如Wikipedia API、Wolfram Alpha、Python executor等）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

AI Agent的算法pipeline是LLM inference和tool execution交替迭代的过程：

```
// AI Agent 单请求执行pipeline
function AgentExecute(query, workflow_type):
    context = [Instruction, FewShot, Query]
    history = []
    
    while not termination_condition():
        // Phase 1: LLM Inference
        prompt = concat(context, history)
        llm_output = LLM_Backend.generate(prompt)  // prefill + decode
        
        if llm_output.is_final_answer():
            return llm_output
        
        // Phase 2: Parse Action
        action = parse_action(llm_output)  // 解析thought/action
        
        // Phase 3: Tool Execution
        tool_result = Tool_Executor.execute(action.tool, action.params)
        // GPU可能在tool执行期间idle
        
        // Phase 4: Update Context
        history.append((llm_output, tool_result))
```

不同agent workflow的区别在于termination condition和action空间：
- **CoT**：无tool调用，仅内部reasoning，单次LLM inference
- **ReAct**：交替reasoning和tool use，直到达到final answer或iteration limit
- **Reflexion**：在ReAct基础上周期性插入self-evaluation和refinement
- **LATS**：Monte Carlo Tree Search，每个tree node展开时发出多个并行LLM calls评估候选path
- **LLMCompiler**：先用planner构造DAG任务依赖，再streaming async执行tool calls

论文测量：tool-augmented agent平均LLM calls是CoT的9.2倍，LATS平均71.0次LLM calls/request。LLM inference和tool execution分别占总延迟约69.4%和30.2%，但由于LLM输出决定下一步调用哪个tool，两者难以重叠（LLMCompiler的DAG并行仅占总延迟18.2% overlap）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

AI Agent通过agent framework实现：将LLM backend（vLLM/OpenAI API等）与tool system连接。常见实现方式：(1) agent worker（Python async process）维护state machine，根据workflow决定下一步是LLM call还是tool call；(2) LLM backend负责prefill+decode；(3) tool system执行本地代码（Python interpreter）或外部API（Wikipedia/Wolfram Alpha/web browsing）；(4) 开源框架包括LangChain、AutoGen、CAMEL等。论文使用各agent原作者的开源实现，统一整合到AgentBench框架（https://github.com/VIA-Research/AgentBench）：ReAct (github.com/ysymyth/ReAct)、Reflexion (github.com/noahshinn/reflexion)、LATS、LLMCompiler (github.com/SqueezeAILab/LLMCompiler)，统一适配到vLLM backend（vLLM 0.6.6, PyTorch 2.6, CUDA 12.8）。

涉及论文标题：
- The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective
- AIMS: A Cost-Efficient Framework for LLM-based Agent Deployment in Cloud-Edge Hybrid Environments

## Test-Time Scaling（测试时扩展）

术语是什么？通过联网搜索让回答具体和精准。

Test-Time Scaling是指在推理时通过增加计算量来提升预训练LLM的推理性能，而不修改模型参数的方法论。代表性技术包括Chain-of-Thought（引导模型生成中间推理步骤）、Tree-of-Thought（探索多条推理路径）、Self-Consistency（采样多个response选最优）等。AI Agent将test-time scaling进一步推进为dynamic reasoning：不仅通过prompt设计增强推理，还通过多步决策集成tool use和维护中间推理状态。这使得agent能根据中间结果动态调整行为，实现对外部环境的自适应。Test-time scaling的核心trade-off：增加计算可提升accuracy，但存在diminishing returns和不可持续的基础设施成本。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Test-time scaling在AI Agent中有两种基本形式：

**Sequential Scaling（顺序扩展）**：逐步增加推理步数
```
for step in range(iteration_budget):
    result = agent_step(context)
    if result.is_final: break
// 延迟随步数线性增长，峰值资源需求低
```

**Parallel Scaling（并行扩展）**：同时探索多条推理路径
```
// LATS tree expansion
children = []
for i in range(num_child_nodes):
    children.append(async_llm_call(context))  // 并行LLM calls
results = await gather(children)
best = select_best(results)
// 可同时降低延迟和提升准确率，但增加瞬时GPU memory和serving contention
```

论文核心发现：test-time scaling存在急剧递减的边际收益。Reflexion从16.9s→25.6s仅获4% accuracy gain；从56.0s→325.5s需31× cost获得同等marginal gain。Parallel scaling (LATS)增加child nodes从1→16可提升14.4pp准确率同时降低196.3s平均延迟（因为更快找到高质量path），但代价是更多并发LLM requests。8B模型配合LATS parallel scaling可接近70B模型性能但energy更低。单次agent query的GPU energy比ShareGPT高62.1×–136.5×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Test-time scaling通过agent framework的参数配置实现：(1) iteration budget：控制最大推理步数和tool调用次数，限制sequential scaling的范围；(2) few-shot examples：适量增加可同时提升准确率并减少推理步数，但过多prompt tokens可能超过模型最优处理区间导致accuracy下降；(3) reflection depth：Reflexion中self-evaluation的轮数；(4) child nodes per expansion：LATS中Monte Carlo Tree Search每步并行探索的分支数；(5) model size scaling：更大模型用更少步骤达到高准确率但energy/query大幅增加（8B→70B：单GPU→8 GPU，单query energy增加约8×–12×）。实现上需在agent worker的state machine中设置stop condition和branching logic。

涉及论文标题：
- The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

## Best-of-N (Outcome Reward Model-based Sampling)

术语是什么？通过联网搜索让回答具体和精准。

Best-of-N是一种parallel test-time scaling方法，在decode阶段生成N条完整的候选completion，然后使用Outcome Reward Model (ORM)或外部verifier对每条候选打分，选择得分最高的作为最终输出。与Beam Search在中间步骤剪枝不同，Best-of-N仅在生成结束后做一次性选择。在数学推理等可通过自动验证检查答案正确性的任务中特别有效。Best-of-N的compute cost与N成正比（每条候选独立生成），但N条候选的decode可以在一个batch中并行执行，利用hardware的batch parallelism。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
Input: prompt P, model M, generation budget N, reward model RM
Output: best completion C*

candidates = []
for i = 1 to N:
    C_i = M.generate(P, temperature=T)  // 独立采样N条
    candidates.append(C_i)

for each C_i in candidates:
    score_i = RM(P, C_i)   // ORM评分，如Skywork-PRM
C* = argmax(score_i)
return C*
```

在Mobile NPU上，Best-of-N的N条候选映射到decode batch，填充HMX 32×32 tile的N行（vs 单路径decode仅1行有效）。论文实测：Qwen2.5-1.5B在MATH500上，N=16时accuracy可达~70%（超越3B base model ~60%），同时per-token decode latency仅轻微增加（HMX计算时间几乎不随batch增大而增加）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Best-of-N的实现要点：(1) Reward model选择：数学推理可用outcome verification（答案匹配），也可用训练好的ORM如Skywork-1.5B-PRM；(2) Temperature设置：需要非零temperature以产生多样化候选；(3) Batch parallelization：将N条候选的decode组织为batch_size=N的推理，利用hardware tile-level parallelism；(4) 在移动NPU上，lm_head/logits常保留在CPU（32-bit虚拟地址空间限制），batch增大时CPU logits计算可能成为瓶颈（B=16时>50%时间）。开源实现见llama.cpp-npu。

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

## Process Reward Model (PRM) for Step-Level Beam Search

术语是什么？通过联网搜索让回答具体和精准。

Process Reward Model (PRM)是一种对LLM生成过程的中间步骤（而非最终结果）进行评分的reward model。与Outcome Reward Model (ORM)仅评估最终completion正确性不同，PRM为每个generation step（如数学推理中的一行推导）输出一个process score，指示该步骤是否正确或向正确答案前进。PRM使Beam Search等step-level搜索算法可以在中间步骤剪枝低质量路径，而不必等待完整生成。典型PRM如Skywork-1.5B-PRM，在每个generated token或step boundary处输出[0,1]的process score。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Step-level Beam Search with PRM:
```
Input: prompt P, model M, beam width W, max steps S, PRM
Output: best completion

beam = {empty_path}  // 每条path含generated tokens + cumulative score
for step = 1 to S:
    candidates = []
    for each path in beam:
        next_token = M.generate_next(path)  // 生成一个step的token(s)
        new_path = path + next_token
        process_score = PRM(P, new_path)   // PRM评分当前步骤
        new_path.cumulative_score *= process_score  // 累积
        candidates.append(new_path)
    beam = top_W(candidates, key=cumulative_score)  // 剪枝到W条
    if all paths in beam are complete: break
return beam[0]
```

论文在Snapdragon平台上使用Skywork-1.5B-PRM实现step-level Beam Search。在decode阶段，beam中的W条路径映射到batch_size=W，同样利用HMX的tile空行。每step结束后PRM打分并剪枝，保留top-W高质量路径。Beam Search中Qwen2.5-1.5B和Llama3.2-1B可达到与各自3B版本相当或略优的accuracy-cost效率。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

PRM的训练：在人工标注的process-level supervision数据上fine-tune（标注每个推理步骤的正确性），输出为[0,1]分数。使用时：(1) 确定step boundary（如每句/每行/每个逻辑段落）；(2) 在step boundary处调用PRM评分；(3) 累积分数（乘积或求和）用于路径比较和剪枝。开源PRM如Skywork-PRM系列，通过HuggingFace Transformers加载。注意PRM评分本身引入额外compute overhead（每次step boundary需额外forward pass）。

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

## Fine-grained Group Quantization (W4A16)

术语是什么？通过联网搜索让回答具体和精准。

Fine-grained Group Quantization是一种LLM权重量化方法，将权重矩阵按较小的group（如group size=32或128）分组，每组独立计算scale和zero-point进行量化。与per-tensor/per-channel量化不同，group quantization的量化参数更精细（group内统计更均匀），因此能在4-bit等低比特下保持更高模型精度。典型方案如GPTQ、AWQ、llama.cpp的Q4_0均采用group quantization。W4A16表示weight 4-bit、activation 16-bit（FP16）的混合精度配置，activation保持浮点避免精度损失，仅在compute前做runtime weight dequantization。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

以Q4_0 symmetric quantization (group size=32)为例：
```
// Quantize: weight matrix W (FP16) → quantized W_q (INT4) + scales (FP16)
for group g in range(num_groups):  // 沿column dim, 每组32个元素
    max_abs = max(abs(W[g*32 : (g+1)*32]))
    scale = max_abs / 7.0            // Q4_0: [-7,7] range, symmetric
    W_q[g*32 : (g+1)*32] = round(W[g*32 : (g+1)*32] / scale)
    scales[g] = scale

// Dequantize at runtime:
for group g:
    W_fp16[g*32 : (g+1)*32] = W_q[g*32 : (g+1)*32] * scales[g]
// Then perform FP16 GEMM: O = A × W_fp16
```

在移动NPU上，QNN仅支持per-tensor/per-channel粗粒度量化：Llama3.2-1B-Instruct W4A16在MATH500上的accuracy从AutoAWQ group quantization的15.9骤降至2.1，证明fine-grained quantization对数学推理任务至关重要。论文通过hardware-aware tile quantization将group quantization的layout重排为HMX tile format，既保留精度又适配硬件。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：llama.cpp的GGUF格式原生支持Q4_0、Q4_1、Q5_0、Q8_0等多种group quantization格式，通过llama-quantize工具转换。AWQ (AutoAWQ)通过calibration data计算per-channel scaling factor并以group方式存储。GPTQ使用二阶信息逐列量化。在runtime，dequantization通常由GEMM kernel内的vector/SIMD指令完成：先加载scale→broadcast→乘quantized values→accumulate。开销取决于group size（越小越精细但dequant overhead越大）。

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

## ReAct (Synergizing Reasoning and Acting)

术语是什么？通过联网搜索让回答具体和精准。

ReAct（Reasoning + Acting）是一种AI agent workflow，将LLM的推理能力和行动能力交替结合。在每个step中，LLM生成一个thought（推理当前状态和下一步计划）和一个action（指定要调用的工具和参数）；系统执行action后返回observation（工具执行结果）；observation被追加到context后进入下一轮thought/action循环。ReAct agent持续这个循环直到生成final answer或达到iteration limit。ReAct是所有tool-augmented agent（Reflexion、LATS）的基础workflow。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

ReAct单步pipeline（以HotpotQA Wikipedia search为例）：

```
Prompt = "You are an assistant solving QA tasks.
  You have access to Wikipedia search and lookup tools.
  Use Thought/Action/Observation format."

当前Context = [Prompt, UserQuery, History(LLM_outputs, Tool_results)]
llm_output = LLM(当前Context)
// llm_output格式: "Thought: <reasoning>\nAction: <tool>[<params>]\n"
// 或 "Thought: <reasoning>\nFinal Answer: <answer>"

action, params = parse(llm_output)
if action == "Final Answer":
    return params
else:
    tool_result = execute_tool(action, params)  // e.g. Wikipedia API (~1.2s)
    History.append(("Action: " + action, "Observation: " + tool_result))
    goto next_step  // input tokens增长（LLM history + Tool history）
```

论文测量：ReAct的LLM和tool call数量相近，tool latency因workload而异——WebShop本地web工具约20ms/call，HotpotQA Wikipedia API约1.2s/call。ReAct在HotpotQA的95th percentile latency为20.7s（vs ShareGPT 9.7s），WebShop 50.8s。prefix caching对ReAct的多轮LLM call特别有效（prefill latency降低60.1%）。ReAct在accuracy-latency trade-off中cost-efficiency表现均衡。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

ReAct通过prompt engineering实现——在system prompt中定义Thought/Action/Observation格式，LLM按此格式生成输出。实现要点：(1) few-shot examples展示标准推理-行动-观察循环；(2) action space定义可用工具及其参数格式（如Wikipedia Search["query"]、Wikipedia Lookup["keyword"]）；(3) output parser提取action并路由到对应tool executor；(4) stop condition检测"Final Answer"或达到max iterations。开源实现：https://github.com/ysymyth/ReAct。论文将其适配到vLLM backend：agent worker异步运行，tool执行期间GPU可被其他concurrent request的LLM call填补（inter-request parallelism）。

涉及论文标题：
- The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective

## LATS (Language Agent Tree Search)

术语是什么？通过联网搜索让回答具体和精准。

LATS（Language Agent Tree Search）是一种基于Monte Carlo Tree Search (MCTS)的AI agent workflow。与ReAct的顺序单路径推理不同，LATS在每步扩展时生成多个候选reasoning/action child nodes（通过并行LLM calls），模拟多条可能的推理路径，并使用MCTS的selection-expansion-simulation-backpropagation循环选择最优路径。LATS的tree search机制使得agent能在多个candidate paths中比较和选择，提升了决策质量但显著增加了LLM call数量（论文测量平均71.0次/request）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// LATS单请求执行pipeline (Monte Carlo Tree Search)
root = Node(state=initial_context)
for iteration in range(max_iterations):
    // Selection: 从root沿tree用UCT选择最有希望的leaf node
    node = select(root, policy=UCT)
    
    // Expansion: 为当前node生成多个child nodes
    children = []
    for i in range(num_child_nodes):  // Parallel scaling参数
        prompt = format_prompt(node.trajectory)
        child_output = async_llm_call(prompt)  // 并行LLM calls
        children.append(parse_node(child_output))
    node.children = children
    
    // Simulation/Evaluation: 评估各child node的value
    for child in children:
        child.value = evaluate(child)  // LLM self-evaluation
    
    // Backpropagation: 更新path上所有node的统计信息
    backpropagate(node, children)
    
// 返回best trajectory
return get_best_trajectory(root)
```

论文核心发现：LATS平均LLM calls/request最高（71.0次），parallel scaling增加child nodes从1→16可提升14.4pp准确率同时降低196.3s平均延迟（更快找到高质量path）。但代价是更多并发LLM requests→增加GPU memory pressure和KV cache占用。prefix caching可平均降低64.8% LATS memory requirement（共享prefix复用）。LATS只保留root→current node path（不concat全部历史），限制了context length增长。8B LATS HotpotQA accuracy达80%，energy/query 22.76 Wh（ShareGPT 0.32 Wh的71.7×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LATS实现基于MCTS算法框架：(1) tree state维护node的visit count和value estimate；(2) UCT (Upper Confidence Bound for Trees) formula用于selection阶段平衡探索与利用；(3) 论文优化了LATS原实现以支持concurrent LLM inference和parallel tool invocation（原版顺序执行加重延迟）；(4) 每个node的trajectory仅包含root→当前node的path（不concat全部历史），控制context length。开源：论文基于LATS官方实现优化（https://github.com/VIA-Research/AgentBench）。LATS在accuracy上优于ReAct和Reflexion，但computational overhead最高。

涉及论文标题：
- The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective

## LLMCompiler (Structured Planning for Parallel Function Calling)

术语是什么？通过联网搜索让回答具体和精准。

LLMCompiler是一种基于结构化多步规划的AI agent workflow。与ReAct的交替thought-action循环不同，LLMCompiler先用planner分析任务依赖并构造一个DAG（有向无环图），将多个tool calls组织成可执行计划。计划生成过程中，中间tool calls可以streaming到execution stage，让scheduler异步执行工具，从而将部分planning和tool execution overlap，降低端到端延迟。这种"plan-then-execute"范式减少了整体LLM call次数。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// LLMCompiler执行pipeline
function LLMCompiler_Execute(query):
    // Phase 1: Planning (单次LLM call)
    plan_dag = Planner_LLM(query)  
    // 输出: DAG nodes = tool calls with dependencies, edges = precedence
    
    // Phase 2: Streaming Execution (与planning重叠)
    scheduled_tasks = []
    for node in plan_dag.topological_order():
        if node.dependencies_met():
            task = async_execute_tool(node.tool, node.params)
            scheduled_tasks.append(task)
    
    // Phase 3: Join results
    results = await gather(scheduled_tasks)
    
    // Phase 4: Final answer generation (第二次LLM call)
    final_context = [plan_dag, results]
    return LLM(final_context)
```

论文测量：LLMCompiler的planning-tool overlap仅占总延迟约18.2%（受任务依赖关系限制）。在HotpotQA上LLMCompiler在accuracy和cost-efficiency上均优于ReAct（结构化规划减少重复推理）。但在WebShop上因tool使用涉及高互依赖（搜索→点击→翻页），DAG-style planning导致不必要的tool invocations，效率低于ReAct。LLMCompiler不适合MATH/HumanEval等需要强顺序推理的任务。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LLMCompiler核心组件：(1) Planner LLM：分析任务、识别tool依赖、生成DAG计划（单次LLM call生成完整计划）；(2) Task Fetching Unit：按DAG topological order stream task到execution stage；(3) Executor：异步执行tool calls（支持并行执行无依赖tasks）；(4) Joiner：收集tool results并format成final LLM prompt。论文将其适配到AgentBench统一框架，使用vLLM backend。开源实现：https://github.com/SqueezeAILab/LLMCompiler。LLMCompiler的关键优势是单次planning LLM call生成完整计划，减少整体LLM call次数和重复推理。

涉及论文标题：
- The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective

## Tool-Augmented Reasoning（工具增强推理）

术语是什么？通过联网搜索让回答具体和精准。

Tool-Augmented Reasoning是AI Agent区别于静态LLM推理的核心能力。在tool-augmented reasoning中，agent不仅进行内部语言推理（reasoning），还调用外部工具（tool use）获取实时数据、执行非语言操作（计算、代码执行、web搜索）。工具调用结果以observation形式返回，被纳入agent上下文中指导后续推理。这种reasoning-acting-observing循环使agent能处理超出LLM训练数据覆盖范围的任务。论文量化了这一范式的系统成本：LLM inference占总延迟69.4%、tool execution占30.2%，两者因sequential dependency难以overlap。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Tool-augmented reasoning的依赖链
LLM_output = f(context)           // 输出包含tool selection和parameters
Tool_result = execute(tool, params)  // 必须等LLM_output确定tool和params
Next_LLM_input = concat(context, LLM_output, Tool_result)  // 必须等Tool_result
// → LLM inference和tool execution难以并行（数据依赖）
```

论文量化了系统成本：(1) Tool latency因类型差异大：Wikipedia API ~1.2s/call, WebShop本地web ~20ms/call, Wolfram Alpha API中等；(2) GPU在tool执行期间最多54.5% idle time（HotpotQA/MATH的CPU/外部tools）；(3) Tool history tokens是context膨胀的主要来源——知识密集型任务中tool返回大段文本；(4) LLMCompiler通过DAG planning尝试overlap planning和tool execution，但仅实现18.2% overlap。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Tool system实现：(1) Tool定义：每个tool有name、description、parameters schema（供LLM理解）；(2) Tool Executor：解析LLM output中的tool call→验证参数→执行（本地Python函数或外部API）→返回结构化result；(3) Tool可以用OpenAI function calling API、Anthropic MCP协议或Google A2A协议标准化；(4) 本地tools（code interpreter, calculator）延迟低不占外部带宽，远程tools（Wikipedia API, Wolfram Alpha）延迟高有网络依赖。论文在AgentBench中集成Wikipedia search/lookup APIs、WebShop navigation、Wolfram Alpha API、Python executor等tools。

涉及论文标题：
- The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective

---

## Reasoning Phase / Answering Phase（推理阶段/回答阶段：Reasoning-Based LLM解码阶段划分）

术语是什么？

Reasoning phase和answering phase是reasoning-based LLMs（如DeepSeek-R1、OpenAI o3）解码阶段内部的两个功能上不同的子阶段。Reasoning phase生成hidden Chain-of-Thought (CoT) reasoning tokens（对用户不可见），answering phase生成user-visible answering tokens。这与conventional LLMs形成根本区别：conventional LLMs的decode阶段仅生成user-visible tokens，而reasoning-based LLMs将decode阶段内部划分出reasoning和answering两个语义不同的phase。

在PASCAL论文中，reasoning phase被定义为包含prefill stage和reasoning tokens的解码，因为两者都贡献于"第一个user-visible token出现前"的延迟。
具体的伪代码：
```
// Reasoning-based LLM decode phase
Input: prompt, model
// Phase 1: Prefill
KV_cache = Prefill(model, prompt)
first_token = Decode(KV_cache)  // could be reasoning or answering
while not end_of_sequence:
    if first_token == <\think> token:  // phase boundary detected
        // Reasoning phase ends, Answering phase begins
        break_from_reasoning = true
    if not break_from_reasoning:
        // Reasoning phase: generate hidden CoT tokens
        token = Decode(model, KV_cache)
        reasoning_tokens.append(token)
    else:
        // Answering phase: generate user-visible tokens
        token = Decode(model, KV_cache)
        answering_tokens.append(token)
TFAT = time from <\think> to first answering token  // TTFAT
TTFT = prefill_time + reasoning_time + TTFAT        // Redefined TTFT
```

从算法pipeline角度拆解术语：

Reasoning-based LLM推理pipeline（以DeepSeek-R1-Distill-Qwen-32B为例）：
1. **Prefill stage**：处理input prompt，生成初始KV cache。这步与conventional LLM一致。
2. **Reasoning phase（decoding）**：模型auto-regressively生成reasoning tokens（如"我们需要计算..."、"首先..."等中间推理步骤）。这些tokens对用户隐藏，但KV cache正常累积。Reasoning token count可从128到2048+不等（取决于问题复杂度）。PASCAL的motivation实验显示：reasoning phase latency受blocking/preemption严重影响——FCFS下128 reasoning tokens请求latency可达oracle的5.14×，RR下2048 reasoning tokens可达1.75×。
3. **Phase transition**：模型生成特殊token（如DeepSeek-R1的`<\think>`）表示reasoning结束。
4. **Answering phase（decoding）**：模型开始生成user-visible answering tokens。这些tokens stream到用户。Answering phase是threshold-sensitive：只需TTFAT ≤ 0.25s + TPOT ≤ 100ms即可满足QoE SLO。即使因preemption导致fragmented execution，SLO仍可保持（Figure 5b显示RR SLO attainment接近oracle）。

术语一般如何实现？如何使用？

Reasoning phase/answering phase的区分是模型训练时通过强化学习（如DeepSeek-R1的GRPO）内化到模型行为中的，不是serving系统手动划分的。模型自动在推理时决定何时进行reasoning、何时输出answer。Serving系统（如PASCAL）通过检测phase boundary token（如`<\think>`）来识别当前phase，并根据phase特性应用不同调度策略。

关键观察（来自PASCAL characterization）：
- Reasoning phase：interruption-sensitive → 需要最小化blocking/preemption。FCFS的HoL blocking显著延长reasoning latency，RR的frequent preemption对long reasoning造成1.75× overhead。
- Answering phase：threshold-sensitive → 可容忍moderate preemption。RR在answering phase保持高SLO attainment（接近oracle），因为只要TTFAT低+TPOT达标即可。
- 两个phase的asymmetric sensitivity是PASCAL phase-aware scheduling的核心motivation。

在conventional LLMs中不存在此区分——decode阶段的所有tokens对用户可见，TTFT=prefill latency，TPOT=decode token generation rate。PASCAL的工作表明这一区分对serving系统的设计有重大影响。

涉及论文标题：
- PASCAL: A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Models

## Resource-oriented Layer-skipping Adaptor（面向资源的跳层适配器）

术语是什么？通过联网搜索让回答具体和精准。

Resource-oriented Layer-skipping Adaptor 是 LEGO 提出的算法-系统协同设计中的算法侧核心组件：在游戏-LLM 共置场景中，当 GPU 资源不足以运行完整 LLM 推理时，必须跳过若干 Transformer 层以满足资源预算。该 adaptor 是一个小型 FFN（Feed-Forward Network），用于近似被跳过层段的知识变换，将资源约束驱动的跳层造成的精度损失降到最低。其设计受知识蒸馏（Knowledge Distillation）启发，但采用自蒸馏（self-distillation）方式——adaptor 从同一模型中被跳过的层学习知识表示，而非从外部 teacher model 学习。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Adaptor 的算法 pipeline 分为离线准备和在线推理两个阶段：

**离线准备流程：**
```
Step 1: Profile 游戏 rendering headroom 范围
  - 采集代表性 gameplay 期间的 rendering task 执行时间
  - 计算最小/最大可用 headroom H_min, H_max
  - 根据 H_min/H_max 确定必须跳过的最小层数 M 和最多层数 N
  - 因此需要准备 N-M+1 种跳层配置

Step 2: 构建 Layer Similarity Heatmap
  - 用训练数据（如 WebInstruct）对 LLM 做前向推理
  - 提取每个 Transformer layer L_i 的输出 tensor T_i
  - 对所有层对 (L_i, L_j) 计算 cosine similarity: sim(T_i, T_j)
  - 构建 similarity heatmap 矩阵

Step 3: 选择跳过层段
  - 当需跳过 N 层时，沿 heatmap 对角线寻找连续层区间 [L_k, L_{k+n}]
  - 选择该区间内平均相似度最高的配置
  - 如 Llama3-8B 跳 4 层: 选 L25-L29（highest similarity）
  - 如 Llama3-8B 跳 8 层: 选 L23-L31

Step 4: 训练 Adaptor
  - Adaptor 是一个单层 FFN: FFN^{k+n}_k
  - 输入: 第 k 层输出 f_k
  - 目标: 逼近第 k+n 层原始输出 f_{k+n}
  - Loss: L_mse = ||f_{k+n} - FFN^{k+n}_k(f_k)||²
  - 仅更新 adaptor 权重（268.8 MB/adaptor）
  - 不同跳层配置可复用中间层输出，减少冗余计算
  - BlackMyth 最多需 14 个 adaptor，总训练时间约 36 小时
```

**在线推理流程：**
```
1. Scheduler 根据 headroom 预测决定跳过 N 层
2. 将对应层段 [L_k, L_{k+n}] 的 Transformer layers 替换为已训练的 adaptor
3. 推理时：输入 f_k → adaptor FFN → 直接映射到 f_{k+n}
4. 剩余层（L_1 到 L_{k-1}, L_{k+n+1} 到 L_last）正常执行
5. decode 阶段以 Transformer layer (~0.4ms) 为粒度调度跳层后推理
6. prefill 阶段以 self-attention (0.5ms) 和 FFN sublayer (1.0ms) 为粒度
```

关键设计决策：
- **连续跳层而非离散跳层**：heatmap 显示后层高相似度但最后层与倒数第二层低相似（最后层编码与 output layer 对接的关键知识，不应跳过）。连续跳层避免 disruption of inter-layer coherent representations
- **资源驱动而非 confidence 驱动**：传统 layer-skipping（LITE/CALM）按 token confidence 决定早退，无法为每个请求提供资源预算保证。LEGO 反转逻辑：资源预算决定跳层数，adaptor 补偿精度
- **跳层上限**：≤12 层时 LEGO 精度优于小模型 baseline（Llama3-3B），跳 13-14 层时精度显著下降但仍优于 LITE

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
- **Adaptor 架构**：标准 FFN layer（两层全连接 + 激活函数），输入输出维度与 Transformer hidden dimension 相同。以 Llama3-8B（hidden=4096）为例，单个 adaptor 参数量约 268.8 MB（FP16）
- **训练数据**：游戏公司使用自己的 fine-tuned LLM + 私有数据集训练 adaptor。论文使用 WebInstruct 作为 upstream training set
- **内存开销**：12 个 adaptor 合计约 3.23 GB；intermediate-result tensor 占 67.5 MB（推理本身需要，无额外开销）
- **兼容性**：与静态量化（如 INT4）、静态 sparsity 兼容；不与动态加速方法叠加（引入执行时间不确定性）
- **跨模型适用**：论文验证 adaptor 在 Llama3-8B、Mistral-7B、DeepSeek-V2-Lite（MoE）、Mixtral-8x7B（MoE）上均有效
- **部署方式**：游戏公司离线训练 adaptor → 与游戏 + LLM 打包 → 用户下载后本地部署；也可集成到云游戏平台如 NVIDIA GeForce NOW

涉及论文标题：
- LEGO: Supporting LLM-enhanced Games with One Gaming GPU

## Layer Similarity Heatmap（层间相似度热力图）

术语是什么？通过联网搜索让回答具体和精准。

Layer Similarity Heatmap 是 LEGO 提出的一种 LLM 层间知识相似度分析方法，用于指导 resource-oriented layer-skipping adaptor 选择要跳过的连续 Transformer 层段。它通过计算所有 Transformer layer 对之间的输出 tensor cosine similarity，构建一个 M×M 热力图矩阵（M 为 Transformer layer 数量），可视化各层输出表示的相似程度。热力图的对角线反映不同跳层配置的候选层段。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

构建流程：
```
输入: LLM model (M transformer layers), dataset D (如 WebInstruct)
输出: M×M similarity heatmap H

for each sample x in D:
    前向传播，提取每层 L_i 的输出 tensor T_i (i ∈ [1, M])
    for each layer pair (i, j):
        提取 T_i 和 T_j 的 hidden states（取前 16 个 output token，与游戏输出长度一致）
        H[i][j] += cosine_similarity(T_i, T_j)

H = H / |D|  // 平均所有样本
```

热力图的关键观察（论文 Fig.8）：
1. **对角线反映跳层配置**：对角线 H[i][j] where j-i=N 代表所有"N 跳层"候选项——即跳过从 L_i 到 L_j 的层段
2. **后层高相似度**：Llama3-8B 和 Mistral-7B 的 latter layers 普遍高相似度（>0.8），说明后续层引入的新信息较少，可安全跳过
3. **最后层与倒数第二层低相似度**：最后 transformer layer 编码与 output layer 对接的关键知识，不应跳过
4. **初始层低相似度**：early layers 输出差异大，跳过会丢失基础语义信息

使用热力图选择跳层配置：
- 跳 4 层时（Llama3-8B）：沿 j-i=4 对角线找到相似度最高的区间 → L25-L29
- 跳 8 层时（Llama3-8B）：沿 j-i=8 对角线找到相似度最高的区间 → L23-L31
- 选择连续区间而非离散层：论文实验显示跳过离散层比跳过连续层造成更大的性能退化（因为 knowledge is distributed both within individual layers and across inter-layer connections）

与 LLM-Streamline 的区别：LLM-Streamline 也提出替换连续 transformer layers，但 LEGO 认为其推理过程不充分——"90% 连续层对超过 80% 相似度"不足以证明可以安全跳过，离散跳层比连续跳层造成的知识损失更大。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
- **相似度度量**：使用 cosine similarity（而非欧氏距离或其他度量），因为 transformer hidden states 的方向（而非幅值）承载更多语义信息
- **采样策略**：使用 2400 samples from WebInstruct，每个 sample 取前 16 个 output token 的 hidden states（与游戏场景 LLM 输出长度一致）
- **计算成本**：O(M² × |D| × d)，M 为层数、d 为 hidden dimension。对 32 层 Llama3-8B，该计算在单 GPU 上数小时内完成
- **通用性**：论文在 Llama3-8B、Mistral-7B、DeepSeek-V2-Lite（MoE 28层）、Mixtral-8x7B（MoE 32层）上均构建了 heatmap，验证了方法的跨模型适用性
- **一次构建，多次复用**：同一 Game-LLM 对的 heatmap 只需构建一次，后续所有跳层配置的选择均基于此 heatmap

涉及论文标题：
- LEGO: Supporting LLM-enhanced Games with One Gaming GPU

## BBC (Bitmap-Bitmap-CSR)

术语是什么？

BBC (Bitmap-Bitmap-CSR) 是 Uni-STC 论文提出的一种统一稀疏矩阵存储格式，通过 CSR 组织 4×4 sparse tile、用两级 bitmap 描述 tile 内非零位置，使 SpMV、SpMSpV、SpMM、SpGEMM 四类稀疏 kernel 共享同一数据结构而无需在线格式转换。BBC 的两级 bitmap 设计为：top-level bitmap 描述 4×4 tile 内 M 维和 N 维是否有非零，bottom-level bitmap 描述每个 4×4 tile 内部具体哪 16 个位置非零。两级 bitmap 使得 TMS 和 DPG 硬件能直接解析生成 task，无需复杂硬件 decoder。

从算法pipeline角度拆解术语：

BBC 格式的数据组织：
```
// top-level CSR 结构
row_ptr[N/4+1]:   // 每 4 行为一个 tile row，记录每个 tile row 的起始位置
col_idx[tile_nnz]: // 非零 tile 的列索引
// 每个非零 tile 内部:
top_bitmap:        // 16-bit，标记 tile 内 4x4 的 M 维和 N 维非零情况
bottom_bitmap[K]:  // 对于 GM＝A×B，沿 K 维每个 tile 配一个 bottom bitmap
val_ptr_lv2:       // 指向 tile 内实际非零数值存储位置
values[]:          // 按 tile 组织存储非零浮点数值
```

BBC 对不同 kernel 的表达：
- **SpMV/SpMSpV**：B 侧为 dense vector x 或 sparse vector x，BBC 仅描述 A 矩阵的稀疏结构，x 作为 operand 直接传入
- **SpMM**：A 稀疏（BBC 描述），B 为 dense matrix（如 64 列），每个 A tile 与 B 的一个 dense tile block 相乘
- **SpGEMM**：A 和 B 均用 BBC 格式描述，C 也用 BBC 作为输出格式，TMS 在 K 维按 bloom filter 判断是否存在非零乘积 tile

相比传统 CSR/CSC 格式的差异：BBC 的 tile-aligned 设计天然匹配 tensor core 的 4×4×4 计算粒度，bitmap 编码比 2:4 structured sparsity 更灵活（不要求固定 50% 稀疏率），且避免了 DS-STC 和 RM-STC 依赖的硬件 decoder（其面积开销大且难以支持多种 kernel）。

术语一般如何实现？如何使用？

BBC 需一次离线构建：输入为 CSR/CSC 格式稀疏矩阵 → 分割为 4×4 tile → 统计每 tile 内非零分布 → 生成 top-level bitmap 和 bottom-level bitmap → 重排非零值数组。构建在 64-core AMD EPYC 7702 CPU <1000ms、NVIDIA A100 GPU <100ms。构建后可用于迭代应用（GNN training、linear solver）一次性摊销 overhead。运行时 warp 通过 stc.load 将 BBC metadata 和 values 直接装载到 Uni-STC 的 Matrix A/B Buffer，无需软件进行格式解码。

涉及论文标题：
- Uni-STC: Unified Sparse Tensor Core

## SpGEMM (Sparse General Matrix-Matrix Multiplication)

术语是什么？

SpGEMM (Sparse General Matrix-Matrix Multiplication) 是计算 C = A × B 的线性代数原语，其中 A 和 B 均为稀疏矩阵，输出 C 也是稀疏矩阵。SpGEMM 是图算法（multi-source BFS 前沿扩展、三角形计数、图粗化）、代数多重网格（Algebraic Multigrid, AMG）的 Galerkin product（计算 R×A×P 的 coarse-grid operator）以及稀疏 DNN（如 GNN 中邻接矩阵的 k 步幂）的核心算子。与 SpMM（A 稀疏 × B dense）不同，SpGEMM 的双输入稀疏性使得：每个 C 非零元素的计算量仅由 A_row 和 B_col 的非零交集决定，执行模型为 Gustavson 算法（CSR-based，对 A 每一行遍历该行非零列→逐列访问 B 的对应行→做 sparse dot product→结果归约到 C 的该行），且 C 的非零模式在计算完成前未知——需通过 symbolic multiplication 预估 C 的 size 和 sparsity pattern。

从算法pipeline角度拆解术语：

Gustavson 算法的 SpGEMM 伪代码：
```
Input: A (CSR), B (CSR)
Output: C (CSR)

// Phase 1: Symbolic (预估 C 的 row_ptr)
C_row_nnz_est[0..N-1] = 0
for i in 0..N-1:
    for k in A.row(i):               // A 第 i 行的所有非零列 k
        for j in B.row(k):           // B 第 k 行的所有非零列 j
            if not marked(i, j):     // 行 i 中 j 列首次出现
                mark(i, j)
                C_row_nnz_est[i]++

// Phase 2: Numeric (计算 C 的实际值)
C_row_ptr = prefix_sum(C_row_nnz_est)
for i in 0..N-1:
    for k in A.row(i):
        val_a = A.val(i, k)
        for j in B.row(k):
            val_b = B.val(k, j)
            C[i][j] += val_a × val_b
```

SpGEMM 的核心难点：
1. **C 非零模式不可预测**：在 numeric 阶段完成前不知道 C 的 sparsity pattern，需要两次 pass 或动态扩容
2. **负载严重不均衡**：A 行长分布可能极度偏斜（power-law），short row 的 C 输出少但 long row 的 sparse dot product 数量以 O(nnz_A_row × avg_nnz_B_row) 爆炸
3. **中间结果膨胀**：每个 A 行可能产生远多于最终 C 实际非零数的中间乘积（称为 intermediate product blowup），需要哈希表或排序去重
4. **访存不规则**：对 B 的访问模式由 A 的非零列索引间接决定，cache miss 率高

在 GNN/科学计算中的典型场景：AMG solver 的 Galerkin product（R×A×P）、GNN 中计算 A²、A³ 以捕捉 k-hop 邻居信息、multi-source BFS 中从多个源点同时扩展 frontier。

术语一般如何实现？如何使用？

GPU SpGEMM 实现策略：
1. **cuSPARSE SpGEMM**：NVIDIA 官方实现，支持 CSR/CSC 输入输出，基于 Hash/排序的 Gustavson 变体
2. **RM-STC**：以 row-row scalar-vector 组合方式，将 SpGEMM 的 Gustavson 算法各 sparse dot-product 映射到固定 shape 的 MAC array 任务
3. **DS-STC**：以 outer-product 方式，将 A 的半列 × B 的半行映射到 MAC array
4. **Uni-STC 方式**：用 BBC 格式统一 A 和 B → TMS 沿 K 维做 bloom filter 判断 C tile 是否非零 → 将 SpGEMM 的计算拆为 4×4×4 T3 任务 → DPG 再拆为 1×1×4 T4 dot-product → SDPU 执行 segmented dot-product 并预合并 partial products
5. **SuiteSparse 方阵子集**：标准 benchmark，Uni-STC 使用 2126 个方阵计算 C=A²

Uni-STC 在 SpGEMM 上相对 RM-STC 和 DS-STC 的平均 speedup 为 1.45x 和 2.40x（64 MAC@FP64），能量效率提升分别为 1.09x 和 3.14x。

涉及论文标题：
- Uni-STC: Unified Sparse Tensor Core

## GEMM-based Operator Chain (FFN/Gated FFN/Conv Chain) as Compute-Intensive Fusion Target（作为计算密集融合目标的GEMM算子链）

术语是什么？通过联网搜索让回答具体和精准。

GEMM-based Operator Chain 是 LLM 和 CNN 中由连续 General Matrix Multiplication (GEMM) 构成的算子序列，在 FlashFuser 论文中是 kernel fusion 的主要目标。三种典型模式：(1) Standard FFN——两个连续 GEMM（GEMM(A,B)→C → activation(如ReLU) → GEMM(C,D)→E），如 GPT/OPT/LLaMA 的 Feed-Forward Network 层；(2) Gated FFN——FFN 带两个平行的 Up-FFN 分支（SwiGLU 结构），其中一个分支经 SiLU 激活后与另一分支 element-wise 乘再进入后续 GEMM；(3) Conv Chain——卷积块可通过 im2col 转换为 GEMM chain 形式（Conv1×1 等价于 GEMM）。这些算子链在 LLM 推理中的典型表现为：序列长度 512 时，FFN 层占 GPT-6.7B 约 61% 执行时间、LLaMA-1B 约 57%、OPT-1.3B 约 53%（Table I）。由于中间 tensor C [M×N] 的数据量很大（如 GPT-6.7B: M=128, N=16384, C ≈ 2M floats ≈ 8MB），远超单 SM SMEM (~227KB)，传统方法无法在片上保留完整中间结果，必须通过 HBM round-trip。

从算法pipeline角度拆解术语：

```
// Standard FFN 推理流程 (GPT/OPT/LLaMA 通用):
def feedforward(x):          // x: [M, K]  (M=batch tokens, K=hidden dim)
  C = x @ W_up.T            // GEMM1: [M,K] × [K,N] → [M,N]
  C = activation(C)         // ReLU/GELU/SiLU
  E = C @ W_down.T          // GEMM2: [M,N] × [N,L] → [M,L] (通常 L=K)
  return E

// Gated FFN (SwiGLU, LLaMA 使用):
def gated_ffn(x):           // x: [M, K]
  C1 = x @ W_gate.T         // Gate branch: [M,K] × [K,N] → [M,N]
  C2 = x @ W_up.T           // Up branch:   [M,K] × [K,N] → [M,N]
  C1 = SiLU(C1)
  C = C1 ⊙ C2               // Element-wise multiply
  E = C @ W_down.T          // GEMM2: [M,N] × [N,L] → [M,L]
  return E

// Conv Chain (ResNet, 可 im2col 转换):
def conv_block(x):          // x: [IC, H, W]
  C = Conv1(x, W_conv1)     // Conv1: [IC,H,W] × [OC1,IC,K1,K1] → [OC1,H,W]
  C = ReLU(C)
  E = Conv2(C, W_conv2)     // Conv2: [OC1,H,W] × [OC2,OC1,K2,K2] → [OC2,H,W]
  return E
```

在 GPU 推理中，这些 GEMM chain 的算术强度（Arithmetic Intensity）较低——当 batch size M 较小时（inference 典型 M=128），中间 tensor C 从 HBM 读取/写入的 overhead 相对 GEMM 算力成为瓶颈，呈现 memory-bound 特性。kernel fusion 的目标是将这些算子链合并为一个大 kernel，使中间 tensor C 留在片上(on-chip)而不是写回 HBM。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GEMM-based Operator Chain 在不同层次中的实现：
1. **库实现（cuBLAS/CUTLASS）**：每个 GEMM 作为独立 kernel launch，中间 tensor C 必须写回 global memory。PyTorch `torch.compile` 可减少 kernel launch overhead 但数据路径不变
2. **图级 fusion（TASO/TVM Relay）**：通过算子替换和 graph optimization 融合 compute-activation pattern，但不支持 sequential GEMM chain 的 compute-intensive fusion
3. **SMEM-based fusion（BOLT/Chimera/Welder）**：利用 register 和单 SM 的 SMEM 保留中间 tile——但当 C tile > SMEM (~227KB) 时 fusion 失败
4. **DSM-based fusion（FlashFuser）**：利用 DSM（cluster 内多 SM SMEM 互联，~3.6MB）保留更大的 C tile，通过 dsm_comm 原语管理 producer-consumer 数据流

涉及论文标题：
- FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection

## Visual Autoregressive (VAR) Model

术语是什么？
Visual Autoregressive (VAR) 是一种将图像生成建模为自回归 next-token prediction 的生成模型范式。与 Diffusion（多步去噪）不同，VAR 首先将图像通过 VQGAN 等 tokenizer 编码为离散的 visual token grid（如 256×256 图像编码为 16×16=256 个 token），再像 LLM 一样逐个预测 next visual token。VAR 天然与 LLM/多模态系统统一，且在大规模下保持较好的质量扩展趋势（scale-out/scale-up 潜力）。但其自回归 token-by-token 解码将生成延迟推高（256×256 需 256–4096 次串行 Transformer 调用），单张图常需 10–60 秒。

从算法pipeline角度拆解术语：
VAR 的标准推理 pipeline：
1. VQGAN encoder 将输入图像编码为离散 visual token grid（如 16×16=256 tokens）
2. 初始化：所有 visual token 位置置为 [MASK]
3. 自回归循环（每步）：
   a. 当前 token sequence（含已解码和 masked token）输入 generative Transformer
   b. Transformer 输出所有位置 logits
   c. 按预定顺序（如 raster scan）选取下一个待解码位置
   d. 对该位置做 Gumbel/argmax sampling 得到新 token
   e. 将该位置 mask 置为 False，token 写入 sequence
4. 所有 token 解码完成→VQGAN decoder 将 visual token grid 还原为像素图像
关键特点：每次 Transformer invocation 仅生成 1 个 token，串行步数等于 token 总数（256-N×N），attention 复杂度 O(N²) 随序列长度二次增长。

术语一般如何实现？如何使用？
VAR 模型通常使用基于 ViT/DeiT backbone 的 generative Transformer + VQGAN tokenizer。训练用 ImageNet + cross-entropy loss + AdamW optimizer。VAR-Turbo 论文使用 DeiT-based Transformer 在 ImageNet 上训练 500 epochs、4×V100、batch size 256。推理时通用平台为 GPU（如 V100），延迟瓶颈主要在串行调用次数和每次调用的 attention/FFN 计算量。

涉及论文标题：
- VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

## Draft-Free Parallel Decoding (PD)

术语是什么？
Draft-Free Parallel Decoding (PD) 是 VAR-Turbo 提出的无需 draft model 的并行视觉 token 解码算法。核心 insight：图像 visual token 的空间相关性远高于语言 token（entropy 分析表明图像 token 更冗余），因此可直接利用生成模型自身的置信度在每轮选择多个高置信 token 同时解码，无需依赖额外的 draft model（如 speculative decoding 中的小模型）。PD 通过 Gumbel sampling 对所有 masked 位置预测 token 和概率，按 schedule r(t) 确定每轮释放的 token 数 K=N×(1-r(t))，再经 TopK 选择最高置信度的 K 个 token 并行替换。

从算法pipeline角度拆解术语：
PD 每轮推理流程：
```
输入：token sequence V_t, mask array M (True=masked), schedule r(t)
1. Transformer forward：对所有当前位置输出 logits
2. Gumbel sampling：对每个 masked 位置采样 predicted token 和 confidence score
3. Mask-out：已 unmasked 位置（M[i]=False）的 confidence 设为 -inf
4. Compute K = N * (1 - r(t))  // 本轮释放的 token 数
5. TopK selection：选 confidence 最高的 K 个位置
6. Token replacement：K 个位置的 token 更新为 predicted token，mask 置 False
7. 其余 token 保持 masked，进入下一轮
```
与 speculative decoding 的关键区别：PD 不依赖 draft model（无额外模型开销），且每轮可并行释放多个 token（最高 64），远高于语言 speculative decoding 的 2-3 token/step。PD 需 PD-aware training 选择 sampling temperature、masking ratio r(t)、guidance scale 等超参数。

术语一般如何实现？如何使用？
PD 将 VAR 的串行步数从 256（256×256）降至 8-32 步（减少 >80%），在 VAR-Turbo-Balance 模式下 256×256 仅需 8 步。实现需配合 TopK 排序硬件（如 Radix Sort Core）加速大 K TopK 选择（N=4096 时 K 可达 1936）。PD 与 TA、DB 叠加使用，从跨迭代和迭代内两个维度联合加速。

涉及论文标题：
- VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

## Token Aggregation (TA)

术语是什么？
Token Aggregation (TA) 是 VAR-Turbo 提出的面向浅层 Transformer 的 attention 压缩算法。核心设计：将每层的 token sequence 划分为 non-overlapped local window，每个 window 内的 token 先经 Small Attention 聚合成一个 representative token（通过 attention score 加权求和），再将所有 window 的 representative token 拼接后送入 Big Attention 做全局建模。TA 基于 attention 作为低通滤波的 insight：浅层（Learning Region）attention map 中局部 token 高度相关但未完全退化，通过两级 attention 在不固定稀疏 pattern 的情况下压缩计算量。

从算法pipeline角度拆解术语：
TA 的伪代码（以单层浅层 Transformer 为例）：
```
输入：token sequence X (shape: [N, D]), local window size W
1. 划分：将 X 分为 N/W 个 non-overlapped local window
2. for each window w:
     Small Attention: Q_w, K_w, V_w = Linear(X_w)
     Score_w = softmax(Q_w × K_w^T / sqrt(d))
     Coeff = mean(Score_w, axis=0)  // 平均所有 query 的 attention
     Rep_w = Coeff × V_w  // representative token
3. Rep = concat(Rep_1, ..., Rep_{N/W})  // shape: [N/W, D]
4. Big Attention: Q_rep, K_rep, V_rep = Linear(Rep)
   Output_rep = softmax(Q_rep × K_rep^T / sqrt(d)) × V_rep
5. 将 Output_rep 映射回原 token 维度用于后续层
```
关键参数：local window size —— 低分辨率用 2，高分辨率用 2/4 混合；size ≥ 8 时质量明显下降。TA 论文声称减少约 41% attention MAC，质量下降 <0.5%。

术语一般如何实现？如何使用？
TA 应用于浅层（0-15 层 Learning Region）。硬件上需 Unified Attention Core 同时支持 Small Attention 的 OP dataflow 和 Big Attention 的 Row dataflow。TA 与 DB 互补：TA 处理浅层仍活跃的 Learning Region，DB 处理深层趋于惯性的 Inert Region。

涉及论文标题：
- VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

## Dynamic Bypass (DB)

术语是什么？
Dynamic Bypass (DB) 是 VAR-Turbo 提出的面向深层 Transformer 的 token 级计算跳过算法。核心设计：在深层（Inert Region），先用轻量 MLP 为每个 token 计算 importance score，再通过 TopK 选出最重要的 K 个 token 进入完整 Transformer（attention + FFN），其余 token 绕过（bypass）Transformer 层并通过 token restoration 在下一层补回原有信息。DB 基于深层 attention map 趋于相似（低通滤波效果）的 insight：深层 token 的重要性高度分化，大部分 token 对输出的信息增量低，可安全跳过。

从算法pipeline角度拆解术语：
DB 的伪代码（以单层深层 Transformer 为例）：
```
输入：token sequence X (shape: [N, D]), schedule parameter α, β
1. Importance scoring：Score_i = LightMLP(X_i)  for i=1..N  // 轻量单层 MLP
2. Compute skip threshold: s(l) = min(α×l + β, max_threshold)  // l=layer index
3. K = N × (1 - s(l))  // 保留的 token 数
4. TopK selection：选出 Score 最高的 K 个 token indices
5. Split：
   - X_keep = X[indices]  // K 个重要 token → 进入 attention + FFN
   - X_bypass = X[others]  // N-K 个不重要 token → bypass
6. 仅 X_keep 经过完整 Transformer 层（attention + FFN）→ Y_keep
7. Token restoration for bypass tokens:
   Y_bypass[i] = X_bypass[i] × JudgeWeight[i] + X_bypass[i]  // 原有信息保留
8. 合并 Y_keep 和 Y_bypass 回到原排列 → 输出 Y
```
关键超参数：α=0.3、β=-0.4、max skip threshold=0.55。DB 额外减少约 58% MAC（覆盖 attention 和 FFN），且 token restoration 避免信息永久丢失。

术语一般如何实现？如何使用？
DB 应用于深层（Inert Region），与 TA 互补。需配合 schedule function 控制逐层 skip rate，skip rate 随层数加深逐步提高。硬件上依赖 Radix Sort Core 执行大 K TopK selection。

涉及论文标题：
- VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

## Learning Region / Inert Region

术语是什么？
Learning Region 和 Inert Region 是 VAR-Turbo 论文对 VAR Transformer 层按其 attention 行为特征所做的两类分区。论文将 attention 解释为低通滤波（low-pass filter）：随着层数堆叠，attention map 逐渐趋于相似（softmax 后的分布平滑化），高频信息被削弱。浅层 attention 仍有较强的学习能力（不同 token/head 的 attention pattern 差异化明显），称为 Learning Region；深层 attention map 高度相似、token 重要性分化、层间信息增量低（趋于"惯性化"），称为 Inert Region。此分区决定 TA 应用于 Learning Region、DB 应用于 Inert Region。

从算法pipeline角度拆解术语：
分区判断依据（基于 attention entropy / similarity 分析）：
- Learning Region（如 0-15 层）：inter-layer attention similarity 低，attention 仍在主动学习 token 间关系。适用 TA：local window 内高度相关但全局需要区分，通过 Small+Big Attention 压缩局部相似性。
- Inert Region（如 16+ 层）：inter-layer attention similarity 高（cosine similarity 接近 1），深层 attention map 趋于均匀和相似。适用 DB：仅重要 token 进入完整计算，其余 bypass。
分区是经验性的，VAR-Turbo 通过实验确定分界层。

术语一般如何实现？如何使用？
分区通过测量各层 attention map 间的 cosine similarity 确定，实际部署时使用固定分界层（如第 15 层为界）。Learning Region 使用 TA（Small+Big hierarchical attention），Inert Region 使用 DB（importance-based token bypass）。此分区方法可推广到其他 deep Transformer 模型以识别不同层的计算冗余特征。

涉及论文标题：
- VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

## Dual Redundancy in Visual Autoregressive Models

术语是什么？
Dual Redundancy（双重冗余）是 VAR-Turbo 论文的核心理论框架，从信息论角度刻画 VAR 图像生成模型的两类冗余：(1) Image Redundancy（图像空间冗余）—— 图像 visual token 的空间相关性远高于语言 token，相邻像素/token 高度相关，熵值低，使得无需 draft model 即可在同一轮中预测和选择多个 token 并行解码；(2) Model Redundancy（模型计算冗余）—— attention 机制作为低通滤波器，随层数加深使 attention map 趋于相似、token 重要性分化，导致深层的大部分 token 可安全跳过完整计算。Dual Redundancy 为 PD（利用 Image Redundancy）和 TA+DB（利用 Model Redundancy）提供了理论依据。

从算法pipeline角度拆解术语：
Dual Redundancy 驱动的算法设计：
1. Image Redundancy → Draft-Free Parallel Decoding (PD)：
   - 图像中相邻 visual token 高度空间相关→模型对相邻位置的预测置信度高
   - 可直接按置信度选 TopK token 并行解码（无需 draft model）
   - 每轮可解码 8-64 token（vs speculative decoding 的 2-3 token）
2. Model Redundancy → Token Aggregation (TA) + Dynamic Bypass (DB)：
   - 浅层 attention 局部 token 相似但全局 pattern 不同 → TA：Small Attention 压缩局部 + Big Attention 保留全局
   - 深层 attention 全盘趋于相似、token 重要性分化 → DB：仅 TopK 重要 token 完整计算，其余 bypass

术语一般如何实现？如何使用？
Image Redundancy 通过 entropy/redundancy 量化分析验证（对比语言 token 和图像 token 的条件熵分布）。Model Redundancy 通过逐层 attention map cosine similarity 和低通滤波频率响应分析验证。PD、TA、DB 三项算法各自对应其中一类冗余，三者叠加实现跨迭代和迭代内的联合加速。

涉及论文标题：
- VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

## Beam Search Speculation（束搜索推测）

术语是什么？通过联网搜索让回答具体和精准。
Beam Search Speculation 是 AdaServe 提出的 speculative decoding 中 draft model 生成 candidate token tree 的方法。与传统 speculative decoding 每次只生成一个候选 token 序列不同，beam search speculation 对每个请求执行 d 步 beam search（每步维持宽度 w 的 beam），形成树状候选 token 结构。每步从当前层 w 个 token 出发各扩展多个 child token，记录由 draft model logits 近似的 path probability（路径上各节点概率的累积积）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Input: root token r, draft model M_d, depth d, beam width w
Output: candidate token tree T

T = {r}
current_layer = {r}
for step = 1 to d:
    candidates = []
    for each node n in current_layer:
        logits_n = M_d(prefix + path_to(n))
        top_k = argmax_k(logits_n, k)
        for each token t in top_k:
            prob = softmax(logits_n)[t]
            path_prob = prob * n.path_prob
            candidates.add(child(n, t, path_prob))
    current_layer = argmax_w(candidates, key=path_prob)
    T.add_all(current_layer)
```
每步扩展多分支形成 tree 而非 chain，保留 w 个最优路径，记录 path probability 供后续 SLO-customized selection 使用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 AdaServe 中由 GPU 上 colocate 的 draft model（如 Llama-3.2-1B-Instruct）执行。深度 d 和宽度 w 由 scheduler 根据活跃请求数动态调节。CUDA Graph 预捕获固定形状 decoding steps 减少 kernel launch overhead。

涉及论文标题：
- AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding

## SLO-Customized Token Tree Construction（SLO定制化Token树构造）

术语是什么？通过联网搜索让回答具体和精准。
SLO-Customized Token Tree Construction 是 AdaServe 的核心算法，将 multi-SLO serving 形式化为带 token budget 约束的 token tree 构造问题。目标：每次 decoding iteration 中，给定硬件可验证 token budget B，为 batch 中各请求构造 speculation token tree，使得 (1) 每个请求期望接受 token 数满足其 TPOT SLO；(2) 同时最大化总 expected accepted tokens。问题分解为两阶段贪心选择。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Input: requests R, token budget B, candidate tree per request T_i,
       per-request SLO target min_tokens_i

// Phase 1: SLO-customized selection
for each r_i sorted by urgency (SLO slack ascending):
    remaining = min_tokens_i - expected_accepted(selected[r_i])
    while remaining > 0 and |selected[r_i]| < per_request_limit:
        best = argmax(T_i.unselected, key=path_prob)
        selected[r_i].add(best)
        remaining -= best.path_prob

// Phase 2: Throughput-optimized selection
remaining_budget = B - Σ|selected[r_i]|
all_remaining = ∪_i (T_i.nodes - selected[r_i])
top_nodes = argmax_k(all_remaining, k=remaining_budget, key=path_prob)
add top_nodes to respective request selected sets

// Submit all selected token trees for tree-based verification
```
min_tokens_i = max(0, latency_elapsed_i / TPOT_SLO_i - tokens_generated_i)。SLO phase 优先满足严格请求；throughput phase 用剩余 budget 最大化总体吞吐。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 AdaServe 的 CPU scheduler 中实现，用 draft model logits 近似真实 path probability。per-request token limit 防止某严格请求吞噬过多低概率 budget。CPU selection overhead 仅 0.41%（Llama-70B）/ 0.31%（Qwen-32B）。

涉及论文标题：
- AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding

## Error-Bound Batch-Level Workload Prediction（误差可控的批级负载预测）

术语是什么？通过联网搜索让回答具体和精准。
Error-Bound Batch-Level Workload Prediction 是 PiLLM 提出的统计预测方法，将大数定律和中心极限定理 (CLT) 应用于 LLM serving 场景中请求输入/输出长度的批级预测。核心公式：对大小 |B| 的 batch，其平均输出长度上界（置信度 1-ε）为 μ_d + σ_d/√|B| · Φ⁻¹(1-ε)。其中 μ_d 和 σ_d 为滑动窗口内历史输出长度的均值和标准差；√|B| 项体现大数定律——batch 越大估计越精确；Φ⁻¹ 为标准正态逆CDF；ε 控制错误容忍度（如 ε=0.05 对应 95% 置信上限）。输入长度同理。预测值再通过离线校准的线性系数转换为 prefill FLOPs、decode FLOPs、prefill KV memory 和 decode KV memory 四维资源需求。

从算法pipeline角度拆解术语，预测计算公式：
```
Input: 滑动窗口历史 W = {(L_in_i, L_out_i)}, batch B,
       错误容忍参数 ε_p (prefill), ε_d (decode)
Output: Prefill_FLOPs, Decode_FLOPs, Prefill_KV_mem, Decode_KV_mem

// Step 1: 滑动窗口统计更新
μ_in = mean({L_in_i for all entries in W})
σ_in = std({L_in_i for all entries in W})
μ_out = mean({L_out_i for all entries in W})
σ_out = std({L_out_i for all entries in W})

// Step 2: 批级上界估计（中心极限定理）
z_p = Φ⁻¹(1 - ε_p)   // prefill 预测的 z-score
z_d = Φ⁻¹(1 - ε_d)   // decode 预测的 z-score
L_in_pred = μ_in + (σ_in / sqrt(|B|)) * z_p
L_out_pred = μ_out + (σ_out / sqrt(|B|)) * z_d

// Step 3: 资源转换（离线校准的线性系数）
Prefill_FLOPs = α_pf * L_in_pred + β_pf
Decode_FLOPs  = α_df * L_out_pred + β_df
Prefill_KV_mem = α_pk * L_in_pred + β_pk
Decode_KV_mem  = α_dk * L_out_pred + β_dk
```
校准系数通过 offline profiling 获得：用不同输入/输出长度运行模型，记录实际 FLOPs 和 KV cache 内存，拟合线性回归。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) 滑动窗口长度需平衡响应速度和稳定性——PiLLM 按管理窗口设置（秒级）；(2) offline profiling 在部署前一次性完成；(3) ε_p 和 ε_d 可独立调节以适应 prefill（compute-bound）和 decode（memory-bound）的不同 SLO 敏感度；(4) CLT 近似在 |B|≥30 时良好，小 batch 时标准差项 σ/√|B| 较大使预测更保守；(5) 分布漂移时预测变差，PiLLM 以 spike reaction 兜底而非在线调参。该方法给出了从 LLM 服务统计量到 FLOPs/memory 的端到端可量化转换链路。

涉及论文标题：
- PiLLM: Resource-Efficient LLM Inference Using Workload Prediction

## MoE (Mixture of Experts / 混合专家模型)

术语是什么？
MoE (Mixture of Experts) 是一种神经网络架构模式，在LLM中将标准Transformer的FFN层替换为多个并行"专家"子网络（每个expert是一个独立FFN），由一个可学习的Gate Network（门控网络/路由器）为每个token选择top-k个expert进行计算。核心特征是sparse activation：每个token只激活少数expert（如top-1或top-2），因此模型总参数量可以很大但每个token计算量保持可控。代表性模型：Mixtral-8x7B（8 experts, top-2, 47B total/13B active）、Qwen1.5-MoE、Phi-3.5-MoE、DeepSeek-V3（256 experts, top-8, 671B total/37B active）。MoE在decoder-only LLM每层有8到256个expert，未被激活的expert参数称为inactive parameters（占72%-84%总参数）。

从算法pipeline角度拆解术语：
在decoder-only LLM的每个Transformer block中，MoE层替代标准FFN：
```
# 第l层，token i：
h = input_hidden_state  # [d_model]
g = GateNetwork(h)      # [num_experts], Softmax归一化的概率分布
topk_idx, topk_prob = TopK(g, k)  # 选概率最高的k个
output = sum(prob * expert_ffn(h) for idx, prob in zip(topk_idx, topk_prob))
```
流程：hidden state→Gate Network计算所有expert概率→Top-K选择→所选expert各自计算FFN→加权累加。稀疏激活使每token计算量≈dense model×active/total parameter ratio。

术语一般如何实现？
常用HuggingFace Transformers `MixtralSparseMoeBlock`实现。Gate Network为线性层(d_model→num_experts) + Softmax。训练时加load-balancing loss鼓励expert均匀使用。推理时Expert Parallelism将不同expert分布到不同GPU。Expert offloading将不活跃expert放CPU memory、按需加载到GPU。Gate Network输出完整概率分布（不仅是top-k选中集），在FineMoE中被用于iteration-level expert map和probability-aware prefetch/cache决策。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

## Gate Network / Expert Router（门控网络/专家路由器）

术语是什么？
Gate Network（Expert Router）是MoE架构中决定每个token由哪些expert处理的小型可学习线性层（W_g ∈ R^{d_model × num_experts}）。输入token hidden state h，输出经Softmax归一化后得到每个expert的概率分数，Top-k选择后加权累加选中expert输出。FineMoE创新地使用gate完整probability distribution（不仅top-k选中集）——概率值表达gate对每个expert的相对置信度，驱动similarity-aware expert selection和probability-aware cache management。

从算法pipeline角度拆解术语：
```
logits = h @ W_g          # [d_model] @ [d_model, num_experts] → [num_experts]
probs = Softmax(logits)   # 对所有expert的概率分布，sum(probs)=1
selected = ArgTopK(probs, k)
norm_probs = probs[selected] / sum(probs[selected])  # 重新归一化选中expert概率
output = Σ norm_probs[i] * ExpertFFN_i(h)
```

术语一般如何实现？
标准实现为`nn.Linear(d_model, num_experts)` + Softmax。Mixtral-8x7B用top-2 gating，DeepSeek-V3用Sigmoid替代Softmax扩展至256 experts。FineMoE将完整probability distribution存入Expert Map Store，用于cosine similarity检索和probability-aware prefetch priority（p/(l-l_now)）及eviction priority（1/(p×freq)）。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

## Load-Balancing Loss in MoE（MoE中的负载均衡损失）

术语是什么？
Load-balancing loss是MoE训练中的辅助损失函数，鼓励gate network将tokens均匀分布到所有expert，防止部分expert过度使用（hot expert）而其他闲置。作为总损失附加项权重较小。虽提升训练稳定性和expert利用率，但对推理时expert offloading有负面影响：均匀路由使expert选择更分散、更难预测，降低了request-level热度统计的prefetch hit rate。

从算法pipeline角度拆解术语：
```
L_lb = α · num_experts · Σ_i(f_i · P_i)
# f_i: expert i被分配的token比例
# P_i: expert i的平均gate probability
# α: 辅助系数 (如0.01)
# L_total = L_main + L_lb
```

术语一般如何实现？
常见于Switch Transformer、GShard、Mixtral等MoE训练。DeepSeek-V3提出auxiliary-loss-free策略用动态bias调整替代。FineMoE指出load-balancing loss使expert使用更均匀但粗粒度统计可预测性下降，因此需要更细粒度的iteration-level expert map预测。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

## Knowledge Precipitation（知识沉淀）

术语是什么？通过联网搜索让回答具体和精准。
Knowledge Precipitation是MFS论文提出的一种离线fine-tuning方法，将LLM model family中最大的模型微调为一个嵌套式multi-tier模型，使低tier获得独立的语言建模能力，同时由高tier的梯度向低tier"沉淀"知识。核心思想是利用同一model family中模型结构（均为stacked decoder-only Transformer）的统一性和Transformer的layer/head冗余性，通过full-parameter fine-tuning在最大模型上将不同模型规模的能力折叠到单一checkpoint中，替代独立维护多个模型。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Knowledge Precipitation的算法pipeline以Llama2-13B-chat转3-tier MFS模型为例：
1. Tier结构设计：测量最大模型（Llama2-13B）各层在目标推理硬件（A100）上的实际latency→确定tier边界（如tier-1取前18层对齐3B latency、tier-2取前32层对齐10B latency、tier-3为全部40层=13B）。切分原则：(a) 只切layer不切head，保持attention中所有head一致性和KV cache兼容性；(b) latency-aligned而非parameter-aligned，使各tier的实际用户体验对齐对应规模独立模型。
2. Step-by-step fine-tuning：
   - Step 1 (Tier-3)：对Llama2-13B-chat做全参SFT，loss仅含tier-3输出头L1 = L_tier3_output。数据为guanaco-llama2 ~9.85k对话，AdamW lr=2e-5, half-period cosine LR, weight decay=0.1, gradient clipping=0.3, 8×gradient accumulation (effective batch=64), seq_len=4096, 1 epoch/2500 iterations。
   - Step 2 (Tier-2)：基于Step 1 checkpoint，在第32层添加tier-2输出头（lm_head），训练目标L = L_tier2 + λ3·L_tier3。tier-3梯度反向传播到前32层共享参数，使tier-2获得独立生成能力且tier-3质量不退化。
   - Step 3 (Tier-1)：基于Step 2 checkpoint，在第18层添加tier-1输出头，训练目标L = L_tier1 + λ2·L_tier2 + λ3·L_tier3。低tier通过接收高tier梯度"沉淀"知识，获得独立语言建模能力。
3. 各tier有独立loss和输出头，但共享低层Transformer参数。推理时请求在对应tier边界采样返回（低tier早退出→低延迟低成本；高tier继续执行→高质量）。

伪代码（简化）：
```
# Knowledge Precipitation fine-tuning
model = load_checkpoint(llama2_13b_chat)
add_lm_head(model, layer=18, name="tier1_head")  # 3B-equivalent
add_lm_head(model, layer=32, name="tier2_head")  # 10B-equivalent
# tier3_head = original lm_head at layer 40  # 13B

# Step-by-step training
for step in [tier3_only, tier2_tier3, tier1_tier2_tier3]:
    for batch in guanaco_llama2_dataset:
        if step == tier3_only:
            loss = cross_entropy(model.tier3_head(hidden[40]), labels)
        elif step == tier2_tier3:
            loss = cross_entropy(model.tier2_head(hidden[32]), labels)
                 + λ3 * cross_entropy(model.tier3_head(hidden[40]), labels)
        else:  # tier1_tier2_tier3
            loss = cross_entropy(model.tier1_head(hidden[18]), labels)
                 + λ2 * cross_entropy(model.tier2_head(hidden[32]), labels)
                 + λ3 * cross_entropy(model.tier3_head(hidden[40]), labels)
        loss.backward()
        optimizer.step()
```
关键设计决策：(a) Step-by-step而非joint training——避免多tier loss梯度冲突导致低tier性能不可控；(b) 对较低tier使用较高λ权重以补偿其较小的模型容量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
论文在2×8 H800 SXM5 (80GB) GPU上实现Knowledge Precipitation，两台服务器通过8×400Gbps NDR InfiniBand互联。约24小时完成Llama2-13B的3-tier fine-tuning（2500 iterations）。论文受资源限制（16×H800）未实际fine-tune Llama2-70B级别模型，大模型家族规模上的成本外推仍需验证。论文未开源Knowledge Precipitation实现代码。该方法目前是MFS系统的专有技术，未见其他系统采用类似方法。通用化使用时需考虑：(1) 需要模型家族中最大的模型的checkpoint作为起点；(2) full-parameter fine-tuning成本随模型规模线性增长；(3) tier边界的latency测量需在目标推理硬件上进行。

涉及论文标题：
- MFS: An Efficient Model Family Serving System for LLMs

## Multi-Tier Model（多层级嵌套模型）

术语是什么？通过联网搜索让回答具体和精准。
Multi-Tier Model（多层级嵌套模型）是MFS论文提出的一种模型结构设计，将LLM model family中不同规模的模型（如7B/13B/70B）折叠为单一checkpoint中的多个执行层级（tier）。每个tier在transformer的不同layer深度有独立的输出头（lm_head），低tier对应小模型（浅层，低延迟），高tier对应大模型（深层，高质量）。所有tier共享低层Transformer参数，不同tier之间通过Knowledge Precipitation保证各自的语言建模能力。该结构是MFS实现高效model family serving的基础——使得不同"模型大小"的请求可以共享参数、KV cache和batch执行。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Multi-Tier Model在推理时的执行流（以Llama2-13B转3-tier模型：tier-1=18层/~3B等效, tier-2=32层/~10B等效, tier-3=40层/13B为例）：

```
def multi_tier_inference(x, requested_tier):
    hidden = x
    # Tiers 1-3 all share these layers
    for layer in range(0, 18):  # Tier-1 prefix
        hidden = transformer_block[layer](hidden)
    kv_cache.store(layers_0_17, hidden)  # Mark as shareable
    
    if requested_tier == 1:
        return tier1_lm_head(hidden)  # Early exit: ~3B quality
    
    # Tiers 2-3 share layers 18-31
    for layer in range(18, 32):  # Tier-2 extension
        hidden = transformer_block[layer](hidden)
    kv_cache.store(layers_18_31, hidden)
    
    if requested_tier == 2:
        return tier2_lm_head(hidden)  # ~10B quality
    
    # Tier-3 only: layers 32-39
    for layer in range(32, 40):
        hidden = transformer_block[layer](hidden)
    return tier3_lm_head(hidden)  # Full 13B quality
```

关键设计约束：
1. **Layer-only split**：只按layer切分tier，不切head维度。因为attention计算需要所有head一致性→切head会破坏各tier KV cache的维度一致性→无法跨tier共享KV cache。
2. **连续性**：tier边界是连续的layer范围，不是删除中间层（deep pruning）。层连续性保证增量计算正确（tier切换时第N+1层的输入hidden state = 第N层的输出）。
3. **Latency-aligned边界**：tier边界不按参数量线性切分（如一半参数量取一半层），而是测量最大模型前若干层在目标硬件的实际推理latency，使各tier的端到端latency对齐对应规模独立模型。例如Llama2-13B前24层latency对齐Llama2-7B，取24层而非20层（按参数量）。
4. **Nested hierarchy**：tier-1 ⊂ tier-2 ⊂ tier-3，即低tier层是高tier层的严格前缀。这确保了group batching和KV cache sharing的正确性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MFS的Multi-Tier Model通过Knowledge Precipitation（step-by-step fine-tuning）从最大模型的checkpoint构造。论文在Llama2-7B-chat → 2-tier、Llama2-13B-chat → 2-tier和3-tier（13B/7B/3B、13B/10B/7B组合）上验证，并在Qwen-14B → Qwen-7B上做泛化验证。质量评估显示各tier在MMLU/PIQA/OpenBookQA/HellaSWAG/BoolQ/ARC/ANLI等10个benchmark上的表现接近或优于对应规模的独立模型。该设计目前仅见于MFS系统，未见其他serving系统使用类似结构。该方法的局限：(1) 仅适用于同一model family（共用相同架构、tokenizer和vocabulary的模型）；(2) tier数量受层数和layer粒度限制；(3) fine-tuning成本随tier数和最大模型规模增长。

涉及论文标题：
- MFS: An Efficient Model Family Serving System for LLMs

## RFLoRA (Resource-Friendly Low-Rank Adaptation)（资源友好型低秩适配）

术语是什么？通过联网搜索让回答具体和精准。

RFLoRA是TailorLLM提出的LoRA变体，针对端云协同LLM推理中的adapter传输和存储开销优化。核心创新：(1) 参数解耦：观察到跨任务微调时LoRA的A矩阵趋于收敛（capture domain-invariant encoder features），B矩阵呈现任务特异性（adapt to domain-specific transformations），因此冻结共享A矩阵（所有任务共用一份）、仅训练和传输B矩阵；(2) 方向-幅度分解：将预训练权重W解耦为magnitude（列范数m = ||W||_c ∈ R^d）和direction（列归一化矩阵W/||W||_c），仅对direction分量应用LoRA低秩分解，幅度分量作为可训练标量独立优化。此设计使端侧仅需预存一份共享A矩阵，传输量从标准LoRA的22MB降至~11.56MB（约50% reduction），同等存储空间可容纳的adapter数量翻倍。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
输入: 预训练权重 W₀ ∈ R^{d×k}, rank r, 下游任务数据
输出: 压缩的task-specific adapter (B ∈ R^{d×r}, m ∈ R^d)

// Step 1: 权重分解
1: m = ||W₀||_c ∈ R^d           // column-wise L2 norm (magnitude)
2: V = W₀ / ||W₀||_c             // column-wise normalized (direction)
3: // W₀ = diag(m) · V

// Step 2: 初始化低秩矩阵（仅一次，跨任务共享A）
4: A ∈ R^{r×k} ~ Kaiming(seed=固定)  // 全局冻结，所有任务共享
5: B ∈ R^{d×r} = zeros(d, r)         // zero初始化，任务特定可训练

// Step 3: 训练（仅更新B和m）
6: for each training step:
7:     ΔV = B @ A                    // [d×r] @ [r×k] → [d×k]
8:     V' = V + ΔV                   // direction update
9:     W' = diag(m) · V' / ||V'||_c  // 重新组合并归一化direction
10:     // 等价于: W' = m · (W₀ + BA) / ||W₀ + BA||_c
11:     loss = task_loss(model(x|W'), y)
12:     m.grad, B.grad ← backward(loss)  // A无梯度
13:     m, B ← optimizer.step()

// Step 4: 传输（仅B + m，约标准LoRA的50%）
14: 云端→端侧: 传输B (d×r FP16) + m (d FP16)
// Step 5: 端侧推理
15: W' = m · (W₀ + BA) / ||W₀ + BA||_c  // 合并后推理，无额外延迟
```

RFLoRA的关键设计决策：(1) A用Kaiming初始化并冻结（与LoRA的随机初始化+可训练不同），因为A主要作为encoder投影输入到子空间；(2) B用zero初始化确保训练起始W'=W₀；(3) 方向归一化||W₀+BA||_c使magnitude和direction分量解耦，magnitude分量不受frozen A约束而独立优化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

RFLoRA在TailorLLM中通过PyTorch实现，在Llama3-1B的所有linear层上应用（r=16）。训练在云端RTX 3090上进行，与标准LoRA共享相同的训练schedule和超参数。每任务仅需存储和传输B矩阵(约11.56MB)和magnitude参数m(约几KB)，端侧预存一份共享A矩阵(~10.5MB)。在8个NLP benchmark上，RFLoRA以3.4M trainable params（0.273% of full model）达到81.6% avg accuracy，超越标准LoRA（81.2%, 0.454% params）和AdaLoRA（81.0%, 0.680% params），与DoRA（82.1%, 0.484% params）精度接近但参数量仅为其56%。论文未明确说明代码开源。

涉及论文标题：
- TailorLLM: Collaborative End-Cloud Inference of Large and Small Language Models Based on Low-Rank Adaptation

## Mamba (State Space Model) for Time Series in LLM Systems（面向LLM系统的Mamba时序模型）

术语是什么？通过联网搜索让回答具体和精准。

Mamba是由Gu和Dao(2024)提出的基于State Space Model (SSM)的序列建模架构，通过选择性状态空间（selective state space）机制突破传统SSM的线性时不变限制。核心公式：h_t = exp(Δt·A)·h_{t-1} + (Δt·A)^{-1}·(exp(Δt·A) - I)·Δt·B·x_t，其中Δt = softplus(W_Δ·x_t + b_Δ)是输入依赖的选择性更新门，A和B是连续时间系统参数。相比RNN，Mamba训练时支持并行扫描（parallel scan），推理时recurrent逐步计算；相比Transformer，Mamba具有线性计算复杂度（而非二次attention），在长序列上参数效率更高。

在TailorLLM的AdapterMgr中：Mamba被用作时间序列特征提取器，从用户历史访问序列中提取时序模式。单层Mamba Block以滑动窗口H=100的访问序列为输入，通过隐藏状态h_t ∈ R^{128}递推式地编码用户行为的长程依赖。训练时使用Parallel模式批量处理序列，推理时使用Recurrent模式逐步更新状态。Mamba在这里替代了传统RNN/LSTM（无法并行训练）和CNN（局部感受野无法捕捉长程依赖）用于cache替换决策的时序编码。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Mamba作为AdapterMgr时序特征提取器的计算流程：
```
输入: 用户历史访问序列 x_{t-H+1:t} ∈ R^{H×d} (H=100, d=128)
输出: 时序特征 h_t ∈ R^d

// 训练时 Parallel Mode:
1: Δ_{t-H+1:t} = softplus(W_Δ @ x_{t-H+1:t} + b_Δ)  // 选择性gate
2: Ā_i = exp(Δ_i · A)                                // 离散化状态转移
3: B̄_i = (Δ_i · A)^{-1} · (exp(Δ_i · A) - I) · Δ_i · B  // 离散化输入投影
4: h_{t-H+1:t} = parallel_scan(Ā, B̄, x)               // 并行扫描

// 推理时 Recurrent Mode:
5: h_t = Ā_t · h_{t-1} + B̄_t · x_t                    // 逐步递推
```

在AdapterMgr pipeline中的位置：
```
6: E(X) = W_x · X + positional_encoding    // 用户序列embedding (H=100, d=128)
7: E(L) = W_l · L                           // cache状态embedding (w=5, d=128)
8: h_t = Mamba(E(X))                        // Mamba提取时序特征 [H×d] → [d]
9: F_fused = Concat(W_f · E(L), h_t)       // 双模态融合
10: F_out = LayerNorm(F_fused)
11: π̂ = Softmax(MLP(F_out))                 // 输出替换策略向量
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Mamba开源实现：https://github.com/state-spaces/mamba (CUDA官方kernel + PyTorch接口)。通过`Mamba(d_model, d_state, d_conv)`类实例化，支持`mamba.forward(x)`。在TailorLLM的AdapterMgr中，使用单层Mamba Block（d_model=128）作为轻量级时序编码器，训练时输入shape为[batch, H, d]的可变长序列。选择Mamba而非Transformer/RRNA的原因：(1) 并行训练效率高；(2) 可建模全局时序依赖；(3) 参数效率——单层即可达到或超过Transformer多层性能。论文未明确说明Mamba block的具体超参数（d_state, d_conv）。

涉及论文标题：
- TailorLLM: Collaborative End-Cloud Inference of Large and Small Language Models Based on Low-Rank Adaptation

## S-L Distance / S-L Similarity（SLM-LLM距离/相似度）

术语是什么？通过联网搜索让回答具体和精准。

S-L Distance（SLM-LLM Distance）是AIMS提出的度量，表示LLM生成的一个subtask需要多少额外的SLM subtask才能产生与之相似（匹配）的输出。具体定义：对于LLM的subtask L_i，如果SLM在执行了d个额外subtask后产生的subtask S_{i+d}与L_i语义相似（SBERT cosine similarity > threshold），则L_i的S-L distance为d。若无法找到匹配的SLM subtask，S-L distance设为infinity。S-L Similarity则是在该匹配点的SBERT余弦相似度值。该度量刻画了SLM和LLM在subtask粒度上的不对齐程度——SLM通常产出更细粒度的subtask，需要多个SLM subtask才能覆盖一个LLM subtask的内容。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

S-L Distance的计算过程（基于offline profiling数据）：

```
// S-L Distance计算（offline profiling阶段）
Input: LLM subtask sequence L = [L1, L2, ..., Ln]
       SLM subtask sequence S = [S1, S2, ..., Sm] (m >= n, SLM更细粒度)
       similarity threshold κ

Output: distance array D[1..n]

for each LLM subtask L_i:
    matched = false
    for d = 0 to (m - i):  // 搜索SLM中对应位置
        if SBERT_cosine_sim(S_{i+d}, L_i) > κ:
            D[i] = d
            matched = true
            break
    if not matched:
        D[i] = INF  // 无匹配SLM subtask
```

论文实验发现（Figure 6）：匹配组（SLM和LLM最终输出一致）中，随着subtask序列推进，S-L distance逐渐增大（后期LLM subtask需要更多SLM subtask才能匹配）；非匹配组中，许多S-L distance达到infinity。例如，对于一个4-ST请求，第1个subtask平均S-L distance≈1，第4个可能达到2-3。这一观察动力了AIMS的SLE和CD组件设计。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

S-L Distance在AIMS中通过Distance Predictor（ModernBERT + LoRA fine-tune）在线预测：输入为当前subtask内容和sequence ID，输出为预测的S-L distance。预测的distance用于SLE组件——SP_SLM预测第(i+d)个subtask内容，与SP_LLM预测的第i个subtask内容比较similarity。若similarity > κ，则SLM执行当前及后续d个subtask。训练数据来自offline profiling中对所有请求生成SLM/LLM binary tree后提取的S-L distance labels。S-L distance使AIMS能在SLM结果与LLM不同时仍找到"追赶路径"，是区别于HybridLLM/Minions独立per-subtask决策的关键机制之一。

涉及论文标题：
- AIMS: A Cost-Efficient Framework for LLM-based Agent Deployment in Cloud-Edge Hybrid Environments

## Subtask Decomposer for Agent Workflows（Agent工作流的子任务分解器）

术语是什么？通过联网搜索让回答具体和精准。

Subtask Decomposer (SD)是AIMS中的一个关键recovery组件，当一个复杂subtask通过SSE/SLE/CD都无法安全分配给SLM时，SD将其拆解为更简单、更细粒度的子subtask序列，使SLM能够逐步处理。SD基于Qwen3-0.6B + LoRA fine-tune，输入为当前subtask内容和SP_LLM预测的next subtask（作为分解目标），输出为分解后的子subtask序列{SSP_1, SSP_2, ..., SSP_m}。之后AIMS对每个子subtask重跑SSE，仅当所有子subtask都通过SSE（即SLM和LLM对每个子subtask的next prediction相似）时才将整组分配给SLM；否则将原始subtask交给LLM。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Subtask Decomposer的决策流程：

```
// Subtask Decomposer工作流
function subtask_decomposer_route(ST_i):
    // 用SD模型将复杂subtask分解为子subtask序列
    next_llm = SP_LLM.predict(ST_i)
    sub_subtasks = SD.decompose(ST_i, next_llm)
    // sub_subtasks = {SST_1, SST_2, ..., SST_m}
    
    // 逐个子subtask评估LLM-SLM一致性
    for each SST_j in sub_subtasks:
        next_slm_j = SP_SLM.predict(SST_j)
        next_llm_j = SP_LLM.predict(SST_j)
        sim_j = SBERT_cosine(next_slm_j, next_llm_j)
        
        if sim_j < κ_position(j):
            // 任一子subtask不通过 → 原始subtask走LLM
            return LLM
    
    // 全部通过 → 整组给SLM
    for each SST_j in sub_subtasks:
        SLM.process(SST_j)
    return combined_result
```

论文示例：HotpotQA中subtask "Verify Shirley Cameron's father, including corroborating biographical details, to confirm his identity as James Cameron's maternal grandfather" → SD分解为：SST1="Search for Shirley Cameron's father" → SST2="Extract father's full name" → SST3="Find key biographical details for verification (e.g., birth/death dates)" → SST4="Confirm and state the maternal grandfather's name"。四个子subtask比原始复杂subtask更容易通过SSE检查。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

SD的训练数据来自offline profiling：对请求的subtask binary tree中SLM生成的更细粒度subtask序列与LLM的对应subtask做配对，从中学习"如何将一个LLM-level subtask分解为SLM-level子subtask序列"。SD采用整组分配策略（全部分解子subtask适合SLM才offload整组，否则走LLM）以避免增加LLM调用次数——若逐个子subtask独立路由，可能部分走SLM部分走LLM，反而增加总LLM调用。实验消融：移除SD使SLM usage下降5.54%、accuracy下降1.58%。SD的局限：(1) 保守策略可能错失混合case（部分子subtask可SLM）；(2) 依赖SP_SLM/SP_LLM的预测准确性。

涉及论文标题：
- AIMS: A Cost-Efficient Framework for LLM-based Agent Deployment in Cloud-Edge Hybrid Environments

## SBERT Similarity for Agent Output Alignment（用于Agent输出对齐的SBERT相似度）

术语是什么？通过联网搜索让回答具体和精准。

SBERT（Sentence-BERT，Reimers & Gurevych, 2019）是BERT的孪生网络变体，通过pooling操作将句子映射到固定维度的embedding空间，使语义相似的句子在embedding空间中有较高的余弦相似度。AIMS使用SBERT embedding的cosine similarity作为衡量SLM和LLM输出对齐程度的核心度量：(1) 请求级：比较整请求在All-SLM和All-LLM下的最终输出embedding相似度；(2) subtask级：比较SP_SLM和SP_LLM预测的next subtask embedding相似度；(3) 收敛检测：比较未来SLM-LLM subtask pair的embedding相似度。阈值默认0.7（empirically determined），低于此阈值认为输出不similar，需走LLM或进入recovery path。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// AIMS中的SBERT相似度计算
function SBERT_similarity(text_a, text_b):
    emb_a = SBERT.encode(text_a)  // shape: [1, 384] (使用all-MiniLM-L6-v2等)
    emb_b = SBERT.encode(text_b)  // shape: [1, 384]
    
    cos_sim = (emb_a · emb_b) / (||emb_a|| * ||emb_b||)
    // 值域: [-1, 1], 0.7+ 认为相似
    
    return cos_sim

// 在SSE中使用
similarity_score = SBERT_similarity(
    SP_SLM.predict(ST_i),   // SLM预测的next subtask
    SP_LLM.predict(ST_i)    // LLM预测的next subtask
)
```

AIMS选择SBERT similarity而非精确匹配（如BLEU/ROUGE）的原因：AI agent的subtask是自然语言描述的动作/决策，语义等价但措辞不同的subtask应视为相似。例如SLM的"Search for James Cameron's mother"和LLM的"Find the director's maternal parent"语义高度相似但文本不同，SBERT能捕捉这种semantic alignment。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

SBERT开源实现：https://github.com/UKPLab/sentence-transformers（`pip install sentence-transformers`）。常用模型包括`all-MiniLM-L6-v2`（384维，快速）和`all-mpnet-base-v2`（768维，更准确）。AIMS使用SBERT替代BERTScore作为主要相似度度量（Section 2.2中BERTScore用于WebShop的ground-truth匹配评估），因为SBERT的sentence-level embedding更适合比较不同长度的subtask描述。SBERT similarity同时用于：(1) URC训练标签（All-SLM vs All-LLM输出相似度）；(2) SSE/CD的在线判断；(3) offline profiling数据分析。相似度阈值0.7是论文在多个数据集上empirically确定的，Section 4.6的sensitivity study显示阈值在0.66-0.74范围内提供良好accuracy-SLM usage balance。

涉及论文标题：
- AIMS: A Cost-Efficient Framework for LLM-based Agent Deployment in Cloud-Edge Hybrid Environments

---

## UI Navigation Graph (UNG)

术语是什么？

UI Navigation Graph (UNG) 是以图结构建模GUI应用导航关系的学术术语。UNG的节点是Windows UI Automation (UIA) 暴露的控件，边表示"点击一个控件后可达的新控件"的导航关系。与静态的UI accessibility tree不同，UNG捕获的是应用功能导航拓扑——哪些菜单/按钮/选项卡的点击会导致哪些新控件/面板/对话框出现。UNG的特点是包含环路（如多路径到达同一功能、反复切换tab）和merge node（不同路径汇聚到同一控件），这使得原始UNG不是树结构而是带环有向图。UNG用于Computer-Use Agent的确定性导航规划——将LLM从"猜测导航路径"解放出来，转为图上的确定性路径求解。

从算法pipeline角度拆解术语：

DMI中UNG的构建pipeline：

```
输入: 目标应用进程 + UIA root element
输出: UNG (节点=UIA控件, 边=点击后可达关系)

1. 初始化: root_node ← UIA.GetRootElement(app_window)
2. DFS构建:
   for each unexplored_node in current_tree:
       state_before ← UIA.CaptureTree()       // 记录当前控件树快照
       UIA.InvokePattern(unexplored_node).Invoke()  // 点击候选控件
       WaitForUIStabilization()                // 等待新UI出现
       state_after ← UIA.CaptureTree()         // 记录点击后控件树快照
       new_controls ← state_after - state_before
       for each new_ctrl in new_controls:
           add_edge(unexplored_node, new_ctrl) // 新出现的控件构成导航边
       if new_top_window_detected:             // 新顶层窗口
           push_to_stack(new_window)           // 递归探索
3. Differential capture处理:
   - 使用process_id + window listener检测新顶层窗口/modal window
   - 跟踪已访问节点，避免无限循环
4. 控件ID分配:
   - XPath-like格式: primary_id|control_type|ancestor_path
   - primary_id优先级: UIA automation_id > 控件名称 > "[Unnamed]"
```

关键设计：differential capture确保只记录"由点击触发的新出现控件"而非所有已有控件，过滤掉UI re-render的噪声。

术语一般如何实现？如何使用？

UNG构建在DMI中实现为自动探索脚本：(1) 对每个Office应用（Word/Excel/PowerPoint），自动化DFS探索 < 3小时；(2) 人工配置约1.5人日（标记access blocklist、选择context-aware exploration的代表对象）；(3) UNG构建结果包含5K+控件节点和对应的导航边；(4) UNG的环路和merge node导致从LLM角度看存在歧义路径，因此需要进一步转为path-unambiguous forest。UNG是实现policy-mechanism separation的基础数据——它将"如何导航"的知识编码为可供确定性算法查询的数据结构，而非让LLM在prompt中推理导航路径。

涉及论文标题：
- From Imperative to Declarative: Towards LLM-friendly OS Interfaces for Boosted Computer-Use Agents

---

## Path-Unambiguous Forest

术语是什么？

Path-Unambiguous Forest 是将带环和merge node的UI Navigation Graph (UNG) 转换为无歧义路径的森林结构的过程。原始UNG中存在两种歧义：(1) 环路——同一功能可通过多条不同路径到达（如通过File菜单和通过Quick Access Toolbar均可到达Save），等价路径使导航算法面临选择；(2) merge node——多条路径汇聚到同一控件（如不同tab下均有指向同一dialog的button），在图遍历中产生路径交汇。Path-Unambiguous Forest通过保留main tree（每个功能的canonical路径）和共享shared subtrees（merge node引用的可复用子树），使得从任意root到任意target的导航路径唯一确定——LLM只需声明目标ID，DMI执行确定性路径求解。

从算法pipeline角度拆解术语：

Path-Unambiguous Forest的转换pipeline：

```
输入: UNG (带环 + merge node的有向图)
输出: Path-Unambiguous Forest (main tree + shared subtrees)

1. 环路检测与消解:
   for each cycle in UNG:
       选择canonical entry作为main tree路径
       其余等价路径在core topology中标注但不作主路径
       消解策略: 优先保留更短/更稳定的路径

2. Merge node处理:
   for each node N with indegree > 1:  // merge node
       在main tree中保留N的一个出现位置
       将N及其子树从main tree中移除（避免重复）
       将N标记为shared subtree (entry_ref = 原始父节点路径)
       在各使用N的父节点处添加引用指针

3. Forest生成:
   - Main tree: root → ... target nodes (唯一canonical路径)
   - Shared subtrees: merge node + 其children (多父节点共享)
   - 每个共享子树节点携带entry_ref_id: 标识从哪个main tree节点进入subtree
```

Forest结构的关键属性：从root到任意控件的导航路径是path-unambiguous的——给定目标控件ID和可选的entry_ref_id，存在唯一的导航控件序列。

术语一般如何实现？如何使用？

在DMI在线执行中：(1) LLM声明目标控件leaf ID（整型），若在shared subtree中则附带entry_ref_id；(2) DMI executor在forest中执行确定性路径求解——从root沿main tree到entry point，再沿shared subtree到目标，整条路径是唯一确定的；(3) forest消除了"LLM不知道选择哪条路径"的歧义——baseline在使用UNG信息时SR反而从44.4%降到42.0%，而DMI通过forest+确定性solver使SR提升到74.1%，部分收益来自消除路径歧义。

涉及论文标题：
- From Imperative to Declarative: Towards LLM-friendly OS Interfaces for Boosted Computer-Use Agents

---

## Core Topology

术语是什么？

Core Topology 是DMI将UI Navigation Forest压缩为LLM context window友好的层级文本描述的技术。原始UNG/Forest包含5K+控件节点，若全部序列化入prompt将远超合理token预算（即使大context，过长控件列表也会稀释task-relevant信息、增加推理成本）。Core Topology通过分层描述+连续整型ID编码，将完整forest保持在一个可管理的token预算内（Excel ~30K tokens, Word ~15K, PowerPoint ~15K），同时保留按需扩展能力（`further_query`机制）。

从算法pipeline角度拆解术语：

Core Topology的压缩pipeline：

```
输入: Path-Unambiguous Forest (main tree + shared subtrees)
输出: Core Topology文本 (分层描述 + 整型ID映射)

1. 整型ID映射:
   将冗长XPath-like控件ID: "Blue|ControlType.Button|ribbon/Design/FormatBackground/FillColor/"
   替换为连续整型ID: 128

2. 层级结构序列化:
   使用紧凑格式: name(type)(description)_id[children]
   例: "Design(Tab)_12[FormatBackground(Group)_45[SolidFill(RadioButton)_67, ...]]"

3. 分层描述策略:
   - Level 1 (Core): 最常用的功能路径 → 默认包含在prompt中
   - Level 2+ (Extended): 低频或专门化功能 → 按需通过further_query扩展
   - 非叶节点: 保留但标记为navigation-only
   - 叶节点: 可直接作为LLM声明访问的目标

4. Shared subtree引用:
   共享子树不重复序列化，在引用处标注entry_ref_id
   例: "...[ApplyToAll(Button)_132@entry_ref=15]"
```

压缩密度：原始5K+控件 → ~15K-30K tokens → LLM可一次加载并在多次交互轮次中复用。

术语一般如何实现？如何使用？

Core Topology在DMI prompt中的使用：(1) 作为system message或task preamble注入LLM prompt；(2) LLM通过阅读层级描述理解应用的功能组织——类似于压缩的"功能地图"；(3) LLM声明目标时引用整型ID（如 `"ids": [128]`），DMI executor反向映射为XPath-like path；(4) 当LLM需要的功能不在core topology中（提示not found），可通过`further_query`请求扩展对应子树（DMI动态追加到prompt context）；(5) 离线构建时决定哪些分支放core vs 需要query——论文报告Excel未压缩core topology ~30K tokens，人工配置决定branch的core/extended划分。与直接将5K+控件全量注入prompt相比，Core Topology既保持了信息完整性（需要时扩展），又避免了context dilution（默认仅core）。

涉及论文标题：
- From Imperative to Declarative: Towards LLM-friendly OS Interfaces for Boosted Computer-Use Agents

## Mask-aware Cached Activation / Y-Activation Cache for Diffusion Models（面向扩散模型的Mask感知激活缓存）

术语是什么？通过联网搜索让回答具体和精准。

Mask-aware Cached Activation是FlashPS提出的针对图像编辑扩散模型推理的激活复用技术。核心思想：在generative image editing serving中，用户请求通常包含mask指定待编辑的局部区域，其余unmasked区域应保持不变。FlashPS预先计算模板在无编辑条件下的transformer block输出activation Y（非K/V cache），在线推理时对unmasked token直接从cache读取Y并注入transformer block输出，仅对masked tokens执行完整attention和feed-forward计算。选择缓存输出activation Y而非K/V cache的原因是：(1) unmasked token的输出Y在相同模板的不同编辑请求间高度相似；(2) Y的缓存量约为K/V的一半，显著降低cache footprint；(3) masked/unmasked cross-attention较弱，仅关注masked tokens的计算不显著影响质量（SSIM可达0.99）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FlashPS mask-aware cached activation的算法pipeline（以SDXL/Flux DiT为例）：

```
输入: latent (spatial tokens), mask (binary, 1=masked/需编辑), template_id
输出: edited_latent

// 离线预计算阶段（每个template执行一次）
1: cached_activations = {}  // {block_id: unmasked_Y}
2: latent_full = VAE.encode(template_image)
3: for each transformer_block in DiT:
4:     Y_full = transformer_block(latent_full, timestep, cond)
5:     cached_activations[block_id] = Y_full[mask == 0]  // 仅保存unmasked tokens的Y
6: store_to_host_memory(template_id, cached_activations)

// 在线推理阶段（每个编辑请求）
7: latent_noisy = add_noise(VAE.encode(input_image), sigma_t)
8: for t in denoising_steps (T→1):
9:     for each transformer_block in DiT:
10:        // DP决定此block是否使用cache（见Bubble-free DP Block Selection）
11:        if use_cache[block_id]:
12:            cached_Y_unmasked = async_load(template_id, block_id)  // CUDA stream异步从host memory加载
13:            Y_masked = transformer_block_compute_masked_only(
14:                latent_noisy[mask==1],    // 仅masked tokens参与attention作为query
15:                latent_noisy,             // 全量tokens作为key/value context
16:                timestep, cond)
17:            Y_full = merge(Y_masked, cached_Y_unmasked, mask)  // masked位置用Y_masked，unmasked用cached
18:        else:
19:            Y_full = transformer_block_full(latent_noisy, timestep, cond)  // 全量计算
20:        latent_noisy = Y_full
21:    latent_noisy = scheduler_step(latent_noisy, predicted_noise, t)
22: edited_image = VAE.decode(latent_noisy)
```

计算复杂度：标准全量attention+FFN对N个tokens为O(N²d + Nd²)。FlashPS仅对masked tokens（≈mN个，m为mask ratio）做attention中的Q·K^T和weighted sum of V，以及对masked tokens做FFN。unmasked tokens的Y直接来自cache（O((1-m)N·d)的读取开销）。理论加速比约为1/m（mask ratio 0.11时理论≈9x）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. **Cache存储**：cached activations存储在host memory（CPU DRAM），也可使用disk或distributed storage作为二级存储。FlashPS生产trace中仅970个templates，每个模板缓存所有transformer block的Y（FP16），总cache大小可控（host memory容纳）。
2. **CUDA Stream异步加载**：创建独立的CUDA stream用于cache load，与computation stream并行。cache load stream通过cudaMemcpyAsync从host memory拷贝cached Y到GPU HBM，computation stream同步执行masked token的attention/FFN计算。
3. **Y-cache vs K/V-cache选择**：论文量化分析——mask ratio 20%时，缓存K/V可将latency从2.27s降至2.06s（比缓存Y额外降低约10%），但cache size翻倍（需存K和V两个tensor）。缓存Y是cache size与speedup的帕累托最优。
4. **适用条件**：收益依赖mask区域较小（生产trace平均mask ratio=0.11）、模板复用率高（970模板各平均复用约35k次）、编辑确实保持unmasked region不变。对style transfer等全局改变任务收益下降。

涉及论文标题：
- FlashPS: Efficient Generative Image Editing with Mask-aware Caching and Scheduling

## ILP/TLP/Arithmetic Intensity Trade-off in DNN Kernels

术语是什么？通过联网搜索让回答具体和精准。
ILP (Instruction-Level Parallelism)、TLP (Thread-Level Parallelism) 和 Arithmetic Intensity 是GPU DNN kernel优化的三维trade-off空间。ILP控制单warp内独立指令的并行度（通过register file中tile size和instruction scheduling控制），TLP控制SM内并发active warp数量（主要受register和shared memory per-thread使用量约束），Arithmetic Intensity = #arithmetic_ops / #bytes_accessed，控制计算与访存的比值（通过tile size调节数据复用度）。三者竞争有限的register file和shared memory资源：更高ILP需要更多register→降低TLP；更高intensity需要更大tile→更多shared memory→可能降低TLP。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Infera的形式化分析（Eq.1-4）：
```
// 执行时间 = 指令数 / IPC
#inst ≈ #inst_mem + #inst_math

// 内存指令数估算（Eq.2+3）
#inst_mem ≈ a × #bytes + b × #ops
         = #ops × (a/I + b)
// I = #ops / #bytes = arithmetic intensity

// Trade-off约束（Eq.4）
min #inst  s.t. ILP constraints, TLP constraints, intensity constraints
  ILP ↑ → #inst ↓ but register ↑ → TLP ↓
  Intensity ↑ → #inst ↓ but shared memory ↑ → TLP ↓
  TLP depends on: register/thread × thread/SM ≤ RF_size/SM
                   shared_memory/block ≤ SMEM_size/SM
```

**具体例子（GEMM kernel on A100）**：
- Low ILP, High TLP (register=64/thread): 更多active warps但每warp内指令串行依赖多
- High ILP, Low TLP (register=128/thread): 每warp内更多independent指令但occupancy降低
- 峰值性能出现在green box (Figure 4): ILP, TLP, intensity三者平衡处

**Infera的tile配置实现trade-off**：
- Register level: 64/96/128 registers/thread → ILP vs TLP
- Shared memory level: 48/80/112/144 KiB/block → tile size控制intensity
- Pipeline stages: 2/3/4 → async copy overlap
- 通过spatial vs reduction axis tile size ratio进一步调节ILP vs intensity

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Infera compiler在生成multi-version kernels时系统化覆盖trade-off空间：register 3种×shared memory 4种×pipeline 3种=36种基础配置组合。每种配置通过top-down tile size derivation从register level向下传播到shared memory和global memory level。Cut-and-patch instruction scheduling进一步在SASS level优化ILP（通过list scheduling减少stall cycles）。推理时SelectKernels根据当前GPU occupancy状态和kernel hazard分析选择最优配置（通过online regression model估计IPC）。该trade-off分析同时指导了Infera compiler的warp specialization设计（4 mainloop + 4 copy warps固定分配，GPU scheduler将每组4连续warp map到同一SM的4 SMSP）。

涉及论文标题：
- Automated End-to-End Model Serving with Cooperative Compilation and Scheduling

---

## Flow Matching（流匹配）

术语是什么？通过联网搜索让回答具体和精准。
Flow Matching（流匹配）是由Lipman et al. (2023)提出的一种生成模型训练范式，作为Classical Diffusion (DDPM/DDIM)的替代方案。它在latent space中学习一个continuous velocity field v_Θ(z_t, t, y)，将噪声平滑地"流动"到数据分布。给定target latent z_0和噪声样本ε ∼ N(0,I)，通过线性插值 z_t = (1-t)z_0 + tε（t∈[0,1]），监督目标为velocity (ε-z_0)而非噪声ε：L_fm = ||(ε-z_0) - v_Θ(z_t,t,y)||²₂。与DDPM的离散马尔可夫链不同，Flow Matching使用连续时间ODE，消除了对固定noise schedule的依赖。代表性模型：Flux（Black Forest Labs, 2024）、Stable Diffusion 3（Esser et al., 2024）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Flow Matching训练pipeline（以Flux在latent space上的训练为例）：

```
// 训练阶段
1: x = real_image ∈ R^{H×W×3}
2: z_0 = VAE.encode(x) ∈ R^{C×h×w}          // 压缩到latent
3: ε ~ N(0, I)                                // 采样高斯噪声
4: t ~ Uniform(0, 1)                          // 连续时间采样
5: z_t = (1-t) * z_0 + t * ε                  // 线性插值
6: v_pred = v_Θ(z_t, t, text_conditioning)    // 预测velocity field
7: target = ε - z_0                           // ground truth velocity（从噪声指向数据）
8: loss = MSE(v_pred, target)                  // 回归velocity

// 推理阶段（从纯噪声生成）
1: z_1 ~ N(0, I)                              // 纯噪声（t=1）
2: for t = 1, 1-dt, ..., dt:                  // ODE求解
3:     z_{t-dt} = z_t - v_Θ(z_t, t, y) * dt  // Euler步或更高级ODE solver
4: image = VAE.decode(z_0)                    // latent→pixel
```

与Classical Diffusion的关键区别：
- DDPM学习预测噪声ε，Flow Matching学习预测velocity (ε-z_0)
- DDPM使用离散timestep t∈{1,...,T} + noise schedule β_t，Flow Matching使用连续t∈[0,1]
- Flow Matching的线性插值路径使得训练更稳定，采样可用更少的ODE步数

**x-prediction vs v-prediction parameterization（ELF论文的关键选择）：**
标准Flow Matching预测velocity v = x - ϵ，但也可reparameterize预测x（clean data）或ϵ（noise）。ELF选择x-prediction的原因：
- x-prediction在高维space（512/768/1024-dim）中保持稳定，v-prediction在高维退化，ϵ-prediction collapse
- x-prediction预测clean embeddings，与final-step token decoding（CE loss）目标一致，使shared-weight denoiser-decoder可行
- 训练loss等价转换：L_MSE = ||v_θ - v||² = ||(x_θ - x)/(1-t)||²（通过v = (x - z_t)/(1-t)关系转换）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Flow Matching已被主流扩散模型采用：Flux-1.dev、SD3、URAE等。LWD论文在Flow Matching基础上引入wavelet-based spatial masking：对latent z_t计算wavelet energy saliency map Awavelet，生成time-dependent mask M_t，最终loss变为 L_masked = ||M_t ⊙ [(ε-z_0) - v_Θ(z_t,t,y)]||²₂。训练后inference与原始Flow Matching完全相同。

ELF论文将Flow Matching扩展到语言建模领域（文本生成为"text-to-text" generation），使用x-prediction parameterization（预测clean embeddings x而非velocity v）替代标准v-prediction。x-prediction使shared-weight denoiser-decoder成为可能（denoising和decoding均predict clean embeddings），在高维embedding space（512/768/1024-dim）中比v-prediction/ϵ-prediction更稳定。ELF使用rectified flow linear interpolant: z_t = t·x + (1-t)·ϵ，训练目标 L_MSE = ||(x_θ(z_t,t) - x)/(1-t)||²。此外ELF还支持SDE-inspired sampler：在每个ODE step注入Gaussian noise（z_back = α·z + (1-α)·ε, α = 1-γ·dt），在perturbed state上重预测x̂，用原z更新。γ=0退化为ODE，γ>0引入stochasticity以纠正early denoising errors。

实现上，Flow Matching使用standard PyTorch/JAX training loop，与logit-normal time schedule（P_mean=-1.5, P_std=0.8）配合使用。in-context conditioning（prepend time/CFG/mode tokens）替代adaLN-Zero可减少参数量。

涉及论文标题：
- Latent Wavelet Diffusion for Ultra-High-Resolution Image Synthesis
- ELF: Embedded Language Flows

---

## Latent Diffusion Model (LDM)（潜空间扩散模型）

术语是什么？通过联网搜索让回答具体和精准。
Latent Diffusion Model (LDM) 由Rombach et al. (2022)提出，是当前主流的扩散模型框架。核心思想：将扩散过程从高维pixel space迁移到低维learned latent space。由一个预训练的VAE (encoder E + decoder D)提供压缩——encoder将图像x∈R^{H×W×3}压缩到latent z∈R^{C×h×w}（典型f=8下采样，h=H/8, w=W/8），扩散模型在latent space中做denoising/flow matching，decoder将生成结果从latent恢复到pixel。比pixel-space diffusion大大降低计算开销，使高分辨率生成在单GPU上可行。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

LDM完整生成pipeline（以Flux text-to-image为例）：
```
// 离线阶段：VAE预训练
1: VAE.encoder: x ∈ R^{H×W×3} → z ∈ R^{C×h×w}     (f=8压缩)
2: VAE.decoder: z ∈ R^{C×h×w} → x ∈ R^{H×W×3}

// 在线生成
3: z_T ~ N(0, I)                                     // 在latent space初始化噪声
4: for t = T, ..., 1:                                // T个denoising steps
5:     v = DiT(z_t, t, text_embedding)               // transformer预测velocity/noise
6:     z_{t-1} = scheduler_step(z_t, v, t)           // ODE/SDE step
7: generated_image = VAE.decoder(z_0)                // latent→pixel
```

关键设计选择：
- **Compression factor f**：决定latent相对于pixel的缩小比。f=8（常用，如Flux-VAE、SD3-VAE 16ch）→latent token数=f²倍减少。更大f（如SD3-F16, f=16）进一步减少token但可能损失细节。
- **VAE quality对UHR生成至关重要**：LWD论文发现标准VAE在UHR下latent space包含cross-scale inconsistent high-frequency artifacts，通过scale-consistency fine-tuning压制伪影可显著提升后续wavelet masking的有效性。
- **LDM vs pixel-space diffusion**: LDM训练/推理快f²倍，但VAE compression可能丢失fine-grained semantic detail——这是LWD承认的limitation（future work建议探索higher-fidelity latent space或latent+pixel联合监督）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
主流实现：CompVis/latent-diffusion（GitHub）、HuggingFace Diffusers、Stable Diffusion系列。LDM的关键模块：1) VAE (encoder+decoder)，通常独立预训练后冻结，扩散模型仅操作在latent；2) U-Net或DiT backbone执行denoising/flow matching；3) text encoder（CLIP/T5）提供条件。训练通常在较低分辨率（256-512）pretrain→UHR fine-tune（LWD的stage 2）。LWD论文中VAE微调独立于扩散模型微调（stage 1 vs stage 2），保持模块化解耦。

涉及论文标题：
- Latent Wavelet Diffusion for Ultra-High-Resolution Image Synthesis

---

## Wavelet Energy Saliency Map（小波能量显著性图）

术语是什么？通过联网搜索让回答具体和精准。
Wavelet Energy Saliency Map是LWD论文提出的、基于Discrete Wavelet Transform (DWT)从latent representation中计算的空间显著性度量。给定一个latent tensor z∈R^{C×H×W}，应用单层DWT分解为四个子带z_LL（低频近似）、z_LH（水平细节）、z_HL（垂直细节）、z_HH（对角细节），计算每个空间位置(i,j)在三个高频子带上的channel-pooled能量：E(i,j) = (1/C) Σ_c[(z_LH^{c,i,j})² + (z_HL^{c,i,j})² + (z_HH^{c,i,j})²]，然后bilinear upsampling + per-sample min-max normalization得到Awavelet∈[0,1]^{H×W}。该图突出latent space中与high-frequency content（纹理、边缘、轮廓）关联的区域，作为频率感知的spatial saliency proxy。不同于基于learned attention的saliency（如DINO），wavelet saliency是deterministic、无训练、直接从信号属性导出的。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Wavelet Energy Saliency Map计算流程：
```
// 输入：latent tensor z_t ∈ R^{C×H×W}
1: z_LL, z_LH, z_HL, z_HH = DWT_2D(z_t, wavelet='haar')
   // 每个子带 ∈ R^{C×H/2×W/2}

2: // 计算高频能量图（Eq.3）
   E_hf = zeros(H/2, W/2)
   for i in 0..H/2-1, j in 0..W/2-1:
       for c in 0..C-1:
           E_hf[i,j] += z_LH[c,i,j]² + z_HL[c,i,j]² + z_HH[c,i,j]²
       E_hf[i,j] /= C                    // channel平均

3: // 上采样 + 归一化
   E_full = bilinear_upsample(E_hf, scale=2)   // 匹配原latent分辨率H×W
   A_wavelet = (E_full - min(E_full)) / (max(E_full) - min(E_full) + ε)
   // A_wavelet ∈ [0,1]^{H×W}
```

关键设计选择：
- **仅用HF子带（LH, HL, HH）**：LL子带编码coarse spatial content，包含局部复杂度的信息极少。类似Sobel/Laplacian边缘检测中gradient magnitude只反映high-frequency transitions。
- **Haar Wavelet选择**：最紧凑support（2 coefficients）→最精确的空间定位、最小cross-position interference。Daubechies wavelet (db2)的wider receptive field在mask边界产生"gray area"→dilute supervision；FFT High-Pass虽计算快但sacrifice spatial localization→Gibbs ringing artifacts污染mask边界（GLCM 0.71 vs Haar 0.74）。
- **无需训练**：wavelet saliency完全基于信号属性计算，不依赖learned attention，零额外参数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
pytorch-wavelets库（Cotter, 2019）提供PyTorch原生的2D DWT实现，支持Haar/Daubechies等多种wavelet basis。在LWD中使用：每个training step对当前latent z_t做DWT→计算HF energy→生成mask→modulate loss。关键前提：VAE latent space需预先通过scale-consistency loss净化——未经regularization的标准VAE latent中"high-frequency energy"多对应spurious artifacts而非真实结构，使wavelet saliency失效。LWD论文的VAE fine-tuning stage (stage 1)通过抑制cross-scale inconsistent高频伪影将latent spectral distribution对齐到clean RGB reference，使得wavelet energy map有意义。该技术也可泛化到其他需要信号驱动的spatial adaptive supervision的场景（如video generation中的temporal attention guidance、depth-aware synthesis）。

涉及论文标题：
- Latent Wavelet Diffusion for Ultra-High-Resolution Image Synthesis

---

## Time-Dependent Frequency-Aware Masking（时间依赖频率感知掩码）

术语是什么？通过联网搜索让回答具体和精准。
Time-Dependent Frequency-Aware Masking是LWD论文的核心训练机制，将wavelet-based spatial saliency转化为时变二元mask来调制扩散模型训练loss。具体地，给定wavelet saliency map Awavelet∈[0,1]^{H×W}和当前的扩散timestep t，对每个spatial位置(i,j)生成binary mask：M_t(i,j) = 1 if T·(Awavelet(i,j)+ℓ) ≥ t else 0，其中T为总timestep数，ℓ∈(0,1)为lower bound（论文设ℓ=0.3）。物理含义：高saliency区域（Awavelet大）的mask=1的timestep范围更广——这些区域在整个扩散过程中被监督更久；低saliency区域仅在t ≤ T·ℓ的早期timestep被监督（至少30%的基础监督）。最终loss：L_masked = ||M_t ⊙ [(ε-z_0) - v_Θ(z_t,t,y)]||²₂。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Time-Dependent Frequency-Aware Masking完整算法：
```
// 输入：latent z_0, noise ε, model v_Θ
// 参数：T（总timesteps），ℓ（lower bound=0.3）

// 扩散混合
1: t ~ Uniform(0, T]                          // 或从noise schedule采样
2: z_t = (1 - t/T) * z_0 + (t/T) * ε          // Flow matching插值

// Wavelet Saliency计算（见Wavelet Energy Saliency Map）
3: z_LL, z_LH, z_HL, z_HH = DWT_2D(z_t, 'haar')
4: A_wavelet = normalize(upsample(LH²+HL²+HH², channel_mean))

// Time-Dependent Mask生成（Eq.6）
5: M_t = zeros_like(z_t[:,0,:,:])              // binary mask
6: for each spatial position (i, j):
7:     if T * (A_wavelet[i,j] + ℓ) >= t:      // 当前timestep t - high saliency or early step
8:         M_t[i,j] = 1                       // 参与loss
9:     else:
10:        M_t[i,j] = 0                        // 跳过（不贡献梯度）

// Masked Loss（Eq.7）
11: target = ε - z_0
12: pred = v_Θ(z_t, t/T, text_conditioning)
13: diff = (target - pred)²                    // per-element squared error
14: loss = mean(M_t ⊙ diff)                    // 仅mask=1位置贡献loss
```

Lower bound ℓ的消融（Table 6, 论文Appendix A）：
```
ℓ=0.0: FID=34.15, GLCM=0.68  ← 平滑区域欠训练
ℓ=0.1: FID=33.21, GLCM=0.72
ℓ=0.3: FID=32.88, GLCM=0.74  ← 最优trade-off
ℓ=0.5: FID=33.46, GLCM=0.71
ℓ=0.7: FID=34.02, GLCM=0.69  ← 退化为接近uniform loss
```

策略的物理意义：
- **高Awavelet位置**（纹理/边缘）：T·(Awavelet+ℓ)大→覆盖更多timestep→被监督更久→detail refinement更充分
- **低Awavelet位置**（平滑区域）：仅在early timestep (t≤T·ℓ) 被监督→只保证基本结构建立
- **空间curriculum learning**：类似课程学习——先全部区域建立全局结构（early steps），后聚焦detail-rich regions精细雕刻（later steps）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该masking机制是纯training-time操作，推理期完全不需要——训练好的LWD模型与baseline model参数量相同、inference pipeline不变。训练期overhead：每step需做一次Haar DWT（通过pytorch-wavelets的CUDA kernel）生成mask，额外memory约3%（存储DWT中间tensor + mask，均为latent map量级，比diffusion backbone参数小3个数量级）。论文设计确保masking仅作用在objective level，不与任何特定模型架构耦合——可应用于任意使用flow-based或score-based trajectory的latent diffusion model。masking策略的通用性暗示可扩展到video generation (temporal attention)，depth-aware synthesis (depth-aligned masking)，multimodal conditioning等场景。

涉及论文标题：
- Latent Wavelet Diffusion for Ultra-High-Resolution Image Synthesis

---

## Scale-Consistency VAE Objective（尺度一致性VAE目标函数）

术语是什么？通过联网搜索让回答具体和精准。
Scale-Consistency VAE Loss是LWD论文stage 1中微调预训练VAE的目标函数，旨在提升latent space在高分辨率下的spectral fidelity和cross-scale coherence。传统VAE训练仅最小化reconstruction loss + KL divergence，对cross-scale frequency consistency无约束，导致UHR下latent representation包含尺度间不一致的高频伪影。Scale-consistency loss引入多尺度强约束：对原图x和其降采样版本x_down同时做encode-decode，要求两者reconstruction质量一致。完整loss：L_VAE = ||D(z)-x||²₂ + α||D(E(z_down))-x_down||²₂ + β D_KL(q(z|x)||p(z)) + λ L_LPIPS(D(z),x)，其中α=0.25, β=0.001, λ=0.05。该loss最早由Skorokhodov et al. (2025)和Kouzelis et al. (2025)为通用reconstruction提出，LWD将其识别为wavelet-guided UHR synthesis的关键前提。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Scale-Consistency VAE训练流程（以Flux-VAE fine-tuning为例）：
```
// 训练一个batch
1: x = batch_of_high_res_images        // 如1024×1024, 2048×2048
2: x_down = downsample(x, scale=2)     // 半分辨率版本（如512×512, 1024×1024）

3: z = E(x)                            // latent of full-res image
4: z_down = E(x_down)                  // latent of downsampled image

5: // 四项loss
6: L_recon = MSE(D(z), x)              // 标准reconstruction
7: L_scale  = MSE(D(downsample(z)), x_down)  // scale-consistency: 对降采样z做decode后与降采样x比较
   // 或等价：L_scale = MSE(D(E(x_down)), x_down)
8: L_kl = KL(q(z|x) || p(z))           // latent regularization
9: L_percep = LPIPS(D(z), x)           // perceptual loss (AlexNet features)

10: L_total = L_recon + 0.25*L_scale + 0.001*L_kl + 0.05*L_percep
11: optimizer.step(L_total)
```

Scale-consistency的核心机制：
- 对z_down（低分辨率latent）decode后与x_down（低分辨率原图）比较→强制encoder在不同分辨率下产生结构一致的latent
- 惩罚"跨尺度不一致的高频分量"→即那些在full-res中存在但在downsampled version中不应出现（或pattern不同）的高频信号
- 效果：将latent frequency spectrum对齐到clean natural image的DCT spectrum（Figure 3）→抑制spurious high-frequency noise

VAE reconstruction metrics（Table 3, 论文Appendix B）：
```
Flux-VAE:       rFID=0.73, LPIPS=0.07, PSNR=27.18, SSIM=0.89
Flux-VAE-SC:    rFID=0.50, LPIPS=0.06, PSNR=28.14, SSIM=0.90  (+SC improvement)

SD3-VAE-F16:    rFID=0.70, LPIPS=0.30, PSNR=19.82, SSIM=0.63
SD3-VAE-F16-SC: rFID=0.70, LPIPS=0.18, PSNR=22.58, SSIM=0.75  (LPIPS大降)
```

Scale-consistency对SD3-VAE-F16（aggressive f=16压缩）的LPIPS改进特别显著（0.30→0.18），说明高度压缩的VAE更容易产生cross-scale artifacts。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在LWD中，Scale-Consistency VAE fine-tuning是stage 1（独立于stage 2的扩散模型微调）。两个stage解耦是设计的核心：先净化latent space使其呈现良好频率特性→再用净化后的latent指导扩散模型训练（wavelet masking）。未经SC调优的VAE中high-frequency energy多对应spurious noise→wavelet mask会指向噪声而非真实结构→"frequency-guided supervision"失效。这就是LWD论文中"Synergy of Frequency Suppression and Utilization"分析的核心：VAE loss压制跨尺度不一致的高频伪影→剩余的高频能量与视觉salient features（edges/textures）相关性更强→增强wavelet attention机制的signal-to-noise ratio。

涉及论文标题：
- Latent Wavelet Diffusion for Ultra-High-Resolution Image Synthesis

---

## Elucidated Diffusion Models (EDM)（阐明扩散模型）

术语是什么？通过联网搜索让回答具体和精准。

Elucidated Diffusion Models (EDM) 是 Karras et al. (2022) 提出的扩散模型统一框架，将训练和采样从离散时间步重新参数化到连续噪声空间。与DDPM使用离散timestep t∈{1,...,T}不同，EDM使用噪声标准差σ作为连续参数来描述数据损坏程度：x_σ = x₀ + σ·ε, ε~N(0,I)。训练目标为L_EDM = E_{x₀,ε,σ}[λ(σ)·||ε − ε_θ(x_σ,σ)||²₂]，其中weighting function λ(σ)在噪声尺度间平衡贡献。EDM的关键洞察是将扩散模型视作学习scale-dependent denoisers的连续统(continuum)，而非离散的逐步去噪过程。EDM支持两类formulation：Variance-Preserving (VP) 和 Variance-Exploding (VE)，分别对应DDPM++和NCSN++架构变体。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

EDM训练与采样（以FFHQ 64×64 unconditional + VP-EDM为例）：

```
// 训练：连续噪声尺度采样
1: sample x_0 ~ p_data, ε ~ N(0,I)
2: sample σ ~ p_train(σ)           // log-normal分布采样噪声尺度
3: x_σ = x_0 + σ·ε                 // 连续加噪（无离散t）
4: loss = λ(σ)·||ε − ε_θ(x_σ, σ)||²₂  // σ为网络conditioning输入
5: θ ← θ − η·∇_θ loss

// 采样：连续确定性路径（DDIM-like）
1: x_N ~ N(0, σ_max²·I)            // 从最大噪声尺度开始
2: for i = N-1, ..., 0:            // N为离散化采样步数
3:     σ_i = schedule[i]           // 预定义噪声尺度序列
4:     denoised = x_{i+1} − σ_{i+1}·ε_θ(x_{i+1}, σ_{i+1})
5:     x_i = denoised + σ_i/σ_{i+1}·(x_{i+1} − denoised)  // 确定性更新
6: return x_0
```

EDM vs DDPM的关键差异：(1) DDPM使用离散timestep t + fixed noise schedule β_t, ᾱ_t = Π(1-β_s)，训练L = ||ε − ε_θ(x_t, t)||² (无λ加权)；(2) EDM使用连续σ + 灵活noise distribution p_train(σ) + 可调λ(σ)，λ(σ)在中等σ区域赋予更高权重（该区域对应最informative的denoising level）。EDM的reweighting λ(σ)使Spectral Regularization论文能在fine-tuning时选择"weighted"（λ=λ_EDM(σ)）或"unweighted"（λ=1），前者保留EDM的per-noise-level重要性分布，后者对所有噪声水平等同对待。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

NVIDIA官方EDM实现（https://github.com/NVlabs/edm）提供pretrained checkpoints for CIFAR-10/FFHQ/AFHQv2等数据集（VP和VE变体），预训练权重以.pkl格式分发。在Spectral Regularization论文中，EDM作为fine-tuning backbone：加载预训练checkpoint→仅5步fine-tuning with spectral auxiliary loss→采样与标准EDM完全相同。PyTorch实现仅需~50行代码的auxiliary loss计算，computational overhead negligible。

涉及论文标题：
- Spectral Regularization for Diffusion Models

---

## Fourier Spectral Regularization for Diffusion Training（扩散训练的傅里叶谱正则化）

术语是什么？通过联网搜索让回答具体和精准。

Fourier Spectral Regularization是一种loss-level训练正则化方法，在标准扩散模型denoising objective上附加可微分的Fourier域L1 penalty。与signal-domain L2 loss（如DDPM的MSE noise prediction loss）仅控制error total energy不同，Fourier spectral loss显式约束reconstruction error在frequency bands间的distribution。定义两种Fourier损失：(1) Fourier Amplitude Loss (LA_F)：LA_F = E[|| |F[x₀]| − |F[x̂₀]| ||₁]，仅匹配幅度谱(amplitude spectrum)，对spatial alignment不敏感，直接控制frequency-wise energy allocation mismatch；(2) Fourier Amplitude-and-Phase Loss (LAP_F)：LAP_F = E[||A₀−Â₀||₁·(1+||ϕ₀−ϕ̂₀||₁)]，将phase penalty通过amplitude magnitude加权，避免对low-amplitude band的insignificant phase noise过度penalize，同时稳定high-amplitude band的fine-scale structure。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Fourier Spectral Regularization的training step（以FFHQ + VP-EDM + LA_F为例）：

```
// 输入：x_0 (ground-truth batch), λ (regularization weight)
1: σ ~ p_train(σ), ε ~ N(0,I)
2: x_σ = x_0 + σ·ε                          // EDM forward corruption
3: ε_hat = ε_θ(x_σ, σ)                      // 网络预测噪声
4: x̂_0 = x_σ − σ·ε_hat                      // DDIM一步reconstruction得干净估计
5: 
6: // Standard EDM denoising loss
7: L_EDM = λ_EDM(σ)·||ε − ε_hat||²₂
8:
9: // Fourier Amplitude Loss
10: F_x0 = torch.fft.fft2(x_0)               // 2D FFT of ground-truth
11: F_x̂0 = torch.fft.fft2(x̂_0)              // 2D FFT of prediction
12: A_0 = torch.abs(F_x0)                      // amplitude spectrum
13: Â_0 = torch.abs(F_x̂0)
14: LA_F = ||A_0 − Â_0||₁                     // L1 amplitude discrepancy
15:
16: L_total = L_EDM + λ·LA_F
17: θ ← θ − η·∇_θ L_total
```

关键设计决策：(1) Spectral loss在**predicted clean sample x̂₀**上计算（DDIM一步reconstruction），而非直接在noisy input x_t上计算——确保spectral supervision与model generation pathway对齐。(2) 使用L1而非L2：有意break Parseval invariance。Parseval-Plancherel identity (||x||²₂ = ||X(ω)||²₂)仅对L2成立，对L1不成立——因此L1 amplitude loss可直接控制error的spectral distribution而非仅total energy。(3) Fourier amplitude loss对spatial translation invariant，amplitude spectrum captures frequency-wise energy分配与spatial shift无关，是天然的"global structural constraint"。(4) PyTorch torch.fft.fft2底层使用cuFFT，GPU计算高效，overhead negligible。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现仅需~20行PyTorch代码。在Spectral Regularization论文中：对pretrained EDM checkpoint做5-step lightweight fine-tuning, λ为唯一新增hyperparameter。Checkerboard toy experiment直观验证：baseline MSE model产生attenuated/broadened spectral responses + visible smoothing，spectral regularizer correctly concentrates energy near correct frequency bands。FFHQ/AFHQ上FID改善0.02-0.07（仅5步fine-tuning）。Audio (DiffWave on LJSpeech)上Fourier amplitude regularization yields strongest FAD improvement (1.994→1.462 at λ=10⁻⁴)。

涉及论文标题：
- Spectral Regularization for Diffusion Models

---

## Wavelet Coefficient Matching Loss for Diffusion Training（扩散训练的小波系数匹配损失）

术语是什么？通过联网搜索让回答具体和精准。

Wavelet Coefficient Matching Loss (LW) 是一种基于离散小波变换(DWT)的loss-level训练正则化方法，在predicted clean sample x̂₀和ground-truth x₀的所有尺度(scales)和方向(orientations)上计算小波系数的L1差异：LW = E[Σ_{s,ℓ} γ_{s,l}·||W₀^{(s,ℓ)} − Ŵ₀^{(s,ℓ)}||₁]，其中s索引尺度，ℓ索引方向/子带(LL/LH/HL/HH)，γ_{s,l}为各尺度/子带的权重。与Fourier spectral loss（基于全局正弦基函数，提供uniform frequency constraint）不同，wavelet coefficients提供localized、scale-aware的representation——每个coefficient同时编码spatial location和frequency content。这使wavelet loss能explicitly target localized oscillations、edges、textures和transient features。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Wavelet Coefficient Matching Loss计算流程（以FFHQ + VP-EDM + Haar wavelet loss为例）：

```
// 输入：x_0 (ground-truth batch), λ, wavelet='haar', level=J
1: σ ~ p_train(σ), ε ~ N(0,I)
2: x_σ = x_0 + σ·ε
3: x̂_0 = x_σ − σ·ε_θ(x_σ, σ)              // DDIM一步干净估计
4:
5: L_EDM = λ_EDM(σ)·||ε − ε_θ(x_σ,σ)||²₂   // 标准EDM loss
6:
7: // Wavelet Coefficient Matching Loss
8: coeffs_x0 = DWT_2D(x_0, wavelet='haar', level=J)
   // coeffs: [(LL_J, (LH_J, HL_J, HH_J)), ..., (LH_1, HL_1, HH_1)]
9: coeffs_x̂0 = DWT_2D(x̂_0, wavelet='haar', level=J)
10: L_W = 0
11: for each scale s, orientation ℓ:
12:     L_W += γ_{s,ℓ} · ||coeffs_x0[s][ℓ] − coeffs_x̂0[s][ℓ]||₁
13:
14: L_total = L_EDM + λ·L_W
15: θ ← θ − η·∇_θ L_total
```

两种小波基对比：(1) Haar wavelet（最简正交小波）：support长度=2（等价于local average vs difference），强调sharp discontinuities和edge-like features，对piecewise-constant structure敏感，但limited smoothness导致对smooth spectral behavior的approximation较粗略。(2) Biorthogonal 1.3 (bior1.3)：非对称双正交小波，analysis wavelet有1个vanishing moment、synthesis wavelet有3个vanishing moment，提供smoother multi-scale separation，high-frequency coefficients捕获larger spatial neighborhoods上的oscillatory behavior。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

PyWavelets (Lee et al., 2019, https://pywavelets.readthedocs.io) 提供Python原生DWT实现，支持Haar/Daubechies/Biorthogonal等wavelet families。在Spectral Regularization论文中，wavelet loss作为training-time auxiliary penalty：图像实验5步fine-tuning，音频(DiffWave on LJSpeech) 150K步fine-tuning。音频实验上Haar wavelet在较高λ下achieve lowest MR-STFT distance (improved multi-resolution temporal coherence)，bior1.3 shows increased sensitivity to λ due to redundant non-orthogonal structure。

与Wavelet Energy Saliency Map (LWD论文)的区别：Wavelet Energy Saliency Map用DWT从latent计算spatial energy map→生成binary mask→modulate diffusion loss，是"where to supervise"的spatial selection机制。Wavelet Coefficient Matching Loss直接比较wavelet coefficients作为loss term，是"what to supervise"的frequency-domain constraint，不涉及masking或spatial selection。

涉及论文标题：
- Spectral Regularization for Diffusion Models

---

## Diffusion Language Model (DLM)（扩散语言模型）

术语是什么？通过联网搜索让回答具体和精准。
Diffusion Language Model (DLM) 是将扩散模型（Diffusion Models）或流模型（Flow-based Models）应用于语言建模的一类生成模型，与自回归（AR）语言模型形成互补范式。DLMs通过迭代去噪（denoising）而非逐token自回归生成文本，支持并行生成、双向上下文和迭代优化。根据Denoising和离散化的空间不同，DLMs分为两大类：(1) Continuous DLMs：将离散token映射至连续表示（embedding/simplex）后执行去噪，如Diffusion-LM、CDCD、ELF；(2) Discrete DLMs：直接在离散token space定义扩散过程，使用masked/uniform transition matrices，如MDLM (absorbing-state masking)、Duo (uniform diffusion)。2024-2025年DLMs快速发展：LLaDA (8B)是首个与AR模型竞争力相当的离散扩散LLM，Dream 7B从AR权重初始化。DLMs的核心优势包括：并行生成（达~1000 tokens/sec）、双向上下文（更丰富的表示）、迭代优化能力、可控生成长度和格式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

DLM的通用算法pipeline（以Continuous DLM如ELF为例）：
```
// 训练阶段
1: tokens s = [s_1,...,s_L] ∈ V^L
2: x = Embed(s)                        // 离散token → 连续embedding
3: t ~ schedule, ε ~ N(0,I)
4: z_t = noisify(x, ε, t)              // 加噪（取决于DLM类型）
5: x̂ = model(z_t, t)                   // 去噪预测
6: loss = L(x̂, s)                      // 不同的DLM在此处分岔：
   // Continuous DLM: L = MSE(x̂, x) 在连续空间
   // 或 L = CrossEntropy(unembed(x̂), s) 每步离散化
   // Discrete DLM: L = -log p(s|z_t) via transition matrix

// 推理阶段
1: z_0 ~ P_noise                      // 初始噪声分布
2: for step = 0 to T-1:
3:     ŝ 或 x̂ = model(z_step, t_step) // 逐步去噪
4:     z_{step+1} = sampler_step(...)  // ODE/SDE/masking step
5: tokens = argmax(unembed(z_T))       // 最终离散化
```

DLM的分类体系（基于ELF论文Tab.2的survey）：
| 类别 | 状态空间 | 训练per-step离散化 | 推理per-step离散化 | 解码器 | 代表方法 |
|------|---------|-------------------|-------------------|--------|---------|
| Embedding-space Diffusion LMs | learn/fix emb | Yes | Yes | No | Diffusion-LM, CDCD, SeqDiffuSeq |
| Simplex Diffusion LMs | simplex | Yes | Yes | No | SSD-LM, TESS, TESS 2 |
| Latent Diffusion LMs | fix enc | No | No | Yes (单独) | LD4LG, PLANNER, Cosmos |
| Flow-based LMs | simplex/one-hot/emb | Yes (部分) | Yes (部分) | Yes/No | FLM, LangFlow, DFM |
| **ELF** | fix enc | **No** (仅最后步) | **No** (仅最后步) | **No** (共享权重) | ELF |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DLMs的实现涉及多个设计选择：(1) 状态空间：Continuous DLMs使用预训练编码器（如T5）或可学习embedding将token映射到连续空间；Discrete DLMs直接在vocabulary空间定义转移矩阵；(2) 噪声调度：Continuous DLMs通常使用Gaussian noise with logit-normal/log-normal schedule，Discrete DLMs使用masking probability schedule；(3) 离散化策略：Continuous DLMs需将连续状态映射回离散token——常用rounding（最近邻embedding）、unembedding layer（可学习投影矩阵）、或argmax over simplex；(4) 采样器：Continuous DLMs使用ODE/SDE solver，Discrete DLMs使用ancestral sampling/predict-and-noise等。ELF展示了Continuous DLMs的minimalist设计——仅在最末步离散化、无需单独decoder、充分发挥连续空间灵活性——取得了与Discrete DLMs竞争力相当甚至更优的质量。开源实现：ELF (https://github.com/lillian039/ELF)、MDLM (https://github.com/kuleshov-group/mdlm)、Duo (https://github.com/s-sahoo/duo)、E2D2 (https://github.com/kuleshov-group/e2d2)。

涉及论文标题：
- ELF: Embedded Language Flows

---

## Self-Conditioning（自条件化）

术语是什么？通过联网搜索让回答具体和精准。
Self-Conditioning是由Chen et al. (2023, "Analog Bits")提出的扩散模型条件化技术。核心思想：将模型在上一去噪步的输出（intermediate prediction x̂'或ŝ'）作为当前步的额外条件输入，使模型利用自身已有预测改进当前估计。在ELF中的具体形式：在Flow Matching的denoising branch训练时，50%概率先用当前状态z_t做一次forward pass得到intermediate prediction x̂'，然后将[z_t, x̂']（channel-wise concatenation）通过线性投影层映射回原始维度作为网络输入进行第二次forward pass；另50%概率使用all-zero作为self-conditioning（即无条件）。推理时，self-conditioning使用上一步的prediction x̂_{t+dt}，无需额外forward pass。Self-Conditioning已在多个DLMs中成为标准组件（SED、CDCD、LD4LG、TESS、TEncDM等）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

ELF中的Self-Conditioning训练与推理流程：
```
// 训练期（Alg.3）
1: z_t = t·x + (1-t)·ε              // 当前noisy state
2: z_no_sc = proj(concat([z_t, 0]))  // 无self-cond: concat zeros
3: x_no_sc = net(z_no_sc, t)         // 第一次forward pass
4: z_sc = proj(concat([z_t, stopgrad(x_no_sc)]))  // 有self-cond: concat前次预测
5: x_sc = net(z_sc, t)               // 第二次forward pass
6: mask ~ Bernoulli(0.5)            // 50%概率使用self-cond
7: x_pred = mask ? x_sc : x_no_sc
8: v_pred = (x_pred - z_t) / (1-t)
9: loss = MSE(v_pred, v_target)

// 推理期（Alg.5）
1: x_pred = 0                         // 初始化为零
2: for each time step t:
3:     z_sc = proj(concat([z_t, x_pred]))  // 用上一步预测
4:     x_pred = net(z_sc, t)           // 单次forward pass（无额外开销）
5:     v = (x_pred - z_t) / (1-t)
6:     z_{t+dt} = z_t + dt·v
```

关键设计选择：(1) Gradient stop：intermediate prediction x̂'的梯度被stop，避免通过self-conditioning形成循环梯度；(2) 投影层：channel维度从2d→d（d为embedding dim），使用线性层；(3) 训练期概率：50%是常用选择，平衡conditional和unconditional mode；(4) 推理零开销：推理时无需额外forward pass（self-cond来自上一步已有预测）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Self-Conditioning在ELF中有双重作用：(1) 直接改善去噪质量——模型在已有预测的引导下做出更consistent的估计；(2) 作为CFG的conditioning signal——ELF的training-time CFG使用self-conditioning prediction作为"条件"，无需外部class label或text prompt即可实现CFG。实现上，Self-Conditioning通过channel-wise concatenation + linear projection融入网络输入，与in-context conditioning（prepend control tokens）协同工作。该技术对训练overhead极小（仅增加一次额外的forward pass和stop-gradient操作，且仅50%概率触发），推理时完全零开销。在ELF的ablation中（Appendix），无self-conditioning的CFG会显著降低生成质量。

涉及论文标题：
- ELF: Embedded Language Flows

---

## Shared-Weight Denoiser-Decoder（共享权重去噪解码器）

术语是什么？通过联网搜索让回答具体和精准。
Shared-Weight Denoiser-Decoder是ELF提出的核心架构设计：单一网络（DiT backbone）在所有time steps t∈[0,1)作为denoiser（预测clean embeddings x̂，用MSE loss训练），在final step t=1作为decoder（预测clean embeddings并通过unembedding layer映射为token logits，用CE loss训练）。关键创新在于：通过共享网络权重、联合训练两个目标，消除了对单独训练decoder的需求。网络上通过binary "mode" token（"denoise" vs "decode"）区分两种操作模式。训练时80% steps分配为denoising mode、20%为decoding mode。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

ELF的Shared-Weight Denoiser-Decoder训练流程：
```
// 联合训练（单batch内混合两种mode）
1: x = T5_encode(tokens)                  // 获得clean embeddings
2: 
3: // === Denoising Branch (80% prob) ===
4: t ~ logit_normal(P_mean=-1.5, P_std=0.8)
5: ε ~ N(0, I)
6: z_t = t·x + (1-t)·ε                    // linear interpolation
7: x̂ = DiT(z_t, t, mode="denoise")       // 网络预测clean embeddings
8: L_MSE = ||(x̂ - x)/(1-t)||²             // x-prediction MSE loss
9:
10: // === Decoding Branch (20% prob, t=1) ===
11: p ~ logit_normal(P_mean=0.8, P_std=0.8) // per-token corruption level
12: ε ~ N(0, I), noise_scale = 5
13: z̃ = p·x + (1-p)·ε                      // 模拟不完美denoiser输出
14: h = DiT(z̃, t=1, mode="decode")         // 同一网络，decode mode
15: logits = W·h                            // unembedding: R^d → R^|V|
16: L_CE = CrossEntropy(logits, s)          // token-level cross-entropy

// 推理期
1: z_0 ~ N(0,I)
2: for t in [0, t_1, t_2, ..., t_{T-1}]:    // ODE/SDE denoising steps
3:     x̂ = DiT(z_t, t, mode="denoise")      // denoise mode
4:     v = (x̂ - z_t) / (1-t)
5:     z_{t+dt} = z_t + dt·v
6: // Final step (t=1):
7: h = DiT(z_T, t=1, mode="decode")         // decode mode
8: tokens = argmax(W·h)                      // 离散化输出
```

解码分支的per-token corruption设计（不同token有不同p值）使网络学会从受污染的embeddings中恢复——模拟推理时denoiser的imperfect outputs。Noise scale=5（OWT）使decode mode对residual errors更加鲁棒。

与baseline的对比：
- Latent Diffusion LMs (LD4LG等)：需要单独训练decoder（AR decoder / NAR decoder），增加参数量和训练阶段
- Per-step discretization DLMs (FLM, Diffusion-LM等)：每步做token prediction+CE loss，denoising trajectory受token-level constraint限制
- ELF：Shared-weight design + 仅在最后步CE loss = minimal treatment of discretization

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Shared-Weight Denoiser-Decoder的实现关键：(1) 网络架构：标准DiT (Diffusion Transformer) with SwiGLU + RMSNorm + RoPE + qk-norm；(2) Mode conditioning：4个可学习的mode tokens（denoise/decode）作为in-context conditioning prepend到输入序列；(3) Unembedding layer：可学习矩阵W ∈ R^{d_model × |V|}，仅在decode mode下使用，与网络联合训练；(4) Denoising mode probability=0.8提供最佳trade-off——过低（0.5）导致denoising训练不足，过高则decoding监督不足。Ablation显示Shared-Weight design比Two-Stage (separate encoder→decoder→denoiser) slightly better，且简化了pipeline（无需预训练decoder的额外stage）。该设计使ELF-B从148M参数（若用adaLN-Zero conditioning）降至105M参数（用in-context conditioning），同时保持competitive性能。

涉及论文标题：
- ELF: Embedded Language Flows
