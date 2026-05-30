## MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models

- 属于算法pipeline的实现是什么？实验比较什么？
  提出基于final image缓存+text-to-image CLIP similarity retrieval的动态去噪步数选择算法。核心设计：(1) Image Caching替代Latent Caching：缓存final generated image（PNG/JPEG压缩后1.4MB/张，对比Nirvana latent caching 2.5MB/张），存储image CLIP embedding仅0.29GB/100K images，cache检索耗时0.05s/100K images；(2) Text-to-Image similarity retrieval：用CLIP image encoder提取cached image embedding，与query text embedding做cosine similarity检索，比text-to-text similarity在CLIPScore (mean 0.28 vs 0.22)和PickScore (mean 20.33 vs 19.52)上均更优；(3) 动态k-selection heuristic：基于text-image similarity score和quality constraint (α≥0.95, 公式5)决定跳过去噪步数k∈{5,10,15,20,25,30}，固定T=50步，cache-hit仅需refine T-k步；(4) Noise re-introduction：对检索图像按扩散模型noise schedule加噪（公式2: Ĩ=σ_tk·ε+(1-σ_tk)·I*），使图像重新进入去噪流程；(5) Quality-constrained retrieval policy：仅当Q_cache-hit(k)≥α·Q_full-gen(α=0.95)时接受cache hit，heuristic在1000 prompt测试集上平均CLIPScore 28.50 vs full pipeline 28.59 (99.7% baseline quality)。实验比较throughput/quality trade-off，对比Vanilla (全量T=50步大模型)、Nirvana (latent caching text-to-text retrieval)、Pinecone (检索无refine)。

- 硬件平台是什么，配置是什么。
  4×NVIDIA A40 GPU (48GB) 单节点 + 16节点各4×AMD MI210 GPU (64GB)。PyTorch + PyTorch RPC。

- 模型是什么。数据集和bench分别是什么。
  Large models: Stable Diffusion-3.5-Large (SD3.5L, 8B params), FLUX.1-dev (12B params)。Small models: Stable Diffusion XL (SDXL, 3B params), SANA (1.6B params)。Distilled baseline: SD3.5L-Turbo (10步)。数据集：DiffusionDB (2M images production dataset)、MJHQ-30k (MidJourney images)。模型均使用T=50 denoising steps生成1024×1024图像，除SD3.5L-Turbo使用10步。SDXL用FP16，其余用BF16。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：https://github.com/stsxxx/MoDM。算法pipeline：
  1. Cache初始化：用大模型生成N张图像（如10000张），每张图像用CLIP image encoder提取embedding (512-dim)，存入cache。存储final image (1.4MB/张 PNG) + embedding (0.29GB/100K embeddings)。
  2. 推理时：新prompt→CLIP text encoder提取query embedding q→对cache中所有image embeddings e_I计算cosine sim (公式1)→取最高sim的cached image I*
  3. 若sim≥threshold: 按heuristic (Fig.5b)确定k（sim≥0.3→k=30; sim≥0.29→k=25; sim≥0.28→k=15; sim≥0.27→k=10; sim≥0.25→k=5）
  4. 用扩散模型noise schedule公式2对I*加噪到timestep t_k→生成Ĩ=σ_tk·ε+(1-σ_tk)·I*
  5. 将Ĩ送入小模型(SDXL/SANA)执行剩余T-k步去噪→输出final image
  6. 若sim<threshold: 发往大模型(SD3.5L/FLUX)执行全量T=50步推理
  7. 总compute savings公式4：C_total_saved = H_cache·ΣP(K=k)·[k/T·C_gen + (T-k)/T·(C_gen-C_small)]，同时跳过k步+用小模型进一步降低每步成本

## VDHA: Vector-Driven Hash Aggregation for Sparse Matrix-Sparse Vector Multiplication on GPUs

- 属于算法pipeline的实现是什么？实验比较什么？
  提出VDHA hash-based aggregation算法，加速weighted SpMSpV中write-back阶段的partial product accumulation。核心算法设计：(1) Vector-driven hash aggregation：将SpMSpV write-back阶段从全局atomic scatter或global sort-reduce改为在shared-memory hash table中做local aggregation。每个CTA维护private hash table（2048-entry），partial products (row_idx, val)通过modulo hash定位→atomicCAS+linear probing插入→命中则accumulate value、未命中则插入新entry→hash table满时flush到global memory。hash table不仅消除intra-block write conflicts，还通过bucket order flush提供partial ordering改善memory coalescing。(2) Column decomposition with reordering：利用real-world graph的skewed列长分布（少数long column占大多数NNZ），将long column按SPLIT_SIZE=256切分为segment→segment metadata按首row index排序（O(S log S)而非O(N log N)）→提升跨列segment overlap，将local overlap ratio ρ从51.0%提升至89.8%（T=2048, density=100%），coalescing factor γ从0.744提升至2.607。(3) Fetch-compute-writeback重叠：利用write-back阶段的高memory stall（>45% long scoreboard stalls），通过double buffering将hash computation叠加到asynchronous memory fetch上，hash computation cost从16.7%降至12.3%。算法保证weighted SpMSpV正确性（支持任意权重），通过FALLBACK_ITER机制在hash probing失败时fallback到global atomic保证结果正确。实验比较：Konect/LAW (>100 web graphs，≥5M NNZ) + SuiteSparse (>200 scientific matrices，≥5M NNZ)，4个vector sparsity levels，对比7个baseline，geomean speedup 1.41× on web graphs、1.13× on SuiteSparse。还提出轻量级predictive model（decision tree，5 features：num_rows, num_nnzs, bandwidth index B, variance index V, vector sparsity）预测VDHA是否优于baseline，91.3% accuracy。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 GPU (40GB HBM2e, peak bandwidth 1555 GB/s, SM80 with 168KB shared memory/SM)，AMD EPYC 7742 CPU (64-core Zen 2)。CUDA nvcc 12.5，-O3。

- 模型是什么。数据集和bench分别是什么。
  无ML模型——VDHA是通用SpMSpV kernel算法。数据集：(1) Konect/LAW web-scale graphs (>100 matrices, ≥5M NNZ)，包含social networks、web graphs如it-2004 (41.2M rows/1.15B NNZ)、sk-2005 (50.6M rows/1.95B NNZ)等。(2) SuiteSparse Matrix Collection (>200 matrices, ≥5M NNZ)，覆盖科学计算、工程、优化等domain，如inline_1 (GHS_psdef)、delaunay_n24 (DIMACS10)、roadNet-CA (SNAP)、atmosmodl (Bourchtein)、G3_circuit (AMD)等。Input vector随机生成，sparsity levels: 0.01/0.05/0.10/0.20。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：论文未明确提供开源链接（PPoPP'26会议论文，可能pending release）。算法pipeline（以it-2004 web graph在sparsity=0.1为例）：
  1. Vector Processing：输入sparse vector x→扫描nonzero entries识别active columns→按LEN_THRES=128分类short/long columns→long columns (如1.4%的columns含>70% NNZ) 按SPLIT_SIZE=256切分为segments→segment metadata按首非零row index排序（10个segments的排序成本远低于对10M nonzeros排序）
  2. Block-level Hash Aggregation：segments block-mapped到CTA→每个CTA维护2048-entry shared-memory hash table→每个thread读取segment中(row_idx, mat_val)→compute partial product val=mat_val×vec_val→hash=(row_idx%2048)→atomicCAS抢占slot→命中则atomicAdd val到hash value→未命中则linear probing→超过FALLBACK_ITER则fallback到global atomicAdd
  3. Flush：hash table接近容量→按bucket order (0→2047) flush entries到global output vector y，flush时entries以hash order排列提供partial ordering改善memory coalescing
  4. Pipeline：while flush当前segment→cp.async异步prefetch下一segment→hash aggregation与memory access重叠
  5. 与传统atomic write-back对比：传统方案每个partial product都global atomicAdd→severe address contention (many-to-one scatter)→仅~270 GB/s bandwidth on A100 (17% peak)。VDHA通过local hash aggregation减少global atomic次数→improved bandwidth utilization

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

## ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression

- 属于算法pipeline的实现是什么？实验比较什么？
  提出TCA-TBE（Tensor-Core-Aware Triple Bitmap Encoding），一种基于BF16权重指数字段冗余的固定长度无损压缩算法。核心设计：(1) 离线分析每层权重矩阵的exponent分布，识别top-7连续exponent值（覆盖>95%权重），编码为3-bit codeword (001-111)，超出top-7的outlier用codeword 000标记并全精度存储；(2) 将每个8×8 weight tile编码为三个独立64-bit bitmap（每个表示codeword的一个bit-plane）、一个PackedSignMantissa buffer（8-bit sign+mantissa）和一个FullValue buffer（完整BF16 fallback值）；(3) 平均每元素11.3 bits，压缩率约1.41×（16/11.3），接近理论下界10.6 bits。实验比较kernel-level speedup、end-to-end latency和throughput，对比cuBLAS_TC、DietGPU (rANS)、nvCOMP (rANS)、DFloat11 (Huffman)和vLLM/Transformers等baseline。算法保证bit-exact无损——解压缩后权重与原始BF16完全一致，无精度损失。

- 硬件平台是什么，配置是什么。
  三平台：(1) 4× NVIDIA RTX4090 (Ada Lovelace, 24GB, Compute Capability 8.9) + Intel Xeon Platinum 8352V (144 cores, 512GB DDR4)；(2) 4× NVIDIA L40S (Ada Lovelace, 48GB) + Intel Xeon Gold 6230R (104 cores, 512GB DDR4)；(3) NVIDIA RTX5090 (Blackwell, 32GB, CC 12.0)。编译：GCC 11.3 + NVCC 12.4 (RTX5090用NVCC 12.8)。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA3.1 (8B/70B/405B)、Qwen2.5 (7B/14B/32B/72B)、Gemma3 (12B/27B)、Mistral (24B/123B)。kernel benchmark使用这些模型真实weight矩阵的linear layer shapes（QKV_proj, O_proj, GateUp_proj, Down_proj, LM head），batch size 8/16/32。end-to-end evaluation使用LLaMA3.1-8B (RTX4090)、Mistral-24B (2×L40S)、LLaMA3.1-70B (4×L40S TP)，batch size 8/32，output length 128-2048 tokens。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  已开源：https://github.com/HPMLL/ZipServ_ASPLOS26.git。TCA-TBE算法pipeline：
  1. 离线压缩：对每个weight matrix，扫描exponent直方图→选择top-7连续exponent值→逐8×8 tile编码为triple bitmap + PackedSignMantissa + FullValue buffer。压缩LLaMA3.1-8B约2.5分钟（16核CPU）。
  2. 在线推理：解压时每个线程独立判断其负责元素的storage mode（通过bitwise OR三个bitmap得到64-bit spatial indicator mask），若indicator=1则从PackedSignMantissa读取sign+mantissa并通过base_exp+codeword算术恢复exponent重建BF16；若indicator=0则直接从FullValue读取完整BF16。codeword lookup通过base_exp + (bit2<<2 | bit1<<1 | bit0)算术实现，无需shared memory table lookup。

## DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU

- 属于算法pipeline的实现是什么？实验比较什么？
  提出ADAPT（Adaptive Dynamic Allocation for Parallel Tree），一种budget-constrained integer programming方法，实现hardware-aware dynamic draft generation。核心设计：(1) 问题形式化：给定computational budget B和draft model在各位置的概率分布，目标为最大化expected number of successfully verified tokens。决策变量x_{i,j,l}∈{0,1}表示是否在第i层第j个node选择第l个token做branching，目标函数max Σ_{i=1}^{D} Σ_{j=1}^{N_i} Σ_{l=1}^{V} p_{i,j,l}·x_{i,j,l}；(2) 三重约束：computational budget constraint（总branch数≤B）、structural constraint（每层branch数≤k_max，k_max设为FPGA parallel support number的整数倍如8/16/32）、pipeline depth constraint（D ≥ D_min = ⌈T_verify / T_draft⌉，确保heterogeneous pipeline的compute overlap最大化）；(3) Temperature-Controlled Probabilistic Sampling Greedy Approximation：NP-hard问题的实时求解——定义Path Cumulative Probability P_cum(i,j,l) = p_{i,j,l}·∏_{k} p_{k,par(a_k),a_k}，用softmax temperature T归一化后通过Gumbel sampling做非重复选择，T→0退化为deterministic top-k selection，T较大趋向uniform exploration；(4) 算法复杂度：O(D·k_max·log k_max) time，O(D·k_max) space。实验比较：(1) acceptance rate稳定在75%-85%（Qwen3-0.6B/8B pair）；(2) 动态draft length分布呈long-tail特征；(3) hyperparameter sensitivity：confidence threshold ε和temperature T对acceptance rate的影响（ε≤0.6时acceptance rate>75%，T≤1时>80%）；(4) HW-Branch ablation贡献（2.21× speedup from baseline）。

- 硬件平台是什么，配置是什么。
  主平台：Intel Xeon 4310 + RTX 4090 (2230MHz, 330 FP16 TOPS, 24GB) + V80 FPGA (300MHz, 10848 DSPs, HBM, 64GB)。附加：A100 + U200/V80 FPGA。算法参数：k_max设为FPGA并行支持数的整数倍（8/16/32），D_min由T_verify/T_draft ratio确定。

- 模型是什么。数据集和bench分别是什么。
  目标模型：Vicuna-7B-v1.3 (4096 hidden, 11008 FFN, 32 layers)、LLaMA-2-7B (4096 hidden, 11008 FFN, 32 layers)、OPT-13B (5120 hidden, 20480 FFN, 40 layers)、Qwen3-8B (4096 hidden, 12288 FFN, 36 layers)。Draft模型：Vicuna-160M (768 hidden, 3072 FFN, 12 layers)、LLaMA-160M (768 hidden, 3072 FFN, 12 layers)、OPT-125M (768 hidden, 3072 FFN, 12 layers)、Qwen3-0.6B (1024 hidden, 3072 FFN, 28 layers)。数据集（Spec-Bench）：MT-Bench (multi-turn对话)、Translation (WMT)、Summarization (CNN/DailyMail, XSum)、Question Answering (SQuAD, Natural Questions)、Math Reasoning (GSM8K, MATH)、Retrieval-Augmented Generation (RAG)。本文Fig.12/13使用Qwen3-0.6B/Qwen3-8B pair。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  已开源：https://github.com/ShaoqiangLu/DFVG（MIT License）。ADAPT算法pipeline：
  1. 每轮decoding iteration：FPGA draft model对当前prefix做forward pass，输出vocabulary-level probability distribution
  2. Budget allocation：根据B（FPGA hardware parallelism limit确定的max branch数）和D_min（T_verify/T_draft ratio）确定本iteration的depth和branching budget
  3. Tree construction：从root node起逐层执行temperature-controlled Gumbel sampling——每层计算各候选token的Path Cumulative Probability，用softmax+温度T归一化→Gumbel noise perturbation→argmax选择k_i个非重复候选→下一层将各selected token作为新node继续扩展
  4. 生成的token tree通过PCIe传输到GPU，TreeSort-Verify做block-parallel attention verification
  5. GPU返回accepted tokens→FPGA根据返回的accepted prefix length检测rollback（对比local sequence），若rollback则reset KV cache并从verified prefix恢复
  例如Qwen3-0.6B/8B pair下，acceptance rate 75%-85%，draft length呈long-tail分布（多数iteration只需短draft，少数需长draft），动态分配避免了static draft length的"too short→frequent rollback / too long→computational waste"困境。k_max设为FPGA PE array的整数倍（如16），确保每次生成的branch数对齐硬件并行粒度。

## BAT: Efficient Generative Recommender Serving with Bipartite Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  提出Bipartite Attention，一种用于生成式推荐(GR)的新注意力机制。核心设计：(1) 利用user和item语义在推荐prompt中是排列不变(permutation-invariant)的关键洞察——交换user和item token的顺序不影响上下文语义；(2) 基于此提出Item-as-prefix attention作为传统User-as-prefix attention的替代方案——输入序列组织为[I1,...,I_N, U, Instr]，item的KV cache可跨用户共享；(3) 调整attention mask使item间无cross-attention，并调整position encoding使所有item共享相同起始位置ID（在User-as-prefix中起始于user token长度，在Item-as-prefix中重置为0），保证每个item的token独立于其他item和后续user/instruction token，从而item KV cache可独立预计算和存储。实验比较：在三个Amazon数据集(Games, Beauty, Books)和一个工业合成数据集(Industry)上对比UP(User-as-prefix)和IP(Item-as-prefix)两种attention策略的Recall@k、MRR@k、NDCG@k指标（k∈[5,10]），结果表明IP在多数情况下保持与UP相当或更好的推荐质量。

- 硬件平台是什么，配置是什么。
  主测试平台：4节点集群（浙江大学），每节点Intel Xeon Silver 4214 CPU (2×24 threads @2.20GHz)、200GB内存、1×A100-40GB GPU (PCIe 3.0x16)、100Gbps网络。生产测试平台：16节点集群，每节点1×NVIDIA H20 GPU、Intel Xeon Platinum 8469C CPU (2 socket×48 cores×2 threads)、500GB host memory、200Gbps网络。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen2-1.5B (L=28, H=2, D=128, KV cache/token=28672 Bytes)、Qwen2-7B (L=28, H=4, D=128, KV cache/token=57344 Bytes)、Llama3-1B (L=16, H=8, D=64, KV cache/token=32768 Bytes)，均使用FP16。数据集：Amazon公开推荐数据集Games (15K users, 8K items)、Beauty (22K users, 12K items)、Books (510K users, 280K items)，以及从真实电商广告workload生成的工业合成数据集Industry (10M users, 1M items)。评价指标：Recall@5/10、MRR@5/10、NDCG@5/10；系统指标：QPS、cache hit rate、P99 latency。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  论文未明确说明代码开源地址。Bipartite Attention算法pipeline：
  1. 离线阶段：预计算所有item的KV cache并加载到各KV cache worker的memory pool中。
  2. 在线推理：
     - User-as-prefix attention：输入=[U, I1,...,I_N, Instr]，user token U的KV cache预计算并缓存，实时仅计算item和instruction token。Attention: Attn(q_{I,Instr}, k_{I,Instr}∪k_U, v_{I,Instr}∪v_U)
     - Item-as-prefix attention：输入=[I1,...,I_N, U, Instr]，item token I的KV cache预计算并缓存，实时仅计算user和instruction token。Attention: Attn(q_{U,Instr}, k_{U,Instr}∪k_I, v_{U,Instr}∪v_I)
  3. 判别token为序列最后一个token，其hidden state投影到vocabulary logits，通过softmax计算每个candidate item的relevance score，输出top-k ranked list。
  4. Item-as-prefix三大优势：(a) item cache可跨成千上万用户共享；(b) 仅需本地内存存储所有item KV cache（如287GB for 1M items vs 430TB for 10M users）；(c) 对不活跃用户节省更多computation（>55%用户每小时仅访问一次，user cache面临compulsory miss）。

## JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  提出RtSmooth，一种面向2-bit KV cache量化的运行时per-token smoothing算法。核心设计：(1) K cache使用per-token smoothing transformation + per-channel group quantization，V cache使用per-token group quantization；(2) 运行时动态计算每个token的smoothing factor = max(|K_i|)^0.5，缩小group内值域范围，降低quantization error上界（ε_smooth(gp) ≤ s_smooth(gp)/2 < s_gp/2）；(3) FAVP (Fast Absmax Value Positioning)离线校准识别每层最可能持有absmax的稀疏channel集（超过90%层涉及<2% channel），运行时仅扫描这些channel计算smoothing factor；(4) decoding期间新生成KV token先保留FP16，每g=32步量化缓冲token。实验比较accuracy (perplexity on WikiText2/C4, LongBench 8 tasks)和efficiency (kernel runtime, serving latency, end-to-end GPU time, KV cache memory usage)。对比baselines包括FP16 FA2、RTN (Round-To-Nearest per-token group quantization)、Atom (2-bit/4-bit)、QServe (2-bit/4-bit)、SKVQ、KVQuant、KIVI (2-bit)、DuoAttention (KV selection)。

- 硬件平台是什么，配置是什么。
  单机4×NVIDIA A100-PCIE-40GB GPU，效率实验（端到端+kernel-level）均使用单张A100。软件栈：PyTorch 2.4.0 + CUDA 12.6。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-30B、Llama2-7B/13B、Llama3-8B、Mistral-7B、Vicuna-7B/13B、Qwen-2.5-32B。Llama-30B/Llama2/Vicuna使用torch.float16，Llama3/Mistral/Qwen使用torch.bfloat16。数据集：WikiText2（perplexity，sequence length 2048）、C4、LongBench（8个long-context multi-task：LCC, TriviaQA, RepoBench-P, QMSum, SAMSum, MultiNews, Qasper, TREC）。校准使用WikiText2 training set中128个8K-length样本。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  论文承诺artifact will be released as open-source repository，但截至分析无法确认公开官方代码仓库。算法pipeline：
  1. 离线校准阶段：用FAVP在128个WikiText2 8K样本上为每层记录最可能持有absmax的稀疏channel集（<2% of total channels），校准仅需数分钟。
  2. Prefill阶段：处理输入prompt生成KV cache，使用full-precision KV token参与attention保证accuracy。
  3. Decoding阶段：新生成token t的K_t、V_t先写入预分配ring buffer保留FP16精度→当buffer中旧segment达到g=32个token时，对K cache执行RtSmooth量化（FAVP快速定位absmax channels→计算smoothing factor max(|K_i|)^0.5→per-token smoothing→per-channel group quantization打包为INT2），对V cache执行per-token group quantization→量化后segment追加到低精度KV cache。
  4. Attention计算：mixed-precision kernel同时处理2-bit quantized KV和FP16 recent KV，在同一kernel内完成INT2-to-FP16 unpacking + dequantization + attention。
  5. Ring buffer容量为用户可配的g整数倍，默认2g；容量为n*g时，每次decoding至少保留(n-1)*g个最近FP16 token。

## High-Throughput Non-Uniformly Quantized 3-bit LLM Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  提出Quantix，一个面向non-uniform 3-bit weight-only quantized LLM inference的GPU执行框架。Quantix本身不是新量化算法，而是将已有non-uniform量化方案（SqueezeLLM、Any-Precision、Bitsandbytes等）的压缩权重高效映射到GPU的执行层优化。核心设计：(1) Hardware-aligned bit shuffling：离线将3-bit weight index拆分为1-bit和2-bit两个矩阵（bit dividing），再按Tensor Core tile访问模式重排为连续segments（bit mapping），使内存访问coalesced且无需padding/spanning；(2) Fused dequantization-matmul kernel：在CUDA cores上完成in-register dequantization（bit concatenation重建3-bit index→centroid lookup生成FP16 weight），直接喂给Tensor Cores做matrix-multiplication；(3) Hierarchical software pipeline：inter-tile层用shared memory double buffering重叠global memory prefetch与计算，intra-tile层用register double buffering重叠dequantization与MMA；(4) Split-K parallelization + 128-bit vectorized memory access。实验比较kernel-level speedup（vs FP16 cuBLAS、SqueezeLLM、Any-Precision LLM、GPTQ）和end-to-end throughput（vs SqueezeLLM、FP16、GPTQ、Marlin），并做ablation study评估各优化组件贡献。

- 硬件平台是什么，配置是什么。
  主要平台：NVIDIA L40 GPU（面向LLM inference优化），NVIDIA A100 GPU。kernel benchmark主要在L40上进行。hardware utilization分析使用NVIDIA Nsight。端到端实验在单张A100和双L40 GPU上测试。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA家族（LLaMA-13B/33B/65B, LLaMA2-7B/13B）、OPT家族（OPT-30B/66B/175B）、Vicuna-13B。kernel benchmark从LLaMA和OPT linear layers提取真实weight矩阵shapes，batch size 1-512。端到端实验Vicuna-13B、OPT-30B、LLaMA-65B，单A100和双L40，prompt length固定128 tokens，output length 128-1024 tokens。accuracy评估使用WikiText-2 perplexity和5-shot MMLU（lm-eval harness），LLaMA2-7B和LLaMA2-13B。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：https://github.com/yuang-chen/Quantix-PPoPP26。Quantix算法pipeline：
  1. 离线阶段：(a) 使用SqueezeLLM/Any-Precision/Bitsandbytes等non-uniform quantization对原始FP16 weight做K-means clustering，得到3-bit index矩阵Wq和每行centroids C；(b) Quantix bit dividing将每个3-bit index拆成1-bit matrix Wq,1和2-bit matrix Wq,2，使32个1-bit元素恰好填32-bit word，32个2-bit元素恰好填64-bit word；(c) Quantix bit mapping按64×64 warp tile→16×16 Tensor Core tile的层次，将每个thread负责的元素收集为连续linear segments W1'和W2'，使在线kernel可用128-bit cp.async指令一次抓取。
  2. 在线推理：(a) Fused kernel初始化时prefetch初始warp tiles到shared memory；(b) 主循环中inter-tile层用cp.async异步预取下一K-tile的W1'/W2'/A到shared memory；(c) intra-tile层从shared memory load subtile到registers→CUDA cores做in-register dequantization：bit concatenation [1-bit]+[2-bit]→3-bit index→shift+mask提取→centroid lookup得到FP16 W†→Tensor Cores执行MMA (A×W†)；(d) 两层double buffering使prefetch/dequant/matmul三级重叠；(e) Split-K将K维度切分并行计算partial sums，最后reduction合并。
  3. 例如LLaMA-65B在A100上用3-bit Quantix，batch size 16、token length 128时相对SqueezeLLM最高11.46× speedup；3-bit Quantix让LLaMA-65B可在单GPU运行（FP16无法）。

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

## RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  提出RTMPQ（Rotated Token-wise Mixed Precision Quantization）算法，通过Hadamard旋转抑制channel-wise outlier后迁移至token维度，再用token-wise mixed precision量化处理双维度outlier。核心设计：(1) Hadamard Rotation：利用Hadamard矩阵（元素+1/-1的正交矩阵）乘activation矩阵，平滑channel维度的极端值，将irregularity从channel维度迁移至token维度。乘法可通过Fast Walsh-Hadamard Transform (FWT)在O(mn log n)复杂度内高效实现。(2) Token-wise Mixed Precision：旋转后outlier呈纯token-wise分布，通过per-token maximum activation value做top-k选择（5% outlier比例），outlier token用INT8量化、其余token用INT4量化，outlier set对称扩展到weight矩阵。(3) 四种精度组合的cross-precision乘法（W4A4/W4A8/W8A4/W8A8），使用INT32累加器防止overflow。实验比较perplexity（WikiText2）和zero-shot accuracy（ARC-C/E, HellaSwag, LAMBADA, PIQA, WinoGrande六个下游任务），对比BF16、INT4 uniform、MixQ、Atom、QuaRot等baseline。Qwen3-8B上RoMeo perplexity 10.97优于QuaRot 11.53、MixQ 14.76，Qwen3-8B zero-shot average 64.41 vs BF16 70.42 vs QuaRot 63.32。

- 硬件平台是什么，配置是什么。
  NVIDIA GeForce RTX 4090 GPU (24GB memory, Ada Lovelace, peak INT4 performance 8× over FP16)。软件: Python 3.12, PyTorch 2.8.0, CUDA 12.8, HadaCore (Hadamard变换), CUTLASS, Triton。多卡serving时Qwen3-14B用2×RTX 4090 TP，Qwen3-32B用4×RTX 4090 TP。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen3 (8B/14B/32B)、Llama-3.1 (8B/70B)。数据集：WikiText2（perplexity评估），六个zero-shot下游任务：ARC-Challenge/ARC-Easy, HellaSwag, LAMBADA, PIQA, WinoGrande（通过lm-eval library评估）。Perplexity用batch size 2、sequence length 2048；下游任务batch size 32。Outlier比例固定为5%（activation和weight各5%）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：https://github.com/thu-pacman/RoMeo。RTMPQ算法pipeline：
  1. 离线阶段：对weight矩阵左乘H^T（Hadamard转置）完成离线旋转→识别weight outlier（per-column max value top-k）→量化weight为INT4+INT8 mixed precision
  2. 在线推理（以Qwen3-8B单token前向为例）：
     a. Activation Hadamard Rotation：输入activation X (FP16, shape [batch, hidden])右乘Hadamard矩阵H→X'=XH，使用FWT高效实现，channel-wise outliers被平滑
     b. Token-wise Outlier Detection：计算X'的per-token max value→top-k选择5% outlier tokens→存入outlier set O_A
     c. Mixed Precision Quantization：outlier tokens→INT8量化（含scaling factor）；normal tokens→INT4量化
     d. Cross-Precision GEMM：四种精度组合（W4A4/W4A8/W8A4/W8A8）的矩阵乘法→INT32 accumulator→dequantize with per-token scaling factors
     e. Post-mul Overwrite：高精度outlier计算结果overwrite对应位置输出
  3. 理论加速比公式：S = 1 / (P_INT4/4 + (1-P_INT4)/2)，其中P_INT4为纯INT4计算比例。当m=n=4096, k_a=k_w=256时，P_INT4=88%，理论加速比3.57×

## GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  提出GyRot量化框架，通过CoRFiG（Coarse Rotation, Fine Grouping）+ HAP（Harmonic-Aligned Permutation）+ 重公式化非对称量化+ceiling-based零點舍入，实现rotation与fine-grained group quantization的协同配合。核心算法设计：(1) CoRFiG：将rotation限制在coarse scope R=1024内局部执行，而非全局rotation，同时保持fine group size G=32，满足R=2^g·G关系。这样rotation在R范围内flatten分布、amortize outliers，同时group quantization保留local adaptation能力，解耦rotation scope与group granularity。(2) HAP：利用Hadamard矩阵的harmonic行结构（递归构造产生长度2^k的全+1或全-1向量），将全局选出的高magnitude outlier channel permute到harmonic rows上，使每个group内outlier乘以一致的符号（全+1或全-1），从而tighten per-group range并降低scale/zero-point精度需求。(3) 重公式化非对称量化：将传统先scale后bias的公式x̂=⌊x/s_x+z_x⌉改为先bias后scale的x̂=⌊(x+z_x)/s_x⌉，zero-point直接从unscaled domain计算z_x=−min(x_g)，避免小scale因子放大zero-point分布尾部。(4) Ceiling-based ZP rounding：用⌈·⌉替代传统round，保证z_Q≥z，消除zero-point量化误差导致的underflow clipping。最终实现INT8 scale factor和INT8 zero-point的fully integer dequantization，在W4A4KV4配置下保持competitive accuracy。实验比较：对比Tender (W8A8/W4A4)、MANT (W4A8/W4A4)、Quarot、SpinQuant、DuQuant、LightRot在WikiText-2 PPL、6个zero-shot task（PIQA/ARC-e/ARC-c/BoolQ/HellaSwag/WinoGrande）、MT-Bench conversational benchmark上的表现。GyRot-INT在W3A4极低bit配置下仍保持competitive PPL。

- 硬件平台是什么，配置是什么。
  算法accuracy评估在NVIDIA GPU上运行（论文未详述具体GPU型号，因accuracy evaluation不依赖特定硬件）。硬件实现评估：GyRot accelerator RTL以SystemVerilog实现，Samsung 28nm工艺，Synopsys Design Compiler综合，1GHz目标频率，片上SRAM由commercial memory compiler生成。对比baseline（Tender, MANT, LightRot）在iso-compute-area约束下综合评估。DRAM功耗使用Micron DRAM Power Calculator (DDR4 model)。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-1 (7B/13B)、LLaMA-2 (7B/13B)、LLaMA-3 (8B, 8B-Instruct)。数据集：WikiText-2（perplexity评估）、PIQA、ARC-easy、ARC-challenge、BoolQ、HellaSwag、WinoGrande（zero-shot task评估，通过LM-Evaluation-Harness框架）、MT-Bench（LLM-as-a-Judge对话质量评估，160 turns）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  论文未明确说明开源。GyRot算法pipeline：
  1. 离线阶段：
     a. 对每层weight矩阵应用CoRFiG：在R=1024 scope内执行Hadamard rotation（R=2^g·G, g=5, G=32）→局部flatten weight distribution
     b. HAP permute：识别全局high-magnitude outlier channels→permute到Hadamard矩阵的harmonic rows（长度2^k的全+1/全-1 vectors）→permutation可fuse进weight矩阵（permutation-invariant性质），无runtime overhead
     c. 对activation应用重公式化非对称量化：z_x=⌈−min(x_g)⌉（ceiling rounding避免underflow），s_x=(max(x_g)+z_x)/(2^b−1)，量化为INT4
     d. Weight对称量化（GPTQ）+ rotation后INT4量化，scale factor SW量化为INT8
  2. 在线推理（以LLaMA-3-8B W4A4KV4为例）：
     a. Activation经FVU做online Hadamard rotation（非线性层如embedding/SwiGLU后需要online rotation，因rotation-invariance不跨非线性层成立）
     b. INT4 activation × INT4 weight → 32-way dot product in PE → 13-bit partial sum
     c. Fully integer dequantization：先乘activation scale SX (INT8)→加zero-point项 ZX×WSUM（WSUM预计算）→乘weight scale SW (INT8)→32-bit integer accumulation
     d. 最终output转FP16写回buffer
  3. CoRFiG+HAP使INT8 SF/ZP即可保持accuracy，消除FP dequantization硬件开销

## BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

- 属于算法pipeline的实现是什么？实验比较什么？
  提出利用Tensor Cores加速低比特KV Cache解码的量化推理pipeline。核心算法设计：(1) 低比特KV Cache量化（4-bit/2-bit Key，8-bit Value），支持channel-wise和tensor-wise两种scaling粒度，适配多种量化算法（KIVI、Gear、KVQuant等）。量化参数（scale + zero-point）压缩为half2格式降低metadata访存。(2) Residual block-based KV cache分区：将KV cache按Tensor Cores对齐的residual block size Nr = Pn × Wn × R分割（Pn=warp tile elements, Wn=N维warp数, R=packing ratio ω/β），前Np entries做packed low-bit存储，后Nr entries保持FP16 residual cache。decode时新tokens先追加到residual cache→达Nr后由Residual Kernel批量量化写入。(3) Query transformation：将Q从[1, (gq, hkv)] reshape为[gq, hkv]，使GQA/MQA的grouped query heads在Tensor Cores上形成更大GEMM block以提高occupancy。(4) 4-bit量化仅0.2% LongBench accuracy degradation（LLaMA-3.1-8B-Instruct, seq_len=32K），2-bit量化2.7% degradation。实验比较：kernel-level speedup vs FP16 FlashDecoding-v2（Blackwell 8.6×, Hopper 8.0×, Ada 7.5×, Ampere 4.8×），end-to-end throughput vs Kivi (non-fused) 和 QServe (fused CUDA-only)，page-setting下over 2× higher throughput than QServe。

- 硬件平台是什么，配置是什么。
  NVIDIA Blackwell (RTX 5090, RTX PRO 6000)、Hopper (H100)、Ada (RTX 4090)、Ampere (A100 80GB)。多GPU: 8×A100 for LLaMA-3.1-70B。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-2-7B (MHA)、LLaMA-3.1-8B (GQA)、LLaMA-3.1-70B (GQA, 8×A100)、Qwen3-8B (GQA)、Qwen3-14B (GQA)。benchmark：LongBench（长上下文理解accuracy评估，seq_len=32K）。kernel benchmark：不同seq_len (1K-128K)、batch_size (1-128)、attention head配置 (h_q=32-128, h_k=8-32, d=128)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源: https://github.com/OpenBitSys/BitDecoding。算法pipeline：
  1. Prefill阶段（prompt length L）：Residual Kernel执行——QK^T和PV mma (FP16 Tensor Cores)→每个thread in-register计算scale/zero→INT4/INT2量化+pack到INT16→前L-(L mod Nr)个KV entries写入packed low-bit KV cache，剩余res_len entries保持FP16在residual cache。
  2. Decode阶段（逐token自回归生成）：
     a. 新生成K/V tokens（FP16）追加到FP16 residual cache
     b. Packing Kernel执行Attention：Q reshape [1, (gq, hkv)]→[gq, hkv] (Query Transformation)→加载low-bit packed K/V到register→lop3-based dequant (INT4→FP16)→QK^T mma (Tensor Cores)→cooperative softmax (cross-warp reduction via shared memory)→PV mma (Tensor Cores)→output
     c. 当residual cache累计达Nr tokens→Residual Kernel触发：将FP16 residual批量量化写入packed KV cache→清空residual cache
  3. Channel-wise quantization example (group_size=128, d=4096, β=4): K tensor (seq_len, d) → 沿seq_len维度分组，每组128 tokens计算channel-wise scale/zero → KEY quantized to INT4 → 4个INT4 pack到1个INT16。Tensor-wise: 沿hidden维度分组。
  4. Blackwell原生mxfp4：跳过dequant→Q (FP16) × K_packed (mxfp4) 直接用mxfp4 mma→softmax→P需dynamic re-quantize to mxfp4→P_packed (mxfp4) × V_packed (mxfp4) mma。
## AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  提出基于Product Quantization (PQ)的PIM-aware KV cache在线量化压缩框架AQPIM。核心算法创新：(1) PQ-based KV cache量化：将高维KV vector分解为m=32个子向量，用k-means clustering生成K=512个centroid per subvector的codebook，在线适应activation分布，compression ratio最高达80%+；(2) Importance-weighted k-means clustering：计算每个token的attention score权重w=sum(S[-t:, :], axis=0)，在k-means迭代中使用weighted centroid update（µ_k = Σ w_n x_n / Σ w_n, n∈C_k），使高attention score的token获得更小的量化误差；(3) Channel pre-sorting optimization：基于cosine similarity将高相关channel分组到同一subvector，sorting matrices P_k, P_v离线生成(calibration dataset: Wikitext-2-v1)并absorb到projection matrices W_q'=W_q·P_k, W_k'=W_k·P_k, W_v'=W_v·P_v, W_o=W_o·P_v^T，无运行时overhead；(4) Page-aware windowed clustering：限制每个window内512个centroid，保证indirect lookup完全在单一DRAM row buffer内完成，消除随机访问penalty；(5) PQ-based direct attention computation：将GEMV qK^T转换为query splits×codebook的inner product matrix lookup + summation，无需dequantization直接在压缩数据上计算。实验比较accuracy vs memory reduction ratio trade-off，对比SnapKV(sparse attention)、PQCache(PQ-based offloading)、SKVQ(channel-reorder quantization)，在LongBench 6个task上的accuracy；ablation study验证weighted clustering和channel pre-sorting对accuracy的提升贡献。

- 硬件平台是什么，配置是什么。
  GPU+HBM-PIM异构系统：1×NVIDIA H100 GPU + 5×16GB HBM（GPU+HBMs baseline）；PIM系统中4×16GB HBM替换为4×16GB HBM-PIM（参考AttAcc!架构）。CPU：Intel Xeon Platinum 8480+ Processor。bfloat16精度用于模型推理，FP16用于架构模拟（与prior work AttAcc!保持一致）。

- 模型是什么。数据集和bench分别是什么。
  模型：Mistral-7B-Instruct-v0.2 (32K context window), Llama-3.2-3B-Instruct (128K context window)。Benchmark：LongBench（含NarrativeQA/HotpotQA/GovReport/TREC/PassageRetrieval-en/LCC六个代表性task，覆盖single-/multi-document QA、summarization、few-shot learning、synthetic tasks、code completion）。Calibration dataset: Wikitext-2-v1（用于channel pre-sorting矩阵离线生成）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  论文代码未直接开源。构建于AttAcc! simulator ([55] https://github.com/scale-attacc-kr/attacc-sim) 和Ramulator2 ([20] https://github.com/CMU-SAFARI/ramulator2) 之上。补充材料见Zenodo (https://zenodo.org/records/17378113)。算法pipeline：
  1. Prefill阶段：GPU生成QKV matrices→KV offload到HBM-PIM→GPU执行attention和projection/FFN→PIM并行生成Key/Value codebook：在BankPE执行distance calculation (DC, FP16 MAC units)→BufferPE执行cluster assignment (CA, MIN unit)→BankPE+BufferPE协同centroid calculation (CC, MUL+SUM+DIV)→迭代4轮收敛。
  2. Decode阶段：GPU生成qkv→offload到PIM→PIM append新token的PQ indices→PIM执行PQ-based attention: query分成m=32 subvectors→BankPE执行query×codebook ATNK (MAC)得inner product matrix→BufferPE lookup indices+softmax (SFM)→BankPE ATNV (attention×value codebook reconstruction)→output回传GPU。
  3. Hyperparameters: m=32 subvectors, K=512 centroids, sink tokens保留前8个full precision, sliding window保留最近32个tokens full precision。Importance weight t=32（window内最近t tokens的attention scores）。

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

## LEGO: Supporting LLM-enhanced Games with One Gaming GPU

- 属于算法pipeline的实现是什么？实验比较什么？
  提出resource-oriented layer-skipping adaptor，在必须根据GPU资源预算跳过若干Transformer层时，用小型FFN adaptor近似被跳过层的知识表示变换，降低精度损失。核心算法设计：(1) Layer Similarity Heatmap：用cosine similarity量化所有Transformer层输出tensor间的相似度，发现后层高相似度但最后层与倒数第二层低相似（最后层编码与output layer对接的关键知识），对角线反映各跳层配置的候选层段；(2) Contiguous Layer Selection：当需跳过N层时，沿heatmap对角线选择相似度最高的连续层区间（如Llama3-8B跳4层选L25-L29，跳8层选L23-L31），避免离散跳层造成的inter-layer coherence disruption；(3) Adaptor Training：训练FFN adaptor（单层feed-forward network），输入第k层输出f_k，输出逼近第k+n层原始输出f_{k+n}，MSE loss L_mse = ||f_{k+n} - FFN^{k+n}_k(f_k)||²，仅更新adaptor权重（268.8MB/adaptor）；(4) Resource-driven skip decision：离线profile游戏rendering headroom范围→计算必须跳过的最小层数M和最多层数N→准备N-M+1个adaptor→运行时由scheduler根据预测headroom选择跳层数。实验比较：LEGO vs LITE（confidence-based early exit layer-skipping）和CALM（classifier-based layer-skipping），在MMLU/ARC-C accuracy和SQuAD-2.0 F1上对比不同跳层数（0/4/8/12/13/14）的精度。LEGO跳12层时相比LITE减少86.3% accuracy loss。还评估SmallModel baseline（Llama3-3B替换8B、Mistral-4B替换7B）。

- 硬件平台是什么，配置是什么。
  Windows 11, CUDA driver 566.36, CUDA SDK 12.1, DirectX 12.1。Intel i9-13900KF @ 3.00 GHz, Nvidia RTX 4090 (24GB)。所有游戏配置4K分辨率、高画质、60 FPS。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama3.2-8B-Instruct (FP16), Mistral-7B-Instruct-v0.3 (FP16)；扩展评估DeepSeek-V2-Lite (28层MoE), Mixtral-8x7B (MoE)。数据集：WebInstruct（adaptor训练集），MMLU/ARC-C (accuracy)、SQuAD-2.0 (F1) 作为下游评估benchmark。LLM推理代表性输入长度512 tokens，输出长度16 tokens。运行100/200/300 APM三种场景。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：论文未提供官方开源仓库。基于llama.cpp (github.com/ggml-org/llama.cpp, commit fc83a9e)。算法pipeline（以Llama3-8B + BlackMyth + 200 APM为例）：
  1. 离线阶段：用WebInstruct数据集对fine-tuned Llama3-8B做推理→每层输出tensor提取→计算所有层对的cosine similarity→构建similarity heatmap→profile游戏rendering trace得H_min/H_max→计算需准备的跳层配置数N-M+1（BlackMyth最多14个adaptor）→对每种跳层N，沿heatmap对角线选择最高相似度的连续层区间→训练FFN adaptor以MSE loss优化→总训练约36小时
  2. 在线推理：Scheduler根据预测headroom选择跳过N层→将相应层段的Transformer层替换为已训练的adaptor→LLM推理时，输入f_k经adaptor直接映射到f_{k+n}→跳过N个transformer层→剩余层正常执行→输出token
  3. 跳层粒度：decode阶段每Transformer layer约0.4ms→调度以layer为粒度；prefill阶段以self-attention (0.5ms)和FFN sublayer (1.0ms)为粒度
  4. 精度保证：跳≤12层时LEGO在MMLU上保持≥40.9，优于Llama3-3B baseline (58.2)；100/200 APM下90% case仅需跳≤5层

## Uni-STC: Unified Sparse Tensor Core

- 属于算法pipeline的实现是什么？实验比较什么？
  提出BBC（Bitmap-Bitmap-CSR）稀疏存储格式和TMS/DPG/SDPU三阶段task decomposition pipeline，统一支持SpMM、SpMV、SpMSpV、SpGEMM四类稀疏算子。BBC用CSR组织4x4 sparse tile，用两级bitmap描述tile内非零位置，使得四类kernel共享同一数据结构而无需在线格式转换。T1(16x16x16)任务经TMS拆分为T3(4x4x4)任务，再经DPG拆分为T4(1x1x4) segmented dot-product任务，最后由SDPU合并执行并预合并partial products。实验比较SpMV、SpMSpV、SpMM、SpGEMM四类kernel的performance、energy和energy efficiency density，对比DS-STC、RM-STC、NV-DTC、GAMMA、SIGMA、Trapezoid，并在DLMC DNN inference（ResNet-50/Transformer，sparsity 70%/98%）和AMG solver上做应用级评估。

- 硬件平台是什么，配置是什么。
  模拟器实验配置：GPU SM内集成Uni-STC tensor core，MAC array配置64 MAC@FP64或128 MAC@FP32。面积用Yosys + FreePDK45 + CACTI 7建模，critical path评估目标频率1.5 GHz。应用级：A100类GPU集成432个Uni-STC单元，额外面积约0.0425 mm²。BBC离线构建在64-core AMD EPYC 7702 CPU和NVIDIA A100 GPU上评估。

- 模型是什么。数据集和bench分别是什么。
  数据集：(1) SuiteSparse 2893个矩阵用于SpMV/SpMSpV/SpMM，2126个方阵用于SpGEMM（C=A²）；(2) DLMC 302个权重矩阵（ResNet-50、Transformer），sparsity 70%和98%；(3) AMG solver的FP64稀疏矩阵。SpMSpV输入向量随机生成50% sparsity，SpMM的B矩阵列数固定64。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：论文未提供GitHub代码仓库。Artifact Appendix标注Publicly available: Yes，提供Google Drive Docker artifact（https://drive.google.com/file/d/1o_pdtPdox7aEdRE2e4GtbEPiMFGpPHCu），含Python/Bash/C++ simulators，Ubuntu 22.04 + GCC/OpenMP/OpenCV，fast verification约5小时，complete verification约75小时需500-600GB存储。算法pipeline：
  1. 预处理：将sparse matrix从CSR/CSC转换为BBC格式（offline，64-core CPU <1000ms或A100 <100ms），BBC=top-level CSR组织4x4 sparse tile + tile内两级bitmap描述非零位置
  2. Load阶段：warp发出stc.load指令，收集A/B矩阵的tile metadata（top-level bitmap, index）和数值到Uni-STC的Matrix A/B Buffer
  3. Task Generation阶段：stc.task指令触发TMS读取top-level bitmap，将16x16x16 T1任务沿M/N/K三维拆为4x4x4 T3 task，写入Tile queue
  4. Task Concatenation阶段：8个DPG并行读取bottom-level bitmap，对T3任务内的A/B tile bitmap做overlay，形成表示sparse dot-product pattern的4-bit code，生成8-bit T4 task code，写入Dot-product queue
  5. Execution阶段：stc.numeric指令驱动SDPU弹出T4任务，执行segmented dot-product（1x1x4），累加到1KB accumulator buffer，最多合并4个partial products后写回C，结果通过register file返回SM

## VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

- 属于算法pipeline的实现是什么？实验比较什么？
  提出VAR-Turbo软件-硬件协同加速框架，包含三项算法优化：(1) Draft-Free Parallel Decoding (PD)：利用图像token的空间冗余（entropy分析表明图像token冗余远高于语言token），每轮同时预测所有masked token，按置信度TopK选择多个token并行解码，无需draft model，将采样步数降低>80%（256×256下从256步降至8-32步）；(2) Token Aggregation (TA)：在浅层Learning Region中，将token sequence划分为non-overlapped local window，先用Small Attention将窗口内token聚合为representative token，再经Big Attention做全局建模，减少约41% attention MAC且质量下降<0.5%；(3) Dynamic Bypass (DB)：在深层Inert Region中，用轻量MLP为token打importance score，仅TopK重要token进入完整Transformer（attention+FFN），其余bypass并通过token restoration避免信息丢失，额外减少约58% MAC。实验比较生成质量（IS/FID）和计算量（TFLOPs），对比Vanilla VAR、MaskGIT、ViTCoD、AdapTiV。VAR-Turbo-Peak在256×256下20 steps、2.8 TFLOPs、FID 2.65、IS 272.4；VAR-Turbo-Balance为8 steps、1.1 TFLOPs、FID 2.67、IS 268.6；512×512下Balance为32 steps、5.7 TFLOPs、FID 3.15、IS 259.6。

- 硬件平台是什么，配置是什么。
  算法训练：4×NVIDIA V100 GPU（ImageNet, 500 epochs, batch size 256, 816 GPU hours, AdamW: lr=1e-4, weight decay=1e-5, momentum (0.9, 0.96)）。通用平台baseline：Intel Xeon Platinum 8168 CPU @2.70GHz、NVIDIA V100 GPU。GPU延迟用torch.cuda.event测量，CPU延迟用time.time；功耗分别用pynvml和s-tui获取。硬件加速器：TSMC 28nm+HPC 1P8M CMOS, TT 25C, 7.09 mm², 1.98 W + 2×64bit HBM2 @2GHz 32GB/s。

- 模型是什么。数据集和bench分别是什么。
  模型：基于DeiT的generative Transformer + VQGAN tokenizer。数据集：ImageNet（主训练集和benchmark）。泛化实验：MS-COCO、CC3M、Places2以及ViT/DeiT/LeViT/SwinT-VAR backbone。评估指标：IS (Inception Score) 和 FID (Frechet Inception Distance)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：论文未提供明确开源仓库（HPCA 2026, DOI: 10.1109/HPCA68181.2026.11408607）。VAR-Turbo算法pipeline（以256×256图像生成8步Balance模式为例）：
  1. Tokenization：VQGAN将图像编码为256个visual token（16×16 grid）
  2. 初始化：文本条件与全masked visual token拼接为V0，mask array全True
  3. 每轮迭代：generative Transformer对所有当前位置输出logits→PD使用Gumbel sampling得pred token和概率→已unmasked token置信度设为负无穷→根据schedule r(t)计算K=N*(1-r(t))个释放位置→Radix Sort Core对confidence array执行TopK→TopK token替换为pred token并mask=False
  4. 每轮Transformer invocation内：浅层0-15层(Learning Region)执行TA——token分local window（低分辨率size=2，高分辨率size=2/4混合）→Small Attention聚合representative token→Big Attention全局建模；深层(Inert Region)执行DB——轻量MLP打分→TopK重要token进入attention+FFN→bypass token通过Token_i × JudgeWeight_i + Token_i恢复信息回下一层
  5. 所有visual token解码完成→VQGAN decoder还原像素图像
  6. PD、TA、DB协同：跨迭代减少Transformer调用次数（PD），每次调用内减少attention MAC（TA）和FFN MAC（DB）。关键trade-off：PD需PD-aware training选择sampling temperature/masking ratio/guidance scale；TA的local window size≥8时质量明显下降；DB需schedule function控制逐层skip rate（α=0.3, β=-0.4, max skip threshold=0.55）

## AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding

- 属于算法pipeline的实现是什么？实验比较什么？
  提出SLO-customized speculative decoding的speculate-select-verify pipeline：将multi-SLO serving形式化为带token budget约束的token tree构造问题，在每次decoding iteration中为不同SLO请求分配不同数量的speculation token tree节点，使严格SLO请求获得更多验证token、宽松SLO请求节省budget给其他请求。核心算法分三阶段：(1) Speculation phase：用小draft model对每个请求做d步beam search（每步width=w），从root（每个请求最后生成的token）开始逐步扩展候选child tokens，记录由draft logits近似的path probability；(2) SLO-customized selection：贪心为每个请求选择最高path probability节点直到累计expected accepted tokens满足其TPOT SLO约束，严格请求优先、宽松请求节省；(3) Throughput-optimized selection：满足所有SLO后剩余budget按全局path probability排序分配给各请求，最大化总expected accepted tokens。系统根据活跃请求数动态调节speculation depth d和beam width w。实验比较vLLM（continuous batching）、Sarathi-Serve（chunked prefill）、vLLM-Spec(n)（固定speculation length n=4/6/8）、SpecInfer（静态speculative decoding），评估不同RPS、严格请求比例、SLO scale下的SLO attainment和goodput。

- 硬件平台是什么，配置是什么。
  4×NVIDIA A100 80GB GPU (NVLink)，AMD EPYC 7763 (64 cores/128 threads)，256GB DRAM。Llama3.1-70B-Instruct (4-way TP) + Llama-3.2-1B-Instruct draft model；Qwen2.5-32B-Instruct (2-way TP) + Qwen2.5-0.5B-Instruct draft model。CUDA 12.4。

- 模型是什么。数据集和bench分别是什么。
  Target LLM: Llama3.1-70B-Instruct（4-way tensor parallelism）、Qwen2.5-32B-Instruct（2-way tensor parallelism）。Draft model: Llama-3.2-1B-Instruct、Qwen2.5-0.5B-Instruct（不做任务特定finetune）。Workload数据集：coding copilot (HumanEval, SLO=1.2×baseline latency)、chatbot (Alpaca, SLO=50ms/token)、summarization (CNN/DailyMail, SLO=150ms/token)。请求到达率：源自真实trace经截断与缩放形成不同RPS。指标：SLO attainment（满足TPOT SLO的请求比例）、goodput（成功满足SLO的请求产生的tokens/s）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：https://github.com/zikun-li/AdaServe-Artifact-Evaluation（artifact evaluation），Zenodo DOI: 10.5281/zenodo.17052619。
  SLO-customized speculative decoding算法pipeline（以batch含coding copilot + summarization两请求为例）：
  1. 初始化：scheduler取活跃请求→计算每请求本轮SLO推进目标：expected_accepted_tokens ≥ latency_elapsed / TPOT_SLO - tokens_generated。coding copilot（SLO紧）需求高→假设需≥3 expected accepted tokens；summarization（SLO宽）需求低→假设需≥1。
  2. Speculation phase：draft model对coding copilot做d步beam search（d=4, w=3）→生成候选tree节点（root→12个candidate nodes），每个节点记录draft logit作为path probability近似。summarization同样生成候选tree。
  3. SLO-customized selection：先处理coding copilot（需求更高）→在候选tree中按path prob从高到低选节点：node_A(0.8)→node_B(0.6)→node_C(0.45)，累计1.85 expected tokens，加入SLO tree。再处理summarization→选node_X(0.7)，累计0.7→已达≥1需求，停止。
  4. Throughput-optimized selection：若token budget=6，SLO阶段已用4个节点→剩余2个budget。全局排序所有请求未选候选节点→选最高path prob两节点补入各自tree。
  5. Verification：所有请求selected token trees提交target LLM做tree-based parallel verification→LLM并行验证共享前缀的多条speculative path→接受匹配token，拒绝错误分支返回correction token。
  6. 状态更新：accepted tokens写回request pool→若coding copilot接受3个token（SLO达成）→下一轮目标降低；若summarization仅接受0个（概率较低）→但SLO宽仍可容忍。结果：同一batch不同请求在一次大模型验证中前进不同token数，打破continuous batching的统一per-token latency。
  7. 关键trade-off：path probability用draft logits近似而非真实分布→效果依赖draft model与target一致性；per-request token limit防止某请求吞过多低概率budget→极端紧SLO可能只能被尽量推进而非完全满足。

## MFS: An Efficient Model Family Serving System for LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  提出Knowledge Precipitation，一种将LLM model family中最大模型微调为嵌套式multi-tier模型的离线fine-tuning方法。核心算法设计：(1) Multi-tier loss：每个tier有独立language modeling loss，总训练目标L = L0 + λ1L1 + ... + λiLi，使低tier获得独立语言建模能力，同时由高tier梯度向低tier传递知识；(2) Step-by-step fine-tuning：对n-tier模型从高tier到低tier逐层fine-tune，避免一次性叠加所有tier loss导致低tier性能不可控；(3) Tier boundary selection：不是按参数量线性切分，而是测量最大模型前若干层实际推理latency，使某tier的latency对齐对应小模型（如Llama2-13B前24层对齐7B latency）；(4) Layer-only tier split：只按layer划分tier而不切head，保持attention计算中所有head一致性，使跨tier KV cache可共享。
  实验比较：(a) 质量评估：MFS-7B (2-tier from Llama2-7B) 在MMLU/PIQA/OpenBookQA/HellaSWAG/BoolQ/ARC-Easy/ARC-Challenge/ANLI-R1-R2-R3共10个指标中8个优于Llama2-7B；MFS-13B/MFS-7B (2-tier from Llama2-13B) 分别8/7个指标优于对应原始模型；(b) 构造方法对比：strawman early-exiting（低tier生成无意义文本）、PEFT/LoRA（不能解决生成质量）、从头训练early-exit（代价极高）、deep pruning（破坏层连续性和KV cache兼容性）——论文以设计论证和局部实验说明它们不满足MFS要求；(c) 三tier实验：13B/7B/3B和13B/10B/7B，各tier质量接近对应独立baseline；(d) 泛化验证：Qwen-14B/Qwen-7B上验证方法可迁移性。

- 硬件平台是什么，配置是什么。
  训练：2台服务器，每台8×NVIDIA H800 SXM5 GPU (80GB)、2×56核Intel Xeon Gold CPU、2TB内存，8×400Gbps NDR InfiniBand互联。推理评估：(1) 2×NVIDIA A100 GPU、2×48核CPU、512GB内存；(2) 8×NVIDIA 3090 GPU、80核CPU、256GB内存。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama2-7B-chat → MFS-7B (2-tier)、Llama2-13B-chat → MFS-13B (2-tier和3-tier: 13B/7B/3B, 13B/10B/7B)。泛化验证：Qwen-14B → Qwen-7B (Qwen1.5 series)。fine-tuning数据：HuggingFace guanaco-llama2（约9.85k对话）。优化器设置：AdamW, learning rate=2e-5, half-period cosine learning rate schedule, weight decay=0.1, gradient clipping=0.3, 8×gradient accumulation (effective batch size=64), input sequence length=4096, fine-tune 1 epoch/2500 iterations, 约24小时 on 16×H800。质量benchmark：MMLU、PIQA、OpenBookQA、HellaSWAG、BoolQ、ARC Easy、ARC Challenge、ANLI-R1/R2/R3共10个指标。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  未找到MFS官方开源仓库。算法pipeline（以Llama2-13B-chat → 3-tier MFS为例）：
  1. Tier结构设计：测量Llama2-13B各层在A100上推理latency→确定tier-1边界为第18层（latency对齐3B）、tier-2边界为第32层（latency对齐10B）、tier-3为全部40层（=13B）。
  2. Step 1 — Fine-tune tier-3 (最高tier)：对Llama2-13B-chat全量参数用guanaco-llama2做SFT，loss仅含tier-3输出头L=L3，产出高质量tier-3 checkpoint作为后续基础。
  3. Step 2 — Fine-tune tier-2：基于tier-3 checkpoint，在第32层添加tier-2输出头（lm_head），训练目标L=L2+λ3L3。tier-3梯度反向传播到前32层共享参数，使tier-2获得独立生成能力同时tier-3质量不退化。
  4. Step 3 — Fine-tune tier-1 (最低tier)：基于step 2 checkpoint，在第18层添加tier-1输出头，训练目标L=L1+λ2L2+λ3L3。低tier通过接收高tier梯度"沉淀"知识，获得独立语言建模能力（如生成简短低延迟回答）。
  5. 输出：单一嵌套模型checkpoint，所有tier共享低层Transformer参数，各tier有独立输出头。推理时请求在对应tier边界采样返回——低tier早退出（低延迟低成本）、高tier继续执行剩余层（高质量）。
  6. 关键设计决策：(a) 只切layer不切head——保持attention中所有head一致性，使跨tier KV cache可共享；(b) step-by-step而非joint training——避免多tier loss梯度冲突导致低tier性能不可控；(c) latency-aligned而非parameter-aligned切分——使每个tier的实际推理延迟匹配用户对不同规模模型的体验预期。

## Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

- 属于算法pipeline的实现是什么？实验比较什么？
  提出面向Qualcomm Hexagon NPU的端到端LLM test-time scaling推理系统。核心算法pipeline：(1) Hardware-aware fine-grained tile quantization：将权重按HMX FP16 tile layout（tile级column-major、tile内每两行permute）重排后，以2×16 tile片段为quantization group做group size 32的4-bit量化。量化后通过group coalescing将8个quantization group合并为一个128-byte super-group，使256个INT4值正好填满一个HVX 1024-bit vector register，scale连续存放。(2) LUT-based Softmax：利用safe softmax保证exp输入非正，预计算64KiB FP16 exp LUT（仅x≤0的32768个entry），通过vgather指令查表替代exp2多项式展开，消除VLIW顺序依赖。(3) LUT-based dequantization：用vlut16指令将INT4权重值直接映射为FP16，并用LUT广播多group scale，替代传统mask-unpack-convert流程。(4) Test-time scaling：将Best-of-N/Beam Search的多候选并行采样映射到decode batch，填充HMX 32×32 tile的空置行。实验比较：accuracy-latency Pareto frontier（Best-of-N/Beam Search vs base model scaling），算子消融（LUT Softmax vs F32/F16 polynomial exp，tile quantized GEMM vs conventional layout/HMX layout only/no dequantization upper bound），端到端decode/prefill throughput对比（Ours vs llama.cpp OpenCL backend vs QNN FP16）。

- 硬件平台是什么，配置是什么。
  NPU性能实验：OnePlus Ace3 (Snapdragon 8 Gen 2, Hexagon V73)、OnePlus 12 (Snapdragon 8 Gen 3, Hexagon V75)、OnePlus Ace5 Pro (Snapdragon 8 Elite, Hexagon V79)。部分准确率实验用NVIDIA RTX3090服务器。Hexagon NPU架构：6-8 scalar VLIW threads、4-6 HVX vector units (1024-bit registers)、1-2 HMX matrix units、1MiB L2 cache、8MiB TCM、DMA ~60GB/s、l2fetch 20-30GB/s。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen2.5-1.5B/3B/7B-Instruct、Llama3.2-1B/3B-Instruct。PRM scorer：Skywork-1.5B-PRM。数据集：MATH500、GSM8K（数学推理 pass@1）、WinoGrande、MMLU、Wikitext-2（PPL）。统一0-shot CoT prompt。量化方案：多数矩阵Q4_0 (~4.5 BPW)，FFN down矩阵Q8_0 (~8.5 BPW)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：主仓库 https://github.com/haozixu/llama.cpp-npu，算子库 https://github.com/haozixu/htp-ops-lib。算法pipeline：
  1. 离线阶段：HuggingFace weight→按HMX tile layout（32×32 tile column-major + tile内2-row permutation）重排→在memory order上按group size 32做4-bit group quantization（等于2×16 tile片段为单位量化）→8个group coalesce为super-group（256个INT4填满一个HVX register，scale连续存放）→输出HMX layout GGUF格式。
  2. 在线decode：CPU侧llama.cpp Hexagon NPU backend将activation/KV cache/operator request通过rpcmem/dmabuf共享内存写入→手动cache flush→FastRPC远程NPU session启动→NPU侧线程轮询共享内存接收请求。HVX用vlut16查表将INT4权重转FP16并广播scale，HMX执行FP16 tile-level inner product。进入attention时FlashAttention按Q/KV tile流式计算，HMX算QK和PV，HVX做rowmax/rowsum/LUT_Exp。
  3. Test-time scaling：Best-of-N保留多个候选路径→batch size=B时HMX的32×32 tile有B行有效计算（vs单路径decode仅1行）→利用原本空置的matrix compute→最终由PRM/ORM scorer选择最优输出。Beam Search在中间步骤用PRM打分剪枝低质量路径。lm_head和logits保留在CPU（因Hexagon NPU 32-bit虚拟地址空间限制），batch size=16时CPU logits计算占比接近或超过50%。

## TailorLLM: Collaborative End-Cloud Inference of Large and Small Language Models Based on Low-Rank Adaptation

- 属于算法pipeline的实现是什么？实验比较什么？
  RFLoRA (Resource-Friendly Low-Rank Adaptation)：一种参数高效的LoRA变体，将预训练权重W解耦为direction和magnitude两个分量（W = m · W/||W||_c），仅对direction分量应用LoRA低秩分解（BA），magnitude标量m和B矩阵可训练，A矩阵冻结并在所有任务间共享。核心发现：(1) 跨任务微调时A矩阵趋于收敛（capture domain-invariant features），B矩阵呈现任务特异性（adapt to domain-specific variations）；(2) 权重可分解为方向和幅度分量，分别优化加速收敛。RFLoRA使端侧只需预存一份共享A矩阵，传输时仅需发送任务特定的B矩阵+m（相比标准LoRA减少~50%传输/存储开销）。实验比较：在Llama3-1B上用8个NLP任务（MRPC/COLA/QNLI/RTE/SST-2/MNLI/QQP/BoolQ）对比Llama3-1B (无微调)、Llama3-70B、LoRA、DoRA、AdaLoRA、HydraLoRA。GSM8K数学任务因1B模型无法通过LoRA达到可接受精度而自动卸载到云端。

- 硬件平台是什么，配置是什么。
  Cloud-side: 4×NVIDIA RTX 3090 GPU (24GB GDDR6X)，Ubuntu 20.04 LTS。End-side: NVIDIA Tesla T4 GPU (16GB limited to 10GB)，Ubuntu 20.04 LTS。微调训练在云端RTX 3090上执行。

- 模型是什么。数据集和bench分别是什么。
  SLM: Llama3-1B。LLM: Llama3-70B。数据集：GSM8K（数学推理，自动卸载到云）、MRPC（语义等价）、COLA（语法可接受性）、QNLI（自然语言推理）、RTE（文本蕴涵）、SST-2（情感分析）、MNLI（多体裁推理）、QQP（查询等价）、BoolQ（是否问答）。80% fine-tune / 20% test split。所有PEFT方法共享相同训练schedule、数据split和超参数（r=16，HydraLoRA因r=16下梯度爆炸使用r=32）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  论文未明确说明开源。RFLoRA算法pipeline：
  1. 预训练权重分解：对LLM每层weight matrix W∈R^{d×k}，计算column-wise norm m=||W||_c∈R^d，direction矩阵 V=W/m（即W/||W||_c column-wise normalization）。初始W = m · V/||V||_c。
  2. LoRA低秩注入direction：仅对direction分量V施加ΔV = B·A。A∈R^{r×k} Kaiming初始化并全局冻结（跨任务共享），B∈R^{d×r} zero初始化可训练。magnitude m可训练。
  3. 更新公式：W' = m · (V + ΔV) / ||V + ΔV||_c = m · (W_0 + B·A) / ||W_0 + B·A||_c。反向传播仅更新m和B。
  4. 传输：云端训练完成后仅传输B矩阵+magnitude参数m给端侧（~11.56MB per adapter，原始LoRA ~22MB）。端侧预存一份共享A矩阵。
  5. 端侧推理：加载对应任务B+m→与预存A构成完整LoRA→注入SLM→推理。
  6. 结果：RFLoRA 81.6% avg accuracy (3.4M trainable params, 0.273% of full model)，vs LoRA 81.2% (0.454%)、DoRA 82.1% (0.484%)、AdaLoRA 81.0% (0.680%)、HydraLoRA 81.2% (1.277%)。在trainable params远少于DoRA的情况下精度接近。将Llama-1B与Llama-70B的精度差距从28.2缩小到3.5个百分点。

## FlashPS: Efficient Generative Image Editing with Mask-aware Caching and Scheduling

- 属于算法pipeline的实现是什么？实验比较什么？
  属于算法pipeline的实现是FlashPS的mask-aware cached activation计算pipeline：将扩散模型transformer block内token按mask划分为masked/unmasked两类，对attention和feed-forward等token-wise计算只对masked tokens执行，unmasked tokens的输出直接复用预先缓存的block输出activation Y（非K/V cache）。同时用DP在block粒度动态决定哪些transformer block使用cache加载+masked-only计算、哪些block直接全量计算以消除pipeline bubble。实验比较baseline：Diffusers（标准全图生成）、FISEdit（mask sparsity，仅SD2.1）、TeaCache（通用activation reuse）。指标包括端到端延迟、图像质量（CLIP/FID/SSIM/用户研究）、P95尾延迟。

- 硬件平台是什么，配置是什么。
  SD2.1: NVIDIA A10 GPU。SDXL和Flux: NVIDIA H800 GPU。在线服务评测使用单台8-GPU机器，每个worker分配1张GPU。SD2.1最大batch size=4，SDXL/Flux最大batch size=8。

- 模型是什么。数据集和bench分别是什么。
  模型：Stable Diffusion 2.1 (SD2.1)、Stable Diffusion XL (SDXL)、Flux。数据集与benchmark：InstructPix2Pix（图像编辑）、VITON-HD（虚拟试穿）、PIE-Bench（图像编辑benchmark）。生产trace：2025年1月14天trace，覆盖20k GPU、3400万张生成图像、970个templates（平均各复用约35k次），平均mask ratio为0.11。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源地址：https://github.com/Sylvia-16/FlashPS；Zenodo: https://zenodo.org/records/17176576。提供Docker镜像jiangxiaoxiao/flashps。
  FlashPS mask-aware算法pipeline：
  1. 图像和mask编码为latent，将latent reshape后的空间位置视为transformer tokens，按mask划分masked/unmasked tokens。
  2. 对于linear projection、feed-forward、LayerNorm、activation等token-wise计算：unmasked token的输出直接从cached activation（存储在host memory/disk/distributed storage）读取，masked tokens执行完整计算。
  3. 对于attention：观察到unmasked token在输出Y上跨请求高度相似，且masked/unmasked token间cross-attention较弱，因此缓存transformer block输出Y中unmasked tokens的activation（非K/V cache），减少cache footprint。
  4. Bubble-free DP：对每个transformer block比较"加载cache后仅计算masked tokens的完成时间"与"全量计算完成时间"，用O(N) DP为每个block决定是否使用cache——避免在cache loading比直接计算慢的block上产生pipeline bubble。
  5. CUDA stream异步加载：cache load stream从host memory异步加载cached activations到GPU HBM，computation stream并行对masked tokens执行attention/feed-forward；block输出时将masked token计算结果与cached unmasked activations合并为完整Y送入下一block。
  6. 理论speedup约为1/m（m为mask ratio），实际mask ratio=0.2时SD2.1/SDXL/Flux加速比分别为1.3x/2.2x/1.9x。SSIM最高达0.99。

## Latent Wavelet Diffusion for Ultra-High-Resolution Image Synthesis

- 属于算法pipeline的实现是什么？实验比较什么？
  属于算法pipeline的实现是Latent Wavelet Diffusion (LWD)，一个纯训练阶段的频率感知框架，不做任何架构修改、推理期零额外开销。包含两个阶段：(1) 用Scale-Consistency (SC) loss微调VAE，抑制跨尺度不一致的高频伪影，使latent空间更适合下游wavelet分解；(2) 用wavelet-masked flow matching objective微调扩散模型——对latent z_t做单层Haar DWT得到LL/LH/HL/HH四个子带，计算高频能量图Awavelet∈[0,1]^{H×W}作为空间显著性度量，再用time-dependent binary mask Mt(i,j) = 1 if T·(Awavelet+l)≥t else 0来调制训练loss，使得高频区域（纹理/边缘）在更多timestep接受监督，平滑区域接受较少监督。l=0.3为消融选出的下界。实验比较baseline：Flux-1.dev、SD3-F16、SD3-Diff4k-F16、PixArt-Sigma-XL、Sana-1.6B、URAE，以及外部baseline SDEdit、I-Max、Diffusion-4K、Lumina-Image 2.0。指标包括FID、LPIPS、MAN-IQA、QualiCLIP、HPSv2.1、PickScore、CLIPScore、Aesthetics、GLCM Score、Compression Ratio、频率域指标HLFR/RDR/WQS/HFE/HFEI。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 GPU（4×A100, 64GB each）。VAE fine-tuning: 60K steps, batch size=4, lr=1×10^{-5}。各backbone训练：LWD+URAE(Flux) 2K约4h/4K约24h, batch size=1; LWD+Diff4K(SD3) 2K约48h, batch size=8; LWD+SANA 2K/4K约24h, batch size=2/1; LWD+PixArt-Σ 2K约24h, batch size=2。训练期peak memory增加~3-4%（如Sana从90.5%到93.9% A100 memory），每20step时间几乎不变（~47s），推理期零开销。

- 模型是什么。数据集和bench分别是什么。
  模型：Flux（flow-matching backbone）、SD3（MMDiT backbone, 16ch VAE）、PixArt-Sigma-XL、Sana-1.6B（linear DiT）、URAE（基于Flux-1.dev的参数高效适配）。数据集：Aesthetic-4K（策展4K benchmark, GPT-4o captions）、LAION-High-Res（50K 2K + 20K 4K image-caption pairs, LAION-5B子集）、HPD prompts（Wu et al., 2023）。评估bench：Aesthetic-Eval、HPD prompt dataset。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源地址：https://github.com/LuigiSigillo/LatentWaveletDiffusion。使用PyTorch + pytorch-wavelets（Cotter, 2019）实现Haar DWT。LWD算法pipeline：
  1. Stage 1 — VAE微调：用scale-consistency loss L_VAE = ||D(z)-x||²₂ + α||D(E(z_down))-x_down||²₂ + β D_KL(q(z|x)||p(z)) + λ L_LPIPS(D(z),x) 微调预训练VAE（Flux-VAE/SD3-VAE/Sana-AE），权重α=0.25, β=0.001, λ=0.05。该阶段被证明是wavelet masking有效的前提——抑制跨尺度不一致的高频噪声，使后续DWT提取的高频能量对应有意义结构而非伪影。
  2. Stage 2 — Wavelet-masked flow matching：对每个training step的latent z_t = (1-t)z_0 + tε，执行单层Haar DWT→计算HF energy map E(i,j) = (1/C) Σ_c [(z_LH)²+(z_HL)²+(z_HH)²]→bilinear上采样+min-max归一化得Awavelet→按Mt(i,j)=1 if T·(Awavelet+l)≥t else 0生成binary mask→计算masked loss L_masked = ||M_t ⊙ [(ε-z_0) - v_Θ(z_t,t,y)]||²₂。Haar wavelet选型的原因：最紧凑support（2 coefficients）提供最精确空间定位，计算效率最高，避免FFT的Gibbs ringing导致的mask边界模糊（Haar GLCM 0.74 vs FFT 0.71）。
  3. 推理期：与baseline完全相同——LWD仅在训练期修改objective，不改变模型参数结构，推理zero overhead。LWD模型与baseline参数量相同、inference time相同。
  4. 收敛加速：LWD仅需baseline论文建议原始训练iteration的10-50%即达收敛。LWD+URAE仅需2k steps（约4h 2K/24h 4K），而Diff4K需要10k steps（约48h）。

## Spectral Regularization for Diffusion Models

- 属于算法pipeline的实现是什么？实验比较什么？
  提出loss-level spectral regularization框架，在标准diffusion training objective（DDPM/DDIM/EDM）基础上augment可微分的Fourier-domain和wavelet-domain L1 penalty terms，不改动diffusion process、模型架构或sampling procedure。核心设计：(1) Fourier Amplitude Loss (LA_F)：在predicted clean sample x̂₀与ground-truth x₀之间计算Fourier amplitude spectrum的L1差异——amplitude discrepancy对应frequency-wise energy allocation mismatch而非local phase misalignment，直接控制reconstruction error在frequency bands间的分布。(2) Fourier Amplitude-and-Phase Loss (LAP_F)：将phase信息通过amplitude coupling引入：LAP_F = E[||A₀−Â₀||₁·(1+||ϕ₀−ϕ̂₀||₁)]，利用amplitude加权phase penalty，避免对low-amplitude band的insignificant phase noise过度penalize，同时稳定fine-scale structure。(3) Wavelet Coefficient Matching Loss (LW)：对Haar和bior1.3两种wavelet，在predicted和ground-truth sample的所有scales和orientations上计算wavelet coefficient的L1差异：LW = E[Σ_{s,ℓ} γ_{s,l}||W₀^{(s,ℓ)}−Ŵ₀^{(s,ℓ)}||₁]，提供localized、scale-aware的multi-resolution control。(4) 最终objective：L_total = L + λ L_S，L为standard denoising loss，λ控制regularization强度。所有spectral loss使用L1而非L2（有意break Parseval invariance以直接控制error的spectral distribution）。Fourier transforms用PyTorch FFT实现，wavelet transforms用PyWavelets。训练方式为lightweight fine-tuning：图像仅需5 optimization steps from pretrained EDM checkpoint，音频需150K steps。实验比较：图像在CIFAR-10/FFHQ/AFHQ上对比EDM baseline (VE/VP variants)，度量FID；音频在LJSpeech上对比DiffWave baseline，度量FAD/UTMOS/PESQ/MR-STFT/NDB。还包含checkerboard toy experiment验证spectral regularizer对高频periodic structure的preservation能力。

- 硬件平台是什么，配置是什么。
  NVIDIA A4000和A6000 GPU。Per-GPU batch size=16。EDM fine-tuning duration=0.5（CIFAR/AFHQ）、learning rate=2×10⁻⁴ (CIFAR)/5×10⁻⁵ (AFHQ/FFHQ)。DiffWave fine-tuning：sample rate=22050 Hz, nmels=80, nfft=1024, hop=256 samples, learning rate=2×10⁻⁴, batch size=16, 150K steps。

- 模型是什么。数据集和bench分别是什么。
  图像：EDM (Karras et al., 2022)，DDPM++ (VP) 和 NCSN++ (VE) variants，pretrained on CIFAR-10 32×32 (conditional+unconditional)、FFHQ 64×64 (unconditional)、AFHQv2 64×64 (unconditional)。预训练权重从NVIDIA官方release获取。采样步数：CIFAR 18 steps, AFHQ/FFHQ 40 steps。FID on 50K generated samples vs full real dataset，3 random seeds average。音频：DiffWave (Kong et al., 2021)，30 residual layers, 64 residual channels, dilation cycle length=10, conditional training。LJSpeech-1.1 dataset (13,100 utterances, single speaker)，排除LJ001*/LJ002*用于evaluation。Inference noise schedule=[10⁻⁴,10⁻³,10⁻²,0.05,0.2,0.5]。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：https://anonymous.4open.science/r/fourierdm-8B8E。算法pipeline（以FFHQ 64×64 unconditional + VP-EDM + Fourier amplitude loss fine-tuning为例）：
  1. 加载pretrained EDM VP checkpoint（NVIDIA官方release的edm-ffhq-64x64-uncond-vp.pkl）。
  2. Fine-tuning loop（仅5 optimization steps）：对每个batch采样diffusion timestep t→加噪得x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε→用DDIM一步reconstruction得x̂₀（Eq.5: x̂₀ = (x_t − √(1−ᾱ_t)·ε_θ(x_t,t)) / √ᾱ_t）→对x̂₀和x₀分别计算Fourier transform（PyTorch FFT）→计算amplitude spectrum L1 loss LAF = |||F[x₀]| − |F[x̂₀]|||₁→总loss L_total = L_EDM + λ·LAF→反向传播更新模型参数。
  3. 关键实现细节：spectral loss在predicted clean waveform x̂₀上计算（通过DDIM一步得到），而非直接在noisy input x_t上做transform。这确保spectral supervision作用在sample-consistent estimate of clean signal上，与model generation pathway对齐。
  4. 采样：与standard EDM完全一致，无额外overhead——spectral regularization仅在训练时作为auxiliary loss生效，不改动sampler或architecture。
  5. 对比baseline（无spectral loss的标准EDM fine-tuning），仅5步fine-tuning即可在FFHQ上获得0.02-0.07的FID improvement，证明spectral bias作为data-efficient inductive prior的有效性。

## ELF: Embedded Language Flows

- 属于算法pipeline的实现是什么？实验比较什么？
  提出ELF（Embedded Language Flows），一种在连续embedding空间中使用连续时间Flow Matching的扩散语言模型（DLM）。核心算法设计：(1) Continuous embedding space：使用frozen pretrained T5-small encoder将离散token映射为连续contextual embeddings（512-dim），通过bottleneck线性投影至128-dim后送入model hidden size 768。在连续embedding space中直接执行denoising，仅在最后一步（t=1）通过共享权重的unembedding层映射回离散token；(2) Flow Matching with x-prediction：使用continuous-time Flow Matching（rectified flow），linear interpolant z_t = t·x + (1-t)·ϵ，网络直接预测clean embeddings x（x-prediction），训练目标为MSE loss L_MSE = ‖(x_θ(z_t,t) − x)/(1−t)‖²；(3) Shared-weight denoiser-decoder：单一网络同时作为denoiser（80%概率，MSE loss）和decoder（20%概率，cross-entropy loss），decode branch使用per-token corruption（logit-normal noise schedule，P_mean=0.8, P_std=0.8）模拟不完美denoiser输出，训练共享权重的unembedding层进行最终离散化。无需单独训练decoder；(4) Training-time CFG with self-conditioning：使用self-conditioning（50%概率concatenate前一预测x̂'作为条件）作为CFG的条件信号，训练时即融入CFG formulation：v_target = v + (1−1/ω)(v_sc − v_no_sc)，CFG scale ω∈[0.5,5]。使用in-context conditioning（4 time tokens + 4 CFG-scale tokens + 4 mode tokens prepended to sequence）替代adaLN-Zero，减少参数量（ELF-B from 148M→105M）；(5) SDE-inspired sampler：在ODE基础上每个step注入Gaussian noise（z_back = α·z + (1-α)·ε, α=1-γ·dt），然后对perturbed state调用denoiser，使用clean prediction更新原state。γ=0退化为ODE，γ>0为SDE。实验比较unconditional generation (OWT) 和conditional generation (WMT14 De-En translation, XSum summarization)，对比MDLM、Duo (discrete DLMs)、FLM/FMLM、LangFlow (continuous DLMs)、SeqDiffuSeq、CDCD、E2D2、AR baseline。

- 硬件平台是什么，配置是什么。
  Google TPU v5p × 64（训练）。训练时间：OWT上ELF-B每epoch约1.5小时。推断使用ODE/SDE sampler，支持32/64/128/256/512/1024 sampling steps。

- 模型是什么。数据集和bench分别是什么。
  模型：ELF-B (105M, 12层, hidden 768, 12 heads)、ELF-M (342M, 24层, hidden 1056, 16 heads)、ELF-L (652M, 32层, hidden 1280, 16 heads)。基于Diffusion Transformer (DiT)架构+ SwiGLU + RMSNorm + RoPE + qk-norm。使用frozen pretrained T5-small encoder (35M)作为embedding encoder。Muon optimizer (lr=0.002, batch size=512)。数据集：OpenWebText (OWT, ~9B tokens, 序列长度L=1024, 5 epochs ≈ 45.2B effective tokens)、WMT14 German-English (De-En, L=128, 144M target tokens)、XSum (L=1088, 6M target tokens)。评价指标：Generative Perplexity (Gen.PPL, 用GPT-2 Large评估)、unigram entropy、BLEU、ROUGE-1/2/L。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：https://github.com/lillian039/ELF。算法pipeline（以OWT unconditional generation + ELF-B + 32-step SDE sampling为例）：
  1. 训练阶段：
     a. Encode：使用frozen T5-small encoder将离散token序列s (L=1024)映射为continuous embeddings x (512-dim)→通过bottleneck线性投影至128-dim→再投影至hidden size 768。
     b. Denoising branch (80%概率)：采样t~logit-normal(P_mean=-1.5, P_std=0.8)→z_t = t·x + (1-t)·ϵ（ϵ~N(0,I), noise scale=2）→self-conditioning（50%概率concatenate前次预测x̂'）→prepend control tokens (time, CFG scale, mode)→送入DiT网络预测x̂→计算MSE loss L_MSE = ‖(x̂−x)/(1−t)‖²（实际为v-prediction转换后的velocity loss）。
     c. Decoding branch (20%概率)：t=1, per-token corruption p~logit-normal(0.8,0.8)→z̃ = p·x + (1−p)·ϵ（noise scale=5）→送入同一网络（mode="decode"）→unembedding层W映射为vocabulary logits→cross-entropy loss L_CE。
     d. Training-time CFG：同时计算conditional和unconditional prediction→v_target = v + (1−1/w)(v_sc − v_no_sc)→以CFG-combined target训练单一网络。
  2. 推断阶段（32-step SDE with CFG scale=3, γ=1.5）：
     a. z_0 ~ N(0,I)→对每个time step (logit-normal schedule, T=32 intervals)：self-condition on previous x_pred→调用denoiser得x̂→v = (x̂−z)/(1−t)→SDE step（注入噪声：α=1-γ·dt, z_back=α·z+(1-α)·ε, 在z_back上重新预测x̂，用原z更新z=z+dt·v）。
     b. 最后一步（t=1）：调用decode mode→unembedding→argmax得离散token→输出文本。
  3. ELF-B在32步SDE下Gen.PPL=24.08（vs MDLM 1024步Gen.PPL≈27, Duo 1024步≈34），使用仅45B training tokens（vs baselines 524-577B），10× fewer training tokens。

