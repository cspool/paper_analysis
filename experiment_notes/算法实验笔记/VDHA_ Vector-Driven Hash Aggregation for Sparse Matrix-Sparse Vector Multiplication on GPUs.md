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
