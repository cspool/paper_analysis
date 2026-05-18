
## FMPQ (Fine-Grained Mixed-Precision Quantization)

术语解释
FMPQ是COMET论文提出的细粒度混合精度后训练量化算法，首次实现对LLM激活的实用INT4量化，使得W4A4KV4配置（INT4权重+混合INT4/INT8激活+INT4 KV cache）可在现代GPU上高效推理且精度损失可忽略。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FMPQ解决的核心问题：LLM激活中存在outlier channels（<1%的feature dimensions有显著大于正常值的magnitude），直接全INT4量化导致严重精度崩溃（WikiText2 PPL增加>5.2）。FMPQ的策略是将激活张量沿channel维度按block_size=k=128划分（对齐A100 FP16 tensor core最小计算粒度64×64×32的整数倍），对每个block独立判别——含outlier的block量化为INT8，正常block量化为INT4。通过calibration data采样定位outlier channels后，使用channel permutation将分散在不同channel的outlier聚集到少数block中，使需要INT8量化的block比例从~50%降至~16%（LLaMA-1-30B甚至仅8%）。KV cache部分：K cache利用RoPE和softmax的outlier正则化特性，V cache本身outlier少，均采用channel-wise INT4量化。权重量化沿用OmniQuant的INT4算法。整体量化配置W4AxKV4中>84% GEMM以W4A4执行（利用INT4 tensor core的1248 TOPS），~16%以W4A8执行。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FMPQ算法流程（以LLaMA-1-30B某线性层为例，激活张量[A] shape [B, L, H_in=6656]）：
```
# === Phase 1: 离线校准与Channel Permutation ===
# Step 1: 采样激活（C4子集校准数据，数千token）
calib_acts = run_inference(model, calib_dataset)  # 各层激活张量

# Step 2: 识别outlier channels（magnitude显著大于中位数）
for each layer:
    act = calib_acts[layer]  # shape [B, L, H]
    channel_mag = mean(|act|, dim=(0,1))  # per-channel平均magnitude
    threshold = median(channel_mag) * FACTOR  # 异常检测阈值
    outlier_chs = where(channel_mag > threshold)

# Step 3: Channel Permutation —— 将outlier channels聚集到少数block
perm = compute_perm_grouping(outlier_chs, block_size=128)
# perm重排后: position 0-895 为正常channels, position 896-1023 为outlier channels等
apply_perm_to_weights(W_quant, perm_col_dims)
apply_perm_to_activation_path(layer, perm)  # 插入perm operator到计算图

# === Phase 2: Block-wise量化（每block独立scale） ===
# H_in=6656 → 52 blocks (k=128)
for b in range(0, H_in, 128):
    act_block = A[:, :, b:b+128]
    if contains_outlier(act_block):
        # INT8 block: scale = max(|act_block|) / 127
        A_quant[b:b+128] = quantize_symmetric_INT8(act_block)
    else:
        # INT4 block: scale = max(|act_block|) / 7
        A_quant[b:b+128] = quantize_symmetric_INT4(act_block)

# === Phase 3: KV Cache量化 ===
# K cache: channel-wise INT4（RoPE/softmax正则化outlier）
K_quant = channel_wise_int4_quant(K_cache)
# V cache: channel-wise INT4（outlier少）
V_quant = channel_wise_int4_quant(V_cache)
```
结果：52个block中约44个为INT4（W4A4 GEMM），8个为INT8（W4A8 GEMM），>84%计算以INT4 tensor core执行。WikiText2 PPL增加仅0.04-0.07 vs FP16。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FMPQ实现基于HuggingFace + PyTorch。使用方式：(1) 加载预训练LLM → (2) 对所有权重执行OmniQuant INT4量化 → (3) 使用校准数据(C4子集)进行一次前向推理，收集各层激活 → (4) 分析activation distribution，定位outlier channels → (5) 执行channel permutation（同时更新权重对应列） → (6) 对激活设置block-wise量化参数（per-block scale + precision flag） → (7) KV cache设置channel-wise INT4量化。量化后的模型与COMET-W4Ax kernel配合使用，通过pybind Python接口调用。开源：https://github.com/rhmaaa/COMET-LLM。

涉及论文标题：
- 82-COMET- Towards Practical W4A4KV4 LLMs Serving.pdf

---

## Block-wise Mixed-Precision Quantization for LLM Activations

术语解释
Block-wise混合精度量化是FMPQ的核心量化策略，沿activation tensor的channel维度按block（k=128）划分，对每个block独立判断是否包含outlier来决定量化精度（INT4或INT8），使得硬件计算粒度与量化粒度对齐。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM激活量化面临两个矛盾需求：(1) 量化粒度必须对齐GPU tensor core的最小计算粒度（A100 FP16 TC: 64×64×32 = m×n×k），否则无法高效利用tensor core；(2) 粒度不能过细（如per-channel或per-tile），否则引入过多软件调度开销和scale参数存储。COMET的block-wise设计取k=128作为block大小——这是tensor core最小k维度(32)的4倍整数倍，确保了tensor core利用率；同时每个block仅需1个scale factor和1个precision flag (1-bit)，总开销可以忽略。block大小选择是trade-off：k过小（如32）→ block数多→ scale参数多→ 硬件利用率不够；k过大（如512）→ 单个outlier污染整个大block→ W4A8比例过高。k=128是实验最优。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Block-wise量化伪代码：
```
# 输入：激活张量 A[B, L, H], block_size=128
# 输出：量化张量 A_q + per-block scale + precision flag

num_blocks = H / 128  # e.g. 6656/128 = 52
for b in range(num_blocks):
    A_block = A[:, :, b*128:(b+1)*128]  # shape [B, L, 128]
    
    # 判断精度
    max_abs = max(abs(A_block))
    median_channel_max = median(channel_abs_max(A_block))
    is_outlier_block = max_abs > OUTLIER_THRESHOLD * median_channel_max
    
    if is_outlier_block:
        scale[b] = max_abs / 127.0  # INT8 symmetric
        A_q_block = round(A_block / scale[b])
        A_q_block = clamp(A_q_block, -128, 127)
        precision_flag[b] = INT8
    else:
        scale[b] = max_abs / 7.0  # INT4 symmetric
        A_q_block = round(A_block / scale[b])
        A_q_block = clamp(A_q_block, -8, 7)
        precision_flag[b] = INT4
    
    A_q[:, :, b*128:(b+1)*128] = A_q_block
```
在COMET-W4Ax kernel中，precision_flag数组作为tile descriptor传递给kernel，kernel据此决定每128列使用INT4 mma还是INT8 mma。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Block-wise量化在COMET中通过Python实现（HuggingFace + PyTorch），量化后的tensor以自定义格式存储：INT4 block以每2个4-bit值打包为1 byte存储，INT8 block以1 byte per value存储。两个格式在连续内存中，通过offset + precision_flag区分。运行时，COMET-W4Ax kernel解析precision_flag，将INT4 block对齐INT4 tensor core计算（warp shape 64×64×128），INT8 block对齐INT8 tensor core（warp shape 64×64×64）。该设计的关键约束是block_size必须为tensor core k维度的整数倍。

涉及论文标题：
- 82-COMET- Towards Practical W4A4KV4 LLMs Serving.pdf

---

## Channel Permutation for Outlier Clustering

术语解释
Channel Permutation for Outlier Clustering是FMPQ中用于降低INT8量化比例的技术：通过重新排列激活张量的channel顺序，将原本分散在多个channel的outlier聚集到少数block中，使得仅这些少数block需要高精度(INT8)量化。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM激活中的outlier channel通常散布在整个hidden dimension中。例如LLaMA-1-30B的hidden_size=6656中，假设约100个channel含outlier（~1.5%），这些channel可能分布在6656维的各个位置。若不做排列，52个block中几乎所有block都可能包含至少1个outlier channel，导致几乎所有block都需要INT8量化。Channel permutation通过离线分析outlier channel位置，计算一个排列perm，将outlier channel尽可能聚集到同一block（或少数block），使大部分block为纯正常channel（可安全INT4量化）。排列后需同步调整权重的对应维度（若排列激活的channel维度，则权重对应列也需排列），以及下一层激活的对应维度（维持计算等价性）。COMET论文实验表明permutation可将INT8 block比例从~50%降至~16%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Channel Permutation算法：
```
# 输入：outlier_channels = [ch_12, ch_45, ch_89, ch_230, ...]  (100个下标)
#       total_channels H=6656, block_size k=128, num_blocks=52
# 输出：perm[0..H-1] 排列索引

# Greedy grouping: 将outlier尽量塞入少数block
perm = list(range(H))
outlier_ptr = 0  # 已经放置的outlier

# 将outlier channel移到前面的少数block
# 例：前8个block(block 0-7)各可容纳最多128个channel
# 将100个outlier填入block 0-7，其余channel填满block 8-51
target_block = 0
for ch in outlier_channels:
    dest_pos = target_block * 128 + (outlier_ptr % 128)
    swap(perm, ch, dest_pos)
    outlier_ptr += 1
    if outlier_ptr % 128 == 0:
        target_block += 1

# 剩余正常channel填入其余block
apply_permutation(activation_tensor, perm)  # 重排激活
apply_permutation(weight_matrix[:, :], perm)  # 重排权重对应列
# 如需：在计算图中插入permute operator (仅0.7% runtime overhead)
```
排列后block状态：block 0-7 含outlier → INT8，block 8-51 纯正常 → INT4。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Channel permutation在LLM量化/剪枝中已有先例（如N:M sparsity的Pool & Yu 2021, E-Sparse, RPTQ）。COMET的独特之处是将permutation服务于混合精度量化（而非sparsity）：通过聚集outlier降低INT8比例而非创造structured sparsity pattern。实现层面：离线阶段通过calibration data确定permutation→将permutation作为static transform baked into模型权重（weight列重排）→在线推理时仅需在激活数据路径中插入轻量级permute kernel（0.7% runtime）。该perm operator在CUDA core上以memory-bound方式高效执行，开销极低。

涉及论文标题：
- 82-COMET- Towards Practical W4A4KV4 LLMs Serving.pdf

---

## Outer-Product (OutP) Execution Flow for SpMM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Outer-Product (OutP) 是SpMM的三种基本执行流之一。OutP计算A的每一列与B的对应行（A column k中的所有非零元素与B row k中的所有非零元素）的外积，产生一个partial matrix of C，然后将所有K个partial matrices合并为最终输出矩阵C。数学上：C = Σ_k (A[:,k] ⊗ B[k,:])。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
OutP dataflow（A: M×K, B: K×N, C: M×N）：
```
for k in [0, K):                    // K次迭代（coiteration outermost）
    for each a_i,k ≠ 0 in column A[:,k]:
        for each b_k,j ≠ 0 in row B[k,:]:
            partial_C_k[i,j] += a_i,k × b_k,j
// Produce K partial matrices, then:
C = merge(partial_C_0, partial_C_1, ..., partial_C_{K-1})
```

OutP的三种reuse特性和缺陷：
- **Input reuse优势**：A的每列和B的每行各fetch一次（K次迭代×单列/行），input data traffic低。
- **Index intersection优势**：仅A全零列或B全零行不产生partial matrix，effectual intersection比例接近100%（vs InP的M×N次intersection中仅NNZ次effectual）。
- **Output management劣势**：partial matrices总size通常远超final C（每个A非零 × B行非零产生一个partial元素，multiple partial matrices中同(i,j)重复），merge阶段memory traffic大。
- **Sync瓶颈**：多PE并行产生merge partial matrices时需同步以避免data race。

ACES中OutP的灵感用于column-by-column traversal增强B reuse（类似OutP的input reuse），但通过row-granularity immediate merging解决OutP的partial matrix transfer问题。

术语一般如何实现？如何使用？
实现案例：(1) **SpArch [HPCA 2020]**：使用aggressive condensed matrix representation减少partial matrix数量 + high-radix merger流水线化merge过程；(2) **OuterSPACE [HPCA 2018]**：OutP-based accelerator，将A和B乘法分配至PE array，使用merge network合并partial matrices；(3) **Spaghetti [HPCA 2021]**：FPGA-based streaming OutP accelerator for highly sparse GEMM。选择InP vs OutP vs ROW的判据：OutP适合input reuse重要且merge overhead可控的sparse pattern。对于extremely sparse matrices（density < 0.01%），merge overhead相对可接受；但density增加时partial matrices爆炸。

涉及论文标题：
- 70-ACES- Accelerating Sparse Matrix Multiplication with Adaptive Execution Flow and Concurrency-Aware Cache Optimizations.pdf

## Fiber (Sparse Matrix Row Representation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fiber是SpMM加速器（SPADA、ACES、Gamma等）中对稀疏矩阵行的压缩表示结构：一个fiber是一个sorted list of (coordinate, value) pairs，按coordinate升序排列。每个fiber代表矩阵的一行（B的row fiber）或partial output row（C的partial fiber）。与CSR的区别：CSR将整矩阵的所有行存储在连续数组中（row_ptr + col_idx + values），而fiber将每行作为独立可寻址的sorted list，更灵活但无全局压缩。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Fiber在ACES中的存储和操作：
```
// Fiber存储格式（sorted by coordinate）
struct Fiber:
    elements: list of (coord: int, val: double)  // sorted by coord
    nnz: int = len(elements)

// B matrix: each row = one fiber
B_row_fiber[k] = [(col_0, val_0), (col_1, val_1), ..., (col_n, val_n)]

// C partial output: partial row = one fiber
C_partial_fiber[row_i] = [(col_a, partial_sum_a), (col_b, partial_sum_b), ...]

// Fiber merge (two-pointer walk, used in immediate merging):
def merge_fibers(f1: Fiber, f2: Fiber) -> Fiber:
    result = []
    i, j = 0, 0
    while i < len(f1) and j < len(f2):
        if f1[i].coord == f2[j].coord:
            result.append((coord, f1[i].val + f2[j].val))
            i++; j++
        elif f1[i].coord < f2[j].coord:
            result.append(f1[i]); i++
        else:
            result.append(f2[j]); j++
    result.extend(f1[i:])
    result.extend(f2[j:])
    return result
```

Fiber的cache line粒度假定：由于fiber可能很大（dense row可达数千elements），一个fiber被split为多个cache lines（ACES中cache line size论文未明确说明）。Fiber Density (FD)即为该fiber占用的cache line总数。

术语一般如何实现？如何使用？
Fiber概念在以下加速器中使用：(1) **Gamma [ASPLOS 2021]**：使用fibercache存储B行fibers；(2) **SPADA [ASPLOS 2023]**：使用fiber-based representation for B rows和C partial rows；(3) **ACES**：扩展为B fibers + C partial fibers，PureFiber policy以fiber为单位管理cache替换。对比CSR：fiber提供per-row独立寻址→适合需要随机访问单行的加速器cache；CSR提供全局压缩→适合顺序遍历的CPU/GPU。二者在ACES中互补使用：A用CSR（用于condensing analysis和顺序遍历），B和C用fiber（用于cache管理和merge操作）。

涉及论文标题：
- 70-ACES- Accelerating Sparse Matrix Multiplication with Adaptive Execution Flow and Concurrency-Aware Cache Optimizations.pdf

## Condensed Matrix Representation for SpMM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Condensed Matrix Representation是将稀疏矩阵的非零元素在columns间进行重排（左移），形成denser columns的技术，用于改变SpMM的执行流特性。每个非零元素保留其original column index以维持计算正确性。根据condensing程度分为none/moderate/aggressive三种：condensing程度越aggressive → columns越dense → MPE并行计算时B row reuse越差（因condensed column混合了不同original columns的A元素→请求不同B rows）但sync冲突越少（同row partial fibers减少）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
三种condensing degree对矩阵A transformation（以4×5矩阵A为例）：
```
原始矩阵A (CSR format):
  row0: [(col1, a01), (col3, a03)]
  row1: [(col0, a10), (col2, a12), (col4, a14)]
  row2: [(col1, a21)]
  row3: [(col0, a30), (col3, a33)]

// None condensing (保持CSR column order):
// 按column 0,1,2,3,4顺序遍历非零元素
// col0: a10, a30 → MPE 0: a10×B0, MPE 1: a30×B0 (B0复用)
//   → 但partial fibers for C row1 and row3 → 无sync冲突
// col1: a01, a21 → MPE 0: a01×B1, MPE 1: a21×B1 (B1复用)
//   → 仅2个MPE work, col2-4各仅1元素 → MPE利用率低

// Moderate condensing (columns分两组, 组内左移):
// Group1 (col0-2): 左移 → condensed col0: a10, a01, a21, a12, a30
//   condensed col1: a14 (从col4移入group)
// Group2 (col3-4): 左移 → condensed col2: a03, a33
// col0: 5个非零→5个MPE并行→各自请求different B rows→B reuse低→partial fibers for 5 different C rows→低sync冲突

// Aggressive condensing (全部左移):
// col0: a01, a10, a21, a30, a03, a12, a14, a33 (8非零全集中)
// col1-4: 空
// col0: 8 MPE并行→请求8不同B rows(B0,B1,B3多次)→最低B reuse→8 different partial rows→最低sync冲突
```

术语一般如何实现？如何使用？
实现方式：condensing = 按column index对非零元素重新sort，aggressive condensing等价于忽略原始column index进行全sort；moderate = partition columns→per-partition sort。ACES在band partitioning后按selected condensing degree在线执行（condensing adapter硬件）；SpArch [61]使用aggressive condensing减少partial matrix数量（但论文argue这会"compromise input reuse"）；SPADA的WA window可视为一种局部condensing。condensing的核心trade-off：B reuse vs sync conflict（或generalized: data locality vs parallelism）。

涉及论文标题：
- 70-ACES- Accelerating Sparse Matrix Multiplication with Adaptive Execution Flow and Concurrency-Aware Cache Optimizations.pdf

## Request-Level Parallelism (RLP) and Token-Level Parallelism (TLP) in LLM Decoding

术语解释
PAPI论文定义的量化LLM decoding并行度的两个关键指标：RLP (Request-Level Parallelism) 即batch size，表示并发处理的request数量；TLP (Token-Level Parallelism) 即speculation length，表示单个decoding iteration内每个request并发生成的token数量。二者乘积RLP×TLP直接决定FC kernel的arithmetic intensity。

术语是什么？
RLP和TLP是PAPI为LLM parallel decoding的并行度建模而定义的指标：(1) RLP由batching技术产生——多个request的decoding合并为一个batch同时处理，每个request的当前output token作为输入。"Request-Level"指跨request维度的并行；(2) TLP由speculative decoding产生——draft model预测的多个future tokens在同一decoding iteration被LLM并行验证。"Token-Level"指同一request内跨token维度的并行。PAPI的key insight是：RLP和TLP在runtime会dynamic变化（batch size因mixed continuous batching变化、speculation length因adaptive optimization变化），导致FC kernel的AI = RLP×TLP×h²×2 / [(2×RLP×TLP×h+h²)×2] ≈ RLP×TLP (当h大时) 在runtime变化，进而kernel在memory-bound和compute-bound间切换。

从算法pipeline角度拆解：
RLP和TLP对decoding并行度的影响：
```
给定: batch_size = B, speculation_length = S, hidden_dim = h

// Serial Decoding: RLP=1, TLP=1
for each request r:
    for t in 1..output_len:
        token = decode(model, token)   // 每iteration只产生1个token

// Batching only: RLP=B, TLP=1
for t in 1..max_output_len:
    tokens = [decode(model, req[t]) for req in active_requests]  
    // 每iteration产生B个tokens (1 per request)

// Speculative Decoding only: RLP=1, TLP=S
for t in 1..output_len step S:
    draft_tokens = draft_model.predict(S tokens)  // 小模型predict
    verified = LLM.verify(draft_tokens)            // LLM并行验证S个tokens

// Batching + Speculative Decoding: RLP=B, TLP=S
for t in 1..max_output_len step S:
    // 每iteration产生 B×S 个tokens (最高并行度)
    tokens_matrix = parallel_decode(all_requests, speculation_length=S)

// RLP和TLP的runtime变化
// RLP变化: mixed continuous batching在batch执行中动态加入/完成requests
// RLP_initial = 64 → during execution, 3 requests finish → RLP_current = 61 → ...
// TLP变化: adaptive speculation length optimization [58, 91]
// TLP随GPU utilization动态调整: small batch → increase TLP; large batch → decrease TLP
```

术语一般如何实现？如何使用？
RLP由batch scheduler管理：(1) static batching——batch固定，RLP随requests完成递减(piecewise constant decreasing)；(2) mixed continuous batching——RLP动态变化以maximize resource utilization，token-level scheduling使新request随时加入batch。TLP由speculation controller管理：(1) static——系统预定义speculation length (如4)；(2) dynamic/adaptive——如Draft & Verify [58]在线调整。在PAPI中，RLP通过每decoding iteration后count <|eos|> tokens追踪；TLP通过dedicated register存储，host CPU修改TLP时通知PAPI hardware更新。RLP×TLP用于估算FC kernel arithmetic intensity，作为dynamic kernel-to-hardware scheduling的输入。

涉及论文标题：
- 83-PAPI- Exploiting Dynamic Parallelism in Large Language Model Decoding with a Processing-In-Memory-Enabled Computing System .pdf

## KV Cache Quantization

术语解释
KV Cache Quantization（KV缓存量化）是一种对LLM推理中Key-Value cache进行低精度压缩的技术，通过减少每个KV entry的bitwidth来降低内存容量和带宽压力。Oaken提出online-offline混合量化方案：offline profiling确定per-layer阈值，online per-token执行量化和编码。

术语是什么？
KV Cache Quantization是将LLM decoder layer中缓存的Key和Value向量从FP16（16-bit）压缩到更低精度（如4-bit）的技术。LLM serving中，每个decoder layer需存储所有past tokens的K和V以计算attention。随batch size和context length增大，KV cache成为内存容量和带宽的主要瓶颈。量化技术的核心挑战是：KV值分布存在outliers（magnitude大的异常值），直接uniform量化会导致显著精度损失。现有方案如KIVI [ICML'24]使用per-channel coarse-grained grouping、KVQuant [NeurIPS'24]使用fine-grained per-vector grouping加nuq、QServe [MLSys'25]使用per-vector mixed-precision（INT4+INT8）、Tender [ISCA'24]使用channel reordering+matrix transformation。Oaken通过threshold-based hybrid grouping（outer/middle/inner三组）、group shift quantization（缩小outlier动态范围）和fused dense-and-sparse encoding（8-bit对齐）实现平均4.4-bit量化，相比FP16压缩3.6×，精度损失仅0.87%。

从算法pipeline角度拆解：
Oaken的KV cache量化pipeline（以Llama2-7B attention计算中单token的K/V处理为例）：

```
// Phase 1: Offline Profiling（一次性，~10 min per model）
for each decoder_layer l:
    // 用WikiText2跑~100次inference，收集所有token的K/V值
    kv_samples[l] = collect_KV_from_n_inferences(n=100, dataset=WikiText2)
    // 按magnitude排序，确定分组阈值
    sorted_vals = sort_by_magnitude(kv_samples[l])
    T_hi[l] = sorted_vals[top_4_percent_idx]    // Outer group上界
    T_lo[l] = sorted_vals[bottom_6_percent_idx]  // Inner group上界
    // Outer group: top 4%（outliers）
    // Middle group: 90%（主要数据，4-bit INT4量化）
    // Inner group: bottom 6%（近零值，sparse→0）

// Phase 2: Online Per-Token Quantization
def quantize_token_kv(v, layer_l):
    // Step 1: Group assignment
    outer_mask = (abs(v) > T_hi[layer_l])
    inner_mask = (abs(v) < T_lo[layer_l])
    mid_mask = ~(outer_mask | inner_mask)
    
    // Step 2: Group Shift Quantization for outer group
    v_outer = v[outer_mask]
    shift = mean(v_outer)  // 缩小动态范围
    v_outer_shifted = v_outer - shift
    scale_out = max(abs(v_outer_shifted)) / 15  // 2^4-1 = 15 for INT5
    v_outer_quant = round(v_outer_shifted / scale_out)  // INT5, range [-15,15]
    
    // Step 3: Middle group 4-bit quantization
    v_mid = v[mid_mask]
    scale_mid = (T_hi[layer_l] - T_lo[layer_l]) / 15
    v_mid_quant = round((v_mid - T_lo[layer_l]) / scale_mid)  // INT4, range [0,15]
    
    // Step 4: Inner group → zero (sparse)
    v_inner = 0  // v[inner_mask] → 0, ~10% sparsity
    
    // Step 5: Fused Dense-and-Sparse Encoding
    // 编码为8-bit对齐格式:
    // [6-bit idx | 1-bit group_flag | 1-bit sign or 5-bit val]
    encoded = fuse_encode(v_mid_quant, v_outer_quant, sparsity_mask)
    // average bitwidth: 4.4 bit/entry (vs FP16 16 bit/entry)
    return encoded

// Phase 3: Dequantization during Attention
def dequantize_for_attention(encoded_kv, layer_l):
    decoded_fp16 = []
    for entry in encoded_kv:
        group_flag = entry.group_flag
        if group_flag == MIDDLE:
            val = entry.val_4bit * scale_mid[layer_l] + T_lo[layer_l]
        elif group_flag == OUTER:
            val = entry.val_5bit * scale_out[layer_l] + shift[layer_l]
        else:  // SPARSE/INNER
            val = 0.0
        decoded_fp16.append(val)
    return decoded_fp16  // 送入MPU计算attention
```

术语一般如何实现？如何使用？
KV Cache Quantization的实现分三个层面：(1) 软件层面——GPU上通过PyTorch实现量化/反量化kernel（如Oaken的公开accuracy evaluation代码，https://github.com/casys-kaist/oaken），在attention计算前对K/V值进行量化并替换原始FP16值；(2) 硬件层面——专用加速器（如Oaken/HyperAccel LPU）集成Quant Engine和Dequant Engine硬件模块，在memory data path上实时执行量化/反量化，与compute/memory传输pipeline重叠；(3) 系统层面——Serving框架（如vLLM）集成量化aware的KV cache内存管理，量化后的KV cache以更少物理内存存储更多tokens的K/V。量化参数（thresholds, scales）通过offline profiling预先计算并随模型加载。

Vector Quantization (VQ) applied to KV cache采用不同的方法：CQ（Coupled Quantization）[69]对每个head的K/V向量按vector_size切分为子向量→k-means聚类→用codebook index替换子向量，将KV cache压缩到1-4 bit等价。Dequantization时按codebook index查codebook获取FP16 centroid→accumulate残差→concatenate sub-spaces恢复原始精度。VQ比element-wise量化在同等bitwidth下accuracy更高，但dequantization latency成为实际部署的瓶颈——VQ-LLM通过codebook cache分层放置、codebook-centric dataflow和hierarchical fusion优化此问题。

ALISA（ISCA'24）采用更简单的fine-grained channel-wise INT8量化压缩KV tensor，用于减少CPU-GPU offload带宽而非attention计算本身。ALISA的KV compression使用标准对称量化公式：x_quant = round(x/λ + z)，其中scale λ = (max-min)/(2^b-1)，zero point z = round(-2^b·min/(max-min))。选择INT8而非更低位宽（如INT4）是为确保跨模型泛化——论文指出OPT模型可承受INT4但其他模型可能不稳定。Quantize在KV tensor存储到CPU内存时执行，Dequantize在KV从CPU reload到GPU用于计算前执行。量化后CPU存储和PCIe传输量减半，实验显示精度影响可忽略（ALISA vs SWA准确率几乎完全track）。与Oaken的混合分组量化不同，ALISA不区分outlier/middle/inner组，采用均匀的per-channel量化。

涉及论文标题：
- 86-Oaken- Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization.pdf
- 2-VQ-LLM_High-performance_Code_Generation_for_Vector_Quantization_Augmented_LLM_Inference.pdf
- 72-ALISA_Accelerating_Large_Language_Model_Inference_via_Sparsity-Aware_KV_Caching.pdf
- 82-COMET- Towards Practical W4A4KV4 LLMs Serving.pdf

COMET的KV cache量化方法：不同于Oaken的三组（outer/mid/inner）精细分组量化，COMET对KV cache采用更简洁的channel-wise INT4量化策略。依据：(1) K cache的RoPE旋转位置编码和attention softmax具有强outlier正则化特性，使得4-bit量化对K引入极小误差；(2) V cache本身outlier少，适合直接4-bit量化。实验表明该策略对WikiText2 PPL影响仅0.05（平均值），且无需Oaken那样的online per-token量化解码开销。COMET的KV cache量化与输入激活的FMPQ量化互补：输入激活用block-wise mixed INT4/INT8以对齐GPU计算粒度（compute-bound optim），KV cache用纯channel-wise INT4（memory-bound optim，无需考虑tensor core计算对齐）。

- 89-AQPIM_Breaking_the_PIM_Capacity_Wall_for_LLMs_with_in-Memory_Activation_Quantization.pdf

AQPIM的KV cache量化方法：采用Product Quantization (PQ)进行clustering-based vector量化，与上述element-wise量化方案根本不同。(1) 将d_head维KV向量分解为m=32个子向量（subvectors）；(2) 每个子向量空间独立执行importance-weighted k-means聚类（K=512 centroids，4 iterations）；(3) 用centroid index（log₂512=9 bit/subvector）替换FP16子向量，总压缩≈6.53×；(4) 聚类在prefilling阶段PIM上在线完成（GPU-PIM并行），decode阶段append新token indices；(5) 不进行dequantization——attention计算直接在compressed data上用codebook lookup+summation完成（PQ-based attention）。与COMET/Oaken等GPU element-wise量化不同，AQPIM是PIM-native设计：无需INT MAC硬件（复用已有FP16 MAC），利用PIM内部高带宽处理online clustering，通过intra-row indirection硬件解决lookup随机访问问题。

---

## Vector Quantization (VQ) for LLM Compression

术语解释
Vector Quantization（向量量化）是一种将多个数据元素组成向量并作为整体压缩单元进行量化的技术。与传统的element-wise量化（每个元素独立压缩）不同，VQ利用元素间的跨维度信息（cross-dimension information），通过聚类将相似向量映射到codebook centroid，能在更低bitwidth下保持reconstruction quality。

术语是什么？
VQ将原始向量空间划分为多个sub-space，在每个sub-space内对子向量执行k-means聚类，用聚类中心（centroid/codebook entry）代表附近的子向量。VQ的核心参数包括：(1) **Vector size**：一次量化处理的元素数量（如2、4、8）；(2) **#Entry**：每个codebook中的聚类中心数量（如256、4096）；(3) **Residual**：残差量化的轮数（1或2），每轮对上一轮的残差再次聚类以提高reconstruction quality。VQ的配置通常标注为VQ<vector_size, #Entry, Residual>，如VQ<4,8,1>表示vector_size=4、8个entries、1轮量化（无残差）。不同VQ算法选择不同的配置和codebook组织方式：Product Quantization (PQ)将向量空间划分为独立sub-space各自训练codebook；Additive Quantization (AQ)对所有sub-space使用加性codebook；Residual Quantization迭代量化残差。VQ-LLM论文调查的SOTA VQ算法包括：QuiP#-4（lattice-based codebook, vector_size=8, 65536 entries, Residual=2）、AQLM-3（additive quantization, vector_size=8, 4096 entries, Residual=2）、GPTVQ-2（product quantization, vector_size=4, 256 entries, Residual=1）、CQ-4/CQ-2（coupled quantization for KV cache, vector_size=2/4, 256 entries, Residual=1）。

从算法pipeline角度拆解：
以VQ<4,2,2>（vector_size=4, #Entry=2, Residual=2）为例：

```
// Phase 1: Offline Quantization（权重或KV cache）
def vector_quantize(tensor_2d, vector_size=4, n_entries=2, residual=2):
    // tensor_2d: [num_rows, d]，如weight matrix [4096, 4096]
    num_subspaces = d // vector_size  // = 1024 subspaces for d=4096, vs=4
    codebooks = []  // list of [n_entries, vector_size] per subspace × residual
    
    for subspace in range(num_subspaces):
        sub_vectors = tensor_2d[:, subspace*vector_size : (subspace+1)*vector_size]  // [num_rows, 4]
        cb_subspace = []
        residual_data = sub_vectors
        
        for r in range(residual):
            // k-means clustering
            centroids, assignments = kmeans(residual_data, n_entries)  // centroids: [n_entries, 4]
            cb_subspace.append(centroids)  // codebook for this residual round
            // compute residual for next round
            reconstructed = centroids[assignments]  // [num_rows, 4]
            residual_data = residual_data - reconstructed
        
        codebooks.append(cb_subspace)  // [residual, n_entries, vector_size]
    
    // Quantized data: [num_rows, num_subspaces, residual] of log2(n_entries)-bit indices
    quantized_indices = ...
    return quantized_indices, codebooks

// Phase 2: Online Dequantization
def vector_dequantize(quantized_indices, codebooks):
    // quantized_indices: [num_rows, num_subspaces, residual]
    // codebooks: [num_subspaces, residual, n_entries, vector_size]
    output = zeros(num_rows, d)
    for subspace in range(num_subspaces):
        reconstructed_sub = zeros(num_rows, vector_size)
        for r in range(residual):
            idx = quantized_indices[:, subspace, r]  // [num_rows]
            centroid = codebooks[subspace][r][idx]    // [num_rows, vector_size]
            reconstructed_sub += centroid             // accumulate residuals
        output[:, subspace*vector_size : (subspace+1)*vector_size] = reconstructed_sub
    return output  // FP16 precision
```

术语一般如何实现？如何使用？
VQ在LLM中的应用有两种方式：(1) **Weight-only VQ**：离线对预训练权重进行VQ压缩（QuiP#、AQLM、GPTVQ），推理时在计算前dequantize权重。量化在模型加载时完成，无运行时overhead。(2) **KV cache VQ**：对每个新生成token的K/V在线量化（CQ），或prefill后对全部KV cache一次性量化，将FP16 KV cache替换为压缩索引+codebook。运行时需dequantization才能用于attention计算。

VQ的核心优势：比element-wise量化更好的accuracy-compression tradeoff——同等4-bit下accuracy更高，或同等accuracy下可压缩到2-bit甚至1-bit（如CQ-2实现12.5% compression ratio, KV cache压缩至1/8）。VQ的核心挑战：dequantization引入额外的codebook lookup和accumulation操作→精心设计的fused kernel（如VQ-LLM）是将memory reduction转化为实际latency improvement的关键。

涉及论文标题：
- 2-VQ-LLM_High-performance_Code_Generation_for_Vector_Quantization_Augmented_LLM_Inference.pdf

---

## Codebook (VQ Codebook / Quantization Codebook)

术语解释
Codebook（码本）是向量量化（VQ）中存储聚类中心（centroid/quantization point）的数据结构。每个codebook包含一组高维向量（entries），代表对应sub-space中所有可能的量化值。Dequantization时，通过量化索引在codebook中查表（lookup）获取对应的FP16精度centroid向量，重建原始数据。

术语是什么？
Codebook是VQ算法的核心数据结构，由k-means聚类生成的centroid向量集合组成。结构上，codebook是一个多维tensor：shape = [num_subspaces, residual_rounds, n_entries, vector_size]，其中n_entries决定了每个索引的bitwidth（=log2(n_entries)）。例如VQ<4,256,1>配置下，每个codebook含256个4维FP16 centroid向量，每个索引占8 bit。在LLM VQ压缩场景中，不同算法选择不同的codebook组织粒度：(1) GPTVQ-2为不同channel组训练独立的codebook（per-channel codebook，易导致重复加载问题）；(2) QuiP#使用一个全局lattice codebook（65536 entries但每次仅查256个）；(3) CQ系列为每head的每group channels训练独立的codebook。Codebook大小变化范围大：QuiP#-4约2KB/block，AQLM-3约128KB/block，GPTVQ-2约32KB/block，CQ-2约64KB/block。

从算法pipeline角度拆解：
Codebook在dequantization流程中的角色（以VQ<4,256,1> attention KV dequantization为例）：

```
// 输入: quantized_kv_index [num_tokens, num_heads, num_subspaces]  (3-bit indices)
//       codebooks [num_subspaces, n_entries=256, vector_size=4]   (FP16)
// 输出: dequantized_kv [num_tokens, num_heads, head_dim]         (FP16)

for token in range(num_tokens):
    for head in range(num_heads):
        for subspace in range(num_subspaces):
            index = quantized_kv_index[token, head, subspace]     // 获取codebook索引
            entry = codebooks[subspace][index]                      // 查codebook获取centroid [4]
            dequantized_kv[token, head, subspace*4:(subspace+1)*4] = entry  // 填充FP16值
```

术语一般如何实现？如何使用？
Codebook在GPU上的存储由VQ-LLM的Codebook Cache管理：按access frequency将entries分层放置在register（hot）、shared memory（medium）和global memory（cold）中，通过reorder-based static mapping实现（offline按frequency排序并重编号索引，runtime通过简单index比较定位）。此分层放置解决了naive全量shared memory存储导致的SM利用率下降（large shared memory footprint）和bank conflict（entries > bank数）问题。

涉及论文标题：
- 2-VQ-LLM_High-performance_Code_Generation_for_Vector_Quantization_Augmented_LLM_Inference.pdf
- 89-AQPIM_Breaking_the_PIM_Capacity_Wall_for_LLMs_with_in-Memory_Activation_Quantization.pdf

AQPIM的codebook用法与VQ-LLM不同：AQPIM的codebook驻留在HBM-PIM的bank内存中（per-subvector per-head），不做GPU式的分层cache。codebook在prefilling阶段由BankPE+BufferPE在线生成（k-means 4 iterations→centroids存储在固定codebook区域）；decode阶段BankPE直接查codebook执行ATNK（query×codebook子矩阵乘）和ATNV（intra-row indirection lookup），无需将codebook传输到GPU。codebook大小：m×K×d/m×2B = 32×512×4×2 = 128KB per head（d_head=128时），总codebook footprint远小于原始KV cache。Page-aware windowed clustering保证任一窗口内的codebook子集（512 entries × 4 bytes = 2KB for key + 2KB for value）fit在单个DRAM row buffer中。

---

## Online-Offline Hybrid Quantization

术语解释
Online-Offline Hybrid Quantization是Oaken提出的KV cache量化范式：offline阶段一次性profiling确定per-layer量化参数（thresholds, scales），online阶段利用预计算参数对每个新token实时执行量化，避免online profiling的运行时开销。

术语是什么？
Online-Offline Hybrid Quantization将KV cache量化分解为两个阶段：(1) Offline Profiling——在模型部署前，用代表性数据集（如WikiText2）运行~100次inference，统计每个decoder layer的KV值分布（magnitude分布、outlier比例），计算最优的量化参数（分组阈值T_lo/T_hi、各组scale factors），保存为per-layer quantizer配置（JSON文件）。此过程为一次性开销（~10分钟/模型），结果可跨输入数据集复用（因为Oaken的Insight 2：KV分布跨输入数据集一致）；(2) Online Quantization——serving时，每个new token的K/V值实时流经Quant Engine，使用预计算的per-layer阈值执行分组、量化和编码操作，写入KV cache。该方案解决了纯online profiling方案（如QServe）的运行时overhead问题——无需对每个token做profiling来决定量化策略，所有策略参数已预计算。

从算法pipeline角度拆解：
Online-Offline Hybrid的工作流程：

```
// Offline阶段（部署前，一次性）
offline_profile(model, calibration_dataset):
    for each layer l in model.decoder_layers:
        samples = run_n_inferences(model, calibration_dataset, n=100)
        kv_distribution = analyze_magnitude_distribution(samples[l])
        
        // 按预设比例确定分组阈值
        T_hi[l] = percentile(kv_distribution, 96)   // top 4%
        T_lo[l] = percentile(kv_distribution, 6)    // bottom 6%
        
        // 预计算各组的scale（基于最大-最小值）
        scale_out[l] = (max_out - min_out) / 15
        shift_out[l] = mean_out
        scale_mid[l] = (T_hi[l] - T_lo[l]) / 15
    
    save_quantizer({layer_l: {T_lo, T_hi, scale_out, shift_out, scale_mid}})

// Online阶段（serving时，per-token）
online_quantize(v_kv_new, layer_l, quantizer):
    // 直接使用预计算参数，无profiling开销
    thresholds = quantizer[layer_l]
    groups = classify_by_thresholds(v_kv_new, thresholds.T_lo, thresholds.T_hi)
    quantized = apply_quantization(groups, thresholds.scales)
    encoded = fuse_encode(quantized)
    write_to_kv_cache(encoded)
```

术语一般如何实现？如何使用？
Online-Offline Hybrid Quantization的实现：(1) Offline profiling脚本（如`oaken_preprocess_activation.py`）在GPU上运行，用calibration dataset做若干次完整inference，收集所有layer的KV值→统计分析→输出JSON/Pickle格式的quantizer配置；(2) Online阶段在serving系统中加载quantizer配置→每个token生成时调用量化函数→量化结果写入KV cache。该范式的关键前提是"KV分布跨输入数据集一致"的insight——若此前提不成立（如不同domain数据导致分布显著偏移），则需要per-dataset profiling或多套quantizer动态切换。Oaken验证了在WikiText2上profiling得到的阈值可用于PIQA/WinoGrande/HellaSwag等不同数据集的inference，精度保持良好。

涉及论文标题：
- 86-Oaken- Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization.pdf

---

## Group Shift Quantization

术语解释
Group Shift Quantization是Oaken提出的一种量化技术，通过将outlier组的值减去组均值（shift）来缩小其动态范围，使其能用更少的bit（INT5代替FP16）表示，从而降低整体平均bitwidth。

术语是什么？
在KV cache量化中，outlier（magnitude异常大的值）是量化精度的主要挑战：uniform量化需要为整个范围的outlier分配足够bitwidth，但outlier仅占约4%的值，导致大部分inlier的量化精度浪费。Group Shift Quantization的核心思路：将outlier组的值整体shift（减去组均值），使shift后的值分布在0附近（范围远小于原始值），从而能用INT5（5-bit, 范围[-15,15]）而非FP16（23-bit）表示。shift和scale作为per-group metadata存储（6-bit scale + 1-bit sign）。该操作将outlier的bitwidth从23降至12（5-bit index + 6-bit scale + 1-bit sign），同时将整体average bitwidth从5.9降至4.8。

从算法pipeline角度拆解：
Group Shift的量化/反量化计算（以outer group的outlier处理为例）：

```
// Quantization（写入KV cache时）
quantize_outer_group(v_outer):
    shift = mean(v_outer)                          // 计算组均值
    v_shifted = v_outer - shift                     // 减去均值，缩小范围
    scale = max(abs(v_shifted)) / 15               // 量化scale（INT5范围[-15,15]）
    v_quant = round(v_shifted / scale)              // 量化到[-15,15]
    // 存储: [5-bit v_quant | 6-bit scale | 1-bit sign] = 12 bit/entry
    return v_quant, scale, shift

// Dequantization（读取KV cache时）
dequantize_outer_group(v_quant, scale, shift):
    v_fp16 = v_quant * scale + shift               // 反量化：恢复原始scale+shift
    return v_fp16
```

对比无shift的量化：v_quant = round(v_outer / scale)，scale = max(|v_outer|) / 15。由于v_outer范围大（含outlier magnitude），scale也大→量化分辨率粗→精度损失大。Group Shift使v_shifted范围约缩小|shift|/max(|v_outer|)倍→scale同比例缩小→量化分辨率提升。

术语一般如何实现？如何使用？
Group Shift Quantization实现为量化函数的一部分：(1) 在Quant Engine硬件中，Shift单元就是一个减法器，在Quantizer之前执行v - shift操作；(2) shift值在offline profiling时预计算（= outer group的均值），与scale和thresholds一起存入quantizer配置；(3) 反量化时，Dequant Engine中的Dequantizer执行val * scale + shift。该技术的代价是额外存储shift值（per-group, negligible vs per-entry data），但由于大幅减少outlier bitwidth（23→12），净收益显著（overall bitwidth 5.9→4.8）。

涉及论文标题：
- 86-Oaken- Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization.pdf

## LUT-NN (Lookup Table-based Neural Network)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LUT-NN是将深度神经网络中计算密集的矩阵乘法（GEMM）替换为基于查找表（Lookup Table, LUT）操作的一种算法范式。核心洞见：对于给定的DNN layer，不同input activation矩阵的features存在block-wise semantic similarity——少量典型特征（centroids/质心）即可近似原始激活值。基于此，GEMM可转换为centroid index search和LUT accumulation两步。

LUT-NN操作分两阶段：
1. **Conversion（转换）**：将预训练模型的weight matrix (F×H) 转换为codebooks和pre-computed Lookup Tables。流程：activation矩阵沿H dim分割为1×V sub-vectors → 对每column做K-means聚类（CT个centroids per column, 共VH个codebooks）→ weight matrix按相同方式拆分 → 与codebooks做inner-product → 预计算出CT个LUTs（每个F×VH大小）。
2. **Inference（推理）**：输入activation (N×H) 分割为1×V tiles → Closest Centroid Search (CCS)：每个tile与对应codebook计算L2-distance寻找最近centroid → 得到index matrix (N×VH) → Table Lookup：根据index从LUT中fetch pre-computed partial-sums (F elements per index) → Reduce：按行accumulate所有VH columns得到F×N输出矩阵。

数学表达：GEMM FLOPs = 2×N×H×F（一半为乘法），LUT-NN FLOPs ≈ 3×N×H×CT + N×F×VH。CT（centroid数，通常4-16）远小于F，multiplication仅占总操作的2.9%~14.3%（PIM-DL论文数据）。LUT-NN将computation-intensive GEMM转换为memory-intensive LUT lookup，FLOPs reduction达3.66×~18.29×。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以BERT-base的FFN1 layer（input N=32768, H=768, F=3072, V=4, CT=16）为例：

```
# ===== Conversion Phase (offline, per model, once) =====
# Input: pre-trained weight matrix W (F×H=3072×768), calibration data
# V=4 → VH = H/V = 192 columns

# Step 1: Centroid Clustering
for col_j in range(VH=192):
    sub_vectors = activation[:, col_j*4:(col_j+1)*4]  # collect all 1×4 sub-vectors
    codebooks[col_j] = K-Means(sub_vectors, k=CT=16)  # 16 centroids, each 1×4
    
# Step 2: LUT Pre-computation
for k in range(CT=16):  # for each centroid index
    LUT[k] = zeros(F=3072, VH=192)
    for col_j in range(VH=192):
        w_sub = W[:, col_j*4:(col_j+1)*4]  # F×4 weight sub-vector
        centroid = codebooks[col_j][k]      # 1×4 centroid vector
        LUT[k][:, col_j] = w_sub @ centroid.T  # F×1 inner-product result
# Output: CT=16 LUTs (each 3072×192), VH=192 codebooks (each 16×4)
```

```
# ===== Inference Phase =====
# CCS operator (host-side, small GEMM for index generation):
# A_tiles = reshape(A, [N=32768, VH=192, V=4])
for col_j in range(VH=192):
    # 32768×4 @ 4×16 = 32768×16 (distance for each centroid candidate)
    dist = A_tiles[:, col_j, :] @ codebooks[col_j].T
    Index[:, col_j] = argmin(dist, dim=1)  # store closest centroid id

# LUT operator (PIM-side, memory-intensive lookup + accumulate):
for row_i in range(N=32768):
    output[row_i] = zeros(F=3072)  # output vector
    for col_j in range(VH=192):
        k = Index[row_i, col_j]  # centroid id ∈ [0, CT)
        output[row_i] += LUT[k][:, col_j]  # fetch F-dim partial sum, accumulate
# Output: F×N = 3072×32768 matrix
```

原始LUT-NN [Tang et al., MobiCom'23]的局限：(1) 无法全层替换——仅能替换BERT-base 6/12 layers，全部替换时GLUE avg从79.0降至35.5；(2) 需100% training set做calibration；(3) 使用Gumbel-Softmax梯度估计导致收敛慢。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LUT-NN的实现和使用：
1. **实现**：Calibration在PyTorch中实现——centroid clustering可用faiss加速K-Means，index search可用CuBLAS/MKL/GGML实现为small GEMM。LUT lookup在特殊硬件上需custom kernel（如PIM-DL Engine）。LUT数据常量化为INT8以压缩存储（≤0.1% accuracy drop）。
2. **适用场景**：所有含linear layer的DNN（Transformer、CNN、MLP）。对Transformer效果最优——因为FFN层（最大inner dim）和projection层（QKV/O）包含最大的GEMM操作，且占end-to-end延迟>85%。计算受限硬件（DRAM-PIM、移动CPU、MCU）受益最大，因为LUT-NN将计算转换为内存查找——match高带宽、弱计算硬件的特征。
3. **与Product Quantization (PQ)关系**：LUT-NN受PQ启发（PQ常用于approximate nearest neighbor search中使用centroids近似向量），将PQ的centroid概念扩展为end-to-end differentiable learning。

涉及论文标题：
- 20-PIM-DL- Expanding the Applicability of Commodity DRAM-PIMs for Deep Learning via Algorithm-System Co-Optimization.pdf

## eLUT-NN Calibration (enhanced LUT-NN)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
eLUT-NN是PIM-DL在baseline LUT-NN基础上提出的增强型校准算法，解决三大缺陷：(1) 全层替换精度大幅下降；(2) 需要大量calibration data；(3) 收敛慢。

核心机制：
1. **Reconstruction Loss（重构损失）**：将每层computation approximation error直接加入overall loss：L = Model Loss + β Σ_{l∈L} ||Â_l W - A_l W||²。其中Â_l = H(A_l)是用最近centroid替换sub-vectors后的近似激活矩阵。(a) 通过direct gradient propagation到centroids（替代layer-by-layer backprop），克服gradient vanishing；(b) 将approximation error融入loss，使centroids学习更准确的激活表征。
2. **Straight Through Estimator (STE)**：由于centroid cluster和table-lookup操作非连续可微，使用STE估算梯度∂L/∂F ≈ (∂L/∂ŷ) · (∂Â/∂F)，将∂Â/∂A赋值为identity matrix以"直通"梯度。替代baseline的Gumbel-Softmax，确保更快收敛。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# eLUT-NN Calibration (PyTorch-based)
# Input: pretrained model M (weights frozen), <1% training data
# Hyperparams: V=4, CT=16, β=1e-3(BERT)/1e-4(ViT), LR=5e-5

# Initialize centroids randomly
for layer l in M.linear_layers:
    centroids_l = nn.Parameter(randn(CT=16, VH=192, V=4))
    
# Calibration loop (≤100K iterations, <1% data)
for batch_x, batch_y in calibration_loader:
    A_cur = batch_x
    rec_loss = 0
    
    for layer l in M.layers:
        if l is linear_layer:
            # Step 1: Split activation into 1×V tiles
            A_tiles = A_cur.reshape(batch, VH, V)
            
            # Step 2: Find closest centroid (STE forward)
            dist = A_tiles @ centroids_l.T  # batch×VH×CT
            indices = argmin(dist, dim=-1)  # hard selection
            
            # Step 3: Replace with nearest centroid (STE backward)
            A_hat = centroid_gather(centroids_l, indices)  # approximate A
            
            # Step 4: Reconstruction loss accumulation
            rec_loss += ||A_hat @ weight[l] - A_cur @ weight[l]||²
            
            # Step 5: LUT-NN forward for next layers
            A_cur = LUT_forward(A_cur, centroids_l, LUTs_precomputed_l)
        else:
            A_cur = l(A_cur)  # non-linear layers unchanged
    
    # Total loss and gradient update
    total_loss = model_loss(model_output, batch_y) + β * rec_loss
    total_loss.backward()  # STE gradients flow through centroid_gather
    optimizer.step()  # only centroids updated, weights frozen
```

关键结果（PIM-DL论文）：
- BERT-base GLUE avg: 79.0→76.9 (-2.1%) vs LUT-NN baseline 35.5
- BERT-large GLUE avg: 81.5→79.3 (-2.7%) vs LUT-NN baseline 36.8
- ViT-base CIFAR-100: 91.4→89.1 vs LUT-NN baseline 1.07
- 仅需~0.78% of training tokens, 收敛≤100K iterations

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
eLUT-NN的实现和使用：
1. **实现**：在PyTorch中实现为custom training loop。STE通过torch.autograd.Function实现——forward执行argmin，backward返回identity gradient。Reconstruction Loss作为额外loss term。centroids作为nn.Parameter使用Adam/SGD更新。weights在整个calibration过程中frozen。
2. **数据效率**：vs LUT-NN baseline需100% training set，eLUT-NN仅需<1%（random sampled）。calibration tokens from dataloader provided by PyTorch。
3. **使用场景**：配合PIM-DL Inference Engine将校准后的LUT-NN模型部署到DRAM-PIM上。calibration是offline一次性流程（~几分钟到几小时取决于模型大小），inference时使用的model structure完全不同（GEMM→LUT lookup+accumulation）。

涉及论文标题：
- 20-PIM-DL- Expanding the Applicability of Commodity DRAM-PIMs for Deep Learning via Algorithm-System Co-Optimization.pdf

---

## Non-GEMM Operations (in Neural Networks)

术语解释
Non-GEMM Operations指深度神经网络中不属于GEneral Matrix Multiplication（GEMM）类别的所有算子。传统NPU设计将>99%的注意力集中在GEMM算子（Conv/MatMul）优化上，将非GEMM算子视为"次要"或通过简单fallback处理。但随着DNN演进（从VGG-16仅3类非GEMM算子到BERT/GPT-2的超10类），非GEMM算子的多样性和运行时占比急剧增加，成为端到端推理的主要瓶颈（在EfficientNet上，GPU执行非GEMM占73% runtime）。

术语是什么？
非GEMM算子根据ONNX实现可分为五大类：(1) **Element-wise Mathematical**：Add, Sub, Mul, Exp, Sqrt, Floor, Ceil, Pow, Reciprocal等逐元素数学运算；(2) **Element-wise Activation**：ReLU, LeakyReLU, Clip, Tanh, Sigmoid, GeLU等激活函数；(3) **Reduction-based**：Depth-wise Conv, MaxPool, GlobalAveragePool, ReduceMean, Softmax等归约操作；(4) **Data Layout Transformation**：Transpose, Reshape, Concat等张量布局变换；(5) **Datatype Cast**：数据类型转换（FXP32↔FXP16↔FXP8↔FXP4），以及BitShift。非GEMM算子的特征：(a) 绝大多数是memory-bound（roofline分析显示除Softmax/GeLU外均落入memory-bound区域），与GEMM的compute-bound性质相反；(b) 访问模式多样——element-wise是一对一映射，reduction是多对一映射；(c) 循环结构在编译时完全确定（静态shape）；(d) 以嵌套循环+基本算术/逻辑原语结构实现。

从算法pipeline角度拆解：
以BERT inference中一个Transformer block的非GEMM算子执行序列为例：
```
# 输入: hidden_states [batch, seq_len, hidden_dim]
# 1. LayerNorm (非GEMM: ReduceMean + Sub + Div + Mul + Add)
mean = ReduceMean(hidden_states, axis=-1)  # reduction
centered = Sub(hidden_states, mean)         # element-wise
var = ReduceMean(Mul(centered, centered))   # reduction + element-wise
normed = Div(centered, Sqrt(var + eps))     # element-wise + sqrt
ln_out = Add(Mul(normed, gamma), beta)      # element-wise

# 2. 经过GEMM (Q/K/V projection) 后

# 3. Softmax (非GEMM: Exp + ReduceSum + Div)
exp_scores = Exp(attention_scores)           # element-wise
sum_exp = ReduceSum(exp_scores, axis=-1)    # reduction
attn_weights = Div(exp_scores, sum_exp)     # element-wise

# 4. 经过GEMM (output projection + FFN) 后

# 5. GeLU (非GEMM: 可分解为基本原语)
# GeLU(x) ≈ 0.5x(1 + tanh(sqrt(2/π)(x + 0.044715x³)))
# 分解：5×Mul + 3×Add + Sign + Abs + Min
```

术语一般如何实现？如何使用？
非GEMM算子在现有系统中的处理方式：(1) **Dedicated hardware blocks**（如NVDLA、Eyeriss）：为ReLU/MaxPool等常见算子设计专用硬件，不支持的算子fallback到CPU；(2) **GPU CUDA kernels**：每种非GEMM算子手写CUDA kernel或通过cuDNN/TensorRT融合到GEMM kernel中（如fused Add+ReLU）；(3) **CPU fallback**：通过ONNX Runtime或框架runtime在CPU上执行（如Intel MKLDNN的element-wise和reduction原语）；(4) **Integer-only decomposition**（I-BERT方法）：将浮点非GEMM算子（GeLU, Softmax, LayerNorm等）分解为整数基本运算序列，消除浮点-整数转换需求。Tandem Processor采用方案(4)——使用INT32 ALU组合执行所有非GEMM算子，通过编译器自动完成分解映射。

涉及论文标题：
- 21-Tandem Processor: Grappling with Emerging Operators in Neural Networks

---

## INT32 Integer-only Inference for Non-GEMM Operators

术语解释
INT32 Integer-only Inference for Non-GEMM是指在神经网络推理中使用INT32整数精度执行所有非GEMM算子，避免浮点运算和与GEMM unit（通常使用INT8/INT32）之间的数据类型转换开销。该方法基于I-BERT等先工作的观察：整数算术可用于CNN和Transformer的推理执行而无精度损失。

术语是什么？
Tandem Processor的ALU固定为INT32精度，核心理由：(1) GEMM unit的累加器通常已经是INT32精度（INT8乘法后累加为INT32），Tandem Processor以INT32读取GEMM输出无需类型转换；(2) 部分非GEMM算子（如ResAdd、Softmax内部累加）需要INT32精度（INT8不足）；(3) 统一的INT32精度消除了非GEMM→GEMM方向的类型转换需求，仅需GEMM→非GEMM方向的INT32→INT8转换（通过DATATYPE_CAST指令支持FXP4/8/16/32）。复杂浮点非GEMM算子（GeLU、Softmax、Sqrt、Exp等）被编译器自动翻译为INT32整数基本原语组合。例如GeLU使用5次乘法+3次加法+sign+absolute+minimum的INT32运算序列实现，基于gemmlowp和I-BERT的整数近似方法。

从算法pipeline角度拆解：
以GPT-2中Softmax的INT32 integer-only实现为例（参考I-BERT方法）：
```
# 浮点版: softmax(x_i) = exp(x_i) / sum(exp(x))
# INT32版（伪代码）：
# Step 1: 找最大值（防止exp溢出）
x_max = INT32_MAX(x, axis=-1)          # reduction: max
# Step 2: 减去最大值（稳定数值）
x_shifted = INT32_SUB(x, x_max)        # element-wise
# Step 3: 整数近似exp（查表或多项式）
# 使用i-exp: exp_int ≈ 2^(x * log2(e))
exp_approx = INT32_EXP_APPROX(x_shifted)  # 查表/多项式
# Step 4: 求和
exp_sum = INT32_REDUCE_SUM(exp_approx)  # reduction
# Step 5: 除法（整数近似）
softmax_out = INT32_DIV_APPROX(exp_approx, exp_sum)
# INT32输出直接供GEMM unit使用（INT8量化由DATATYPE_CAST完成）
```

术语一般如何实现？如何使用？
INT32 integer-only inference的实现路径：(1) 参考I-BERT（Kim et al., ICML 2021）的整数算子近似库——提供二阶多项式近似exp、牛顿迭代法sqrt、分段线性GeLU等；(2) 参考gemmlowp（Google）的定点算术库——提供量化乘法和累加的低精度原语；(3) Tandem Processor编译器内置这些整数近似的操作模板，自动将ONNX浮点算子翻译为INT32指令序列。INT32的代价是4×于INT8的scratchpad带宽和存储需求，但Tandem Processor通过128KB scratchpad + Data Access Engine的批量传输 + tile粒度流水设计来抵消这一开销。与需要硬件特殊函数单元（如Google VPU的exp/sqrt专用指令）的方案相比，integer-only方法牺牲一定的单算子效率（论文记录~0.8× slowdown for not having special function hardware），但换来了更高的可编程性和面积效率。

涉及论文标题：
- 21-Tandem Processor: Grappling with Emerging Operators in Neural Networks

## Einsum Notation (Einstein Summation for Tensor Algorithms)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
Einsum（Einstein Summation，爱因斯坦求和约定）是一种简洁表达张量代数操作的符号系统，在Orojenesis和Timeloop等加速器建模框架中被用作workload specification语言。Einsum通过隐式求和（implicit summation over repeated indices）描述多维张量间的计算关系。例如，矩阵乘法表达为B_{m,n}^{M,N} = A_{m,k}^{M,K} W_{k,n}^{K,N}，上标表示各rank的shape，重复下标k表示沿K维度的reduction。Orojenesis支持single Einsum（如GEMM/Conv/BMM/Grouped BMM）和chain of Einsums（producer-consumer序列，如GPT-3-6.7b LLM building block中Q_proj→bmm_QK→bmm_QKV→Final_proj→mm_0→mm_1）。Einsum的变体TeAAL、NumPy einsum()、TACO均基于相同核心思想：通过index notation表达任意rank的tensor contraction。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
以GPT-3-6.7b MHA中BMM的Einsum表示为例：
```
# Batched MM (Q_proj): B_{h,m,n}^{H,M,N} = A_{h,m,k}^{H,M,K} W_{h,k,n}^{H,K,N}
# H=32 heads, M=32768 (2048 seq_len×16 batch), K=128, N=128

# Grouped BMM (K_proj in MQA/GQA): B_{h,m,n}^{H,M,N} = A_{h,m,k}^{H,M,K} W_{g,k,n}^{G,K,N}
# G=1(MQA) to 32(MHA), W shared across H/G heads per group

# 2D Convolution: B_{p,q,n}^{P,Q,N} = Atw(p+dw_r, q+dh_s, c) W_{c,n,r,s}^{C,N,R,S}
# dw, dh: dilation offsets; r,s: filter coords; c: input channel
```
在Orojenesis中，Einsum被翻译为Timeloop problem specification (YAML)，包含各rank shape、density和operator type。Mapper基于Einsum的index structure自动推导reuse pattern和legal loop nest structure。

术语一般如何实现？如何使用？
Einsum notation被广泛使用于：NumPy/PyTorch的`einsum('mk,kn->mn', A, W)`提供灵活tensor contraction接口；TACO使用Einsum-like input language compile到优化后的CPU/GPU kernel；Timeloop/Orojenesis使用Einsum作为accelerator modeling input；TeAAL扩展Einsum支持sparse tensor accelerator modeling。在ML compiler pipeline（XLA/TVM）中，Einsum作为high-level IR representation，在lowering阶段被分解为tiled primitive操作供backend code generation使用。

**Cascade of Einsums**（FuseMax论文引入的概念）：将多个具有依赖关系的Einsum组织为DAG（有向无环图），描述更大的kernel。例如attention的1-pass cascade包含~15个Einsum（BQK、LM、RM、SLN、SLD、SLNV、PRM、SPD、RD、SPNV、RNV等），每个Einsum定义精确的iteration space和computation，cascade express它们之间的data dependency。Cascade的is-ﬁbertree（iteration space ﬁbertree）序列可用于分析：(1) pass count lower bound——对任意mapping/scheduling，某一tensor fiber需被遍历的次数；(2) algorithmic minimum live footprint——在mapping独立下，kernel中各tensor所需的最小on-chip footprint。Einsum在不同pass间无法fuse（因read-after-write dependency），同pass内可任意fuse。这使得cascade of Einsums成为mapping-agnostic的kernel分析和优化工具。

**Extended Einsums (EDGE)**：FuseMax使用的Einsum扩展。分离computation为三个action：map（pair-wise computation between shared ranks of two tensors）、reduce（reduction step of an Einsum）、populate（=，将RHS值放入LHS）。每个map/reduce action包含merge（指定iteration space中哪些点被touched：intersection ∩、union ∪、或pass-through）和compute（用户自定义操作，如×、+、max、÷、max(∪)、÷(←)、sub-then-exp(1)等）。支持iterative ranks（generative/iterative ranks）表达递归和迭代计算（如prefix sum: S_{i+1} = S_i + A_i）。FuseMax使用EDGE shorthand：drop add+union reduce、infix notation for map、max(Am,Bm)替代max(∪) map、Am/Bm替代÷(←)、e^{Am-Bm}替代sub-then-exp(1)。

涉及论文标题：
- 23-Mind the Gap: Attainable Data Movement and Operational Intensity Bounds for Tensor Algorithms
- 5-FuseMax_Leveraging_Extended_Einsums_to_Optimize_Attention_Accelerator_Design.pdf

---

## Arithmetic Intensity (for CIM Resource Allocation)

术语解释
Arithmetic Intensity (AI, 算术密度) 在CIM编译优化上下文中是衡量DNN算子或模型计算-访存需求比例的关键指标，定义为FLOPs per Memory Operation (FLOPs/MOP)。CMSwitch使用AI指导dual-mode CIM array的compute/memory资源分配——高AI算子分配更多compute array，低AI算子分配更多memory array。

术语是什么？
Arithmetic Intensity = Total FLOPs / Total Memory Operations。在CIM编译中：(1) **模型级AI差异**：不同DNN模型AI差异显著——ResNet50平均AI≈66（高AI，需要更多compute array），LLaMA2单batch inference平均AI≈2（低AI，需要更多memory bandwidth而非compute power）。Transformer-based模型（LLaMA2, GPT, BERT）普遍AI较低，而CNN模型（VGG, ResNet）AI较高。(2) **层级AI差异**：同一模型不同层AI也不同——ResNet50早期层AI<100 FLOPs/MOP，后期层AI>700 FLOPs/MOP；BERT的FC层AI远高于QKV attention计算。(3) **动态AI变化**：Transformer模型AI随input/output sequence length变化——BERT AI从150 FLOPs/MOP (seq_len=128)到>1000 FLOPs/MOP (seq_len=4096)。

从算法pipeline角度拆解：
AI如何影响CIM资源分配（以CMSwitch的MIP延迟模型为例）：
```
给定算子Oi的arithmetic intensity AI_Oi、total operations OP_Oi：
  Compute capacity per cycle: C = ComOi · OPcim  (ComOi个compute array × 每array MAC数/cycle)
  Memory bandwidth per cycle: M = MemOi · Dcim + Dmain  (MemOi个memory array × 每array数据率/cycle + 外部带宽)
  有效computation rate: R = min(C, M · AI_Oi)
  算子延迟: LOi ∝ OP_Oi / R

当 AI_Oi 大（如CNN后期层 AI>700）:
  R ≈ C (compute-bound) → 分配更多array为compute mode

当 AI_Oi 小（如LLaMA2 AI≈2）:
  R ≈ M · AI_Oi (memory-bound) → 分配更多array为memory mode以提升M
```

例如OPT-6.7B single layer：QKV projection AI≈50-100 → 约40-50% compute + 50-60% memory；FFN FC1 AI≈200 → 约60% compute + 40% memory；attention QK^T AI≈10 → 约60-70% compute（数据消费后即丢弃，无需大量memory buffer）。

术语一般如何实现？如何使用？
AI通过模型profile获取：对每个算子的FLOPs和内存访问量进行静态分析（类似roofline model）。在CMSwitch中，AI_Oi是MIP latency model的输入常量（Table 1中的Constants），由application和CIM chip在编译前确定。AI的准确度直接影响latency model的精度和资源分配的最优性。除了CIM编译，AI也广泛用于roofline model analysis、accelerator architecture design（如Orojenesis的OI mesa）和模型-硬件co-design。

在Duplex论文中，Arithmetic Intensity被称为Op/B (Operations per Byte)，用于characterize LLM inference中各layer的计算-访存特征以决定合适的处理单元：(1) MoE layer decoding-only stage: Op/B>1（因为多个requests共享相同expert，但每个expert仅处理部分tokens）；(2) Attention layer with GQA (deggrp=4-8): Op/B=4-8（heads共享K/V slices形成narrow GEMM）；(3) Attention layer with MHA: Op/B~1-2（每head独立K/V, GEMV-like）；(4) FC layers: 高Op/B（batched GEMM on all tokens）。Duplex的Logic-PIM target Op/B=1-32范围，compute-to-bandwidth ratio=8，而Bank-PIM target Op/B<1（per-bank MACs有限）。Continuous batching下mixed stage中prefill tokens arrival transiently升高MoE layer Op/B——这一fluctuation是Duplex设计xPU+Logic-PIM dual-processor的key motivation。

在AUM论文中，ARI（Arithmetic Intensity）被用于AU Selecting——判断LLM serving的底层矩阵操作应使用AMX还是AVX。ARI公式：Prefill 阶段 ARI = 6(1/d_model + 3/(B×L))^{-1}，Decode 阶段 ARI = 6(1/d_model + 3/B)^{-1}，其中 B=batch size, L=input sequence length, d_model=model dimension。ARI 越高 → AMX 使用越密集（U_AU 越高）。QKV mapping 等 GEMM 操作的 ARI 随 d_model 增大而升高（因 O(n^2) FLOPs vs O(n) memory），随 B 增大而升高（GEMM M 维度增大，从 GEMV memory-bound → GEMM compute-bound 转变）。ARI 阈值被用于划分 High-AU / Low-AU / None-AU 使用等级，指导 Processor Dividing 和资源分配决策。

涉及论文标题：
- 25-Be CIM or Be Memory- A Dual-mode-aware DNN Compiler for CIM Accelerators.pdf
- 61-Duplex- A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching.pdf
- 83-PAPI- Exploiting Dynamic Parallelism in Large Language Model Decoding with a Processing-In-Memory-Enabled Computing System .pdf
- 90-AUM_Unleashing_HPCA-2026-Wang.pdf

## Sparse Attention (稀疏注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sparse Attention是Transformer模型中self-attention的优化变体，通过消除弱连接来降低attention计算的二次复杂度（O(dn²)→近似线性）。核心观察：大多数token之间仅有弱关联，可被安全剪枝，仅保留对预测重要的连接并引入轻微准确率损失。稀疏注意力分为两大类：(1) Static Sparsity——稀疏mask矩阵在推理前预先确定，与输入无关（如Longformer sliding window、Big Bird block-based pruning）；(2) Dynamic Sparsity——通过量化阶段确定稀疏mask矩阵，与输入绑定（如Sanger quantize-and-pruning、DOTA）。静态和动态稀疏都通过两个核心操作实现：SDDMM（稀疏mask M控制Q×K^T仅计算non-zero位置的点积）和SpMM（稀疏S×密集V的矩阵乘）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以Sanger dynamic sparsity（ASADI采纳方法）在BERT上为例：
1. Pre-processing：输入X→线性层得Q,K→计算密集S̃=Q·K^T→量化：每行取top-τ值标记non-zero→得稀疏mask M→S=S̃⊙M(NNZ=τ·n)。
2. SpMM计算：
```
for each non-zero S[i][j]:
    for k=0 to d-1:
        Z[i][k] += S[i][j] * V[j][k]  // NNZ次乘加 vs 全连接n²次
```
FLOPs从O(n²d)降至O(NNZ·d)，sparsity=90%时降至10%。Longformer的sparsity pattern为固定窗口的对角线带状（M[i][j]=1 iff |i-j|≤w/2）；Sanger为输入依赖的top-τ选择。

ALISA提出的Sparse Window Attention (SWA)是一种混合静态-动态稀疏方法（ISCA'24）。SWA在每个解码步选择caching ratio r比例的token：将2k=⌊nr⌋个选中token均分为k个locally static tokens（最近k个token，保留语言顺序语义）和k个globally dynamic tokens（通过"local attention sum"机制选择）。Local attention sum：对最近k步的attention weights沿head和step维度求和，得到每个prior token的累积重要性分数S[j]，选top-k作为全局动态token。SWA的关键洞察：(1) multiple preceding steps能比single step更好地提示重要token；(2) 混合稀疏模式避免纯local attention（Longformer）丢失远距离重要token和纯动态选择（如H2O的全局attention weight sum）计算开销过大的问题。SWA中稀疏KV tensor通过gather操作打包为dense tensor（K_s=K[I,:], V_s=V[I,:]），后续执行dense matmul，因此计算保持规整。与dense attention的Spearman attention score correlation接近1.0。ALISA实验显示OPT/LLaMA模型上SWA在80% KV sparsity下准确率几乎无损（<5% drop），而Longformer和SparseTransformer的sparse attention在同等sparsity下准确率崩溃。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
训练阶段在dense model上fine-tune引入稀疏mask（Longformer用multi-pattern attention；Sanger用STE梯度传播动态剪枝）。推理阶段用稀疏矩阵库（CPU: MKL Sparse BLAS; GPU: cuSPARSE PyTorch sparse）或专用加速器（ASADI/SPRINT/CPSAA/Sanger）以CSR/DIA等格式解码并跳过bubble计算。关键trade-off：sparsity越高→计算越少但准确率损失越大；对角线局部性越好→压缩效率越高。

SOFA (MICRO'24)提出面向LTPP (Large-scale Token Parallel Processing)的动态稀疏加速，针对传统DS三阶段在大量token并行时暴露的瓶颈：(1) Pre-compute阶段低精度预测开销占>57%总延迟；(2) 全行处理导致大量DRAM access (MAT ratio 72%)；(3) FlashAttention-2的tiling引入surging计算。SOFA通过cross-stage coordinated tiling将三阶段分解为fine-grained sub-stage tiles实现流水线化，并利用前阶段信息减少后阶段计算。其DS pipeline：DLZS (log-domain multiplier-free prediction) → SADS (distributed sub-segment top-k sorting) → SU-FA (sorted-updating FA leveraging top-k info)。LTPP下128 token并行处理，中间结果完全留在SRAM，消除off-chip DRAM store/load。与传统DS相比：计算复杂度降低28%，memory access降低79%，PE利用率达85.2%。与GPU A100相比9.5× speedup，71.5× energy efficiency。

涉及论文标题：
- 29-ASADI_Accelerating_Sparse_Attention_Using_Diagonal-based_In-Situ_Computing.pdf
- 72-ALISA_Accelerating_Large_Language_Model_Inference_via_Sparsity-Aware_KV_Caching.pdf
- 74-SOFA_A_Compute-Memory_Optimized_Sparsity_Accelerator_via_Cross-Stage_Coordinated_Tiling.pdf

## Diagonal Locality in Sparse Attention (稀疏注意力的对角线局部性)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
对角线局部性指稀疏注意力矩阵中的非零值倾向于集中分布在中心对角线附近的现象。ASADI对Sanger动态稀疏的实验统计：中心ω条对角线区域（ω=n/16,n/8,n/4,n/2）内非零值占比NNZ_ω/NNZ，ω=n/8时超50%非零在对角线区域，且该区域数据密度是其他区域的7×以上。根本原因：词或像素与其邻近词/像素在语义层面关联更紧密，转换为attention矩阵即非零值沿对角线分布。SparseBERT [33]也验证多种稀疏注意力均有此特性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
用于驱动DIA压缩策略：(1) Longformer sliding window天然为bubble-free对角线带状；(2) Sanger动态稀疏需bubble-containing DIA——中心ω=n/8对角线区域bubble-free压缩，区域外非零元素移至区域内bubble，条件为NZ_ω≥NNZ_o（区域内零值数≥区域外非零值数），记录原坐标(Rd,Ro)。ω=n/8时NZ_ω刚好略大于NNZ_o，为最佳平衡。对角线局部性直接决定DIA压缩效率——局部性越强→压缩越紧凑→有效计算比例越高。消融实验：对角线局部性从60%降至10%时性能持续退化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
评估方法：在fine-tuned sparse attention模型上推理，统计每层S矩阵的NNZ_ω/NNZ比例。利用方法：根据比例自适应选ω（高sparsity时缩小ω至n/16或n/32；低sparsity时增大至n/4）。适用于NLP Transformer和CV ViT。

涉及论文标题：
- 29-ASADI_Accelerating_Sparse_Attention_Using_Diagonal-based_In-Situ_Computing.pdf

## DIA Format (Diagonal Compression Format, 对角线压缩格式)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DIA Format是按对角线方向存储稀疏矩阵的压缩格式。每条对角线上元素具有相同的(row-col)偏移，天然对齐列坐标。结构：(1) value lists——每条对角线元素按行序排列；(2) diagonal index (DI)——每条对角线的偏移，中心DI=0，右下DI>0，左下DI<0。ASADI将DIA用于二种场景：(a) Bubble-free DIA——对角线全为非零（如Longformer）直接DIA存储；(b) Bubble-containing DIA——对角线上含bubble（如Sanger），中心ω对角线bubble-free压缩，区域外非零移入bubble并附(Rd,Ro)坐标映射。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
n=6,ω=2示例：原始S矩阵→中心DI_0和DI_{-1}按列式存储（每列n个元素）。SpMM优势：DIA列坐标天然对齐右矩阵行坐标，直接vector-vector multi，无需CSR的row-wise remapping。DIA仅需ω次迭代(vs CSR的n次)。Longformer场景对角线局部性7.5×→DIA比CSR节省7.5×迭代。解压：中心对角线按DI恢复列位置→灰色元素按(Rd,Ro)映射回原坐标。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
存储用三数组：values[ω][n]、diagonal_index[ω]、可选Rd[]和Ro[]。SpMM时加载对角线与V行做point-wise multiply→accumulate；SDDMM时DI控制Q矩阵shift（DI>0下移，DI<0上移）→与K做vector-vector multi。适合对角线局部性强的矩阵（稀疏注意力），不适合随机sparsity pattern（bubble过多时效率退化）。

涉及论文标题：
- 29-ASADI_Accelerating_Sparse_Attention_Using_Diagonal-based_In-Situ_Computing.pdf

## CSR Format (Compressed Sparse Row, 行压缩稀疏格式)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CSR是稀疏矩阵最常用的压缩格式。三个数组：row_ptr[n+1]（每行起始索引）、col_idx[NNZ]（每个非零的列坐标）、values[NNZ]（非零值按行序存储）。保留行局部性（同行元素连续）但破坏列坐标连续性（每行元素可来自任意列）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SpMM with CSR:
```
for i=0 to n-1:                    // n次迭代
    for idx=row_ptr[i] to row_ptr[i+1]-1:
        j=col_idx[idx]; val=values[idx]
        for k=0 to d-1:
            Z[i][k] += val * V[j][k]  // row-wise remapping
```
CSR在稀疏注意力中的问题：(1) 每次迭代仅处理一行非零（平均稀疏度×n个），大量bubble致PE空闲；(2) 稀疏注意力行局部性弱→V[j][k]访问随机且跨bank→PIM中频繁cross-bank transfer；(3) SDDMM迭代间共享K行需串行。ASADI的CSR-ASADI消融证明：CSR在ReRAM上in-situ计算因bubble降低并行度，短序列时性能低于PIM baseline。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
广泛用于CPU(MKL Sparse BLAS)/GPU(cuSPARSE, PyTorch sparse)稀疏矩阵库和专用加速器（Sanger, SPRINT, CPSAA）。优势：通用性（任意sparsity pattern有效）；劣势：无法利用对角线局部性——这正是ASADI用DIA替代CSR的核心动机。

在ACES SpMM加速器中，CSR的row_ptr数组被用于band partitioning——通过计算相邻行row length差（|row_len[i] - row_len[i-1]| > 10则新建band），将相似稀疏模式的行聚合到同一band，从而为per-band adaptive condensing提供分区依据。

涉及论文标题：
- 29-ASADI_Accelerating_Sparse_Attention_Using_Diagonal-based_In-Situ_Computing.pdf
- 70-ACES- Accelerating Sparse Matrix Multiplication with Adaptive Execution Flow and Concurrency-Aware Cache Optimizations.pdf

## Opacity in Transactional Memory

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Opacity（不透明性）是 Transactional Memory（TM）中最严格的安全保证/正确性准则之一，由 Guerraoui & Kapalka [PPoPP 2008] 提出。Opacity 要求：**所有事务（包括最终 abort 的事务）在任何时刻观察到的共享内存状态，都可以被解释为某个串行执行历史的一部分**。简单说，即使一个事务最终 abort 了，它在执行过程中也从未见过不一致的状态——不会因为读到 uncommitted writes 导致除零、段错误等不可恢复的错误。

Opacity 比 serializability 更强——serializability 只要求 committed 事务可序列化，aborted 事务允许看到任意"脏"数据。在 DBMS 中事务运行在 sandbox 环境（通过 SQL 引擎抽象），读到 dirty data 通常可恢复。但在 STM 中，事务直接操作内存地址——若读到 uncommitted write 后触发非法操作（如以此为指针 dereference），可能直接 crash 而非简单 abort→rollback。因此 opacity 对 STM 是必要的安全保证。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Opacity 在 STM 实现中通过以下机制保证（以 PIM-STM 的 7 种实现为例）：

```
// Opacity要求: 事务的所有read都必须来自consistent snapshot
// 不能看到uncommitted writes或inconsistent state

// 机制1: Write-Back (WB) 策略天然保证
// 所有write缓冲在私有写集，commit时才对外可见
tx_write(addr, val):
    writeset.buffer(addr, val)  // 不写入共享内存
// → 并发事务看不到uncommitted writes → 自动满足opacity

// 机制2: Write-Through (WT) + Encounter-Time Locking (ETL)
// WT直接写入共享内存，但ETL先acquire lock→block其他事务读
tx_write(addr, val):
    acquire_write_lock(lock_table[hash(addr)])  // ETL: 立即lock
    old_val = *addr
    undo_log.append((addr, old_val))
    *addr = val  // 直接写入共享内存
// → lock阻止并发读到uncommitted values → 保证opacity
// 但WT+CTL不可行: commit时才lock→uncommitted writes在lock前已被外部读到→违反opacity

// 机制3: Invisible Reads + Validation
tx_read(addr):
    // 先检查是否有并发update事务改变了snapshot
    if sequence_lock_changed():
        validate_readset()  // 逐地址确认值未变
    val = *addr
// 若值已被并发事务修改→validation失败→abort→保证看到的是consistent snapshot

// 机制4: Visible Reads + rw-lock
tx_read(addr):
    acquire_read_lock(rw_lock_table[hash(addr)])  // register as reader
    val = *addr
// 写事务想修改该地址→发现read lock→需等待或abort→保证reader看到consistent state
```

PIM-STM 中 opacity 的关键设计约束：
- **WT 仅与 ETL 兼容**（图2 taxonomy）：若 CTL+WT→uncommitted writes 无 lock 保护直接可见→违反 opacity。
- **NOrec 的 opacity 保证**：commit 时 sequence lock 保护 write-back→所有 update 事务串行化提交→aborting 事务通过 value-based validation 确认 readset 未变。
- **VR 的 opacity 保证**：读时获取 rw-lock read-mode→写事务在升级 lock 时 abort（若有并发 reader）→防止 reader 看到不相容的 writes。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Opacity 在两篇奠基论文中定义：
- Guerraoui & Kapalka [PPoPP 2008]: 提出 opacity 概念和 formal definition，证明其比 serializability 更严格。
- 几乎所有 mainstream STM（TinySTM, NOrec, TL2, SwissTM, PIM-STM）都实现 opacity。

实现方式取决于 STM 算法：
- **WB 策略**天然支持 opacity（write 延迟对外可见直到 commit）。
- **WT 策略**需配合 ETL（immediate lock）来防止 uncommitted data exposure，commit 时 release + undo on abort。
- **Invisible reads** 需 validation 来检测 stale reads——若 readset 中某值被并发 update 事务修改→abort→opacity 保证该事务始终看到 consistent snapshot。
- **Multi-version STM**（如 JVSTM, CSMV）：通过保留多个 version 实现 opacity——每个事务读自己 snapshot version 的值，无需 validation。

更弱的正确性条件（用于特定场景）：
- **Snapshot Isolation**：允许 write skew anomaly，不保证 opacity。
- **Elastic Transactions** [Felber et al.]：放松 opacity 以允许事务在冲突时部分重做而非全部 abort。

涉及论文标题：
- 31-PIM-STM- Software Transactional Memory for Processing-In-Memory Systems.pdf

---

## SpTRSV (Sparse Triangular Solve, 稀疏三角矩阵求解)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SpTRSV（Sparse Triangular Solve）是求解线性系统 Lx = b 或 Lx = b（L/U为稀疏三角矩阵）的核心kernel。如Algorithm 1（标准dot product-based算法）：对第i行，s = Σ(v_e × x[c_e])累积c_e < i的所有非零×已求出的x值→用对角元素归一化x[i] = (b[i]-s)/v_l。SpTRSV的关键挑战是行间数据依赖——第i行的计算依赖前i-1行所有已求解的x[j]（j < i且矩阵位置(i,j)非零）→并行度受限于三角矩阵的sparsity pattern和行依赖DAG的level结构。这在GPU上导致低arithmetic intensity和低利用率——因为GPU并行执行大量线程，但SpTRSV的行依赖限制可并行行数。cuSPARSE使用row reordering（将独立行分组为level，同level行可并行）；pSyncPIM [ISCA 2024]是首个将SpTRSV mapping到PIM的学术提案。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
标准Dot Product-based SpTRSV（Algorithm 1）：

```
// Lower triangular solve Lx = b, where L diagonal = v_l (non-1)
M: n×n lower triangular matrix in COO format
b: input vector, x: output vector

for i = 0 to n-1:
    s = 0
    for all e = (i, c_e, v_e) ∈ M where c_e < i:
        s += v_e × x[c_e]    // 依赖所有前序x[c_e]值
    l = (i, i, v_l) ∈ M      // 对角线元素
    x[i] = (b[i] - s) / v_l  // 除法在关键路径上
```

pSyncPIM的Scalar Multiplication-based SpTRSV（Algorithm 3, 用于PIM）：

```
// 假设ILDU预处理: L*=L-I (diagonal=0, lower=original), D⁻¹对角单独存储
// Column-first COO, batch按列分组为levels (独立列可并行)

for each batch (column group with independent columns):
    for each column j in batch:
        scale = b[j] × D⁻¹[j]     // host预计算scale
    for all e = (r_e, j, v_e) ∈ L* where r_e > j:
        x[r_e] = x[r_e] - scale × v_e   // scalar broadcast × column values
    // 各bank并行执行上述loop, 每PE处理其分配到的column (predicated)
```

关键区别：标量乘法式消除标准算法的行依赖——改为列遍历，每列scale广播到所有非零行进行x[r] -= scale × v_e更新，不同列间无依赖（对于同batch内独立列）→适配PIM的all-bank parallel column处理。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通用实现方式：
- GPU（cuSPARSE [NVIDIA]）：row reordering（Cuthill-McKee, level scheduling）后按level并行执行行→同level行无依赖→多thread并行。CapelliniSpTRSV [ICPP 2020]使用thread-level和warp-level无锁同步。Block Algorithm [Ahmad et al., TPDS 2021]：递归二分三角矩阵为L0/M/L1子块，L0/L1递归SpTRSV而M用SpMV求解→减少总迭代数。
- pSyncPIM on PIM：采用block algorithm + scalar multiplication kernel + recursive decomposition。预处理：ILDU分解归一化对角线（D⁻¹存储值消除除法的在线计算）→row reordering最大化level内行数→recursive block decomposition生成unitriangular子问题→column-first COO批处理。Kernel使用scalar multiplication确保column内操作独立和local——PE执行x[r] -= scale × v_e仅访问自身bank内的x和v。
- 通用库：MATLAB pcg()用SpTRSV做preconditioning；许多HPC应用（电磁场[Um]、CFD[Wissgott]、电路仿真[Hamm]）在conjugate gradient类迭代求解器中大量使用SpTRSV。pSyncPIM vs GPU (RTX 3080 cuSPARSE)：6个FP64矩阵geomean 3.53× speedup，但hyper-sparse near-diagonal矩阵（parabolic_fem）GPU更优因GPU可并行更多行。

涉及论文标题：
- 32-pSyncPIM_Partially_Synchronous_Execution_of_Sparse_Matrix_Operations_for_All-Bank_PIM_Architectures.pdf

---

## COO Format (Coordinate List, 坐标列表稀疏格式)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
COO（Coordinate List）是最直接的稀疏矩阵压缩格式：用三个数组——row_idx[NNZ]、col_idx[NNZ]、values[NNZ]——分别存储每个非零元素的行坐标、列坐标和数值。与CSR/CSF格式不同，COO不维护行指针（row_ptr）结构→无间接寻址→单次access即可获取完整的(row, col, val)三元组。缺点是无行局部性保证——同行元素可能分散在数组中（特别是按列排序时）。COO在pSyncPIM中作为首选格式，因为：(1) 不需要CSR的row_ptr indirection（indirection需要cross-row access，与all-bank PIM约束冲突）；(2) 三元组结构天然适配pSyncPIM的sparse vector queue（每个queue有独立的row/col/value 64B子队列）。对于<1% density的HPC矩阵，COO无显著的元数据存储开销劣势。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
COO在pSyncPIM SpMV中的应用：

```
// COO存储: row[0..NNZ-1], col[0..NNZ-1], val[0..NNZ-1]
// 按行排序 → 同batch包含连续行的非零元素

// pSyncPIM SpMV loop over COO (per PE, Algorithm 2):
Read SpVQ0 ← Bank // push 32B of (row, col, val) triplets
// SpVQ0.row sub-queue: [r₀, r₁, ..., rₖ]
// SpVQ0.col sub-queue: [c₀, c₁, ..., cₖ]
// SpVQ0.val sub-queue: [v₀, v₁, ..., vₖ]

IndMOV SRF ← Bank[SpVQ0.col]  // 用col值做间接地址读input vector x
                                // 无需CSR的row_ptr→row_start→row_end两层indirection
SSpV SpVQ1 ← SRF × SpVQ0.val  // scalar-×-sparse vector
SpVDV DRF0 ← SpVQ1 + Bank     // accumulate
```

与CSR对比（pSyncPIM观点）：CSR需额外metadata access (row_ptr[i]→row_start, row_ptr[i+1]→row_end)，这要求per-PE访问可能不在同一memory row的数据→与all-bank统一row access约束冲突。COO将完整三元组嵌入同一memory row内→在all-bank PIM下更可行。CSR支持需额外4个32-bit index寄存器和integer adder（Section IV-C），area overhead minor。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
COO广泛用于HPC稀疏矩阵benchmark（SuiteSparse Collection [Davis & Hu, TOMS 2011]）和多个稀疏加速器（ExTensor [Hegde et al., MICRO 2019]使用COO变体）。双精度COO：row/col 32-bit each + val 64-bit = 16B per non-zero。pSyncPIM column-first COO for SpTRSV：按列排序→同列元素在内存中连续→batch处理按列分group。对多精度支持：INT8 value减小element size→同1KB row可容纳更多COO triplets→减少batch数量→减少external I/O。COO可与CSR互转（scan-based转换），主机侧preprocessing完成。

涉及论文标题：
- 32-pSyncPIM_Partially_Synchronous_Execution_of_Sparse_Matrix_Operations_for_All-Bank_PIM_Architectures.pdf

---

## Recursive Block Algorithm for SpTRSV (SpTRSV递归分块算法)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Recursive Block Algorithm for SpTRSV是Ahmad等人 [TPDS 2021]提出的sparse triangular solve分治算法，pSyncPIM将其适配为PIM加速的核心算法框架。算法将稀疏下三角矩阵L递归分解为子块结构：L = [L₀ 0; M L₁]，其中L₀和L₁为更小的lower triangular子矩阵，M为sparse square submatrix。原问题Lx=b分解为三步：(1) L₀x₀ = b₀（递归SpTRSV解上半部分）；(2) b₁' = b₁ - Mx₀（SpMV更新右端向量）；(3) L₁x₁ = b₁'（递归SpTRSV解下半部分）。递归持续到子三角矩阵尺寸≤1KB（HBM2单memory row）——此时整个子矩阵可被PIM单次all-bank操作处理。此算法将大三角矩阵求解转化为多个小三角矩阵求解和SpMV的组合，既降低单次kernel复杂度又利用PIM高效处理SpMV。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Recursive Block SpTRSV on PIM (Section VI-A):
spTRSV_recursive(L, b):
    if L.n ≤ 1KB_memory_row_limit:
        // Base case: unitriangular submatrix fits in 1 memory row
        // Column-first COO layout, column batches organized by levels
        for each column_batch:
            for each level in batch (all columns independent):
                for each column j in level:
                    scale = b[j] × D⁻¹[j]  // host pre-compute
                    for all non-zero e = (r, j, v) ∈ L*|r>j:
                        x[r] -= scale × v   // PIM scalar multiplication kernel
        return x
    else:
        // Split L into [L₀ 0; M L₁], b into [b₀; b₁]
        L₀, M, L₁ = partition(L)  // L₀ dim = L.n/2 (roughly)
        b₀, b₁ = partition(b)
        
        x₀ = spTRSV_recursive(L₀, b₀)       // Step 1: 递归上三角
        b₁' = b₁ - SpMV(M, x₀)              // Step 2: SpMV更新 
        x₁ = spTRSV_recursive(L₁, b₁')       // Step 3: 递归下三角
        return [x₀; x₁]
```

SpTRSV kernel处理unitriangular子矩阵（L*=L-I，对角线元素为1省略），column-first COO排序。Column batch内分为多个level——同level内所有列之间无依赖→可并行。对每个level执行scalar multiplication-based algorithm（Algorithm 3）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
pSyncPIM实现：(1) host侧preprocessing执行recursive decomposition + ILDU normalization (对角线→1) + row reordering (最大化level内独立行数)；(2) 对每个base case子三角矩阵生成column-first COO triplets→按batch分到各bank；(3) PIM执行scalar multiplication kernel处理column batches，SpMV kernel处理square M子矩阵。pSyncPIM vs GPU on SpTRSV：geomean 3.53× speedup (6 matrices)，但parabolic_fem（hyper-sparse near-diagonal）GPU更优——其小数据依赖允许GPU并行更多行超过PIM的1KB row size constraint。原算法[TPDS 2021]在CPU上验证，pSyncPIM首次将其mapping到PIM并证明recursive decomposition使PIM处理任意大三角矩阵成为可能。

涉及论文标题：
- 32-pSyncPIM_Partially_Synchronous_Execution_of_Sparse_Matrix_Operations_for_All-Bank_PIM_Architectures.pdf

---

## ILDU Decomposition (不完全LDU分解)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ILDU（Incomplete LDU Decomposition）是稀疏线性代数中的预处理技术。将矩阵A近似分解为A ≈ LDU，其中L为unit lower triangular matrix（对角线=1）、D为diagonal matrix、U为unit upper triangular matrix（对角线=1）。"不完全"(Incomplete)表示分解仅保留部分fill-in（新产生的非零元素）以控制sparsity和存储开销——通常通过drop tolerance或fill-in level限制。在pSyncPIM的SpTRSV acceleration中使用ILDU的目的：(1) 归一化对角线元素为1——将标准SpTRSV中的division操作（x[i] = (b[i] - s) / v_l, 见Algorithm 1 line 10）removed from online computation path（division需数十周期和额外除法器）；(2) 将对角线值存储为D⁻¹（diagonal inverse），使PIM kernel只需乘法无需除法；(3) 生成unitriangular矩阵L*和U*——对角线元素不存储（均为1）→节省存储空间并简化PIM kernel logic。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// ILDU应用pipeline (host preprocessing, Section VI-D):
原始稀疏三角矩阵 L (diagonal not 1):
  ILDU(L) → L_hat * D * U_hat   // 不完全分解
  Store: L* = L_hat - I           // unitriangular lower (diagonal=0, 存储非对角元素)
         D⁻¹ = diag(D)⁻¹         // diagonal inverse matrix (only diagonal entries)
         U* = U_hat - I           // unitriangular upper

// PIM SpTRSV kernel使用ILDU后数据:
// 原算法: x[i] = (b[i] - Σv_e×x[c_e]) / v_l  (line 10需要除法)
// ILDU后: x[i] = D⁻¹[i] × (b[i] - Σv_e×x[c_e])  (乘法替代除法)
// 简化: 标量乘法based SpTRSV (Algorithm 3) 列操作:
for column j:
    scale = b[j] × D⁻¹[j]         // host预计算, 仅乘法
    for each e = (r, j, v) ∈ L*:
        x[r] = x[r] - scale × v    // PIM内仅乘加, 无除
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ILDU广泛用于迭代法（Conjugate Gradient、GMRES、BiCGStab）的preconditioning——求解M⁻¹Ax = M⁻¹b中的M ≈ A（preconditioner矩阵）。ILU（Incomplete LU）是其变体（无D分离）。pSyncPIM中ILDU在host端离线执行（matrix preprocessing step, 不计入kernel execution time），其overhead被大量迭代求解的benefit摊销。实现库：SPARSKIT [Saad]、ILUPACK [Bollhöfer]、MATLAB ilu()。pSyncPIM评估使用p-BiCGStab和p-CG两种preconditioned iterative methods——二者均使用ILDU作为preconditioner——验证ILDU+Scalar Multiplication SpTRSV在PIM上的端到端效果：pSyncPIM+PIM SpTRSV vs GPU cuSPARSE SpTRSV = 1.68×~2.88× speedup for linear system solvers。

涉及论文标题：
- 32-pSyncPIM_Partially_Synchronous_Execution_of_Sparse_Matrix_Operations_for_All-Bank_PIM_Architectures.pdf

## Multi-Camera Deep Learning-based 3D Spatial Computing Pipeline

术语解释
Multi-Camera 3D Spatial Computing Pipeline是AR/VR、自动驾驶和智能手机中处理多相机图像并提取语义特征的标准化四阶段流程，通过坐标系变换（球面↔切线平面）连接多相机图像系统与DNN推理系统。

术语是什么？
该pipeline由4个基础阶段组成：
1. **Stage 1 (iProj + Stitch)**：将多个不同位置/旋转/畸变的相机图像通过inverse perspective projection (iProj)映射到统一球形坐标(θ,ϕ)，再stitch成统一球形图像。
2. **Stage 2 (Proj + Tangent Generation)**：从球形图像通过perspective projection (Proj)生成多张切线平面图像（虚拟相机视角），以匹配DNN模型预训练所用的rectilinear图像数据集（如ImageNet、KITTI）。
3. **Stage 3 (DNN Execution)**：在每张切线图像上执行DNN模型（如monocular depth estimation、image segmentation），提取语义特征。
4. **Stage 4 (iProj + Feature Stitch)**：将DNN feature maps通过iProj映射回球形坐标并stitch成360°语义输出。

根据应用，某些阶段可跳过（如Stage 1和2可由商业360°相机替代）。核心瓶颈在于：Stage 1和Stage 4的iProj操作是内存密集型的非线性图像变形，无法像DNN那样通过batch processing加速——多相机mapping index不可共享导致延迟随相机数线性增长。

从算法pipeline角度拆解：
以360° RGB-D generation（18相机@4纬度, 256×256）为例的完整流程：
```
// Stage 1: Multi-camera iProj + Stitch
for each camera cam_i in multi-camera rig:
    // 公式(2): 切线平面(u,v) → 球形(θ,ϕ)
    θ = θc + atan(u·sin(c) / (γ·cos(ϕc)·cos(c) - v·sin(ϕc)·sin(c)))
    ϕ = asin((cos(c)·sin(ϕc) + v·sin(c)·cos(ϕc)) / γ)
    where γ = sqrt(u²+v²), c = atan(γ)
    // 预计算LUT: mapping_index[cam_i][u][v] = (θ, ϕ)
    // Remap: I_i(u,v) → S_i(θ,ϕ)  (inverse warping: 4邻域双线性插值)
// Stitch: 相邻S_i通过image blending合并为统一S(θ,ϕ)

// Stage 2: Proj + Tangent Images
for each target_view j:
    // 公式(1): 球形(θ,ϕ) → 切线平面(u,v)
    u = cos(ϕ)·sin(θ−θc_j) / cos(c)
    v = (cos(ϕc_j)·sin(ϕ)−sin(ϕc_j)·cos(ϕ)·cos(θ−θc_j)) / cos(c)
    where cos(c) = sin(ϕc_j)·sin(ϕ) + cos(ϕc_j)·cos(ϕ)·cos(θ−θc_j)
    // Remap: S(θ,ϕ) → T_j(u,v)

// Stage 3: DNN on each T_j
for each tangent_image T_j:
    depth_map_j = Monodepth2(T_j)  // or U-Net for segmentation

// Stage 4: iProj + Feature Stitch
for each depth_map_j:
    // 反投影: depth feature → 球形坐标
    S_depth_j(θ,ϕ) = iProj(depth_map_j)
// Stitch: S_depth = blend(S_depth_0, ..., S_depth_N)
// Output: 360° RGB-D
```

术语一般如何实现？如何使用？
GPU实现（baseline）：所有阶段在GPU上执行，DNN (Stage 3)受益于batch processing，Proj (Stage 2)可共享单个球形输入加速，但iProj (Stage 1/4)因不可共享mapping index导致延迟随相机数线性增长。GPU上8相机仅image projection需87.3ms（RTX2080Ti）。CamPU实现：Stage 1/2/4的图像投影和blending由CamPU专用硬件加速（RTL, 28nm, 500MHz, 0.54mm²），Stage 3 DNN由DSPU/GPU执行。CamPU+DSPU端到端360° RGB-D generation 94.1ms vs RTX2080Ti 270.1ms (2.9× faster)。适用场景：AR/VR（4+相机）、自动驾驶（8+相机）、智能手机（2+相机）、360°深度估计、3D目标检测。

涉及论文标题：
- 33-CamPU_A_Multi-Camera_Processing_Unit_for_Deep_Learning-based_3D_Spatial_Computing_Systems.pdf

## Inverse Perspective Projection (iProj) and Perspective Projection (Proj)

术语解释
iProj和Proj是3D空间计算系统中坐标变换的核心操作对——iProj将相机切线平面坐标(u,v)映射到球形坐标(θ,ϕ)，Proj将球形坐标映射回切线平面。它们是连接多相机图像与统一球形表示及DNN切线输入的数学桥梁。

术语是什么？
- **Perspective Projection (Proj)**：从球形坐标(θ,ϕ)到切线平面坐标(u,v)的投影。公式(1)：u = cos(ϕ)sin(θ−θc)/cos(c)，v = (cos(ϕc)sin(ϕ)−sin(ϕc)cos(ϕ)cos(θ−θc))/cos(c)，cos(c)=sin(ϕc)sin(ϕ)+cos(ϕc)cos(ϕ)cos(θ−θc)。(θc,ϕc)为切线平面中心在球形坐标中的位置。用于Stage 2——从统一球形图像生成DNN输入切线图像。
- **Inverse Perspective Projection (iProj)**：从切线平面坐标(u,v)到球形坐标(θ,ϕ)的投影。公式(2)：θ = θc + atan(u·sin(c)/(γ·cos(ϕc)·cos(c)−v·sin(ϕc)·sin(c)))，ϕ = asin((cos(c)·sin(ϕc)+v·sin(c)·cos(ϕc))/γ)，γ=√(u²+v²)，c=atan(γ)。用于Stage 1和Stage 4——将相机图像或DNN特征映射回球形坐标。

二者在固定多相机rig下映射关系不变，因此可预计算为LUT避免实时三角函数运算——LUT-based image projection将计算密集型坐标变换转为内存密集型的查表+remap操作。

从算法pipeline角度拆解：
以Stage 1 iProj的inverse warping+双线性插值为例：
```
// Input: tangent image I(u,v), 预计算LUT mapping[θ][ϕ] = (u,v)
// Output: spherical image S(θ,ϕ)
for θ in range(θ_min, θ_max):
    for ϕ in range(ϕ_min, ϕ_max):
        (u,v) = LUT[θ][ϕ]  // lookup pre-computed mapping
        // Inverse warping: 4邻域双线性插值
        u0=floor(u), v0=floor(v), u1=u0+1, v1=v0+1
        du=u-u0, dv=v-v0
        S(θ,ϕ) = (1-du)*(1-dv)*I(u0,v0) + du*(1-dv)*I(u1,v0)
               + (1-du)*dv*I(u0,v1) + du*dv*I(u1,v1)
```
每个输出像素需4次remap（加载4个邻域像素），导致4倍内存访问。
Proj类似但方向相反（从球形图像生成切线图像），inverse warping同样使用4邻域插值。

术语一般如何实现？如何使用？
GPU实现：使用CUDA kernel执行公式计算或LUT查表→remap→双线性插值。存在非规则内存访问（23% cache miss rate, RTX2080Ti）、不可batch共享mapping index（多相机iProj）、full-size矩形扩展（88.3%冗余数据）等问题。CamPU实现：预计算LUT（一次计算，多帧复用）→ inter/intra-data reuse压缩mapping index（节省94.4% LUT footprint）→ CamPU硬件加速remap+插值。Coordinate converter unit（SIMD分段线性逼近三角函数）仅在mapping index需要更新时才激活。适用场景：所有涉及相机坐标变换的3D空间计算系统——360°图像/视频生成、AR/VR渲染、多相机SLAM、全景深度估计。

涉及论文标题：
- 33-CamPU_A_Multi-Camera_Processing_Unit_for_Deep_Learning-based_3D_Spatial_Computing_Systems.pdf

## Inter-data and Intra-data Reuse on Mapping Indices

术语解释
Inter-data Reuse和Intra-data Reuse是CamPU针对多相机LUT-based图像投影提出的两种mapping index压缩方法——前者利用纬度对齐多相机mapping index的形状相似性共享数据，后者利用同一mapping index内相邻元素的数值相似性进行差分编码，组合压缩率达94.4%。

术语是什么？
LUT-based图像投影将公式(1)(2)的三角函数计算预先完成并存为mapping index LUT，投影时只需查表+remap。但iProj的mapping index很大（1 MB/image，256×256图像），多相机（如18相机）下LUT footprint和带宽成为瓶颈。
- **Inter-data Reuse**：同纬度多相机（I₀, I₁, I₂, I₃）的mapping index形状相同（球形矩形），仅中心索引(θc)不同（基于公式(2)）。因此4相机共享一份mapping index，投影输出通过各自θc offset区分。节省75% LUT footprint和带宽。
- **Intra-data Reuse**：同一mapping index内相邻元素(θn,ϕn), (θn+1,ϕn+1)可能指向同一输入像素(uk,vk)，差分编码可将这些相邻元素表示为(0,0)差分值——bit-precision从8-bit降至2-bit（256×256图像）。节省75%。
- 组合效果：18 camera @4纬度 → 94.4% LUT压缩。

从算法pipeline角度拆解：
以18相机iProj为例的mapping index处理流程：
```
// Pre-computation (offline, once per camera rig):
for each camera_i at same latitude:
    // 计算基准mapping index (仅第一相机)
    for (θ,ϕ) in output_spherical_grid:
        (u,v)_0 = iProj公式(2) → 存储为baseline LUT
    // Inter-data: 同纬度其他相机共享此mapping index
    // 仅需存储各自的中心偏移θc_i

// Intra-data Reuse: 差分编码baseline LUT
prev = (u₀,v₀)  // reference
for each (θn,ϕn) sequentially:
    (Δu,Δv) = (un-prev.u, vn-prev.v)
    // 大量相邻元素(Δu,Δv)=(0,0) → 2-bit编码足够
    store (Δu,Δv) with 2-bit precision
    prev = (un,vn)

// Runtime (CamPU index decoder unit):
// Step 1: 从压缩LUT读取差分序列(Δu,Δv)
// Step 2: Index recovery: 累加差分恢复(uk,vk)
// Step 3: 同纬度4相机共享此(uk,vk)
// Step 4: 各相机输出offset θc_i添加各自中心索引→得最终(θ,ϕ)
```

术语一般如何实现？如何使用？
Inter-data reuse仅适用于纬度对齐的多相机rig（相机在经度方向排列）。不满足时仅使用intra-data reuse（仍节省75% LUT）。CamPU index decoder unit硬件实现：mapping index LUT存储差分编码→index recovery unit（累加器）逐元素恢复原始mapping index→发送给image projection unit做remap。该方法是2D图像中利用空间locality进行压缩的经典思路在3D空间计算中的具体应用，可推广到任何固定相机rig的LUT-based坐标变换场景。

涉及论文标题：
- 33-CamPU_A_Multi-Camera_Processing_Unit_for_Deep_Learning-based_3D_Spatial_Computing_Systems.pdf

## Tree-based Speculative Inference (基于token树的投机推理)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Tree-based Speculative Inference是一种用于加速LLM生成的投机解码技术。它与传统sequence-based speculative inference的核心区别在于：使用小型投机模型（Small Speculative Model, SSM）同时预测多个候选token序列（而非单一序列），将这些候选token组织为一棵token树（Token Tree）——树的每个节点包含一个候选token，从根到任意节点的路径代表一条完整的候选token序列。LLM不再作为逐token增量解码器，而是一次性验证整棵token树中所有候选序列的正确性。由于同时考虑了多个候选分支（如top-5），token验证成功率从sequence-based的52-57%大幅提升至96-97%（stochastic decoding, LLaMA-7B/68M）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Tree-based Speculative Inference的算法pipeline（以expansion-based + greedy decoding为例）：
```
输入: prompt tokens S
输出: 生成的token序列

1. S = prompt
2. while True:
3.     // Step A: Speculation - 构造token tree
4.     N = {}  // token tree
5.     for each SSM_i (可并行):
6.         以S为输入，用SSM_i做推理，按expansion config <k1,k2,...,km>
7.         在每步取样top-k个token，构造子树
8.     N = Merge(N, 所有SSM的子树)  // Token Tree Merge (Definition 3.2)
9.     
10.    // Step B: Tree Verification - LLM并行验证整棵树
11.    O = TreeParallelDecode(LLM, N)  // 一次LLM forward pass
12.    // O[u] 是LLM对tree node u所在序列的next-token预测
13.    
14.    // Step C: Greedy或Stochastic Verification
15.    if greedy:
16.        u = root(N); V = []
17.        while exists v: parent(v)=u and token(v)=O[u]:
18.            V.append(token(v)); u = v
19.        V.append(O[u])
20.    else: // stochastic (MSS)
21.        V = VerifyStochastic(O, N)  // multi-step speculative sampling
22.    
23.    for t in V:
24.        S.append(t)
25.        if t == <EOS>: return S
```
Token tree construction有两种方法：
- **Expansion-based**：单个SSM在每步取top-k token（而非top-1），按预设向量<k1,...,km>在各步展开，构造token tree。原理：LLM选中的token即使不是SSM的top-1，也通常在top-5内。
- **Merge-based**：使用adaptive boosting无监督训练多个SSM（OpenWebText corpus做prompt→LLM生成→逐SSM用前一个SSM失败的样本finetune），然后按Definition 3.2合并所有SSM的输出为一个统一token tree。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Tree-based Speculative Inference的典型实现方式：
1. **SSM选择**：使用与LLM同模型家族的预训练小模型（如LLaMA-68M用于LLaMA-7B，OPT-125M用于OPT-13B/30B），这些模型使用相同数据集预训练，天然具有一定对齐度。也可以使用distilled、quantized、pruned的LLM变体。
2. **Token tree组织**：树结构在GPU memory中通过parent指针和DFS linearization存储。Tree的宽度（分支数）和深度（speculation步数）可根据GPU spare resources动态调整。
3. **Verification保证**：Theorem 4.2确保stochastic decoding下MSS生成的token分布与LLM原始incremental decoding等价（P_SpecInfer = P_LLM）。
4. **加速效果**：单节点multi-GPU场景1.5-2.5×加速，多节点2.4-2.8×加速，offloading-based 2.6-3.5×加速。每步平均验证2-4个token（vs incremental的1个）。
5. 开源实现：SpecInfer artifact (https://github.com/goliaro/specinfer-ae)，基于FlexFlow runtime。

涉及论文标题：
- 38-SpecInfer- Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification.pdf

## Multi-step Speculative Sampling (MSS, 多步投机采样)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-step Speculative Sampling (MSS)是SpecInfer提出的用于stochastic decoding下验证token tree的算法。当LLM使用随机采样（从概率分布sample而非argmax）生成token时，MSS保证verified output的分布与LLM原始incremental decoding的分布完全等价（Theorem 4.2），同时最大化能被验证的token数量。MSS的核心是rejection sampling的树形扩展：对token tree的每个节点，按顺序尝试其各子分支（对应不同SSM的预测），以概率min(1, P_LLM/P_SSM)接受token；若拒绝，则将LLM概率分布中该SSM预测分量减去（normalize residual），继续尝试下一个SSM分支；若所有SSM分支均被拒绝，从residual distribution中直接采样一个token。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MSS算法伪代码（VerifyStochastic, Algorithm 2 of SpecInfer）：
```
输入: LLM输出的token tree logits O, 推测的token tree N
      LLM参数 Θ_LLM, SSM参数集合 {Θ_SSM_j}
输出: 已验证的token序列 V

VerifyStochastic(O, N):
    V = []; u = root(N)
    while u 不是叶子节点:
        H = children(u)  // u的所有子节点集合
        while H 非空:
            s ~ random_index(H)           // 随机选一个子节点索引
            r ~ Uniform(0, 1)              // 随机数
            x_s = H[s]                     // 该节点对应的token
            // Rejection sampling:
            if r <= P(x_s | u, Θ_LLM) / P(x_s | u, Θ_SSM_s):
                V.append(x_s)              // 接受该token
                u = s                      // 移动到该子节点继续
                break
            else:
                // 拒绝：normalize residual distribution
                P(x | u, Θ_LLM) = norm(max(0, P(x | u, Θ_LLM) - P(x | u, Θ_SSM_s)))
                H.remove(s)
        if H 为空:  // 所有SSM预测均被拒绝
            x_next ~ P(x | u, Θ_LLM)      // 从residual分布直接采样
            V.append(x_next)
            break
    return V
```

关键数学保证（Theorem 4.2）：
P_SpecInfer(u_i | U; Θ_LLM, {Θ_SSM_j}) = P(u_i | U; Θ_LLM)
即MSS的输出分布恒等于LLM原始incremental decoding的输出分布。这是通过rejection sampling的标准性质保证的：每一步的接受/拒绝决策保持了目标分布的正确性。

Theorem 4.3进一步证明MSS的rejection概率uniformly lower than Naive Sampling（直接从LLM采样再检查是否在token tree中）：
P(reject | MSS) ≤ P(reject | NS)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MSS的实现要点：
1. **多SSM支持**：MSS天然支持多个SSM——对同一tree node，不同子分支可以来自不同SSM的预测。VerifyStochastic按SSM顺序逐个尝试接受。
2. **Residual normalization**：每次拒绝后，residual distribution = norm(max(0, P_LLM - P_SSM))。Max(0,·)保证非负，norm保证和为1。
3. **实际性能**：相比Naive Sampling，MSS的verified tokens数提升1.2-1.3×（Table 3, 5个数据集）。
4. 开源实现：https://github.com/goliaro/specinfer-ae（spec_infer.cc中的VerifyStochastic实现）。

涉及论文标题：
- 38-SpecInfer- Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification.pdf

## 8-bit Posit (Posit8)

术语解释
8-bit Posit（Posit8，本文使用Posit(8,1)，即1 exponent bit）是第三代universal number（Type III unum）系统的8-bit实例，设计为IEEE 754浮点数的直接替代方案。Posit数由四个字段组成：sign (1-bit)、regime (可变长度)、exponent (最多es=1 bits)、fraction (剩余bits)。与FP8不同，Posit具有tapered precision——数值越接近1精度越高——适合DNN weights和activations的Gaussian分布特征。

术语是什么？
Posit是John Gustafson于2017年提出的浮点数替代方案。8-bit Posit (Posit(8,1))的decoding公式为：`x = (-1)^s × 1.f × (2^(2^es))^k × 2^e`，其中k为regime值（通过统计sign bit后连续相同bit数得到run length m：若相同bit为0则k=-m，若为1则k=m-1）。Regime field起额外指数的作用，赋予Posit更广的动态范围（Posit(8,1)可表示2^(-12)至2^12）和tapered precision。Decoding从左到右：sign→regime (leading one counter)→exponent→fraction。Encoding则是将浮点指数分解为regime（除2^es）和exponent（模2^es），组装后round-to-even。与FP8 E4M3相比，Posit8对接近1的值有更多fraction bits但数值极大/极小时fraction bits少，这天然匹配DNN中权重/激活的分布（大部分值集中在0附近）。

从算法pipeline角度拆解：
以MobileBERT推理中一个GEMM输出激活的Posit8量化和后续使用为例：
```
# Step 1: GEMM输出为BF16 accumulator (形状: [batch, seq_len, hidden_dim])
accum = W @ X  # BF16 accumulator (32-bit fp mantissa semantics)

# Step 2: 将accum转为Posit8 (decoded to E5M4 for MAC)
# Decoding Posit8: sign + regime (leading one counter) + exponent + fraction
pos8_vals = encode_to_posit8(accum, es=1)
# range: [-4096, 4096], 饱和截断至[-4096, 4096]
# round-to-even when value < 2^(-12), round down to 0 instead of up

# Step 3: Fused operation - 所有element-wise操作在32-bit accumulator完成
fused_result = pos8_vals + residual  # residual add
fused_result = layer_norm(fused_result)  # layer norm in accumulator
fused_result = gelu(fused_result)  # activation in accumulator
fused_result = fused_result / sqrt(d_k)  # attention scaling in accumulator
# 最后一步才量化回Posit8
output = encode_to_posit8(fused_result, es=1)

# Step 4: GEMM用decoded Posit8 → E5M4 (5 exponent, 4 fraction) MAC
# Posit8 decoded: max 4 fraction bits, exponent range -12 to 12 → E5M4
```
训练时使用per-tensor scaling：activation gradients在量化前scale amax到64（而非Posit8最大值4096），因为tapered precision使大数的fraction bits极少，无法精确表示。

术语一般如何实现？如何使用？
Posit算术的实现通常通过HLS或RTL将Posit decoding/encoding作为独立的硬件模块。8-bit posit的训练通常在GPU上通过模拟实现：在PyTorch中使用自定义clipping函数截断值到Posit8的表示范围。本文开源代码（https://github.com/jeffreyyu0602/quantized-training）将Posit8的clipping和rounding嵌入PyTorch autograd的forward/backward hook中。Posit(8,2)有更大的range (2^(-24)至2^24)但更少的fraction bits，适合大型模型（如Whisperlarge, LLaMA 2）。Pytorch原生不支持Posit，需要自定义posit library（如SoftPosit, https://gitlab.com/cerlane/SoftPosit）或本文的模拟实现。

涉及论文标题：
- 39-8-bit Transformer Inference and Fine-tuning for Edge Accelerators.pdf

## FP8 (8-bit Floating Point, E4M3 / E5M2)

术语解释
FP8是NVIDIA提出的8-bit浮点数格式，包含两种变体：E4M3（4-bit exponent + 3-bit mantissa，高精度）和E5M2（5-bit exponent + 2-bit mantissa，宽动态范围），用于DNN训练和推理的混合精度方案。本文采用Hybrid FP8（E5M3 MAC format），可同时支持两种格式的GEMM。

术语是什么？
NVIDIA FP8格式的定义（图2）：(1) E4M3: `(-1)^s × 1.f × 2^(e-7)`。1 sign + 4 exponent + 3 mantissa bits。最大值±448。Forward pass用于weights和activations（需要更高精度）。(2) E5M2: `(-1)^s × 1.f × 2^(e-15)`。1 sign + 5 exponent + 2 mantissa bits。最大值±57344（支持±inf和NaN）。Backward pass用于gradients（需更宽动态范围但对精度不敏感）。FP8 MAC accumulation在高精度（BFloat16或FP32）中进行，与int8不同，FP8无需per-channel scaling factors。本文的Hybrid FP8 MAC用E5M3实现，可兼容E4M3和E5M2。

从算法pipeline角度拆解：
以FP8 LoRA微调的forward/backward pass为例：
```
# Forward Pass (E4M3)
x_fp8 = quantize_e4m3(x, amax)  # scale amax to 448 (E4M3 max)
# GEMM: E4M3 × E4M3 → BF16 accumulator
h = fused_gemm_eltwise(W_e4m3, x_e4m3)  # + residual + LN + activation + attn scaling
output = quantize_e4m3(h, amax)

# Backward Pass (E5M2)
grad_output_e5m2 = quantize_e5m2(grad_output, amax)  # scale amax to 57344 (E5M2 max)
# GEMM: E5M2 × E5M2 → BF16 accumulator for gradient computation
grad_W = grad_output_e5m2 @ x_e4m3_T  # weight gradient
grad_x = W_e4m3_T @ grad_output_e5m2   # input gradient
```

量化策略：NVIDIA的delayed scaling使用历史amaxes的列表，对当前iteration取max预测值作scale factor。训练中FP8仅应用于GEMM的inputs (如Micikevicius et al. 2022)，其余操作留在BF16。

术语一般如何实现？如何使用？
NVIDIA在H100/H200 GPU上提供FP8 Tensor Core（第4代Tensor Core），通过Transformer Engine库提供FP8支持：`fp8_autocast()` context manager自动将compatible layers转换为FP8。使用流程：(1) 用`fp8_autocast()`包裹forward pass → (2) transformer layers的Linear被自动替换为FP8 GEMM → (3) amax history在每次iteration更新 → (4) scale factor延迟一步更新（防止overflow）。PyTorch 2.1+通过`torch._scaled_mm`和`torch.float8_e4m3fn`/`torch.float8_e5m2`数据类型原生支持FP8。Megatron-LM、NeMo、TensorRT-LLM均支持FP8训练和推理。

涉及论文标题：
- 39-8-bit Transformer Inference and Fine-tuning for Edge Accelerators.pdf

## Operation Fusion (for Quantization Error Reduction)

术语解释
Operation Fusion在此论文context中指将Transformer中element-wise操作（residual add、layer normalization、GeLU/GELU activation、attention scaling）融合进前序GEMM操作，在32-bit accumulator中完成所有computation，最后一次性量化到8-bit，从而减少多次量化/反量化引入的舍入误差。

术语是什么？
传统8-bit量化（如int8）的pipeline：GEMM (int8) → dequant to FP32 → residual add (FP32) → quantize → layer norm (FP32) → quantize → activation (FP32) → quantize → attention scaling → quantize。每步量化引入独立rounding error，累积误差导致精度下降。Operation Fusion的核心思想：在GEMM后的accumulator（32-bit float）中连续执行residual + layer norm + activation + attention scaling，与accumulator的计算精度（FP32或BF16的mantissa宽度）共享中间精度，最后统一量化。这消除了中间量化/反量化步骤，将多次rounding error压缩为一次。

从算法pipeline角度拆解：
以MobileBERT中一个Transformer block的fused inference为例：
```
# 不融合baseline（多次量化）
q = Quant(W_q @ Quant(x))  # GEMM量化 → dequant
k = Quant(W_k @ Quant(x))
v = Quant(W_v @ Quant(x))
attn = Quant(Quant(q @ k^T) / sqrt(d_k))  # attn scaling后需量化
attn = Softmax(Dequant(attn))  # softmax需FP32输入
out = Quant(Quant(attn @ v) + residual)  # residual后量化
out = Quant(LayerNorm(Dequant(out)))  # LN后量化
out = Quant(GeLU(Dequant(out)))  # 激活后量化

# Operation Fusion（本文方法）
# GEMM + residual + LN + activation + attn scaling全部在accumulator中完成
q_accum = W_q @ x  # Posit8/FP8 MAC → BF16 accumulator
k_accum = W_k @ x
v_accum = W_v @ x
attn_accum = (q_accum @ k_accum^T) / sqrt(d_k)  # attn scaling in accumulator
attn_accum = softmax(attn_accum)  # softmax in BF16
out_accum = attn_accum @ v_accum + x  # residual in accumulator
out_accum = layer_norm(out_accum)  # LN in accumulator
out_accum = gelu(out_accum)  # activation in accumulator
out = Quant(out_accum)  # 仅一次量化！
```

融合优先级（按对accuracy影响降序，Table 1）：attention scaling > activation > layer norm > residual add。MobileBERT需要全部融合才能<1% accuracy loss；BERTbase无需融合即可达标。

术语一般如何实现？如何使用？
在本文开源代码（https://github.com/jeffreyyu0602/quantized-training）中，operation fusion通过PyTorch的自定义autograd Function实现：修改Transformer层的forward方法，将GEMM输出后的各element-wise操作串联在accumulator中执行，最后调用quantize函数。这与NVIDIA FP8 training中的`fp8_autocast`思想一致（但FP8 training仅fuse GEMM inputs）。在定制加速器上，vector unit的multi-stage pipeline（4 stages: arithmetic → exp → reduce → ReLU）天然支持此类fusion。

涉及论文标题：
- 39-8-bit Transformer Inference and Fine-tuning for Edge Accelerators.pdf

## Low-Rank Adaptation (LoRA)

术语解释
LoRA (Low-Rank Adaptation) 是一种参数高效微调（PEFT）方法，通过注入低秩可训练矩阵适配预训练模型到下游任务。本文创新地将LoRA适应到8-bit数据格式（Posit8/FP8），使LoRA权重merge和GEMM均在8-bit完成，消除了传统int8 LoRA需要dequant到高精度的问题。

术语是什么？
LoRA由Hu et al. (2021)提出，核心公式：`h = W0·x + ΔW·x = W0·x + α·B·A·x`，其中W0是预训练权重（frozen），ΔW = α·B·A是低秩更新（rank r << min(d_in, d_out)），A∈R^(r×d_in)，B∈R^(d_out×r)，α是scaling factor。仅训练A和B（参数量从d_in×d_out降至r×(d_in+d_out)，通常减少100-1000×）。传统LoRA用于int8量化的方案（如QLoRA）需将int8 W0 dequantize为BF16/FP32后再与BF16的LoRA merge，无法使用高效8-bit MAC。

本文的8-bit LoRA方案：`h = quant(W0_8 + α·quant(B_16)quant(A_16))·x`。LoRA矩阵B、A存储为BF16（提供足够精度的weight update），乘法前量化为8-bit（Posit8或FP8），merge到8-bit pretrained weights后整体再量化一次。这使得：(1) 所有GEMM均用8-bit MAC；(2) LoRA merge在8-bit完成；(3) 无需dequant pretrained weights。

从算法pipeline角度拆解：
```
# 传统int8 LoRA（QLoRA式）
W0_int8 = quantize_int8(W0)  # pretrained weights 8-bit
B_bf16, A_bf16 = init_bf16()  # LoRA matrices
# Forward:
W0_bf16 = dequantize(W0_int8)  # 反量化到BF16
delta = B_bf16 @ A_bf16  # BF16 matmul
h = (W0_bf16 + delta) @ x  # BF16 matmul -- 无法用8-bit MAC!

# 本文8-bit Native LoRA
W0_8 = quantize_8bit(W0)  # Posit8 or E4M3
B_bf16, A_bf16 = init_bf16()  # LoRA in BF16
# Forward:
B_8 = quantize_8bit(B_bf16)
A_8 = quantize_8bit(A_bf16)
delta_8 = B_8 @ A_8  # 8-bit GEMM
W_merged_8 = quantize_8bit(W0_8 + alpha * delta_8)  # merge + requantize in 8-bit
h = W_merged_8 @ x_quant  # 8-bit GEMM -- 全部用8-bit MAC!

# Backward (仅更新LoRA参数A, B):
grad_A = B_8^T @ grad_h @ x^T  # E5M2 GEMM
grad_B = grad_h @ (A_8 @ x)^T  # E5M2 GEMM
A_bf16 -= lr * dequant_bf16(grad_A)  # BF16 update
B_bf16 -= lr * dequant_bf16(grad_B)
```

术语一般如何实现？如何使用？
LoRA通常通过HuggingFace PEFT库实现：`from peft import LoraConfig, get_peft_model`，配置rank r、target_modules（如["q_proj","v_proj"]）、alpha scaling。训练时仅LoRA参数requires_grad=True。本文对LoRA的adaptation在开源代码中通过自定义quantized linear层实现。LoRA rank选择：MobileBERT/MobileBERTtiny用r=8应用于所有dense layers；RoBERTa用r=8仅应用于W_q和W_v。8-bit LoRA将fine-tuning内存从BF16的~500MB降低到~165MB（3× reduction for MobileBERTtiny）。

在Chameleon等多adapter LLM serving系统中，LoRA adapter以多rank（8/16/32/64/128）形式同时服务于不同下游任务。adapter weights不merge进base model，而是作为独立参数块存储在host memory（CPU）或GPU cache中，推理时通过MBGMM等batch gather kernel对batch内不同请求的不同rank adapter统一计算 `y = W_base·x + B_adapter·(A_adapter·x)`。adapter rank越大→推理延迟越高（rank 128的adapter loading+compute占TTFT的~60%）、memory占用越大（rank 128 adapter ~512MB per Llama-7B adapter），因此多adapter serving场景需考虑adapter rank对调度和缓存的影响。

在ELORA (FAST LIBRA) 的Multi-LoRA serving场景下，KV cache因含LoRA branch修正（KV_Cache_q,t = W_{k,v} q + A_t B_t q）而具有per-LoRA隔离性——不同LoRA的KV caches不可互换。这一特性使Multi-LoRA caching系统面临额外的"usage dependency"挑战：KV cache仅在其对应LoRA adapter也cached在HBM中时才有用。ELORA通过将LoRA和KV统一管理（unified caching pool + dependency tree）解决了此问题。

涉及论文标题：
- 39-8-bit Transformer Inference and Fine-tuning for Edge Accelerators.pdf
- 66-Chameleon- Adaptive Caching and Scheduling for Many-Adapter LLM Inference Environments..pdf
- 67-ELORA- Efficient LoRA and KV Cache Management for Multi-LoRA LLM Serving.pdf

## Tapered Precision

术语解释
Tapered Precision（渐变精度）是Posit数系统的核心特性：数值越接近±1，分配的fraction bits越多，精度越高；数值远离±1（极大或极小），fraction bits减少（regime bits增加），精度降低但动态范围扩大。这一特性基于DNN中weights和activations通常遵循Gaussian分布（集中在0附近）的观察。

术语是什么？
在Fixed-Width格式（如IEEE 754 FP8 E4M3）中，所有可表示值的fraction bits数恒定（3 bits），即uniform precision。Tapered precision则不同：通过variable-length regime field，接近1的值使用短regime（如k=0, regime=01, 仅2 bits），剩余bits分配给fraction（可达4-5 bits）；极大值使用长regime（如k=3, regime=11110, 5 bits），仅剩0-1 fraction bits。公式：`value = (-1)^s × 1.f × (2^(2^es))^k × 2^e`。对Posit(8,1)，min positive=2^(-12)，max=2^12，在1附近可达4 fraction bits（≈6 bit mantissa有效精度），在±4096仅有1 fraction bit。

从算法pipeline角度拆解：
对比FP8 E4M3和Posit8在MobileBERT量化推理中的行为（图4和图6）：
- E4M3：uniform 3-bit mantissa across all values [-448, 448]。MobileBERT stacked FFN产生wider activation distribution → E4M3的uniform precision更好覆盖。
- Posit8：接近0的值（大部分activations）有更多fraction bits → 精度更高。但MobileBERT stacked FFN产生大量大值activations（>64），这些区域的fraction bits少（tapered off）→ 精度下降。BERTbase无stacked FFN → Gaussian分布 → Posit8更好。
实际tradeoff分析（Table 2）：MobileBERT with Posit8 no-fusion F1=65.1 vs E4M3=82.7（E4M3显著更好，因tapered precision对大activation不利）。而BERTlarge Posit8 no-fusion F1=92.3 vs E4M3=93.0（差距缩小，因大模型activation分布更集中）。

术语一般如何实现？如何使用？
Tapered precision是posit格式的固有属性，由regime field实现。硬件上通过leading one counter统计regime长度→计算剩余fraction bits数。在量化和训练中，利用tapered precision的策略包括：(1) 选择合适的es参数（本文Posit(8,1) vs Posit(8,2)的对比表明，大模型倾向于更大的es以获得更宽的range）；(2) scaling amax到64而非4096（因大值fraction bits少，scaling到中等range利用更多fraction bits）；(3) 训练时per-tensor scaling按tensor的分布选择scale factor。Tapered precision的劣势是大值精度不足（Takum format尝试用对数渐变精度解决此问题）。

涉及论文标题：
- 39-8-bit Transformer Inference and Fine-tuning for Edge Accelerators.pdf

## Per-Tensor Scaling (in 8-bit Quantization Training)

术语解释
Per-Tensor Scaling是在8-bit量化训练中，对每个tensor独立应用一个scaling factor，将tensor值的amax（最大绝对值）映射到目标数据类型的可表示范围，使更多值落入可精确表示区域。本文用于8-bit fine-tuning的activation gradients量化。

术语是什么？
与int8量化常用的per-channel scaling（每个channel独立scale factor）不同，per-tensor scaling对整个tensor（如一个activation gradient tensor [batch, seq_len, hidden_dim]）使用单一scale factor。对于FP8 E5M2：scale factor = 57344 / amax（使amax对齐E5M2最大表示值）。对于Posit8：由于tapered precision，scale amax到64而非4096（因大值仅有1-2 fraction bits，无法精确表示；中等值有更多fraction bits），即scale factor = 64 / amax。Scale factor通常fused到前序操作中：在产生tensor的kernel末尾apply scaling，避免将高精度值写入memory再读回。

从算法pipeline角度拆解：
```
# Training backward pass中activation gradient的per-tensor scaling
def backward_with_scaling(grad_output, layer_state):
    # grad_output shape: [batch, seq, hidden]
    # Step 1: 使用历史amaxes预测当前scaling factor
    historical_amaxes = layer_state.amax_history  # 维护每个tensor的历史amax列表
    predicted_amax = max(historical_amaxes[-K:])  # 取最近K步的max
    scale = 64 / predicted_amax  # Posit8: scale to 64; FP8 E5M2: 57344/amax
    
    # Step 2: 计算gradient（在前序GEMM中用缩放后的值）
    grad_input_scaled = grad_output * scale  # apply scaling
    grad_input_8 = quantize_8bit(grad_input_scaled)
    
    # Step 3: GEMM with 8-bit gradients
    grad_W = grad_input_8 @ x_8^T  # E5M2 GEMM
    
    # Step 4: 更新amax history
    current_amax = max(abs(grad_output))
    layer_state.amax_history.append(current_amax)  # 滑动窗口
    if len(layer_state.amax_history) > window_size:
        layer_state.amax_history.pop(0)
    
    return grad_W
```

对于部分任务（如SQuAD + MobileBERT），activation gradient分布极宽，单一loss scaling（全局scale）不足，per-tensor scaling不可或缺。与int8的per-channel scaling相比，per-tensor scaling硬件开销更低（无需per-channel scale/zero-point存储和在线计算），但要求数据格式（如Posit8/FP8）本身有足够的动态范围。

术语一般如何实现？如何使用？
NVIDIA FP8 training中per-tensor scaling通过delayed scaling recipe实现：每一步根据上一步的amax计算scale factor（延迟一步以避免overflow）。Transformer Engine的`fp8_autocast`内置此机制。本文开源代码自定义了per-tensor scaling的quantizer类，在forward/backward hook中维护amax history并计算scale。关键超参数：history window size（取K个历史amax的max）、Posit8的target amax（本文found 64 optimal）。

涉及论文标题：
- 39-8-bit Transformer Inference and Fine-tuning for Edge Accelerators.pdf

## Post-Training Quantization (PTQ)

术语解释
PTQ是在模型训练完成后，不经过额外训练（fine-tuning），直接对模型weights和/或activations进行低精度量化的技术。本文采用PTQ评估FP8和Posit8的Transformer推理accuracy，并通过operation fusion减小PTQ的量化误差。

术语是什么？
与Quantization-Aware Training (QAT)不同，PTQ不需要在量化过程中重新训练——它直接对已训练好的模型应用量化。核心步骤：(1) 收集calibration data（少量representative samples），统计各tensor的activation值分布；(2) 确定量化参数（scale factor, zero-point等，int8需要；本文FP8/Posit8不需要scale/zero-point）；(3) 将weights/activations截断/舍入到目标精度。PTQ的优势是无需训练代价、部署简单；缺点是精度可能低于QAT（尤其对低bit-width如4-bit）。本文展示FP8/Posit8 PTQ通过operation fusion可在大多数模型上实现<1% accuracy loss vs BF16。

从算法pipeline角度拆解：
本文PTQ的具体流程（以MobileBERT SQuAD推理）：
```
# Step 1: 模型加载（无训练，直接使用pretrained weights）
model = load_huggingface_model("google/mobilebert-uncased")

# Step 2: 逐层量化（PTQ, 无需calibration的统计信息，因FP8/Posit8不需scale factor）
for layer in model.encoder.layers:
    # 量化weights
    layer.q_proj.weight = quantize_to_posit8(layer.q_proj.weight.data)
    layer.k_proj.weight = quantize_to_posit8(layer.k_proj.weight.data)
    # ... all linear layers
    
# Step 3: Inference with operation fusion (PTQ的精度保证)
def fused_forward(x):
    # GEMM + residual + LN + activation + attn_scaling全部在accumulator
    qkv = quantize(attn_weights @ quantize(x))  # Posit8 GEMM
    # ... subsequent fused operations with single final quantization

# 输出：F1 score on SQuAD v1.1
```
本文Table 2展示了逐级增加fusion的PTQ accuracy变化：No Fusion → +Attn Scaling → +Activation → +LayerNorm → +Residual（F1从65.1逐步回升至89.4, 接近BF16=89.9, MobileBERT Posit8）。

术语一般如何实现？如何使用？
常见PTQ工具：(1) PyTorch quantization API (`torch.quantization.quantize_dynamic`, `torch.ao.quantization`)；(2) NVIDIA TensorRT (INT8 PTQ with calibration)；(3) HuggingFace Optimum (`optimum.quantization`). 本文的FP8/Posit8 PTQ不需要calibration（直接按格式max/min截断），简化了PTQ流程。对于需要scale factor的格式（如int8），PTQ通常需要少量calibration data（100-1000 samples）统计activation范围。

Anda（HPCA 2025）将PTQ拓展到activation精度搜索：复用weight-only PTQ的calibration data（数千token, 128 random sequences），通过自适应精度组合搜索算法（Adaptive Precision Combination Search）在32次迭代内为4类关键激活张量（A_qkv, A_o, A_u, A_d）搜索最优mantissa长度组合（1-16 bit连续可调），无需retraining或backward propagation。精度评估通过校准集上的前向pass完成，每次迭代约等于一次前向传播时间。相比GPTQ（weight-only PTQ），Anda的搜索速度快约10×。

涉及论文标题：
- 39-8-bit Transformer Inference and Fine-tuning for Edge Accelerators.pdf
- 79-Anda_Unlocking_Efficient_LLM_Inference_with_a_Variable-Length_Grouped_Activation_Data_Format.pdf

## Block Floating Point (BFP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block Floating Point (BFP) 是一种介于定点（INT）和浮点（FP）之间的数值表示格式：将一组数值划分为groups，组内所有元素共享一个公共指数（shared exponent），每个元素保留独立的符号位和mantissa（尾数）。与标准FP32/FP16每个元素独享指数不同，BFP通过共享指数大幅减少指数存储开销，同时保留了比纯定点更宽的动态范围。BFP格式由两个关键参数决定：**group size (GS)**——共享指数的元素数量；**mantissa length (M)**——每个元素保留的尾数位数。BFP转换过程（FP16→BFP）：(1) 将FP16张量按GS分组；(2) 每组内取最大指数作为共享指数；(3) 其他元素的尾数根据指数差值右移对齐；(4) 超出M位的尾数截断；(5) 全零尾数表示数值0。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
BFP在LLM推理中的激活量化流程（Anda, GS=64, M可变）：
```
# FP16 activation tensor → BFP conversion
# Input: FP16 activations A ∈ R^{H×W}, group_size=GS, mantissa_bits=M

for group_idx in range(0, H*W, GS):
    group = A[group_idx : group_idx+GS]
    
    # Step 1: Find max exponent in group
    exp_max = max(extract_exponent(x) for x in group)  # 5-bit exponent
    
    # Step 2: Align mantissas to shared exponent
    for i, x in enumerate(group):
        exp_diff = exp_max - extract_exponent(x)
        # Right-shift mantissa by exp_diff, truncate to M bits
        aligned_mant[i] = (extract_mantissa(x) >> exp_diff) & ((1<<M)-1)
        # Zero detection: if all M bits are 0, value is 0
        if aligned_mant[i] == 0: zero_flag[i] = 1
    
    # Step 3: Store as BFP element
    # Each element: sign(1b) + mantissa(M b)
    # Group metadata: shared_exponent(5b)
    
# BFP FP-INT GeMM: group内dot product为INT算术
# result = Σ (mant_i × weight_i) × 2^{shared_exp} × weight_scale
```

Anda论文（HPCA 2025）的实验表明：LLM activation对BFP mantissa长度的敏感度因模型和模块而异。OPT系列模型对mantissa缩减更不敏感（可直接移除5-bit），LLaMA系列较敏感（仅能移除4-bit）。同一模型内，A_qkv（Q/K/V投影）对精度最敏感需更多mantissa位，A_d（下投影）可激进压缩。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
BFP的三种主要使用方式（按mantissa长度灵活性分类）：(1) **Uni-Length BFP**（VS-Quant、BOOST、FIGNA、Flexpoint）：所有组使用固定mantissa长度（4-16 bit），硬件实现简单但无法适应不同模块的精度需求；(2) **Multi-Length BFP**（FAST、DaCapo、FlexBlock）：提供2-3种预设mantissa长度选项（如2b/4b或4b/8b/16b），chunk-serial计算，灵活性有限；(3) **Variable-Length BFP**（Anda）：支持1-16 bit连续可调mantissa，bit-serial计算使cycle数与mantissa长度成正比（vs bit-parallel固定cycle数），适合精度敏感度不同的LLM模块。BFP计算时共享指数消除了组内的指数对齐和normalization开销，将FP乘加简化为INT乘加+最终移位。对于weight-only quantized LLM (W4A16)，BFP激活可与INT4权重直接执行INT dot-product。

涉及论文标题：
- 79-Anda_Unlocking_Efficient_LLM_Inference_with_a_Variable-Length_Grouped_Activation_Data_Format.pdf

## Weight-Only Quantization (W4A16)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight-Only Quantization是LLM推理中的一种量化策略：仅将模型权重（weights）量化为低精度整数（如INT4），而保持激活值（activations）为高精度浮点（如FP16），形成W4A16方案。与此相对的是Weight-Activation Quantization (W8A8)，两者均量化。Weight-only方案的优势：(1) 权重占模型存储的主体（如LLaMA-7B约14GB FP16），4-bit量化可将模型大小减少近4×；(2) 保持激活为FP16避免了激活中的outlier对精度的严重破坏，因为激活outlier对模型准确率影响远大于权重量化误差。代价是推理时执行FP-INT GeMM（浮点激活×整数权重），能耗约1.7×于W8A8 INT-only操作。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Weight-Only Quantization在LLM推理中的pipeline（以W4A16g128为例）：
```
# Offline PTQ: 权重量化
for each linear layer weight W ∈ R^{out×in}:
    # 按group_size=128分组
    for g in range(0, in, 128):
        W_group = W[:, g:g+128]
        scale[g] = max(|W_group|) / 7.0     # INT4 range [-7,7]
        W_int4[:, g:g+128] = round(W_group / scale[g])
        W_int4 = clamp(W_int4, -8, 7)        # 4-bit signed

# Online Inference: FP16 activation × INT4 weight
# A_qkv ∈ FP16^{seq×hidden}, W_q_int4 ∈ INT4^{hidden×hidden}
# FP-INT GeMM: C[i,j] = Σ_k A_fp16[i,k] × (W_int4[k,j] × scale[k//128,j])
#   = Σ_groups (Σ_k_in_group A_fp16[i,k] × W_int4[k,j]) × scale_g[j]
```

LLM中涉及FP-INT GeMM的4类关键模块（Anda论文分类）：(1) A_qkv（Q/K/V投影，激活×W_q/W_k/W_v）；(2) A_o（输出投影，激活×W_o）；(3) A_u（FFN上投影，激活×W_up）；(4) A_d（FFN下投影，激活×W_down）。Anda论文的Fig.2显示这些FP-INT GeMM在sub-4K token应用中占总操作>90%，在10K+序列中仍占多数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
主流weight-only量化工具：(1) **GPTQ**（ICLR 2023）：基于OBQ的逐层量化，使用Hessian信息确定量化顺序；(2) **Omniquant**：训练-free PTQ，支持W4A16g128方案；(3) **AWQ**：通过per-channel scaling保护显著权重。硬件支持：(1) NVIDIA GPU通过dequantize INT4→FP16后用Tensor Core执行FP16 matmul，或有专用FP-INT kernel（如NVIDIA FP-INT GeMM kernel）；(2) 专用加速器如FIGNA（HPCA 2024）和Anda（HPCA 2025）直接支持FP-INT操作，避免dequantization开销。Anda进一步将FP16激活替换为变长BFP格式，消除反复的格式转换。

涉及论文标题：
- 79-Anda_Unlocking_Efficient_LLM_Inference_with_a_Variable-Length_Grouped_Activation_Data_Format.pdf

## FP-INT GeMM Operations

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FP-INT GeMM (General Matrix Multiplication) 是weight-only quantized LLM中特有的矩阵乘法操作：一个操作数为浮点格式（FP16 activation），另一个为整数格式（INT4/INT8 weight）。与标准FP-FP或INT-INT GeMM不同，FP-INT GeMM需要处理两种不同数值格式的乘法-累加。计算过程：(1) INT权重乘以per-group scale factor以恢复浮点范围；(2) FP激活与恢复后的权重值执行浮点乘加；或(3) 将FP激活转为等价整数格式后执行INT乘加以降低硬件开销。在weight-only quantized LLM (W4A16)中，FP-INT GeMM占总计算的>90%（sub-4K context），是推理延迟和能耗的主要瓶颈。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FP-INT GeMM的四种计算方案（Anda论文Fig.8对比）：
```
方案(a) GPU FP-FP: INT4 weight → dequantize → FP16
     FP16 activation × FP16 weight → FP32 accum → FP16 output
     缺点: 反复转换 + FP计算开销

方案(b) GPU FP-INT: 专用FP-INT单元
     FP16 activation × INT4 weight → FP32 accum → FP16 output
     缺点: exponent对齐和normalization硬件复杂

方案(c) FIGNA: FP16 activation存储 → 动态转换BFP(14b mantissa) → INT计算
     INT(BFP mant) × INT4 weight → INT32 accum → FP32 → FP16
     缺点: FP16存储无压缩 + 每次计算需动态转换

方案(d) Anda: Anda格式activation直接存储 → INT bit-serial计算
     Anda(mant M b) × INT4 weight → INT32→FP32→Anda output
     优势: 存储压缩 + 无转换开销 + 变长mantissa按需计算
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
硬件实现：(1) GPU: NVIDIA的FP-INT GeMM kernel（如CUTLASS中W4A16 kernel，将INT4 weight dequantize后送入FP16 Tensor Core）；(2) 专用加速器: FIGNA的bit-parallel INT单元（14-bit固定mantissa BFP），Anda的bit-serial INT单元（1-16 bit可变mantissa BFP）。关键设计trade-off：bit-parallel（如FIGNA）在固定精度下面积效率更高，bit-serial（如Anda）在可变精度下利用率和能效更高。Anda通过自适应精度搜索使平均mantissa长度降至4-8 bit（vs FIGNA固定14/11/8 bit），在1%精度损失下BOPs reduction达2.44-3.31×。

涉及论文标题：
- 79-Anda_Unlocking_Efficient_LLM_Inference_with_a_Variable-Length_Grouped_Activation_Data_Format.pdf

## Anda Data Format (Variable-Length Grouped Activation Data Format)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Anda是HPCA 2025提出的一种变长分组激活数据格式（Variable-Length Grouped Activation Data Format），属于Block Floating Point (BFP)格式的扩展。Anda格式的核心特征：(1) **共享指数**：每64个激活元素共享一个5-bit指数（group size=64）；(2) **变长尾数**：每个元素独立1-bit符号+可变长度mantissa（1-16 bit连续可调），不同LLM模块可使用不同mantissa长度；(3) **模块级精度分配**：以4元组[M_qkv, M_o, M_u, M_d]指定4类关键激活张量的mantissa长度，各模块内所有层的同类型张量使用相同长度。与已有BFP格式的核心区别：Uni-Length（固定长度）和Multi-Length（2-3种预设）均不支持连续可变精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Anda格式在LLM推理中的使用流程：
```
# Offline: 自适应精度搜索 (compile time)
Input: LLM model L, calibration data D, accuracy tolerance δ
Output: [M_qkv, M_o, M_u, M_d]

# 搜索算法（Algorithm 1 in paper）:
Q = PriorityQueue([4,4,4,4], ..., [13,13,13,13])  # BOPs最小优先
best_comb = null
while iterations < 32:
    curr = Q.pop_min_BOPs()
    if EvaluateAccuracy(L, D, curr) >= (1-δ) × FP16_baseline:
        best_comb = curr  # 更新最佳
        # 生成邻居候选: 每个维度分别减1
        for dim in [qkv, o, u, d]:
            neighbor = curr.copy(); neighbor[dim]--
            Q.push(neighbor)

# Online: Anda格式推理
for each Transformer block:
    # A_qkv: FP16 → BFP转换 (GS=64, M=M_qkv)
    # 组内: max_exp = max(extract_exp(x)), mant[i] = x.mant >> (max_exp - x.exp)
    # 截断到M_qkv bit
    # INT bit-serial dot-product: Σ mant[i] × W_int4[i]
    # 结果 × 2^max_exp × W_scale → FP32 accum
```

实例（WikiText2, 1% loss）：OPT-30B最优组合[6,4,5,4]（A_qkv=6b, A_o=4b, A_u=5b, A_d=4b），LLaMA-13B最优组合[7,7,6,7]。OPT模型的A_d可压缩到4-bit而LLaMA的A_d需7-bit，反映不同架构的精度敏感度差异。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Anda格式的硬件支持需要三个关键组件：(1) Bit-plane data layout memory：将Anda值的sign/mantissa/exponent按bit-plane view组织存储，变长尾数仅影响地址深度；(2) Bit-serial PE (APU)：mantissa长度直接决定计算cycle数（M-bit需M个compute cycle），bit-parallel变体如FIGNA-M8/M11需分离硬件实现；(3) Runtime Bit-plane Compressor (BPC)：在线将FP16输出压缩为Anda格式。RTL综合（16nm, 285MHz, 0.8V）：总面积2.17mm²，总功耗81.18mW。Anda以per-module粒度分配精度，比per-layer方法（如HAWQ）搜索更快，直接集成到weight-only PTQ部署pipeline。

涉及论文标题：
- 79-Anda_Unlocking_Efficient_LLM_Inference_with_a_Variable-Length_Grouped_Activation_Data_Format.pdf

## BOPs (Bit Operations) Metric

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BOPs (Bit Operations) 是衡量量化神经网络计算复杂度的指标：统计所有乘加操作在bit-level的总操作次数。与只计算操作数（MACs/FLOPs）不同，BOPs将每次乘法的bit宽度纳入考量，更精确地反映量化对计算量的实际影响。对于INT4×M-bit mantissa乘法，BOPs = 4×M per operation。1次FP16×INT4操作等价于约64 BOPs（作为近似的bit-level计算复杂度）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Anda论文使用BOPs作为精度搜索算法中的代价函数，无需完整模型评估即可快速估计组合的计算效率：
```
# 精度组合 [M_qkv, M_o, M_u, M_d] 的BOPs估计:
BOPs([M_qkv, M_o, M_u, M_d]) = Σ_{layer} (
    OPS(A_qkv_layer) × 4 × M_qkv +      # INT4 weight × M_qkv-bit mantissa
    OPS(A_o_layer)   × 4 × M_o +
    OPS(A_u_layer)   × 4 × M_u +
    OPS(A_d_layer)   × 4 × M_d
)

# BOPs reduction相对于FP16 baseline:
# BOPs_reduction = BOPs_FP16 / BOPs_combination
# 其中 BOPs_FP16 ≈ OPS_total × 64 (1次FP16×INT4 ≈ 64 BOPs)
```

BOPs作为轻量级proxy metric使Anda的搜索算法每次迭代仅需1次前向pass（评估精度），而非昂贵地执行硬件模拟。这使Anda搜索比GPTQ快约10×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
BOPs广泛应用于量化研究：(1) ResQ（ICCV 2023）：使用BOPs作为混合精度搜索的优化目标；(2) HAWQ（ICCV 2019）：结合Hessian信息引导per-layer BOPs分配。在Anda中，BOPs作为优先队列（priority queue）的排序key，确保每次迭代首先评估BOPs最小的候选组合，加速向高效精度组合收敛。BOPs的局限：仅衡量计算量不包含memory access energy，因此Anda最终使用system-level cycle-accurate simulation评估完整能效。

涉及论文标题：
- 79-Anda_Unlocking_Efficient_LLM_Inference_with_a_Variable-Length_Grouped_Activation_Data_Format.pdf

## Prefill and Decode Stages in LLM Inference (LLM推理的预填充与解码阶段)

术语是什么？
LLM（Large Language Model）推理分为两个阶段：Prefill（预填充）和Decode（解码/自回归生成）。Prefill阶段：接收用户输入的prompt tokens，并行处理整个prompt序列，计算所有层的attention和FFN，产生第一个输出token。Decode阶段：以自回归（auto-regressive）方式逐个生成后续token，每次只处理一个新token，但需要访问之前所有token的KV Cache。两个阶段的计算特征截然不同：prefill是compute-bound（处理大批量tokens，GEMM的M维度大），decode是memory-bound（每次只处理1个token，但需要读取大量KV Cache数据）。

从算法pipeline角度拆解术语：
LLM推理pipeline（以Llama3-8B为例）：
```
输入: prompt = ["What", "is", "the", "capital", "of", "France", "?"]
// === Prefill Phase ===
for token t_i in prompt (并行处理7个tokens):
    for each Transformer layer:
        // Multi-Head Attention: Q,K,V projection (GEMM)
        Q = W_Q @ input_embedding  // [7, 4096] x [4096, 4096]
        K = W_K @ input_embedding
        V = W_V @ input_embedding
        // Scaled Dot-Product Attention (BMM + Softmax)
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
        // KV Cache: 保存K,V供后续decode使用
        KV_cache.append(K, V)
        // FFN: 两个linear层+激活
        FFN(x) = W_2 @ ReLU(W_1 @ x)

// === Decode Phase (迭代直到生成EOS或达到max_len) ===
output_token = first_token
while output_token != EOS:
    for each Transformer layer:
        // 只计算新token的Q，复用cache中的K,V
        Q_new = W_Q @ embedding(output_token)  // [1, 4096] x [4096, 4096]
        // Attention: Q_new与所有历史K交互
        Attention(Q_new, KV_cache.K, KV_cache.V)
        // 追加新K,V到cache
        KV_cache.append(K_new, V_new)
        // FFN: 只处理当前token
        FFN(x_new)
    output_token = argmax(logits)

// 关键维度变化: Prefill M=batch*seq_len (可达131K+), Decode M=batch (通常1~8)
```

Prefill阶段的GEMM是大型矩阵乘法（M大→compute-bound），大量使用Tensor Core。Decode阶段的GEMM是小型矩阵乘法（M=1~8→memory-bound），受限于KV Cache读取带宽。这种GEMM异构性（compute-bound vs memory-bound在attention层和FFN层之间切换）是LLM推理的独特特征，也是AMALI需要建模warp instruction分布差异的根本原因。

术语一般如何实现？如何使用？
主流LLM推理框架（vLLM, TensorRT-LLM, llama.cpp等）均区分prefill和decode阶段。优化策略：(1) Prefill优化——通过FlashAttention、kernel fusion减少HBM访问，利用大batch充分利用Tensor Core；(2) Decode优化——通过KV Cache量化/压缩减少内存占用，通过continuous batching提高GPU利用率；(3) Chunked prefill——将长prompt拆分为多个chunk与decode交替执行。AMALI作为底层kernel性能预测工具，需要分别为prefill和decode阶段的kernel提供准确的cycle预测（prefill MAPE 15.56%, decode MAPE 34.90%）。

在 CPU AU 平台上（AUM 论文），prefill/decode 还呈现出独特的 AU 使用差异：
- **AU Usage Pattern**：Prefill 的 GEMM（8192×4096×22016, batch=16×seq=512）适合 AMX TMUL 加速（40.57 TFLOPS），AMX cycle ratio 14.4%。Decode 的 GEMV（16×4096×22016, batch=16×1 token）因 M 维度小，AVX 更高效（3.87 TFLOPS），AMX cycle ratio 仅 1.5%。
- **ARI 公式**：Prefill 阶段 ARI = 6(1/d_model + 3/(B×L))^{-1}，Decode 阶段 ARI = 6(1/d_model + 3/B)^{-1}，其中 B=batch size, L=input sequence length, d_model=model dimension。ARI 越大 AU 使用越密集。
- **资源 bound 差异**：Prefill 的 Backend Bound 59.9%，Serializing Operations 20.4%；Decode 的 Backend Bound 71.8%，DRAM Bound 31.2%（因 KV cache 读取的密集内存访问）。
- **SLO 差异**：Prefill 关注 TTFT (Time-To-First-Token)，Decode 关注 TPOT (Time-Per-Output-Token)。不同场景 TTFT/TPOT SLO 不同：chatbot 250ms/100ms, code completion 75ms/150ms, summarization 1.5s/100ms。
- AUM 通过 FCFS 调度 prefill prompts，对 decode tokens 使用 LAG 指标（Σ(dTPOT - e_token)）分析实时进度，自适应调整 AU 资源分配。

涉及论文标题：
- 3-AMALI- An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs.pdf
- 90-AUM_Unleashing_HPCA-2026-Wang.pdf

## Structured Sparse Pattern（结构化稀疏模式）

术语是什么？
结构化稀疏模式（Structured Sparse Pattern）是DNN剪枝中为平衡硬件效率与模型精度而施加的非零元素空间约束规则。与无结构逐元素剪枝不同，它规定保留的非零元素必须遵循特定空间布局（块状、条状、向量状），将稀疏访问转为规则内存访问。Fractal论文将所有结构化稀疏模式统一抽象为四个设计因素：(1) Pruning Pattern尺寸H×W——集体剪枝的元素维度；(2) Selection Region尺寸P×Q——相邻剪枝单元统一排名排序的区域，区域内维持均匀稀疏率；(3) 稀疏率；(4) 模式方向（行向/列向）。现有模式如BW（Block Wise, 块粒度剪枝）、TW（Tile Wise, 细粒度tile剪枝）、VW（Vector Wise 2:4，每4元素保留2个）均为上述因素的特化组合。

从算法pipeline角度拆解术语：
以BERT GEMM算子Y=XW为例，结构化稀疏模式决定W的pruning mask的空间约束：
1. BW pattern 32×32：W被划分为32×32块，每块内元素全保留或全清零，剪枝算法在块粒度排序importance。
2. TW pattern 1×32：W的每行按32列分组，组内按importance排序决定保留/剪枝。
3. VW 2:4 pattern：每4个连续元素必须恰好保留2个，需要Sparse Tensor Core硬件支持。
每种模式对模型表达力施加不同程度限制：粗粒度模式（BW）accuracy损失大但硬件效率高；细粒度模式（VW）accuracy保留好但需要特殊硬件。Fractal提出Hybrid模式——在多个tiling层级同时应用稀疏——搜索accuracy-performance最优解。

术语一般如何实现？如何使用？
(1) 计算weight importance scores（magnitude/gradient/LAMP等）；(2) 按Selection Region对importance分组；(3) 区域内按scores排序，按目标稀疏率剪去低分元素；(4) 以对应存储格式（BlockELL/TileWise/cuSPARSELt 2:4）存储非零元素和索引；(5) 推理时对应稀疏算子库解码索引跳过零值计算。PyTorch中通常通过剪枝后应用mask并调用对应sparse kernel。

涉及论文标题：
- 40-Fractal- Joint Multi-Level Sparse Pattern Tuning of Accuracy and Performance for DNN Pruning.pdf

## Mask Diversity（MD，掩码多样性）

术语是什么？
Mask Diversity（MD）是量化结构化稀疏模式对模型精度潜在影响的度量，定义为给定稀疏模式和目标张量形状下所有可能剪枝掩码的组合数。对N元素张量保留K个元素，无结构pruning的MD=C(N,K)。结构化模式因空间约束减少有效组合数，MD随之减小。实际使用log(MD)作为可比较指标——log(MD)越大表示模式越灵活、对精度约束越小。Fractal论文证实log(MD)与speedup呈反比关系，刻画了模式设计的accuracy-performance权衡。

从算法pipeline角度拆解术语：
Fractal用MD建立稀疏模式与accuracy之间的快速关联桥梁，使auto-tuning无需每次都做完整retraining。例如75%稀疏率1024×1024 GEMM：无结构EW log(MD)极大；2:4 VW每4选2→每组C(4,2)=6种，MD=6^262144, log(MD)≈262144×log(6)；BW 32×32每块保留256个→MD=C(1024,256), log(MD)显著更小。Fractal在搜索中可视化log(MD) vs speedup散点图观察trade-off关系，配合importance score threshold做精确过滤。

术语一般如何实现？如何使用？
纯组合数学计算，不涉及实际剪枝。在auto-tuning中作为快速预筛选指标——丢弃MD过低的模式，减少后续不必要的pruning和evaluation。Fractal将MD作为importance score的补充：MD提供expressiveness理论下界，importance score提供具体accuracy影响的近似。

涉及论文标题：
- 40-Fractal- Joint Multi-Level Sparse Pattern Tuning of Accuracy and Performance for DNN Pruning.pdf

## Multi-Level Sparse Tiling（多级稀疏Tiling）

术语是什么？
多级稀疏Tiling是将稠密GEMM的多级循环Tiling与结构化稀疏结合的概念。现代GPU稠密GEMM使用多级tiling利用各级内存层次（CUTLASS的thread block tile/warp tile/thread tile三级）。Fractal观察到现有结构化稀疏模式实质上是在**单一固定tiling层级**跳过计算——BW在coarse block级跳过，TW/VW在fine tile/vector级跳过。多级稀疏Tiling提出在**多个tiling层级同时应用稀疏**：外层tile穿孔跳过整个coarse block，内层tile穿孔在保留block内进一步跳过fine元素，形成Hybrid稀疏模式，实现更灵活的accuracy-performance权衡。

从算法pipeline角度拆解术语：
以1024×1024 GEMM C[i,j]=ΣA[i,k]B[k,j]为例：
1. Dense Tiling：Split(i,4,8,32)→I0^4 I1^8 I2^32; Split(k,32,8,4)→K0^32 K1^8 K2^4
2. 一级稀疏（Outer）：Perforate(K0,28)——32个K0 block中保留28个
3. 二级稀疏（Inner）：Perforate(K1,5)——每个保留K0 block内8个K1 tile保留5个
4. 形成Hybrid PatternIR：`I0^4 I1^8 K0_28^32 K1_5^8 K2^4 I2^32 J^1024`
效果：既跳过coarse block节省计算，又在block内跳过fine tile进一步削减，而保留的block数量多维持数据复用。比单级BW更精确匹配weight的intrinsic structure（如attention head边界），accuracy损失更小。

术语一般如何实现？如何使用？
(1) 剪枝从外到内（coarse→fine）贪心执行：最外层Selection Region内按importance排序决定coarse block保留/跳过，保留的block内递归对内层执行同样操作；(2) 算子采用分层索引存储格式（Fractal-ELL），每级稀疏循环维护独立index vector；(3) 编译框架自动生成多级稀疏循环代码。

涉及论文标题：
- 40-Fractal- Joint Multi-Level Sparse Pattern Tuning of Accuracy and Performance for DNN Pruning.pdf

## LLM Inference Phase Characterization (Prompt vs Token Generation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM Inference Phase Characterization是对生成式LLM推理中两个固有阶段——prompt computation（prompt processing phase）和token generation（autoregressive decoding phase）——在计算特性、内存使用、功耗、延迟和批处理效率方面的系统化表征。Splitwise论文基于Azure production traces（coding和conversation服务）和DGX-A100/DGX-H100硬件进行了一系列实验，揭示了两个phase的根本性差异：(1) prompt phase处理所有input tokens并行执行单次forward pass生成第一个output token和完整KV-cache——compute-intensive，GPU利用率高，功耗接近TDP，吞吐随batch tokens增加先升后降（2048 tokens后下降）；(2) token generation phase逐token串行生成，每step仅处理1个new token + 全部累积KV-cache——memory bandwidth/capacity bound，GPU compute单元大部分时间stall等待HBM数据，功耗仅约TDP的50%且几乎不随batch size变化，吞吐随batch增大持续提升直至memory满。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LLM推理两个phase的算法pipeline对比（以BLOOM-176B, single request, P=prompt_tokens, G=generated_tokens）：

```
// === Prompt Computation Phase (1次forward, P tokens并行) ===
// Input: prompt_tokens shape [P]
hidden_states = Embedding(prompt_tokens)  // [P, d_model]
for layer in 1..L:  // L=70 for BLOOM-176B
    // Self-Attention: Q·K^T is [P × P] matmul, compute-intensive
    Q, K, V = Linear_QKV(hidden_states)  // each [P, n_heads, d_head]
    attn = Softmax(Q @ K^T / sqrt(d_head)) @ V  // [P, n_heads, d_head]
    hidden_states = hidden_states + Linear_O(attn)
    // FFN: two large matmuls, compute-intensive
    hidden_states = hidden_states + Linear_2(GELU(Linear_1(hidden_states)))
    // KV-cache写入: 本层K, V保存供后续token generation使用
    KV_cache[layer].K = K  // 每层 ~ P × n_heads × d_head × 2(FP16) bytes
    KV_cache[layer].V = V
first_token = Sample(LM_Head(hidden_states[-1]))  // 取最后position
KV_cache_size = P  // 累积P个position

// === Token Generation Phase (G次forward, 每次1 token) ===
for step in 1..G:
    new_hidden = Embedding(output_tokens[-1])  // [1, d_model]
    for layer in 1..L:
        Q, K, V = Linear_QKV(new_hidden)  // each [1, n_heads, d_head]
        // Q·K^T: [1, H, d] × [H, d, cache_size] → data from HBM
        // compute/byte ≈ 2 FLOP/B (memory-bound on H100)
        K_full = Concat([KV_cache[layer].K, K])  // read all history K from HBM
        V_full = Concat([KV_cache[layer].V, V])
        attn = Softmax(Q @ K_full^T / sqrt(d_head)) @ V_full
        new_hidden = new_hidden + Linear_O(attn)
        new_hidden = new_hidden + Linear_2(GELU(Linear_1(new_hidden)))
        KV_cache[layer].K = Concat([KV_cache[layer].K, K])  // append to HBM
        KV_cache[layer].V = Concat([KV_cache[layer].V, V])
    next_token = Sample(LM_Head(new_hidden))
    if next_token == EOS: break
```

关键特征数值对比（BLOOM-176B on DGX-H100, tensor parallelism=8）：
| 特性 | Prompt Phase | Token Generation Phase |
|------|-------------|----------------------|
| 每forward处理的token数 | P (数百~数千) | 1 (per seq in batch) |
| 瓶颈类型 | Compute-bound | Memory bandwidth/capacity bound |
| GPU利用率 | 高 (SM, Tensor Core near full) | 低 (SM频繁stall等HBM) |
| 功耗 vs batch size | 随batch增长，接近TDP | 几乎不变 (~50% TDP) |
| Throughput vs batch | 先升后降 (peak ~2048 total tokens) | 持续升 (至HBM满, batch≈64) |
| TTFT/TBT scaling | TTFT linear to P | TBT仅2×增长(batch 1→64) |
| E2E时间占比 | 小 (1500-token prompt ≈ 6-token gen) | 主导coding/conversation请求 |
| 内存使用特征 | KV-cache快速增长(P tokens同时写入) | KV-cache缓慢增长(每step+1 token) |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LLM Inference Phase Characterization的实用价值：
1. **系统设计指导**：Phase characterization直接驱动Splitwise的phase splitting设计——prompt phase需要高compute+中等memory capacity硬件，token phase需要中等compute+高memory capacity/bandwidth硬件。这使Splitwise-HA（H100 prompt + A100 token）等异构部署可行。
2. **批处理策略**：Prompt phase的throughput在2048 total tokens后下降→prompt machine MLS限制batch≤2048 tokens。Token phase的throughput持续提升到memory满→token machine MLS尽可能增大batch。
3. **功耗管理**：Token phase对power cap不敏感（≥50% cap几乎无延迟影响）→Splitwise-HHcap对token GPU施加power cap节省cluster power budget。Prompt phase对power cap敏感→prompt GPU维持full TDP。
4. **硬件选型**：H100 vs A100对比——H100 compute 3.43× A100，memory bandwidth仅1.64×，capacity无增长。对token phase而言H100的额外compute浪费（memory bound），A100具有更好的Perf/$和Perf/W。
5. **通用性**：此characterization适用于所有基于transformer decoder的autoregressive LLM（GPT系列、Llama系列、BLOOM、OPT等），因为prompt/token两阶段是autoregressive decoding的固有属性。MoE变体同样具有这两个phase。

涉及论文标题：
- 41-Splitwise_Efficient_Generative_LLM_Inference_Using_Phase_Splitting.pdf

## Temporal Differential Computing（时序差分计算）

术语解释
一种针对扩散模型推理的算法加速技术：利用扩散模型去噪过程中相邻时间步（timestep）输入数据的高度相似性，将每步的激活值 X_t 分解为上一时间步的激活值 X_{t-1} 与差值 ΔX_t 之和（X_t = X_{t-1} + ΔX_t），仅对delta值而非原始值执行卷积等线性运算，利用delta值更窄的数值范围用更低位宽表示，从而减少计算开销。

术语是什么？
扩散模型在去噪过程的每一时间步需要对整个U-Net网络完整执行一次推理。由于每步仅对图像做微小去噪，相邻时间步的U-Net输入之间高度相似——两者的差值（temporal delta, 沿时间维度的差值）数值范围远小于原始激活值。在Stable-Diffusion的典型层中，原始激活值需要FP16表示，而temporal delta可使用INT3表示（entropy从3.37降至1.60，降低2.11×），且无需增加量化误差（因为量化分辨率取决于scale factor s，而delta的自然范围缩小允许不改变s的情况下用更少bit表示）。

具体流程：
1. 第t-1时间步的激活值 X_{t-1} 和卷积结果 Conv(X_{t-1}) 已保存。
2. 第t时间步计算 delta ΔX_t = X_t - X_{t-1}（通常由上层输出自然地得到）。
3. 对delta执行低位宽卷积 Conv(ΔX_t)（如INT3×FP16乘法，替代FP16×FP16）。
4. 利用卷积的线性性恢复最终结果：Conv(X_t) = Conv(X_{t-1}) + Conv(ΔX_t)。

类比：已知123×7=861，计算124×7时只需算1×7=7（delta乘法更简单），再加到861得到868。

与Spatial Differential的关键区别：Diffy [15]使用spatial differential（沿图像空间维度的相邻像素差值），但扩散模型的中间激活含噪声，空间平滑性差，spatial delta数值分散；temporal differential的delta数值范围远更窄，是更高效的选择（论文Fig. 5对比了两种delta的分布）。

从算法pipeline角度拆解：
给定U-Net的第t时间步推理，pipeline逻辑为：

```
# 输入：上层的delta输出 ΔX_t（已在上层计算完成，on-chip）
# 权重：预训练U-Net权重 W（FP16）

# Step 1: 差分卷积（对delta执行，替代对raw值执行）
ΔY_t = Conv2D(W, ΔX_t)          # INT3×FP16 → 节省约3.3×算术成本

# Step 2: 差分激活函数（需要sign-mask approximation保证non-linear correctness）
ΔY'_t = DifferentialReLU(ΔY_t, sign(Y_{t-1}))   # 见Sign-Mask ReLU Approximation

# Step 3: 差分Group Normalization（使用相邻timestep平均值）
ΔY''_t = DifferentialGN(ΔY'_t, μ_avg, σ²_avg)

# Step 4: 更新raw值（在DRAM侧由NDP engine完成）
Y_t = Y_{t-1} + ΔY''_t          # read-add-write in DRAM
```

核心约束：仅线性算子（卷积、上/下采样、dropout、残差连接）可直接使用差分计算——因为F(X+ΔX) = F(X) + F(ΔX)对线性算子成立。非线性算子（ReLU/SiLU、GroupNorm、Softmax/Attention）需要特殊处理才能维持全网络差分。

术语一般如何实现？如何使用？
Temporal differential computing的实现要求：
1. **离线分析**：对目标扩散模型在各时间步的各层统计delta值的数值分布，确定可用的最小bitwidth（保证量化误差不增加的min bitwidth）。
2. **量化方案**：论文中INT3可表示>98%的delta值（非outlier部分）；剩余~1.86%的outlier需用FP16保持精度——由此引出outlier-aware量化设计。
3. **差分算子实现**：线性算子（Conv）直接对delta执行低位宽计算，节省compute；非线性算子需要额外近似（sign-mask, GN averaging）将delta forwarding维持在全网络。
4. **Delta更新机制**：每个时间步结束后，使用当前delta更新DRAM中的raw激活值和sign bits（供下个时间步使用），可由NDP引擎在DRAM侧执行以避免加载整个raw tensor。
5. **适用范围**：主要用于扩散模型（Stable Diffusion, Guided-Diffusion等）的U-Net backbone（卷积占~76.4%计算时间）。Attention层（占<1%计算）直接回退到raw值计算。

涉及论文标题：
- 42-Cambricon-D_Full-Network_Differential_Acceleration_for_Diffusion_Models.pdf

## Sign-Mask ReLU Approximation（差分ReLU sign-mask近似）

术语解释
在temporal differential computing中，为使非线性激活函数ReLU无需加载FP16原始激活值即可对delta值执行，利用相邻时间步的sign一致性假设（sgn(Y_t) ≈ sgn(Y_{t-1})），将差分ReLU简化为delta值与1-bit sign的AND mask操作的一个近似方法。

术语是什么？
ReLU定义为 ReLU(Y) = Y · sgn(Y)（其中sgn为正则1、非正则0）。差分ReLU需要计算 ΔY' = ReLU(Y_t) - ReLU(Y_{t-1}) = Y_t · sgn(Y_t) - Y_{t-1} · sgn(Y_{t-1})。由于Y_t = Y_{t-1} + ΔY_t，直接计算需要同时加载Y_{t-1}和Y_t（均为FP16），这将引入大量memory traffic——这正是Diffy baseline性能下降23.4%的根本原因。

Sign-mask近似的关键洞察：因为在扩散模型的去噪过程中ΔY_t相对于Y_{t-1}非常小（delta值集中在极窄范围），所以在绝大多数情况下sign(Y_t) ≈ sign(Y_{t-1})（在Stable-Diffusion中成立概率为99.59%）。利用此假设：
  ΔY' ≈ ΔY_t · sgn(Y_{t-1})
这意味着只需从DRAM加载1-bit的sign tensor Sgn_{t-1}（而非16-bit的raw value Y_{t-1}），差分ReLU变为对delta输出做AND mask操作（sign=1保留delta值，sign=0将delta值置零）。

额外考虑：(1) SiLU激活函数（扩散模型实际使用的激活）定义为 SiLU(x) = ReLU(x) · σ(x)，论文发现用纯ReLU替换SiLU精度损失<0.5%，因此可安全替换。(2) sign bits需要跨时间步更新：在第t步使用Sgn_{t-1}做近似，同时计算Sgn_t = sgn(Y_t)供第t+1步使用——由NDP engine在DRAM侧自动更新。

从算法pipeline角度拆解：
```
# 输入：差分卷积输出的delta张量 ΔY_t (INT3)
# 输入：从DRAM加载的上步sign bits Sgn_{t-1} (1-bit/tensor element)
# 硬件上的SFU（Special Function Unit）执行：

for each element i in ΔY_t:
    if Sgn_{t-1}[i] == 1:     # 上步raw值Y_{t-1}为正
        ΔY'_t[i] = ΔY_t[i]    # 保留delta值
    else:                      # 上步raw值Y_{t-1}为负/零
        ΔY'_t[i] = 0           # mask out (ReLU结果为0)
```
此操作在硬件上是一个简单的AND gate（delta值 AND sign bit），硬件成本极低（仅占1.08%面积/1.05%功耗）。

术语一般如何实现？如何使用？
1. **Sign bit存储**：在DRAM中为每个raw激活张量额外维护一个separate sign bit tensor（每元素1-bit），确保对sign bits的访问不因DRAM burst granularity浪费带宽。
2. **Sign bit更新**：每个时间步卷积/ReLU完成后，delta值（经压缩）传输到DRAM，由NDP engine执行：decompress delta → convert INT3 to FP16 → FP adder: Y_t = Y_{t-1} + ΔY_t → extract sign: Sgn_t = sgn(Y_t) → 写入sign bit tensor供下一时间步使用。此过程完全在DRAM侧完成，不占用PE array和core-chip bandwidth。
3. **精度保证**：sign-mask近似的精度依赖于假设sgn(Y_t) ≈ sgn(Y_{t-1})的成立比例。在扩散模型U-Net各层中该假设成立率>99.5%，近似引入的总精度损失<0.5%。对于成立比例更低的场景（如delta变化剧烈的层），sign bits的更新机制保证了不会累积误差——每个时间步更新后sign bits反映的是真实的sgn值。
4. **与outlier处理的交互**：sign-mask操作作用于所有delta值（inlier INT3和outlier FP16），因为sign仅由raw值Y_{t-1}的符号决定，与delta本身的精度无关。

涉及论文标题：
- 42-Cambricon-D_Full-Network_Differential_Acceleration_for_Diffusion_Models.pdf

## SSMP（Scaling Sub-matrix Partition / 缩放子矩阵划分）

术语解释
MECLA (ISCA'24)提出的参数高效矩阵划分方法，将LLM大规模权重矩阵分解为小尺度source sub-matrices (SS)和derived sub-matrices (DS)，每个DS由对应SS乘以一个scaling scalar得到，从而大幅减少权重存储和计算量。

术语是什么？
SSMP是一种利用权重矩阵内部二维空间相似性的压缩方法。核心观察：LLM超级大的权重矩阵维度（如LLaMA的4096×11008）使得pigeonhole principle（鸽笼原理）频繁发生——大矩阵中必然存在相似的子矩阵块。SSMP将权重矩阵划分为多个region，每个region包含1个source sub-matrix (SS，尺寸[x, y])和(nx·ny-1)个derived sub-matrices (DS)，每个DS通过对应的SS乘以一个scaling scalar生成：DS[i,j] = SS × S[i,j]。配置由四元组(x, y, nx, ny)表示：x/y为SS的垂直/水平维度，nx/ny为DS在垂直/水平方向的扩展数量。例如配置(8,8,4,4)时，一个4096×11008权重矩阵被划分为n=4096·11008/(8·4·8·4)=128×344=44032个region，每个region含1个8×8 SS和15个8×8 DS。存储需求从原始的Dx·Dy个参数降至n·x·y + n·(nx·ny-1)个scalar——例如Bloom-7B FFN层从67.1MB降至5.2MB（节省92.3%）。SSMP可配置为标准（GLUE精度损失<2%）和aggressive（<5%损失）模式。

从算法pipeline角度拆解：
SSMP在LLM推理中的算法流程（以Bloom-7B单个线性层，配置(8,8,4,4)为例）：

```
// 离线阶段：SSMP weight preparation
输入：预训练权重 W ∈ R^{Dx × Dy}（如4096×11008）
输出：SSMP参数 {WSS, S}
  num_regions_x = Dx / (x * nx)  // 4096/32 = 128
  num_regions_y = Dy / (y * ny)  // 11008/32 = 344
  WSS = zeros(num_regions_x, num_regions_y, x, y)      // [128, 344, 8, 8]
  S   = zeros(num_regions_x, num_regions_y, nx, ny)    // [128, 344, 4, 4]

// 在线推理：权重on-the-fly恢复（实际在MECLA硬件上不显式恢复）
  def weight_lookup(out_ch, in_ch):
      region_x = out_ch // (x * nx)
      region_y = in_ch  // (y * ny)
      ss_row = (out_ch % (x * nx)) // nx
      ss_col = (in_ch  % (y * ny)) // ny
      ds_x = (out_ch % (x * nx)) // nx 的 quotient 部分 → (out_ch // x) % nx
      ds_y = (in_ch  % (y * ny)) // ny 的 quotient 部分 → (in_ch // y) % ny
      if ds_x == 0 and ds_y == 0:
          return WSS[region_x, region_y, ss_row, ss_col]  // SS itself
      else:
          // DS: scale SS by scalar
          return WSS[region_x, region_y, ss_row, ss_col] * S[region_x, region_y, ds_x, ds_y]

// 矩阵乘法（利用PSum重用，详见kernel调度层）
  y = x @ W^T  // 实际通过PSum reuse完成，不完全展开W
```

关键数学性质：对于同一SS对应的不同DS，若它们作用于相同的input channels，则其partial sum (PSum)相差一个scalar倍——PSum_DS = PSum_SS × S[ds_x, ds_y]。这一性质是SSMP不仅能减少存储、还能减少计算的基础。

术语一般如何实现？如何使用？
1. **配置选择**：SSMP超参数{x,y,nx,ny}通过grid search + successive halving在搜索空间{2,4,8,16}^4中确定。Standard setting选accuracy degradation <2% (GLUE)的最小模型；Aggressive选<5%的最小模型。由于SSMP专为推理效率设计（fine-tuning参数也因SSMP大幅减少），搜索开销可接受。
2. **Fine-tuning转换**：预训练LLM不能直接应用SSMP——需要通过SSMP-oriented fine-tuning（见"SSMP Fine-tuning with Forget Factor"条目）将权重转换为SSMP格式。与LoRA本质区别：LoRA为训练参数效率，SSMP为推理memory-compute efficiency。
3. **硬件适配**：SSMP需要专用硬件（如MECLA）才能充分利用PSum reuse——在GPU上SSMP仅减少weight memory但需额外recovery计算，加速比有限（2.32-2.88× on V100）。专用PE array + scaling multiplier可将reuse转化为实际计算节省（4.25-5.28× vs naive V100 inference）。
4. **压缩率与模型规模**：更大模型（如Bloom 7B vs 1B7）在同一accuracy loss下可实现更高压缩率——因更大的权重矩阵维度增加sub-matrix similarity概率。
5. **与蒸馏/量化对比**：SSMP压缩结果可比肩KD和MiniLLM（Rouge-L仅差0.2 vs MiniLLM 7B，但参数量少57.1%），并提供连续压缩选项（5B-7B）而非仅fixed-size small model。

涉及论文标题：
- 44-MECLA_Memory-Compute-Efficient_LLM_Accelerator_with_Scaling_Sub-matrix_Partition.pdf

## SSMP Fine-tuning with Forget Factor（带遗忘因子的SSMP微调）

术语解释
MECLA提出的将预训练LLM转换为SSMP格式的参数高效微调方法。通过freeze预训练权重、仅训练微小的SSMP参数（WSS和S）、并引入forget factor σ（遗忘因子）逐步从预训练权重过渡到纯SSMP权重。

术语是什么？
直接训练SSMP格式的LLM from scratch成本极高。SSMP fine-tuning（Algorithm 2）提供了一种参数高效的后训练转换方法：
- 输入：预训练LLM权重W（全部freeze）
- 创建微小可训练参数：WSS（source sub-matrix weight，尺寸n·x·y）和S（scaling parameter，尺寸n·nx·ny），参数量远小于原始W
- 引入forget factor σ（标量，初始=1.0）：W_combined = σ·W + (1-σ)·W_SSMP
- 训练时对σ施加正则化惩罚 L(σ)，推动σ→0
- 当σ < 10^-4（实验确定阈值）时，安全移除预训练权重W，仅保留SSMP参数{WSS, S}

该过程类似LoRA的低参数微调范式，但目的根本不同：LoRA为降低训练成本，而SSMP fine-tuning为让推理时的memory和compute更高效。

从算法pipeline角度拆解：
SSMP fine-tuning算法流程（Algorithm 2伪代码）：
```
Algorithm: SSMP Fine-tuning
Input: Pre-trained LLM weight W (frozen)
Output: SSMP-style LLM weight W' = {WSS, S, σ≈0}

1  Initialization:
2    WSS, S, σ ← Init()  // σ = 1.0
3    FreezeParam(W)       // 预训练权重不可训练
4
5  Training Loop:
6    for trainSet in trainLoader:
7      W_DS = WSS × S                    // broadcast mul: [n,x,y] × [n,nx,ny]
8      W_SSMP = Concat(WSS, W_DS)        // 完整SSMP权重重建
9      W_new = σ · W + (1-σ) · W_SSMP    // 混合预训练+SSMP
10     Output = Forward(trainSet, W_new)  // 标准forward pass
11     Loss = LLM_loss(Output) + λ·|σ|   // LLM loss + σ正则化
12     Update(WSS, S, σ)                  // 仅更新SSMP参数
13   end
14
15 return W' = {WSS, S, σ}               // σ已接近0
```
训练参数：learning rate {5e-4, 1e-4, 5e-5}，batch size {16, 32}，20 epochs。使用knowledge distillation（大模型为teacher）提升fine-tuning效果（如RoBERTa large/Bloom 7B/LLaMA 13B作为teacher）。

术语一般如何实现？如何使用？
1. **Forget factor初始化和退火**：σ从1.0开始，使得训练初期几乎完全使用预训练权重（W_combined ≈ W），然后逐步过渡到SSMP权重。正则化项λ·|σ|推动σ线性衰减。当σ magnitude < 10^-4时手动移除预训练分支。
2. **参数效率**：可训练参数量 = |WSS| + |S| + 1（σ）。例如Bloom-7B FFN层原始权重4096×11008=45M参数，SSMP配置(8,8,4,4)的可训练参数仅128×344×(64+16)≈3.5M参数（~7.8% of original）。
3. **任务特定fine-tuning**：每个下游任务（GLUE各task、Dolly、WikiText2）需独立fine-tune——因SSMP配置和σ退火过程依赖具体任务和模型。但与full fine-tuning相比，参数效率高使得开销可接受。
4. **策略与LoRA的对比**：LoRA学习低秩偏置ΔW=BA加到冻结权重上；SSMP学习完整权重矩阵的压缩表示{WSS,S}并通过σ逐步替换冻结权重。LoRA不减少推理memory（需同时加载W和BA），SSMP减少83.6%推理memory access。

涉及论文标题：
- 44-MECLA_Memory-Compute-Efficient_LLM_Accelerator_with_Scaling_Sub-matrix_Partition.pdf

## Autoregressive LLM Inference（自回归LLM推理）

术语解释
LLM推理的核心特征：每次生成一个token，每个新token的计算依赖于所有之前生成的token和完整模型权重，导致每次token生成需完整遍历模型全部参数。

术语是什么？
根据概率链式法则 P(x1,...,xn) = ∏P(xn|x1,...,xn-1)，LLM将已生成的token序列作为新输入，预测下一个token。一次推理迭代（Algorithm 1）：输入序列→所有decoder layers（每层QKV Linear→Attention→FFN Linear）→输出token→拼接到输入序列→下一迭代。该过程重复数十到数千次直至生成完整输出或达到max length。与BERT等encoder的非自回归推理（权重仅需访问一次）或GPT-2等小型decoder模型相比，LLM的自回归生成导致权重参数被反复访问——每生成一个新token都要从memory hierarchy完整读取模型权重。这使LLM推理的memory footprint随输出长度线性增长。

从算法pipeline角度拆解：
LLM autoregressive推理的算法伪代码（来自MECLA Algorithm 1）：
```
Algorithm: Autoregressive Transformer Inference
Input:  Input prompt I, Transformer model M, max output length Lmax
Output: Generated output sequence Sout

1  X ← I                          // 初始化输入为prompt
2  Sout ← {}                      // 空输出序列
3  while len(Sout) ≤ Lmax and Sout[-1] ≠ [EOS]:
4    for Layer in M:              // 遍历所有decoder layers
5      if Layer is attention layer:
6        q, k, v ← Layer.QKV_Linear(X)     // QKV线性投影
7        A ← softmax(q × k^T / sqrt(M.dim)) // 注意力计算
8        X ← A × v                          // 加权value
9        X ← Layer.Linear(X)                // 输出线性层
10     else:                        // FFN层
11       X ← Layer(X)              // FFN Linear + Activation
12     end
13   end
14   Sout.append(decode(X))        // 将输出解码为token
15   X ← Sout                      // 更新输入为全部已生成序列
16 end
17 return Sout
```
关键特征：(1) 权重遍历——每次while循环都要完整遍历所有Layer的权重（line 4-13），无法cache在PE local storage（因权重太大且下次使用前需跑完所有其他layers）；(2) Matrix-Vector Multiplication——decode阶段batch=1，每次MatMul是GEMV而非GEMM，缺少input tensor reuse，arithmetic intensity极低；(3) 线性层主导——QKV Linear (line 6)和FFN Linear (line 9/11)占>98%计算和memory，attention计算（line 7-8）占比<2%（当token length<1024时）。

术语一般如何实现？如何使用？
1. **Prefill vs Decode阶段**：Prefill阶段处理input prompt（batch=prompt_len），compute-bound（GEMV→GEMM）；Decode阶段每次1 token（batch=1），memory-bound（GEMV），是MECLA优化的主要目标。
2. **KV Cache**：为避免重复计算历史token的K/V，标准实现使用KV cache存储每层的Key和Value——Attention只用新token的Q与cache中的K计算。但MECLA主要优化线性层，KV cache作为辅助存储在DDR中。
3. **Memory Wall问题**：以LLaMA-7B为例，嵌入维度4096，FFN维度11008。单层QKV权重3×4096×4096=150M参数≈300MB（FP16），FFN权重3×4096×11008≈406MB（FP16）。单层总权重~700MB远超GPU L2 cache（~40MB），每token必须从HBM完整重载。生成32 token需访问~14GB权重数据。

涉及论文标题：
- 44-MECLA_Memory-Compute-Efficient_LLM_Accelerator_with_Scaling_Sub-matrix_Partition.pdf

## Sub-matrix Similarity in LLM Weights（LLM权重的子矩阵相似性）

术语解释
MECLA提出的观察：LLM超大的权重矩阵中，由于pigeonhole principle（鸽笼原理），存在显著的二维空间子矩阵相似性——相似的权重子矩阵块反复出现，可利用此相似性进行压缩和计算复用。

术语是什么？
LLM权重矩阵维度的极大扩展（如LLaMA 4096×11008 vs GPT-2 768×3072）使得input/output channel数量远超传统Transformer。根据pigeonhole principle：当元素数量远大于可能状态数时，必然存在重复/相似模式。MECLA将这一观察从传统工作的channel-wise（一维）相似性扩展到spatial-wise（二维）——在整个权重矩阵的二维空间中寻找相似子矩阵。这种相似性表现为：某些子矩阵块之间存在近似scalar倍的缩放关系。SSMP通过将相似子矩阵用共同的SS+不同scalar表示来压缩。实验验证：更大模型（Bloom 7B vs 1B7）在同一accuracy loss下可实现更高压缩率——因更大矩阵维度增加了sub-matrix similarity的发生概率，与pigeonhole principle预测一致。

从算法pipeline角度拆解：
子矩阵相似性在LLM权重矩阵中的数学表达：
```
给定权重矩阵 W ∈ R^{Dx × Dy}（如4096×11008）
定义region R_{i,j} 为以SS_{i,j}为source的 (x·nx)×(y·ny) 子区域
  R_{i,j} 包含 nx·ny 个 size [x,y] 的sub-matrix blocks
  其中 block(0,0) = SS_{i,j}（source sub-matrix）
       block(p,q) = SS_{i,j} × S_{i,j}[p,q]（derived sub-matrix, (p,q) ≠ (0,0)）
  
SSMP假设：R_{i,j}内所有block可由SS_{i,j}通过scalar乘法近似表示
即：∥block(p,q) - SS_{i,j}·S_{i,j}[p,q]∥_F ≤ ε

压缩率 = 1 - (n·x·y + n·(nx·ny-1)) / (Dx·Dy)
        = 1 - 1/(nx·ny) - 1/(x·y·nx·ny)  // 近似（忽略SS自身）
```
这一性质也出现在之前的工作中——如[64][87]发现权重张量中的similarity/duplication可被利用。MECLA的贡献是将此从channel-wise推广到2D spatial-wise，并为计算复用（而非仅存储压缩）设计相应硬件。

术语一般如何实现？如何使用？
1. **相似性检测**：论文未显式检测权重矩阵中的相似子矩阵——而是通过SSMP fine-tuning强制权重服从SSMP结构（W_SSMP逼近原始W），利用梯度下降"发现"最优SS和scaling参数使重构误差最小。
2. **与pruning的差异**：Pruning移除"不重要"权重（基于magnitude），SSMP用共享SS+scalar表示"相似"权重——保留所有output channel但共享参数，避免pruning的稀疏计算开销。
3. **与量化正交**：SSMP可与量化叠加——SS和scaling scalar可进一步量化（如INT8存储），论文中MECLA采用INT8 PE array + FP16 scaling scalar。
4. **适用范围**：主要指FFN线性层（gate/up/down projection）和QKV线性层的权重矩阵。Attention的Q×K^T和P×V计算不适用（动态matrix，维度随token数变化）。少量无法partition的矩阵区域用标准dense MatMul补偿。

涉及论文标题：
- 44-MECLA_Memory-Compute-Efficient_LLM_Accelerator_with_Scaling_Sub-matrix_Partition.pdf

## SSM-based Global Convolution Models（基于状态空间模型的全局卷积模型）

术语解释
SSM-based Global Convolution Models是一类使用状态空间模型（State Space Model, SSM）生成卷积滤波器，并通过FFT在频域执行全局卷积的序列建模架构。它们替代Transformer的self-attention层，以O(l log l)的次二次方复杂度处理长序列输入。

术语是什么？
SSM-based Global Convolution是self-attention的高效替代方案。其核心思想：(1) 使用Linear Time Invariant (LTI) SSM的参数(A, B, C, D)生成长度匹配输入的卷积滤波器K = {CA^0B, CA^1B, ..., CA^{l-1}B}；(2) 利用卷积定理——时域卷积等价于频域逐点乘法——通过FFT将输入u和滤波器K分别变换到频域，逐点相乘后IFFT得到卷积输出。复杂度O(l log l)远优于self-attention的O(l²)。为降低A的非对角形式导致的高昂滤波器生成复杂度O(lm³)，A被约束为复数对角矩阵diag(C^m)，将复杂度降至O(lm)。代表性模型：S4、S4D、H3、Hyena等。H3模型在WikiText103数据集上的perplexity（21.0）接近同规模Transformer（20.6），且在Long-Range Arena (LRA) benchmark上优于self-attention模型。

从算法pipeline角度拆解：
以一个长度为l=4K的输入序列u，hidden dimension d=768，SSM state size m=64的H3模型为例：
```
1. 输入x (l×d) 经3个FC层 → Q, K, V (均为l×d)
2. 对每个hidden dimension i (0..d-1):
   a. 1D Conv: K_i 经短卷积(small kernel)平滑
   b. PointMult: shift_K_i ⊙ V_i → gated_V_i
   c. SSMConv (全局卷积):
      - 状态传递: 将l分为C=l/L个chunk (L=2048)
      - For each chunk c:
        * State Update: x_c = A^L·x_{c-1} + Mux·u_c
          (Mux ∈ C^{m×L} 由参数(A,B)构造的Vandermonde矩阵)
        * FFTConv: y_fft = IFFT(FFT(K) ⊙ FFT(u_c))
        * Output Projection: y_c = Mxy·x_{c-1} + y_fft + D·u_c
          (Mxy ∈ C^{L×m} 由参数(A,C)构造的Vandermonde矩阵)
   d. PointMult: Q_i ⊙ SSMConv_i → result_i
3. 结果经FC层 → 输出
```
关键性质：滤波器K的元素通过递归关系关联（K_{t+1} = K_t × A），使得状态向量x_c可总结所有前序chunk的信息，chunk间无需重新计算完整FFT。

术语一般如何实现？如何使用？
1. **Filter生成**：利用A的diagonal性质，K的各元素可通过循环复数乘法生成：K_i = C·diag(A)^i·B，复杂度O(l·m)而非O(l·m³)。
2. **Chunk-wise convolution**：将长序列分块处理使得每chunk的FFT fit in on-chip SRAM，避免DRAM往返。Chunk size L是2的幂次以使用Radix-2 FFT。State passing的额外计算开销（~2次batched matrix multiplication）被DRAM带宽节省所抵消。
3. **与Parallel Scan选择**：FFT convolution比parallel scan快~3.3×（在128K序列时），但parallel scan提供更大模型灵活性（无需预计算滤波器）。
4. **Head dimension并行**：不同hidden dimension的SSMConv运算互相独立，可在多GPU/多PE上完全并行。
5. 开源实现：H3 https://github.com/danfu09/H3；S4 https://github.com/state-spaces/s4。

涉及论文标题：
- 45-VGA_Hardware_Accelerator_for_Scalable_Long_Sequence_Model_Inference.pdf

## State Passing Algorithm（状态传递算法）

术语解释
State Passing Algorithm是SSM-based全局卷积模型的核心算法，将长输入序列分割为固定大小的chunk，在每个chunk内独立执行FFT卷积，并通过状态向量在chunk间传递上下文信息。

术语是什么？
State Passing利用LTI SSM的递归性质——前序输入对当前输出的影响可通过状态向量x_c完全概括。将长度N的序列分为C=N/L个长度为L的chunk，定义状态更新方程：x_c = A^L·x_{c-1} + Mux·u_c，输出投影方程：y_c = Mxy·x_{c-1} + K∗u_c + D·u_c。其中Mux ∈ C^{m×L}和Mxy ∈ C^{L×m}是Vandermonde矩阵，由SSM参数(A,B,C)构造。这使得chunk size L可自由选择以fit在SRAM内，避免了长序列FFT必须反复访问DRAM的瓶颈。当chunk数C过大时（如L过小），状态向量递归链x_0→x_1→...→x_{C-1}的串行依赖会成为新瓶颈。

从算法pipeline角度拆解：
```
Input: 长序列 u (长度N), SSM参数 (A,B,C,D), chunk size L
       C = ceil(N/L)
Output: 输出序列 y (长度N)

初始化: x_0 = 0 (零向量 ∈ C^m)
预计算: Mux[0..L-1] = {A^{L-1}B, A^{L-2}B, ..., A^0B}
        Mxy[0..L-1] = {C·A^0, C·A^1, ..., C·A^{L-1}}
        A^L = diag(A)^L  // 对角矩阵的L次幂

For c = 0 to C-1:
    u_c = u[c·L : (c+1)·L]  // 取当前chunk
    
    // State Update: x_{c-1} → x_c
    x_scaled = A^L ⊙ x_{c-1}        // 逐元素复数乘法 (长度m)
    update = Mux ⊗ u_c              // 矩阵向量乘 (m×L)·(L) → m
    x_c = x_scaled + update
    
    // FFT Convolution on current chunk
    y_fft = IFFT( FFT(K) ⊙ FFT(u_c) )
    
    // Output Projection: x_{c-1} → y_c
    y_proj = Mxy ⊗ x_{c-1}          // 矩阵向量乘 (L×m)·(m) → L
    
    y_c = y_proj + y_fft + D·u_c    // 逐元素加法和乘法
```

术语一般如何实现？如何使用？
1. **Chunk size选择**：L选为2的幂次以使用Radix-2 FFT，同时需fit in on-chip SRAM。L过小导致chunk数C过大，串行状态更新链成为瓶颈；L过大导致单chunk FFT超出SRAM容量。
2. **矩阵存储优化**：Mux和Mxy均为Vandermonde矩阵，完整存储需要O(L·m)空间。利用Vandermonde性质，仅需存储首列/首行和scaling factor，SRAM需求减少数个数量级。
3. **与FlashAttention类比**：State passing在概念上类似FlashAttention的tiling——将全局操作分解为SRAM-friendly的chunk级操作，用额外计算换取DRAM访问减少。
4. 开源：H3官方repo未提供state passing实现，VGA论文自行实现该算法作为GPU baseline。

涉及论文标题：
- 45-VGA_Hardware_Accelerator_for_Scalable_Long_Sequence_Model_Inference.pdf

## H3 Model（Hungry Hungry Hippos）

术语解释
H3 (Hungry Hungry Hippos) 是由Fu等人提出的state-of-the-art SSM-based全局卷积语言模型，用SSMConv层替代Transformer block中的self-attention层，在保持接近Transformer语言建模质量的同时支持长序列高效处理。

术语是什么？
H3是首个在自然语言建模任务上实现与Transformer可比perplexity的SSM-based模型。其核心设计：(1) H3 block结构类似Transformer block，但self-attention被H3 layer替代；(2) H3 layer包含：短1D Conv平滑K→PointMult(K,V)→SSMConv(全局卷积+state passing)→PointMult(Q, result)→FC输出；(3) SSM参数基于S4D的对角化参数化（diagonal A矩阵），降低滤波器生成复杂度。H3-GPT配置：12个H3 blocks替代GPT-125M的12个self-attention blocks，h=768，SSM state size m=64。在WikiText103上perplexity为21.0（同等规模GPT-2约20.6）。H3-Speech是H3在语音分类SC10上的变体，使用6个单向H3 layers，h=128。

从算法pipeline角度拆解：
H3模型推理的ROI在H3 layer中，以单个hidden dimension列为例：
```
1. K经过短1D Conv（长度<128的小kernel FFT卷积）
2. V ⊙ Conv(K) → 门控V
3. SSMConv(u=门控V):
   a. 若序列长度N很大 → State Passing分块
      - For c in 0..C-1:
        * State Update (Mux·u_c + A^L·x_{c-1})
        * Output Projection (Mxy·x_{c-1})
        * FFTConv (FFT→CTF mult→IFFT on chunk u_c)
        * y_c = Proj + FFTConv + D·u_c
   b. 若N fit in SRAM → 直接FFT卷积
4. Q ⊙ SSMConv_output → 最终输出
```

术语一般如何实现？如何使用？
1. **与Transformer的互换性**：H3 layer可直接替换Transformer的self-attention layer，保留FFN、LayerNorm、residual connection和dropout。
2. **Hybrid H3模型**：部分H3 layer替换为self-attention layer的混合架构在短序列上可能性能更好，但长序列时self-attention成为瓶颈。
3. **开源实现**：https://github.com/danfu09/H3（含GPT-125M-like配置和WikiText103训练的checkpoint）。
4. **精度与效率tradeoff**：H3在LRA benchmark上显著优于self-attention，但在自然语言perplexity上略差（差距随模型规模缩小）。FFT convolution比parallel scan快3.3×但灵活性更低。

涉及论文标题：
- 45-VGA_Hardware_Accelerator_for_Scalable_Long_Sequence_Model_Inference.pdf
- 46-Fast On-device LLM Inference with NPUs.pdf

## Shadow Outlier Execution

术语解释
Shadow outlier execution 是 llm.npu 提出的 W8A8 per-tensor 量化推理中处理 activation outlier 的技术。利用加法分配律将 MatMul 分解为"NPU 端 per-tensor INT8 MatMul + CPU 端稀疏 outlier 浮点 MatMul"，在不损失 NPU 效率的前提下保持推理精度。通过 hot/cold channel 分离和层级 outlier 剪枝最小化额外内存和同步开销。

术语是什么？
LLM 的 activation 中存在少量 outlier channels（值显著大于相同 tensor 的其他 channels），使得直接 per-tensor INT8 量化（用单一 scale 将整个 tensor 映射到[-127,128]）产生显著误差。现有方法要么用 per-group 量化（如 K-Quant、AWQ，每组独立 scale，但 NPU 不友好），要么用 SmoothQuant（per-tensor 但需迁移 outlier 到 weight，精度损失 3.9–8.4%）。Shadow outlier execution 将 quantized MatMul 等效分解为：
```
MatMul(x, w) = MatMul(clip(x/s, -127, 128), w) [NPU INT8]
              + MatMul(outlier_channels, w)         [CPU FP16]
```
其中 outlier 定义为 `⌊x/s⌋ * 128`（超出 INT8 范围的小数部分）。由于 outlier 极稀疏（0.1–0.3% channels），CPU 侧 MatMul 计算量极小，可与 NPU 执行完全重叠。

从算法pipeline角度拆解：
```
# 1. 离线准备阶段
for each linear_layer in model:
    # Profiling: 使用大规模语料库（如 wikitext）统计各 channel 的 outlier 出现频率
    for batch in calibration_corpus:
        activations = forward_pass(layer, batch)
        for c in channels:
            outlier_val = activations[:, c] / scale_s
            if abs(outlier_val) > 128:
                outlier_count[c] += 1
    
    # 识别 hot channels（<3% channels 贡献 >80% outliers）
    hot_channels = top_k_percent(outlier_count, 3%)
    # 存储 hot channel 的 FP16 权重到 CPU 内存，cold 权重到磁盘
    hot_weights[layer] = W_fp16[hot_channels, :]

    # 层级剪枝：测量各 layer 的 outlier importance
    importance[layer] = max(outlier_val / scale_s)
    # 剪枝 85% 最不重要 layers 的 outlier 处理

# 2. 运行时推理
def shadow_outlier_matmul(x_fp16, W_int8, s, layer_id):
    # Part A: NPU per-tensor W8A8 MatMul
    x_int8 = clip(round(x_fp16 / s), -127, 128)  # [seq, hidden]
    y_npu_int32 = INT8_MatMul_NPU(x_int8, W_int8)  # NPU INT8 MatMul
    
    # Part B: CPU shadow execution（并行于 NPU 执行）
    if layer_id in important_layers:
        outlier_mask = abs(x_fp16 / s) > 128
        outlier_channels = find_nonzero_channels(outlier_mask)
        outlier_vals = x_fp16[outlier_channels]  # 仅提取 outlier 值（不含量化 clip）
        W_cpu = lookup_weights(outlier_channels, hot_weights, disk)
        y_cpu = FP16_MatMul_CPU(outlier_vals, W_cpu)
    else:
        y_cpu = 0  # pruned layer，跳过
    
    # 合并：反量化 NPU 结果 + 加上 shadow 补偿
    y = y_npu_int32 * s + y_cpu
    return y
```

术语一般如何实现？如何使用？
(1) 量化 scale s 的选择使用 max-min symmetric quantization：`s = max(|x|) / 127`，无需 calibration data（仅需知道 max 值）。(2) Hot channel 缓存：仅缓存 <3% 的最频繁 outlier channels 的 FP16 权重到 CPU 内存，降低 34.3% 内存开销；cold channels 的权重在需要时从磁盘按需加载（加载可与 NPU 执行重叠）。(3) 层级剪枝率 85%：仅对 top 15% 最重要 layers 执行 shadow execution，消除 CPU-NPU 同步开销（从 29.7% 端到端延迟降至可忽略），精度损失 <1% vs FP16。(4) 与现有方法对比：比 SmoothQuant（per-tensor，迁移 outlier 到 weight）精度提升最高 32.9%（因 shadow 在 CPU 上以 FP16 精确处理 outlier）；比 K-Quant（per-group）在 NPU 上性能提升 8.1–10.7×（因避免子 MatMul + float sum）。

涉及论文标题：
- 46-Fast On-device LLM Inference with NPUs.pdf

## Early Exiting in LLMs (大语言模型中的早退机制)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Early Exiting（早退）是动态神经网络中的一种推理优化技术，在 LLM 推理中指让某些 token 在遍历全部 transformer decoder layers 之前就终止计算并输出结果。核心思想：LLM inference 的本质是从完整词汇表中通过级联 decoder layers 生成最高概率 token 的在线搜索过程，但并非所有 token 都需要完整遍历所有层——简单 token（如标点、冠词）可能在浅层已有足够置信度。Early Exiting 在每层后部署 predictor（预测器），判断是否可以在当前层直接输出结果，从而减少平均前向层数。LLM 场景下的早退与 vision 领域不同：LLM 的搜索空间是完整词汇表（~3×10⁴ tokens），predictor 需要感知整个词汇空间来做出 exit 决策，使得 predictor 自身的开销（词汇表遍历 + 高维特征提取）成为关键瓶颈。现有方法如 AdaInfer 使用 SVM predictor，RAEE 使用检索增强数据库，但均面临 predictor 开销高（~20-30% 端到端延迟）的问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LLM Early Exiting 的算法流程：
```python
# 输入: prompt tokens P, LLM with L layers
# 输出: generated token sequence
for layer_idx in range(L):
    hidden_states = decoder_layer(hidden_states)  # Attention + FFN
    
    # Early Exiting predictor
    features = extract_features(hidden_states)  # 从 hidden states 提取特征
    should_exit = predictor(features)            # 预测是否早退
    
    if should_exit:
        token = lm_head(hidden_states)           # 映射到词汇空间
        return argmax(token)                     # 输出 token
# 最终层输出
return argmax(lm_head(hidden_states))
```
不同 LLM 的 exit layer 分布具有 skewness（偏态性）：统计表明约 50% 的层 exit 概率低于平均水平（如 3.2%），意味着大部分 predictor 的计算在多数情况下是无效的。不同 token 需要的 forward layers 也不同（如 Llama2-7B 中不同 token 可能在第 5 层到第 28 层不等 exit）。

术语一般如何实现？如何使用？
现有实现方式包括：(1) AdaInfer——每层部署 SVM predictor，需要遍历完整词汇表（hidden_states × lm_head）获取特征（top token prob、entropy、confidence），内存开销低但预测开销高（~30% 计算量）；(2) RAEE——构建早退信息数据库（>数 GB），通过 embedding 相似度检索匹配历史退出数据，内存开销高且检索延迟影响端到端性能；(3) SpecEE——利用 speculative model 生成推测 token 来将搜索空间从 32000 降至 4，MLP predictor 参数仅 6.7K，predictor overhead ~5.6% 总延迟；(4) MoD/D-LLM——训练路由网络或动态决策模块，需要预训练/微调 LLM 参数，训练成本高。关键设计选择：predictor 的搜索空间大小、特征维度、模型复杂度、以及是否需要修改原 LLM。

涉及论文标题：
- 47-SpecEE- Accelerating Large Language Model Inference with Speculative Early Exiting.pdf

## Speculative Early Exiting (基于推测的早退)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Speculative Early Exiting 是 SpecEE 提出的新范式，用 speculative model（推测模型，如 EAGLE DLM）将 LLM early exiting predictor 的在线搜索空间从完整词汇表（~3×10⁴ tokens）缩减到推测 token 范围（~3-4 tokens），实现 ~100× 搜索空间缩减。核心逻辑：投机解码中的 DLM（Draft Language Model）输出的 speculative tokens 为 TLM（Target Language Model）的 token 选择提供了有界范围——通过训练使 DLM 结果与 TLM 对齐，足够强的 DLM 可以确保 TLM 的最终输出几乎总是落在 speculative tokens 集合中。SpecEE 利用这一性质，让 early exiting predictor 不再需要遍历完整词汇表，而只关注 DLM 推测的少数几个 token，大幅降低 predictor 开销。该范式将 predictor 从每层 ~6.7M params/FLOPS（原始 AdaInfer-style traversal）降至 ~6.7K params/FLOPS。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Speculative Early Exiting pipeline（以 SpecEE Llama2-7B 为例）：
```python
# Step 1: Speculative model 生成推测 token，提取推测子词表
spec_tokens = speculative_model.generate(prompt, num_tokens=4)  # [T1,T2,T3,T4]
spec_lm_head = lm_head[:, spec_tokens]  # [4096, 4] 提取对应列

# Step 2: LLM 逐层前向，每层执行 predictor
for layer_idx in range(L):
    hidden = decoder_layer(hidden)  # [1, 4096]
    
    # Predictor: 仅在推测子空间内操作（~100× 缩减）
    spec_logits = hidden @ spec_lm_head           # [1, 4] vs original [1, 32000]
    local_probs = softmax(spec_logits)            # [1, 4]
    prob_variation = local_probs - prev_probs     # [1, 4]
    feat = concat([spec_logits, local_probs, prob_variation])  # [1, 12]
    pred = MLP(feat)                              # 12→512→1, Sigmoid
    
    if pred > 0.5:
        # Verification: 用完整 lm_head 验证
        global_logits = hidden @ lm_head  # [1, 32000]
        if argmax(global_logits) in spec_tokens:
            return argmax(global_logits)  # Early exit
    prev_probs = local_probs
```

术语一般如何实现？如何使用？
实现要点：(1) 选择 speculative model——SpecEE 选用 EAGLE DLM（仅需 24 hours RTX 3090 训练，~3% LLM 内存开销），也可用其他 DLM（如 Medusa heads）；(2) Predictor 训练——用目标 LLM 在 MT-Bench 等数据集上推理，收集各中间层的隐藏状态和 speculative token 信息，生成约 16K 训练数据 per predictor，label 为 True 若 early exit token = 最终 token；仅需 ~2% 训练数据即可收敛（~5 分钟/A100 total）；(3) Predictor 模型——2 层 MLP（12×512+512×1），ReLU + Sigmoid，利用 GPU tensor core 并行；(4) 不修改原始 LLM 参数，predictor 与 LLM 完全解耦。可集成到 HuggingFace、vLLM、llama.cpp 等 serving 框架。

涉及论文标题：
- 47-SpecEE- Accelerating Large Language Model Inference with Speculative Early Exiting.pdf

## Probability Shift in LLM Early Exiting (LLM早退中的概率偏移)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Probability Shift（概率偏移）是 SpecEE 论文发现的关键现象，用于指导 early exiting predictor 的特征选择。核心观察：在 LLM 逐层前向过程中，如果最终输出 token 确实在 speculative tokens 的缩减词汇空间内，则该 token 的 local probability 会在某一层急剧上升，而其他 speculative tokens 的概率保持低位稳定；反之若最终输出不在缩减空间内，所有 speculative tokens 的概率都保持低位稳定。这一现象将 "是否正确推测" 转化为 "概率曲线的形态变化"，从而可以用低维特征（speculative token logits + local probabilities + 概率变化量）替代原始的高维特征（完整词汇分布，~32000 维）。概率偏移使得 predictor 可以用极少的输入特征（12 维 = 4 tokens × 3 特征）实现有效的 exit/continue 分类。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
概率偏移的特征提取算法：
```python
# 输入: 当前层 hidden_states [1, hidden_dim]
#       speculative_lm_head [hidden_dim, 4]（4个推测token对应的lm_head列）
#       prev_local_probs [1, 4]（前一层local probabilities）
# 输出: 12-dim predictor input features

spec_logits = hidden_states @ speculative_lm_head   # [1, 4]
local_probs = softmax(spec_logits)                   # [1, 4]
prob_variation = local_probs - prev_local_probs      # [1, 4]

# 特征选择的重要性——仅用概率变化不够：
#   案例A: old_probs=[0.32, 0.68] → new_probs=[0.44, 0.56]  Δ=[0.12, -0.12]
#   案例B: old_probs=[0.46, 0.54] → new_probs=[0.58, 0.42]  Δ=[0.12, -0.12]
#   Δ相同但案例A不该exit（正确token概率从0.68降到0.56），案例B可以exit
# 因此需要同时输入 spec_logits 和 local_probs 作为补充特征

feat = concat([spec_logits, local_probs, prob_variation], dim=-1)  # [1, 12]
pred = MLP(feat)  # 二分类: exit or continue
```

术语一般如何实现？如何使用？
具体观察：(1) 当最终正确 token 在 speculative set 内时，该 token 的 local probability 从初始 ~25%（均匀分布）上升到 ~94%（later layers），其他 token 降至 ~6%；(2) 当最终正确 token 不在 speculative set 内时，所有 4 个 speculative token 的 probability 保持低位振荡（~20-30%）；(3) 概率偏移的现象在不同 dataset（MT-Bench/MMLU/GSM8K）和不同模型（Llama2-7B/13B）上均成立。使用方式：(a) 离线数据收集——在多数据集上运行 LLM + speculative model，记录每层 hidden states、speculative logits、local_probs、最终 token；(b) 特征工程——确认 3 类特征的充分性（如 Figure 6 的 counterexample analysis）；(c) MLP 设计空间探索——测试不同 hidden dim (64-1024) 和 layers (1-4)，选择 accuracy/time tradeoff 最优配置。

涉及论文标题：
- 47-SpecEE- Accelerating Large Language Model Inference with Speculative Early Exiting.pdf

## DLRM (Deep Learning Recommendation Model) — Sparse + Dense Dual-Network Architecture

术语解释
DLRM (Deep Learning Recommendation Model) 是Meta提出并广泛部署的推荐系统模型架构，将sparse features（类别型特征如Post ID、Ad ID）通过embedding table lookup处理，dense features（连续型特征如年龄）通过MLP处理，两者交互后由final MLP输出预测。DLRM是当前推荐系统infrastructure中占主导地位的model architecture。

术语是什么？
DLRM的canonical architecture由三个主要组件构成：(1) **Sparse Network**（Remote Network）：处理类别型特征（categorical features）。每个类别型特征（如user ID、item ID）通过embedding table lookup转换为dense vector（embedding dimension D=32-256）。多个embedding table的lookup结果通过pooling操作（如sum pooling/mean pooling）聚合为统一的sparse representation。TBE (Table Batched Embedding) operator专门实现batched embedding lookup+pooling。Embedding tables可占模型大小的90%（数十GB至TB级）。(2) **Dense Network**（Merge Network）：处理连续型特征（dense features），通常是多层MLP（FC layers + activation + LayerNorm），将dense features映射到与sparse representation相同的维度空间。(3) **Interaction + Final MLP**：sparse和dense representations通过interaction操作（如Hadamard product、dot product、或DHEN的deep interaction）融合，然后输入final MLP产生预测（如点击率CTR预测的sigmoid输出）。DLRM的compute pattern特征是：sparse部分memory-intensive（embedding lookup涉及大量irregular memory access），dense部分compute-intensive（FC layers GEMM dominated），两者对硬件需求差异大。

从算法pipeline角度拆解术语：
DLRM推理的完整pipeline（以Meta生产环境为例）：
```
# Input: user_features (sparse: user_id, country, etc.; dense: age, history_ctr, etc.)
#        + candidate_items (sparse: item_id, category, etc.; dense: price, etc.)

def DLRM_inference(user_features, candidate_items):
    # == Sparse Network ==
    user_embeddings = []
    for table, feature_id in zip(embedding_tables, user_features.sparse):
        # TBE: Table Batched Embedding lookup
        emb_vector = table.lookup(feature_id)  # [B, D] or [B, T, D] for sequence
        user_embeddings.append(emb_vector)
    user_sparse = pooling(user_embeddings)  # sum/mean pooling -> [B, D1]
    
    item_embeddings = []
    for table, feature_id in zip(embedding_tables, candidate_items.sparse):
        emb_vector = table.lookup(feature_id)
        item_embeddings.append(emb_vector)
    item_sparse = pooling(item_embeddings)  # -> [B, D2]
    
    # == Dense Network ==
    user_dense = dense_mlp(user_features.dense)  # MLP: [B, D_dense] -> [B, D1]
    item_dense = dense_mlp(candidate_items.dense)  # -> [B, D2]
    
    # == Interaction ==
    # Hadamard product interaction
    interaction = user_sparse * user_dense + item_sparse * item_dense  # [B, D1]
    
    # == Final MLP (late-stage ranking) ==
    for fc_layer in final_mlp_layers:
        interaction = fc_layer(interaction)  # FC weight: [D1, H1], [H1, H2], ...
        interaction = relu(interaction)
    prediction = sigmoid(final_fc(interaction))  # [B, 1] -> CTR prediction
    return prediction
```
MTIA 2i上的关键适配：(1) sparse network的TBE操作利用MTIA的DMA_IN index-based address calculation和SE的128-row accumulation；(2) dense network的FC layers利用DPE GEMM + autotuning选择optimal kernel variant；(3) SRAM缓存embedding table热rows→40-60% sparse access hit rate。

术语一般如何实现？如何使用？
DLRM的实现：(1) Facebook开源DLRM (https://github.com/facebookresearch/dlrm) ——包含reference PyTorch实现和benchmark (Criteo Kaggle, Criteo Terabyte)。(2) 生产级变体——DHEN（Deep and Hierarchical Ensemble Network, [30]) 增加hierarchical interaction层；Wukong ([29]) 扩展DHEN横跨两个数量级scaling；HSTU ([28]) 引入sequence-based Transformer-like推荐。(3) DLRM对硬件的影响——推动Meta自研MTIA芯片的主要workload，其sparse+dense双网络特征直接影响MTIA 2i的异构PE设计（DPE专为dense GEMM，SE支持TBE accumulation，大SRAM缓存embedding热数据）。

涉及论文标题：
- 48-Meta_s Second Generation AI Chip- Model-Chip Co-Design and Productionization Experiences.pdf

## Jagged Tensor in Recommendation Models

术语解释
Jagged Tensor（锯齿张量）是一种支持变长序列的稀疏张量表示，其中连续的rows可以有不同的长度。在推荐模型中，当引入sequence embeddings（如用户行为序列）时，不同用户的序列长度不同，产生jagged tensor结构。

术语是什么？
Jagged tensor是PyTorch中定义的数据结构（参见FBGEMM Jagged Tensor Operators [4]），其结构包含：(1) values——所有序列的elements连续存储的一维tensor；(2) offsets——各序列在values中的起始偏移量（长度为batch_size+1的整数tensor）。例如，3个用户的浏览序列长度分别为[3, 1, 2]，则values=[v₀⁰, v₀¹, v₀², v₁⁰, v₂⁰, v₂¹]，offsets=[0, 3, 4, 6]。Jagged tensor对推荐模型的意义：(a) 高效存储——避免padding浪费（若pad到max length=3，存储9 elements vs jagged的6 elements）；(b) 支持ragged attention（HSTU [28]需要处理不同用户的不同历史长度）；(c) 变长操作——支持jagged tensor上的数学运算（linear transform、Hadamard product、sparse↔dense convert）。Jagged tensor在MTIA 2i上的支持：由于SE的ISA有限，使用RISC-V vector core（64B vector extension）更灵活；SE的LUT被repurpose用于gather操作（piecewise loading of weight/timestamp tables）。

从算法pipeline角度拆解术语：
Jagged tensor在HSTU sequence-based推荐中的使用：
```
# 用户行为序列：user1有4个历史行为，user2有2个，user3有7个
# 传统方法：pad to max_len=7, shape=[3,7,D] -> 21 elements
# Jagged方法：values存储12个有效elements

# TBE lookup with jagged indices
jagged_indices = JaggedTensor(
    values=[123, 456, 789, 101, 202, 303, ...],  # flat indices
    offsets=[0, 4, 6, 13]  # 3 users: lengths 4, 2, 7
)

# Embedding lookup -> jagged embeddings
jagged_embeddings = embedding_table.lookup(jagged_indices)
# values shape: [sum(lengths), D] = [13, 128] for D=128

# Jagged linear transformation (per-sequence-element)
jagged_transformed = jagged_linear(jagged_embeddings, weight)  # [13, D] @ [D, H] -> [13, H]

# Ragged attention (HSTU)
attention_output = ragged_self_attention(
    query=jagged_transformed,
    key=jagged_transformed,
    value=jagged_transformed,
    offsets=jagged_indices.offsets,  # tells attention which elements belong to same sequence
    bias=positional_timestamps_bias   # calculated from positional weights + timestamps
)
# Output: jagged tensor with same offsets, attended values
```

术语一般如何实现？如何使用？
Jagged tensor的实现：(1) PyTorch FBGEMM库提供C++/CUDA jagged tensor operators (https://pytorch.org/FBGEMM/fbgemm_gpu-overview/jagged-tensor-ops/JaggedTensorOps.html)；支持操作包括：jagged_to_padded_dense, dense_to_jagged, jagged_elementwise_add/mul, jagged_linear；(2) GPU实现利用CUDA block-stride loop——每个thread block处理一个jagged dimension的元素；(3) MTIA 2i实现利用RISC-V vector core——由于jagged tensor的数据级并行度较低（相比dense tensor），64B vector extension的较短vector length反而提供更高的灵活性。使用场景：推荐模型的sequence embeddings、LLM中变长prompt batching的attention mask。

涉及论文标题：
- 48-Meta_s Second Generation AI Chip- Model-Chip Co-Design and Productionization Experiences.pdf

## Test-Time Compute (TTC) for LLM Reasoning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Test-Time Compute (TTC, 测试时计算) 是一种LLM推理范式，在推理阶段通过投入额外计算资源来增强模型推理能力，而非仅依赖扩大模型参数量。TTC将复杂推理任务分解为多步骤的迭代过程：每个step中，policy model（解码模型）生成多个候选"分支"（candidate solutions），process reward model (PRM, 过程奖励模型) 对这些候选进行评分验证（verification），选择最有前景的分支进入下一个step。Tree search结构由"宽度"（每step候选分支数，通常2~8）和"深度"（推理步骤数，任务依赖）定义。TTC使小型模型（如1B参数）在推理benchmark上可匹敌或超越大型模型（如405B参数）[18, 28]。典型TTC pipeline包括：text-based pipeline (如Compute-Optimal Test-Time Scaling [18]) 和 vision-based pipeline (如LLaVA-o1 [36])。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TTC-based LLM Reasoning pipeline算法伪代码（以ORCHES使用的pipeline [18] 为例）：
```
Input: question Q, search width W, max steps S
Output: answer A

prompt = Q
for step s = 1 to S:
    // Generation Phase (Policy Model, e.g., Llama3.2-1B)
    candidates = []
    for branch b = 1 to W:
        candidate_b = PolicyModel.decode(prompt)  // autoregressive decoding
        candidates.append(candidate_b)
    
    // Verification Phase (PRM, e.g., Llama3.1-8B)
    scores = []
    for candidate in candidates:
        score = PRM.prefill(prompt + candidate)   // prefilling all tokens
        scores.append(score)
    
    // Selection: keep top-k candidates (beam search / best-of-N)
    selected = TopK(candidates, scores, k=K)     // K ≤ W
    
    // Update prompt for next step
    prompt = prompt + Concatenate(selected)
    
return FinalAnswer(prompt)
```
Pipeline中两个模型角色不同：Policy model执行decoding（token-by-token generation，memory-bound, arithmetic intensity低），PRM执行prefilling（一次性处理全部候选tokens，compute-bound, arithmetic intensity高）。Shared KV cache（来自共享前缀prompt）随step增多逐渐accumulate→workload特征从memory-bound向compute-bound动态转变。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TTC的实现依赖：(1) 两个独立模型（policy model和PRM）或同一模型的不同使用模式（如LLaVA-o1用同一Vision-Language model同时做generation和verification）；(2) Tree search策略（beam search: beam width B从W个候选中选top-B；best-of-N: 每step独立生成N个候选取最优）；(3) PRM scoring机制（可为learned reward model输出标量分数，也可以rule-based如math verifier）。实际部署中policy model通常较小（1B-3B），PRM较大（7B-8B）以保证verification accuracy。常用PRM训练方式：在process supervision数据上fine-tune（标注中间步骤正确性），或从outcome-level reward蒸馏。TTC的主要开销来自多分支的KV cache存储（W倍memory footprint）和step间串行依赖（generation必须等verification完成）。ORCHES通过GPU-PIM协同、branch prediction pipelining和memory structuring系统性地解决这些efficiency瓶颈。

涉及论文标题：
- 55-ORCHES- Orchestrated Test-Time-Compute-based LLM Reasoning on Collaborative GPU-PIM HEterogeneous System..pdf

## Process Reward Model (PRM) in TTC Reasoning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Process Reward Model (PRM, 过程奖励模型) 是TTC-based LLM reasoning中负责候选分支验证（verification）的模型。不同于outcome reward model仅评估最终答案的正确性，PRM对推理过程的每个中间步骤进行评分，判断每个候选分支在推理路径上的合理性。在TTC pipeline中，PRM以prefilling模式执行：将policy model生成的所有候选tokens（prompt + candidate）作为input一次性编码，输出每个候选的reward score。PRM通常比policy model更大（如policy=1B, PRM=8B），因为verification quality直接影响最终推理准确率。PRM-version用来在ORCHES中做speculative branch prediction：使用large PRM的前若干层（如前10层）作为small PRM进行快速预判，与large PRM的full verification并行执行。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
PRM在TTC pipeline中的verification计算流程：
```
Input: prompt P (shared prefix from previous steps), 
       candidates C = [c_1, c_2, ..., c_W] (W branches generated by policy model)
Output: selected candidates C'

for each candidate c_i in C:
    // Concatenate: [shared_prompt | unique_candidate_tokens]
    input_seq_i = P + c_i                     // total length L_i tokens
    
    // Prefilling (parallel encoding of all L_i tokens)
    K_i, V_i = PRM.encode(input_seq_i)       // compute KV for all tokens at once
    
    // Final hidden state or special [SCORE] token
    h_i = PRM.final_layer(K_i, V_i)
    score_i = PRM.reward_head(h_i)           // scalar reward score

// Selection
selected_indices = argmax(score_i, k=K)      // K ≤ W for beam search
C' = [C[i] for i in selected_indices]
```
PRM prefilling是高arithmetic intensity的compute-bound操作（batch_size = L_i通常100+ tokens），与Policy model的low arithmetic intensity decoding形成对称。这种asymmetry是ORCHES Adaptive Assignment (T1) 将PRM固定分配给GPU、Policy model根据batch size动态分配GPU/PIM的理论基础。PRM验证的latency可能等于或超过generation（当PRM model size远大于Policy model或candidates很多时），造成generation→verification之间的pipeline stall（C2）。ORCHES的branch prediction (T2) 正是利用small PRM预判来提前启动generation，消除此stall。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PRM实现方式：(1) **Base model selection**——使用通用LLM（Llama3.1-8B、Qwen2.5-7B等）在process supervision数据上fine-tune为PRM；(2) **PRM-Tuned变体**——在标注了中间推理步骤正确性的数据上训练（如Qwen2.5-7B-PRM-Tuned），使模型学习评估partial reasoning path的质量；(3) **Scoring方式**——最终hidden state经reward head（小型线性层）输出scalar score；也可用整个序列的平均log-probability作为confidence score。PRM的compute cost与candidate count × average candidate length成正比。优化方向包括：PRM量化以减少memory footprint和prefilling latency；使用small PRM做快速pre-filter再large PRM精细评分（multi-stage verification）；ORCHES的history alignment用large PRM的历史步scores提升small PRM预测准确率（从~52%→~78%）。

涉及论文标题：
- 55-ORCHES- Orchestrated Test-Time-Compute-based LLM Reasoning on Collaborative GPU-PIM HEterogeneous System..pdf

## Dense/Sparse KV Cache Separation with Prefix Tree for RAG Generation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dense/Sparse KV Cache Separation 是HeterRAG在locality-aware generation优化中提出的KV cache管理方法，结合prefix tree组织文档序列，将每个文档的KV cache分为两部分：(1) **Dense KV**——文档自身的完整KV，独立于文档在序列中的位置和上下文，计算于文档本身（self-context）。当同一文档出现在不同序列中时，dense KV可跨序列复用。(2) **Sparse KV**——文档出现在特定序列中时，仅保留某些重要tokens（10-20%）的KV。利用attention sparsity现象（少数tokens贡献大部分attention），保留这些tokens的KV几乎不降低生成质量（约0.2% attention deviation [CacheBlend, EuroSys'25]），但大幅减少多副本存储开销。Dense KV全量缓存，sparse KV按LRU evict。该方案解决了传统prefix tree KV cache方案的两大缺陷：长序列prefix overlap少导致cache效率低、文档在新序列中重复出现时仍需recompute KV。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Dense/Sparse KV Cache分离在RAG generation中的算法伪代码：
```
// 数据结构：prefix tree node
struct TreeNode:
    token: str                    // 文档token序列
    dense_kv_addr: addr           // 文档自身的dense KV在HBM中的地址
    sparse_kv_entries: list[addr] // 该文档在不同序列中的sparse KV地址列表
    children: dict[str, TreeNode] // prefix tree子节点

// Prefilling with Dense/Sparse KV Cache
def prefill_with_cache(sequence: list[Document]):
    kv_cache_for_prefill = []
    recompute_list = []

    node = tree_root
    for doc in sequence:
        if doc.id in node.children:
            // 匹配：文档dense KV + 当前序列的sparse KV
            treenode = node.children[doc.id]
            dense_kv = hbm_read(treenode.dense_kv_addr)
            sparse_kv = find_sparse_kv(treenode, current_sequence_id)
            if sparse_kv:
                // Selective combine: 用sparse KV中的important token KV替换dense KV对应位置
                combined_kv = kv_substitution(dense_kv, sparse_kv)
                kv_cache_for_prefill.append(combined_kv)
            else:
                kv_cache_for_prefill.append(dense_kv)
            node = treenode
        else:
            // 未匹配
            if doc.id in dense_kv_store:
                // dense KV已缓存但未在prefix tree此位置
                dense_kv = dense_kv_store[doc.id]
                // Token Filtering: 选择important tokens (10-20%) selective recompute
                important_idx = select_important_tokens(dense_kv, top_k=10-20%)
                recompute_list.append((doc, important_idx))
            else:
                // 完全未缓存：完整compute
                recompute_list.append((doc, full_compute=True))
            // 创建新tree node
            node = create_new_path(node, doc)

    // 执行计算（仅recompute未缓存或selective part）
    full_kv = prefill_compute(recompute_list)
    // 更新prefix tree
    update_prefix_tree(full_kv)  // 写入dense KV（如有新文档）或sparse KV（新序列context）

    // KV cache满了按LRU evict sparse KV entries
    if cache_full():
        evicted_doc = lru_evict_sparse()
        if evicted_doc.sparse_entries.empty():
            reclaim_dense_kv(evicted_doc)

    return concat(kv_cache_for_prefill, full_kv)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Dense/Sparse KV分离的实现依赖HeterRAG AccelHBM TPM中的三个专用硬件单元：(1) Tree Search Unit——在prefix tree中匹配新文档序列，输出matched path和unmatched documents；(2) KV Substitution Unit——将matched document的dense KV和当前序列的sparse KV合并（hardware支持position-based select和result update逻辑）；(3) Token Filtering Unit——对unmatched但dense KV已缓存的文档，选择10-20% important tokens（基于deviation computation和sort logic），仅对这些tokens做selective recompute。该技术与RAGCache [arXiv:2404.12457]的knowledge tree结构、CacheBlend [EuroSys'25]的cached knowledge fusion理念相通，但HeterRAG通过专用硬件（TSU/KVSU/TFU）实现硬件加速，并将dense/sparse分离引入了前缀树的KV共享中。

涉及论文标题：
- 49-HeterRAG- Heterogeneous Processing-in-Memory Acceleration for Retrieval-augmented Generation.pdf

## Bi-directional Bit-level Sparsity (BBS)

术语解释
Bi-directional Bit-level Sparsity (BBS) 是一种新型位级稀疏范式，通过对称地利用位级稀疏——当bit-vector中zero-bits > 50%时skip zero-bits，当zero-bits < 50%时invert bit-vector skip one-bits——确保任意bit-vector始终展现≥50%的bit sparsity。BBS同时支持prune全零或全一的bit column（bi-directional sparse columns），提供比仅prune zero bit column的baseline方法更大的模型压缩空间。

术语是什么？
BBS基于2's complement二进制表示，利用bit-serial dot product的数学等价性：
- Eq. 2: Σ_{i: W_ib=1} A_i（skip zero-bits，当zero-bits > 50%时用）
- Eq. 3: ΣA_j - Σ_{i: W_ib=0} A_i（skip one-bits，当zero-bits ≤ 50%时用，即invert bit-vector）
第3式意味着：当bit column中zero-bits少于50%（即one-bits多于50%），可以将bit-vector invert，原one-bits变为sparse，计算转为从总激活和ΣA中减去跳过bits对应的激活。

BBS的核心优势：(1) 保证任意长度bit-vector至少50% bit sparsity，解决bit-serial计算中因zero-bit随机分布导致的load imbalance问题；(2) 兼容2's complement算术——无需像sign-magnitude格式那样引入2's complementer的开销；(3) BBS不依赖于operand precision，适用于任意bit-width的DNN；(4) BBS + binary pruning可实现>50% model compression而无需retraining。

从算法pipeline角度拆解：
BBS在DNN推理中的bit-serial dot product计算流程（8-bit weight, group size N=16, activation group {A0..A15}）：

```
Input: 16 weights W[0..15] (INT8), 16 activations A[0..15] (INT8)
Output: dot product result

for each bit significance b in 0..7:
    // Step 1: Extract bit column
    bit_col = [W[0][b], W[1][b], ..., W[15][b]]  // 16-bit vector
    
    // Step 2: Count sparsity and decide direction
    num_ones = popcount(bit_col)
    num_zeros = 16 - num_ones
    
    if num_zeros > num_ones:  // zero-bit sparsity > 50%
        // Eq. 2: sum activations where weight bit = 1
        partial_sum = 0
        for i in 0..15 where bit_col[i] == 1:
            partial_sum += A[i]
        sparse_flag = 0  // zero-skipping mode
    else:  // one-bit sparsity > 50%
        // Eq. 3: ΣA - sum activations where weight bit = 0
        sum_all = A[0] + A[1] + ... + A[15]  // pre-computed constant
        partial_sum = sum_all
        for i in 0..15 where bit_col[i] == 0:
            partial_sum -= A[i]
        sparse_flag = 1  // one-skipping (inverted) mode
    
    // Step 3: Shift and accumulate
    result += (partial_sum << b)  // multiply by 2^b
```

关键属性：无论bit_col中zero/one比例如何，上述循环每次最多处理ceil(16/2)=8个effective bits。

术语一般如何实现？如何使用？
BBS的实现：(1) **软件侧**（binary pruning算法）：PyTorch实现，对quantized INT8 weight做group-level binary pruning（rounded averaging或zero-point shifting），生成bi-directional sparse columns和BBS compression metadata。开源：https://github.com/yc2367/BBS-MICRO.git。(2) **硬件侧**（BitVert accelerator）：PE内scheduler通过priority encoder统计bit column中zero-bits数量→决定skip方向→生成sel/val信号控制bit-serial multiplier selectively process effective bits。BBS Multiplier根据BBS constant metadata对sparse columns做常数乘累加。

BBS可应用于任何quantized DNN（CNN/Transformer/LLM），post-training执行，无需calibration dataset或retraining。在7个代表性模型（VGG-16/ResNet/ViT/BERT/Llama-3-8B）上验证，平均准确率损失<0.5%。

涉及论文标题：
- 4-BBS_Bi-Directional_Bit-Level_Sparsity_for_Deep_Learning_Acceleration.pdf

## Bit-level Binary Pruning

术语解释
Bit-level Binary Pruning 是BBS框架中的模型压缩算法，通过生成structured bi-directional sparse bit columns来减少DNN权重的有效位宽，同时最大化地保留原始8-bit权重分布的统计特性。包含两种策略：Rounded Column Averaging（适合prune少量bit columns）和Zero-point Shifting（适合aggressive prune多个bit columns），均无需retraining或calibration dataset。

术语是什么？
Bit-level Binary Pruning以weight group（N个weight）为单位，对每个group生成指定数量（target N_sparse）的bi-directional sparse columns：
- **Bi-directional sparse column**：bit column中所有bits全为0或全为1。全0 column的bit-serial dot product = 0（可跳过），全1 column的bit-serial dot product = ΣA（组内所有activation之和，仅需常数乘）。
- **Rounded Column Averaging**：Step 1 识别并移除redundant columns（最高significant bit之后重复的bit columns，移除后不影响原始值）+ Step 2 对lower significant bits计算rounded average→用常数替换所有weight的低位→生成zero sparse columns。
- **Zero-point Shifting**（Algo. 1）：遍历所有可能的BBS constant（6-bit precision: [-32, 31]），将constant加到weight group后clip→识别redundant columns→zero out lower bits→计算MSE→保留MSE最小的配置。核心思想：通过shift weight的zero-point来促进更多zero column的生成。

压缩后每个group存储：8-bit metadata（2-bit #RedunCol + 6-bit BBS constant）+ 剩余非sparse bit columns。BBS constant每一位指示对应bit column是全0还是全1。

从算法pipeline角度拆解：
Bit-level Binary Pruning的完整算法流程（以moderate pruning, N=32, target=4 sparse columns为例）：

```
# Algo. 1: Zero-Point Shifting（对单个weight group）
Input:  W ∈ Z^{32} (INT8 weights in a group)
        p = 6 (BBS constant precision)
        N = 4 (target # sparse columns)
Output: W_C (compressed weights), D = {numRedunCol, constant}

bestMSE = ∞
# 遍历所有可能的zero-point shift
for constant in range(-32, 31):
    W_tmp = clip(W + constant, -128, 127)  # shift + 防溢出
    
    # Step A: 识别redundant columns（MSB后重复的bit columns）
    numRedunCol = count_redundant_columns(W_tmp)
    W_tmp = remove_lower_bits(W_tmp, numRedunCol)
    
    # Step B: 生成zero sparse columns
    numSparseCol = N - numRedunCol  # 还需生成的sparse columns数
    W_tmp = zero_out_lower_bits(W_tmp, numSparseCol)  # 最小化MSE的zeroing
    
    # Step C: 评估MSE
    newMSE = mean((W_tmp - W)^2)
    if newMSE < bestMSE:
        bestMSE = newMSE
        W_C = W_tmp
        D = {numRedunCol, constant}

return W_C, D

# Algo. 2: Global Binary Pruning（per-channel, hardware-aware）
Input:  Model M, per-channel scaling factors S, β=0.2, CH=32
Output: Pruned model M_P

# 1. Global敏感channel识别
all_channels = flatten([L.channels for L in M.layers])
sorted_channels = sort_by(all_channels, key=S, descending=True)
num_sensitive = ceil(β * len(sorted_channels))
sens_channels = sorted_channels[:num_sensitive]  # top β% by scale

# 2. 逐层对齐+prune
for L in M.layers:
    # 选本层敏感channel（对齐到CH倍数）
    layer_sens = intersect(L.channels, sens_channels)
    num_layer_sens = ceil(len(layer_sens) / CH) * CH
    # 不足CH的补齐（选本层scale最大的channel）
    top_L = sort_by(L.channels, key=S[L], descending=True)[:num_layer_sens]
    
    # 对normal channels执行binary pruning
    normal_channels = L.channels - top_L
    if aggressive:
        M_P[normal_channels] = zero_point_shifting(normal_channels)
    else:
        M_P[normal_channels] = rounded_averaging(normal_channels)
    
    # Sensitive channels keep 8-bit
    M_P[top_L] = L.weights[top_L]  # unchanged

return M_P
```

术语一般如何实现？如何使用？
Binary pruning实现为PyTorch代码（开源https://github.com/yc2367/BBS-MICRO.git），在NVIDIA RTX 3090 GPU上执行。压缩整个ResNet-50约需15秒（全层并行vectorized搜索optimal constant）。使用方式：
1. 对per-channel INT8 quantized model应用global binary pruning（选择β=10%/20%，CH=32）
2. Conservative pruning (2 columns, rounded averaging): 1.29×压缩，0.25%准确率损失
3. Moderate pruning (4 columns, zero-point shifting): 1.66×压缩，0.45%准确率损失
4. 输出compressed weight + BBS metadata，供BitVert或兼容的bit-serial加速器推理使用

Binary pruning与PTQ的关键区别：不需要calibration dataset（data-free），保留所有8-bit量化级别（通过允许bit column同时包含0和1），KL divergence远低于PTQ和BitWave bit-flip。

涉及论文标题：
- 4-BBS_Bi-Directional_Bit-Level_Sparsity_for_Deep_Learning_Acceleration.pdf

## DDPM (Denoising Diffusion Probabilistic Model，去噪扩散概率模型)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DDPM (Denoising Diffusion Probabilistic Model) 是Ho et al. (2020) 提出的一类生成模型，通过两个马尔可夫过程实现图像生成：(1) **前向扩散过程 (Forward Process)**——逐步向真实图像添加高斯噪声，经T个timestep后变为纯噪声：q(x_t | x_{t-1}) = N(x_t; √(1-β_t)·x_{t-1}, β_t·I)，其中β_t为预定义的噪声调度（noise schedule）。任意timestep t的采样可直接从x_0计算：x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε，其中ᾱ_t = Π_{s=1}^t (1-β_s)，ε ~ N(0,I)。(2) **反向去噪过程 (Reverse Process)**——从纯噪声x_T ~ N(0,I)开始，通过训练好的去噪网络ε_θ逐步去除噪声，恢复为清晰图像：p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t²·I)。训练目标为简化的噪声预测损失：L_simple = E_{t, x_0, ε}[||ε - ε_θ(x_t, t)||²]。推理时需执行T步（通常T=1000）连续去噪，每一步调用完整的U-Net ε_θ，计算成本极高。
S-DMA论文涉及的DDPM具体配置：U-Net架构含downsample blocks（卷积+SiLU activation+GroupNorm）、middle block（含self-attention）、upsample blocks（含skip connection）。输入分辨率CIFAR-10 32×32或CelebA 64×64。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DDPM推理（图像生成）pipeline伪代码：
```
// 输入：训练好的去噪网络 ε_θ，总timesteps T，噪声调度 {β_t}
x_T ~ N(0, I)  // 从纯噪声开始
for t in T, T-1, ..., 1:  // 反向去噪，共T步
    // 步骤1：噪声预测
    ε_pred = ε_θ(x_t, t)  // U-Net前向推理，计算量最大的部分
    
    // 步骤2：DDPM采样（从x_t求x_{t-1})
    α_t = 1 - β_t
    ᾱ_t = Π_{s=1}^t α_s
    z = N(0, I) if t > 1 else 0  // 最后一步不加随机噪声
    x_{t-1} = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_pred) + σ_t * z
// 输出：x_0 为生成图像
```

U-Net ε_θ内部计算（每个timestep内）：
1. **输入处理**：x_t（带噪图像）与timestep embedding（t经sinusoidal encoding+MLP）拼接。
2. **Encoder（Downsample path）**：ResBlock（Conv→GroupNorm→SiLU→Conv→GroupNorm→SiLU → +skip connection）→Downsample（Conv stride-2或AvgPool）。通道数逐步增加（如64→128→256→512），空间分辨率逐步减半。
3. **Bottleneck**：ResBlock + Self-Attention（QKV projection→attention score (SoftMax(QK^T/√d)) → ×V → output projection） + ResBlock。
4. **Decoder（Upsample path）**：Upsample（nearest-neighbor或transposed conv）→Concat skip connection from encoder→ResBlock。通道数逐步减少，空间分辨率逐步恢复。
5. **Output head**：Conv→输出与x_t同shape的噪声预测ε_pred。

计算特征：U-Net每timestep包含大量Conv（~76%计算量）、少量Attention（~1%）、激活和归一化操作。1000步串行去噪导致总FLOPs为单步U-Net的1000倍，端到端延迟极大。

术语一般如何实现？如何使用？
DDPM的实现通常基于PyTorch等深度学习框架。核心组件：(1) U-Net ε_θ使用标准Conv2D、GroupNorm、SiLU/ReLU激活、Multi-Head Self-Attention；timestep embedding通过sinusoidal position encoding + 2层MLP生成。(2) 噪声调度{β_t}：原论文使用linear schedule from β_1=1e-4 to β_T=0.02，后续工作提出cosine schedule等改进。(3) 使用方式：训练——对真实图像集随机采样x_0和t→生成x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε→训练ε_θ预测ε→minimize ||ε-ε_θ||²。推理——从x_T开始逐步去噪，每步运行完整U-Net。加速方向：(a) 减少timesteps T（DDIM等sampler，用更少步数达成相似质量）；(b) 单步U-Net加速（量化、剪枝、稀疏、蒸馏）；(c) 硬件加速（如S-DMA的专用sparsity accelerator）。

涉及论文标题：
- 50-S-DMA- Sparse Diffusion Models Acceleration via Spatiality-Aware Prediction and Dimension-Adaptive Dataflow..pdf

## Post-Transformer LLM / SU-LLM (State Update-based LLM)

术语解释
Post-Transformer LLM 是一类替代传统 Transformer 架构的大语言模型，包括状态空间模型（SSM）、线性注意力（Linear Attention）和循环神经网络（RNN）。Pimba 将它们统称为 SU-LLM（State Update-based LLM），因为这些模型的生成阶段共享一个统一的计算原语——state update。

术语是什么？
Post-Transformer LLM 包含三大类：(1) **SSM（如 Mamba-2）**：通过 structured state space duality 用 select state update 传播跨 token 的上下文信息，每 head 维护 dim_head×dim_state 的状态矩阵；(2) **Linear Attention（如 RetNet、GLA）**：用线性函数替代 softmax，K^T·V 乘积作为 state 持续更新，等同于去掉 decay 或使用 input-dependent gate 的 SSM state update；(3) **RNN（如 HGRN2）**：将传统 1D state 扩展为 2D outer product-based state，forget gate 类似 decay 机制。这些模型的核心优势是：相比 Transformer O(n²) 的 compute 和 O(n) 的 memory（KV cache 随 seq len 增长），Post-Transformer 的资源需求与序列长度无关——constant compute 和 constant memory。但生成阶段 state update 操作成为新的 memory bandwidth 瓶颈（batch 下 arithmetic intensity 极低）。混合模型（如 Zamba2）在 SSM 层间插入 attention 层以恢复 in-context learning 能力。

从算法pipeline角度拆解：
SU-LLM 单 token 生成（以 Mamba-2 为例）的计算流程：
```
输入: 当前 token embedding x_t, 上一 token state S_{t-1}
for each head h in 1..H:
    a_h, B_h, C_h, X_h = project_h(x_t)
    A_h, B_h = discretize(a_h, B_h, Δ_t)       // 连续→离散
    S_t = A_h ⊙ S_{t-1} + B_h ⊗ X_h            // state update
    y_h = S_t^T · C_h                            // GEMV
output = concat(y_1..y_H) · W_out + FFN(x_t)
```
RetNet/GLA 的 state update 与 Mamba-2 等价：d_t 替代 A_h（scalar/vector decay），k_t v_t^T 替代 outer product。HGRN2 使用 forget gate vector + outer product update。这些统一为 Equation 2：S_t = d_t ⊙ S_{t-1} + k_t v_t^T, y_t = S_t^T q_t。

术语一般如何实现？如何使用？
- SSM 代表 Mamba-2 [Dao & Gu, ICML 2024] 是当前最广泛采用的 Post-Transformer，已被集成到 Zamba2、Codestral Mamba、Nemotron-H 等模型中。
- Linear Attention 系列：RetNet [Sun et al., 2023]、GLA [Yang et al., ICML 2024]（gated linear attention）、GSA [Zhang et al., NeurIPS 2024]。
- RNN 系列：RWKV [Peng et al., EMNLP 2023]、HGRN2 [Qin et al., COLM 2024]、xLSTM [Beck et al., NeurIPS 2024]。
- 混合架构：Zamba2 [Zyphra]、Nemotron-H [NVIDIA]、Samba [Ren et al., ICLR 2025]。
- 训练框架通常使用 CUDA/PyTorch，Mamba-2 使用 selective scan 算法在 GPU 上高效实现。

涉及论文标题：
- 52-Pimba- A Processing-in-Memory Acceleration for Post-Transformer Large Language Model Serving..pdf

## State Update Operation (Unified Formulation for SU-LLMs)

术语解释
State Update 是 Post-Transformer LLM（SU-LLM）生成阶段的核心操作，统一了 SSM、Linear Attention 和 RNN 中用于传播和演化跨 token 上下文信息的状态更新过程。广义公式为 S_t = d_t ⊙ S_{t-1} + k_t v_t^T, y_t = S_t^T q_t。

术语是什么？
State Update 操作包含三个子步骤：(1) **State Decay**：d_t vector broadcast 到 S_{t-1} 每个元素做 element-wise 乘法，衰减旧信息的影响；(2) **Outer Product Update**：k_t v_t^T 产生 dim_head × dim_state 的矩阵，叠加到 decayed state；(3) **Output GEMV**：S_t^T q_t 产生 dim_head 维 output。该公式统一了 Mamba-2 的 selective state update（A_h 为 decay，B_h X_h^T 为 outer product）、RetNet 的 retention（标量 decay + K V outer product）、GLA 的 gated linear attention（vector decay/gate + K V outer product）和 HGRN2 的 forget gate + outer product update。此操作的 arithmetic intensity 约 0.25 FLOPS/byte（fp16），在 batch inference 下 strongly memory-bound。每 request 独立维护 state 无参数复用，state update latency 随 batch size 线性增长（RetNet BS=128 时占比 74%）。

从算法pipeline角度拆解：
单 head 的 state update 伪代码（MX8 量化版本）：
```
// S_{t-1}: [dim_head, dim_state] in MX8 format stored in DRAM
// d_t[dim_head], k_t[dim_head], v_t[dim_state], q_t[dim_head] in MX8

// Step 1: State Decay (element-wise multiply)
for j in 0..dim_state:
    for i in 0..dim_head:
        S_decay[i][j] = MX_mul(d_t[i], S_{t-1}[i][j])

// Step 2: Outer Product (MX multiply each pair)
for i in 0..dim_head:
    for j in 0..dim_state:
        outer[i][j] = MX_mul(k_t[i], v_t[j])

// Step 3: Update (MX element-wise add)
for i in 0..dim_head:
    for j in 0..dim_state:
        S_t[i][j] = MX_add(S_decay[i][j], outer[i][j])

// Step 4: Output GEMV (dot product)
for i in 0..dim_head:
    y_t[i] = Σ_{j=0}^{dim_state} S_t[i][j] * q_t[j]
```

术语一般如何实现？如何使用？
- GPU 实现：PyTorch/CUDA 中 element-wise multiply/add 和 GEMV 分别调用 cublas GEMV + custom element-wise kernel。Batch 下 memory-bound 严重。
- PIM 加速：Pimba 将 state update 完整 offload 到 HBM PIM 的 SPU 中执行四阶段流水线，14.6× latency 降低 vs GPU。
- 量化：MX8+stochastic rounding 存储 state 使带宽减半，且维持准确率。
- 所有 SU-LLM（RetNet/GLA/HGRN2/Mamba-2）的 state update 本质同构，使统一加速成为可能。

涉及论文标题：
- 52-Pimba- A Processing-in-Memory Acceleration for Post-Transformer Large Language Model Serving..pdf

## MX8 with Stochastic Rounding for LLM State Quantization

术语解释
MX8（Microscaling 8-bit 变体）+ 随机舍入 是 Pimba 识别出的 Pareto-optimal 低精度量化方案，用于在 SU-LLM 的 PIM 加速场景中对 state 矩阵进行量化，同时达到高准确率和低面积开销。

术语是什么？
MX（Microscaling）[Darvis Rouhani et al., ISCA 2023] 是 Microsoft 提出的 block floating point 格式，在组级别共享指数以降低指数存储开销。MX8 组织：每 16 个值组成一个 group，共享一个 8-bit shared exponent（E_g）；group 内每 2 个值为一个 sub-group，共享 1-bit microexponent（μexp）；每个值存储 7-bit sign + mantissa。随机舍入：量化时以概率正比于值到最近两个可表示值距离进行舍入（而非确定性最近邻），通过 LFSR 生成随机数加到 mantissa 低位实现。与 fp8 的关键差异：fp8（e4m3/e5m2）尾数仅 2-3 位（+隐式 1），连续 state update 中 swamping 效应严重——小量级值在累加中被丢失。MX8 以 6-7 位有效尾数抵抗 swamping，perplexity 与 fp16 持平（Mamba-2: 11.51 vs 11.46）。与 int8 相比，int8 需 per-32-element scaling factor，element-wise 加法需 dequantize→加→requantize，引入额外乘法器和比较逻辑——MX8 的共享 exponent 使加法仅需 shift 对齐（无乘法），面积开销显著更小。

从算法pipeline角度拆解：
MX8 State 量化与算术流程：
```
// State 量化（GPU→PIM 传输时）:
for each group of 16 values in state S:
    E_g = max_i(floor(log2(|x_i|)))
    for each pair (x_2j, x_{2j+1}) in group:
        μexp = max(e_2j, e_{2j+1}) - E_g   // 1-bit
        mant_2j = stochastic_round(x_2j / 2^{E_g + μexp}, 6)

// MX 乘法（SPE 内部）:
E_g_result = E_g_A + E_g_B
μexp_result = μexp_A + μexp_B  // 若=2则设为1并右移mantissa
mant_result[i] = int_mul(mant_A[i], mant_B[i]) >> shift

// MX 加法（SPE 内部）:
max_exp = max(E_g_A, E_g_B)
aligned_mant_B[i] = mant_B[i] >> (max_exp - E_g_B - μexp_B)
mant_result[i] = int_add(mant_A[i], aligned_mant_B[i])
μexp_result = 0
```

术语一般如何实现？如何使用？
- 开源：Microsoft MX 格式规范及软件模拟库；Pimba 实现 MX Multiplier 和 MX Adder 的 RTL（Synopsys Design Compiler + FreePDK 45nm→DeepScaleTool 10nm）。
- MX Multiplier（3 级层次）：Group 级加 exponent → Sub-group 级加 μexp（溢出处理）→ Element 级整数乘法。
- MX Adder：比较 exponent 取 max → 较小侧 mantissa 右移对齐 → 整数加法。
- 面积效率：MX8 SR 在 Mamba-2 上 perplexity ~11.5（接近 fp16 11.46）的同时面积开销显著低于 int8（避免 scale 乘法/比较逻辑）。
- Pimba 端到端准确率：geomean 差距 ≤0.3% vs GPU fp16。

涉及论文标题：
- 52-Pimba- A Processing-in-Memory Acceleration for Post-Transformer Large Language Model Serving..pdf

## Activation Sparsity in Diffusion Models（扩散模型的激活稀疏性）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Activation Sparsity in Diffusion Models 指扩散模型（DDPM）U-Net去噪网络在推理过程中，激活函数（SiLU/ReLU）后产生大量零值输出的现象。这是S-DMA加速器利用的核心特性。具体表现：(1) SiLU激活函数定义为 SiLU(x) = x·σ(x)（σ为sigmoid），当输入x<0且|x|较大时SiLU(x)≈0；ReLU(x)=max(0,x)直接对负值输出0。(2) 在U-Net各层中，大量输入特征落入激活函数的负值区间→输出零→有效稀疏性。(3) 稀疏性模式具有**空间相关性**——相邻像素输入分布相似→激活后的零值位置也相邻→形成空间聚簇的稀疏模式。(4) 稀疏性随timestep变化——去噪初期（t接近T，噪声大）激活值分布较随机→稀疏性较低且不规则；去噪后期（t接近1，接近真实图像）激活值分布结构化→稀疏性呈现更清晰的空间模式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
扩散模型中的激活稀疏性在U-Net计算中的具体表现（以某一ResBlock的Conv→SiLU为例）：
```
// U-Net某一层的Conv+SiLU计算
// 输入: X ∈ R^{C_in×H×W}, 权重: W ∈ R^{C_out×C_in×K×K}

Y = Conv2D(X, W)          // Y ∈ R^{C_out×H×W}, 密集计算
Z = SiLU(Y)               // Z[c][h][w] = Y[c][h][w] * sigmoid(Y[c][h][w])

// --- 激活稀疏性出现在Z中 ---
// 观察：对于大量 (c,h,w) 位置，Z[c][h][w] ≈ 0
// sparsity ratio ≈ count(Z≈0) / (C_out × H × W)
// 典型值：30%-70%稀疏性（取决于层和timestep）

// 下一层计算（如另一个Conv）
Y2 = Conv2D(Z, W2)        // 此时大量Z值≈0
// → 若实现zero-skip: 跳过Z≈0位置的MAC→节省计算
// → 传统dense执行: Z≈0位置仍然执行乘加→浪费

// S-DMA的SAP预测流程：
// 在计算Y时，SAP预测Z的sparsity mask
// SAP输入: X (或pre-SiLU的Y) → 预测输出: M_Z ∈ {0,1}^{C_out×H×W}
// M_Z[c][h][w]=0 → Z[c][h][w]≈0 → 下一层Conv跳过此位置
```

稀疏性分析的三个关键维度：
1. **Per-channel sparsity**：整channel是否全为零。若某channel的所有spatial位置SiLU输出都接近0→C-first数据流可跳过整channel。
2. **Per-spatial-location sparsity**：某spatial位置的所有channel是否全为零。若某(h,w)位置所有channel输出都接近0→S-first数据流可跳过整位置。
3. **Spatial correlation**：相邻位置的sparsity pattern高度相关——这是SAP预测的基础（可通过spatial convolution捕获）。

术语一般如何实现？如何使用？
Activation sparsity的利用方式：(1) **Runtime detection**——每层计算后检测实际零值→下一层跳过（简单但引入检测延迟，且无法提前调度）。(2) **Prediction-based**（S-DMA的SAP方式）——在计算当前层时预测下一层的sparsity mask→无延迟，可提前调度。(3) **Structured sparsity**——强制要求稀疏模式符合硬件友好结构（如2:4 structured sparsity），但可能须retraining。(4) **Unstructured sparsity**——利用模型自然产生的稀疏性（无需retraining），如S-DMA利用的SiLU自然稀疏。该术语在Cambricon-D [42]中以sign-mask近似形式使用，在BitVert [4]中使用weight sparsity（BBS），在S-DMA中的独特贡献在于将activation sparsity的空间相关性与硬件数据流调度（DAD）直接协同——SAP预测空间稀疏→DAD自适应选择最优skip粒度。

涉及论文标题：
- 50-S-DMA- Sparse Diffusion Models Acceleration via Spatiality-Aware Prediction and Dimension-Adaptive Dataflow..pdf

## RETRO Model (Retrieval-Enhanced Transformer)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RETRO（Retrieval-Enhanced TRansfOrmer）是Google DeepMind提出的代表性RALM（Retrieval-Augmented Language Model）架构，将autoregressive language model与基于向量搜索的外部知识检索器深度耦合。RETRO的核心设计：(1) **分块检索** — 输入序列被划分为固定大小的chunk（chunk_size个token），每retrieval_interval个token用最近chunk_size个token作为query检索外部知识库；(2) **Chunked Cross-Attention (CCA)** — 检索到的文本chunks经frozen BERT Encoder编码为embedding向量，在decoder block的CCA层中以K/V矩阵的形式与language model的Q向量进行cross-attention；(3) **迭代检索** — 每生成retrieval_interval个token触发新一轮检索，新的检索结果更新CCA层的K/V cache。RETRO的Decoder Block结构：Embedding → Self-Attention (SA, Q/K/V from input tokens) → Chunked Cross-Attention (CCA, Q from SA output, K/V from retrieved chunks) → Feed Forward Network (FFN) → next block。RETRO的参数量包括language model部分和frozen Encoder部分（Encoder不参与training/generation梯度更新）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RETRO模型推理pipeline（MNM paper使用的配置：12 decoder blocks, embedding_dim=384, input seq len=64, output seq len=2048）：
```
# RETRO autoregressive generation with iterative retrieval
chunk_size = 64
retrieval_interval = 8  # Config 5
tokens = input_tokens   # [64] initial tokens
retrieved_chunks = []   # 初始检索用前64 tokens

for step in range(2048):  # 生成至多2048 tokens
    # 检查是否需要检索
    if step % retrieval_interval == 0:
        query_tokens = tokens[-chunk_size:]  # 最近64 tokens
        query_embedding = Encoder(query_tokens)  # 384-dim向量
        
        # IVF-PQ检索（FAISS）
        clusters = cluster_selection(query_embedding)  # 选top nprobe clusters
        candidates = []
        for cluster in clusters:
            for codeword in cluster.codewords:
                distance = L2(query_embedding, reconstruct(codeword))
                candidates.append((distance, codeword))
        top_k = topk_select(candidates, k=16)  # Config 5: top-16
        
        # 获取原始文本并编码
        raw_data = database.fetch(top_k.ids)
        retrieved_chunks = Encoder.encode(raw_data)  # [k, chunk_embedding_dim]
    
    # Decoder Block forward（逐层）
    embedding = Embedding(tokens[-1])  # 最新token的embedding
    
    for block in range(12):  # 12 decoder blocks
        # Self-Attention (SA)
        Q, K, V = embedding × W_Q, embedding × W_K, embedding × W_V
        # K, V追加到cache: K_cache.append(K), V_cache.append(V)
        attn_scores = Q × K_cache^T        # GEMV-dominant Score
        attn_weights = softmax(attn_scores)
        sa_output = attn_weights × V_cache  # GEMV-dominant Context
        sa_output = sa_output × W_Proj      # GEMM Projection
        
        # Chunked Cross-Attention (CCA)
        Q_cca = sa_output × W_Q_cca
        # K/V_cca来自检索结果（retrieved_chunks），在两次检索之间cache不变
        attn_scores_cca = Q_cca × K_cca^T
        attn_weights_cca = softmax(attn_scores_cca)
        cca_output = attn_weights_cca × V_cca
        
        # FFN
        ffn_output = GELU(cca_output × W_FFN1) × W_FFN2
        embedding = ffn_output + cca_output  # residual
    
    # LM Head
    next_token_logits = embedding × W_LMHead
    next_token = argmax(softmax(next_token_logits))
    tokens.append(next_token)
```

RETRO的关键算法特性（影响硬件加速设计）：(1) **SA MHA是memory-bound GEMV** — 每token的K/V矩阵唯一（unique per request per decode step）→无法使用GEMM batched → GEMV memory-bound → MNM选择PIM加速；(2) **CCA MHA同样是GEMV** — K/V来自retrieval结果，在两次retrieval（retrieval_interval=8 tokens）之间保持不变→cache hit期间K/V复用→但score仍是GEMV；(3) **QKV generation, Projection, FFN是GEMM** — 可batch并行 → GPU tensor core高效 → 保留在GPU。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **开源实现**：RETRO的Google DeepMind开源代码（TensorFlow实现，GitHub: google-research/retro）提供了完整的training和inference pipeline。MNM paper使用此开源RETRO代码进行GPU profiling和workload characterization。FAISS-GPU [37]作为IVF-PQ检索后端。
- **模型大小**：RETRO-0.5B (~500M params, 12 decoder blocks), RETRO-1.5B, RETRO-7.5B。Encoder（frozen BERT）参数量不计入生成模型参数。各模型的num_heads和d_head不同但embedding_dim=384保持不变（匹配检索向量的384-dim）。
- **数据集**：Realnewslike subset of C4 [15]（140M vectors, 384-dim FP16 embedding）；Wikipedia [17]（80M vectors, 384-dim FP16）。
- **Perplexity评估**：RETRO使用perplexity（lower is better）评估检索配置对生成质量的影响——Config 1 (nprobe=16) perplexity ~20.1, Config 5 (nprobe=256) perplexity ~18.62→检索越充分perplexity越低。
- **检索配置tradeoff**：Config 1→Config 5: nlist减小(32768→2048), nprobe增大(16→256), interval减小(64→8), top-k增大(1→16)→检索latency从4.4%升至88.5% of total latency→高度hardware-dependent。

涉及论文标题：
- 53-Accelerating Retrieval Augmented Language Model via PIM and PNM Integration.pdf

## Chunked Cross-Attention (CCA) in RALM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Chunked Cross-Attention (CCA)是RETRO模型Decoder Block中的特殊attention层，将language model的self-attention输出（作为Q）与从外部知识库检索到的文本chunks的编码表示（作为K和V）进行cross-attention。与Self-Attention（Q/K/V均来自同一输入序列）不同，CCA的K和V来自外部检索数据，在两次连续检索之间保持不变（cache），使生成过程能持续融入外部知识。CCA的设计使得language model可以在不改变自身参数的情况下动态访问外部知识库——检索到的chunks被frozen Encoder（如BERT）编码为固定维度embedding，直接作为CCA的K/V矩阵。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CCA在RETRO Decoder Block中的计算流程（位于SA和FFN之间）：
```
# === CCA Layer Forward (per token, per decoder block) ===

# Input: sa_output [1 × d_model] — SA层的输出
# Stored: K_cca_cache [k × d_model], V_cca_cache [k × d_model]
#          — 来自最近检索的top-k个retrieved chunks的Encoder encoding
#          — 在retrieval_interval个token间保持不变

# Step 1: Q projection (from SA output)
Q_cca = sa_output × W_Q_cca        # [1 × d_model] × [d_model × d_head*n_head]
Q_cca = reshape(Q_cca, [n_head, d_head])  # 分head

# Step 2: Cross-attention Score (Q_cca · K_cca^T)
# K_cca_cache: [k × d_model] → reshape to [n_head, k, d_head]
for head in range(n_head):
    for i in range(k):  # k=top-k retrieved chunks (1~16 depending on config)
        # GEMV: 每个chunk的K向量与Q_cca向量dot-product
        score[head][i] = Σ(Q_cca[head][j] * K_cca[head][i][j])  # j=0..d_head-1

# Step 3: Softmax
attn_weights = softmax(score / sqrt(d_head))  # [n_head × k]

# Step 4: Context (attn_weights · V_cca)
for head in range(n_head):
    context[head] = Σ(attn_weights[head][i] * V_cca[head][i])  # i=0..k-1
    # weighted sum over retrieved chunks

# Step 5: Concatenate & Project
cca_output = concat(context[0..n_head-1])  # [1 × d_model]
cca_output = cca_output × W_Proj_cca       # [1 × d_model] × [d_model × d_model]

# Output: cca_output [1 × d_model] — 喂入FFN
```

CCA的memory-bound特征：每次CCA的Score计算是Q_cca向量[1×d_head]与K_cca矩阵[k×d_head]的GEMV→当k较小时（如Config 1, k=1）→几乎纯GEMV→memory bandwidth critical。当k较大时（如Config 5, k=16）→接近small GEMM→但仍memory-bound（因为k*d_head << GPU SM compute capacity）。这就是MNM用PIM加速CCA（和SA）MHA的根本原因——K_cca_cache存储在HBM bank中，PIM MAC直接在bank内做dot-product。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **Chunk encoding**：Retrieved text chunks被frozen Encoder（如BERT-base）编码为固定维度向量。Encoder输出经过额外的线性projection（W_Proj_chunk）映射到d_model维度后存入K/V cache。Encoder可以offline预计算（对静态database）或online编码（对dynamic/web data）。
- **K/V cache管理**：CCA的K/V cache在两次retrieval之间有效（interval内cache hit）→新retrieval触发时更新K/V cache。Cache占用memory：k × d_model × 2 (K+V) × n_decoder_blocks × FP16 bytes。示例：RETRO-7.5B, d_model=4096, k=16, 12 blocks → 16 × 4096 × 2 × 12 × 2B ≈ 3.1MB per request。
- **与Self-Attention K/V cache的区别**：SA的K/V cache per token持续增长（每生成1个token追加1行K/V）→长度线性增长。CCA的K/V cache是per retrieval固定大小（k个chunks）→两次retrieval间不变→但retrieval时全部替换（不同retrieval结果的chunk通常不同）。
- **在MNM paper中的角色**：CCA的MHA是PIM acceleration的主要target之一（与SA MHA并列）。CCA K/V cache存在HBM bank中→PIM MAC可直接使用PIM_ACT_AB激活对应row→无需将K/V搬到GPU→消除GPU-HBM bandwidth压力。

涉及论文标题：
- 53-Accelerating Retrieval Augmented Language Model via PIM and PNM Integration.pdf

## IVF-PQ with Precomputed LUT (预计算查找表的倒排文件乘积量化检索)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
IVF-PQ with Precomputed LUT是FAISS [16]向量相似性搜索库中的核心检索算法，通过Inverted File（倒排索引）做粗粒度cluster筛选、Product Quantization（乘积量化）压缩向量、预计算LUT（Lookup Table）加速L2距离计算。该算法专门服务于RALM系统的外部知识检索——给定查询向量x（384-dim FP16 embedding），在包含数千万到数十亿向量的数据库中找出top-k最近邻。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
IVF-PQ检索的完整算法pipeline（以MNM Config 5: nlist=2048, nprobe=256, M=64, ksub=256, top-k=16为例）：
```
# === 预处理阶段（offline，在检索前完成） ===
# Step A: IVF聚类
# 将数据库中N=140M vectors用k-means分为nlist=2048个cluster
# 每cluster有centroid y_c [384-dim]
# 每data vector y存储为residual: y_R = y - y_c (相对于centroid的差值)

# Step B: PQ Codebook训练
# 对每个cluster的residual vectors y_R:
#   将每个384-dim y_R分为M=64个sub-vectors (每sub-vector dsub=384/64=6维)
#   对每sub-vector用k-means (ksub=256)聚类→256个codebook entries per sub-vector
#   每data vector的PQ编码: 64个8-bit indices (共64B) → 384-dim→64B 压缩比6×
#   每个8-bit index指向对应sub-vector的codebook entry (存储FP16的6-dim sub-vector)
PQ_codebook[subvec_id=0..63][code_id=0..255] = 6-dim FP16 sub-vector centroid

# Step C: Precomputed LUT (per cluster)
# 对每个cluster c:
for code_id in 0..255:  # ksub=256
    for subvec_id in 0..63:  # M=64
        y_R_sub = PQ_codebook[subvec_id][code_id]  # 6-dim FP16
        y_c_sub = cluster_centroid[c][subvec_id*6 : (subvec_id+1)*6]  # 6-dim
        LUT[c][code_id][subvec_id] = ||y_R_sub||^2 + 2 * dot(y_c_sub, y_R_sub)
        # 大小: nlist × ksub × M = 2048 × 256 × 64 = 33.6M FP16 entries ≈ 67MB
        # 每cluster: 256 × 64 = 16K FP16 entries ≈ 32KB

# === 在线检索阶段（per query） ===
# Step 1: Coarse-grained search (Cluster Selection)
query_x = [384-dim FP16]
dist_to_centroid = []
for c in 0..nlist-1:  # nlist=2048
    d = ||query_x - cluster_centroid[c]||^2  # 384-dim L2, 可在GPU上batched GEMV
    dist_to_centroid.append(d)
selected_clusters = topk(dist_to_centroid, nprobe=256)  # 选最近的256个cluster
# 记每个selected cluster的 ||x - y_c||^2

# Step 2: Fine-grained search (PQ Code Scan) — 对每个selected cluster
for cluster_c in selected_clusters:
    # 加载该cluster的precomputed LUT (16K entries)
    lut_c = LUT[cluster_c]
    
    # 遍历cluster内所有PQ codewords
    for codeword in cluster_c.codewords:  # cluster内可能有数千个codewords
        # 每个codeword: 64B = M=64个8-bit indices
        
        # 计算 x · y_R (residual dot-product)
        dot_product = 0
        for subvec_id in 0..63:  # M=64
            code_id = codeword.indices[subvec_id]  # 8-bit index
            y_R_sub = PQ_codebook[subvec_id][code_id]  # 6-dim FP16
            # dot product over 6 dimensions
            for dim in 0..5:
                dot_product += query_x[subvec_id*6 + dim] * y_R_sub[dim]
        
        # 查表获取预计算项
        lut_sum = 0
        for subvec_id in 0..63:
            code_id = codeword.indices[subvec_id]
            lut_sum += lut_c[code_id][subvec_id]
        
        # 最终L2距离
        # d = ||x - y||^2 = ||x - y_c - y_R||^2
        #   = ||x - y_c||^2 + ||y_R||^2 + 2(y_c·y_R) - 2(x·y_R)
        #   = centroid_dist + lut_sum - 2 * dot_product
        distance = centroid_dist[cluster_c] + lut_sum - 2 * dot_product
        distances.append((codeword.id, distance))

# Step 3: Top-k Selection
final_topk = topk_select(distances, k=16)  # 全局top-16最近邻
# 返回: [(distance_1, id_1), ..., (distance_16, id_16)]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **FAISS实现**：Meta FAISS库提供CPU和GPU两种版本的IVF-PQ实现。GPU版本（FAISS-GPU [37]）使用CUDA kernel并行化PQ code scan中的dot-product，但LUT access受GPU shared memory容量限制（每cluster 16K entries × 2B = 32KB，但shared memory仅132KB/SM→多cluster并存时超出）→FAISS-GPU通常on-the-fly计算部分距离项而非使用precomputed LUT→增加算术但减少memory access。MNM paper对比了precomputed LUT vs on-the-fly的EDAP，prove LUT方案在PNM硬件上更优（384-dim时EDAP降36%）。
- **MNM PNM的硬件LUT**：PNM的dedicated SRAM LUT可每cluster一次加载（GPU DMA→PNM_WR_MMIO），所有codewords共享。PNM LUT area仅占总PQ scanner面积的13.7%→EDAP优于on-the-fly recomputation。
- **GPU vs PNM tradeoff**：GPU上precomputed LUT的working set > shared memory →需global memory access →LUT lookup latency高→不如recompute。PNM上dedicated SRAM→LUT access latency低→查表优于recompute。这体现了硬件平台对算法参数选择的根本影响。
- **Embedding维度影响**：MNM paper评估了192-D到1536-D多种embedding维度（对应不同embedding模型：all-MiniLM-L6-v2 [88] 384-dim, OpenAI embedding 1536-dim [79], DPR [39] 768-dim等）。PNM w/ LUT在所有维度下都优于w/o LUT→但improvement随维度变化（192-D: 13%, 384-D: 36%, 768/1536-D: 27-29%）。
- **检索质量配置**：nprobe/nlist/top-k/M的组合决定检索质量(retrieval accuracy)和延迟tradeoff。MNM paper Table 1中的5种config从低质量低延迟(Config 1)到高质量高延迟(Config 5)。

涉及论文标题：
- 53-Accelerating Retrieval Augmented Language Model via PIM and PNM Integration.pdf

## Hybrid Transformer-Mamba Language Model

术语解释
Hybrid Transformer-Mamba是一种将传统Transformer注意力层与Mamba-2状态空间模型(SSM)层交错组合的混合语言模型架构，兼顾Transformer的表达能力和Mamba的长序列效率。

术语是什么？
Hybrid Transformer-Mamba模型通过交替排列attention layers和Mamba-2 layers（典型比例为每6-10层Mamba-2后插入1层attention），以两方互补的方式解决各自的计算瓶颈：(1) Mamba-2用线性计算复杂度和常数内存纠正Transformer的二次复杂度和KV cache膨胀；(2) Attention用强语言建模能力弥补Mamba-2在recall和in-context learning上的退化。HLX论文评估的Hybrid-2.7B (Mamba2attn-2.7B)包含64层：6 attention layers（multi-head attention, 30 heads × dhead 128）+ 58 Mamba-2 layers（SSD with 80 heads × dhead 64, dstate 128）。前向过程：token embedding → 循环执行 [attention layer (RMSNorm → QKV gen → FA-2 → Out Proj → Residual) 或 Mamba-2 layer (RMSNorm → In Proj → Conv1D → SiLU → SSD → z-gating → RMSNorm → Out Proj → Residual)] → LM Head → output token。随seqlen增长（1K-128K），latency bottleneck从Mamba-2 dominated（短seqlen，Mamba-2层数多）切换到attention dominated（长seqlen > 128K，attention二次复杂度体现）。

从算法pipeline角度拆解术语：
Hybrid-2.7B推理pipeline (per token，prefill phase):
1. Token Embedding: input_ids [batch, seqlen] → embeddings [batch, seqlen, dim]
2. For each layer l in 0..63:
   - 若l为attention layer: x = RMSNorm(x); Q,K,V = x @ W_QKV; O = FA-2(Q,K,V); x = x + O @ W_O
   - 若l为Mamba-2 layer: x = RMSNorm(x); dt,xBC,z = x @ W_in; 其中z=gate; xBC = Conv1D(xBC) → SiLU; x,B,C = split(xBC); Y = SSD(A,dt,x,B,C); x = x + (z ⊙ Y) @ W_out
3. LM Head: x @ W_lm_head → logits [batch, seqlen, vocab_size]
HLX paper仅加速prefill phase。

术语一般如何实现？如何使用？
代表性Hybrid模型：Jamba [27]、Samba [45]、Zamba [16]、Jamba-1.5 [49]、Mamba2attn [11]。实现上通常复用现有attention kernel (FA-2/FA-3) 和Mamba-2/SSD kernel。HLX paper指出：因为两个kernel的compute pattern不同，在GPU上两者compute utilization差距大(FA-2 ~61% vs SSD ~27%)→需要统一加速方案。HLX的URSC通过reconfigurable pipeline同时高效支持两种operation。

涉及论文标题：
- 54-HLX- A Unified Pipelined Architecture for Optimized Performance of Hybrid Transformer-Mamba Language Models..pdf

## Fused SSD (Fused State-Space Duality)

术语解释
Fused SSD是将Mamba-2的SSD算法的5个独立GPU kernel合并为单一fused kernel的技术，类似于FA-2将attention的多个操作合并为单一fused kernel。

术语是什么？
SSD原始实现由5个独立kernel组成：chunk cumsum、chunk state、state passing、BMM chunk、chunk scan。每个kernel产生中间数据写入DRAM，下一个kernel再从DRAM读取→大量DRAM流量且无数据复用→memory-bound（GPU compute utilization ~27%）。Fused SSD将5个kernel合并为一个，模仿FA-2的block-level fusion：用单个for循环遍历chunk blocks（而非KV blocks的双重循环），在每个block内完成所有6个operation：(1) sdt=softplus(dt+dtbias); dA=sdt×A; dACS=cumsum(dA); decay_states=exp(dACS[-1:]-dACS); d2t=decay_states×sdt; (2) CBT=C×B^T; CBTLdt=CBT×L×sdt; YDiag=CBTLdt×x; (3) dCOff=exp(dACS)×C; YOff=dCOff×states(j-1); (4) YFinal=YDiag+YOff; (5) states(j)=exp(dACS[-1])×states(j-1)+dBdt^T×x。但fused SSD在GPU上性能反退化1.74×——因为中间数据642KB/block超出per-SM memory（256KB register + 164/224KB shared memory）→register spilling→occupancy下降。

从算法pipeline角度拆解术语：
Fused SSD算法流程 (per chunk block j):
```
# Input: dt[j], dtbias, A, x[j], B[j], C[j]; states(j-1)
sdt = softplus(dt[j] + dtbias)
dA = sdt * A
dACS = cumsum(dA)
decay_states = exp(dACS[-1:] - dACS)
d2t = decay_states * sdt
CBT = matmul(C[j], B[j]^T)
CBTLdt = CBT * L * sdt
YDiag = matmul(CBTLdt, x[j])
dCOff = exp(dACS) * C[j]
YOff = matmul(dCOff, states(j-1))
YFinal = YDiag + YOff
dBdt = d2t * B[j]
statesN = matmul(dBdt^T, x[j])
states(j) = exp(dACS[-1]) * states(j-1) + statesN
Write YFinal to DRAM
```
与FA-2 fused kernel的关键差异：FA-2用双重循环(Q blocks × KV blocks, O(N²))；fused SSD用单循环(chunk blocks, O(N))，但存在column-wise dependency (states(j-1)→states(j)) 和row-wise dependency (YDiag+YOff, statesint+statesN)。

术语一般如何实现？如何使用？
HLX paper将fused SSD进一步发展为PipeSSD——在fused SSD基础上加入fine-grained pipelining，分解为3个pipeline stage并控制各行处理粒度→中间数据从642KB降至58.5KB (11× reduction)→可在HLX URSC硬件上高效执行。Fused SSD本身作为纯软件方案在GPU上因register pressure失败→说明了硬件-算法co-design的必要性。

涉及论文标题：
- 54-HLX- A Unified Pipelined Architecture for Optimized Performance of Hybrid Transformer-Mamba Language Models..pdf

## Floating-Point Multiplication Approximation (FPMA) / Mitchell's Logarithmic Approximation

术语解释
FPMA（浮点乘法近似）是一种用整数加法替代浮点乘法的方法，基于Mitchell对数近似原理。根据IEEE 754，标准浮点数 x=(-1)^S·2^(E-B)·(1+M)，取对数得 log2(|x|)=E-B+log2(1+M)≈E-B+M。由此 FP 乘法 r=x·y 可近似为 log2(|r|)≈(E_x+M_x)+(E_y+M_y)-2B，即 R=X+Y-B，其中 X=E_x+M_x, Y=E_y+M_y。结果R本身即为标准浮点格式（exponent+mantissa顺序一致），无需特殊反转换。所有运算均为整数加法，彻底消除浮点乘法器。

术语是什么？
FPMA将浮点乘法转化为整数加法：对齐两个操作数的二进制表示（E_bits||M_bits），做整数加法，再减去一个bias。近似误差来源于 log2(1+M)≈M 的线性化——该近似在M=0和M≈1时准确，M≈0.5时误差最大（~0.086）。对于多层的LLM，该误差逐层累积可导致显著的PPL退化（如在OPT-6.7B上，不加补偿的mpFPMA使PPL从FP16的10.86升至11.83）。

从算法pipeline角度拆解：
以AxCore的混合精度FPMA (mpFPMA, FP16 activation×FP4 weight) 为例：
```
输入: A[FP16: S1E5M10], Wq[FP4: S1E2M1 或 S1E1M2 或 S1E3M0], B1, C1
1. Align: Wq_aligned = Wq << (10 - NM_FP4)  # 对齐mantissa位宽至FP16域
2. R = A + Wq_aligned - B1 + C1  # 所有操作均为整数加法
   - B1 = Ba + Bwq - Br  # bias修正: FP16 bias=15, FP4 E2M1 bias=1
   - C1: precomputed mean error compensation constant
输出: R[≈ FP16], 与A×Wq的精确FP16乘积近似
```
关键参数：B1为格式感知的bias修正项，在activation和result同为FP16时简化为B1=Bwq。C1为per-format-pair平均误差补偿常数，由 C1=(1/(2^NMa·2^NMw))·Σ_{ma,mw} ε(ma,mw) 一次性precompute。

术语一般如何实现？如何使用？
- Mitchell (1962) 首创，Gustafsson & Hellman (ARITH 2021) 理论化。AxCore首次将FPMA扩展到混合精度场景(mpFPMA)并应用于quantized LLM inference。
- 实现要点：(1) 对齐操作数的fixed-point表示，确保radix point一致；(2) 格式感知bias修正B1处理不同exponent bias；(3) 补偿策略——AxCore使用单常数均值补偿(比per-mantissa-pair查表法省storage)，对FP16×FP4/FP16×FP8/BF16×FP4等格式对均适用；(4) 低比特FP格式下必须处理subnormal值（SNC），因subnormal无hidden leading 1→FPMA数学基础不成立。
- 硬件实现：PE内仅需轻量整数加法器（W4A16场景下7-bit adder）替代24-bit FP乘法器，同时将bias/compensation计算移出PE（PreAdd）共享。Dequantization也可用FPMA实现(AxScale)：O=Oq+S-B+C2，两次整数加法替代缩放乘法。

涉及论文标题：
- 56-AxCore- A Quantization-Aware Approximate GEMM Unit for LLM Inference.pdf

## Weight-Only Quantization with Direct Mixed-Precision GEMM (mpGEMM)

术语解释
Weight-only quantization 是LLM推理中只将模型权重量化到低比特（INT4/FP4/INT8/FP8），而保持activations在较高精度（FP16/BF16）的量化策略。Direct mpGEMM 是指在GEMM计算中直接使用低比特权重与高精度activation做混合精度乘加，将dequantization推迟到accumulation之后，而非先dequantize权重再计算。

术语是什么？
LLM推理中，权重占总参数量绝对多数且为静态（离线可知），易于离线量化；激活是动态的、输入依赖的，激进量化导致显著精度损失。因此weight-only quantization成为学术界和工业界的标准（GPTQ [Frantar+, ICLR'23]、AWQ [Lin+, MLSys'24]、TensorRT-LLM、NVIDIA FP4）。GEMM的执行模式有三种：(a) Standard GEMM: FP16×FP16；(b) Indirect GEMM: Wq→dequantize→Wr(FP16)，再与A(FP16)做标准GEMM；(c) Direct mpGEMM: A(FP16)×Wq(INT4/FP4)直接计算，accumulation后再dequantize。Direct mpGEMM更高效因为GEMM unit datapath更轻量（处理低比特权重而非全精度），且避免了per-weight dequantization开销。

从算法pipeline角度拆解：
```
# 三种GEMM模式对比 (W4A16)
# (a) Standard GEMM: 无量化 → 最大精度、最大硬件成本
O[i,j] = Σ_k A[i,k] * W[k,j]          # FP16×FP16, FP16乘法器+FP32累加

# (b) Indirect GEMM: 先反量化
W_r[k,j] = W_q[k,j] * s[j]            # N次FP16乘法(per-weight dequant)
O[i,j] = Σ_k A[i,k] * W_r[k,j]        # FP16×FP16 GEMM

# (c) Direct mpGEMM (AxCore): 直接混合精度，延迟反量化
O_q[i,j] = Σ_k A[i,k] * W_q[k,j]      # FP16×FP4, 轻量加法器(FPMA)或INT乘法器(FIGNA)
O[i,j] = O_q[i,j] * s[j]              # per-channel FP16缩放(可FPMA近似)
```
Group-wise quantization 进一步将权重tensor分成更小的group（如128或64 elements），每组分配独立scaling factor（FP16），更好捕获局部分布→减少量化误差，代价是额外存储scales。

术语一般如何实现？如何使用？
- 工业实现：NVIDIA TensorRT-LLM (FP8/INT8/INT4 weight-only)、Apple MLX、Qualcomm AI Engine。常用量化方法：GPTQ (逐列贪心+Cholesky分解更新未量化权重)、AWQ (activation-aware per-channel scaling)、GPTAQ (asymmetric calibration)。
- 硬件支持：FIGNA [HPCA'24] 使用INT4-FP16 mpGEMM unit (INT multiplier)、FIGLUT [HPCA'25] 使用LUT-based FP-INT GEMM、AxCore [MICRO'25] 使用multiplier-free FPMA。TPUv4/v5支持INT8 weight-only via 其systolic array。
- 格式选择：INT4均匀分布→实现简单但精度次优（LLM权重呈Gaussian-like分布）；FP4 (E2M1/E1M2/E3M0) 非均匀分布→更多表示能力在零附近→更高的精度潜力。NVIDIA FP4格式(NVIDIA TensorRT 10.11+)和LLM-FP4 [Liu+, EMNLP'23]都舍弃inf/NaN表示，将所有bit pattern用于有效数值。
- Group size tradeoff: smaller group→better accuracy but more scale storage overhead。典型值：OPT group_size=128、LLaMA2 group_size=64。

涉及论文标题：
- 56-AxCore- A Quantization-Aware Approximate GEMM Unit for LLM Inference.pdf

## Adaptive Format-Aware Quantization (Block-wise FP4 Format Selection)

术语解释
Adaptive format-aware quantization 是一种block-wise的自适应量化策略：将权重矩阵划分为小的block，对每个block在候选低比特格式集合中选择使量化误差最小的格式，而非全局使用单一格式。

术语是什么？
传统量化使用统一的低比特格式（如全部INT4或全部FP4 E2M1），但LLM各层甚至同层不同block的weight分布差异巨大。以Llama2-7B为例：Layer 0的attention output weight分布呈sharp peaks，适合power-of-two-like格式(E3M0)；Layer 29分布更宽且均匀，适合E2M1/E1M2。Uniform format无法同时适配这两种分布，导致某些block量化误差过大。Adaptive format-aware quantization通过block-wise format selection，使每个block使用最优格式，整体量化误差最小化。

从算法pipeline角度拆解：
```
# AxCore block-wise format-aware quantization (W4A16, group_size=g, tilesize=n)
候选格式 D = {E3M0, E2M1, E1M2}

for weight_matrix W [K, N]:
    for block in partition(W, block_size=(g, n)):
        for format d in D:
            W_d = quantize_dequantize(W[block], format=d)
            error_d = ||A_calib * W_d - A_calib * W[block]||^2
        best_format = argmin_d error_d
        store(W_q[block], best_format_flag, S[block])
```
在线推理时，SNC单元根据FormatSel信号选择对应格式的decoder通路，将不同格式统一为S1E3M2内部表示→下游logic完全格式无关。block_size要求g和n均为GEMM array size的倍数以保证array调度对齐。

术语一般如何实现？如何使用？
- 类比方法：ANT [Guo+, MICRO'22] 提出tensor-wise adaptive numerical data type但粒度粗糙；M-ANT [Hu+, 2025] 的group-wise approach更细粒度但仍用单一格式per tensor。AxCore首次将block-wise格式选择与FPMA co-design。
- 实现考虑：(1) 候选格式集合的穷举：需要设计覆盖不同分布特性的格式集合（power-of-two→uniform）。AxCore选择{E3M0, E2M1, E1M2}覆盖三种典型分布。(2) Format flag存储：per block需额外2-bit格式标识，相比per-group scale的FP16开销可忽略。(3) 硬件支持：PE需要format-decode逻辑（SNC），但输出统一格式后下游logic可复用。

涉及论文标题：
- 56-AxCore- A Quantization-Aware Approximate GEMM Unit for LLM Inference.pdf

## Subnormal Number Handling in Low-Bit Floating-Point

术语解释
在IEEE 754浮点数中，subnormal（次正规数，旧称denormal）是exponent全零且没有hidden leading 1的极小值，公式为 x=(-1)^S·2^(1-B)·M。在FP16/FP32中subnormal是极小值的罕见边界情况（<10^-38 in FP32），但在低比特格式如FP4中，由于exponent bits极少（2 bits→4种exponent level），subnormal可represent up to 0.5的值且频繁出现。

术语是什么？
FPMA依赖近似 log2(1+M)≈M，这建立在"mantissa有hidden leading 1"的前提上。Subnormal没有这个leading 1（即log2(0+M)≠M），因此FPMA对subnormal值数学上不成立。在FP4量化中，subnormal出现比例远高于高精度格式，不可忽视。AxCore的Subnormal Number Conversion (SNC)通过硬件查表将subnormal编码映射到数值最近邻的normalized编码。

从算法pipeline角度拆解：
以E1M2格式为例，Subnormal→Normal映射表：
```
Subnormal编码 → 十进制值 → 最近Normal编码 → Normal十进制值
S-0-00        → 0        → return 0        → 0
S-0-01        → 0.5      → S-0-00 = 0.5↑ / return 0↓ (stochastic)
S-0-10        → 0.5      → S-00-0 = 0.5    → 0.5
S-0-11        → 0.75     → S-00-1 = 0.75   → 0.75
```
无法精确映射的值使用stochastic rounding（随机bit取自activation mantissa最高位）→交替平衡累积误差。消融实验：mpFPMA+SNC将PPL从11.83降至11.45 (OPT-6.7B)。

术语一般如何实现？如何使用？
- 现有工作大多忽略低比特FP的subnormal问题。AxCore的SNC实现为per-PE硬件模块：运行时检测subnormal→查表转换→输出unified internal S1E3M2格式。PE内SNC开销仅占PE总面积的3.5%。Stochastic rounding利用pre-existing activation bits→无额外RNG开销。SNC coverage涵盖FP4三种子格式(E3M0/E2M1/E1M2)，扩展原理适用FP8等格式。

涉及论文标题：
- 56-AxCore- A Quantization-Aware Approximate GEMM Unit for LLM Inference.pdf

## Accuracy-Throughput Pareto Frontier (Model Variant Selection)

术语解释
Accuracy-Throughput Pareto Frontier是指在推理服务系统中，从所有可能的模型变体-设备映射配置中筛选出的帕累托最优（Pareto optimal）配置集合——对于给定吞吐需求，帕累托前沿上的配置提供最高的可达系统精度。它是在异构集群上执行accuracy scaling时选择模型变体组合的理论基础。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
在包含M个模型变体和N个异构设备的推理服务集群中，共有O(M^N)种可能的模型变体-设备分配方案。每种方案产生不同的(system throughput capacity, system accuracy)点。这些点中的许多被其他配置严格支配（strictly dominated）——即存在另一配置在同等或更高throughput下有更高accuracy。帕累托前沿就是去除所有被支配配置后的optimal trade-off曲线。例如图1b：5个EfficientNet变体+5个设备=3125种可能配置→筛选出帕累托前沿上的少数关键配置——图1b右子图中红色曲线上的点。对于给定目标吞吐（如100 QPS），在帕累托前沿上选择对应的配置即可得到highest reachable accuracy（约84%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
构建Accuracy-Throughput Pareto Frontier的计算流程：
```
Input: M model variants, N heterogeneous devices
       For each (m, d): throughput P_d,m (profiled), accuracy A_m

1. Enumerate all configurations:
   For each assignment f: [1..N] → [1..M] (device d hosts model f(d)):
       system_throughput = sum_{d} P_d, f(d)
       system_accuracy = weighted_avg(A_{f(d)})  // per query type weight

2. Filter Pareto-optimal:
   Sort configurations by throughput ascending
   Pareto_set = []
   max_acc_so_far = -inf
   For each config in sorted:
       if config.accuracy > max_acc_so_far:
           Pareto_set.append(config)
           max_acc_so_far = config.accuracy

3. Given target throughput T:
   Select config in Pareto_set with throughput >= T and max accuracy
```
实际中，Proteus并不显式枚举所有O(M^N)配置并构建帕累托前沿，而是将帕累托选优嵌入MILP的目标函数（max Σ A_m·z）和约束（Σ z ≥ S_target），让MILP solver隐式搜索帕累托前沿上的最优解。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
帕累托前沿的特性：(1) 凸性（convexity）不保证——离散的model-device assignment产生non-convex Pareto front→不能用简单的convex optimization。(2) 离散变量——x_d,m是布尔变量→MILP是合适的求解方法。(3) 多目标——帕累托前沿本质上是throughput和accuracy的双目标trade-off。Proteus通过将throughput作为约束、accuracy作为目标来单一化。(4) 在Proteus中，帕累托前沿的概念用于验证MILP解的最优性——实验显示MILP总是能找到帕累托前沿上的配置，而INFaaS的greedy heuristic则常落在被支配区域（lower accuracy at same throughput）。

涉及论文标题：
- 59-Proteus- A High-Throughput Inference-Serving System with Accuracy Scaling.pdf

## System Accuracy / Effective Accuracy (in Inference Serving)

术语解释
System Accuracy（也称为Effective Accuracy）是推理服务系统的顶层度量，定义为所有被系统成功服务的查询请求的平均精度。与单个模型的accuracy不同，system accuracy反映了系统在运行期间由于accuracy scaling（切换不同精度的模型变体）导致的整体服务质量。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
System Accuracy的计算：Effective Accuracy = Σ_q a_q，其中a_q = Σ_m A_m · (Σ_d x_d,m · z_d,q · s_q)。展开解释：(1) s_q是query type q的到达速率（QPS）；(2) z_d,q是设备d服务query type q的QPS；(3) x_d,m表示设备d是否host模型变体m；(4) A_m是模型变体m的精度（通常归一化，如除以该模型家族最精确变体的精度）；(5) a_q是所有query type q请求的total accuracy sum。System Accuracy = 各query type的accuracy加权和除以total queries。注意它只计成功服务的请求——timeout/SLO violation的请求不计入accuracy（proteus中timeout请求被drop）。
Accuracy Drop = 100% - Effective Accuracy（归一化后），衡量因accuracy scaling造成的精度损失。Proteus在高峰期最大accuracy drop仅4.85%。

从算法pipeline角度拆解，给出具体例子：
以Twitter trace的1分钟窗口为例——同时服务EfficientNet（80% of queries, 归一化精度范围80%-100%）和ResNet（20% of queries, 归一化精度范围82%-100%）：
- 低峰期：所有query由最精确变体服务→a_EfficientNet = 80% × 100% = 80.0, a_ResNet = 20% × 100% = 20.0→System Accuracy = 100%
- 高峰期：EfficientNet的50% query切换到B0（80%精度），ResNet不变→a_EfficientNet = 40% × 100% + 40% × 80% = 72.0, a_ResNet = 20.0→System Accuracy = 92.0%→Accuracy Drop = 8%

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
System Accuracy必须与throughput和SLO violation ratio一起解读——单独的accuracy可以很高（如Clipper-HA始终用最精确变体→100% accuracy）但可能伴随极高的SLO violations（高峰期50%+的请求timeout被drop，不计入accuracy统计）。因此，Proteus报告三个指标联立：(1) throughput (QPS)、(2) effective accuracy、(3) SLO violation ratio。实验证明Proteus在三个指标上同时优于baseline：throughput比无scaling的Clipper-HA高60%、accuracy drop仅为4.85%（vs INFaaS 13.7%、Sommelier 16%）、SLO violation ratio最低。

涉及论文标题：
- 59-Proteus- A High-Throughput Inference-Serving System with Accuracy Scaling.pdf

---

## Attention Algorithm Pass Classification (3-pass / 2-pass / 1-pass)

术语解释
Attention Algorithm Pass Classification是FuseMax基于Cascade of Einsums形式化提出的attention算法分类学，以算法对attention的M fiber（sequence length dimension）所需遍历次数（pass数）为标准，将文献中所有numerically stable attention算法分为三类。

术语是什么？
Pass定义为对某一tensor某一rank的fiber的完整遍历。每次在遍历完所有fiber元素后需要再次访问同一元素的次数，即为additional pass。对于attention的M fiber（对应sequence length），不同算法需要1/2/3次pass。**3-pass cascade**（PyTorch/TensorFlow/FLAT/E.T.）：Pass 1→compute QK + global max (GM)；Pass 2→compute numerator SN=e^(QK-GM) + denominator SD=ΣSN（需re-read M fiber of QK）；Pass 3→compute A=SN/SD + AV（需re-read M fiber of SN）。Read-after-write dependency between passes（如GM必须在pass1全部完成后才能用于pass2）导致无法fusion跨pass，live footprint为O(M)。**2-pass cascade**（TileFlow/Choi et al.）：先partition input→comput per-partition local max→form local numerator/denominator→second pass用global max correction修正。**1-pass cascade**（FlashAttention/FlashAttention-2/Rabe and Staats）：用running max（而非global max）迭代式building结果，每iteration处理M chunk后即时update running max/numerator/denominator，correction factor e^{old_max-new_max} rescale旧值到新max。仅需1次M fiber遍历，live footprint为tile size O(M0×P0)，与总seq len无关。

从算法pipeline角度拆解术语：
以attention的1-pass cascade（Cascade 5, FlashAttention-2 variant）为例，对比3-pass：
```
3-PASS (Cascade 4):
  # Pass 1 — 遍历所有M fiber element k:
  QK_{m,p} = Σ_e Q_{e,p} × K_{e,m}        // QK matmul
  GM_p = max_m(QK_{m,p})                   // global max (需要读完所有M)
  # Pass 2 — 重新遍历M fiber:
  SN_{m,p} = e^{QK_{m,p} - GM_p}          // numerator
  SD_p = Σ_m SN_{m,p}                      // denominator
  # Pass 3 — 再次遍历M fiber:
  A_{m,p} = SN_{m,p} / SD_p               // softmax output
  AV_{f,p} = Σ_m A_{m,p} × V_{f,m}        // attention output

1-PASS (Cascade 5, FuseMax):
  for m1 in 0..M1-1:  # M = M1 × M0, 单次遍历
    BQK_{m1,m0,p} = Q_{e,p} × BK_{e,m1,m0}       # tile QK
    LM_{m1,p} = max_{m0}(BQK_{m1,m0,p})           # local max
    RM_{m1+1,p} = max(RM_{m1,p}, LM_{m1,p})      # running max update
    SLN_{m1,m0,p} = e^{BQK_{m1,m0,p} - RM_{m1+1,p}}  # local numerator
    SLD_{m1,p} = Σ_{m0} SLN_{m1,m0,p}             # local denominator
    SLNV_{f,m1,p} = Σ_{m0} SLN_{m1,m0,p} × BV_{f,m1,m0}  # local num×V
    PRM_{m1,p} = e^{RM_{m1,p} - RM_{m1+1,p}}      # correction factor
    SPD_{m1,p} = RD_{m1,p} × PRM_{m1,p}           # rescale old denom
    RD_{m1+1,p} = SLD_{m1,p} + SPD_{m1,p}         # running denom update
    SPNV_{f,m1,p} = RNV_{f,m1,p} × PRM_{m1,p}     # rescale old num×V
    RNV_{f,m1+1,p} = SLNV_{f,m1,p} + SPNV_{f,m1,p} # running num×V update
  AV_{f,p} = RNV_{f,M1,p} / RD_{M1,p}             # final division
```
关键区别：1-pass每m1 iteration同时执行所有Einsum（BQK→LM→RM→SLN→SLD→SLNV→PRM→...），而3-pass必须等某一类Einsum全部完成（如所有BQK+GM）才能start下一类。1-pass代价：correction factor PRM和rescaling增加额外compute（exp+multiply-add），但消除了O(M) live footprint和memory spill。

术语一般如何实现？如何使用？
Pass分类用于分析attention accelerator设计中on-chip buffer sizing和memory traffic trade-off。3-pass在seq len短时可充分on-chip buffer → 无spill；seq len超过buffer capacity → 被迫spill → memory-bandwidth bound（如FLAT在seq len≥256K时性能骤降）。1-pass的live footprint仅为tile size（M0×P0 on-chip），与总seq len无关 → 无spill → 始终保持compute-bound。FuseMax选择1-pass cascade作为mapping基础，通过deep fusion + fine-grain pipeline binding在spatial architecture上实现~100% PE utilization。Pass lower bound是mapping independent的——给定cascade的Einsum dependencies决定了minimum pass count，任何mapping/scheduling都无法突破此下限。

涉及论文标题：
- 5-FuseMax_Leveraging_Extended_Einsums_to_Optimize_Attention_Accelerator_Design.pdf

---

## Division Reduction for Attention (softmax后置除法优化)

术语解释
Division Reduction是attention中将softmax除法和后续matmul乘法的执行顺序重构以大幅减少division操作次数的优化。原本Am,p = SNm,p/SDp需M×P次division，之后AVf,p = Am,p × Vf,m。重构为：先SNVf,p = SNm,p × Vf,m（reduce across M），再AVf,p = SNVf,p / SDp（仅F×P次division）。因为M（seq len） >> F（embedding dim），division减少约M/F倍。

术语是什么？
Attention的标准形式：AV = softmax(QK^T/√d) × V。softmax输出Am,p（shape M×P）需对每个(m,p)做division by SDp → M×P次division。Division Reduction观察到：division和后续V乘法可交换——SNVf,p = Σ_m SNm,p × Vf,m（先multiply+reduce across M），然后一次性AVf,p = SNVf,p / SDp（每个(f,p)只做1次division）。正确性保证：因Am,p = SNm,p/SDp，Σ_m Am,p × Vf,m = Σ_m (SNm,p/SDp) × Vf,m = (Σ_m SNm,p × Vf,m) / SDp（SDp独立于m，可提出summation）。注意：此优化与numerical stability兼容——SNm,p需先经e^{QKm,p - GMp}稳定化。

从算法pipeline角度拆解术语：
```
# 标准（无Division Reduction）:
SN_{m,p} = e^{QK_{m,p} - GM_p}              # M×P exp
SD_p = Σ_m SN_{m,p}                          # M×P次加法 (reduction)
A_{m,p} = SN_{m,p} / SD_p                   # M×P次division
AV_{f,p} = Σ_m A_{m,p} × V_{f,m}            # M×F×P次multiply + M×P reduction

# Division Reduction:
SN_{m,p} = e^{QK_{m,p} - GM_p}              # 同上
SNV_{f,p} = Σ_m SN_{m,p} × V_{f,m}          # M×F×P multiply + reduction (新增)
AV_{f,p} = SNV_{f,p} / SD_p                 # 仅F×P次division (大幅减少)
```
效果：M=256K, F=64 → 标准需256K×P次division，Division Reduction仅64×P次 → division减少4000×。注意SNV引入额外F×M×P次multiply操作，但因multiply远快于division（在硬件中multiply ~1 cycle vs division ~10s of cycles），且memory bandwidth-limited workload可trade compute for memory。

术语一般如何实现？如何使用？
此优化最初在FlashAttention-2中使用，FuseMax指出它可独立于pass数应用于任意cascade（3-pass/2-pass/1-pass均可受益）。在FuseMax中，Division Reduction被编码为Cascade 5的Einsums 49+55：先用Einsum 49 `SLNV_{f,m1,p} = Σ_{m0} SLN_{m1,m0,p} × BV_{f,m1,m0}` 做local num×V（在2D array），最后用Einsum 55 `AV_{f,p} = RNV_{f,M1,p} / RD_{M1,p}` 做final division（在1D array）。在GPU实现中，FlashAttention-2将此优化与tiling+recomputation结合。在FPGA/ASIC中，FP除法器面积大（~10× of FP multiplier），减少division直接减少area/power。Xia et al. [59]的low-latency configurable precision FP divider被FuseMax用于energy modeling（缩放到45nm）。

涉及论文标题：
- 5-FuseMax_Leveraging_Extended_Einsums_to_Optimize_Attention_Accelerator_Design.pdf

---

## Fibertree (Tensor Representation for Accelerator Analysis)

术语解释
Fibertree是一种format-agnostic的tensor表示方式，将tensor表示为fiber的树形结构，是Einsum和cascade of Einsums分析的基础数据抽象。

术语是什么？
在fibertree抽象中，tensor被表示为一棵fiber（纤维）树：每个rank对应树的一层，fiber由该rank下所有共享高层coordinates的coordinates集合组成，每个coordinate耦合一个payload。Payload可以是：(1) 指向下一层rank的fiber的reference；(2) leaf data value（对于最底层rank）。一个N-tensor对应一棵N层fibertree。例如，2-tensor（matrix）A_{k,m}的fibertree：顶层K-fiber包含所有k coordinates→每个k payload指向一个M-fiber→M-fiber包含所有m coordinates→每个m payload为leaf data value A[k][m]。**is-ﬁbertree**（iteration space ﬁbertree）：将Einsum的iteration space也表示为fibertree。例如GEMM的iteration space为K×M×N的3-tensor。Cascade of Einsums的pass分析基于比较不同Einsum的is-ﬁbertree中同一tensor的同一fiber是否存在read-after-write dependency。

从算法pipeline角度拆解术语：
以attention中QK matmul (Einsum 22: QK_{m,p} = Q_{e,p} × K_{e,m}) 的fibertree为例：
```
Q tensor (E×P):           K tensor (E×M):           QK tensor (M×P):
  E-fiber:                   E-fiber:                   M-fiber:
    e0 → P-fiber               e0 → M-fiber               m0 → P-fiber
      p0 → Q[e0][p0]             m0 → K[e0][m0]             p0 → QK[m0][p0]
      p1 → Q[e0][p1]             m1 → K[e0][m1]             p1 → QK[m0][p1]
    e1 → P-fiber               ...                        ...
      ...                    ...                        m1 → P-fiber
                                                          ...
```
QK Einsum的iteration space is-ﬁbertree有shape E×M×P。每个(e,m,p)点project到Q的(E,P) space取Q[e][p]、到K的(E,M) space取K[e][m]、乘法结果写入QK的(M,P) space QK[m][p]处（需reduce across E）。Fibertree的format-agnostic性质意味着：无论tensor存储为dense/CSR/COO/CSF等format，fibertree的逻辑结构相同，分析（如live footprint计算）成立。

术语一般如何实现？如何使用？
Fibertree最初在TACO、TeAAL等sparse tensor algebra framework中提出。TeAAL [35]使用fibertree作为sparse tensor accelerator modeling的核心抽象——将accelerator的data access pattern映射为fibertree的traversal。FuseMax使用fibertree进行pass分析：比较不同Einsum的is-ﬁbertree中对同一tensor的同一rank fiber的访问模式→检测read-after-write dependencies→推导minimum pass count。Fibertree提供mapping-independent的kernel特性分析——给定cascade，任何mapping/scheduling都无法突破fibertree dependency导出的pass下限和live footprint下限。类似概念：TACO的iteration graph、SparseLoop的fibertree-based analytical model。

涉及论文标题：
- 5-FuseMax_Leveraging_Extended_Einsums_to_Optimize_Attention_Accelerator_Design.pdf

---

## Running Maximum and Correction Factor in 1-Pass Softmax

术语解释
Running Maximum（运行最大值）是1-pass attention中替代Global Maximum的迭代统计量，配合Correction Factor（修正因子）实现数值稳定的逐chunk softmax计算，无需等待整个M fiber完成。

术语是什么？
在标准numerically stable softmax中，先计算global maximum GM_p = max_m(QK_{m,p})，再用它归一化所有QK值：SN_{m,p} = e^{QK_{m,p} - GM_p}。但这要求已知所有M fiber的QK值→必须完成pass1才能start pass2。Running maximum替代为：每处理一个M chunk m1→计算local max LM_{m1,p}→update running max RM_{m1+1,p} = max(RM_{m1,p}, LM_{m1,p})→立即用当前RM计算local numerator/denominator。但问题：之前已累积的denominator/numerator（用旧RM计算）与新RM不匹配→需correction factor PRM_{m1,p} = e^{RM_{m1,p} - RM_{m1+1,p}} rescale旧accumulator。rescale后新旧值在同一scale下→可安全相加。数学上等价于global max方法（因最终所有值都除以同一e^{RM_{M1,p}}，而RM_{M1,p}=GM_p）。

从算法pipeline角度拆解术语：
```
# 每iteration m1:
LM = max(BQK[m1, :, :])                    # local max of current chunk
RM_new = max(RM_old, LM)                   # update running max
# 用新的RM计算当前chunk的numerator/denominator
SLN = exp(BQK[m1, :, :] - RM_new)         # local numerator
SLD = sum(SLN)                              # local denominator
# Correction: 将旧running值rescale到新RM
PRM = exp(RM_old - RM_new)                 # correction factor ∈ (0, 1]
SPD = RD_old × PRM                         # rescale旧denominator (downscale)
RD_new = SLD + SPD                         # 新running denominator (同scale相加)
# 同理对numerator×V做correction
SPNV = RNV_old × PRM                       # rescale旧num×V
RNV_new = SLNV + SPNV                      # 新running num×V

# 迭代结束后RM_{M1} = global max, RD_{M1}和RNV_{M1}都除以同一e^{RM_{M1}} → 等价于global max softmax
AV = RNV_{M1} / RD_{M1}
```
额外compute开销：(1) per-iteration PRM = exp(RM_old - RM_new) — 1次subtract+1次exp；(2) SPD = RD × PRM — 1次multiply；(3) SPNV = RNV × PRM — F次multiply。总共per-chunk overhead: 1 exp + (1+F) mult + (1+F) add。但消除了：(a) 存储/加载完整QK tensor的O(M×P) memory traffic；(b) 等待global max完成的pipeline stall。

术语一般如何实现？如何使用？
Running max/correction是FlashAttention系列（FlashAttention [15], FlashAttention-2 [14]）中1-pass attention的核心机制，在GPU上通过online softmax (tiling + rescaling)实现。FuseMax将其形式化为Cascade 5的Einsums 46、50-54，并在spatial architecture上实现为fine-grain pipelined binding：2D array计算BQK和SLN（含exp(BQK-RM)），1D array计算RM、SLD、PRM、SPD、RD、SPNV、RNV。因correction涉及1D array的multiply-add（而非MAC），1D array的PE需支持MAC+exp或dedicated multiply unit。类似概念见于Rabe and Staats [47]的O(1) memory self-attention、Dao et al.的online softmax。

涉及论文标题：
- 5-FuseMax_Leveraging_Extended_Einsums_to_Optimize_Attention_Accelerator_Design.pdf

## Multi-Task Multi-Modal (MT MM) Model Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Task Multi-Modal (MT MM) Model Training是一种训练范式，在统一的foundation model框架下同时处理多个任务（如text classification、image captioning、speech recognition）和多种数据模态（如vision、audio、text、motion、thermal、depth）。MT MM模型采用sub-model sharing架构（图1上部）：shared base model（如Transformer layers）包含common knowledge被不同task/modality共享，各modality有独立的encoder（如vision encoder、audio encoder），各task可激活不同的modality encoder组合和task-specific head。每个training iteration中，包含多种modality的数据同时输入模型，不同model components被选择性激活和更新（如speech recognition task激活audio encoder+text encoder+shared LM，image captioning task激活vision encoder+text encoder+shared LM）。OFASys [9]、Flamingo [4]、AnyMAL [46]等采用此范式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MT MM训练的一个iteration（以Multitask-CLIP 4 tasks, 6 modalities为例）：
```
// Model components
encoders = {vision_enc, audio_enc, text_enc, motion_enc, thermal_enc, depth_enc}
shared_lm = TransformerLM()  // shared cross-modal module
task_heads = {task1_head, task2_head, task3_head, task4_head}

// Data: mixed-modality batch
batch = {
  task1: {vision: img_batch1, text: text_batch1},
  task2: {audio: audio_batch2, text: text_batch2},
  task3: {vision: img_batch3, motion: mot_batch3},
  task4: {thermal: thm_batch4, depth: depth_batch4}
}

// Forward pass (conceptual - actual execution order optimized by Spindle)
for task, data in batch:
  // 1. Modality encoding
  modal_features = []
  for modality in task.modalities:
    feat = encoders[modality](data[modality])
    modal_features.append(feat)
  
  // 2. Feature fusion (e.g., concatenation or cross-attention)
  fused = concat(modal_features)
  
  // 3. Shared cross-modal processing
  shared_output = shared_lm(fused)
  
  // 4. Task-specific head
  task_output = task_heads[task](shared_output)
  loss = compute_loss(task_output, labels[task])

// Backward: shared parameters accumulate gradients from ALL tasks
// Spindle handles this via Parameter Device Group Pool
```

MT MM训练面临的核心挑战：(1) workload heterogeneity——不同modality encoder（如Audio encoder轻量vs Vision encoder重量）和不同task组合的workload差异大；(2) execution dependency——shared components被多个task激活导致执行barrier；(3) dynamic modality proportion——不同task中modality比例随时间变化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MT MM训练在实践中通常使用PyTorch分布式训练。Spindle之前的方法（Megatron-LM、DeepSpeed）将MT MM模型的各sub-model在时间维度解耦串行训练，导致严重的资源浪费。Spindle通过wavefront scheduling实现operator-level交错执行，保持高GPU利用率。用户通过SpindleTask API定义MT MM训练任务（可自定义PyTorch modules并通过add_flow连接，或由FX Tracer从统一model自动拆分），Spindle自动完成graph contraction、profiling、resource allocation和wavefront scheduling。该训练范式适用于需要多模态理解和多任务泛化的场景（如通用AI助手、多模态搜索、自动驾驶感知）。

涉及论文标题：
- 63-Spindle- Efficient Distributed Training of Multi-Task Large Models via Wavefront Scheduling.pdf

## Variable-Length Encoding for Quantized Parameters (面向量化参数的变长编码)

术语解释
Variable-Length Encoding（变长编码）是SPARK提出的一种对已INT8量化的DNN参数进行bit-level自适应精度编码的方案。利用量化值高位bit（MSB）的天然稀疏性，通过1-bit标识符嵌入数据本身来区分低精度（4-bit）和高精度（8-bit）编码，以固定长度（4-bit对齐）存储变长数据，实现对齐memory access下的混合精度压缩。SPARK编码可离线执行（weight）或在线硬件执行（activation），与pruning等压缩方法正交。

术语是什么？
Variable-Length Encoding for Quantized Parameters的核心思想：DNN模型经INT8量化后，约80%的参数值落在[0,7]区间（仅需4-bit有效表示），约20%落在[8,255]区间（需8-bit）。传统方法使用uniform 8-bit存储所有值导致大量高位bit浪费（高位稀疏）。SPARK的变长编码通过以下规则压缩：
- 使用最高位(c0)作为1-bit identifier：c0=0→低精度4-bit（值域[0,7]），c0=1→高精度8-bit（值域[8,255]）
- 低精度值占1个4-bit存储slot，高精度值占2个连续4-bit slots
- Accuracy Compensation Mechanism：对需舍入的值通过b0⊕b3 XOR check决定舍入方向（上舍入或保留），error≤16（对[0,255]范围）
- 95%+值可无损编码，整体accuracy loss <0.1%（CNN），对attention模型可能反而提升准确率

与传统coordinate list方法（OLAccel/GOBO需额外存储index标记outlier位置）相比，SPARK的变长编码：
- 不需要额外的coordinate list或index存储
- 内存访问完全对齐（所有code以4-bit基本长度对齐）
- Decoder仅需MUX+OR+NOT gates（极简硬件实现）
- 支持intra-tensor mixed-precision而非per-layer或per-channel

从算法pipeline角度拆解：
SPARK变长编码的完整encoding流程（以8-bit unsigned INT8 weight为例）：

```
# Input: W ∈ [0,255] (INT8量化后的weight/activation张量)
# Step 1: Per-element SPARK encoding
for each element x in W:
    b = binary(x, 8)  # {b0, b1, b2, b3, b4, b5, b6, b7}, MSB first
    
    if x in [0, 7]:                           # Case 1: Low precision
        # Short code: 4-bit, identifier c0=0
        code = {0, b5, b6, b7}                # Store last 3 data bits
        # No information loss (lossless)
        
    elif x in [8, 127]:                        # Case 2: High precision, 7 valid bits
        c0 = 1                                 # Identifier = high precision
        if (b0 XOR b3) == 0:                   # No rounding needed
            code = {1, b1, b2, b3, b4, b5, b6, b7}  # 8-bit, lossless
        else:                                   # Round up
            # Set b3→0, b4..b7→1111 (maximize lower bits)
            code = {1, b1, b2, 0, 1, 1, 1, 1}  # 8-bit, lossy (error ≤ 16)
            
    else:  # x in [128, 255]                   # Case 3: High precision, 8 valid bits
        # identifier c0 is also a valid bit
        if b3 == 1:                             # No rounding needed
            code = {b0, b1, b2, b3, b4, b5, b6, b7}  # 8-bit, lossless
        else:                                   # Round
            # Set b3→1, b4..b7→0000 (minimize lower bits)
            code = {b0, b1, b2, 1, 0, 0, 0, 0}  # 8-bit, lossy

# Step 2: Aligned storage
# Low-precision codes: 1 × 4-bit slot
# High-precision codes: 2 consecutive × 4-bit slots
# All memory accesses are 4-bit aligned

# Step 3: Decoding (hardware side, see hardware/kernel entries)
# Read 4 bits + enable signal per cycle
# if c0 == 0: output 4-bit low-precision value directly
# if c0 == 1 and EN == 0: output based on c3 (3 or 4 bits)
# if c0 == 1 and EN == 1: output 4-bit as high-precision post part
```

术语一般如何实现？如何使用？
SPARK变长编码的实现：
1. **Software侧**（PyTorch）：对预训练的INT8量化模型，offline执行weight的SPARK encoding；activation encoding通过模拟SPARK decoder行为来评估accuracy。
2. **Hardware侧**（Verilog RTL）：Encoder用5-bit Leading Zero Detector (LZD) + MUX + XOR gate实现，Decoder用MUX + OR + NOT gates实现。28nm TSMC下encoder/decoder面积仅占core的~0.5%。
3. 编码与pruning（如DBB 50% sparsity）正交叠加，实现联合优化（joint optimization）。
4. 无需finetuning——Accuracy Compensation Mechanism通过舍入策略最小化误差，但可选finetuning进一步提升accuracy。
5. 适用于CNN-based（VGG/ResNet）和attention-based（ViT/BERT/GPT-2）模型。

涉及论文标题：
- 68-SPARK_Scalable_and_Precision-Aware_Acceleration_of_Neural_Networks_via_Efficient_Encoding.pdf

## Intra-Value Bit Sparsity (值内比特稀疏性)

术语解释
Intra-Value Bit Sparsity（值内比特稀疏性）是SPARK论文揭示的一种量化值内部的bit-level冗余现象：经过INT8量化后，DNN参数的MSB（高位bit）普遍为零——约80%的值落在[0,7]区间，其b0..b4（高位5-bit）全为0。这种每个值内部的bit-level稀疏性（区别于value-level sparsity/pruning）提供了压缩机会：高位bit不携带有效信息，可直接丢弃，用更少的bit表示大部分值。

术语是什么？
Intra-Value Bit Sparsity描述量化后单个数值内部高位bit的稀疏特征。与value-level sparsity（pruning整值为零）和inter-value bit sparsity（如BBS的bit column sparsity）不同，intra-value bit sparsity关注单个quantized value的bit representation中使用不到的leading bits。SPARK发现：
- INT8量化后，[0,7]区间值的前5-bit（b0..b4）全为零，仅最后3-bit（b5..b7）有效
- 这80%的值实际只需3-4 bit表示（加上1-bit identifier）
- 这不同于coordinate list-based outlier方法（OLAccel）需要额外存储outlier位置

这一观察推动SPARK设计variable-length encoding：用1-bit identifier替换无效高位bit的信息量，实现4-bit对齐存储。

从算法pipeline角度拆解：
Intra-Value Bit Sparsity在SPARK中的利用流程：

```
# 原始INT8量化参数分布分析
# 步骤1：统计量化后值的分布
histogram = count_frequency(quantized_weights, bins=[0,7], [8,127], [128,255])
# 观测结果（以ResNet-50/BERT为例）：
#   [0,7]: ~80%   → 仅需3有效bit + 1 identifier = 4 bit
#   [8,127]: ~15% → 需7有效bit + 1 identifier = 8 bit
#   [128,255]: ~5% → 需8有效bit (identifier也是有效位) = 8 bit

# 步骤2：确定编码效率
avail_bit = 8           # INT8总bit数
effective_bits_most = 3 # [0,7]区间的有效bit数
wasted_bits = avail_bit - effective_bits_most  # 5-bit浪费

# 步骤3：SPARK encoding回收wasted bits
# 浪费的5-bit → 替换为1-bit identifier (c0)
# 剩余4-bit → 存储有效数据 (b5,b6,b7) 或对齐的8-bit高精度code
# 结果：80%值以4-bit存储，节省50%存储和传输带宽
```

术语一般如何实现？如何使用？
Intra-Value Bit Sparsity的利用方式：
1. **SPARK encoding**：利用高位bit稀疏设计1-bit identifier方案，减少存储和传输bit-width。
2. **正交性**：与value-level压缩（pruning/sparsity）正交——SPARK编码后可叠加DBB 50% sparsity，进一步减少cycle count。
3. **通用性**：该特征在CNN和attention-based模型中都存在（attention模型甚至更明显，因为其参数分布方差更大），使SPARK对两类模型都有效。
4. 与NVIDIA Ampere GPU的compute data compression（压缩zero值和相似bytes）互补——Ampere在DRAM/L2 cache压缩，SPARK在数据表示层压缩比特冗余。

涉及论文标题：
- 68-SPARK_Scalable_and_Precision-Aware_Acceleration_of_Neural_Networks_via_Efficient_Encoding.pdf

## Bit-Slice (BS) Decomposition for LLM Weights

术语是什么？
Bit-Slice (BS) Decomposition是将LLM的k-bit整数量化权重矩阵分解为k个独立的1-bit二值矩阵（每个对应一个bit位）的技术。对于INT8权重矩阵W∈R^{H×H}，BS分解产生8个bit-slice矩阵：W_bs[0]=W&1 (LSB), W_bs[1]=(W>>1)&1, ..., W_bs[7]=(W>>7)&1 (MSB/sign bit)。分解后的bit-slice矩阵揭示两个关键特征：(1) BS-Sparsity——高bit位（接近MSB）的bit-slice矩阵中零bit比例远高于低bit位（因量化权重呈Gaussian-like分布，大部分值落在小数值区间，高位bit为零），例如Llama13B中第7th BS矩阵sparsity ratio可达95%；(2) BS-Repetitiveness——bit-slice矩阵中存在大量重复的列向量，因为bit列向量只有2^m种可能（m为group size），而LLM的hidden dimension H（4k-12k）远大于2^m。BS分解保持了完整的计算等价性：k-bit GEMV等价于k个1-bit bit-slice矩阵的shift-and-accumulate（第b位左移b位后求和），因此可以在bit-level利用sparsity和repetitiveness加速计算而不损失精度。

从算法pipeline角度拆解术语：
BS分解在LLM推理pipeline中的具体流程（以INT8 Llama7B GEMV为例）：
```
# Input: INT8 weight W∈R^{H×H}, INT8 activation X∈R^{H}
# Output: INT8 result Y∈R^{H}

# Step 1: Bit-Slice Decomposition
for b in 0..7:
    W_bs[b][i][j] = (W[i][j] >> b) & 1  # Extract b-th bit

# Step 2: Per-bit-slice independent computation
for b in 0..7:
    # Each W_bs[b] is a binary matrix ∈ {0,1}^{H×H}
    # GEMV: Y_bs[b][i] = Σ_j W_bs[b][i][j] * X[j]
    # (Can be accelerated via sparsity or repetition)
    Y_bs[b] = bit_slice_gemv(W_bs[b], X)

# Step 3: Shift-and-accumulate to reconstruct INT8 result
Y = Σ_{b=0}^{7} (Y_bs[b] << b)
```
BS分解暴露了value-level representation无法观察的细粒度优化机会。在value-level，一个2-bit的零值需要两个bit位同时为零（概率低）；在bit-level，高位bit为零的概率独立且更高（Llama13B中bit sparsity平均是value sparsity的10.1倍）。

术语一般如何实现？如何使用？
MCBP对INT8 weight离线执行BS分解→仅高bit位（3rd-7th BS矩阵，SR>65%）进行BSTC压缩→低bit位（1st/2nd/8th）无压缩直接存储。在线推理时，BSTC decoder解压高bit位BS矩阵→BRCR unit对每个BS矩阵独立执行group-wise GEMM→结果经shift-accumulate和quantize scale/bias恢复INT8输出。与value-level processing的关键区别：BS分解使数据存储和计算自然对齐bit维度，消除了value-to-bit reorder开销（在FuseKNA/BitWave中该开销占energy的18-30%）。

涉及论文标题：
- 6-MCBP- A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness.pdf

## BS-Repetitiveness-enabled Computation Reduction (BRCR)

术语是什么？
BRCR是MCBP提出的利用bit-slice (BS)向量间重复性消除GEMM冗余计算的算法。核心思想：将BS矩阵按group size m行分组为Group Matrix G∈R^{m×H}（H为hidden dimension），G的每一列为m-bit向量→至多2^m种可能类型（pigeonhole principle）。当H>>2^m时，大量列为重复项→通过合并重复列对应的activation，将GEMV从H次乘加降为至多H×(1-bs̃)+m×2^{m-1}次加法。BRCR分两步：(1) Merge：遍历G的H列，将每列的非零索引对应的activation累加到Merged Activation Vector (MAV, 长度2^m)；(2) Reconstruct：用Enumeration Matrix（记录每行包含哪些列类型）乘以MAV恢复GEMV结果。

从算法pipeline角度拆解术语：
以m=4, H=4096, k=8为例的BRCR GEMM：
```
# 输入: k个BS矩阵W_bs[k]∈R^{H×H}, activation X∈R^{H}
# 参数: group size m=4

for each BS matrix b in 0..7:
    for row_group_start in 0..H step m:  # 每次处理m=4行
        G = W_bs[b][row_group_start:row_group_start+m][:]  # R^{4×H}
        
        # Step 1: Merge (最多H×(1-bs̃) adds)
        MAV = zeros(2^m)  # length 16
        for j in 0..H-1:
            col_val = (G[0][j]<<0 | G[1][j]<<1 | G[2][j]<<2 | G[3][j]<<3)  # 0..15
            if col_val != 0:
                MAV[col_val] += X[j]
        
        # Step 2: Reconstruct (至多 m×2^{m-1} adds)
        for i in 0..m-1:
            Y_bs[b][row_group_start+i] = 0
            for each col_val where Enumeration[i][col_val] == 1:
                Y_bs[b][row_group_start+i] += MAV[col_val]

# Bit-level accumulation
Y = Σ_{b=0}^{7} Y_bs[b] << b
```
BRCR的加法总量为k(H×(1-bs̃)+m×2^{m-1})，在bs̃≈0.70, vs̃≈0.07下相比value sparsity减少12.1×计算量。

术语一般如何实现？如何使用？
BRCR需要硬件支持：CAM (Content Addressable Memory)在单周期内识别G的列值；Index Converter将CAM bitmap翻译为activation地址；AMU (Addition Merge Unit)合并重复activation。MCBP选定m=4作为sweet spot（m过小→重复不足，m过大→reconstruct开销2^{m-1}指数增长→抵消merge收益）。BRCR为lossless优化（利用固有数据冗余，不损失精度），对prefill阶段（GEMM-dominated）效果最显著（Dolly 1k prompt, 3.9× latency reduction）。

涉及论文标题：
- 6-MCBP- A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness.pdf

## BS-Sparsity-enabled Two-State Coding (BSTC)

术语是什么？
BSTC是MCBP提出的沿bit-slice维度对LLM权重进行无损压缩的编码方案。核心观察：量化权重的bit-slice矩阵中，高bit位（如第7th BS, 第6th BS）具有极高sparsity（SR>65%，甚至>95%），且零bit以列向量的形式co-occurring（一个m-bit列向量全为零的概率随m减小而增大）。BSTC采用two-state编码：全零列向量→1'b0（1 bit）；非零列向量→{1'b1, m-bit原始数据}（m+1 bits）。编码以与BRCR相同的group size m=4独立应用于每个BS矩阵，避免bit-reorder开销。BSTC为正收益的条件：SR>65%（因编码为每个非零列增加1-bit indicator开销）。

从算法pipeline角度拆解术语：
BSTC的离线编码和在线解码流程：
```
# Offline encoding (per BS matrix, m=4)
for each BS matrix b in {3,4,5,6,7}:  # 仅SR>65%的高bit位
    for col_group in 0..(H/m)-1:
        v = W_bs[b][col_group*m : col_group*m+m]  # m-bit column vector
        if v == all_zeros:
            encoded.append(1'b0)
        else:
            encoded.append({1'b1, v})  # m+1 bits

# BS matrices 1st, 2nd, 8th: no compression (SR too low)

# Online decoding
for each encoded token:
    if token[0] == 0:  # all-zero column
        output = [0,0,0,0]  # m zeros
    else:  # non-zero column
        output = token[1:m+1]  # extract original m-bit data
```
BSTC实现lossless compression，没有accuracy loss。编码仅应用于sign-magnitude格式下SR>65%的BS矩阵。

术语一般如何实现？如何使用？
MCBP使用sign-magnitude (SM) weight格式以最大化BS sparsity（SM格式使符号位为独立MSB，其余bit更sparse）。BSTC CODEC硬件轻量：Encoder含4-bit CMP+MUX，Decoder含1-bit CMP+5-bit SIPO+leading one eliminator。BSTC与BRCR共享group size m=4以对齐编解码和计算粒度。在Llama7B/13B等模型上，BSTC平均减少weight memory access 75.8%。

涉及论文标题：
- 6-MCBP- A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness.pdf

## Bit-Grained Progressive Prediction (BGPP)

术语是什么？
BGPP是MCBP提出的bit-grained progressive top-k attention sparsity预测机制，用于减少LLM解码阶段的KV cache访问。传统value-level top-k使用4-bit MSB估计attention matrix→sort选top-k→full-precision计算被选KV。BGPP改进：(1) Progressive——多轮过滤，每轮只加载Keys的当前1 bit（从MSB到LSB）计算partial QK估计；(2) Bit-grained——使用bit-serial inner product逐位计算attention；(3) Early termination——当partial QK结果超出feasible top-k范围时提前终止，避免加载后续bits。每轮阈值θ_i^r = max(Â_i^r) - α_r×radius（radius默认3，α_r∈[0,1]控制aggressiveness）。

从算法pipeline角度拆解术语：
以单个Query、6个Keys、2轮过滤为例：
```
# 输入: Query Q (4-bit MSB), Keys K[0:5] (INT8)
# 参数: radius=3, α_1=α_2=0.5, rounds=2

# Round 1: Load 1st bit (MSB) of all 6 Keys
K1_msb = [K[0][7], K[1][7], ..., K[5][7]]  # 仅1 bit per Key
Â_1 = bit_serial_dot(Q, K1_msb)  # 1-bit × 4-bit, 6 results
threshold_1 = max(Â_1) - 0.5 * 3  # α_1×radius
selected = where(Â_1 > threshold_1)  # e.g., indices [1,3,5]

# Round 2: Load 2nd bit only for selected Keys
K2_msb = [K[1][6], K[3][6], K[5][6]]  # 仅加载3个Keys的1 bit
Â_2 = Â_1[selected] + bit_serial_dot(Q, K2_msb) << 1  # 累加
threshold_2 = max(Â_2) - 0.5 * 3
selected = where(Â_2 > threshold_2)

# Early termination: 如果某partial结果已≤threshold→后续bits不必加载
# Final: 仅对被选Keys做full-precision attention (8-bit QK + softmax + 8-bit PV)
```
BGPP减少KV cache access的原因是：high bit位的partial估计已足以判断大量KVs不属于top-k→它们的剩余bits无需从HBM加载。

术语一般如何实现？如何使用？
BGPP实现需bit-serial inner product units (64-input AND-based adder tree)、Progressive Filter (Threshold Updating + Clipping modules)、Clock-gated design (threshold<min时自动跳入下一轮省电)。α_r∈[0.5, 0.6]平衡准确率和sparsity：α_r越小→更aggressive pruning→更高sparsity但更低accuracy。BGPP对长decoding任务（MBPP 4k decode）效果最显著（2.1× latency reduction）。

涉及论文标题：
- 6-MCBP- A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness.pdf

## Sign-Magnitude (SM) Weight Format for Bit-Slice Processing

术语是什么？
Sign-Magnitude (SM) format是一种将有符号整数表示为1-bit符号位（sign）和(k-1)-bit magnitude（绝对值）的数值格式。与2's complement格式不同，SM中符号位独立于数值位，使得MSB位（符号位）与其他bit位的sparsity特性分离。MCBP采用SM格式存储INT8权重的原因：在2's complement下，MSB既是符号又参与数值→MSB bit-slice矩阵中零bit比例被污染；在SM下，符号位独立为第8个BS矩阵，数值位（1st-7th BS）的sparsity按magnitude自然分布——小数值的magnitude高bit位为零，sparsity随bit位升高而增加（第3rd-7th BS的SR均>65%），BSTC可有效压缩这些高sparsity BS矩阵。

从算法pipeline角度拆解术语：
```
# 2's complement INT8 → SM INT8 转换（无精度损失）
def to_sm(x_int8):
    if x >= 0:
        return (0 << 7) | x  # sign=0(positive), mag=x
    else:
        return (1 << 7) | (-x)  # sign=1(negative), mag=|x|

# SM INT8的BIT分解
# bit 7 (8th): sign bit (0=positive, 1=negative)
# bits 6-0 (7th-1st): magnitude bits

# 计算时: 对sign bit的bit-slice矩阵需negate处理
# BGPP中通过Sign Decision Unit (SDU)在bit-serial adder tree前处理符号
```

术语一般如何实现？如何使用？
SM格式在bit-serial computing中广泛使用（如BitWave也采用SM），因它使bit-slice矩阵的sparsity更可预测。MCBP利用SM格式：(1) BSTC仅压缩3rd-7th BS（magnitude的高bit位），1st/2nd BS和8th BS（sign bit）无压缩存储；(2) BGPP在bit-serial IP unit前增加SDU处理符号位。MCBP的hardware支持SM到2's complement的隐式转换（在accumulation阶段）。

涉及论文标题：
- 6-MCBP- A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness.pdf

## Grouped Bit-Slice (Group Matrix)

术语是什么？
Grouped Bit-Slice是将bit-slice矩阵的m行合并为一个Group Matrix G∈R^{m×H}的技术，是MCBP中BRCR和BSTC共享的核心组织粒度。G的每一列为m-bit向量（每位来自一个不同的原始weight行），因此有至多2^m种可能的列向量类型。Group size m的选择决定redundancy exploitation和reconstruction overhead的trade-off：m越小→2^m越小→列重复概率越高（pigeonhole principle, 因H>>2^m）→redundancy利用更充分→但每个BS矩阵需更多group→overhead累加；m越大→单个group覆盖更多行→但2^m指数增长→reconstruction开销（m×2^{m-1} adds）很快超过merge收益。

从算法pipeline角度拆解术语：
MCBP通过Design Space Exploration确定m=4为最优（Fig.18）：
```
# m=1: CR<1（indicator overhead超过compression gain）, CPR最低
# m=2-4: CPR和CR同时增长
# m=4: CR峰值（all-zero column最多），CPR接近峰值
# m=5: CPR达到峰值（更多行被merge）但CR开始下降
# m>5: CPR下降（reconstruction 2^{m-1} overhead主导），CR持续下降（更大group中all-zero column减少）
# 此外m=4是大多数Transformer hidden dimension的公约数（H=4096→4的倍数）
```
Grouped BS效应解释了为什么BS分解后redundancy远高于value-level：在value-level，两个H维列向量全等概率极低；在bit-level，m-bit列向量仅2^m种可能类型→随着m减小，pigeonhole principle保证重复无限增加。

术语一般如何实现？如何使用？
Grouped BS是BRCR和BSTC的基础：BRCR以group为单位做merge-and-reconstruct（CAM match size=m bits, MAV size=2^m=16）；BSTC以group为单位做two-state coding（编码单位是m-bit列向量）。MCBP accelerator的data layout沿group size维在HBM bank间interleave存储→确保一次burst读取完整group。m=4也使得CAM basic block为2-bit匹配粒度，可通过re-matching适应其他m值。

涉及论文标题：
- 6-MCBP- A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness.pdf

## Dynamic-Shape Neural Networks（动态形状神经网络）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dynamic-Shape Neural Networks是一类tensor operator的输入/输出tensor shape在编译时未知、仅在runtime才确定的神经网络。与传统的static-shape网络（operator shape编译时固定）不同，dynamic-shape网络因以下场景产生变化的tensor shape：(1) **Dynamic sequence length**：NLP模型（如BERT）处理不同长度的输入句子，导致GEMM operator的M/N/K维度随sequence length变化；(2) **Dynamic batch size**：训练中自适应调整batch size以平衡收敛速度和资源消耗；(3) **Dynamic image resolution**：CV模型（如Faster R-CNN）处理不同分辨率的输入图像，导致convolution operator的feature map尺寸变化。Dynamic-shape特性给tensor compiler带来新挑战：传统static-shape compiler（如TVM）要求编译时已知shape以进行auto-tuning，而dynamic场景中编译昂贵且不现实。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以BERT-base处理dynamic sequence length的pipeline为例：
```
// 输入：句子长度len（runtime确定，如len=84）
input_ids = tokenizer("The quick brown fox ...")  // shape: (batch, len)
// Embedding lookup → (batch, len, hidden_dim=768)
embeddings = embedding_table[input_ids]
// Multi-Head Self-Attention中的GEMM:
// Q = Linear_q(embeddings): (batch, len, 768) × (768, 768) → (batch, len, 768)
// 对应GEMM: M=batch×len, N=768, K=768
// 当len=84时M值变化→GEMM shape在runtime确定
for layer in range(12):
    Q = matmul(embeddings, W_q)  // dynamic-shape GEMM
    K = matmul(embeddings, W_k)
    V = matmul(embeddings, W_v)
    attn_output = softmax(Q @ K^T / sqrt(d_k)) @ V
    ffn_output = gelu(matmul(attn_output, W_ffn1)) @ W_ffn2
```
Dynamic-shape的挑战在于：每个operator的tiling策略（如GEMM的TM.0, TN.0, TK.0参数）对该shape可能是最优的，但对另一个shape（如len=512时）则可能因load imbalance或memory bounded而严重退化（A100上GEMM M=4096时262.2 TFLOPS vs M=105时仅22.3 TFLOPS）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Dynamic-shape网络的处理方式：(1) **Padding方法**：将所有输入pad到预定义最大长度，但浪费计算和内存。(2) **Static-shape compiler (TVM/XLA)**：在编译时为已知shape auto-tune，无法处理未知shape。(3) **Dynamic-shape compiler (DietCode/Nimble)**：使用shape-generic search space，在编译时按shape range预生成一组实现→runtime按实际shape选择，但range外shape报错。(4) **MikPoly方法**：two-stage micro-kernel polymerization → offline生成固定尺寸micro-kernels + online按需聚合→支持任意shape，无range限制。Dynamic-shape网络在LLM和CV中广泛使用，且随模型架构演进持续产生新的dynamic-shape需求。

涉及论文标题：
- 69-Optimizing Dynamic-Shape Neural Networks on Accelerators via On-the-Fly Micro-Kernel Polymerization.pdf

## Mixture-of-Experts (MoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mixture-of-Experts (MoE) 是一种稀疏激活的神经网络架构，核心思想是将传统 Transformer 的 Dense FFN 层替换为多个并行的 Expert FFN 层（每个 Expert 与原始 FFN 维度相同），通过 Gate Function 为每个输入 token 动态选择激活少量 Experts（通常 top-1 或 top-2），从而在 scaling 模型容量（增加 Expert 数量）的同时保持计算量 sub-linear 增长。MoE block 由 Gate Function（计算各 Expert 对当前 token 的激活概率 → softmax → top-k 选出 activated experts）和 Expert Layers（多个并行的 FFN，仅被选中的执行）组成。Google SwitchTransformer 激活 top-1 of 128 experts（仅 0.8% 参数被使用），Meta NLLB-MoE 激活 top-2 of 128 experts。MoE 的 compute efficiency 体现在：无论 Expert 数量如何增加，计算 FLOPs 保持恒定（仅 selected experts 参与计算），但模型总参数量随 Expert 数量线性增长。MoE 的主要挑战是：(a) 巨大内存占用（Expert 参数占总参数绝大部分，如 Switch-Large 105.6GB）；(b) 动态稀疏激活导致推理时难以预测哪些 Experts 被激活，多 GPU 部署时 compute utilization 低；(c) Expert selection 与 Expert execution 存在串行数据依赖。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MoE 推理 pipeline（传统 Gate Function）：
```
# Input: hidden_states shape (batch, seq_len, d_model)
# For each MoE block:
gate_logits = gate_layer(hidden_states)         # (batch, seq, num_experts)
gate_probs = softmax(gate_logits)
expert_indices = topk(gate_probs, k=top_k)      # 选中 expert ids
expert_weights = gate_probs[expert_indices]      # gate weight for selected experts

# 仅对选中 experts 执行 FFN（sparse computation）:
output = zeros_like(hidden_states)
for expert_id in unique(expert_indices):
    mask = (expert_indices == expert_id)         # tokens assigned to this expert
    tokens = hidden_states[mask]                 # gather assigned tokens
    expert_out = expert_ffn[expert_id](tokens)   # W1(gelu(W2(x))) expert FFN
    output[mask] += expert_weights[mask] * expert_out  # scale by gate weight
```
Dense LLM 的全连接 FFN 等价于 MoE 只有 1 个 Expert 的特例。MoE 的优势：128 Experts 时计算量 ≈ 1 个 Expert 的计算量（仅激活 top-1），但模型 capacity = 128×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MoE 模型通常先大规模预训练（pretraining on general corpus, thousands of GPUs for months），再针对下游任务 fine-tune。Google SwitchTransformer [8] 是最具代表性的 MoE 模型，开源 pretrained weights 在 HuggingFace。MoE 的 Gate Function 可用多种策略训练：BASE Layers [20] 使用 balanced assignment，Hash Layers [36] 使用 deterministic hashing，Expert Choice Routing [46] 让 experts 自己选 tokens。MoE 推理部署通常用多 GPU Expert Parallelism（experts 分布到多 GPU 内存），或用 CPU offloading（expert 参数放 CPU，按需迁移到 GPU）。DeepSpeed-MoE [30] 和 FasterTransformer [25] 支持 MoE 推理。常见 MoE 模型：SwitchTransformer、GLaM [7]、NLLB-MoE [41]、Mixtral 等。

涉及论文标题：
- 71-Pre-gated_MoE_An_Algorithm-System_Co-Design_for_Fast_and_Scalable_Mixture-of-Expert_Inference.pdf
- 81-Klotski- Efficient Mixture-of-Expert Inference via Expert-Aware Multi-Batch Pipeline .pdf

## Pre-gate Function

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Pre-gate Function 是 Pre-gated MoE 论文提出的新型门控函数，它修改了传统 MoE 中 Gate Function 的角色：传统 Gate 为当前 MoE block 选择激活的 Experts（intra-block gating），而 Pre-gate 为下一个 MoE block 选择激活的 Experts（inter-block gating）。具体地，第 N 个 MoE block 的 pre-gate 接收第 N 个 block 的输入 hidden states，输出第 (N+1) 个 block 应激活的 expert indices。该设计的关键价值在于完全消除 Expert selection 和 Expert execution 在同一 MoE block 内的串行数据依赖——在 Block N 执行 expert FFN 的同时，系统可以提前将 Block N+1 的 activated experts 从 CPU 迁移到 GPU，实现 compute 与 communication 的完全重叠。第一个 MoE block 额外保留传统 first-gate（选当前 block experts），最后一个 MoE block 无需 pre-gate。Pre-gate 是一个轻量 MLP（d_model → num_experts logits → softmax），计算量 < MoE block 总 FLOPs 的 1%。Pre-gate 在 fine-tuning 阶段训练（利用已有 pretrained MoE weights），不需要修改资源密集的 pretraining 阶段。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Pre-gated MoE 推理 pipeline（pre-gate 激活层级 N=1）：
```
# Block 0 (first MoE block): 两个 gate
gate0_out = first_gate(hidden_0)          # 传统 gate：选 block 0 experts
expert_idx_0 = topk(gate0_out, k=top_k)
pregate0_out = pre_gate_0(hidden_0)      # pre-gate：选 block 1 experts
next_expert_idx_1 = topk(pregate0_out, k=top_k)
output_0 = expert_ffn[expert_idx_0](hidden_0)  # 执行 block 0
# [ASYNC] 同时启动 expert_idx_1 参数从 CPU→GPU 迁移

# Block 1..N-1 (中间 blocks): 全流水线化
# expert_idx_1 已在 GPU ready
output_1 = expert_ffn[expert_idx_1](hidden_1)  # 直接执行
pregate1_out = pre_gate_1(hidden_1)
next_expert_idx_2 = topk(pregate1_out, k=top_k)  # 选 block 2 experts
# [ASYNC] 启动 expert_2 迁移 + 并发执行 block 1 FFN

# Block N (last): 无 pre-gate
output_last = expert_ffn[expert_idx_N](hidden_N)
```
数学：Pre-gate G_N^pre: h_N → p_{N+1} ∈ R^E, h_N 为第 N block 输入 hidden states, p_{N+1} 为第 (N+1) block E 个 experts 的激活概率。训练损失为标准 LM loss，pre-gate 参数通过反向传播学习。激活层级 N 可为 1/2/3（预选 1/2/3 个 block 后的 experts），论文实验表明 N=1 准确率最优（更远的 pre-gate 因输入信息衰减导致 expert 选择准确度下降）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Pre-gate 实现为一个小型线性变换（d_model × num_experts），在 fine-tuning 阶段与模型其他参数一起训练。训练配置：SwitchTransformer pretrained weights → pre-gate 随机初始化 → fine-tune 2,048 steps, batch 256 seq × 256 tokens, LR 0.0001（与 conventional MoE fine-tune 相同配置）。Pre-gate 对 model accuracy 几乎无影响：SQuAD F1 分数 Switch-Base-128 Pre-gated 89.4 vs conventional 89.2，Switch-Large-128 Pre-gated 90.2 vs conventional 90.1。推理时 pre-gate 计算开销极低（<1% block FLOPs），因为只是一个小 MLP forward。Conventional MoE 模型只需修改 gate 输出的 target block 并添加 pre-gate 层即可转换为 Pre-gated MoE，无需从头 pretrain。

涉及论文标题：
- 71-Pre-gated_MoE_An_Algorithm-System_Co-Design_for_Fast_and_Scalable_Mixture-of-Expert_Inference.pdf
- 81-Klotski- Efficient Mixture-of-Expert Inference via Expert-Aware Multi-Batch Pipeline .pdf

## Hot Experts / Cold Experts (MoE Expert Activation Skew)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hot Experts / Cold Experts 是 MoE 推理中观察到的专家激活不对称现象：在 MoE 模型的自回归推理过程中，少数 Experts（hot experts）承载了绝大多数 token 的路由分配，而其余 Experts（cold experts）仅被少数 token 选中。例如，在 Mixtral-8×7B（top-2 gate）的 layer 14 中，experts 1 和 3 合计覆盖了 53.7% 的 tokens；类似的不均匀分布在其他 layer、Switch Transformers 和 DeepSeekMoE 中均被观测到。该现象的成因与 gating network 对输入数据的敏感性有关——训练后 gate 学习到某些 experts 对通用语义模式（如常见词、标点、连接词）的处理能力更强，因此高频输入 token 倾向于路由到这些 experts。Hot experts 在推理时呈现"高计算需求 + 低 I/O 需求"特性（被频繁命中，传输一次可为大量 token 复用），Cold experts 则呈"低计算需求 + 高 I/O 需求"特性（传输耗时但计算量小）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Klotski 利用 Hot/Cold Expert 现象进行调度优化的过程（Mixtral-8×7B, top-2, multi-batch）：
```
# 在 attention layer 多 batch 计算期间，仅 prefetch gate + K hot experts
# K 的选取基于 expert correlation table 中的历史激活频率
hot_experts = expert_correlation_table.query(layer_idx, prev_selections)

# 仅 prefetch hot experts，而非全部 experts
prefetch(gate_weights, layer_idx)
for e in hot_experts:
    prefetch(expert_weights[e], layer_idx)

# Gate 计算后，检查每个 token 的 expert assignment
for token in all_tokens:
    selected = gate(token).topk(k)
    for e_id in selected:
        if e_id not in transferred_experts:
            async_transfer(expert_weights[e_id])  # cold expert 按需传输

# Expert layer 按 expert 维度组织计算
# Hot experts 优先执行（已在 GPU 就绪），其大量计算时间覆盖 cold experts I/O
for e_id in hot_experts:        # 优先：hot experts
    tokens = all_tokens[expert_assignments == e_id]
    output += expert_ffn[e_id](tokens)
for e_id in cold_experts:       # 后执行：cold experts 传输已完成
    tokens = all_tokens[expert_assignments == e_id]
    output += expert_ffn[e_id](tokens)
```
Hot/Cold 特性使得仅 prefetch K（=top-k）个 experts 即可覆盖多数 token 计算，其余按需加载。这让 Klotski 的 prefetch 不等式从 `n*tc_A >= tIO_all_experts` 降为 `n*tc_A >= tIO_gate + K*tIO_expert`，所需 batch group 数量 n 显著降低。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Hot/Cold Expert 现象在多个 MoE 模型中均有观测：Lina [23] 首次在 DeepSpeed-MoE 训练中发现 expert selection skew，并在推理中使用 hot expert 优先调度平衡负载；OpenMoE [42] 和 MoE-LLaVA [24] 进一步证实了数据依赖的 expert activation skew。在系统实现中，hot experts 的识别可通过：(a) offline profiling（预跑少量数据统计各 layer 各 expert 的激活频率，记录到 correlation table）；(b) online 动态追踪（inference 过程中持续更新频率统计）。Klotski 采用 offline pre-run（wikitext-2, batch=8, seq=512）建立 expert correlation table（JSON），在线推理阶段查表确定 hot experts 并动态更新。

涉及论文标题：
- 81-Klotski- Efficient Mixture-of-Expert Inference via Expert-Aware Multi-Batch Pipeline .pdf

## Token Merging (TM) for ViT

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Merging (TM) 是一种针对 Vision Transformer (ViT) 模型的推理加速技术，通过将相似token合并为代表性token来减少输入序列长度，从而降低整个模型的FLOPs和内存占用。TM操作分为两个阶段：(1) Token Matching (TMatch)——计算token间相似度，确定相似token分组（clustering）；(2) Cluster Aggregation——合并cluster内token：Prune Merge（保留一个representative token）或Average Merge（求平均）。TM可应用于ViT每层attention block之前或之后。ToMe[12]是该方向代表性工作，通过随机bipartition + brute-force bipartite matching实现O(N²) TM。关键洞察：图像中存在大量同质区域→相邻token高度相似→合并后几乎不损失信息但大幅减少计算量。CLS token不参与merge。Proportional Attention利用cluster population对Softmax attention score rescaling保持精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TM在ViT推理中的计算流程（ToMe prune merge, ViT-base-patch16-224, N=197）：
```
# Step 1: Bipartition
A, B = random_split(X, ratio=0.5)  # |A|,|B| ≈ N/2
# Step 2: Cosine similarity O(N²)
sim = A @ B.T / (norm(A)·norm(B).T)  # [N/2 × N/2], d×N²/4次FP16乘法
# Step 3: Bipartite matching
for a in A: best[a] = argmax(sim[a, :])  # 找最相似pair
# Step 4: Merge top r×N most similar pairs → N' = N×(1-r)
X_reduced = prune_merged_tokens(X, top_k_pairs, r)
# Step 5: Continue ViT backbone
Q,K,V = Linear(X_reduced); attn = Softmax(QK^T/√d)V; FFN(attn)
```
Fixed merge rate r逐层递增（浅层0.1→深层0.8），对所有图像统一。AdapTiV将TM中的cosine similarity替换为Sign similarity (1-bit XNOR)，将O(N²) brute-force替换为O(N) LMatch，将fixed MR替换为Dynamic MR。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Software: PyTorch/timm中通过similarity computation+clustering实现。ToMe开源(github.com/facebookresearch/ToMe)，monkey-patching插入预训练ViT forward，无需retrain。
- Hardware (AdapTiV): 专用ASIC AdapTME模块：SSCU用XNOR gates计算Sign similarity + SP存储comparison token sign bits + SPMU管理semantic→physical mapping + TIM跟踪merge状态。Sign-Driven Scheduling将TM嵌入LayerNorm→latency overhead降至零。AdapTME仅占1.49% area和1% power。
- 使用: 适用于任意预训练ViT（DeiT、Swin等变体），off-the-shelf无需训练。

涉及论文标题：
- 73-AdapTiV_Sign-Similarity_Based_Image-Adaptive_Token_Merging_for_Vision_Transformer_Acceleration.pdf

## Sign Similarity

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sign Similarity是AdapTiV提出的轻量级token相似度度量，替代TM中的cosine similarity。对两个d维向量a,b，定义为Σ_{i=1}^d (1 if sign(a_i)=sign(b_i) else 0)，即对应元素符号相同个数。直觉：向量方向越接近（夹角越小）→各维度符号相同概率越高→可近似cosine similarity。ImageNet-1K验证：与cosine similarity的Pearson correlation=0.95, mutual information=0.95。核心优势：(1)计算从d个n-bit乘法器→d个1-bit XNOR门（约1/n开销）；(2)存储从n-bit/element→1-bit/element（约1/n存储）。sign bits在LayerNorm的x_i-μ_i减法后天然可用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 替代: cosine_sim = (a·b) / (||a||×||b||) → d×n-bit乘法+d次加法+norm+除法
# Sign Similarity:
sign_a = [1 if a[i]>=0 else 0 for i in range(d)]  # d-bit vector
sign_b = [1 if b[i]>=0 else 0 for i in range(d)]
sign_sim = Σ_i (sign_a[i] XNOR sign_b[i])  # scalar ∈ [0,d], d个1-bit XNOR+PopCount
if sign_sim >= threshold: merge(a, b)  # 阈值判定
```
d=768, FP16: cosine sim需768×16-bit乘法 vs Sign sim需768×1-bit XNOR→理论16×更高效。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 硬件(AdapTiV SSCU): 2 lanes × 64 XNOR lines + PopCount + threshold comparator。d=768需12 cycles完成（64-bit batches）。XNOR gate约1/20面积和1/10功耗 vs FP16 multiplier。
- 使用: 替换TM中所有cosine similarity计算。局限: 仅考虑方向不反映模长，对模长差异大但方向相似的token可能over-merge，但accuracy loss<1%。

涉及论文标题：
- 73-AdapTiV_Sign-Similarity_Based_Image-Adaptive_Token_Merging_for_Vision_Transformer_Acceleration.pdf

## Local Matching (LMatch) in Token Merging

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Local Matching (LMatch) 是AdapTiV提出的token matching搜索空间限制策略。传统TM（ToMe）使用brute-force全对全搜索O(N²)。LMatch利用图像空间局部性先验——相邻token更可能视觉相似——将TMatch限制为每个token仅与left neighbor（同行左侧）和above neighbor（同列上方）比较，复杂度降至O(N)。ImageNet-1K实验证实：LMatch将effective TMatch比例（相似度>0.75的匹配）从brute-force的9.6%提升至36%，因相邻token确实更可能相似。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 按top-left→bottom-right扫描，image width W
for k in range(N):
    col = k % W
    if col > 0: sim_left = SignSim(T[left_idx[k]], T[k])   # 左侧邻居
    sim_abv = SignSim(T[abv_idx[col]], T[k])               # 上方邻居
    if sim_left >= thresh: merge(T[k], left_cluster)
    elif sim_abv >= thresh: merge(T[k], above_cluster)
    else: keep(T[k])  # 独立token，存入SP作为future comparison token
```
N=196: brute-force ~19,208次比较 vs LMatch ~392次→约49× fewer。空间局部性保证搜索质量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 硬件(AdapTiV): SP存储最多W个comparison tokens sign bits; SPMU用W×(W+1)-bit bitmap管理semantic address (Left + Abv#1~#W)到physical address映射，单cycle lookup。
- 适用: 空间局部性强的视觉数据。不适用于文本等。
- 局限: 仅比较left/above可能miss diagonal相似性，但accuracy loss<1%表明命中率足够。

涉及论文标题：
- 73-AdapTiV_Sign-Similarity_Based_Image-Adaptive_Token_Merging_for_Vision_Transformer_Acceleration.pdf

## Dynamic Merge Rate / Image-Adaptive Token Merging

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dynamic Merge Rate (MR) 是AdapTiV提出的图像自适应token合并策略。传统TM用Fixed MR（每层固定r_l，所有图像统一）。Dynamic MR让merge rate完全由图像内容决定：按top-left→bottom-right扫描所有effective tokens→LMatch+Sign similarity判定→相似即merge→不预设合并数量。具有累积特性：每层TM从上一层结果继续→cluster跨层累积扩展。结果：简单图像merge rate可96.5%，复杂图像可0%，实现per-image自适应。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
effective = all_tokens  # N initially
for layer l in 1..L:
    for t_k in effective:  # top-left→bottom-right扫描
        sim_left, sim_abv = LMatch_SignSim(t_k, SP)
        if similar: merge(t_k, cluster); early_stop_LN(t_k)
        else: complete_LN(t_k); SP.store(sign(t_k))
    effective = [t for t in effective if t.independent]  # 累积减少
# N'完全由图像内容决定: 天空→7 (96.5% merged); 人群→196 (0% merged)
```
对比Fixed MR: Fixed固定合并N×r tokens→不论图像内容。Dynamic合并数=Σ(similar pairs)→per-image自适应。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 硬件(AdapTiV): TIM跟踪per-token merge状态和origin address; SPMU管理SP; SSCU streaming sign bits comparison。通过Sign-Driven Scheduling与LN并行→零额外latency。AdapTiV ASIC 2.49mm²@28nm, 14.6× speedup over edge GPU。
- GPU局限: 论文指出"dynamically changing input size is unfavored on GPUs"→专用硬件使Dynamic MR可行且高效。
- 效果: accuracy loss<1% (ImageNet-1K), 无需training。AdapTiV-Lite co-processor: +edge CPU 2.95×, +edge GPU 2.65× speedup。

涉及论文标题：
- 73-AdapTiV_Sign-Similarity_Based_Image-Adaptive_Token_Merging_for_Vision_Transformer_Acceleration.pdf

## DLZS (Differential Leading Zero Summation, 差分领先零求和)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DLZS是SOFA提出的一种log-domain无乘法计算范式，用于动态稀疏attention的pre-compute阶段预测Q-K pairs的重要性。核心思想：将乘法 `x·y` 转换为 `×OR(sign_x, sign_y) × M_x · 2^(W - LZ_y)`，即仅需对y提取leading zero count (LZ)作为移位量，对x做移位操作和add，完全消除乘法器。其中 `x = Sign × M × 2^(W-LZ)`，M为归一化mantissa ∈ [0,1]。差分(Differential)的含义：仅将乘法的一个operand通过LZE转对数域，另一个直接移位，相比vanilla leading zero方案（两operand都转one-hot）减少一半converter开销和一半误差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DLZS分两阶段操作（以8-bit token prediction, 16-bit attention prediction为例）：
```
# Phase 1: DLZS Key Prediction (预转换Wk为LZ格式存储，4-bit)
# x: 8-bit token, Wk: pre-converted LZ format {sign, M, LZ}
K̂ = XOR(sign_x, sign_Wk) × M_Wk × (x << LZ_Wk)  # shift + add, no multiply

# Phase 2: DLZS Attention Prediction
# Q: 16-bit, K̂: 8-bit cached result
LZ_Q = LeadingZeroCount(Q)  # 5-bit
Â = XOR(sign_Q, sign_K̂) × M_Q × (K̂ << LZ_Q)  # shift + sum
```
DLZS用128×32 systolic shift array实现。首先预将Wk权重转换为LZ格式并存储。Key prediction阶段：8-bit token × 4-bit LZ format weight，通过shift-sum得到K̂ (8-bit)，缓存在output buffer。Attention prediction阶段：16-bit Q经LZE得到5-bit LZ，与K̂一起送入shift array产生Â。相比baseline (4-bit direct multiplication)，DLZS减少18%计算复杂度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
硬件实现：configurable LZE支持8/16-bit混合精度——每个LZE含两个8-bit LZC (Leading Zero Counter)串联，8-bit时独立工作，16-bit时通过AND级联。Zero eliminator跳过与零相关的计算减少switching activity。128×32 systolic shift array执行移位和求和。适用场景：任何需要低精度快速预测矩阵乘结果的场景（尤其是attention sparsity prediction），可作为multiplication-free近似计算引擎。

涉及论文标题：
- 74-SOFA_A_Compute-Memory_Optimized_Sparsity_Accelerator_via_Cross-Stage_Coordinated_Tiling.pdf

## SADS (Sphere-Search-Aided Distributed Sorting, 球搜索辅助分布式排序)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SADS是SOFA提出的一种利用数据分布特性的低复杂度top-k排序算法。其设计基于DCE (Distributed Cluster Effect)发现：attention矩阵中Type-I（少数token主导，占~25%）和Type-II（均匀分布的多dominant tokens，占>76%）共占>95%分布。DCE的核心属性：一个长序列可划分为若干较短的sub-segments，每段内的较大值能充分代表全局较大值。SADS将一行长度为S的attention vector分为n=4个sub-segments，每段独立执行top-(k/n)选择，利用sphere search（以previous max为benchmark，减去search radius r确定feasible range）裁剪排序范围以降低比较开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SADS流程（以S=1024, n=4, k=25%为例）：
```
# Input: One row of Â (size S), n=4 sub-segments
For each iteration:
  benchmark = Max from previous iteration
  feasible_range = {entries | value >= benchmark - r}  # r: search radius
  For each sub-segment s_i (size S/4):
    Clip: block entries outside feasible_range (substitute with 0)
    Bitonic sort (16-to-4): input 12 new values + 4 retained max from prev round
    Select top-(k/4) values → FC_i  # sub-segment candidate set
  FC = ∪ FC_i  # merge all sub-segment candidates
  
  # Specific handling:
  - top-1 and top-2: kept precisely ordered (for SU-FA)
  - 3rd to k-th: order not preserved (eliminate redundant comparators)
  
  Threshold update for next iteration:
  top_margin = Max_current - r
  low_bound = Min value in current output buffer
  threshold = max(top_margin, low_bound)
```
对于Type-I分布（dominant value集中在某sub-segment），SADS必然捕获该dominant value；对于Type-II（均匀分布），SADS能有效选出所有相对较大的值。边缘值影响小，排序要求可适当放松。实验（S=1024, n=4, top 25%）显示平均误差仅~3%（以不同K index计为错误）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
硬件实现：Iterative SADS unit含128个并行16-to-4 bitonic sort core，支持flexible-input长度（tile size随模型/任务变化）。每轮输入12新元素 + 4 retained max → 输出4新max。Clipping module通过Threshold Updating Unit动态更新threshold，将小于threshold的值替换为零消除switching activity。排序只保留top-1和top-2的精确序（3rd到k-th消去冗余比较器）。SADS与DLZS组成LP (Low-complexity Prediction)机制，共同在pre-compute和top-k阶段以低开销预测vital tokens。

涉及论文标题：
- 74-SOFA_A_Compute-Memory_Optimized_Sparsity_Accelerator_via_Cross-Stage_Coordinated_Tiling.pdf

## SU-FA (Sorted-Updating FlashAttention, 排序更新FlashAttention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SU-FA是SOFA提出的一种同时优化计算和内存访问的attention计算机制。传统FlashAttention-2通过tiling减少off-chip memory access，但需在多tile间频繁刷新global MAX值，导致Exp和比较操作随tile数Tc激增（S=2048时比vanilla多9×10^6次Exp）。SU-FA利用top-k阶段提供的sorting信息绕过MAX比较：采用descending order更新——从predicted MAX index开始，按值从大到小依次处理k个vital Q-K pairs。Descending updating使li更新仅需1 Exp + 1 Add，而ascending updating需1 Exp + 1 Mul + 1 Add，vanilla FA-2需频繁comparison + Exp + Mul + Add。平均降低25%计算复杂度（vs vanilla FA-2），11%（vs ascending SU-FA）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SU-FA descending order流程：
```
# Input: Q_i (query row), sorted K/V indices [idx[1]...idx[k]] (desc by predicted value)
# K/V divided into Tc tiles, tile size Bc
For j = 1 to Tc:
  For q = 1 to k (descending order):  # from largest predicted to smallest
    s_{iq} = Q_i · K_{idx[q]}^T           # in Systolic Array 1
    
    if first computation in this tile:
      m_i = s_{iq}                         # initialize MAX (Mode 1 of AP module)
    else:
      P_i = exp(s_{iq} - m_i)              # Mode 0 of AP module
      l_i += rowsum(P_i)                   
      O_i += P_i · V_{idx[q]}              # in Systolic Array 2
    
  # Between tiles: AP module Mode 1 checks s vs cached MAX, updates if needed

O_i = O_i / l_i   # Final normalization
```
Ascending vs Descending对比：
- Ascend: m_i^(j) = x_i^(j), l_i^(j)需要 1 Exp + 1 Mul + 1 Add
- Descend: m_i^(j) = x_i^(j), l_i^(j)仅需 1 Exp + 1 Add（消除Mul）
因为descending时m_i^(j)恒等于当前x_i^(j)，而ascending时m_i^(j-1) ≠ x_i^(j)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
硬件：SU-FA engine含两个128×4 Output Stationary Systolic Array（SA-1 for Q×K^T, SA-2 for Score×V），128个EXP单元，128个DIV单元。AP (Auxiliary Process) module为folded设计——同一电路在双mode下时分复用：Mode 0 (computation): s − Max → Exp；Mode 1 (max update): 比较s与Reg中的Max并更新。Tiled computation controller管理tile间和tile内phase切换。由于DLZS的log-domain近似可能引入MAX估计误差，AP module的MAX确保电路在tile切换或首phase激活Mode 1进行runtime校正。SU-FA与SADS协同实现cross-stage tiling：SADS的tiled sorting信息直接引导SU-FA的tiled computation。

涉及论文标题：
- 74-SOFA_A_Compute-Memory_Optimized_Sparsity_Accelerator_via_Cross-Stage_Coordinated_Tiling.pdf

## Cross-Stage Coordinated Tiling (跨阶段协同分块)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cross-Stage Coordinated Tiling是SOFA提出的动态稀疏加速核心理念：将传统动态稀疏的pre-compute → top-k → formal-compute三阶段分别分解为fine-grained sub-stage tiles，使后阶段无需等待前阶段完全结束即可开始处理，实现跨阶段流水线化。关键洞察：传统DS各阶段存在信息解耦（information decoupling），前阶段的中间结果可引导后阶段减少计算和内存访问，但现有方案因全行处理（whole-row-processing）的row dependency而难以实现。SOFA通过DLZS（tiled log-domain prediction）→ SADS（sub-segment distributed sorting）→ SU-FA（sorted-updating tiled attention）实现首个跨阶段协同分块的动态稀疏pipeline。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以128 token parallel, S=2048, 每tile Bc=16为例的Cross-Stage Tiled Pipeline：
```
Tile-level pipeline (fine-grained, not stage-level serialization):

Tile[0] of K/V (Bc=16 rows):
  → DLZS predicts Â_tile0 (128×16)
  → SADS sorts each row's tile of 16 elts → top-4 indices per row
  → Mask guides on-demand K/V generation for selected indices
  → SU-FA computes sparse attention for this tile

Tile[1] of K/V (Bc=16 rows):  
  → DLZS predicts Â_tile1 (in parallel with Tile[0]'s SU-FA)
  → SADS sorts (in parallel with Tile[0]'s SU-FA)
  → ... pipeline continues

Tile Synchronization:
  - Between tiles: AP module Mode 1 compares s with cached MAX, updates if needed
  - Cross-stage info flow: SADS top-k indices → guide SU-FA descending order
                           SADS top-1 index → potential MAX position for SU-FA
```
相比传统串行三阶段流水线（pre-compute all → store to DRAM → load + top-k all → store → load + formal compute all），cross-stage tiling使中间结果完全留在SRAM（无需DRAM往返），Tiling size Bc和top-k per layer通过Bayesian optimization DSE确定：目标 min L(R)=L_en + α·Σ(Bc_i·k/S·k) + β·Σ(S/Bc_i)，搜索空间Bc_i ∈ {2,4,...,32}，top-k ∈ {5%,10%,...,50%}。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
硬件层面由Tiled & Out-of-Order Controller管理tile同步、KV乱序执行和数据预取。Pipeline将DLZS prediction、SADS sorting、KV generation和SU-FA computation融合为单一连续数据流。适用场景：任何需要将长序列处理分解为fine-grained tile以降低memory access和latency的动态稀疏attention加速。SOFA在128 token parallelism下实现85.2% PE utilization，消除79% memory access（vs vanilla DS），latency降至45ms（Llama7B attention, vs FACT 296ms）。

涉及论文标题：
- 74-SOFA_A_Compute-Memory_Optimized_Sparsity_Accelerator_via_Cross-Stage_Coordinated_Tiling.pdf

## LTPP (Large-scale Token Parallel Processing, 大规模Token并行处理)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LTPP指在现代LLM推理中以高并行度同时处理大量token的处理模式。随着LLM上下文长度不断增长（GPT-4 32k, LongLLaMa 256k），快速处理扩展上下文变得至关重要。LTPP在prefill阶段尤为关键——整个上下文被同时处理，高token并行度能提升效率。此外speculative inference可将decode操作转化为prefill任务，进一步增加LTPP需求。SOFA的分析显示：增加token parallelism能有效提升attention模块的Operation Intensity (OI)，在相同计算能力下减少data movement需求（因数据重用增加），理论上将性能瓶颈从memory-bound推向compute-bound。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LTPP的挑战（以T=512 token parallel处理Llama-13B, S=2048为例）：
- Pre-compute复杂度O(T·S·H)：需10^11次比较和10^8次乘法，prediction占>57%总延迟
- Top-k复杂度O(T·S·k)：全行排序要求整行Â就绪，T=512, S=2048时需5MB SRAM（5.47mm² @ TSMC 28nm）
- Memory access：T=128时MAT ratio达72%，off-chip DRAM成为主要瓶颈
SOFA的LTPP策略：通过cross-stage tiling将并行处理的中间结果限制在SRAM内（total 316KB on-chip），每个tile独立处理无需等待全局数据就绪。128 token并行处理 @ 1GHz，总吞吐量4905 GOPS。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通用实现需考虑：(1) tiling strategy分解长序列避免SRAM溢出；(2) cross-stage信息传递避免重复计算；(3) memory hierarchy设计保证tile间数据留在on-chip。GPU上TP (Tensor Parallelism)是LTPP的一种形式但受限于GPU的向量引擎难以高效处理fine-grained control flow。专用加速器如SOFA通过算法-硬件协同设计实现高效LTPP：85.2% PE utilization vs GPU ~30%。

涉及论文标题：
- 74-SOFA_A_Compute-Memory_Optimized_Sparsity_Accelerator_via_Cross-Stage_Coordinated_Tiling.pdf

## On-Demand KV Generation (按需KV生成)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
On-Demand KV Generation是SOFA提出的一种策略：仅生成被prediction阶段判定为重要的token对应的K和V向量，trivial tokens的K/V从源头就不计算。传统方案先生成全部K/V再进行sparse selection，导致大量计算和内存浪费。On-demand策略：pre-compute阶段先用DLZS预测K̂和Â → top-k selection确定vital indices → 仅对这些indices执行 Ki=xi·Wk, Vi=xi·Wv的线性投影。这要求prediction必须发生在K/V generation之前，打破传统Transformer层的QKV同时生成的顺序。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 传统方案 (compute then prune):
Q, K, V = X @ Wq, X @ Wk, X @ Wv     # 全量生成
Â = low_precision(Q × K^T)            # prediction
mask = top_k(Â)                        # selection
O = sparse_attention(Q, K[mask], V[mask])  # 仅用稀疏KV

# On-Demand KV Generation (predict then compute):
Q = X @ Wq                             # Q 总是需要
K̂ = DLZS_Key_Prediction(X, Wk_LZ)     # low-cost K estimation
Â = DLZS_Attn_Prediction(Q, K̂)        # low-cost attention prediction
mask = SADS(Â)                         # distributed top-k selection
K_sparse = X[mask] @ Wk               # 仅生成选定token的K
V_sparse = X[mask] @ Wv               # 仅生成选定token的V
O = SU_FA(Q, K_sparse, V_sparse, mask) # 稀疏attention
```
SOFA实验：on-demand策略结合DLZS sparsity prediction，在0%/1%/2%准确率损失下减少Attention+QKV总计算量56.8%/62.6%/67.4%，单独Attention减少81.3%/87.7%/92.6%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
硬件实现需要：(1) prediction engine（如DLZS）先行估算重要性；(2) data fetcher根据mask选择性加载X中对应token行；(3) PE array执行sparse KV projection。SOFA中由TOOC (Tiled & Out-of-Order Controller)管理这一过程的tile级调度。此概念可推广至任何"先预测后计算"的sparse computing范式。

涉及论文标题：
- 74-SOFA_A_Compute-Memory_Optimized_Sparsity_Accelerator_via_Cross-Stage_Coordinated_Tiling.pdf

## DCE (Distributed Cluster Effect, 分布式聚类效应)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DCE是SOFA通过对attention token分布的统计分析发现的数据特性。"Distributed"指一个长序列可划分为多个较短sub-segments；"Cluster"指每个sub-segment内包含其主要信息（即该段内的较大值能代表全局较大值）。SOFA分析了BERT/L、ViT/B、GPT-2、Llama7B四种模型的4096行attention分布，发现三种分布类型：Type-I（少数token主导，占~25%）→Dominated by a few tokens；Type-II（均匀分布的多个dominant tokens，占>76%）→Dominated by several tokens evenly distributed；Type-III（集中在某区域的larger elements，接近0%）→Concentrated in one region。Type-I和Type-II共占>95%，因此基于well-segmented partitions的排序对整体性能影响可忽略。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DCE直接驱动SADS算法设计：
- 对Type-I分布：无论dominant value落在哪个sub-segment，该sub-segment的top-(k/n)必然捕获它 → SADS保证正确性
- 对Type-II分布：每段均匀分布的相对较大值通过sub-segment排序被选出，合并后近似全局top-k → SADS高效
- 对Type-III分布（极少出现）：可能需要跨segment比较，但因其极低概率(~0%)，影响可忽略

DCE使SADS在n=4, S=1024, k=25%时平均误差仅~3%，为distributed tiled sorting提供理论支撑。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DCE的发现通过统计分析得来（BERT-B/L on GLUE, ViT on ImageNet, GPT-2 on WikiText2, Llama7B on WikiText2），验证了"聚类效应"在不同模型/任务中普遍存在。使用方式：在pre-deployment阶段分析target模型的attention分布特征，确定sub-segment数量n和search radius r等超参数，指导运行时SADS的tile配置。此效应也可能适用于其他需要distributed approximate top-k的场景。

涉及论文标题：
- 74-SOFA_A_Compute-Memory_Optimized_Sparsity_Accelerator_via_Cross-Stage_Coordinated_Tiling.pdf

## Activation Sparsity in LLMs（大语言模型激活稀疏性）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM中的Activation Sparsity指在transformer-based LLM推理过程中，由于激活函数（如ReLU、SiLU、GELU）的非线性特性，大量中间激活值被置零或接近零，导致对应的权重参数（神经元）无需被加载和计算的现象。具体机制：(1) 在MLP block中，ReLU(x)=max(0,x)对负值输入输出0→后续FC层的对应列/行权重成为"inactive neuron"——因为activation=0使得该neuron对最终输出的contribution为零，加载和计算该neuron是冗余操作。(2) 该稀疏性遵循power-law分布：约20%的神经元（hot neurons）承担80%的计算量（activation频繁非零），其余80%（cold neurons）仅承担20%计算量，hot neuron的计算强度（computation intensity）是cold的16×。(3) 对self-attention block，可在QKV generation前人工插入ReLU函数[38]来人为引入激活稀疏性。(4) 对于原生非ReLU激活的模型（如LLaMA用SiLU、Falcon用GELU），可替换为ReLU[38][52]，准确率损失<1%但稀疏性可达70%-90%。(5) 稀疏性具有token-wise temporal locality（相邻token激活模式>90%相似）和layer-wise correlation（连续层间激活neuron高度相关）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在LLM推理的token generation阶段（以OPT-66B单层MLP为例，原始dense计算 → activation sparsity优化后）：

```
原始dense FC1计算:
Input: activation x (dim H), weight W_FC1 [H, 4H]
Output: y = x @ W_FC1  # 所有H×4H次乘法

利用activation sparsity后的计算:
1. 前一层输出经ReLU: a = ReLU(x)  # 产生稀疏activation
2. 识别inactive positions: mask = (a == 0)  # zero positions
3. 仅计算active columns: y[j] = Σ_{i where mask[i]==0} a[i] * W_FC1[i,j]
   # 跳过mask[i]==1的i维度乘加，减少约70%-90%的MAC操作
```

更具体的Hermes hot/cold neuron版本：
```
对于每层:
  已知neuron state table S[n] (4-bit per neuron, 0~15)
  hot_neuron = [i for i where S[i] > Th]  # Th=10, ~20% of neurons
  cold_neuron = [i for i where S[i] <= Th]  # ~80% of neurons
  
  # GPU仅计算hot neurons
  y_gpu = Σ_{i in hot} a[i] * W[:,i]  # GPU Tensor Cores GEMM
  
  # NDP-DIMM计算cold neurons中实际被激活的部分
  for i in cold_neuron where a[i] != 0:
      y_dimm += a[i] * W[:,i]  # DIMM GEMV unit
  
  y = y_gpu + y_dimm  # merge in NDP-DIMM
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) **Native ReLU models**（OPT系列）：直接利用MLP block中的ReLU激活函数，无需模型修改。LLaMA/Falcon等模型需要替换激活函数[38]或插入额外ReLU[52]。(2) **Replacement approach**：将SiLU/GELU替换为ReLU后fine-tune或直接使用（准确率损失<1%），修改版模型在https://huggingface.co/SparseLLM开源。(3) **Insertion approach**：在self-attention的QKV generation前插入ReLU函数，扩展稀疏性到attention block。(4) **Prediction-based loading**（Deja Vu [34]）：用per-layer MLP predictor预测activated neurons→只加载预测激活的neuron→减少memory access和computation。(5) **Hermes approach**：用lightweight FSM-based predictor（4-bit state table+token-wise+layer-wise）替代MLP predictor，预测accuracy 98%但内存<1MB、runtime overhead<0.1%。(6) **Hardware协同**：在Hermes中，activation sparsity直接驱动hot/cold neuron分区→决定GPU vs NDP-DIMM的workload分配。

涉及论文标题：
- 77-Make_LLM_Inference_Affordable_to_Everyone_Augmenting_GPU_Memory_with_NDP-DIMM.pdf

## Hot/Cold Neuron Partition（热/冷神经元分区）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hot/Cold Neuron Partition是基于LLM中activation sparsity的power-law分布将权重参数（neurons）分为两类的策略。"Neuron"在此定义为weight matrix中的特定行或列（如FC1层的某列weights），当其对应的activation为0时该neuron不被激活。"Hot neuron"：频繁被激活（activation非零频率高），占总参数约20%但承担约80%的计算量，计算强度（computation intensity）是cold neuron的16×。"Cold neuron"：较少被激活，占总参数约80%但仅承担约20%的计算量。分区是input-specific的——约52%的offline初始hot neuron在runtime会表现不同的activity pattern，因此需要online动态调整。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Hermes中的hot/cold分区和动态调整流程（Algorithm伪代码）：

```
// Offline phase: ILP求解初始optimal mapping
Input: 每neuron的activated frequency f_i (profiled over C4/Pile 128 samples),
       每个neuron的memory size M_i, 每processing unit的storage S_j,
       每processing unit的per-neuron compute time T_l^j, sync time Tsync
Objective: min Σ_l max(T_GPU_l, T_DIMM_l)
  where T_GPU_l = T_l^GPU · Σ_i f_i · x^GPU_il + 2·Tsync
        T_DIMM_l = max_j( T_l^dimm-j · Σ_i f_i · x^dimm-j_il )
Constraints: Σ_i M_i · x^GPU_il ≤ S_GPU,  Σ_i M_i · x^dimm-j_il ≤ S_dimm-j
Solver: PulP (Open-source ILP solver), ~110 seconds

// Online phase: per-token dynamic adjustment
For each generated token t:
  // Step 1: Update neuron state table (token-wise prediction)
  For each neuron i:
    if activated(t):  S[i] += s  (s=4)
    else:             S[i] -= 1
    S[i] = clamp(S[i], 0, 15)
  
  // Step 2: Layer-wise prediction enhancement
  For each neuron i in layer l:
    corr_count = count of activated neurons among
                 top-2 correlated neurons (from layer l-1 correlation table)
    if S[i] + λ·corr_count > T:  predict activated  (λ=6, T=15)
  
  // Step 3: Hot/cold reclassification
  For each neuron i:
    if S[i] > Th:  status = HOT   (Th=10)
    else:          status = COLD
  
  // Step 4: Neuron relocation (during projection computation)
  new_hot = [i where status==HOT and current_location==DIMM]
  evict_cold = [j where status==COLD and current_location==GPU, sorted by S[j] asc]
  For each h in new_hot (limited by GPU free space):
    Copy W[h] from DIMM to GPU memory (overwrite evict_cold pop's slot)
    // Note: all weights already stored in DIMM, only overwrite GPU copy
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) **Offline ILP formulation**：将neuron mapping建模为整数线性规划问题，用PuLP求解器一次性求解初始最优mapping（~110秒）。(2) **Online predictor**：Hermes使用轻量级FSM-based predictor（4-bit neuron state table，借鉴分支预测的2-bit saturating counter设计扩展到4-bit），而非Deja Vu的per-layer MLP predictor（需2GB存储，占10%-25% runtime）。(3) **Neuron relocation**：因为所有weights在DIMM中都有完整副本，hot neuron重定位只需覆盖GPU memory中的cold neuron slot，无额外的weight copy开销。(4) **Threshold selection**：Th=10是empirical选择——state range 0-15, state>10表示该neuron在最近token中频繁激活→应被视为hot。(5) 该策略适用于所有具有activation sparsity的LLM（OPT原生ReLU + LLaMA/Falcon的ReLU替换版本）。

涉及论文标题：
- 77-Make_LLM_Inference_Affordable_to_Everyone_Augmenting_GPU_Memory_with_NDP-DIMM.pdf

## Lightweight FSM-based Neuron Predictor（基于有限状态机的轻量级神经元预测器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Lightweight FSM-based Neuron Predictor是Hermes系统中用于在线预测LLM推理中每个neuron是否将被激活的轻量级预测器。它替代了Deja Vu等系统中昂贵的per-layer MLP predictor（需2GB存储、占10%-25% runtime）。设计灵感来自CPU分支预测中的两级自适应训练策略[60]。核心组件：(1) 4-bit Neuron State Table——每个neuron一个4-bit饱和计数器（0~15），类似分支预测器的2-bit predictor扩展到4-bit。(2) Token-wise Finite State Machine——激活则state+=s（s=4），未激活则state-=1，clamp至[0,15]。(3) Layer-wise Correlation Table——offline从previous layer采样每个neuron的top-2最相关neuron及其correlation weight。(4) 联合预测公式：若 S[i] + λ·C[i] > T 则预测activated（λ=6, T=15, S[i]=token-wise state, C[i]=前层相关neuron中被激活的数量）。

从算法pipeline角度拆解术语，给出具体例子：
```
初始化 (prompting phase结束后):
  统计prefill阶段各neuron的激活频率 f_i
  将f_i的分布分为16个bin（等频分箱）
  S[i] = bin_index(f_i)  // 0~15的初始state

Token generation阶段每token的预测:
  Given: neuron_state_table S[n], correlation_table C[n][2] (per neuron存top-2 correlated)
  
  // Token-wise update (FSM step)
  For each activated neuron i in current token:
    S[i] = min(15, S[i] + 4)
  For each inactive neuron j:
    S[j] = max(0, S[j] - 1)
  
  // Layer-wise prediction (for next layer l+1)
  For each neuron k in layer l+1:
    corr_active = count(C[k][0] is activated, C[k][1] is activated)
    predicted_active[k] = (S[k] + 6 * corr_active > 15)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 存储开销：每个neuron仅需4-bit state + correlation table（每neuron存2个correlated neuron index+weight），LLaMA-7B的state table仅需232KB，correlation table可small。(2) 运行位置：predictor集成在host CPU scheduler中，neuron state table存在CPU last-level cache以实现快速访问。(3) 准确率：98%（联合token-wise+layer-wise），远高于单独token-wise（受token间activity fluctuation影响）或layer-wise（依赖静态采样表）。(4) 与MLP predictor对比：MLP predictor需要per-layer的独立MLP模型，LLaMA-7B需要2GB额外存储（每层~64MB），推理时需forward each MLP（10%-25% runtime overhead），而FSM predictor only needs table lookup+arithmetic，runtime overhead <0.1%。

涉及论文标题：
- 77-Make_LLM_Inference_Affordable_to_Everyone_Augmenting_GPU_Memory_with_NDP-DIMM.pdf

## Per-Group Weight Quantization

术语是什么？
Per-Group Weight Quantization（分组级权重量化）是LLM权重量化的中间粒度，介于per-tensor（整张张量共享一个scaling factor）和per-channel（每个输出通道独立scaling factor）之间。具体方法：将权重矩阵 W^{K×D}（K个输出通道，每通道D维）沿通道维度进一步划分为 D/G 个group，每个group含G个连续权重，每个group独立分配scaling factor（和zero-point）。常用group size G=128，由AWQ [30]和GPTQ [20]等SOTA量化框架采用，在精度和内存开销间取得平衡。Key insight: 更小的group意味着更小的maximum value和value range（图2），使得scaling factor ∆更小、量化误差 error ∝ ∆ 更低。但memory overhead = (scaling_factor_bits + zero_point_bits) × K × D/G，group越小overhead越大。

从算法pipeline角度拆解：
以per-group INT4-Asym量化OPT-1.3B的权重矩阵 W^{2048×2048} 为例（G=128）：
```
# Input: Wf ∈ FP16^{2048×2048}, G=128
# Output: Wq ∈ INT4^{2048×2048}, ∆ ∈ FP16^{2048×16}, z ∈ INT8^{2048×16}

for k in 0..K-1:  # 每输出通道
    for g in 0..(D/G)-1:  # 每group
        Wg = Wf[k, g*G : (g+1)*G]  # 128个权重
        Wg_max = max(Wg)
        Wg_min = min(Wg)
        ∆[k,g] = (Wg_max - Wg_min) / (2^4 - 1)  # INT4 scaling factor
        z[k,g] = round(-Wg_min / ∆[k,g])         # zero-point
        # 量化: Wq = round(Wg/∆ + z), clamp to [0, 15]
        Wq[k, g*G:(g+1)*G] = clamp(round(Wg/∆[k,g] + z[k,g]), 0, 15)

# 推理时dequantize (per-group):
# partial_sum = Σ_i (Wq[i] - z[g_i]) * ∆[g_i] * A[i]
# 注意: 不同group的∆不同→每计算完一个group的dot product后必须立即dequantize
```
Per-group vs per-channel的关键差异：per-channel dequantization可fuse到后续element-wise操作中（如layer-norm），而per-group dequantization必须在dot product计算过程中插入（因为同一channel内不同group有不同scaling factor），对硬件设计提出更高要求。

术语一般如何实现？如何使用？
Per-group量化实现：(1) GPU上（AWQ, GPTQ, OmniQuant）：INT weight load from memory→INT-to-FP16 dequantization (乘∆+加z)→FP16 Tensor Core GEMM。因per-group dequantization在GEMM循环内执行，无法完全利用Tensor Core流水线，效率低于per-channel量化的纯GEMM。(2) Custom ASIC（BitMoD, FIGNA）：专用per-group dequantization硬件（如BitMoD的bit-serial dequantization unit）在partial sum产生后立即rescaling，避免FP pipeline开销。group size tradeoff: G=64→更低量化误差但2× metadata开销；G=256→更低开销但更高误差。SOTA框架（AWQ, GPTQ）默认G=128。

涉及论文标题：
- 78-BitMoD_Bit-serial_Mixture-of-Datatype_LLM_Acceleration.pdf

## Extended Floating-Point Data Types for Low-Precision Quantization (FP3-EA/ER, FP4-EA/ER)

术语是什么？
Extended Floating-Point Data Types是BitMoD提出的新型低精度浮点数据类型，通过重新利用浮点格式的冗余负零（-0）引入额外量化值来扩展基本FP3/FP4。基本FP格式因符号-幅度表示同时存在+0和-0，在低精度下冗余零占据大量量化层级（如3-bit中占1/8=12.5%）。BitMoD将冗余零替换为预定义特殊值(special value)，构建两类扩展：(1) ER(Extra Resolution/额外分辨率)——特殊值在原始FP数值范围内（如FP3-ER的特殊值±3在FP3范围[-4,4]内），增加分辨率；(2) EA(Extra Asymmetry/额外非对称性)——特殊值超出原始范围（如FP3-EA的特殊值±6超出[-4,4]），引入非对称极值。具体设计：FP3基础值{0, ±1, ±2, ±4}→FP3-ER={±3或±3}替换-0, FP3-EA={±6或±6}替换-0。FP4基础值{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}→FP4-ER={±5}, FP4-EA={±8}。每组权重从4个特殊值(SV_reg)中选1个，与基础FP值组合为量化集合，通过非线性量化（映射每个权重到最近的quantization value）最小化MSE。特殊值选择由per-group encoding metadata (2-bit)记录。关键洞察：per-group权重分布可能对称（Gaussian-like）或非对称（含全部正或全部负的outlier），同时提供ER和EA两种扩展让每组自适选择最优数据类型。

从算法pipeline角度拆解：
以FP3 weight quantizing OPT-1.3B为例（G=128）：
```
# 特殊值候选
basic_FP3 = {0, ±1, ±2, ±4}     # 7个distinct values (含±0)
special_values = {+3, -3, +6, -6}  # FP3-ER + FP3-EA

for each weight group Wg of 128 weights:
    min_MSE = +∞
    for sv in special_values:
        quant_set = basic_FP3 ∪ {sv}  # 替换-0为sv, 得到8个distinct values
        # 非线性量化: 每个权重映射到最近quant_set值
        Wq_tmp = NonLinearQuantize(Wg, quant_set)
        error = MSE(Wg, Wq_tmp)
        if error < min_MSE:
            best_Wq = Wq_tmp
            best_sv = sv
    # 存储: 3-bit quantized weight + 2-bit sv_index + 8-bit INT8 scaling factor

# INT8 per-group scaling factor: 对称量化 scaling factors of a channel
# ∆_fp = max(|Wq|) / (2^3 - 1), ∆ = Round(∆_fp / ∆_channel_scale * 255)
```
NonLinearQuantize: 对每个权重w, 找到quant_set中使|w - q_i * ∆|最小的q_i→Wq_i=q_i。

术语一般如何实现？如何使用？
BitMoD量化在PyTorch上实现（GitHub: https://github.com/yc2367/BitMoD-HPCA-25），量化Llama-2-7B仅需~10秒 on A6000 GPU。硬件上：4个特殊值存储于Special Value Register File (SV_reg)，LLM部署前一次性编程。Bit-serial PE通过LOD解码extended FP值为bit-serial terms（最多2 terms），与FP16 activation做混合精度dot product。特殊值可灵活替换——例如特殊值7可表达为2^3 - 2^0 = 两个bit-serial terms。BitMoD paper Table VIII消融实验证明同时提供ER和EA比单独ER或EA的perplexity更低；Table IX证明{±3, ±6}组合最优。

涉及论文标题：
- 78-BitMoD_Bit-serial_Mixture-of-Datatype_LLM_Acceleration.pdf

## Fine-Grained Data Type Adaptation

术语是什么？
Fine-Grained Data Type Adaptation（细粒度数据类型自适应）是BitMoD提出的per-group量化策略：每个weight group独立从预定义的特殊值集合中选择最优特殊值来扩展基础浮点数据类型，使量化数据类型自适应每组权重的数值分布特性。核心算法（Algo.1）：对每个weight group——遍历所有special values→将special value加入basic FP quantization values→执行非线性量化→计算MSE→选择最小MSE对应的special value。选择结果用2-bit encoding (log2(4) for 4 special values) per group存储。该策略与现有量化优化（AWQ的activation-aware scaling, OmniQuant的clipping threshold optimization）正交兼容——仅替换原始INT-Asym quantizer为BitMoD数据类型的search过程。

从算法pipeline角度拆解：
向量化GPU实现（同时处理所有权重tensor的全部groups）：
```
# Input: W ∈ FP16^{K×D}, precision p ∈ {3,4}
# Output: Wq (quantized), SV_idx (per-group special value index), ∆ (INT8 scaling factor)

# GPU向量化: 所有groups同时处理
all_groups = reshape(W, [-1, G])  # [num_groups, G] where num_groups = K*D/G
basic_vals = GetBasicValues(p)      # shape: [num_basic]
special_vals = GetSpecialValues(p)  # shape: [4]

# Parallel search for all groups
errors = zeros(num_groups, 4)  # 4 special values
for i, sv in enumerate(special_vals):
    quant_vals_i = basic_vals ∪ {sv}  # shape: [num_basic+1]
    # 非线性量化: per element, find nearest quant_val (vectorized with broadcasting)
    Wq_tmp = non_linear_quantize_vectorized(all_groups, quant_vals_i)
    errors[:, i] = mean_squared_error(all_groups, Wq_tmp, axis=1)

best_sv_idx = argmin(errors, axis=1)  # per group optimal special value
# Select quantized values per group based on best_sv_idx
Wq = select_by_index(Wq_all_candidates, best_sv_idx)
∆ = INT8_quantize_per_group_scaling_factors(W, Wq)
```
GPU上Llama-2-7B全部groups的search约10秒(A6000)，因每个group的search完全独立可全并行。

术语一般如何实现？如何使用？
实现：(1) 软件侧——PyTorch vectorized操作，内存开销per group=2-bit(special value encoding)+8-bit(INT8 scaling factor)=10 bits/128 weights=0.078 bits/weight overhead，低于AWQ等INT-Asym的24-bit overhead (16-bit ∆ + 8-bit z per group)。(2) 硬件侧——Special Value Register File (4个可编程SV值)在LLM加载时一次性配置，Bit-serial Term Generator的LOD+比较器在解码时将选定的special value替换冗余负零。设计灵活性：若新LLM需要不同special values，只需更新SV_reg和term decoder的微小修改（如special value 7表达为23-20）。

涉及论文标题：
- 78-BitMoD_Bit-serial_Mixture-of-Datatype_LLM_Acceleration.pdf

## Collaboration-of-Experts (CoE) Model

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Collaboration-of-Experts (CoE) 或 Composition of Experts 是一种多专家模型架构，由多个独立训练的 expert 模型和一个 routing module 组成。与 Mixture-of-Experts (MoE) 不同，CoE 的每个 expert 是独立的完整模型（可各自训练、微调、部署和移除），而非共享同一模型内的 expert sub-layer。Routing module 可手动配置规则或独立训练，决定了输入经过哪些 expert 处理的 inference chain。CoE 的专家协作模式是链式的：输入 → Routing module selects preliminary expert → 执行推理 → 输出决定 next expert selection（如 circuit board inspection 中先由分类 expert 判断 defect，再由 object detection expert 验证 alignment）→ 最终输出。CoE 的核心优势：(1) 可达到单模型无法达到的精度（如电路板检测 99.9% accuracy vs 单模型 <92%）；(2) 各 expert 可独立训练和微调，无需联合优化；(3) expert 可灵活增减；(4) routing 可预定义 → 提供 expert usage probability 的先验知识用于 serving 优化。CoE 的核心挑战：大量 experts 导致内存需求大（如 300+ experts, 60GB+），在资源受限设备上需频繁 expert switching。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CoE 模型的推理 pipeline（以 circuit board defect detection 为例）：
```
输入: circuit board component image I
输出: defect detection result (pass/fail + alignment status)

1. Routing Module(I) → selects preliminary expert E_classify
   // E_classify 是 ResNet101 classification expert，针对特定组件类型训练
2. result_1 = E_classify(I)  // forward pass: ResNet101(I) → class logits
3. if result_1 == "no_defect":
      Routing Module(result_1, component_type) → selects E_detect
      // E_detect 是 YOLOv5m/l object detection expert
      result_2 = E_detect(I)  // forward pass: YOLOv5m(I) → bounding boxes + class
      if result_2.alignment_ok and result_2.solder_direction_correct:
          return "PASS"
      else:
          return "FAIL: alignment/soldering issue"
   else:
      return "FAIL: defect detected by E_classify"
```

对比 MoE pipeline（如 Mixtral-MoE）：
```
输入: token embedding h
输出: next-token logits

For each MoE layer in [0, N_layers):
    1. Gate(h) → selects top-k experts (e.g., top-2 from 8)
    2. For each selected expert_i:
         expert_out_i = ExpertFFN_i(h)  // parallel execution
    3. h = sum_i(gate_weight_i × expert_out_i)  // weighted sum
    4. h += SelfAttention(h)
return h  // 经过所有层后的 hidden state
```

CoE vs MoE 的关键算法差异：
- CoE: sequential expert chain, output of one expert determines next expert selection
- MoE: parallel expert execution per layer, gate selects experts, outputs are merged by weighted sum
- CoE: routing can be pre-defined → usage probability known a priori
- MoE: routing determined at runtime → only historical statistics available for prediction
- CoE: experts are independent models (different architectures possible: ResNet + YOLO)
- MoE: experts share the same architecture within a model (e.g., all are FFN variants)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CoE 的实现包括：(a) **Routing Module 实现方式**：(i) 用户预定义规则（如 if component_type == "A" use Expert_A），直接可推导 usage probability；(ii) 独立训练的 router（轻量级分类器，输入为上一 expert 的输出或原始输入），需在 sample data 上评估得到 usage probability；(iii) 查找表（Lookup Table）方式，按 component- expert 映射。 (b) **Expert 模型独立性**：每个 expert 是完整的 PyTorch/TensorFlow 模型，有独立的 state_dict 和 forward pass，可存放在各自的 checkpoint 文件中。CoE 不需要联合训练。 (c) **代表系统**：Samba-CoE [MICRO 2024]（大规模 CoE 部署）、CoServe [ASPLOS 2025]（边缘设备 CoE serving）、Qihoo 360 CoE（code/math/law 多领域 CoE）、Bench-CoE [2024]（CoE benchmark 框架）。应用场景：精密制造（99.9% 检测准确率）、多领域知识问答（code/math/law 专家协作）、自动化决策系统。CoE 不适用于需要所有 experts 联合训练以学习路由策略的场景——这种场景下 MoE 更合适。CoE 和 MoE 可互补使用，如 CoE routing 的 preliminary expert 本身可以是 MoE 模型。

涉及论文标题：
- 80-CoServe- Efficient Collaboration-of-Experts (CoE) Model Inference with Limited Memory.pdf


## Microscaling (MX) Data Format for Foundation Model Quantization

术语解释
Microscaling (MX) 是 Open Compute Project (OCP) 标准化的 block data representation (BDR) 格式，通过多级共享scale factor实现高效的组量化。MicroScopiQ论文中使用MX-FP格式量化离群值(outliers)、MX-INT格式量化内点(inliers)，两者结合实现FMs的高精度低比特量化。

术语是什么？
MX数据格式由四个组件定义：(1) scale factors (level-1, level-2)，(2) 数据格式类型 τ (INT/FP)，(3) bit-width b，(4) group sizes (k1, k2)。MX的核心思想是在block级别共享scale信息以降低metadata开销。
- **MX-FP-b_{k1,k2}**: 采用两级scaling——level-1 scale factor (power-of-two, 2^{O_sf_l1}) + level-2 microExponent (μX，即共享的FP exponent field)。例如MX-FP-4_{8,8}表示4-bit FP元素，每8个元素共享level-1和level-2 scale。
- **MX-INT-b_{k1}**: 仅采用单级scaling（INT无exponent field），类似标准INT group quantization但使用E8M0格式的scale factor。例如MX-INT-2_{128}表示2-bit INT元素，每128个元素共享scale factor。
- **Level-2 microExponent (μX)**: MX-FP独有的概念——提取μB内所有FP值的公共指数作为level-2 scale共享，等效于FP格式的shared exponent field。MicroScopiQ利用此特性将FP-outlier转换为INT格式在homogeneous INT PE中处理（通过提取μX将FP mantissa映射为INT可处理的值）。

与标准Block Floating Point (BFP)的关系：MX是BFP的标准化扩展，支持多级scaling和多种数据类型(INT/FP)。

从算法pipeline角度拆解：
MicroScopiQ中MX量化的完整流程（以MX-INT-2_{128} inlier + MX-FP-4_{8,8} outlier为例）：

```
# Step 1: MX-INT Inlier Quantization (per MaB, B_M=128)
I_sf = log2(max(|W_in|)) - log2(max_INT_2)  # max_INT_2 = 1 (2-bit symmetric)
# I_sf 始终为负的 2 的幂 → 2^{I_sf} (power-of-two scale)
Q_in[i] = clip(round(W_in[i] / 2^{I_sf}), -1, 1)  # MX-INT-2_{128}, 每128个inlier共享I_sf

# Step 2: Outlier Pre-processing (利用I_sf)
W_out_scaled = W_out * 2^{I_sf}  # 预缩小动态范围

# Step 3: MX-FP Outlier Quantization (per μB, B_μ=8)
# Level-1 scale: O_sf_l1 = log2(max(|W_out_scaled|)) - log2(max_FP_4)
# FP4 format: e1m2 (1-bit exponent, 2-bit mantissa)
for each outlier w in μB:
    Q_out[w] = quantize_FP4(w / 2^{O_sf_l1})  # {sign, exp(1b), mantissa(2b)}

# Level-2 microExponent: 提取μB内所有outlier的公共指数
μX = common_exponent(Q_out in μB)  # shared exponent field

# Final outlier scale factor
O_sf = O_sf_l1 + μX - I_sf  # 包含预缩放补偿
```

术语一般如何实现？如何使用？
- **OCP标准**: MX格式由微软、Intel、NVIDIA等支持，OCP发布MX V1.0 Spec (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)。
- **MicroScopiQ使用**: 离群值用MX-FP-4_{8,8} (或MX-FP-8_{8,8})保持高精度(2× inlier bit-width)，内点用MX-INT-2_{128} (或MX-INT-4_{128})实现极致压缩。关键洞察：inlier scale factor (2^{I_sf}) 始终为负的2的幂→可预缩放离群值降低量化难度。
- **MX-FP到INT的转换**: MicroScopiQ通过提取μX将FP格式离群值转换为INT格式(sign+mantissa halves)→在homogeneous INT PE阵列中处理→避免复杂的FP PE设计。
- **与现有MX8 entry的区别**: MX8 (Pimba)侧重MX8+随机舍入用于LLM state量化的特定配置；MicroScopiQ展示MX格式的另一种用法：MX-FP用于离群值(高动态范围)+MX-INT用于内点(极致压缩)，并通过μX特性实现INT PE的FP处理。

涉及论文标题：
- 84-MicroScopiQ- Accelerating Foundational Models through Outlier-Aware Microscaling Quantization.pdf
- 88-Amove- Accelerating LLMs through Mitigating Outliers and Salient Points via Fine-Grained Grouped Vectorized Data Type.pdf
- **Amove对MX的扩展**: Amove在MX格式shared scale factor基础上额外引入shared residual和per-cluster encoding（2-bit），形成Fine-Grained Grouped Vectorized Data Type。每个coarse-grained group包含：S_base(FP8)+R(FP8)+K/C个encoding(2b)+K个quantized elements(INT4/FP4)。通过residual approximation mechanism实现fine-grained cluster-wise scale recovery: `S_ci = S_base - R * E_ci`，在不存储per-cluster scale factor的情况下达到0.25~1 bit/value的scale factor overhead（vs MX 2 bit/value at g=4）。Amove灵活配置group size (4/8/16/32/128)、cluster size (4/16)和encoding bit-width (2 bit)，在NVIDIA Ampere GPU tensor core和systolic array上实现W4A4全量化（含attention层），area overhead <2%。

## Outlier Bit Redistribution via Hessian-guided Structured Pruning

术语解释
MicroScopiQ提出的核心技术：通过Hessian-guided剪枝将离群值(outlier)额外的比特位重新分布到被剪枝的内点(inlier)位置，在保持每个tensor元素统一bit-budget和数据格式的同时实现离群值的高精度量化。

术语是什么？
这是MicroScopiQ解决Group A (高精度但非对齐内存+复杂PE)和Group B (对齐内存但低精度)矛盾的核心机制：
1. 离群值以2×精度量化(如4-bit MX-FP)产生额外比特位(e.g., 2 extra bits per outlier)
2. 在微块(μB=8)内通过Hessian矩阵(H^{-1} = (2XX^T + λI)^{-1})识别n个最不重要的内点位置
3. 将离群值的额外比特位(如{s,m1}和{s,m0}两半)分布到剪枝的内点位置
4. 每个μB形成(B_μ-n):B_μ structured pruning pattern → 统一bit-budget per element → aligned memory

从算法pipeline角度拆解：
MicroScopiQ Outlier Bit Distribution 伪代码（Algorithm 1, Step 2-3）：

```
# Input: μB weights W_μB (B_μ=8), Hessian H^{-1}
# Output: Quantized weights + permutation list

# Step 2.0: Count outliers in μB
n = min(B_μ/2, NumOutliers(W_out_μB))  # max 4 outliers per μB=8

# Step 2.2: Hessian-guided pruning (选择n个最不重要的inlier)
M = {}  # pruned position list
for it in range(n):
    # 基于Hessian的saliency metric: w_p^2 / [H^{-1}]_{pp}
    p = argmin_{p in W_in_μB, p not in M} (w_p^2 / [H^{-1}]_{pp})
    w_p = 0  # prune to 0
    M = M ∪ {p}

# Step 2.5: Quantize outliers at 2x precision (MX-FP-4_{8,8})
# Each outlier: {sign(1b), exponent(1b), mantissa(2b)}
# After sharing μX: each outlier has {sign(1b), m1(1b), m0(1b)}
# Split into two 2-bit halves mimicking MX-INT structure:
#   Upper half: {sign, m1}
#   Lower half: {sign, m0}

# Step 3: Distribute outlier LSBs to pruned inlier positions
# For μB with n=1 outlier at position 3:
#   Position 0: inlier MX-INT-2
#   Position 1: inlier MX-INT-2
#   Position 2: pruned → receives {s, m0} (Lower half)
#   Position 3: outlier → keeps {s, m1} (Upper half)
#   ... remaining positions: inliers
# permutation_list[μB] = {(3,2)}  # tracks (outlier_pos, lower_half_pos)
```

关键设计决策：
- **Hessian而非magnitude**: 使用w_p^2/[H^{-1}]_{pp} metric而非简单的|w|——这确保剪枝对最终量化误差影响最小（低saliency权重剪枝后通过Hessian update补偿误差）
- **(B_μ-n):B_μ而非固定N:M**: 剪枝数量n根据μB内的实际离群值数量动态决定→更灵活→更高精度
- **共享scale的层次设计**: inlier scale (2^{I_sf}) per MaB(128)→粗粒度共享→低metadata；outlier scale (O_sf_l1 + μX) per μB(8)→细粒度共享→低量化误差

术语一般如何实现？如何使用？
- **实现**: PyTorch calibration-based PTQ, 单张H100 GPU执行, 量化时间30min–9h (3.8B–175B参数)
- **EBW计算**: 若μB无离群值→EBW=b_b(2或4)；若有离群值→EBW=(perm_bits + 2×B_μ + O_sf_bits)/B_μ, e.g., (24+16+8)/8=6 bits per μB with outliers
- **可集成性**: 与OmniQuant的LWC (Learnable Weight Clipping)和LET (Learnable Equivalent Transformation)正交组合→Omni-MicroScopiQ→额外精度提升up to 22%
- **与OliVe的对比**: OliVe剪枝离群值相邻的inlier作为identifier→依赖outlier locality假设(modern FMs不成立)→significant accuracy drop；MicroScopiQ通过Hessian选择剪枝位置→不依赖locality→preserve all outlier values
- **与SDQ的对比**: SDQ也组合剪枝+量化但使用固定N:M pattern和分离的inlier/outlier向量→unaligned memory；MicroScopiQ在统一bit-budget内完成分布→aligned memory

涉及论文标题：
- 84-MicroScopiQ- Accelerating Foundational Models through Outlier-Aware Microscaling Quantization.pdf

## Effective Bit Width (EBW) for Quantized Foundation Models

术语解释
Effective Bit Width (EBW) 是衡量量化模型实际存储开销的指标，计算每个权重参数平均使用的比特数（包含所有metadata开销：scale factors, permutation lists, outlier identifiers等）。与名义bit-width (b_b)不同，EBW反映模型的真实压缩率。

术语是什么？
EBW = (总存储比特数) / (总参数量)。对于量化方案，metadata（scale factor, zero-point, permutation list等）和不同精度的混合使用使得实际每参数比特数高于名义精度。MicroScopiQ的EBW计算考虑了：
- Inlier data bits (per element)
- Outlier data bits (per element, 2× inlier precision)
- Permutation list bits (per μB with outliers)
- Inlier scale factor bits (per MaB, shared across 128 elements → negligible per-element)
- Outlier MXScale bits (per μB with outliers)
- Outlier presence identifier (1-bit per μB → negligible)

从算法pipeline角度拆解：
MicroScopiQ的EBW计算公式（Equation 4）：

```
# 对于有l层的FM，每层有m个μB，x%的μB含离群值

# 无离群值的μB EBW
EBW_I = b_b  # e.g., 2 bits for MX-INT-2

# 有离群值的μB EBW (以b_b=2, B_μ=8, MXScale=8b, perm_list=24b为例)
EBW_O = (perm_bits + 2*B_μ + O_sf_bits) / B_μ
      = (24 + 16 + 8) / 8 = 6 bits

# FM整体EBW
EBW_FM = Σ_{i=1}^{l} (x·m·EBW_O + (1-x)·m·EBW_I) / m / l
       = x·EBW_O + (1-x)·EBW_I

# MicroScopiQ vs baselines:
# b_b=2: EBW ≈ 2.36 bits (取决于模型离群值比例)
# b_b=4: EBW ≈ 4.15 bits
# GOBO (group A): EBW = 15.6–18.17 bits (离群值保留FP32全精度)
# OliVe (group B): EBW = b_b (所有metadata软件管理)
# 其他算法baselines (GPTQ, AWQ, OmniQuant): EBW = b_b
```

关键EBW组成 (MicroScopiQ W2A16, B_μ=8):
- Per-μB metadata (仅含离群值的μB): permutation list (24b = 6b/element × 4 pairs) + MXScale (8b) = 32b per μB
- Per-MaB metadata (所有MaB): inlier scale factor (8b, 共享于128 elements → ~0.06b/element)
- Outlier presence identifier: 1b per μB (→ ~0.125b/element, 通常忽略)

术语一般如何实现？如何使用？
- **报告惯例**: 遵循GPTQ [Frantar et al., 2022]和Atom [Zhao et al., 2024]的EBW报告方式
- **动态变化**: MicroScopiQ的EBW随模型和层的离群值比例动态变化——离群值更多的模型/层具有更高的EBW
- **权衡**: EBW_I = b_b (最优压缩) vs EBW_O (含metadata)→通过调整B_μ控制：小B_μ增加metadata overhead但降低量化误差；大B_μ减少metadata但增加量化误差；MicroScopiQ在B_μ=8找到最优平衡点
- **实际意义**: EBW 2.36 bits表示MicroScopiQ在W2A16配置下达到约6.8× compression vs FP16 (16/2.36)，而GOBO仅约1.0× (16/15.6)，OliVe约4.0× (16/4)但在微低bit-width精度不可接受

涉及论文标题：
- 84-MicroScopiQ- Accelerating Foundational Models through Outlier-Aware Microscaling Quantization.pdf

---

## Einsum Notation for Sparse Tensor Algebra

术语解释
Einsum（Einstein Summation）notation是一种紧凑的数学表示法，用于描述多维张量上的代数运算。它源自爱因斯坦求和约定（1916），在现代稀疏张量代数中被广泛采用以清晰表达多维迭代空间上的contraction操作。

术语是什么？
Einsum notation以"output_indices = input_operands"的形式简洁表达张量运算。例如，SpMSpM表示为 Cij = Aik × Bkj over (I, J, K)，每个张量标注有参与计算维度的下标：i/j为output维度，k为contracted dimension（在A和B中均出现但在C中不出现，其值在迭代中聚合）。Einsum比传统的矩阵乘法符号（C=A×B）更精细，因为它明确揭示了可能的循环顺序选择——这是sparse accelerator dataflow设计的核心输入。

关键概念：
- **Output indices**：出现在output tensor中的维度（如i, j for Cij）
- **Contracted dimension**：出现在多个输入中但不在output中的维度（如k），其值通过sum/max等聚合
- **Iteration space**：(I, J, K)多维空间→compute遍历此空间
- **Point**：张量中一个元素，由坐标元组确定（如Xijk at (i,j,k)），包含零值或非零值

从算法pipeline角度拆解：
以SpMSpM Cij = Aik × Bkj over (I, J, K)为核心，Einsum揭示不同kernel variant的维度角色：
1. **SpMSpM (S×S)**：A(I×K)稀疏, B(K×J)稀疏→C(I×J)稀疏。Einsum="ik,kj->ij"
2. **FT×F**：Cjj' = Fij × Fij' over (J, J', I)→tall-skinny matrix self-multiplication。Einsum="ij,ij'->jj'"
3. **F×D (SpMM)**：Cij = Fik × Dkj over (I, J, K)→B是dense matrix。Einsum="ik,kj->ij"（同上但B dense）
4. **FT×S (MS-BFS)**：Csource,j = Fsource,i × Sij over (J, I)→sparse source vector × sparse graph matrix。Einsum="ij,ij->j"（with i contracted）

Dataflow选择基于Einsum的维度排列：IP(i⊲j⊲k)时k在innermost做内积，OP(k⊲i⊲j)时k在outermost做外积，Gust(i⊲k⊲j)时k在中间做hybrid。

术语一般如何实现？如何使用？
- **框架API**：NumPy `np.einsum("ik,kj->ij", A, B)`、PyTorch `torch.einsum("ik,kj->ij", A, B)`→自动推导optimal contraction order→GPU backend映射到cuBLAS kernel
- **加速器设计**：ExTensor[12]通过Einsum定义通用sparse tensor algebra的iteration space→自动生成intersector/merger硬件配置；HYTE遵循Einsum→推导per-dataflow的A/B/C复用特性（Table 1）→指导tiling scheme搜索
- **与tiling关系**：Einsum定义全迭代空间→tiling将此划分为逻辑子空间（tiles）→每个tile内的Einsum计算一样但数据量限制在buffer容量内

涉及论文标题：
- 85-HYTE- Flexible Tiling for Sparse Accelerators via Hybrid Static-Dynamic Approaches.pdf

---

## Fiber in Sparse Tensor Formats

术语解释
Fiber（纤维）是稀疏张量压缩存储格式中的基本结构单元。在CSR/CSC等分层压缩格式中，张量维度被组织为包含fibers的层次结构（fiber tree[35]）。一个fiber是固定所有维度坐标仅保留一个维度变化而得到的一维sub-array。

术语是什么？
在稀疏张量文献[35]中，fiber定义为张量中固定除一个维度外所有坐标后沿该维度的元素序列。例如在3D张量X中固定(i,k)后沿j维度的所有元素构成一个j-fiber；在SpMSpM中，CSR格式A张量的每行是i-fiber（固定i，沿k维度的非零元素列表）。

关键属性：
- **Position vs Coordinate**：position是fiber中某点在compressed存储中的实际存储位置（offset in CSR arrays），通常与坐标不同（因零值compression和null pointer removal）
- **Fiber segment**：tiling将维度空间切分后，fiber被tile边界截断产生的部分→例如CSR的一行被Tk切成多个k-segments
- **Fiber metadata**：每个segment需记录begin position用于tile间高效跳转

从算法pipeline角度拆解：
以CSR格式SpMSpM中fiber的流转为例（A: CSR row沿k, B: CSR row沿j）：
1. **CSR组织**：A的ptr[i]→ptr[i+1]标记第i个i-fiber的在cols[]/vals[]中的[begin,end)范围。fiber内每个元素= (k坐标, 值)
2. **Tiling截断**：HYTE tiling controller发tile Ti×Tk→accessor fetch Ti个i-fibers→每个被Tk截断为segment（仅取cols在[tile_k_begin, tile_k_begin+Tk)内的元素）→产生Ti个fiber segments→每个segment的begin position记录为metadata
3. **Format-access direction mismatch**：当inter-tile order与CSR format方向不匹配时（如i⊲j⊲k order下B以CSR row沿k存储但需沿j遍历columns）→fetch沿k方向被截断的fiber需要更多metadata（O(K) positions）→HYTE的cost model自动权衡tiling收益与metadata cost

术语一般如何实现？如何使用？
- **Compressed formats**：CSR以ptr+cols+vals三数组存储；CSC类似以column为主；COO以(row,col,val)三元组；Block CSR以固定block（32×32）为粒度→fiber概念扩展到block-fiber
- **Fiber tree**[35]：将任意维度张量抽象为fiber tree层次结构→每level的fiber包含下一level fibers序列→根节点=整个张量、叶子=非零元素值→支持任意维度通用压缩
- **硬件的fiber处理**：HYTE hardware accessor以fiber segment（非单个element）为粒度做buffer管理→扩展自Buffets[30]的element粒度→每个segment的metadata在buffer中与data协同管理

涉及论文标题：
- 85-HYTE- Flexible Tiling for Sparse Accelerators via Hybrid Static-Dynamic Approaches.pdf

## Residual Approximation Mechanism

术语是什么？
Residual Approximation Mechanism是Amove提出的细粒度量化scale factor压缩方法。核心思想：利用细粒度group-wise量化下scale factor分布的light-tailed特性（大多数LLM kurtosis < 3，即分布集中在均值附近、极端值稀少），用粗粒度group的shared base scale factor + 1个shared residual + per-cluster compact encoding（2-bit）来近似每个fine-grained cluster的真实scale factor，避免存储per-cluster scale factor。对cluster i的scale factor恢复公式：`S_ci = S_shared - R * E_ci`，其中S_shared是base scale factor（取group内所有cluster scale的最大值），R是shared residual，E_ci是per-cluster encoding（取值于[-2^{E-1}-1, 0]）。Residual对权重采用offline search-based MSE最小化（搜索范围[-1,1]步长0.01），对激活采用online average deviation（平均每个cluster scale与base scale的偏差），支持online quantization和校准分布偏移。配置参数：group size K、cluster size C、encoding bit-width E。Scale factor memory overhead公式：`(R_bits + S_bits)/K + E_bits/C` bits per value。

从算法pipeline角度拆解术语：
该机制是Amove量化框架的核心压缩算法，运作于quantize阶段：
```
// Input: FP16 tensor to quantize, group_size G, cluster_size C, encoding_bits E
for each coarse-grained group of G elements:
    // Step 1: Partition into K = G/C clusters
    clusters[] = partition(group, C)

    // Step 2: Compute per-cluster true scale factors (symmetric integer)
    for each cluster c_i:
        Δ_i = max(|c_i|) / (2^(b-1) - 1)

    // Step 3: Select base scale (largest cluster scale)
    Δ_base = max(Δ_1, ..., Δ_K)

    // Step 4: Compute residual R
    if is_activation:  // online: average deviation
        R = (1/(C*E)) * Σ|Δ_i - Δ_base|
    else:  // weight (offline): search-based MSE minimization
        R = argmin_R Σ(Δ_i - (Δ_base - e_i * R))^2
        // e_i = floor((Δ_i - Δ_base) / R), e_i ∈ [-2^{E-1}-1, 0]
        // search R ∈ [M, N] with step Q (e.g., [-1, 1], step 0.01)

    // Step 5: Quantize with approximate cluster scales
    for each cluster c_i:
        S_ci = Δ_base - R * E_ci           // approximate scale factor
        X_q = round(c_i / S_ci)             // quantize to INT4
        X_hat = X_q * S_ci                  // dequantize back

    // Pack: FP8(Δ_base) | FP8(R) | encodings(K×2b) | INT4 elements(K×C)
```

术语一般如何实现？如何使用？
在PyTorch中通过向量化操作实现：weight residual在offline calibration一次性搜索完成，activation residual在inference时online计算（极低overhead）。Amove-Aggressive配置（linear: G=128, C=16, E=2b → overhead=0.25 bit/value；attention: G=32, C=4, E=2b → overhead=1 bit/value）。Amove-Conservative配置（G=32, C=4 uniform → overhead=1 bit/value）。对light-tailed分布假设的鲁棒性：当模型kurtosis > 3时（如Bloom-3B kurtosis≈3.31），通过减小residual group size（128→32→16→4）仍能有效建模非理想分布。与VS-Quant的per-vector scale factor量化不同，Amove避免了scale factor本身量化导致的>50% accuracy loss。

涉及论文标题：
- 88-Amove- Accelerating LLMs through Mitigating Outliers and Salient Points via Fine-Grained Grouped Vectorized Data Type.pdf

## Fine-Grained Grouped Vectorized Data Type (Amove Data Type)

术语是什么？
Amove提出的Fine-Grained Grouped Vectorized Data Type是一种支持LLM细粒度W4A4和低比特weight-only量化的向量化数据格式，在MX (Microscaling)格式基础上扩展。数据格式结构：`[Base Scale (FP8 E4M3) | Residual (FP8 E4M3) | K/C encodings (2-bit each) | K elements (INT4/FP4/scalar)]`，其中K是group size，C是cluster size。关键特性：(1) 通过residual+encoding机制而非单一shared scale恢复fine-grained cluster scale: `S_ci = S_shared - R * E_ci`；(2) 支持灵活配置group size (4/8/16/32/128)、cluster size (4/16)和encoding bit-width (2)；(3) Element字段可集成任意scalar data type（INT4、FP4、M-ANT等）；(4) Byte-aligned memory access设计（如W3A3: 32×3b+8b base+8b residual+8×2b encoding=128b per group）；(5) 支持weight-activation (W4A4)和weight-only (W3A16/W2A16)双模式。

从算法pipeline角度拆解术语：
```
// Quantize: FP16 matrix → Amove data format
Amove_Quantize(W_fp16, group_size=G, cluster_size=C, encoding_bits=E):
    for each group g in W_fp16:
        Δ_base = max(compute_cluster_scales(group, C))
        R = compute_residual(group, Δ_base, C, E)  // see Residual Approx.
        for each cluster i in 0..(G/C-1):
            Δ_i = compute_cluster_scale(cluster_i)
            E_i = floor((Δ_i - Δ_base) / R)  // ∈ [-2^{E-1}-1, 0]
            S_i = Δ_base - R * E_i
            for each element e in cluster_i:
                e_q = round(e / S_i)  // → INT4
        pack(FP8(Δ_base), FP8(R), E_0..E_{G/C-1}, Q_0..Q_{G-1})

// Dequantize: per-cluster scale recovery + scale to FP16
Amove_Dequantize(packed_group):
    Δ_base, R, E[], Q[] = unpack(packed_group)
    for each cluster i:
        S_i = Δ_base - R * E[i]
        X_hat[i] = Q[i] * S_i  // INT4 * FP16 → FP16
```

术语一般如何实现？如何使用？
在PyTorch中实现为modular data representation layer，可无缝插入GPTQ、AWQ、OmniQuant的量化pipeline（替换scalar weight format为Amove vectorized format，无需修改core logic）。Configuration: Amove-Aggressive（linear G=128/C=16 → 0.25b/value; attention G=32/C=4 → 1b/value）追求低overhead；Amove-Conservative（G=32/C=4 uniform → 1b/value）追求更高精度。集成M-ANT时在每个cluster内使用M-ANT数学自适应格式替代INT4→perplexity降低约20% at equal overhead。

涉及论文标题：
- 88-Amove- Accelerating LLMs through Mitigating Outliers and Salient Points via Fine-Grained Grouped Vectorized Data Type.pdf

## Cluster-wise Quantization

术语是什么？
Cluster-wise Quantization是组量化(group-wise quantization)的进一步细化：将每个coarse-grained group（如128或64个元素）再划分为更小的cluster（如4或8个元素），每个cluster拥有独立的scale factor。比group-wise精度更高（更细粒度捕获局部数据分布），但引入更多scale factor memory overhead（例如group size=4时overhead=16/4=4 bits/value vs group size=128时仅0.125 bits/value）。在Amove中，cluster-wise的scale factor不直接存储，而是通过Residual Approximation Mechanism用`S_ci = S_base - R * E_ci`近似恢复，实现cluster-wise精度但大幅降低存储开销。

从算法pipeline角度拆解术语：
```
// Comparison of granularity levels:
// Group-wise (g=64):  [group 0: 64 elem | scale Δ_0(FP16)]
//                      Overhead: 16/64 = 0.25 bits/value
// Cluster-wise (g=64, c=4):
//   [group 0: cluster0:4elem|Δ_0_0(FP16), cluster1:4elem|Δ_0_1(FP16), ...]
//   Overhead: 16/4 = 4 bits/value (16 clusters per group)
// Amove cluster-wise (g=32, c=4):
//   [group: FP8(Δ_base)|FP8(R)|8×2b encoding|32×INT4 elem]
//   Overhead: (8+8)/32 + 2/4 = 0.5 + 0.5 = 1 bit/value
//   Scale recovery: S_ci = Δ_base - R * E_ci
```

术语一般如何实现？如何使用？
Cluster-wise量化在VS-Quant、FineQ等工作中被使用，但直接存储per-cluster scale factor导致memory overhead过高。Amove通过residual approximation避免了该问题，在保持cluster-wise精度优势的同时将overhead降至0.25~1 bit/value。硬件层面，Amove的GIPU和scale factor decoder直接支持vectorized dequantization at cluster size=4。

涉及论文标题：
- 88-Amove- Accelerating LLMs through Mitigating Outliers and Salient Points via Fine-Grained Grouped Vectorized Data Type.pdf

## Product Quantization (PQ) for KV Cache

术语解释
Product Quantization for KV Cache（乘积量化KV缓存）是将Product Quantization（PQ）[Jégou 2011, TPAMI]应用于LLM推理中KV cache压缩的技术。PQ将高维key/value向量沿head_dim维度分解为m个子向量→每个子向量空间独立执行k-means聚类（K个centroids）→用centroid index（log₂K bit）替换原始子向量，实现从FP16(16-bit)到log₂K/m bit的压缩。与element-wise量化（per-element scaling+rounding）不同，PQ是vector-level聚类量化，能自适应数据分布中的locality和similarity。

术语是什么？
PQ是一种vector quantization技术，最初用于近似最近邻搜索（ANNS）。其核心思想：(1) Vector Splitting：将D维向量分解为m个d/m维子向量（d/m为subvector dimension），各子向量空间独立处理；(2) Clustering-based Quantization：每个子向量空间用k-means聚类生成K个centroids（codebook），原始子向量由最近centroid的index（log₂K bit）表示。codebook大小=m×K×d/m×2B，indices大小=m×N×log₂K/8 bit。对于d=128, m=32, K=512的典型配置（AQPIM），压缩率约6.53×（vs FP16的每token 4d bytes）。PQ的codebook生成需要O(K·N)的聚类迭代，带宽需求大，传统上被认为不适合online inference使用。AQPIM的关键insight是将PIM的高内部带宽反向赋能给PQ聚类，使online PQ变得可行。

从算法pipeline角度拆解：
AQPIM的PQ用于KV cache量化流程：
```
# Step 1: Vector Splitting (per head)
K_head = [N, d_head]  # Key matrix for one head
V_head = [N, d_head]  # Value matrix for one head
m = 32  # subvectors
K_splits = K_head.reshape(N, m, d_head/m)  # [N, m, d/m]
V_splits = V_head.reshape(N, m, d_head/m)

# Step 2: Per-subvector k-means clustering (4 iterations)
for s in range(m):
    # Distance computation (BankPE, all banks parallel)
    dist[n,k] = ||K_splits[n,s,:] - centroids[s,k,:]||^2
    
    # Cluster assignment (BufferPE, argmin)
    assignment[n] = argmin_k(dist[n,:])
    
    # Centroid update (importance-weighted, see separate term)
    for k in range(K=512):
        members = {n: assignment[n]==k}
        centroids[s,k] = Σ(w[n]*K_splits[n,s,:] for n in members) / Σ(w[n] for n in members)

# Step 3: Store compressed format
key_codebook = centroids  # [m, K, d/m]  FP16
key_indices = assignment  # [m, N]  int (log2K=9 bit)
# Compression ratio: (m*K*d/m*2 + m*N*9/8) / (N*d*2) ≈ K*d*2/(N*d*2) + m*9/(8*d*2) ≈ 6.53x
```
Codebook生成在prefilling阶段完成，与GPU的attention计算并行（聚类O(K·N) vs attention O(N²)），被完全隐藏。

术语一般如何实现？如何使用？
PQ deployment有三种模式：(1) CPU offloading retrieval（PQCache [arXiv 2024]）：CPU存储全量KV，用PQ做近似检索找到relevant tokens后fetch original KV到GPU，非pure quantization；(2) GPU online PQ（SqueezedAttention [arXiv 2025]）：GPU上在线聚类量化KV，用于sparse attention的token selection，保留full KV copy；(3) PIM online PQ（AQPIM [HPCA 2026]）：纯PIM quantization，无full KV copy，quantized后直接在PIM中计算attention，完全消除CPU/GPU KV传输。AQPIM是首个将PQ既用做compression又直接作为attention计算source的方案。PQ压缩率可灵活调节——通过改变m(子向量数)和K(centroid数)控制（m=32, K=512为default），无需硬件修改。

涉及论文标题：
- 89-AQPIM_Breaking_the_PIM_Capacity_Wall_for_LLMs_with_in-Memory_Activation_Quantization.pdf

---

## Importance-Weighted K-Means Clustering

术语解释
Importance-Weighted K-Means Clustering（重要性加权K均值聚类）是AQPIM提出的对标准k-means的改进：在聚类目标函数中加入token重要性权重（来自attention scores），使对模型accuracy影响大的token获得更小的量化误差。

术语是什么？
标准k-means平等对待所有token，但LLM attention中不同token的重要性不同——sink tokens和recent tokens consistently receive higher attention scores [StreamingLLM, SnapKV]。AQPIM的importance-weighted k-means将attention scores转化为聚类权重：w = sum(S[-t:, :], axis=0)，其中S是attention score matrix (N×N)，t=32是recent tokens窗口。聚类时，centroid更新变为加权平均：μ_k = Σ(w_n·x_n)/Σw_n (n∈C_k)。这意味着高attention tokens"拉近"centroid，使其量化误差更小。实验表明在高压缩场景（K=128 centroids）下，weighting贡献显著accuracy提升（+6.82 avg score vs standard PQ on LongBench）。

从算法pipeline角度拆解：
```
# 输入: attention scores S[N, N] (prefilling时GPU计算)
#       key vectors K[N, d]
# 输出: weighted centroids

# Step 1: 计算token重要性权重
w = sum(S[-32:, :], axis=0)  # 最后32个token对各position的attention score之和, shape [N]

# Step 2: Weighted k-means (4 iterations)
for iter in range(4):
    # E-step: 加权距离分配
    for n in range(N):
        for k in range(K):
            dist[k] = ||K[n] - centroids[k]||^2  # 距离本身不加权
        assignment[n] = argmin(dist)
    
    # M-step: 加权centroid更新
    for k in range(K):
        C_k = {n: assignment[n]==k}
        W_k = sum(w[n] for n in C_k)  # 分母在BufferPE算倒数
        centroids[k] = sum(w[n]*K[n] for n in C_k) / W_k  # 加权平均
```
权重计算复用attention计算中的S矩阵（FlashAttention已生成），overhead极小。

术语一般如何实现？如何使用？
权重w的计算在GPU上完成（与attention计算共用S矩阵），然后offload到PIM供k-means使用。分母W_k的计算利用BufferPE的DIV单元取倒数→传给BankPE做一次乘法（退化为MUL），避免BankPE新增除法器。该技术专门针对KV cache场景：attention scores直接反映token重要性，无需额外calibration。通用场景下可使用gradient-based importance、frequency-based importance等替代方案。

涉及论文标题：
- 89-AQPIM_Breaking_the_PIM_Capacity_Wall_for_LLMs_with_in-Memory_Activation_Quantization.pdf

---

## Channel Sorting for PQ Vector Splitting

术语解释
Channel Sorting for PQ Vector Splitting（PQ子向量划分的通道排序优化）是AQPIM的一种预处理技术：在将向量分解为子向量前，先按channel间cosine similarity对channel进行重排序分组，使高相关性的channel进入同一子向量，提高聚类内聚性、降低量化误差。

术语是什么？
标准PQ将相邻channel连续打包成子向量，未考虑channel之间的关联性——若高相关channel分散在不同子向量中，每个子向量空间都需要独立的centroids来近似相似的信息，浪费codebook expressiveness。AQPIM的解决方案：(1) 随机选一个reference channel；(2) 计算所有其他channel与该reference的cosine similarity；(3) Greedily选取top-k most similar channels组成一个group（subvector）；(4) 重复m次直到所有channel被分配。排序后，同一group内的channel具有高互cosine similarity，聚类时inner-cluster variance更小。与Channel Permutation for Outlier Clustering（COMET的术语）不同：COMET聚集outliers到少数block以减少高精度量化比例，AQPIM聚集correlated channels到同一subvector以提高聚类质量。

从算法pipeline角度拆解：
```
# 输入: K/V activation samples from calibration data [N_sample, N_head, d_head]
# 输出: 排序矩阵 P_k [d_head, d_head], P_v [d_head, d_head]

# Step 1: 对于key activation的channel维度
for head in range(n_heads):
    channels = list(range(d_head))
    groups = []  # m个group，每个d_head/m个channel
    remaining = set(channels)
    
    for g in range(m=32):
        ref = random.choice(list(remaining))
        similarities = []
        for ch in remaining:
            sim = cosine_similarity(K_activations[:,head,ref], K_activations[:,head,ch])
            similarities.append((ch, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        group = [ch for ch, _ in similarities[:d_head//m]]
        groups.append(group)
        remaining -= set(group)

# Step 2: 构建排序矩阵并吸收到projection权重中
P_k = build_permutation_matrix(groups)  # [d_head, d_head]
# Runtime: 排序矩阵吸收到projection中
W_q' = W_q @ P_k   # query projection
W_k' = W_k @ P_k   # key projection  
W_v' = W_v @ P_v   # value projection
W_o' = W_o @ P_v.T # output projection
```
排序矩阵离线生成（校准数据集：Wikitext-2-v1），吸收到projection weights后runtime无额外开销。

术语一般如何实现？如何使用？
Channel sorting的处理在offline完成：(1) 用calibration dataset跑一次inference收集各层K/V activation samples；(2) 对每层各head执行cosine similarity-based grouping；(3) 生成P_k/P_v permutation matrices；(4) 将permutation baked into projection weights（矩阵乘融合），不引入runtime permute kernel。与COMET的channel permutation不同：COMET需在runtime数据路径中插入permute operator（0.7% overhead），AQPIM通过offline absorption完全消除runtime cost。排序效果取决于activation distribution的cluster结构——论文visualization (UMAP)表明KV activation确实具有tight cluster结构，适合此方法。高压缩率(K=128 centroids)下pre-sorting贡献+4.47 avg score vs w/o pre-sort。

涉及论文标题：
- 89-AQPIM_Breaking_the_PIM_Capacity_Wall_for_LLMs_with_in-Memory_Activation_Quantization.pdf

---

## PQ-Based Attention Computation (Lookup+Summation)

术语解释
PQ-Based Attention Computation（基于乘积量化的注意力计算）是将传统GEMV-based attention替换为codebook lookup和summation操作的技术，直接在compressed data上计算attention，无需dequantization。

术语是什么？
传统attention decoding：q·K^T (GEMV, O(d·N) complexity) → softmax → a·V (GEMV, O(N·d))。PQ-based attention将这两个GEMV转换为：(1) Inner Product Matrix构建：query分m子向量→各子向量与对应key codebook子矩阵乘（固定规模K×d/m，非N）→生成m×K inner product matrix（O(K·d)）；(2) Key lookup：用key indices[n,s]在inner product matrix中查表取值，沿子向量轴求和得qK^T近似值；(3) Value reconstruction：用value indices查value codebook→乘以attention scores→累加。GEMV的复杂度从O(d·N)降为O(K·d)+O(m·N)，其中K=512是常数，m·N远小于d·N（当m<<d时）。关键优势：(a) 不dequantize→不需要解压步骤→不需要INT MAC硬件；(b) 固定规模矩阵乘（K×d/m而非N×d），解耦了计算量与sequence length的关系。

从算法pipeline角度拆解：
```
# 输入: query q[1, d_head], key_codebook[m, K, d/m], key_indices[m, N],
#       value_codebook[m, K, d/m], value_indices[m, N]
# 输出: attention_output[1, d_head]

# ① Split query
q_splits = split(q, m)  # [m, d/m]

# ② Inner Product Matrix (固定规模: m × K)
inner_prod = zeros(m, K)
for s in range(m):
    inner_prod[s,:] = q_splits[s] @ key_codebook[s].T  # [d/m] × [K, d/m]^T → [K]

# ③ Key lookup (replaces q·K^T GEMV)
qK_T_approx = zeros(1, N)
for n in range(N):
    for s in range(m):
        k_idx = key_indices[s, n]  # centroid assignment
        qK_T_approx[0, n] += inner_prod[s, k_idx]  # 查表+累加

# ④ Softmax
attn_scores = softmax(qK_T_approx)  # [1, N]

# ⑤ Value reconstruction (replaces a·V GEMV)
output = zeros(1, d_head)
for s in range(m):
    for n in range(N):
        v_idx = value_indices[s, n]
        vs = s * (d_head/m)
        ve = (s+1) * (d_head/m)
        output[0, vs:ve] += attn_scores[0,n] * value_codebook[s, v_idx]
```
对于GQA/MQA场景（多个query head共享同一KV head），inner_prod matrix可跨query head复用。

术语一般如何实现？如何使用？
HBM-PIM实现：(1) ATNK kernel (BankPE)：并行计算各subvector的query×codebook→生成inner_prod；(2) Intra-row indirection：查表时column decoder读取GRF中的indices信号→通过MUX顺序输出对应row buffer中的inner_prod值（仅1次ACT）；(3) Softmax (BufferPE)：接收所有bank的部分和→执行EXP/SUM/DIV；(4) ATNV kernel (BankPE)：indirection lookup取value codebook entries→乘以attention scores→累加。查表的major bottleneck（随机访问）通过page-aware windowed clustering + intra-row indirection解决。对比GPU上实现：GPU不支持高效的低位宽indirection lookup（需要gather指令→scattered memory access→high latency），这也是PQ-based attention在GPU上不实际的原因之一。

涉及论文标题：
- 89-AQPIM_Breaking_the_PIM_Capacity_Wall_for_LLMs_with_in-Memory_Activation_Quantization.pdf

---

## Page-Aware Windowed Clustering

术语解释
Page-Aware Windowed Clustering（页感知窗口聚类）是AQPIM提出的算法-硬件协同设计：将长序列划分为多个窗口，限制每个窗口内的centroids数量不超过DRAM row buffer容量（如512个FP16值=1KB），使所有indirection查表操作发生在一个DRAM page内，消除跨row的随机激活开销。

术语是什么？
PQ的indices查表产生随机访问模式——不同token的centroid assignment不同，indirection时可能跳转到不同DRAM row。传统DRAM每访问不同row需PRE+ACT，延迟高（tRC≈45ns）。AQPIM的解决方案：(1) 算法上，将序列划分为多个window（每个window内最多512个centroids），window内的所有centroids数据co-locate在单个DRAM row中；(2) 硬件上，intra-row indirection配合column decoder流式输出，将随机逻辑访问转为物理上单row内的顺序流。当window advance时，旧centroids copy到新page，后续incrementally更新。此方法使row activation次数降至window数量（而非token数量），大幅减少DRAM的ACT/PRE开销。

从算法pipeline角度拆解：
```
# Windowed clustering
N_seq = 4096  # sequence length
K_per_window = 512  # centroids per window (fits 1 DRAM row = 1KB = 512×FP16)
num_windows = ceil(N_seq * K / (K_per_window * ...))  # typically = 1 for most lengths

for w in range(num_windows):
    win_start = w * window_stride
    win_end = win_start + window_len
    window_tokens = K_splits[:, win_start:win_end, :]  # 当前window的tokens
    
    # 对当前window执行k-means (K=512)
    centroids_w, assignments_w = kmeans(window_tokens, K=K_per_window)
    
    # centroids_w存储在同一DRAM row中
    # assignments_w的indices => column decoder streaming时全是row-buffer hit
```
当K=512时，大多数长上下文场景（up to 128K）仅需1个window（512 centroids足够），row activations = 1。

术语一般如何实现？如何使用？
实现依赖两个保证：(1) 算法层面：window化centroids数量≤512（K=512配置），如需更多centroids则增加window数；(2) 硬件层面：HBM-PIM的1KB row buffer + column decoder + GRF（存储indices）。page-aware design也extend到value codebook的lookup——value矩阵维度更大但可多次loop over indices。Window间的partial results累加由BankPE的accumulator处理。与GPU上的gather-based lookup对比：GPU gather需要scattered global memory access→high latency+L2 cache thrashing→性能远差于PIM的intra-row indirection。

涉及论文标题：
- 89-AQPIM_Breaking_the_PIM_Capacity_Wall_for_LLMs_with_in-Memory_Activation_Quantization.pdf

## Composition of Experts (CoE) Inference Pipeline (CoE推理流水线)

术语是什么？
CoE推理流水线是Composition of Experts系统的端到端推理过程，包含三个核心阶段：(1) **Routing**：输入prompt经Router model（本身是一个specialized model）处理，输出目标expert identity。(2) **Expert Switching**：若目标expert未在accelerator memory中，执行model loading。（3）**Expert Execution**：目标expert执行推理，通常为autoregressive decoding（prefill + multi-step decode循环）。与Standard LLM inference的区别：CoE在pipeline中插入了Router execution和条件性的expert switching阶段。Router execution的开销通常较低（Router也可以是小模型，或与expert共享compute），switching的开销取决于memory hierarchy设计（HBM-only vs HBM+DDR vs HBM+Host DRAM）。
与Mixture of Experts (MoE) Inference的区别：MoE的routing在模型内部完成（每层有gating network选择top-k experts），expert switching在layer粒度、由硬件tensor core切换完成（experts co-resident in HBM）。CoE的routing在模型外部完成（独立的Router model），switching在model粒度（整个expert model）、由runtime完成（DMA from DDR/Host）。

从算法pipeline角度拆解术语：
CoE推理的伪代码算法：
```python
def coe_inference(prompt, router_model, experts_dict, coe_runtime, 
                  max_tokens=20, use_speculative=False):
    # ===== Phase 1: Routing =====
    router_output = router_model.forward(prompt)  
    # Router执行完整的prefill+单token decode
    expert_id = router_output.expert_selection  
    # 可能是一个expert或概率分布(top-k)
    
    # ===== Phase 2: Expert Activation =====
    if expert_id not in coe_runtime.active_experts:
        coe_runtime.switch_expert(expert_id)  
        # DDR→HBM DMA (SN40L) 或 Host→GPU copy
        # 可能触发LRU eviction
    
    expert_model = experts_dict[expert_id]
    
    # ===== Phase 3: Expert Execution (Autoregressive Decoding) =====
    kv_cache = []
    tokens = tokenize(prompt)
    generated = []
    
    # Prefill phase (batch process input tokens)
    first_token, kv_cache = expert_model.prefill(tokens)  
    # 所有input tokens并行处理构建KV cache
    generated.append(first_token)
    
    # Decode phase (autoregressive loop)
    for i in range(max_tokens - 1):
        next_token, kv_cache = expert_model.decode(
            generated[-1], kv_cache
        )  # 单token decode, KV cache增量更新
        generated.append(next_token)
        if next_token == EOS_TOKEN:
            break
    
    return detokenize(generated)
```

Samba-CoE的具体pipeline（Figure 2和Figure 9）：
```
Input Prompt → Pre-processing (Language Detection) 
→ Router Prompt Engineering → Router Prediction
→ Expert Detector (@Expert Detector模块确定target expert)
→ Expert Prompt Engineering
→ Expert Execution (selected expert model)
→ Post-processing → Completion
```

术语一般如何实现？如何使用？
- **Router模型设计**：(a) Classification-based——fine-tuned Llama2-7B输出固定类别的expert ID；(b) Embedding-based——计算prompt embedding与各expert description embedding的cosine similarity选最近；(c) LLM-as-judge——用大模型直接判断prompt的domain（更高准确率但更高延迟）。
- **Multi-Expert CoE**：复杂query可能需要多个experts协作——如"用Python实现傅里叶变换"先route到coding expert生成代码，再route到math expert验证正确性。Router需支持multi-step routing decisions。
- **Batch CoE**：多个prompts batch处理时，Router BS>1执行得到各prompt的expert IDs → 不同prompts可能需要不同experts → 每个(prompt, expert) pair顺序执行（因不同expert weights不同，无法batch across experts）。这是CoE serving的key challenge（parallelism limited by expert diversity）。
- CoE与MoE的计算量对比：CoE的switching cost来自DDR→HBM DMA (~10ms)，MoE的switching cost来自HBM内的expert weight loading (layer-level gating)。CoE适合大粒度task specialization（整段对话用一个expert），MoE适合细粒度token-level routing（每token可能用不同experts组合）。

涉及论文标题：
- 8-SambaNova_SN40L_Scaling_the_AI_Memory_Wall_with_Dataflow_and_Composition_of_Experts.pdf
