
## Memory Bandwidth Contention between Prefill and Decode（Prefill与Decode间的HBM带宽竞争）

术语是什么？

在PD Multiplexing中，Prefill和Decode虽然使用不同的SM分区和CUDA Stream并行执行，但它们共享同一GPU的HBM（高带宽内存）带宽和L2 Cache。当两阶段同时访问HBM时，产生HBM带宽竞争（Memory Bandwidth Contention），导致各自性能下降。Prefill是计算密集型（compute-bound），Decode是内存密集型（memory-bound），因此Decode对HBM带宽竞争更为敏感。

论文Table 2给出两阶段不同场景的compute complexity分析（d=hidden dimension, L=总token长度, n=new context长度, r=reused context长度）：

| Phase | Attention复杂度 | FFN复杂度 |
|-------|----------------|----------|
| Prefill (无cache) | O(L²d + Ld²) | O(Ld²) |
| Prefill (有cache) | O(nd²) | O(nd²) |
| Decode | O(d² + (r+1)d) | O(d²) |

该分析揭示：Prefill的attention对总context长度呈二次复杂度（无cache时），Decode的attention对reused context呈线性复杂度。当reused context极长时（multi-turn场景可>50K tokens），Decode需大量读取KV cache，与同时执行的Prefill竞争HBM带宽，导致Decode显著变慢。

从kernel调度角度拆解术语：

伪代码表示带宽竞争的kernel执行交互：

```
// 两个CUDA Stream并行执行
Stream_Prefill (GreenContext_Prefill):  // 更多SM
  for each prefill_block:
    // 计算密集：大量矩阵乘法和注意力计算
    QKV = matmul(input, W_qkv)         // HBM读取W_qkv权重
    attn_out = flash_attention(Q, K, V) // HBM读写KV Cache
    ffn_out = matmul(attn_out, W_ffn)  // HBM读取W_ffn权重

Stream_Decode (GreenContext_Decode):    // 少量SM
  for each decode_iteration:
    // 内存密集：主要瓶颈在KV Cache和权重的HBM访问
    q = matmul(token, W_q)             // HBM读取W_q (<1% 计算量)
    attn_out = paged_attention(q, KV)  // HBM读取KV Cache (主要瓶颈)
    ffn_out = matmul(attn_out, W_ffn)  // HBM读取W_ffn权重

// HBM带宽竞争发生在：
// - Prefill的W_qkv读取 vs Decode的KV Cache读取
// - Prefill的W_ffn读取 vs Decode的W_ffn读取
// - L2 Cache行冲突（特别是attention的KV页访问模式与matmul的连续访问模式冲突）
```

从kernel执行角度看：
1. Decode的每token计算量极小（~2×参数量 FLOPs），但HBM访问量很大（读取所有权重+KV Cache页面）。
2. Prefill的每token计算量大（批量矩阵乘），HBM访问模式为连续流式访问。
3. 两者并行时，DRAM控制器需要在两种访问模式间仲裁，L2 Cache的命中率因竞争而下降。
4. 结果是Decode的迭代延迟显著增加（因为Decode对带宽更敏感），可能导致ITL SLO违规。

术语一般如何实现？如何使用？

MuxWise通过Contention-tolerant Estimator来建模和量化这种带宽竞争：离线profile不同SM分配下的Decode延迟曲线和Prefill吞吐曲线，在线性回归基础上加入竞争因子，实时预测给定SM分配下的实际ITL。当预测ITL接近SLO时，调度器减少Prefill的SM分配或降低Prefill并发度。Bullet使用SM-scaling Roofline Model (SRM)作为替代建模方法：基于roofline分析建模compute/memory/network三个维度的性能上界随SM数量变化的关系，用稀疏concurrent sample校准（<1小时 vs Estimator的~12小时），通过roofline边界预测而非全网格查表。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing
- Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration

---

## Contention-tolerant Estimator（容忍竞争的估计器）

术语是什么？

Contention-tolerant Estimator是MuxWise中用于建模和预测PD Multiplexing下性能的核心组件。它量化了当Prefill和Decode在同一个GPU上通过SM空间分区并行执行时，由于HBM带宽竞争导致的性能退化程度。该估计器离线profile不同配置下的延迟/吞吐，在线进行插值预测，为SLO-aware Dispatcher提供决策依据。

Estimator由两个子组件构成：(1) **Solo-run Predictor**：基于离线profiling训练的回归模型，预测无竞争时的prefill和decode延时；(2) **Contention Guard**：通过网格采样profiling建立的最坏情况slowdown因子查表，作为保守保护叠加到solo-run预测上。

Solo-run Predictor的回归公式基于Table 2的compute complexity分析（d=hidden dimension，L=总token长度，r=reused context长度，n=L-r=new context长度）：

```
T_Prefill = θ1·Σ(ni²) + θ2·Σ(ni·ri) + θ3·Σ(ni) + θ4   (Equation 1)
T_Decode  = θ1·Σ(ri) + θ2·bs + θ3                        (Equation 2)
```

公式1中prefill延时由new context的二次项（attention中QK计算O(n²d)）、new×reused交叉项（O(L²d)含reused部分）和线性项构成。公式2中decode延时主要由reused context线性项（KV cache访问O(rd)）和batch size线性项构成。论文报告该模型最大偏差prefill 8.16%、decode 8.84%。

Contention Guard使用网格采样profiling覆盖5个变量（prefill new tokens、prefill reused tokens、decode batch size、decode total reused tokens、partition configuration），以16 SM为partition粒度（A100产生6种配置，H100产生7种），token长度按4的幂次采样（2K到128K），总计约7K个样本对/模型-机器对，profile耗时约12小时。Contention guard返回当前配置所在grid cell的最大slowdown factor，A100上最大slowdown≤20%，H100上≤30%。

从kernel调度角度拆解术语：

估计器的建模和计算流程（伪代码）：

```
// 离线Profile阶段
for SM_decode in range(8, SM_total, step=8):   // Arch 9.0最小8 SM
  for SM_prefill in range(0, SM_total - SM_decode, step=8):
    // 同时运行Decode和Prefill benchmark
    decode_latency[SM_decode][SM_prefill] = measure_decode_iteration_latency()
    prefill_throughput[SM_decode][SM_prefill] = measure_prefill_tokens_per_second()

// 在线估计阶段
function estimate_itl(SM_decode, SM_prefill, batch_size):
    // 基线：无竞争时的Decode延迟（仅Decode运行）
    base_decode_latency = profile_no_contention(SM_decode, batch_size)
    
    // 竞争系数：由SM_prefill带来的额外延迟比例
    contention_factor = lookup_or_interpolate(contention_table[SM_decode][SM_prefill])
    
    // 预测实际ITL
    predicted_itl = base_decode_latency * (1 + contention_factor)
    
    return predicted_itl

function find_min_sm_for_decode(batch_size, itl_slo):
    // 二分搜索满足ITL SLO的最小SM_decode
    for SM_decode in [8, 16, 24, ...]:
        predicted_itl = estimate_itl(SM_decode, SM_total - SM_decode, batch_size)
        if predicted_itl <= itl_slo:
            return SM_decode
    return SM_total  // fallback: 所有SM给Decode
```

关键建模挑战：
1. **DRAM访问模式的非线性竞争**：Decode的随机KV页访问与Prefill的连续权重读取在DRAM控制器层面产生复杂的交互，简单的线性模型不足。
2. **L2 Cache污染**：Prefill的连续访问容易刷掉Decode的KV Cache相关L2缓存行。
3. **Warp调度粒度**：不同SM分区上的warp调度器独立运行，但DRAM调度器全局共享。

术语一般如何实现？如何使用？

实际实现结合离线profile和在线校准：先在目标GPU型号上建立竞争矩阵（不同SM组合下的延迟/吞吐），部署时通过少量在线测量校准模型参数。估计器被SLO-aware Dispatcher周期性调用，用于SM分区决策。对于未在profile矩阵中的配置点，使用插值或小规模神经网络拟合。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

---

## libsmctrl / CUDA Stream SM Mask（CUDA流SM掩码控制）

术语是什么？

libsmctrl是一个底层CUDA库，用于在运行时修改CUDA Stream的SM（Streaming Multiprocessor）执行掩码。通过调用`libsmctrl_set_stream_mask()`，可以将一个CUDA stream上的后续kernel launch限制在GPU特定的SM子集上执行，实现进程内（intra-process）的GPU SM空间分区。与NVIDIA GreenContext相比，libsmctrl通过直接修改CUDA stream metadata（GPC配置掩码）绕过context切换开销，实现微秒级（~4us）的SM重分区。

从kernel调度角度拆解术语：

libsmctrl修改stream mask的kernel执行流程：

```
// 1. 获取GPU GPC/TPC拓扑信息
gpc_info = libsmctrl_get_gpc_info()  // 返回GPC数量和每个GPC的TPC数量

// 2. 构建SM mask（A100: 108 SM, 以16 SM为粒度, 6种配置; H100: 132 SM, 7种配置）
// mask是一个GPC配置掩码，指定哪些GPC中的哪些TPC可用于后续kernel执行
sm_mask_prefill = build_sm_mask(sm_ids=[0..71])   // 72个SM给prefill
sm_mask_decode  = build_sm_mask(sm_ids=[72..107]) // 36个SM给decode

// 3. 将CUDA stream绑定到SM mask
cudaStream_t stream_prefill, stream_decode;
cudaStreamCreate(&stream_prefill);
cudaStreamCreate(&stream_decode);
libsmctrl_set_stream_mask(stream_prefill, sm_mask_prefill);
libsmctrl_set_stream_mask(stream_decode, sm_mask_decode);

// 4. 后续kernel launch自动限制在mask指定的SM子集
// Prefill engine: 在其stream上发射layer-wise kernel
for layer in transformer_layers:
    launch_qkv_kernel(stream_prefill, ...)    // 仅在SM 0-71上执行
    launch_attention_kernel(stream_prefill, ...)
    launch_mlp_kernel(stream_prefill, ...)

// Decode engine: 在其stream上发射CUDA Graph decode step
cudaGraphLaunch(decode_graph, stream_decode)  // 仅在SM 72-107上执行

// 5. Scheduler动态下发repartition command（平均4.1us延迟）
// 当prefill队列堆积或decode接近SLO边界时，更新stream mask
new_sm_mask_prefill = build_sm_mask(sm_ids=[0..89])  // 扩展prefill SM
libsmctrl_set_stream_mask(stream_prefill, new_sm_mask_prefill)
// 后续kernel立即在新SM子集上运行
```

术语一般如何实现？如何使用？

libsmctrl通过修改CUDA stream的内部metadata（具体是GPC配置掩码）工作。当stream上发射一个grid时，CUDA的GigaThread Engine根据stream的SM mask决定将thread block分发到哪些SM。修改mask后，已排队的后续kernel自动遵从新mask，无需重新创建stream或context。

Bullet使用libsmctrl而非GreenContext的原因：libsmctrl的mask更新开销仅~4us，而GreenContext的context切换需要重新初始化CUDA Graph等资源。Bullet的resource manager收到scheduler的repartition command后，直接调用libsmctrl_set_stream_mask()更新相应stream的mask。

依赖与限制：libsmctrl是用户空间库，利用CUDA driver未文档化的内部API修改stream metadata，依赖特定NVIDIA driver版本兼容性。Bullet论文使用CUDA 12.4 + NVIDIA driver。GitHub仓库包含修改版libsmctrl（https://github.com/zejia-lin/BulletServe）。

涉及论文标题：
- Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration

---

## SM-scaling Roofline Model (SRM)（SM缩放Roofline模型）

术语是什么？

SM-scaling Roofline Model (SRM)是Bullet提出的性能预测模型，用于建模LLM推理中prefill和decode阶段在不同SM数量下的延迟上界。与传统roofline模型不同，SRM将SM数量作为自变量，同时建模compute（Tensor Core FLOPS）、memory（HBM bandwidth）和network（NVLink bandwidth，多GPU场景）三个维度的性能饱和边界随SM分配变化的动态关系。SRM仅需少量offline concurrent sample校准，声称profiling overhead低于1小时，在线预测和更新开销为微秒级。

从kernel调度角度拆解术语：

SRM的建模和计算流程：

```
// SRM核心公式：给定SM数量s，预测kernel执行延迟T(s)
// T(s) = max(T_compute(s), T_memory(s), T_network(s))

function predict_latency(sm_count, op_type, tensor_shapes):
    // 1. Compute bound: T_compute = FLOPs / (peak_FLOPS_per_SM * sm_count * efficiency)
    //    小SM数量下compute是瓶颈（prefill场景）
    roofline_compute = total_FLOPs / (PEAK_TFLOPS_PER_SM * sm_count * EFF_COMPUTE)

    // 2. Memory bound: T_memory = bytes / (HBM_bandwidth * contention_factor)
    //    大SM数量下memory bandwidth饱和成为瓶颈（decode场景）
    roofline_memory = total_bytes / (PEAK_HBM_BW * BANDWIDTH_EFF)

    // 3. Network bound（多GPU场景）: T_network = bytes / NVLink_bandwidth
    roofline_network = comm_bytes / (NVLink_BW * sm_count / total_SMs)

    return max(roofline_compute, roofline_memory, roofline_network)

// 校准：运行少量concurrent sample（prefill+decode并行的代表性配置）
// 用实际测量修正contention_factor和efficiency参数
function calibrate():
    for (sm_prefill, sm_decode) in sparse_samples:  // 稀疏采样，非全网格
        run_concurrent_prefill_decode(sm_prefill, sm_decode)
        // 采集实际延迟
        actual_decode_latency = measure_decode_step()
        actual_prefill_latency = measure_prefill_layer()
        // 修正模型参数（线性回归或简单拟合）
        update_model_params(actual_latency)

    // Bullet声称profiling overhead <1小时（对比MuxWise的Contention Guard约12小时）
```

SRM与MuxWise Contention-tolerant Estimator的关键区别：
- **Estimator**：基于全网格profiling（~7K样本对/模型-机器对，12小时），用solo-run predictor + contention guard查表预测
- **SRM**：基于roofline分析模型，用稀疏concurrent sample校准（<1小时），通过roofline边界预测

术语一般如何实现？如何使用？

SRM在Bullet的SLO-aware scheduler中作为performance estimator的核心组件运行：
1. 初始化阶段：对目标模型-硬件组合运行SRM校准（少量代表性SM配置下的concurrent prefill+decode sample）
2. 在线阶段：scheduler调用SRM预测不同SM分配方案下的TTFT和TPOT
3. 自适应修正：在线统计持续修正contention_factor等参数
4. 多维度建模：compute saturation（prefill对小SM数量敏感）、memory saturation（decode对HBM带宽敏感）、network saturation（多GPU tensor parallelism场景）

涉及论文标题：
- Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration

---

## Wave Quantization in GPU Compute（GPU计算的波量化效应）

术语是什么？

Wave Quantization是GPU计算利用率的一种损失现象：当kernel发射的thread block数量不能被GPU的SM数量整除时，最后一"波"（wave）执行中只有部分SM有活干，其余SM空闲。在LLM推理中，prefill阶段的attention和FFN计算的thread block数由序列长度决定，短序列或小chunk下O-proj和attention的block数较少，导致严重的wave quantization —— 部分SM idle，compute utilization从理论峰值下降至约70%-76%。Bullet识别这是chunked prefill利用率低的根源之一。

从kernel调度角度拆解术语：

Wave Quantization在LLM prefill kernel中的表现：

```
// 以Llama3.1-8B的O-proj为例，hidden_dim=4096, block_size=128
// O-proj kernel: matmul(attention_output, W_o), shape=[seq_len, 4096] × [4096, 4096]

// GPU有108个SM，每个SM可运行若干thread block
// 总thread block数 = ceil(seq_len / block_size) * ceil(4096 / thread_tile)
// 假设seq_len=256（1k chunk prefill），matmul tile=128
// blocks_per_seq_dim = ceil(256 / 128) = 2
// blocks_per_hidden_dim = ceil(4096 / 128) = 32
// total_blocks = 2 * 32 = 64

// GPU A100有108个SM，64个block只能填满59%的SM
// Wave 1: 64个block → 64个SM在工作, 44个SM idle
// 这就是wave quantization：block数不足导致SM idle
// compute utilization = 64/108 ≈ 59%

// 实际完整transformer layer有多个kernel（QKV, attention, O-proj, gate, up, down）
// 每个kernel的block数因tensor shape不同而异
// Bullet测量：完整layer的average compute utilization ≈ 70%-76%
```

当chunked prefill使用1k token chunk时：
- 每个chunk的compute efficiency进一步下降至61%
- 最后一个chunk处理时间是第一个chunk的1.9x（因为KV reload叠加wave quantization）
- 总prefill latency比unchunked高1.13x

术语一般如何实现？如何使用？

在LLM serving系统中，缓解wave quantization的方法包括：
- **增大batch/sequence长度**：更多token产生更多thread block，自然填满SM（但受SLO约束不可无限增大）
- **Kernel fusion**：将多个小kernel融合为大kernel，减少wave量化损失（如FlashInfer的融合attention kernel）
- **Persistent kernel**：使用persistent thread block设计，每个block从global work queue动态取活，消除静态block分配导致的SM idle
- **Bullet的方案**：通过intra-GPU prefill-decode concurrency，在prefill SM idle时让decode kernel在这些SM上执行，利用互补的compute/memory特性填充bubble

涉及论文标题：
- Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration

---

## Flashinfer（高性能推理Attention Kernel库）

术语是什么？

Flashinfer是一个针对LLM推理优化的高性能attention kernel库，由华盛顿大学和CMU开发。它提供模块化的GPU kernel实现，涵盖prefill attention、decode attention、paged attention等多种场景。Flashinfer的核心优势是**将（原本分开的）prefill attention和decode attention融合为单一kernel**（即POD-attention的前身），使chunked-prefill场景下prefill chunk的attention和decode batch的attention在同一kernel内完成，消除两次kernel launch和KV cache重复读取的开销。该特性被SGLang默认集成。

从kernel调度角度拆解术语：

Flashinfer的融合attention kernel执行流程（伪代码）：

```
// SGLang + Flashinfer 融合 attention kernel
// 输入: prefill_chunk_tokens (Q_prefill, K_prefill, V_prefill)
//       decode_batch_tokens (Q_decode)
//       kv_cache (所有历史K, V)
function fused_prefill_decode_attention(
    Q_prefill, K_prefill, V_prefill,  // prefill chunk的数据
    Q_decode,                          // decode batch的query
    kv_cache                           // 历史KV Cache
):
    // Step 1: Prefill self-attention
    // Q_prefill @ K_prefill^T (当前chunk内的attention)
    attn_prefill = flash_attention_block(
        Q_prefill, K_prefill, V_prefill
    )
    
    // Step 2: Prefill cross-attention with KV cache
    // Q_prefill @ KV_cache^T (当前chunk与历史context的attention)
    K_cached = kv_cache.load_all_layers()
    V_cached = kv_cache.load_all_layers()
    attn_prefill_cross = paged_attention_block(
        Q_prefill, K_cached, V_cached
    )
    
    // Step 3: Decode attention
    // Q_decode @ [KV_cache + new_KV]^T (decode token与所有历史的attention)
    K_all = concat(K_cached, K_prefill)
    V_all = concat(V_cached, V_prefill)
    attn_decode = paged_attention_block(
        Q_decode, K_all, V_all
    )
    
    // Key insight: 三步在单一kernel内完成
    // - 避免prefill和decode的两次独立kernel launch
    // - KV cache在shared memory / register中复用
    // - 实现了POD-attention的等效性能
    return attn_prefill + attn_prefill_cross, attn_decode
```

相比非融合方案（SARATHI-Serve原始实现的serial execution），Flashinfer融合kernel通过shared memory复用KV cache，消除了一次HBM往返。

术语一般如何实现？如何使用？

Flashinfer在SGLang中通过`from flashinfer import ...`直接调用。其kernel实现以CUDA/CUTLASS模板编写，针对不同head_dim（64/128/256）、不同dtype（fp16/bf16）和不同GPU架构（SM 8.0/9.0）编译多组特化版本。在MuxWise中，Flashinfer被用于decode iteration的attention kernel（通过CUDA Graph launch）。开源地址：https://github.com/flashinfer-ai/flashinfer。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

## Sequence Parallelism (SP/Ulysses) for Inference（面向推理的序列并行）
**attention 之前（MLP、LN、softmax的不同token之間不存在計算，拆分獨立計算）：
  别让 GPU 只拿一段 token
  而是让 GPU 拿完整 token，但只拿部分 heads

attention 计算时（不同token之間需要一起計算）：
  每张 GPU 独立计算自己负责的 heads 的完整 attention

attention 之后：
  再转回按 sequence 切分，继续跑 MLP / LN / 后续层**

术语是什么？

Sequence Parallelism (SP)，亦称Ulysses SP，最初由DeepSpeed Ulysses（arXiv:2309.14509）提出用于长序列训练。核心机制是沿sequence维度切分输入数据到多个GPU，而不是沿model weight维度（如TP）或请求维度（如DP）。在attention计算前通过all-to-all通信将数据从sequence-parallel layout转换为head-parallel layout，attention完成后再通过all-to-all转回。与TP不同，SP的all-to-all通信量不随sequence length增长（Table 2），在长序列和大batch下提供接近DP的throughput。

Shift Parallelism将SP从训练场景改造为inference可用，补充了三个inference关键特性：(1) GQA支持——处理Q head数与KV head数不匹配时通过fused all-to-all中的KV cache replication完成；(2) 小batch load balancing——通过padding到SP degree倍数避免load imbalance（如batch=9, SP=8时效率仅50%，padding后100%）；(3) 任意(SP, TP)组合——支持mixed parallelism应对大模型（如Llama-17B-16E需TP=2才fit单GPU）。

从kernel调度角度拆解术语：

SP for inference的forward pass（Algorithm 1，以(SP, TP)组合为例）：

```
1: embed[n/SP, d] ← SP.slice(input_embeds[n, d])
2: for i = 1, ..., L do
3:   qkv_heads[n/SP, 3×h/TP] ← embed * layer_i.qkv[d, 3×h/TP]
4:   qkv_heads[n, 3×h/(SP×TP)] ← SP.all_to_all(qkv_heads)  // fused QKV通信
5:   attn_o[n, h/(SP×TP)] ← layer_i.attn(qkv_heads)          // head-parallel attn
6:   attn_o[n/SP, h/TP] ← SP.all_to_all(attn_o)              // 返回sequence layout
7:   embed[n/SP, d] ← attn_o * layer_i.o[h/TP, d]
8:   TP.all_reduce(embed)
9:   act[n/SP, d'/TP] ← embed * layer_i.mlp_up[d, d'/TP]
10:  embed[n/SP, d] ← act * layer_i.mlp_down[d'/TP, d]
11:  TP.all_reduce(embed)
12: end for
13: output_embeds[n, d] ← SP.all_gather(embed[n/SP, d])
```

关键点：(a) Line 3：QKV projection使用TP分片权重，每个GPU只处理`h/TP`个heads；(b) Line 4：SP all-to-all将sequence-partitioned的QKV重分布到head-parallel layout，使每个GPU获得完整sequence但仅部分heads的数据；(c) Line 5：attention在head-parallel layout下执行，无需跨GPU通信；(d) Line 6：第二个all-to-all将结果返回到sequence-parallel layout；(e) Lines 8,11：MLP路径使用TP all-reduce（TP沿weight维度切分，需同步partial results）。

术语一般如何实现？如何使用？

SP for inference在ArcticInference/vLLM中通过`--ulysses-sequence-parallel-size N`启用。SP可单独使用（SP=P）或与TP组合使用（SP×TP=P）。典型配置：8 GPU节点，(SP=4, TP=2)作为base config处理大batch；shift config使用(SP=1, TP=8) full TP处理小batch。SP的通信模式使用NCCL all-to-all collective（可fused为单次调用处理Q/K/V）。GQA扩展通过将QKV projection的head数从`3×h`替换为`h + 2×h_kv`实现（h_kv为KV head数），当h_kv < SP degree时通过all-to-all send/receive buffer复制KV head。

涉及论文标题：
- Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads
- TetriServe: Efficiently Serving Mixed DiT Workloads

### DiT场景下的Sequence Parallelism

与LLM SP不同，DiT的SP有显著差异：(a) **目的不同**：DiT模型通常小到可fit单GPU（FLUX.1-dev仅12B/80GB H100），SP仅用于降低延迟而非解决显存容量；(b) **序列内容不同**：SP切分的不是text token序列而是image latent token序列（256×256=256 tokens, 2048×2048=16384 tokens）；(c) **通信开销与分辨率高度相关**：小分辨率（256×256）在SP=8时通信占比超30%导致scaling效率极低，大分辨率（2048×2048）通信占比<10%受益于更多GPU——这是固定SP度在异构workload下失效的kernel级原因；(d) **无KV cache**：DiT是stateless的（每步独立计算全部latent tokens），SP无需维护跨步KV cache一致性，但这也意味着无法像LLM那样通过切换SP度复用KV cache；(e) **两种实现方式**：Ulysses attention使用all-to-all collectives（适合NVLink高带宽），Ring attention使用P2P ring passing（overlap通信与计算）。TetriServe使用xDiT中的Ulysses SP实现，在8×H100 NVLink 4.0和4×A40 PCIe 4.0平台上运行。每步的scaling efficiency通过offline profiling获得，形成cost model lookup table供调度器查询。

## Combined (SP, TP) Forward Pass（组合序列并行+张量并行的前向传播）

术语是什么？

Combined (SP, TP) Forward Pass是Shift Parallelism中base configuration的核心执行路径（Algorithm 1），将Sequence Parallelism沿sequence维度的数据切分与Tensor Parallelism沿model weight维度的计算切分组合在同一forward pass中。SP负责沿sequence维度拆分batch中的token、通过all-to-all转换attention layout；TP负责沿weight维度拆分每层的QKV、O、MLP权重矩阵。关键约束：`SP × TP = P`（总GPU数），且SP group内执行all-to-all通信，TP group内执行all-reduce通信。

从kernel调度角度拆解术语：

Algorithm 1的kernel级执行流程（以SP=4, TP=2, 8 GPU, Llama-70B为例）：

1. **Sequence分片**（Line 1）：输入embedding `[n, d]` 被SP.slice沿sequence维度均分为4份`[n/4, d]`，每份分发到一个SP rank。

2. **QKV Projection**（Line 3）：每个GPU用本地sequence slice和本地TP weight shard `qkv[d, 3×h/2]`做矩阵乘法，产生`[n/4, 3×h/2]`的QKV heads。因为TP=2，每个GPU只持有half heads。

3. **SP All-to-All**（Line 4）：在SP group内执行fused all-to-all。对于SP=4，4个GPU互相交换sequence slices和head partitions。结果：每个GPU获得完整sequence length `n`但仅`3×h/8`（即3×h/(SP×TP)）个heads。通信复杂度O(n×h×d/SP)，不随SP degree增长。

4. **Head-Parallel Attention**（Line 5）：每个GPU对本地`h/8`个heads执行attention计算，使用本地KV cache。由于所有GPUs处理不同的attention heads，attention阶段无通信。

5. **SP All-to-All返回**（Line 6）：第二个all-to-all将结果从`[n, h/8]`转回`[n/4, h/2]`（sequence-parallel + TP head layout）。

6. **O Projection + TP All-Reduce**（Line 7-8）：`attn_o[n/4, h/2] * o[h/2, d]`得`embed[n/4, d]`，然后TP all-reduce在TP group内同步partial sums。因为TP=2，每对TP rank做all-reduce。

7. **MLP + TP All-Reduce**（Lines 9-11）：MLP up projection用TP column-parallel，MLP down用TP row-parallel，各产生一次all-reduce（总计每layer 2次TP all-reduce + 2次SP all-to-all）。

术语一般如何实现？如何使用？

在ArcticInference中，通过`--tensor-parallel-size TP --ulysses-sequence-parallel-size SP`同时指定。系统自动构建SP groups和TP groups，并编译对应CUDA graphs。基Config的约束是SP×TP=P，TP越小（SP越大）通信开销越低（TP的all-reduce随TP degree增高，SP的all-to-all不随SP增高），但TP需确保每GPU能fit模型权重+KV cache。论文中Llama-70B用(SP=4, TP=2)或(SP=8, TP=1)；Llama-17B-16E因109GB内存需TP=2配合SP=4。

涉及论文标题：
- Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads

## Fused All-to-All in Sequence Parallelism（序列并行中的融合全交换通信）

术语是什么？

Fused All-to-All是Sequence Parallelism中将Q、K、V三组attention投影的all-to-all通信融合为单次collective操作的技术。原始Ulysses SP需要对Q、K、V分别执行all-to-all（3次），而融合版本将QKV projection的结果`qkv_heads[n/SP, 3×h/TP]`视为一个整体矩阵，通过单次all-to-all完成sequence layout↔head layout的转换。GQA场景下（Q head数≠KV head数），融合all-to-all变得更关键：将`qkv_heads`的head维度从`3×h`替换为`h + 2×h_kv`，单次通信同时处理Q、K、V的重分布和KV cache replication（当#KV heads < SP degree时）。

从kernel调度角度拆解术语：

Fused All-to-All的通信模式：

1. **发送端**：每个SP rank持有`[n/SP, h + 2×h_kv]`的QKV数据。需将数据重排列为head-parallel layout：每个接收GPU应获得完整序列长度n但仅处理`(h + 2×h_kv)/(SP×TP)`个heads。

2. **Buffer准备**：rank i构建send buffer，将`(h + 2×h_kv)/SP`个heads的数据打包发送给rank j（j ∈ SP group）。若#KV heads不足，在send buffer中复制KV数据——这是KV cache replication的机制。

3. **单次NCCL All-to-All调用**：通过NCCL all-to-all collective完成SP group内所有rank的全交换。通信量约为`O(n × (h + 2×h_kv) × d / SP)` per rank。

4. **接收端**：每个rank从all-to-all接收完整序列的`(h + 2×h_kv)/(SP×TP)`个heads，直接用于本地attention计算。

5. **反向All-to-All**：attention输出需再次通过fused all-to-all从head-parallel layout（`[n, h/(SP×TP)]`）转回sequence-parallel layout（`[n/SP, h/TP]`）。

融合的优势：相比3次独立all-to-all，融合减少NCCL launch overhead、提高网络利用率（更大message size），并在GQA场景天然支持KV cache replication（send buffer中复制KV head）。

术语一般如何实现？如何使用？

在ArcticInference实现中，fused all-to-all通过单次NCCL collective调用实现。send/recv buffer按head维度layout预先分配，GQA path通过send buffer中的KV head复制处理h_kv < SP的场景。用户无需手动配置——系统根据模型配置（#Q heads, #KV heads）和(SP, TP)组合自动选择通信策略。fused all-to-all是SP inference path的核心通信原语，在base config每次forward中执行2次（attention前后各一次）。

涉及论文标题：
- Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads

---

## Bit-Serialization with Carry Save Adder (CSA) for Neural Computation / 神经网络计算中的位串行化与进位保存加法器

术语是什么？
Bit-Serialization with CSA是一种硬件算术优化技术，将多bit输入信号从LSB到MSB串行化（每次处理1 bit），利用Carry Save Adder（CSA）树以多时钟周期替代单周期全宽累加，从而以时间换面积。在HNLPU的Hardwired-Neuron架构中，这是第三步关键优化：gpt-oss 120B的hidden size为2,880，若单周期累加需要大规模加法器，而bit-serial化将单周期累加展开为多周期CSA树（CSA将三个输入压缩为两个输出，避免进位传播延迟），显著减小了硬件面积。CSA是计算机算术中的经典技术（Kai Hwang, 1979），其特点是在进位传播之前通过多级3:2压缩器逐步减少操作数。

从kernel调度角度拆解：
HN中的bit-serial计算过程（伪代码）：
```
// 输入: x[0..2879] 每个4-bit，串行化LSB-first
// 权重: w[0..2879] 由金属线编码（源→颜色区域的连接）
// 16个颜色区域，每个有1-bit POPCNT累加器

for bit_pos = 0 to 3:  // 4-bit精度，4个周期
    for 每个颜色区域 color in 16:
        // POPCNT: 统计路由到该区域且当前bit=1的输入数
        count[color] = POPCNT(输入x_i中满足w_i=color且x_i[bit_pos]=1者)
        // CSA树累加: 将count压缩为partial sum和carry
        (psum[color], carry[color]) = CSA_tree(count[color], psum[color], carry[color])
    
    // 每bit位置累加完成后移位
    psum[16] <<= 1
    
// 4周期后: psum[color] = Σ x_i (对于所有权重=color的输入)
// 乘法阶段:
for color in 16:
    result[color] = psum[color] × weight_value[color]  // 16个固定4b乘法器

// 最终加法树:
output = SUM(result[0..15])  // 4b×16加法树
```
关键设计：CSA树避免每步进位传播（仅最后需要），使关键路径极短适合高频；bit-serial化将2,880宽的全并行累加变成4周期流水线，面积大幅减少。

术语一般如何实现？如何使用？
HN以Verilog实现bit-serial CSA树。CSA采用Full Adder阵列级联，每级将3个输入压缩为2个输出（和+进位）。在HN的16个颜色区域中，每个区域独立配有CSA树，4个bit周期后通过移位累加完成全精度累加。该技术配合weight constancy和distributive law使HN的面积比CE降低93.4%。bit-serial以吞吐换面积——HN仍可在1.0 GHz运行，通过6级pipeline×36层=216并发batch来补偿每操作多周期。

涉及论文标题：
- Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates

## Constant Multiply-and-Accumulate (CMAC) / 常数乘累加单元

术语是什么？
Constant Multiply-and-Accumulate (CMAC) 是一种针对固定权重优化的硬件乘累加单元。在hardwired神经网络中，由于权重参数物理固化在芯片中，乘法器不需要支持任意两数的乘法，而是仅需实现"乘以常数"——这比通用乘法器的布尔复杂度低数倍。在HNLPU论文中，FP4 CMAC比GPU的FP4通用乘法器小约6×；累加也可利用EDA工具的常数优化。CMAC的极致形式是HN中的Metal-Embedding——权重由金属连线表达（源-目的地路由），硅器件仅执行参数无关的POPCNT和固定乘法操作。

从kernel调度角度拆解：
CMAC与通用MAC的差异（以1×1024 input × 1024×128 FP4 weight为例）：
```
// 通用MAC Array (MA): 
// 1024个MAC单元，每个从SRAM读取weight
for i in 0..127:  // 128个输出
    for j in 0..1023:  // 1024个输入
        weight = SRAM_read(addr[i][j])  // 从SRAM取FP4权重
        acc[i] += x[j] × weight          // 通用FP4乘法器

// CMAC (Cell-Embedding):
// 权重固定→乘法器被优化为multiply-by-constant
for i in 0..127:
    for j in 0..1023:
        acc[i] += x[j] ×_constant W[i][j]  // 优化后的constant乘法器
```
论文的实验显示：相比MA（64KB SRAM + 1024 MACs），CE和ME分别将执行周期降低至MA的~1/100（全并行计算+无SRAM fetch），ME的能耗最低（消除SRAM访问+较小面积减少leakage）。

术语一般如何实现？如何使用？
CMAC通过逻辑综合工具的常数传播优化自动实现——当综合工具检测到一个乘法器的一个输入为常数时，自动简化电路。在HNLPU中，CMAC演进为更极致的HN架构——先POPCNT累加输入再乘以常数权重值（accumulate-multiply-accumulate），用16个通用4b乘法器替代2,880个CMAC。CMAC适合无法用Metal-Embedding的常规hardwired实现（如灵活的printed/flexible electronics场景）。

涉及论文标题：
- Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates

## ZipGEMM (Fused Decompression-GEMM Kernel)

术语是什么？通过联网搜索让回答具体和精准。
ZipGEMM是ZipServ提出的fused decompression-GEMM CUDA kernel。它将TCA-TBE格式压缩权重的解压与Tensor Core矩阵乘法融合为单一kernel，实现"load-compressed, compute-decompressed"执行模型。权重从DRAM以压缩格式加载→在register file内解压→直接送入Tensor Core mma.m16n8k16指令计算，消除传统decoupled pipeline中intermediate global memory buffer的redundant memory traffic。ZipGEMM在RTX4090上达1.31× average/1.71× peak speedup over cuBLAS，L40S上达2.21× peak，是首个超越高度优化的cuBLAS Tensor Core GEMM的压缩推理kernel。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ZipGEMM kernel伪代码（split-K tiling，per thread block）：
```
for k_tile in range(0, K, K_TILE):
    // Stage 1: Tile Loading (async)
    cp.async.load(compressed_weight_tile + activations → shared_mem)
    cp.async.wait_group<0>(); __syncthreads()

    // Stage 2: Warp-Level Decoding (per warp, 32 threads)
    for each assigned 8×8 FragTile:
        M = B1 | B2 | B3  // spatial indicator (64-bit)
        for i in {0,1}:  // two elements per thread
            pos = 2*lane_id + i
            if (M >> pos) & 1:  // compressed
                idx_H = popc(M & ((1<<pos)-1))  // dynamic addressing
                val = PackedSignMantissa[start_H + idx_H]
                codeword = (B3[pos]<<2) | (B2[pos]<<1) | B1[pos]
                exponent = base_exp + codeword  // implicit lookup
                bf16_val = MakeBF16(val.sign, exponent, val.mantissa)
            else:  // fallback
                idx_L = pos - idx_H  // complementary offset
                bf16_val = FullValue[start_L + idx_L]

    // Stage 3: Activation Register Transfer
    LDSM.M88(activation_tile → registers)  // layout matches mma requirement

    // Stage 4: Tensor Core Computation
    mma.m16n8k16(weight_regs, activation_regs, accum_regs)

    // Next k_tile iteration...
```
关键micro-architecture优化：
- LDGSTS.128 bypass L1 cache直接写shared memory
- cp.async + __syncthreads() barrier做tile double buffering
- Fine-level: slice-wise interleaving（Tensor Core算slice i时ALU load+decompress slice i+1）
- 全程无shared memory bank conflict（仅~4.7K触发，vs DietGPU百万级）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ZipGEMM以CUDA C++实现（约2.5K行），编译为独立.so库（nvcc 12.4/12.8）。使用方式：
1. 编译：`mkdir build && cd build && cmake .. && make` → 生成libzipgemm.so
2. 调用：通过C++ API传入TCA-TBE格式压缩权重buffer、激活tensor、matrix dimensions (M,N,K)
3. 集成：PyBind11桥接到vLLM的linear layer execution
4. Profiling：Nsight Compute (NCU) 分析micro-architecture counters（DRAM read volume, ALU utilization, Tensor Core utilization, bank conflicts）
适用场景：memory-bound decode阶段效果最好（如batch 8-32的token generation）。compute-bound prefill阶段回退到decoupled pipeline以避免ALU overhead超过memory saving。

涉及论文标题：
- ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression

## Hierarchical Software Pipeline for GPU Kernels（GPU Kernel的两级软件流水线）

术语是什么？通过联网搜索让回答具体和精准。
Hierarchical software pipeline是GPU kernel中使用多级double buffering隐藏memory latency和compute latency的流水线设计。ZipGEMM首次提出两级pipeline用于fused decompression-GEMM：Coarse-level（tile级）用shared memory double buffering重叠global→shared传输与计算；Fine-level（slice级）用ALU/Tensor Core交错重叠decompression与MMA。Quantix进一步发展了该技术用于non-uniform dequantization-matmul：Inter-tile级用Smem0/Smem1双buffer重叠cp.async prefetch与dequant+MMA；Intra-tile级用Reg0/Reg1双buffer重叠CUDA core dequantization与Tensor Core MMA。两级barrier协调：cp.async.wait_group<0>() + __syncthreads()同步inter-tile buffer切换；intra-tile内warp的SIMT lockstep执行天然同步。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
两级pipeline执行timeline（以ZipGEMM为例，单个thread block内）：
```
时间 →
Buffer A: [async load tile 0] [---idle---] [async load tile 2] ...
Buffer B: [---idle---] [async load tile 1] [---idle---] ...
Compute:  [---idle---] [decomp+mma tile 0 slices] [decomp+mma tile 1 slices] ...

单个tile内的fine-level slice interleaving：
  slice 0: [load+decomp w0→regs] [mma w0 × act]
  slice 1:                           [load+decomp w1→regs] [mma w1 × act]
  slice 2:                                                     [load+decomp w2→regs] ...

  ALU:  [decomp s0][decomp s1][decomp s2]...
  TC:            [mma s0] [mma s1] [mma s2]...
```

Quantix的两级pipeline执行timeline：
```
// Inter-tile (K-tile granularity, shared memory double buffering):
//   Smem 0: [cp.async W'/C/A tile 0] [dequant+mma tile 0 subtiles]
//   Smem 1:                          [cp.async W'/C/A tile 1]              [dequant+mma tile 1 subtiles]

// Intra-tile (subtile granularity, register double buffering):
//   Reg 0: [ld.shared + dequant s0] [mma s0] [ld.shared + dequant s2] [mma s2] ...
//   Reg 1:            [ld.shared + dequant s1] [mma s1]           [ld.shared + dequant s3] ...

// 三级overlap:
//   Global→Shared: [cp.async tile 0] [cp.async tile 1] ...
//   Shared→Reg + CUDA Cores (dequant): [dequant s0][dequant s1][dequant s2]...
//   Tensor Cores (mma):                     [mma s0] [mma s1] [mma s2]...
```

ZipGEMM设计使ALU利用率达66.0%（来自decompression的LOP3/IADD/POPC指令），Tensor Core利用率保持cuBLAS的71.6%。Quantix的ablation显示：禁用pipeline（全部序列执行）性能降至完整版本的约41%，证明两级pipeline对隐藏dequantization latency的关键作用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：
1. **Coarse/Inter-tile级**：两个shared memory buffer（weights + activations + centroids），通过cp.async将数据从global memory异步加载到"下一个"buffer，__syncthreads() barrier确保所有线程完成当前tile计算后再切换
2. **Fine/Intra-tile级**：ZipGEMM在每个tile内手动unroll K维度的slice循环，交错安排load指令和mma指令；Quantix用两个register buffer（Reg0/Reg1），当Reg0做dequantization时Reg1被Tensor Cores消费
3. Barrier策略：inter-tile用cp.async.wait_group<0>()等待所有async copy完成 + __syncthreads()所有线程同步；intra-tile内warp的SIMT lockstep执行天然同步
4. Quantix的pipeline需要3类数据同时流动：W1'/W2' (packed indices)、C (centroids)、A (activations)，比ZipGEMM的2类数据（compressed weights + activations）更复杂
5. 该技术可推广至其他需要重叠memory/preprocessing/computation的GPU kernel设计

涉及论文标题：
- ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression
- High-Throughput Non-Uniformly Quantized 3-bit LLM Inference
- RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

## Dynamic Addressing with POPC for Parallel Decompression（基于POPC动态寻址的并行解压）

术语是什么？通过联网搜索让回答具体和精准。
Dynamic addressing with POPC是ZipGEMM Decompressor的核心技术：在GPU warp内并行解压时，每个线程需要确定其负责元素在compact value buffer中的偏移量。由于TCA-TBE以bitmap存储每个元素的存储模式（1=压缩存储在PackedSignMantissa buffer，0=fallback存储在FullValue buffer），偏移量通过POPC（population count，即__popc() intrinsic）在spatial indicator mask上做并行prefix sum得出。例如线程i处理tile内位置2i的元素，需计算indicator mask中bits[0, 2i-1]内值为1的个数（compressed元素偏移）或值为0的个数（fallback元素偏移）。这种将非均匀索引转换为确定性SIMT-friendly前缀和的技巧，是TCA-TBE实现无分支并行解码的关键。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Decompressor的dynamic addressing过程（per thread, per FragTile）：
```
Input: thread lane_id l (0-31), spatial indicator M (64-bit), 
       bitmap B1/B2/B3 (each 64-bit), base_exp, H_buffer_ptr, L_buffer_ptr

For k in {0, 1}:  // two assigned elements a0, a1
    pos = 2*l + k  // global position in 8×8 tile
    mask = (1 << pos) - 1
    idx_H = __popc(M & mask)  // count 1s before pos = compressed elem count
    if (M >> pos) & 1:  // this element is compressed
        val = H_buffer[start_H + idx_H]  // read packed sign+mantissa
        c = (B3[pos]<<2) | (B2[pos]<<1) | B1[pos]  // reconstruct codeword
        exp = base_exp + c  // arithmetic recovery
        result[k] = MakeBF16(val.sign, exp, val.mantissa)
    else:  // this element is fallback
        idx_L = pos - idx_H  // count 0s = total positions - count 1s
        result[k] = L_buffer[start_L + idx_L]  // read full BF16
```
关键insight：idx_H（compressed count）和idx_L（fallback count）通过popc一次计算得出，idx_L = pos - idx_H（因为前pos个元素中非0即1）。这避免了两次popc调用，仅需一次__popc() + 一次整数减法。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖NVIDIA GPU intrinsic：
- `__popc(unsigned int)`：统计32-bit word中1的个数，映射为单条POPC PTX指令
- `__shfl_sync()`：warp shuffle，用于跨线程共享prefix sum结果
- 该技术完全在register内操作，无shared memory往返
- 前提条件：tile大小≤64元素（对应64-bit bitmap）使单warp内popc足够；更大tile需分层prefix sum（先warp内reduction再跨warp）
该技术可推广至其他需要sparse/non-uniform decompression的场景（如sparse matrix decompression、bitmap-based activation sparsity）。

涉及论文标题：
- ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression

## TreeSort-Verify (树排序验证机制)

术语是什么？通过联网搜索让回答具体和精准。
TreeSort-Verify是DFVG提出的高效tree-based speculative decoding验证机制。传统tree-based验证需为每个token sequence维护复杂拓扑感知causal mask（irregular sparse pattern），导致GPU attention计算中memory access不规整，无法充分利用向量化计算能力。TreeSort-Verify通过path-packing对token tree节点重排序，将irregular causal mask转换为block-diagonal lower triangular矩阵形式，使tree attention计算分解为K个独立block的标准attention，每个block直接调用高度优化的cuBLAS GEMM kernel，显著提升GPU验证计算效率。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TreeSort-Verify的kernel级执行流程：
```
Input: token tree T = {t_1, t_2, ..., t_n}, draft tokens with positions

// Step 1: Path-Packing Reordering
π: T → {1,2,...,n}  // 定义排序函数
// 约束: 若t_i是t_j的ancestor，则π(t_i) < π(t_j)
// 效果: 所有ancestor在其所有descendant之前
// 实现: DFS/BFS遍历tree，按深度优先顺序分配全局index

// Step 2: 构建Block-Diagonal Causal Mask
// 重排序后的mask M_reordered:
// M[i,j] = 1 if π(t_j) ≤ π(t_i) AND t_j∈ancestors(t_i), else 0
// 性质: M_reordered是block-diagonal lower triangular

// Step 3: Block Decomposition
// 将重排序序列划分为K个连续block {B_1, B_2, ..., B_K}
// B_k = tokens with indices [start_k, end_k]
// 每个B_k内causal mask是标准lower triangular

// Step 4: Parallel Block Attention
Att_tree = ⊕_{k=1}^{K} Att_block(Q_Bk, K_Bk, V_Bk, M_Bk)
// M_Bk为标准lower triangular mask
// ⊕表示按原始index顺序recombine
// 每个block独立调用cuBLAS GEMM
```
效率来源：(1) block内标准causal mask → 直接调用cuBLAS，无需custom sparse kernel；(2) block-diagonal结构天然支持GPU SM间pipeline并行；(3) 连续block布局improve memory locality，compact KV-cache存储减少bandwidth waste。FPGA侧配套的Multi-Branch Mapping：shared prefix使多branch共享weight loading→Q×K^T复用prefix KV仅改loading address→S×V最后round accumulation归并。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TreeSort-Verify在DFVG的GPU verify path中实现（CUDA C++）。每轮iteration：FPGA通过PCIe发送token tree→GPU host code执行path-packing排序→按block划分→每个block launch cuBLAS GEMM→结果按原始index顺序recombine→acceptance decision。TreeSort-Verify在ablation中贡献2.21×→2.46× speedup（相比仅HW-Branch）。对比传统tree-based验证（SpecInfer的irregular mask），TreeSort-Verify消除sparse mask的memory divergence和vectorized computing underutilization。开源：https://github.com/ShaoqiangLu/DFVG。

涉及论文标题：
- DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU

## Block-Diagonal Causal Mask (块对角因果掩码)

术语是什么？通过联网搜索让回答具体和精准。
Block-Diagonal Causal Mask是DFVG的TreeSort-Verify机制将tree-based speculative decoding的irregular causal attention mask转换后的矩阵形式。传统token tree的attention mask因树结构不规则而呈sparse pattern（如SpecInfer中的topology-aware mask），导致GPU memory access不规整。TreeSort-Verify通过path-packing重排序使mask变为block-diagonal lower triangular——即矩阵由K个沿对角线排列的稠密lower triangular子块组成，块间为零。每个block内部是标准因果attention mask，可直接调用cuBLAS高度优化的GEMM kernel。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Block-Diagonal Causal Mask的构建和使用：
```
// 原始token tree T，节点按ancestor关系排序后:
// T_sorted = [t_π(1), t_π(2), ..., t_π(n)]
//      其中π满足: t_i是t_j的ancestor ⇒ π(t_i) < π(t_j)

// 排序后causal mask M_reordered具有block-diagonal结构:
//
//     B1  │      │      
//     ────┼──────┼──────
//         │  B2  │      
//     ────┼──────┼──────
//         │      │  B3  
//
// 其中每个Bi是mi×mi的下三角全1矩阵（标准causal mask）
// Bi之间可能有零或少量cross-block dependency（由tree topology决定）

// Block-Diagonal在GPU attention中的使用:
for each block B_k in parallel:
    Q_k = Q[T_sorted[start_k : end_k]]
    K_k = K[T_sorted[start_k : end_k]]
    V_k = V[T_sorted[start_k : end_k]]
    // 标准attention计算（无稀疏mask overhead）
    attn_k = softmax(Q_k × K_k^T / √d) × V_k  // 完全在cuBLAS GEMM中

// 跨block零区域天然skip，无额外masking开销
// 最终recombine按原始tree index顺序
```
关键特性：(1) 密度→每个block内部mask全为1（lower triangle），无sparsity overhead；(2) 规整性→每个block shape对齐cuBLAS tile size偏好；(3) 并行性→K个block可在GPU不同SM上并行执行。与FlashInfer的融合prefill+decode attention类似，通过mask结构变换将irregular pattern转为硬件友好形式。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在DFVG中，Block-Diagonal Causal Mask通过path-packing排序自动产生（无需手动构造mask矩阵）。排序后连续block内的tokens天然具有标准因果依赖（ancestor在前，descendant在后，但同block内关系简单）。实现为GPU host code中的预处理步骤（token tree→重排序→block划分），开销极低（token数少，通常≤64）。该技术可推广至任何需要tree-structured attention的场景。开源：https://github.com/ShaoqiangLu/DFVG。

涉及论文标题：
- DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU

## Prefix-Aware Attention Kernel（前缀感知注意力内核）

术语是什么？通过联网搜索让回答具体和精准。
Prefix-Aware Attention Kernel是PAT论文提出的面向LLM decode阶段的GPU attention kernel实现范式。其核心思想是：在decode batch内识别跨请求的共享KV cache prefix，将共享同一prefix的多个query打包进同一个CTA执行，使共享KV blocks仅从GPU global memory加载一次，在CTA内shared memory中复用。这与传统的query-centric attention kernel（每query独立CTA，重复加载共享KV）和KV-centric kernel（固定tile+padding）形成对比。PAT的prefix-aware kernel遵循pack-forward-merge范式：pack阶段将vLLM block table转为prefix tree并基于memory-centric profit model生成CTA partition；forward阶段用multi-tile kernel和multi-stream执行；merge阶段用online softmax合并partial results。在A100上，prefix-aware design使PAT相对FlashAttention的KV cache traffic减少4.1-7.5×，attention latency平均降低53.5%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Prefix-Aware Attention在PAT中的执行流程（以Conversation trace三层prefix为例）：

```
// --- Pack Stage: Prefix Tree Construction ---
Input: decode_batch block_table  // 每行是query的KV block IDs
For each query:
  Path = block_table[query]  // 如 [KV-0, KV-1, KV-3, KV-6]
  Insert Path into prefix_tree  // shared leading blocks → internal nodes

// internal node u: {l_u: shared_KV_length, s_u: num_sharing_queries}
// leaf: one query's full KV path

// --- Pack Stage: TreeHeuristic ---
Function PackTree(root):
  P = []
  For each child c of root:
    if 4 * c.size < root.length:  // Scheme 1: Split
      P += PackTree(c)  // child becomes independent CTA
    else:  // Scheme 2: Merge
      P += PackTree(c with root.blocks)  // merge parent blocks into child's CTA
      root.RemoveQueries(c.queries)
  P += PackCTA(root.remaining_queries, root.shared_blocks)
  return P

// --- Forward Stage: Per-CTA Execution ---
For each CTA in P:
  q = CTA.num_queries         // e.g., 3 queries share prefix
  kv_len = CTA.KV_length      // e.g., 4096 tokens
  
  // Tile Selector
  m = ceil_pow2(q)            // e.g., q=3 → m=4, q=20 → m=32
  n = SelectKVTile(kv_len)    // long KV → n=128, short KV → n=32
  
  // Multi-tile forward kernel execution
  Launch on CUDA stream for (m,n):
    // Loop over KV tiles with double buffering
    for kv_tile in range(0, kv_len, n):
      cp_async_load(K_tile[kv_tile:kv_tile+n] → shared_mem)
      cp_async_load(V_tile[kv_tile:kv_tile+n] → shared_mem)
      
      // QK^T: Q[m, head_dim] × K[n, head_dim]^T → S[m, n]
      // 同一个shared_mem K tile被CTA内所有q个query复用
      for each query i in CTA:
        S[i] = mma(Q[i], K_shared)  // Tensor Core MMA
      
      // Online softmax stats
      m_new = max(m_old, rowmax(S))
      l_new = exp(m_old - m_new) * l_old + rowsum(exp(S - m_new))
      
      // PV: P[m, n] × V[n, head_dim] → O[m, head_dim]
      for each query i in CTA:
        P[i] = exp(S[i] - m_new) / l_new
        O[i] += mma(P[i], V_shared)  // 同一个shared_mem V tile复用

    // Output partial results per query per head
    WriteToGlobalMem(partial_max, partial_lse, partial_O)

// --- Merge Stage ---
For each query q:
  Load all partial results for query q from different CTAs
  // Online softmax merge across CTAs
  m_global = max(all partial_max)
  l_global = sum(exp(partial_max[i] - m_global) * partial_lse[i])
  O_final = sum(exp(partial_max[i] - m_global) * partial_O[i]) / l_global
  WriteToGlobalMem(O_final)
```

关键设计决策：(1) pack阶段决定哪些queries共享KV加载——这是prefix-aware的核心，通过profit model权衡KV读节省与intermediate写开销；(2) multi-tile用round-up规则避免query维度padding（如q=20选m=32而非m=64）；(3) multi-stream让不同tile配置的CTAs并行执行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PAT的prefix-aware attention kernel以约3k行Cutlass/CuTe + C++实现。Multi-tile kernel基于CUTLASS/CuTe的MMA抽象，使用cp_async + double buffering做global→shared memory异步搬运。在vLLM中通过设置环境变量`VLLM_ATTENTION_BACKEND=PAT`启用。Prefix-aware设计依赖三个前提：(1) decode batch中存在跨请求共享prefix（如system prompt、RAG context、tool templates）；(2) GPU memory-bound场景（compute/memory ratio高），HBM bandwidth是瓶颈；(3) serving framework的block table提供logical KV block ID mapping。论文指出prefix-aware attention对batch中共享prefix比例敏感：prefix ratio越高收益越大，无共享prefix时收益显著缩小（仅剩multi-tile/multi-stream的较小优化）。开源实现：https://github.com/flashserve/PAT。

涉及论文标题：
- PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

## Multi-Tile Kernel with Runtime Tile Selector（多Tile内核与运行时Tile选择器）

术语是什么？通过联网搜索让回答具体和精准。
Multi-Tile Kernel是PAT提出的GPU attention kernel设计方法：不再使用单一硬编码的tile size配置（如FlashAttention固定m=64,n=128或FlashInfer固定m=16,n=128），而是offline求解一组可行的(m,n) tile配置，并为每个CTA在运行时选择最优配置。Tile selector是配套的运行时决策逻辑：对每个CTA，根据其query数q选择最小可行Q tile size m（round-up规则，如q=20→m=32避免64的padding浪费），根据其KV length选择最优KV tile size n（长KV偏大n降低per-SM concurrency减少tail execution bubble，短KV偏小n避免最后tile的compute bubble）。该设计解决了已有KV-centric kernel的one-size-fits-all资源浪费问题（当共享prefix的query数少于固定m时需要padding浪费shared memory/register，当CTA KV长度差异大时固定n造成execution bubble）。Ablation显示，替换为固定tile的PAT-fixed比完整PAT慢39%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Multi-tile kernel的设计分为offline和online两个阶段：

```
// ============ Offline: Feasible Tile Set Derivation ============
Input: GPU hardware params (S_smem, S_reg_thr, S_register, S_num, L, B, h, b, b')
Output: feasible_tile_set = {(m,n)} 满足所有约束

For m in {16, 32, 64, 128}:   // power-of-2, >=16 (CUTLASS requirement)
  For n in {16, 32, 64, 128, 256}:
    // Constraint 1: Shared memory upper bound
    if m*h*b + n*h*b + m*h*b' > S_smem:  continue
    
    // Constraint 1: Register bounds (via offline compilation)
    R_thr, R_CTA = compile_and_profile(m, n)
    if R_thr > S_reg_thr:  continue
    CTA_per_SM = floor(S_register / R_CTA)
    if CTA_per_SM < 1:  continue
    
    // Constraint 2: Bandwidth lower bound
    D_flight = S_num * CTA_per_SM * n * h * b
    if D_flight < L * B:  continue  // insufficient in-flight data
    
    // Constraint 3: CUTLASS requirement (implicitly satisfied by loop)
    feasible_tile_set.add((m, n))

// A100 result: 11 feasible configs
// H100 result: 12 feasible configs (移除64,32和64,64)
```

```
// ============ Online: Runtime Tile Selection Per CTA ============
Input: CTA with q queries, KV length kv_len
Output: (m, n) tile configuration

// Q tile selection: round-up rule
m = min{mi in feasible_m_set | mi >= q}
// e.g., q=1→m=16, q=20→m=32, q=40→m=64, q=100→m=128

// KV tile selection: piecewise decision tree (offline profiled)
n = DecisionTree(kv_len):
  if kv_len <= 64:    return 16
  elif kv_len <= 256: return 32
  elif kv_len <= 1024: return 64
  elif kv_len <= 4096: return 128
  else:               return 256  // 需配合Long-KV Split
```

Kernel equivalence验证：在无共享prefix和execution bubble的batch下（batch size设为所有配置CTA concurrency的公倍数，A100用1134，H100用1188），所有feasible配置达到83%-86%（A100）或92%-94%（H100）带宽利用率且latency差异<2%，证明了tile selector可在不损失单kernel性能的前提下实现自适应。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Multi-tile kernel以CUTLASS/CuTe实现，每个feasible (m,n)配置编译为一个独立的kernel variant。Offline solver需要在目标GPU上运行micro-benchmark获取：memory latency L和bandwidth B（通过不同data size的global→shared transfer测latency vs data size曲线），per-thread和per-CTA register使用量（通过nvcc编译+static analysis获取）。移植到新GPU架构需重新运行offline solver推导等价tile set。PAT在A100和H100上都验证了该方法的通用性。Multi-tile kernel使用方式和PAT整体一致：通过pybind11暴露为vLLM backend，环境变量启用。在典型batch中，每个decode step使用的active tile config数量为1-5个（共11个feasible configs中）。

涉及论文标题：
- PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

## Pack-Forward-Merge Execution Paradigm（打包-转发-合并执行范式）

术语是什么？通过联网搜索让回答具体和精准。
Pack-Forward-Merge是PAT提出的面向decode attention的三阶段GPU kernel执行范式，直接针对memory-bound decode attention的两大瓶颈：(1) 共享prefix导致的redundant global memory accesses；(2) 动态query数和KV长度导致的resource inefficiency。Pack阶段将decode batch的block table转为prefix tree，用memory-centric profit model决定哪些queries应被打包进同一CTA（共享KV加载），生成CTA partition；Forward阶段为每个CTA选择最优tile配置并用multi-stream并行执行；Merge阶段用online softmax将同一query被拆分到多个CTA的partial results合并为最终输出。该范式区别于query-centric的one-query-per-CTA和KV-centric的fixed-tile packing，实现KV读复用+资源自适应。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Pack-Forward-Merge作为整体pipeline的执行流程：

```
// ====== Phase 1: Pack ======
Input: batch of queries + vLLM block_table
// block_table: each row = [block_id_0, block_id_1, ..., block_id_k]
// shared prefix → identical leading block_ids

// Step 1: Build prefix tree
prefix_tree = BuildPrefixTree(block_table)
// internal node u: {l_u: shared KV length, s_u: #sharing queries}
// leaf: one query with non-shared KV suffix

// Step 2: Memory-centric profit-driven packing
CTA_list = TreeHeuristic(root)
// Profit model for packing node u into a CTA:
//   Saving: (s_u - 1) * l_u * d  (减少s_u-1次shared KV加载)
//   Overhead: 8 * s_u * d  (FP32 partial intermediate写回+读回)
//   Profit ratio: l_u / 16 ≥ 1 (since KV block size ≥ 16)
// Inter-node: merge child c into parent u when 4*s_c > l_u
// Complexity: O(|V|+|E|) — linear in tree nodes+edges

// Step 3: Lazy update
if block_table unchanged since last step:
  reuse CTA_list from cache  // 跳过重建和调度

// ====== Phase 2: Forward ======
// Step 1: Tile selection per CTA
streams = {}  // map: (m,n) → CUDA stream
For each CTA in CTA_list:
  q = CTA.num_queries
  kv_len = CTA.KV_length
  (m, n) = TileSelector(q, kv_len)  // O(1) lookup
  CTA.tile_config = (m, n)
  streams[(m,n)].enqueue(CTA)

// Step 2: Long-KV split
mean_kv_len = mean(CTA.KV_length for CTA in CTA_list)
For each CTA in CTA_list:
  if CTA.KV_length > mean_kv_len:
    Split CTA into ceil(CTA.KV_length / mean_kv_len) sub-CTAs

// Step 3: Multi-stream parallel execution
For each (m,n) in streams:
  Launch kernel_{m,n} on CUDA stream_{(m,n)}
  // Different streams execute concurrently on GPU
  // Within each stream, CTAs execute sequentially

// Kernel_{m,n} internal:
For each CTA assigned to this stream:
  For kv_block in CTA.KV_blocks step by n:
    cp_async_load(K_tile, V_tile)  // async global→shared
    For each query in CTA:  // shared KV reuse within CTA
      QK^T [m, n]  // Tensor Core MMA
      Online softmax update (max, lse)
      PV [m, n]   // Tensor Core MMA
  Write partial results (max_i, lse_i, O_partial_i) to global memory

// ====== Phase 3: Merge ======
For each query q in batch:
  Gather all partial results for query q
  // Online softmax merge:
  m_final = max(m_1, m_2, ..., m_k)
  For each partial i:
    weight_i = exp(m_i - m_final) * lse_i
  l_final = sum(weight_i)
  O_final = sum(weight_i * O_partial_i) / l_final
  Write O_final to global memory
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PAT将pack-forward-merge实现为vLLM的attention backend plugin。Pack scheduler在CPU上以异步线程运行，与pre-attention tasks（metadata preparation, QKV projection）重叠执行，平均scheduling latency比pre-attention task latency低42-50%，因此不增加端到端延迟。Forward和Merge kernel在GPU上以CUDA/CUTLASS实现。整个pipeline通过pybind11暴露给Python，约1.2k行Python glue code集成到vLLM v0.9.0。打包决策基于logical block IDs（而非physical KV cache data），复用vLLM的paged KV cache机制。开源：https://github.com/flashserve/PAT。

涉及论文标题：
- PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

## Query-Centric vs KV-Centric Attention Execution（面向查询 vs 面向KV缓存的注意力执行）

术语是什么？通过联网搜索让回答具体和精准。
这是两种不同的GPU attention kernel CTA组织策略，用于LLM推理的decode阶段。Query-Centric执行将每个decode query和其完整KV cache独立映射到一个CTA（one-query-per-CTA），各CTA并行执行。该方法调度简单（每个query独立），但在batch内多个query共享prefix时造成冗余的global memory访问：同一shared KV blocks被不同CTA重复从HBM加载到各自的shared memory。代表性实现包括FlashAttention (v2.5.9, tile m=64,n=128)、FlashInfer (v0.2.5, tile m=16,n=128, dynamic CTA partitioning改善load balance)。KV-Centric执行将共享同一prefix的多个queries和对应KV cache打包进一个CTA，共享KV仅从global memory加载一次在CTA内共享复用。该方法减少redundant memory access，但常采用one-size-fits-all tile设计（如FastTree固定两种tile configs、DeFT固定(32,16)、Cascade Inference固定打包参数），当query数小于tile size m时需padding浪费shared memory（I_mem），当KV长度参差不齐时造成execution bubble（I_exe）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
两种策略的对比（以batch含4个query，Q1/Q2共享KV-0/KV-1/KV-3前缀，Q3/Q4共享KV-0/KV-2前缀为例）：

**Query-Centric (FlashAttention)**:
```
// 每个query → 独立CTA
CTA_0: load KV-0, KV-1, KV-3, KV-6 for Q1 → compute → output
CTA_1: load KV-0, KV-1, KV-3, KV-7 for Q2 → compute → output
CTA_2: load KV-0, KV-2, KV-4 for Q3 → compute → output
CTA_3: load KV-0, KV-2, KV-5 for Q4 → compute → output
// 问题: KV-0被加载4次, KV-1被加载2次, KV-2被加载2次 → 冗余
// NCU profiling: FlashAttention KV cache traffic = 4.3-8.7× 理论最小值
```

**KV-Centric (FastTree)**:
```
// 共享KV的queries → 同一个CTA
CTA_0: Q1, Q2 share KV-0, KV-1, KV-3 → compute partial for shared part
CTA_1: Q1 finish KV-6 → compute remaining
CTA_2: Q2 finish KV-7 → compute remaining
CTA_3: Q3, Q4 share KV-0, KV-2 → compute partial for shared part
CTA_4: Q3 finish KV-4 → compute remaining
CTA_5: Q4 finish KV-5 → compute remaining
// Merge partial results across CTAs
// 问题: 使用固定tile (64,32), Q1+Q2只有2个query → 62行padding浪费shared memory
//       CTA_0 (KV-0,1,3) vs CTA_4 (KV-4) KV长度差异大 → tail execution bubble
```

**PAT (Memory-Centric Prefix-Aware)**:
```
// 动态CTA partitioning + multi-tile
CTA_0: Q1, Q2 share KV-0, KV-1, KV-3 → tile (32, 128) for q=2, kv_len~=long
CTA_1: Q3, Q4 share KV-0, KV-2 → tile (32, 64) for q=2, kv_len~=medium
CTA_2: Q1 finish KV-6, Q2 finish KV-7 → tile (32, 32) for q=2, kv_len=short
CTA_3: Q3 finish KV-4, Q4 finish KV-5 → tile (32, 32) for q=2, kv_len=short
// Multi-stream: CTA_0 on stream_A, CTA_1 on stream_B, CTA_2+3 on stream_C
// KV-0仅加载一次（被CTA_0和CTA_1 packing后shared，实际通过prefix tree split决策）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Query-centric实现相对简单——将batch中每个query和其block table entries直接映射到CTA，无额外packing逻辑。FlashAttention/FlashInfer是代表性开源实现。KV-centric实现需要prefix identification和query packing逻辑：FastTree使用compute-oriented cost model做packing并运行两个串行kernel；RelayAttention pack first-level shared prefix使用FlashAttention kernel；DeFT聚合shared KV queries并均衡KV length。PAT在KV-centric基础上引入memory-centric profit model、multi-tile kernel和multi-stream forward，在synthetic batch下相对query-centric FlashAttention/FlashInfer平均降低attention latency 67.8%/52.1%，相对KV-centric FastTree降低3.8-68.9%。选择query-centric还是KV-centric取决于workload特征：无共享prefix时query-centric更简单高效（PAT在无prefix配置下仅1.6% improvement），共享prefix比例高时KV-centric显著更优。

涉及论文标题：
- PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

## Memory-Centric Cost Model for Attention CTA Packing（面向内存的注意力CTA打包代价模型）

术语是什么？通过联网搜索让回答具体和精准。
Memory-Centric Cost Model是PAT在pack阶段使用的CTA打包决策模型。与FastTree的compute-oriented cost model（以最小化计算量为优化目标）不同，PAT的memory-centric model以最小化global memory访问量为目标，因为decode attention是memory-bound的（瓶颈在HBM bandwidth而非Tensor Core compute）。Model计算将queries打包进同一CTA的profit（节省的KV cache加载量）与overhead（因CTA splitting产生的FP32 partial intermediate写回和读回），通过profit-overhead ratio决定parent-child node之间的split/merge策略。该model使得PAT的memory read/write比compute-oriented model（PAT-compute ablation）低10.9%，比naive每node独立pack（PAT-naive ablation）低16.7%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Memory-centric profit model的核心公式：

```
给定prefix tree node u (shared KV length l_u, s_u个共享queries):

// === Intra-node profit (node u自身打包) ===
Saving = (s_u - 1) * l_u * d     // d = head_dimension
  // 相比one-query-per-CTA: s_u次KV加载 → 1次，节省(s_u-1)次
Overhead = 8 * s_u * d            // FP32 intermediate writes + reads
  // s_u个query × 2(写+读) × 2(来自node u + 来自children)
  // × 2(FP32 vs FP16) = 8*s_u*d
Profit_Ratio = l_u / 16 ≥ 1     // 因为KV block size ≥ 16

// === Inter-node profit (parent u + child v_i) ===
// Scheme 1: Split (父和子各自独立CTA)
Profit_split = (s_u - 1)*l_u*d - 4*s_u*d + Σ_i (s_i - 1)*l_i*d
// 开销项4*s_u*d: s_u个query × 2(写+读) × 2(仅来自node u overhead)

// Scheme 2: Merge (child v_i merge进parent u的CTA)
Profit_merge = (s_u - s_i - 1)*l_u*d - 4*(s_u - s_i)*d 
              + Σ_{k≠i} (s_k - 1)*l_k*d + (s_i - 1)*(l_u + l_i)*d
// 合并后: v_i的queries不再需要intermediate write/read
// 增量: Profit_merge - Profit_split = 4*s_i*d - l_u*d

// === 决策规则 ===
if 4*s_i > l_u:  选择 Scheme 2 (Merge)
  // child query数足够多且parent prefix足够短时merge更优
else:             选择 Scheme 1 (Split)
  // parent prefix长时split避免过大的CTA和intermediate overhead
```

关键设计理念：(1) profit以global memory bytes saved计量，而非compute FLOPs——与decode attention的memory-bound性质一致；(2) overhead明确计入FP32 intermediate results的双向global memory traffic；(3) 线性复杂度O(|V|+|E|)适合online serving（每个node和edge仅处理一次）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PAT在pack scheduler中以C++实现TreeHeuristic算法。Scheduler读取vLLM block table→构建prefix tree→对每棵树调用TreeHeuristic递归遍历→按profit model决策每个internal node的split/merge→生成CTA partition。Lazy update机制使得调度结果在block table不变时跨continuous-batching iterations复用，并与pre-attention tasks异步重叠。对比实验显示：PAT-compute（替换为FastTree的compute-oriented cost model）attention latency比PAT高4.6%，memory read/write高10.9%；PAT-naive（简单每node独立pack）latency比PAT高10.4%，memory read/write高16.7%。这验证了memory-centric model设计对memory-bound decode attention的适配性。

涉及论文标题：
- PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

## FAVP (Fast Absmax Value Positioning / 快速绝对值最大值定位)

术语是什么？通过联网搜索让回答具体和精准。

FAVP是JanusQuant提出的快速absmax值定位技术，用于在运行时以极低开销定位每个token中绝对值最大的channel值，以计算RtSmooth的smoothing factor。其核心insight是：K cache的per-token absmax值出现的channel具有跨token的稀疏性和规律性——少数channel持续持有每个token的absmax值（论文Figure 8显示超过90%的层仅涉及少于2%的channels）。FAVP通过部署前一次性离线校准（数分钟）识别每层这些稀疏channel集，运行时仅扫描这些候选channel（而非全量hidden_dim个channel）来获取每个token的absmax，将absmax计算开销降低超过50倍。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FAVP在JanusQuant fused quantization kernel中的执行流程：

```
// === 离线校准阶段（deployment前，一次性，数分钟）===
// 输入：128个WikiText2 8K samples, model
FAVP_channel_sets = {}  // 每层一个稀疏channel索引集
for each layer l in model.layers:
    absmax_channel_counter = zeros(hidden_dim)  // 统计每个channel作为absmax的频率
    for each sample in calibration_samples:
        K_cache = forward_model_to_layer_l(sample)  // 获取该层K cache
        for each token t in K_cache:
            absmax_ch = argmax(|K_cache[t, :]|)  // 该token absmax所在的channel
            absmax_channel_counter[absmax_ch] += 1
    
    // 选择覆盖大多数absmax的稀疏channel集
    sorted_channels = argsort_descending(absmax_channel_counter)
    cumulative = cumsum(absmax_channel_counter[sorted_channels]) / total_tokens
    // 论文：>90%的层仅需<2% channels即可覆盖大多数absmax
    FAVP_channel_sets[l] = sorted_channels[0 : top_k]  // top_k ≈ 0.02 * hidden_dim

// === 运行时：Fused Smoothing + Quantization Kernel ===
// 每g=32 token触发一次（而非每token每step）
__global__ void fused_smoothing_quantization_kernel(
    FP16* K_segment,      // g × hidden_dim FP16 K values
    int* FAVP_channels,   // 预校准的稀疏channel索引（~2% × hidden_dim）
    int num_candidate_ch,  // 候选channel数量
    FP16* K_quantized,    // 输出：INT2 packed K values
    ParamBlock* params    // 输出：unified parameter block
) {
    for each token t in block's assigned range:
        // Step 1: FAVP - 仅扫描稀疏候选channel
        FP16 absmax_val = 0.0;
        for each ch in FAVP_channels[0:num_candidate_ch]:
            absmax_val = max(absmax_val, abs(K_segment[t][ch]));
        
        // Step 2: Smoothing factor（无需扫描全部hidden_dim channels）
        FP16 gamma = pow(absmax_val, 0.5);  // lambda = 0.5
        gamma = (gamma == 0) ? 1.0 : gamma;
        
        // Step 3: Per-token smoothing
        for each ch in range(hidden_dim):
            K_segment[t][ch] = K_segment[t][ch] / gamma;
        
        // Step 4: Per-channel group quantization + INT2 packing
        // ... (后续量化步骤，smoothing factor存入param block)
        params[t].smoothing_factor = gamma;
}

// 无FAVP的naive runtime smoothing: absmax需扫描全部hidden_dim
// cost: O(g × hidden_dim) reads → O(g × 0.02 × hidden_dim) with FAVP
// 论文量化kernel breakdown (Figure 15a):
//   - naive smoothing: 4.43× overhead @ 64K seq len vs no-smoothing baseline
//   - RtSmooth + FAVP: overhead降至接近1× (near no-smoothing baseline)
```

关键实现细节：(1) FAVP依赖于absmax channel的跨token稳定性，论文在128个8K样本上验证——跨样本absmax分布保持稳定；(2) 候选channel集按per-layer存储，推理时作为constant memory或read-only buffer传入kernel；(3) FAVP不要求精确匹配真实absmax（它只是smoothing factor的近似计算），论文Figure 11显示predicted-to-actual ratio的偏差在极小比例token中出现，对perplexity无负面影响。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FAVP使用时需要：① 部署前用代表性calibration数据运行一次离线校准（论文用128个WikiText2 8K样本，数分钟完成）；② 为每个attention layer生成并存储稀疏channel索引集（存储开销可忽略，仅<2% × hidden_dim × sizeof(int) per layer）；③ 在fused quantization kernel中，以calibrated indices作为gather操作的索引，通过coalesced或random memory access读取候选channel值。由于候选集极小且across tokens重复使用，L1 cache命中率高。FAVP的最大优势在于将quantization kernel中占比>80%的absmax计算开销降至可忽略水平（论文Figure 15a）。其局限性是依赖cross-token absmax channel稳定性，对新架构/非标准attention的迁移需要重新校准。

涉及论文标题：
- JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference

## Mixed-Precision Attention Kernel with Fused Dequantization（融合解量化的混合精度注意力内核）

术语是什么？通过联网搜索让回答具体和精准。

Mixed-Precision Attention Kernel with Fused Dequantization是JanusQuant提出的自定义CUDA attention kernel，在同一kernel内同时处理2-bit quantized historical KV cache和FP16 recent KV tokens，将INT2 dequantization与attention computation融合消除独立dequantization kernel的memory-bound开销。在baseline准确率导向方法（SKVQ/KIVI/KVQuant）中，dequantization作为独立kernel先于attention执行，造成额外的kernel launch overhead和全局内存往返（论文SKVQ breakdown：dequantization占decode 80% runtime）。JanusQuant kernel利用RtSmooth保留的positional alignment特性——2-bit KV和FP16 KV在hidden dimension排列一致，可直接分段由不同thread block处理——在单一kernel内完成unpack、dequantize、attention三个步骤。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Mixed-Precision Attention Kernel的执行流程（以Llama2-7B decoding为例）：

```
// === Mixed-Precision Attention Kernel（单次decoding step）===
// 输入：
//   Q: FP16 query token (1 × num_heads × head_dim)
//   history_KV_quantized: INT2 packed KV cache (seq_len_hist tokens)
//   recent_KV_FP16: FP16 ring buffer KV (≤ n*g tokens)
//   param_blocks: unified parameter blocks per thread block segment

// Grid: 2D thread blocks — dim 0沿KV sequence, dim 1沿heads
// 每个thread block处理一段KV + 一个head

__global__ void mixed_precision_attention(
    FP16* Q,                     // [num_heads, head_dim]
    INT2_packed* K_hist,        // [seq_hist, num_kv_heads, head_dim] (packed)
    INT2_packed* V_hist,        // [seq_hist, num_kv_heads, head_dim] (packed)
    FP16* K_recent,             // [num_recent, num_kv_heads, head_dim]
    FP16* V_recent,             // [num_recent, num_kv_heads, head_dim]
    ParamBlock* param_blocks,   // unified parameter blocks aligned to thread blocks
    FP16* output                // [num_heads, head_dim]
) {
    int block_id = blockIdx.x;   // KV segment index
    int head_id = blockIdx.y;    // attention head index
    int kv_head_id = head_id_to_kv_head(head_id);  // GQA mapping

    // === Phase 1: 加载Q到shared memory ===
    __shared__ FP16 Q_shared[head_dim];
    if (threadIdx.x < head_dim) Q_shared[threadIdx.x] = Q[head_id * head_dim + threadIdx.x];
    __syncthreads();

    // === Phase 2: 处理2-bit historical KV segment ===
    if (block_id < num_hist_segments):
        // Step 2a: 加载unified parameter block（scale, zero_point, smoothing_factor, offsets）
        ParamBlock pb = param_blocks[block_id * num_kv_heads + kv_head_id];
        
        // Step 2b: 从global memory加载INT2 packed KV segment到shared memory
        // 注意：unified parameter layout将4类参数合并对齐 → 减少memory transactions
        load_packed_KV_to_smem(K_hist, V_hist, block_id, kv_head_id, pb);
        
        // Step 2c: INT2-to-FP16 unpacking + dequantization（在register中）
        for each value in thread's KV chunk:
            // 高效unpacking: 利用FP16 exponent manipulation
            FP16 k_val = int2_to_fp16_unpack(packed_K_chunk, thread_local_idx);
            FP16 v_val = int2_to_fp16_unpack(packed_V_chunk, thread_local_idx);
            
            // Dequantization: v_hat = q * s + z
            k_val = k_val * pb.K_scale + pb.K_zero_point;
            v_val = v_val * pb.V_scale + pb.V_zero_point;
            
            // Inverse smoothing: 恢复RtSmooth的缩放
            k_val = k_val * pb.smoothing_factor;
        
        // Step 2d: 计算QK attention scores + softmax（in register/shared memory）
        FP16 attn_score = dot_product(Q_shared, k_vals) / sqrt(head_dim);
        // ... softmax across all KV tokens (online softmax with running max/sum)
        
        // Step 2e: 加权求和V
        output_acc += attn_weight * v_vals;

    // === Phase 3: 处理FP16 recent KV segment（同一kernel内） ===
    else if (block_id >= num_hist_segments && block_id < num_hist_segments + num_recent_segments):
        // 直接加载FP16 KV，无需dequantization
        load_FP16_KV_to_smem(K_recent, V_recent, recent_block_id, kv_head_id);
        
        // 标准attention计算
        attn_score = dot_product(Q_shared, k_fp16) / sqrt(head_dim);
        // ... softmax ...
        output_acc += attn_weight * v_fp16;

    // === Phase 4: 输出 ===
    output[head_id * head_dim + threadIdx.x] = output_acc;
}

// === INT2-to-FP16 高效unpacking（3条指令处理2个值）===
// 利用FP16格式特性：[1024, 2047)区间共享exponent=1024
// mantissa直接编码offset
__device__ FP16 int2_to_fp16_unpack(uint32_t packed, int idx) {
    // Step 1: 提取2-bit值 (lop3 bitwise extract)
    uint32_t val_2bit = (packed >> (idx * 2)) & 0x3;
    
    // Step 2: 放入FP16 mantissa + 设置exponent (or with 0x64006400)
    // 0x6400 = exponent 25 (1024) + sign 0 → FP16 exponent field
    uint32_t fp16_repr = (val_2bit << 10) | 0x6400;  // int in mantissa, exp=1024
    
    // Step 3: 减去1024得到最终FP16 (sub)
    FP16 result = __int2float_rn(fp16_repr) - 1024.0f;
    // 等价于：result = (FP16)(int_val)，但避免了通用INT→FP转换指令
    return result;
}
```

关键thread block调度：kernel以task parallelism组织——不同thread block分别处理quantized和FP16 segments，异步执行重叠计算与访存。由于quantized segment的thread block做更多计算（unpack+dequantize），而FP16 segment的thread block做更少计算，这种分工自然平衡了block间的工作量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现需约3500行CUDA/C++，包含：① fused dequantization-attention CUDA kernel（如上），通过FlashInfer API包装为Python extension；② INT2-to-FP16 unpacking的PTX inline assembly（利用lop3/or/sub指令）；③ unified parameter block layout（scale/zero_point/smoothing_factor/offsets按thread block KV segment访问模式重排）；④ thread block到KV segment的mapping逻辑（区分quantized/FP16区域）。Kernel可编译为standalone .so，通过Pybind暴露为Python compatible torch.nn.Module。使用Nsight Compute Roofline分析指导优化：初始kernel memory-bound → dequantization fused后变为compute-bound → INT2 unpacking优化降低compute intensity → 参数block layout减少memory transactions。论文Figure 14/15b显示：128K seq/hidden 4096/32 KV heads下JanusQuant attention kernel speedup 6.17× over KIVI、1.69× over QServe、平均1.99× over FA2。

涉及论文标题：
- JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference

## Hardware-Aligned Bit Shuffling (Bit Dividing + Bit Mapping / 硬件对齐的比特重排)

术语是什么？通过联网搜索让回答具体和精准。
Hardware-aligned bit shuffling是Quantix提出的一种离线权重布局转换技术，用于解决3-bit non-uniform quantized weights与GPU native word size（32/64-bit）及Tensor Core operand layout之间的不对齐问题。核心idea是将"odd-bit packing"难题转换为"easy-bit packing"问题：通过bit dividing将每个3-bit index拆分为1-bit和2-bit两个独立矩阵，使每路均与GPU INT类型天然对齐；再通过bit mapping将元素按Tensor Core tile访问模式重排为连续segments，确保coalesced memory access和Tensor Core operand匹配。该变换是lossless w.r.t. quantized model——只重排index bits，不修改centroids，因此完全保留原non-uniform量化精度。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Bit shuffling分两步，均在offline完成：

```
// ===== Step 1: Bit Dividing =====
// 输入: Wq ∈ INT^{K×M} (每个元素3-bit)
// 输出: Wq,1 ∈ {0,1}^{K×M}, Wq,2 ∈ {0,1,2,3}^{K×M}

for each element e in Wq:
    // 3-bit binary: e = b2 b1 b0 (MSB to LSB)
    // 任意选1个bit分离，例如MSB:
    Wq,1[i,j] = (e >> 2) & 0x1   // 1-bit (可以选任意bit)
    Wq,2[i,j] = e & 0x3          // 2-bit (剩余)

// Packing: 32个1-bit元素 → 1个32-bit word (perfect fit)
//          32个2-bit元素 → 1个64-bit word (perfect fit)
// 关键: 1和2都是32和64的因子，因此无padding浪费、无spanning跨界
```

```
// ===== Step 2: Bit Mapping =====
// 目标: 将packed elements按Tensor Core (TC) tile访问模式重排
// 每个warp负责1个64×64 warp tile
// warp tile分为16个16×16 TC tiles (TC Tile 0-15)

// 在每个TC tile内，每个thread负责4 pairs of elements
// Pair 0: element (0,0/1), element (8,0/1)
// Pair 1: element (0,8/9), element (8,8/9)
// Pair 2: element (0,0/1), element (8,0/1) [不同位置]
// Pair 3: element (0,8/9), element (8,8/9) [不同位置]

// 按配对的元素索引，批量映射输出

// Bit mapping: 收集同一thread在16个TC tiles中的4×16 pairs×n bits
// = 128n bits (n=1: 128 bits; n=2: 256 bits)
// 组织为连续linear memory segment Wn'

// 每个thread的segment布局:
// W1': 128 bits → 1次cp.async (128-bit)抓取
// W2': 256 bits → 2次cp.async (128-bit)抓取
```

```
// ===== Physical layout in memory =====
// warp 0: [thread0_W1' | thread1_W1' | ... | thread31_W1']
//         [thread0_W2' | thread1_W2' | ... | thread31_W2']
// warp 1: [thread0_W1' | thread1_W1' | ... | thread31_W1']
//         ...
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Bit shuffling实现为Python/C++预处理脚本，在模型部署前执行一次。输入为已non-uniform quantized的权重（由SqueezeLLM/Any-Precision/Bitsandbytes等生成），输出为W1'、W2'和reordered centroids C。具体使用：
1. 加载由任意non-uniform量化方案生成的Wq (3-bit indices)和C (per-row FP16 centroids)
2. 执行bit dividing: Wq→Wq,1 (1-bit) + Wq,2 (2-bit)
3. 执行bit mapping: 按warp tile→TC tile层次重排，生成W1'/W2' linear segments
4. 输出打包为GPU可读取的binary格式
该变换的代价被所有后续推理分摊（offline, one-time），因此对在线推理无性能影响。Quantix论文开源：https://github.com/yuang-chen/Quantix-PPoPP26

涉及论文标题：
- High-Throughput Non-Uniformly Quantized 3-bit LLM Inference

## In-Register Dequantization（寄存器内解量化）

术语是什么？通过联网搜索让回答具体和精准。
In-register dequantization是Quantix fused kernel中使用的技术，将non-uniform quantized weights的所有dequantization操作完全在GPU register file内完成，无需任何global memory或shared memory的中间读写。该技术通过bit concatenation（1-bit+2-bit→3-bit index）和centroid indexing（3-bit index查per-row FP16 centroids）两条寄存器内路径重建FP16权重，避免了SqueezeLLM等传统方法中"dequantize→写回内存→从内存读回→matmul"的多级内存路径和cache-unfriendly centroid pointer chasing。Ablation study显示in-register dequantization是Quantix最大性能贡献组件——移除后性能降至完整版本的约40%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
In-register dequantization在fused kernel的一个TC tile (16×16)内的执行流程：

```
// ===== 寄存器布局 =====
// 一个TC tile各thread负责4 pairs (8个元素/thread)
// Pair的结构: row0 element1, row0 element2, row8 element1, row8 element2
// 原因: Tensor Core MMA的ldmatrix指令需要交错来自不同行的数据

// W1' (1-bit) segment registers: 128 bits/thread
//  R1_low:   bit positions 0-31 (32个1-bit values for W1')
// W2' (2-bit) segment registers: 256 bits/thread
//  R2_low, R2_high: bit positions 0-63, 64-127 (32个2-bit values for W2')
// Centroids C registers: 8个FP16 per row × 2 rows = 16个FP16 registers
```

```
// ===== Step 1: Bit Concatenation =====
// 处理4个pairs (i=0,1,2,3)，每个pair含2行×2列=4个元素
for pair_idx in range(4):
    // 从W1'和W2'提取对应bits
    bit1_row0_col0 = extract_bit(R1, pair_idx*2+0)     // 1-bit
    bit2_row0_col0 = extract_2bits(R2, pair_idx*2+0)    // 2-bit
    
    // Concatenate: [1-bit] + [2-bit] → [3-bit]
    index_3bit_row0_col0 = (bit1_row0_col0 << 2) | bit2_row0_col0
    // 例如: [1] + [10] → [110] = 6 (binary)
    
    // 8个3-bit indices打包到1个32-bit register
    // Register layout: [row0_pair0 | row0_pair1 | row8_pair0 | row8_pair1 | ...]
```

```
// ===== Step 2: Centroid Indexing =====
// 每行有2^k个FP16 centroids (k=3时8个)
// Centroids C均已在registers中（从shared memory加载）
// 例如: Row 0's centroids = [33.14, -48.24, 1.32, 0.90, -7.82, 53.13, 73.96, -27.63]

// Step 2a: Extract individual 3-bit index from packed 32-bit register R
// qi = (R >> (3*i)) & 0x7
// 避免条件分支的bitwise操作
for i in range(8):
    qi = (R >> (3 * i)) & 0x7  // 提取第i个3-bit index
    
// Step 2b: Centroid lookup (register-to-register)
// 用qi索引centroids数组
    w_deq[i] = centroids_row[i//4][qi]
    // 论文未明确说明centroid lookup的具体register-level实现
    // 可能使用PRMT（permute）指令或conditional select
```

```
// ===== 为什么In-Register关键 =====
// Naive方法 (SqueezeLLM等):
//   W† = C[Wq]  → pointer chasing: Wq读取→地址计算→memory load→返回→存入memory
//   latency: global memory load (数百cycles)
// In-register方法:
//   qi = (R >> (3*i)) & 0x7 → 1条shift + 1条AND (各1 cycle)
//   centroid select from register → ~1-5 cycles
//   latency: 数cycle vs 数百cycles
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
In-register dequantization在Quantix fused kernel中以CUDA PTX实现。关键点：
1. 数据准备：bit-divided和bit-mapped的W1'/W2' segments通过cp.async从global预取到shared memory，再通过ld.shared加载到registers
2. Dequantization在CUDA cores上执行（ALU指令：shift, AND），无需shared memory参与
3. 重建的FP16 weights直接在registers中被Tensor Core MMA消费，中间不写入任何内存
4. 寄存器布局精心设计以匹配Tensor Core的ldmatrix指令对数据interleaving的要求（如图7所示row0/row8交替）
5. 该技术要求centroids也在registers中——每行8个FP16 centroids (3-bit)，共需16个FP16 registers（2行），对于32-wide的warp register file可承受
6. 限制：过大batch时register pressure增加可能导致spilling，影响ALU utilization——论文观察到batch≥32时ALU utilization下降即因register spilling

涉及论文标题：
- High-Throughput Non-Uniformly Quantized 3-bit LLM Inference

## Fused Dequantization-Matmul Kernel（融合解量化-矩阵乘的内核）

术语是什么？通过联网搜索让回答具体和精准。
Fused dequantization-matmul kernel是Quantix的核心在线计算kernel，将non-uniform quantized weights的dequantization与Tensor Core matrix multiplication融合为单一CUDA kernel。该kernel以hardware-aligned bit-shuffled weights (W1', W2')、activations A和centroids C为输入，在单次kernel launch中完成prefetch→load→dequantize→matmul全流程。与分开执行的"先dequantize kernel→写memory→matmul kernel→读memory"常规做法相比，fused kernel消除了中间全局内存往返，使dequantization latency被Tensor Core computation隐藏。kernel通过两层double buffering实现三级流水线重叠（global memory prefetch / dequantization on CUDA cores / MMA on Tensor Cores）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Fused kernel执行伪代码（Algorithm 1 in paper）：

```
// Algorithm 1: Fused Kernel in Quantix
// Input: W1' (1-bit), W2' (2-bit), A (FP16 activations), C (FP16 centroids)
// Output: Y = A × Dequant(W1', W2', C)

for each thread block in Split-K slices: // parallel
    // === Initialization ===
    Smem[0] = cp.async(W1'[tile_0], W2'[tile_0], C[tile_0], A[tile_0])
    cp.async.wait()
    
    // Load first subtile to registers and dequantize
    Reg[0] = ld.shared(Smem[0], subtile_0)
    Reg[0].Wt = dequantize(Reg[0].W1', Reg[0].W2', Reg[0].C)
    // Reg[0] now holds FP16 reconstructed weights
    
    // === Main Loop with Hierarchical Pipeline ===
    for k in range(num_K_tiles):
        // --- Inter-tile level: overlap prefetch with compute ---
        // Prefetch next K-tile to alternate shared memory buffer
        Smem[(k+1) % 2] = cp.async(W1'[k+1], W2'[k+1], C[k+1], A[k+1])
        
        // --- Intra-tile level: overlap dequant with MMA ---
        for s in range(1, num_subtiles):
            // Load next subtile from shared memory
            Reg[s % 2] = ld.shared(Smem[k % 2], subtile_s)
            
            // Dequantize on CUDA Cores (current subtile)
            Reg[s % 2].Wt = dequantize(Reg[s % 2].W1', Reg[s % 2].W2', Reg[s % 2].C)
            
            // MMA on Tensor Cores (previous subtile)
            Y_partial = mma(Y_partial, Reg[(s-1) % 2].A_frag, Reg[(s-1) % 2].Wt_frag)
            // mma.m16n8k16 on Tensor Cores
        
        cp.async.wait() // ensure prefetch complete
    
    // Reduction: merge partial sums from Split-K
    Y = reduction_kernel(Y_partials)

// === Helper: in-register dequantization ===
function dequantize(W1, W2, C):
    R_3bit = 0 // 32-bit register for 8 3-bit indices
    for pair in range(4):
        bit1 = W1[pair]         // 1-bit
        bit2 = W2[pair]         // 2-bit
        idx = (bit1 << 2) | bit2
        R_3bit |= (idx << (3 * pair))
    
    Wt = [] // reconstructed FP16 weights
    for i in range(8):
        qi = (R_3bit >> (3 * i)) & 0x7
        row = i // 0-3 for row0, 4-7 for row8 (interleaved layout)
        Wt[i] = C[row][qi] // centroid lookup
    
    return Wt
```

```
// ===== 数据流timeline =====
// Inter-tile (K-tile granularity):
//   Smem 0: [cp.async tile 0] [mma/dequant tile 0 subtiles] 
//   Smem 1:                    [cp.async tile 1]              [mma/dequant tile 1 subtiles]
//
// Intra-tile (subtile granularity):
//   Reg 0: [load+dequant s0]    [mma s0]   [load+dequant s2]  [mma s2] ...
//   Reg 1:          [load+dequant s1] [mma s1]          [load+dequant s3] ...
//
// Tensor Cores:  [idle] [mma s0] [mma s1] [mma s2] [mma s3] ...
// CUDA Cores:    [dequant s0] [dequant s1] [dequant s2] ...
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Quantix fused kernel以CUDA/C++实现，约数千行代码。关键实现技术：
1. **Memory access**: 128-bit cp.async (UINT4 reinterpret) 实现vectorized global→shared传输；ld.shared实现shared→register加载；ldmatrix实现activation register准备（Tensor Core operand format）
2. **Tensor Core调用**: 通过PTX内联汇编调用mma.m16n8k16等指令，支持FP16输入/输出
3. **Double buffering**: 两个shared memory buffer + 两个register buffer实现三级流水线
4. **Split-K**: 沿K维度切分，多个thread block group并行计算partial sums，最后lightweight reduction kernel归并
5. **集成**: Kernel集成进HuggingFace Transformers替换SqueezeLLM默认backend，uniform baselines (GPTQ/Marlin)使用AutoGPTQ library
6. **Accuracy**: Kernel不改变量化模型精度——bit shuffling是lossless，dequantization是bit-exact重建

开源：https://github.com/yuang-chen/Quantix-PPoPP26

涉及论文标题：
- High-Throughput Non-Uniformly Quantized 3-bit LLM Inference

术语是什么？通过联网搜索让回答具体和精准。

INT2-to-FP16 Efficient Unpacking是JanusQuant在mixed-precision attention kernel中使用的一种低开销2-bit整数到FP16浮点数的转换方法。其核心insight来自FP16格式特性：在[1024, 2048)数值区间内，所有FP16值共享相同的exponent（2^10=1024），而mantissa的10位直接编码0-1023的整数偏移量。因此，对于[0, 1023]范围内的整数，可以通过直接将该整数放入FP16 mantissa位并设置exponent=1024来"合成"FP16表示，再减去1024得到最终值。这种方法避免了通用INT→FP转换指令的高延迟，用3条指令（lop3 bitwise extract, or设置exponent, sub减偏移）处理两个2-bit值。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

INT2-to-FP16 unpacking在attention kernel中的具体实现（PTX级别）：

```
// === PTX Register-Level INT2→FP16 Unpacking ===
// 输入:
//   R1 (32-bit): 打包的16个2-bit values（共32 bits）
//   R2 (32-bit): 工作寄存器
// 输出:
//   每轮处理2个值 → 2个FP16 values packed in R2

// Step 1: 用lop3提取第i个2-bit值到R2
// lop3是NVIDIA GPU的三输入bitwise逻辑指令，可单指令完成复杂bitwise提取
lop3.b32 R2, R1, 0x3, RZ, extract_pattern;  
// R2 = (R1 >> (i*2)) & 0x3  → 提取的2-bit值在R2低位

// Step 2: 移位到mantissa位置 + OR设置exponent
shl.b32 R2, R2, 10;           // 将2-bit值左移10位到mantissa位置(bit 10-19)
or.b32  R2, R2, 0x64006400;   // 设置两个FP16值的exponent字段
// 0x6400 = 0110 0100 0000 0000
//   bit 15: sign = 0
//   bits 14-10: exponent = 01100 = 24 (bias 15 → actual exponent = 24-15 = 9, 
//                wait - 论文说exponent = 1024. 让我修正:
//   实际上: exponent_field = 1024 (actual exponent) → with bias 15:
//   stored_exponent = 1024 + 15 = 1039 = 0b10000001111 → 
//   but in FP16, stored exponent is 5 bits for values >= 1.0
//   
//   更准确的解释: FP16对于整数1024+n的表示：
//   1024 = 2^10 → stored exponent = 10+15 = 25 = 11001
//   mantissa = 0 (since 1024 is exact power of 2)
//   1024+n 其中n∈[0,1023]: stored exponent = 10+15 = 25,
//   mantissa = n (n的10-bit binary)
//   所以 0x6400 = 0110 0100 0000 0000
//     bit 15: 0 (sign)
//     bits 14-10: 11001 = 25 (exponent for 2^10)
//     bits 9-0: 0000000000 (mantissa = 0 → 代表1024)

// 处理两个值: 低位FP16和高位FP16
// R2低16位: val_low in mantissa bits 9-0, exponent = 25 → FP16(1024 + val_low)
// R2高16位: val_high in mantissa bits 9-0, exponent = 25 → FP16(1024 + val_high)
// 所以0x64006400将两个16-bit half的exponent都设为25

// Step 3: 减去1024得到最终值 (sub指令)
// 1024.0 in FP16 = 0x6400
sub.f16 R2_low, R2_low, 1024.0;   // FP16(1024+val_low) - 1024 = FP16(val_low)
sub.f16 R2_high, R2_high, 1024.0; // FP16(1024+val_high) - 1024 = FP16(val_high)

// 对比naive实现：__int2float_rn()需要完整INT→FP转换流水线
// 至少4条指令/值（包括range check, leading zero count, shift, round）
// JanusQuant方法: 3条指令处理2个值 (lop3 + or + sub×2 = 平均1.5指令/值)
```

此方法的关键约束：① 仅适用于[0,1023]范围的整数（2-bit值永远在[0,3]内，天然满足）；② 依赖FP16格式中exponent=1024区间mantissa直接编码整数的特性；③ 需要R1和R2均为32-bit register，一次处理16个2-bit值。论文称这是"inspired by prior work on 8-bit and 4-bit dequantization [17]"的适配。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在JanusQuant attention kernel中以PTX inline assembly实现。CUDA kernel中每thread处理一段KV cache segment，从shared memory加载INT2 packed data到register（R1），循环调用lop3→shl→or→sub序列解包每个2-bit值。该实现需要CUDA compute capability ≥ 7.0（支持lop3指令）。在A100上验证：unpacking优化相比naive INT→FP转换使attention kernel平均加速1.99×（Figure 15b），且与parameter block layout combined后达到3.05× speedup。此技术可推广到其他低位量化（3-bit/4-bit等）的dequantization kernel，但需调整bit提取逻辑和mantissa移位偏移量。

涉及论文标题：
- JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference

## Two-level Sparse Storage Format (BSR + Bitmap) for Sparse MHA

术语是什么？通过联网搜索让回答具体和精准。

Two-level Sparse Storage Format是STOF提出的用于在GPU MHA kernel中统一表示任意sparse attention mask的两级存储格式，结合Block Compressed Sparse Row (BSR)和bitmap。该格式将mask矩阵按两级抽象组织：OuterTile (OT) 级——每个OT为64个8×8 InnerTile (IT)，OT作为coarse-grained skipped block单元；InnerTile (IT) 级——每个IT为8×8元素块，内部64个元素的mask pattern用一个uint64 bitmap_mask值精确表示。OT被分为"full"（内部所有IT均非空）和"part"（内部至少一个IT含mask元素但非全满）两类。存储结构由6个数组组成：full_row_ptr/full_col_idx（CSR格式表示full OTs位置）、part_row_ptr/part_col_idx（CSR格式表示part OTs位置）、load_row_ptr/load_col_idx（将full和part OTs按row-major合并的统一加载索引）、bitmap_mask（每个part OT对应64个uint64值，每个表示一个8×8 IT的mask pattern）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Two-level storage format在block-wise MHA kernel中的使用流程（STOF Algorithm 1）：

```
// ===== Mask Preprocessing (offline) =====
// Input: mask matrix M ∈ {0,1}^{seq_len × seq_len}
// OT_Size_M = OT_Size_N = 64 (64个8×8 IT = 512×512 elements per OT)
//
// 1. Partition M into OT grid
// 2. For each OT: if all 64 ITs non-empty → "full"
//               elif at least one IT non-empty → "part"
//               else → skip (empty)
// 3. Build arrays: full_row_ptr, full_col_idx, part_row_ptr, part_col_idx,
//    bitmap_mask[part_ot_idx][64] (uint64 per IT)
//    load_row_ptr/load_col_idx = merge of full + part

// ===== Kernel Runtime =====
for i in [0, ceil(seq_len / OT_Size_M)):        // Row-Parallel Dimension
    Q_i = Load_from_HBM(Q_HBM_i)                  // Q_i in registers (resident)
    load_num = load_row_ptr[i+1] - load_row_ptr[i]
    part_num = part_row_ptr[i+1] - part_row_ptr[i]
    tmp_part_idx = 0
    
    for kv_idx in [0, load_num):
        j = load_col_idx[load_row_ptr[i] + kv_idx]
        K_Tj = Load_from_HBM(K_HBM_j)
        V_j = __async_memcpy(Load_from_HBM(V_HBM_j))  // async: overlap with GEMM
        
        P_ij = Compute_GEMM(Q_i, K_Tj)  // register × SMEM → register
        
        // Fine-grained mask for "part" OTs
        if tmp_part_idx < part_num and j == part_col_idx[part_row_ptr[i] + tmp_part_idx]:
            Apply_Mask(P_ij, bitmap_mask[tmp_part_idx])  // uint64[64] per part OT
            tmp_part_idx++
        
        S_ij, alpha = Softmax(P_ij)       // Online Softmax with scaling
        O_i = O_i × alpha + Compute_GEMM(S_ij, V_j)
    
    result_HBM = Write_back_to_HBM(O_i)
```

8×8 IT设计理由：正好匹配NVIDIA Tensor Core的mma.m16n8k16操作粒度；64个elements恰好填满一个uint64（每element 1 bit），bitwise操作（POPC/OR/shift）可直接用于mask判断无额外memory lookup。IT列主序存储消除bank conflict（同warp内线程访问不同列映射到不同bank），OT行主序支持Softmax的iterative row-wise reduction。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

STOF以CUDA/C++实现（约2,500 LOC），基于FA2的CuTe结构扩展。预处理在Python端将mask转换为6个数组通过torch::Tensor传入GPU。BSR格式是NVIDIA cuSPARSE原生支持的稀疏矩阵格式（提供cusparseXcsr2bsrNnz等API）。bitmap_mask存储为uint64一维数组。该格式的通用性：causal、sliding window、Longformer、Bigbird等任意mask pattern均可用同一套数据结构和kernel处理——仅预处理不同。相比之下FlashMask的column-range数组仅支持column-continuous mask。

涉及论文标题：
- Accelerating Sparse Transformer Inference on GPU

## Block-wise Sparse MHA Kernel with Kernel Selection Analytical Model

术语是什么？通过联网搜索让回答具体和精准。

Block-wise Sparse MHA Kernel是STOF框架中针对GPU优化的sparse multi-head attention fused kernel，以OT (OuterTile)为粒度partition Q/K/V张量，利用two-level sparse storage format跳过无效计算，将QK^T GEMM、mask application、Softmax和PV GEMM全部融合在单一kernel中执行。与之互补的是Row-wise Kernel（以Q行为粒度partition，warp内shuffle通信，适合小seq_len+高稀疏率场景）。Kernel Selection Analytical Model（公式1）基于valid OT ratio和seq_len自动选择row-wise或block-wise kernel。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Kernel Selection公式和核心优化：

```
// ===== Kernel Selection Analytical Model (STOF Eq. 1) =====
threshold = (load_row_ptr[ceil(seq_len/16)] / (ceil(seq_len/16))²) 
          - (τ / (log₂(ceil(seq_len/16)))²)
// τ = 1.2 (empirically set)
// 第一项 = valid OT ratio per row
// 第二项 = log penalty（压制extreme sparse长序列场景）
//
// threshold < 0 → row-wise kernel (warp内shuffle, 零warp间sync)
// threshold ≥ 0 → block-wise kernel (partition到SMEM利用memory hierarchy)

// ===== Block-wise Kernel Key Optimizations =====
// 1. Q Register Resident: Q_i保持在register跨KV tile循环复用（vs FA2需每次SMEM读）
// 2. Async Data Copying: cp.async异步加载V_j，与Q_i×K_Tj GEMM重叠
// 3. OT-level Compute Skipping: 仅加载valid OTs，Bigbird (80.8% sparsity)仅~19.2% OTs计算
// 4. IT-level Fine-grained Mask: 对part OTs用bitmap_mask做per-element mask (POPC/shift)
// 5. SMEM Double Buffering: K_Tj/V_j共享同一SMEM物理区域
// 6. Tensor Core Alignment: 8×8 IT对齐mma.m16n8k16 operand tile

// Performance: (batch=16, seq_len=4096, sliding window 93.8% sparsity)
// STOF block-wise vs FA2: 4.8× on A100
// STOF block-wise vs FlexAttention: 4.9× on A100
```

Kernel selection实例：
```
// 场景1: BERT-Base, batch=1, seq_len=128, causal (50% sparsity)
// OT grid: 2×2, load_row_ptr[2]=3 → threshold≈0.75-1.2=-0.45<0
// → row-wise kernel: warp内shuffle, 零warp间sync

// 场景2: BERT-Base, batch=16, seq_len=4096, Bigbird (80.8% sparsity)
// OT grid: 64×64, load_row_ptr[256]≈12/row
// → block-wise kernel: SMEM hierarchy利用, Q register resident
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

STOF实现：block-wise kernel基于FA2的CuTe结构，扩展引入two-level storage format和对应优化，约2,500 LOC C/CUDA。通过torch/cpp_extension JIT编译为.so。CuTe提供tile-level抽象（TiledMMA、TiledCopy等），允许类型安全地组合tensor core操作、shared memory staging和global memory access。block_size/num_warps/num_stages等launch配置通过AutoTune搜索。

涉及论文标题：
- Accelerating Sparse Transformer Inference on GPU

## Online Row-wise Normalization (RowNorm Online / 在线行归一化)

术语是什么？通过联网搜索让回答具体和精准。

Online Row-wise Normalization (RowNorm Online) 是 MetaAttention 提出的一种通用 online 行归一化接口，将 row-wise normalization（如 softmax、RetNet 的 reduceAbsSum normalization）拆分为 online_prologue、online_forward、online_epilogue 三段式，使得 normalization 可以在分 tile 遍历 K/V sequence 时逐步更新归一化状态，无需物化完整 score matrix 到 global memory。该设计源自 FlashAttention 的 online softmax 思想，但被泛化为通用接口以支持任意 row-wise normalization（不仅仅是 softmax）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

RowNorm Online 三段式接口的 kernel 执行流程：

```
// 以 RetNet reduceAbsSum-based RowNorm 为例
// 用户定义 RowNorm Online 接口的三个函数

class scores_RowNorm_Online:
    def online_prologue():
        row_sum_wo_clamp = 0           // 初始化未 clamp 的累积和
        row_sum = 0                     // 初始化 clamp 后的累积和
        return row_sum_wo_clamp, row_sum

    def online_forward(scores_tile, row_sum_wo_clamp_prev, row_sum_prev):
        // 当前 tile 的局部分量
        row_sum_cur = scores_tile.reduceAbsSum()
        // 更新全局未 clamp 和
        row_sum_wo_clamp = row_sum_wo_clamp_prev + row_sum_cur
        // clamp 防止除零
        row_sum = max(row_sum_wo_clamp, 1)
        // 用当前全局和归一化，并 rescale 之前 tiles 的输出
        scores_tile = scores_tile / row_sum
        scale = row_sum_prev / row_sum   // 传递给 aggregation 阶段 rescale 已累积的 output
        return scores_tile, row_sum_wo_clamp, row_sum, scale

    def online_epilogue(scores_tile):
        return scores_tile                // 最终输出（通常 identity）
```

Kernel 执行时，runtime 在遍历 KV tile 的主循环中：
```
// MetaAttention Parallel Pattern kernel 简化伪代码
row_sum_wo_clamp, row_sum = online_prologue()
output = zeros[head_dim_v]

for kv_tile in range(0, seq_len_kv, kv_tile_size):
    // 1. 异步加载 K_tile, V_tile 从 global → shared memory (TMA on H100)
    // 2. relevance scoring: scores_tile = Q × K_tile^T (Tensor Cores MMA)
    // 3. 应用 scores_Mod (如 mask, scaling) - SIMT fused
    // 4. online_forward: 更新归一化状态 + 归一化 scores_tile + 计算 scale
    scores_tile, row_sum_wo_clamp, row_sum, scale = online_forward(scores_tile, ...)
    // 5. 用 scale rescale 之前累积的 output
    output = output * scale
    // 6. aggregation: output += scores_tile × V_tile (Tensor Cores MMA)
    // 7. online_epilogue (通常 identity)

return output
```

按 RowNorm Online 标准的 online softmax 表达：
```
class scores_RowNorm_Online:
    def online_prologue():
        row_max = -inf; row_sum = 0
        return row_max, row_sum

    def online_forward(scores_tile, row_max_prev, row_sum_prev):
        row_max_cur = scores_tile.reduceMax()
        row_max = max(row_max_prev, row_max_cur)
        // rescale: 用新 max 修正之前累积和
        row_sum = row_sum_prev * exp(row_max_prev - row_max)
        row_sum += scores_tile.exp().reduceSum() * exp(row_max_cur - row_max)
        scores_tile = exp(scores_tile - row_max) / row_sum  // 归一化当前 tile
        scale = row_sum_prev / row_sum * exp(row_max_prev - row_max)
        return scores_tile, row_max, row_sum, scale

    def online_epilogue(scores): return scores
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

RowNorm Online 接口的泛化设计使得 MetaAttention 能支持任意 row-wise normalization（softmax、sigmoid、ReLU norm、RetNet reduceAbsSum norm 等），而不仅仅局限于 FlashAttention 内置的 online softmax。在 MetaAttention 实现中：online_prologue/forward/epilogue 作为 customizable function 被 trace 为 tensor DAG，其产生的中间状态（row_sum、row_max 等）作为 IntermediateTensor 纳入 scheduling（通常分配在 register 以最小化 latency），forward 中的 elementwise/scaling 操作被 SIMT fused，reduce 操作使用 intra-warp reduction。该接口的实现受到 FlashAttention 的 online softmax [Milakov & Gimelshein 2018] 和 FlashAttention [Dao et al. 2022] 的启发，但被抽像为通用接口。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends

## Chunk Parallelism in Recurrent Attention（递归注意力中的块并行）

术语是什么？通过联网搜索让回答具体和精准。

Chunk Parallelism 是一种用于 recurrent/linear attention 的并行优化技术：将长序列沿 sequence 维度切为多个 chunk，chunk 内使用并行矩阵运算（intra-chunk，可并行化），chunk 间以 recurrent 方式传递压缩 state（inter-chunk，需顺序执行）。该技术使 recurrent attention（如 Mamba2 SSM、RetNet Recurrent、Gated Retention）能在训练和长序列 prefill 时利用 GPU 并行性，避免完全的 token-by-token 串行执行。MetaAttention 在 Recurrent Pattern kernel template 中集成了 chunk parallelism，由 scheduler 根据序列长度和硬件资源自动确定 chunk size。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Chunk Parallelism 在 Recurrent Attention 中的执行流程：

```
// 假设序列长度 seq_len，chunk size = C，共 num_chunks = seq_len / C 个 chunk
// 全局 hidden state H 的 chunk-parallel 更新

H_initial = zeros[dimqk, dimv]         // 初始 state

for chunk_id in range(num_chunks):     // inter-chunk: 顺序传递 state
    chunk_start = chunk_id * C
    chunk_end = chunk_start + C

    // 以下 intra-chunk: GPU 上并行执行（matrix operations）
    Q_chunk = Q[chunk_start:chunk_end] // [C, dimqk]
    K_chunk = K[chunk_start:chunk_end] // [C, dimqk]
    V_chunk = V[chunk_start:chunk_end] // [C, dimv]

    // 并行 relevance scoring: 用当前 state 输出所有 chunk token
    output_chunk = matmul(Q_chunk, H_initial)  // [C, dimv]

    // 并行 state 累积: K 和 V 的 chunk 内累积
    H_update = matmul(K_chunk^T, V_chunk)      // [dimqk, dimv] via outer product sum
    H_initial = H_initial + H_update           // 更新 state 传给下一个 chunk

    // customizable Mod/RowNorm 在 chunk 内 elementwise+reduction 融合
    output[chunk_start:chunk_end] = output_chunk
```

与 Parallel Pattern (FlashAttention-style) 的对比：
- Parallel Pattern: O(seq_len²) memory（需完整 score matrix），通过 online tiling 避免物化
- Recurrent Pattern + Chunk Parallelism: O(chunk_size² + dimqk×dimv) memory，通过 chunk 内并行 + chunk 间 state 传递实现 O(seq_len) 计算复杂度

Chunk size 的 trade-off：大 chunk → 更高 intra-chunk 并行度 + 更高算术强度，但 O(C²) memory + O(C²) intra-chunk 计算；小 chunk → 更多 inter-chunk 串行步骤，但 memory 占用低。MetaAttention scheduler 通过 IntermediateTensor-based scheduling 权衡确定 chunk size（考虑 shared memory 容量、register 数量等硬件约束）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Chunk Parallelism 在多个框架中有实现：(1) Mamba2 chunk kernel（Mamba2 论文的 Triton 实现）——用 chunk-based scan 替代 sequential scan；(2) Flash-Linear-Attention (FLA v0.2.0) ——提供 chunkwise parallel operators for DeltaNet、GatedDeltaNet、RetNet Recurrent 等；(3) MetaAttention Recurrent Pattern runtime——自动为 recurrent attention 应用 chunk parallelism。在 MetaAttention 中，用户选择 Recurrent Pattern 后无需手动配置 chunk 参数，scheduler 自动确定 chunk size，runtime 生成 chunk-parallel kernel（将 elementwise/reduction 逻辑融合到 recurrent kernel 中，避免额外 kernel launch）。chunk 大小通常为 64-256，受 shared memory 容量约束。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends

---

## Ragged Batching Kernels via Round-Robin Tile-to-Thread-Block Mapping（基于Round-Robin Tile映射的Ragged批处理Kernel）

术语是什么？通过联网搜索让回答具体和精准。

Ragged Batching Kernel是Difflow为支持异构shape请求的高效批处理而实现的kernel层技术。核心分为两类：(1) **Ragged data-independent operation kernels**：对不跨请求共享数据的操作（transpose/reduce等），基于已有regular operator的tiling plan和microkernels，将每个请求划分为tile集合，通过round-robin policy在batch执行时映射tile到GPU thread blocks；(2) **Redundancy Memory Access Elimination**：对attention操作中冗余K/V tensors，运行时压缩K/V沿redundant batch dim + concat Q tensors → 使用FlashAttention等标准kernel执行压缩计算。Difflow实现了四个ragged data-independent operation kernels（Triton + CUDA）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Round-Robin Tile-to-Thread-Block Mapping伪代码：

```
// === Ragged Data-Independent Operation Kernel ===
// 输入: batch of requests, each with different shape
// 已有: regular operator的tiling plan (tile_H × tile_W)和microkernel

1. Function RaggedDataIndependentKernel(batched_requests, op_type):
2.     // Step 1: Tile decomposition (per-request独立)
3.     all_tiles = []
4.     for each request r in batched_requests:
5.         r_tiles = decompose_to_tiles(r.input, op.tiling_plan)
6.         // 例如: 256×256→4个128×128 tiles; 512×768→24个128×128 tiles
7.         all_tiles.append((r.id, r_tiles))
8.     
9.     // Step 2: Round-robin tile dispatch to thread blocks
10.    total_tiles = sum(len(r.tiles) for r in batched_requests)
11.    num_blocks = min(total_tiles, max_thread_blocks)
12.    // Launch num_blocks thread blocks
13.    for block_id in 0..num_blocks-1:
14.        // Round-robin: 每个block轮流处理不同请求的tiles
15.        tile_idx = block_id
16.        while tile_idx < total_tiles:
17.            (request_id, tile) = flatten_tiles[tile_idx]
18.            execute_microkernel(tile, request_id)
19.            tile_idx += num_blocks  // stride = num_blocks

// === Redundancy Memory Access Elimination (Attention) ===
// 例: 3个请求，共享相同prompt → 相同K/V tensors
// Baseline: Q1×K1^T, Q2×K2^T, Q3×K3^T → 3次独立attention (K1=K2=K3)
// Eliminated:
20. K_compressed = compress(K, along redundant batch dim)  // [1, H, S, d] (去重)
21. V_compressed = compress(V, along redundant batch dim)  // [1, H, S, d]
22. Q_concatenated = concat([Q1, Q2, Q3], along batch dim)  // [3, H, S_q, d]
23. output = FlashAttention(Q_concatenated, K_compressed, V_compressed)
24. // 沿batch dim split + broadcast 恢复各请求的output
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Difflow基于Triton和CUDA实现了4个ragged data-independent operation kernel。Round-robin mapping策略的关键优势：(1) 无需为不同shape组合设计specialized tiling——每个请求的tile独立decompose然后通过round-robin负载均衡到thread blocks；(2) 与regular kernel共享microkernel实现，仅调度层不同。Redundancy Memory Access Elimination通过等价线性代数变换实现——compress+concat操作是lightweight的（无额外compute），仅需改变tensor layout，随后直接调用FlashAttention等高度优化的regular kernel。该方法避免了为冗余attention场景重写优化kernel。

涉及论文标题：
- Difflow: A Data-Characteristic-Aware Serving System for Diffusion Models

## Compressed Sparse Patch (CSP) Format

术语是什么？通过联网搜索让回答具体和精准。

Compressed Sparse Patch (CSP) Format是MixFusion提出的一种受Compressed Sparse Row (CSR)启发的patch管理数据结构。CSR是稀疏矩阵的经典压缩存储格式（三数组：values/column_indices/row_offsets），CSP将其思想迁移到图像patch管理场景：将不同分辨率请求产生的heterogeneous patches通过resolution reorder + offset-based compression高效存储和定位。核心差异在于CSR的block size固定且位于矩阵左上角，而CSP支持diverse block sizes（各resolution产生不同数量patches）且patches按请求连续分布。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

CSP格式的内存布局与查询流程：

```
内存数组（以4请求, resolutions=[1024,512,768,1024], patch_size=256为例）:
  RequestStart[] = [0,0,0,0, 4,4,...,4, 13,13,...,13, 17,17,...,17]
                   // 每个patch对应其所属请求的起始patch ID
  RequestEnd[]   = [3,3,3,3, 12,12,...,12, 16,16,...,16, 32,32,...,32]
                   // 每个patch对应其所属请求的终止patch ID
  ResolutionOffset[] = [0, 8, 17, 33]
                   // 累计patch数：4+9+4+16=33(实际), 用于Self-Attn分组
  RequestOffset[] = [0, 4, 13, 17]  // 每个请求的首patch在全局数组中的偏移

查询操作：
  patch_id → 二分查找RequestOffset确定所属请求request_idx
  request_idx → RequestOffset[idx]到RequestOffset[idx+1]遍历该请求所有patches
  request_idx → ResolutionOffset[idx]提供Self-Attention reconstruction偏移

构造流程：
1. 请求按arrival时间排序 (Figure 9a)
2. 按resolution重排序→patches视为sparse array (Figure 9c)
3. 仅记录每个patch的request和resolution metadata
4. 计算累积offset：RequestOffset（per-request首patch偏移）
5. 计算分辨率偏移：ResolutionOffset（per-request resolution分组偏移）
```

CSP的O(1)定位能力来自offset预计算：任何patch通过其global index二分查找RequestOffset即可确定所属请求，再通过RequestStart/End确定在请求内的patch范围。这比naive per-patch metadata存储（需记录resolution/shape/offset等）大幅节省内存。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

CSP在MixFusion中作为Python + C++ runtime实现。patch splitting时根据batch内各resolution的GCD确定patch_size→计算每个请求的N_h×N_w patches→按resolution排序后算出所有offset arrays→denoising过程中所有patch操作通过CSP提供的O(1) lookup定位上下文（如Convolution边界stitching所需的邻接patch、Self-Attention全图reconstruction所需的分组信息）。CSP的设计使patch管理overhead modest（Figure 17中splitting overhead仅占latency的极小比例），且支持dynamic batch composition（新请求到达时只需重新计算offset arrays，无需重组patch数据）。

涉及论文标题：
- MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models

## Patch Edge Stitcher

术语是什么？通过联网搜索让回答具体和精准。

Patch Edge Stitcher是MixFusion提出的一种fused CUDA kernel，将扩散模型中Convolution算子的跨patch边界stitching操作融合(fuse)到GroupNorm kernel中执行。核心观察是T2I扩散模型（SDXL等U-Net架构）中Convolution通常紧接GroupNorm操作——因此可在normalization过程中overlap boundary data exchange，消除独立的stitching kernel调用和额外的global memory round-trip。替代方案包括：(a) naive stitching——在Convolution前fetch所有邻接patch的boundary pixels并concat→额外的memory movement开销完全抵消patch parallelism收益（Figure 5）；(b) Ghost Zone——直接replicate边界像素（在科学计算stencil中常用），但扩散模型从noise生成图像，相邻patch间无natural locality→产生明显patch边界artifact（Figure 6, PSNR仅9.54/SSIM 0.45）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Patch Edge Stitcher fused with GroupNorm kernel的执行流程（Figure 10d）：

```
每个Thread Block (TB) 处理一个patch:
1. TB加载patch数据到registers/shared memory
2. 执行GroupNorm计算：
   - 计算patch内mean和variance
   - normalize: (x - mean) / sqrt(var + eps)
   - scale and shift: gamma * x_norm + beta
3. 同步检查依赖metadata（在patch splitting时预计算）：
   - 该patch的哪些boundary pixels被邻接patch需要？
   - 例如：P0的右边界被P1需要，下边界被P2需要
   - 例如：P3的左边界被P2需要，上边界被P1需要
4. 将被需要的boundary pixels写入TB的shared memory
5. __syncthreads() // 等待所有TB完成normalization
6. 根据metadata定位目标patch：
   - 遍历该TB需要写入boundary的目标patch
   - 将shared memory中的boundary data写回global memory对应位置
   - 处理position diversity：
       * row boundary: 连续内存访问（对齐memory layout）
       * column boundary: 通过shared memory中转（避免irregular global memory access）
7. 目标patch的TB可直接读取准确的boundary值用于Convolution
```

关键设计：(1) Shared memory作为boundary exchange的中间缓冲——column boundary的irregular memory access通过shared memory中转local化，row boundary直接高效对齐读写；(2) 消除额外synchronization——stitching与GroupNorm在同一个fused kernel内完成，不需要kernel launch overhead或global barrier；(3) Direction & Position Diversity处理——通过pre-computed dependency metadata统一处理不同位置patches的不同stitching方向（四个边界的子集）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Patch Edge Stitcher在MixFusion中以CUDA C++实现，作为PyTorch custom operator集成到diffusion pipeline的Convolution模块前。Dependency metadata在patch splitting阶段预计算（记录每个patch的四邻接patch ID及需要的boundary范围→生成TB间通信计划）。在SDXL上的evaluation显示：(a) latency overhead minimal——Figure 5中PES的latency接近理论optimal（无stitching overhead的patched execution latency）；(b) quality显著优于alternatives——Table 4中PES w/ 4 patches达到PSNR 28.82/SSIM 0.88 vs Ghost Zone 9.54/0.45, vs Distrifusion 10.96/0.49；(c) SD3模型无Convolution→自然100% accuracy（PSNR inf/SSIM 1.0）。

涉及论文标题：
- MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models

## Batched Patch Cache Operations

术语是什么？通过联网搜索让回答具体和精准。

Batched Patch Cache Operations是MixFusion中用于高效管理patch级缓存的批量操作机制。由于patch-level caching需要在每个denoising step的每个block前后进行cache query（判断哪些patch可复用）/update（更新重算patch的结果）/insert（新增patch）/delete（清理退出GPU的patch），per-patch逐一操作会产生巨大开销——在SD3中每step约40-50ms含24 blocks，每个block的cache操作必须<2ms才能获得净收益。Batch coalescing将同一block中的所有patches的cache操作聚合为一次batch调用，通过三集合分类（Common Set/New Set/Expired Set）并行处理，amortize per-patch overhead。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Batched Patch Cache Operations的执行流程：

```
Cache系统以map结构存储（patch unique ID → cached_data），每个block维护独立cache。

输入：patch_indices[N] + intermediate_results[N]（N=当前batch的patch数）

1. Compare（索引比对）：
   input_set = set(patch_indices)
   cache_set = set(cache_pool.keys())
   common_set = input_set ∩ cache_set   // IDs同时存在→需verification
   new_set = input_set - cache_set      // IDs仅在input→插入
   expired_set = cache_set - input_set  // IDs仅在cache→删除

2. Slice（分片处理）：
   对common_set中每个patch：
     if MSE(intermediate_result, cached_data) > threshold:
         标记为recompute（mask=0）
     else:
         标记为reuse（mask=1）
   → 生成reuse_mask[N]

3. Compose（输出组合）：
   对common_set中reuse的patch：
     用cached_data替换intermediate_result对应位置
   对common_set中recompute的patch：
     保留新计算的intermediate_result
   → 生成masked_output[N]（所有position均有有效数据）

4. Update（批量更新）：
   对common_set中recompute的patch + new_set中所有patch：
     batch_insert(cache_pool, {id: new_data})
   对common_set中reuse的patch：
     batch_update_timestamp(cache_pool, ids)  // 仅更新时间戳
   对expired_set：
     batch_delete(cache_pool, ids)  // 对应patch已退出GPU
```

关键优化：(1) 所有cache操作在一次调用中完成→避免per-patch kernel launch overhead；(2) Common Set的verification（MSE comparison）使用vectorized GPU操作； (3) Expired Set检测使得cache自动清理退出GPU的patch，无需显式preemption支持。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MixFusion中用C++/CUDA实现batch cache manager：使用hash map存储patch ID到cached data的映射，batch操作通过GPU kernel并行处理（vectorized comparison, batch insert with coalesced writes）。Cache predictor（Random Forest, GPU端cuML）的输出reuse_mask直接作为batch cache operation的输入，形成predict→compute→combine→update的pipeline。batch size scaling study（Figure 17）显示cache management overhead随batch size modest增长（batch size 3→12, overhead仅增加约10%），验证了batching策略的scalability。

涉及论文标题：
- MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models

---

## OP-MCF (Outer-Product-friendly Masked Column-merging Format)

术语是什么？

OP-MCF 是 ASM-SpMM 提出的面向 ARM SME outer-product 执行模型的稀疏矩阵压缩格式。它按 SME vector length 将稀疏矩阵的连续行切分为 row window，在每个 window 内删除空列并将非零位置不重叠的列合并为一个 compressed slot，同时用 bitmask 记录每个原始列内的有效行位置，使 SME predicate register 可以只对有效非零位置参与 outer product 计算。与 GPU Tensor Core 格式（TCF、ME-TCF）的根本区别在于：OP-MCF 消除硬性 block padding，适应 SME 的 predicate-driven vector outer-product 语义。

从 kernel 调度角度拆解术语：

OP-MCF 的格式结构和 SpMM kernel 中对一个 row window 的处理：
```
// OP-MCF 格式结构（四数组）
RowWindowOffset[i]       → row window i 的起始行索引
ColumnOfRowWindow[i]     → row window i 的 compressed column 数量
SparseAtoB[j]            → compressed slot j 对应的原始列索引（指向 B 的行）
ColumnPositionMaskBit[j] → 64-bit bitmask，标识 slot j 内哪些行有有效非零

// SpMM kernel 对一个 row window 的处理
for each row_window_r:
    ZA_tile.clear()                           // 清零目标 ZA tile
    for j in compressed_slots_of(r):
        col_idx = SparseAtoB[j]               // 找到 B 的对应列
        sparse_vals = svld1(Z_reg, A_sparse[j])  // 加载 compressed sparse values
        pred = whilelt(ColumnPositionMaskBit[j])  // bitmask → predicate register
        B_tile = svld1(B[col_idx:col_idx+SVL])   // 加载 dense B tile
        ZA_tile = svmopa(ZA_tile, pred, pred, sparse_vals, B_tile) // outer product
        _svprfw(A_sparse[j+1])                // 预取下一 slot
    svst1_hor(C[row_window_r], ZA_tile)       // 写回输出
```

关键设计决策：
1. **Column compaction**：删除 row window 内的空列（全零列）→ 减少无效 slot 遍历
2. **Masked multi-column merging**：非零位置不重叠的多个原始列可以合并为一个 compressed slot → 提高每 slot 的有效非零密度。mean NNZ per slot 达到约 4-6（CSR 约 1-2）
3. **Bitmask 而非 explicit index**：ColumnPositionMaskBit 为 64-bit（对应 FP64 下 8 行 row window 内的每行），直接转 predicate register，无需 runtime 解压 row index
4. **与 GPU Tensor Core 格式的差异**：TCF 要求 left-aligned tile 和 fixed block padding（2×2/4×4），zero padding 浪费算力；OP-MCF 无对齐约束，predicate 按需屏蔽

术语一般如何实现？如何使用？

OP-MCF 需要一次性格式转换——遍历原始稀疏矩阵，按 SVL 划分 row window，对每个 window 做空列检测、列重叠分析和 bitmask 生成。转换时间复杂度 O(nnz + n_cols_per_window)。适合同一稀疏矩阵被重复执行的场景（GNN inference 中邻接矩阵固定、迭代 solver、超参搜索）。若矩阵频繁变化且复用次数少，格式转换成本可能难摊销。OP-MCF 的实现细节（压缩算法、bitmask 编码）论文通过四数组描述但未开源。

涉及论文标题：
- ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration

---

## SME Outer-Product SpMM Microkernel

术语是什么？

SME Outer-Product SpMM Microkernel 是 ASM-SpMM 的核心 kernel，利用 ARM SME 的 MOPA outer-product 指令将稀疏矩阵-稠密矩阵乘法（SpMM）映射到 ZA tile 的外积累加。与传统的 CPU SpMM kernel（使用 SVE/Neon SIMD 做 vector inner-product/dot-product）不同，该 kernel 利用 outer-product 语义直接形成 sparse vector × dense tile 的二维外积并累加到 ZA，避免将 SME 降级为普通 SIMD 使用。Kernel 同时整合多 ZA tile 并发、显式 prefetch pipeline 来隐藏 SME 指令的高延迟和稀疏访存的不规则性。

从 kernel 调度角度拆解术语：

SME outer-product SpMM microkernel 的伪代码执行流（FP64, SVL=512, 8 rows/window）：
```
// 输入：OP-MCF formatted sparse matrix A, dense matrix B
// 输出：C = A × B，row_window_size = SVL/64 = 8 rows

for each row_window_r in 0..num_row_windows:
    // Phase 1: Clear and prepare
    svzero_za()                                  // 清零所有 ZA tile
    z_reg_pool = allocate_Z_registers(4)          // 4 Z regs for operand streaming
    
    // Phase 2: Process compressed slots with multi-tile concurrency
    for each compressed_slot_s in row_window_r:
        // ---- 当前 slot 的计算 ----
        col = SparseAtoB[s]
        pred = ColumnPositionMaskBit[s]           // bitmask→predicate
        
        sparse_vec = svld1_f64(z_reg[0], A_values[s])    // 加载 sparse values
        B_tile = svld1_f64(z_reg[1], &B[col*ldb])         // 加载 B 的对应 tile
        
        // outer product accumulate: sparse_vec ⊗ B_tile → ZA_tile[slot % num_tiles]
        ZA[slot % 4] = svmopa_za64_f64_m(ZA[slot % 4], pred, all_true, 
                                          sparse_vec, B_tile)
        
        // ---- 预取下一 slot（与当前计算 overlap）----
        _svprfw(A_values[s+1])                    // 预取 sparse data
        _svprfw(&SparseAtoB[s+1])                 // 预取 column index
        _svprfw(&ColumnPositionMaskBit[s+1])      // 预取 bitmask
        _svprfw(&B[SparseAtoB[s+1] * ldb])        // 预取 dense B tile
    
    // Phase 3: Writeback
    for each active_ZA_tile:
        svst1_hor_za64_f64(C[row_window_r + tile_offset], ZA_tile)
```

关键执行特征：
1. **Outer-product 语义**：sparse_vec[i] × B_tile[j] 直接产生二维结果→累加到 ZA[i][j]。无中间 dot-product reduction
2. **Predicate mask 控制**：pred 确保只有有效非零行参与 outer product，零行对应的 ZA row 保持原值
3. **多 tile 流水线**：4 个 ZA tile 轮转使用，当前 tile 计算时前一 tile 结果的写回可与下一 slot 的加载重叠
4. **Prefetch 策略**：_svprfw 预取指令提前将下一 slot 的 sparse data/mask/column index/dense B 片段带入 L2 cache，将 LLC miss rate 从 30%-61% 降至 23%-48%

术语一般如何实现？如何使用？

实现依赖 ARM SME intrinsics（`arm_sme.h`）和 streaming SVE 模式。编译需 Clang 16.0+（支持 SME intrinsics 和 `-march=armv9-a+sme`）。Apple M4 上 key parameter：SVL=512→row window=8 rows（FP64），ZA 可划分 8 个独立 tile。使用注意：(1) SME 指令延迟高（M4 上 10-20 cycles），必须多 tile 并发+prefetch 隐藏；(2) ZA tile 间无直接数据通路，跨 tile 累加需 Z register 中转；(3) 对非常稀疏的 block（per-slot NNZ 接近 1），outer-product 的 tile 利用率低→应分配 SVE/NEON vector path。

涉及论文标题：
- ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration

---

## Hybrid SME/SVE Matrix-Vector Co-execution

术语是什么？

Hybrid SME/SVE Matrix-Vector Co-execution 是 ASM-SpMM 提出的混合执行策略：将稀疏矩阵中密度足够高的 block 分配给 SME outer-product path 以获得高吞吐，将低密度/碎片化 block 分配给 SVE/Neon vector path 以避免 SME 在低利用率 block 上空转。关键是 interleaved instruction scheduling——vector path 的计算被安排在 SME path 的固定执行窗口内（如 ZA tile 切换间隙或 prefetch 等待周期），使 vector 工作量被 SME 延迟隐藏。

从 kernel 调度角度拆解术语：

Hybrid kernel 的调度决策和执行流：
```
// Phase 1: Block partitioning 决策
for each row_window_r:
    density = nnz_in_window / (SVL * num_cols_in_window)
    if density < THRESHOLD:
        assign_to_vector_path(r)    // SVE/Neon path
    else:
        assign_to_sme_path(r)       // SME outer-product path

// Phase 2: Interleaved execution schedule
sme_blocks  = sorted_by_density_desc(sme_path_blocks)
vec_blocks  = sorted_by_density_asc(vector_path_blocks)

// 交错执行：SME path 执行一个 block → vector path 执行若干 block
while sme_blocks or vec_blocks:
    if sme_blocks:
        block = sme_blocks.pop()
        ZA_tile = sme_outer_product_spmm(block)      // SME 高延迟(10-20 cycles)
        // vector path 在 SME 执行期间并行执行
        while estimated_vec_time < sme_latency:
            if vec_blocks:
                vec_block = vec_blocks.pop()
                vec_result = sve_vector_spmm(vec_block)  // SVE/NEON low latency
    // 合并 SME 和 vector path 结果
    merge_to_output(ZA_tile, vec_result)
```

关键设计考量：
1. **SME/SVE microbenchmark 延迟估计**：预先 profile 不同密度 block 的 SME 和 SVE 执行时间，运行时用于 decide block allocation 和 interleave granularity
2. **Interleaved scheduling**：要求在 SME 固定执行窗口内完成 vector 工作量——vector block 太小则调度开销淹收益，太大则超越 SME 窗口产生额外等待
3. **Hybrid/theory ratio**：论文报告 hybrid 实际性能/理论性能为 0.78-0.90——差距来自 register partition（ZA 和 Z register 划分给 SME/vector path）、资源竞争（SVE 和 SME 共享 load/store 带宽）和 workload partition 开销
4. **收益场景**：在 rCA、FY-RSR、ddi、ppi 等包含大量低密度 block 的矩阵上，hybrid kernel 比 matrix-only (纯 SME path) 快 8%-18%

术语一般如何实现？如何使用？

实现需同时支持 ARM SME intrinsics 和 SVE/NEON intrinsics，在同一个 kernel function 内做条件分支和 instruction interleaving。Apple M4 的 E-core 上 SME 性能极低（约 P-core 1/8-1/16），hybrid path 的 SVE/NEON fallback 在 E-core 上尤为重要。LOOPS（同期工作，arxiv 2511.08158）采用了类似的 hybrid 思路：row-wise CSR 分配给 NEON + vector-wise BCSR 分配给 SME，在 M4 Pro 上达 9.93× (FP32) / 14.4× (FP64) over TACO。

涉及论文标题：
- ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration

---

## Heterogeneous Work Stealing Scheduling for SpMM

术语是什么？

Heterogeneous Work Stealing Scheduling 是 ASM-SpMM 提出的面向异构 ARM CPU（如 Apple M4 的 P-core+E-core）的 SpMM 多核动态负载均衡策略。与传统的 static task partitioning（按行/非零元/core 能力预分配）不同，它先做 hardware-aware initial task mapping，再用 progress monitoring 和 work stealing 在运行时动态再平衡。核心挑战：M4 的 P-core SME unit 和 E-core SME unit 性能不对称（P-core >> E-core），且不同 row window 的非零分布不可预测→静态分配必然导致某些 core 过早完成而等待。

从 kernel 调度角度拆解术语：

Work stealing 调度伪代码：
```
// Phase 1: Hardware-aware initial task mapping
row_windows = partition_by_svl(sparse_matrix_A)    // 按 SVL 切 row window
core_capability = [core.sme_peak_flops for core in cores]  // P-core > E-core
initial_assignments = weighted_round_robin(row_windows, core_capability)

// Phase 2: Parallel execution with work stealing
shared_state = {remaining_windows: List[RowWindow], 
                progress_per_core: int[cores], 
                lock: spinlock}

function execute(core_id):
    local_queue = initial_assignments[core_id]
    while local_queue or shared_state.remaining_windows:
        // Step 2a: Process local work
        while local_queue:
            window = local_queue.pop()
            result = spmm_kernel(window, B, ZA_tile)  // SME or hybrid kernel
            shared_state.progress_per_core[core_id]++
        
        // Step 2b: Try work stealing
        victim = find_slowest_core(shared_state)       // 最大剩余工作量
        if victim and shared_state.remaining_windows[victim]:
            lock(shared_state.lock)
            stolen = steal_half_windows(victim)         // 从 victim 窃取一半剩余 window
            unlock(shared_state.lock)
            local_queue.extend(stolen)
    
    barrier()
    merge_partial_results()                             // 合并各 core 的部分 C
```

关键设计决策：
1. **Hardware-aware initial mapping**：根据 core 的 SME 性能（P-core vs E-core）按比例分配 row window——P-core 获得更多 window
2. **Steal granularity**：以 row window 为最小窃取单位——row window 内部是不可分割的 SME kernel 调用单元
3. **Progress monitoring**：各 core 通过共享 counter 报告已完成 window 数，用于 victim selection
4. **Steal policy**：从完成度最低的 core 窃取其一半剩余 window——避免窃取过多导致新不均衡
5. **无 OS 依赖**：纯 user-space 实现，不依赖 OS 的异构 core 调度支持

术语一般如何实现？如何使用？

实现为 C/C++ pthread 或 std::thread 多线程程序，使用 spinlock 保护共享队列（window 粒度 coarse enough，lock contention 低）。LX2（对称 SME，12 核等效）上 12 thread vs 2 thread 达 8x-11x scaling（static scheduling 有效）。M4（异构 SME）上多线程增益受 SME unit 数量限制（仅 2 个 SME unit 对应 2 个 cluster），但 work stealing 克服了 E-core SME 性能低的问题——E-core 完成后可窃取 P-core 的任务使总体时间接近 P-core 单独完成的分摊时间。

涉及论文标题：
- ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration

## ArbitWeave

术语是什么？

ArbitWeave 是 Drawloom 提出的面向 GPU Tensor Core 的 SpMV 稀疏矩阵映射策略，支持将任意稀疏矩阵灵活映射到不同 shape 的 TC 和 SpTC 硬件单元。核心设计：(1) 根据 TC shape 的结构比 V = mma_m / mma_n 将稀疏矩阵切分为 V-width row strip（如 V=2 时每 2 个连续行组成一个 strip），作为映射到 TC block 的基本计算单元；(2) 按每个 row strip 的非零元数量（NNZ）分为 Long Mapping（nnz > T1，独占 TC block）、Medium Mapping（T2 < nnz ≤ T1，多个 strip 聚合到 TC block）、Short Mapping（nnz ≤ T2，利用 2:4 sparsity 映射到 SpTC block）三种策略；(3) 阈值 T1 = (mma_m / V) × mma_k × WarpLoad, T2 = mma_k 由 TC shape 和 warp 负载导出。ArbitWeave 的核心创新在于同时调度 TC 和 SpTC 处理不同稀疏度区域。

从kernel调度角度拆解术语：

ArbitWeave 的映射计算过程（以 m16n8k16 TC, v=2 为例）：

```
// 步骤1: 确定 TC shape 和 strip
mma_m=16, mma_n=8, mma_k=16, V = mma_m/mma_n = 2
row_strip_width = V = 2   // 每2行组成一个strip

// 步骤2: 按NNZ分类每个row strip
for each row_strip in sparse_matrix:
    nnz = count_nonzeros(strip)
    if nnz > T1:
        classify as Long Mapping   // 独占TC blocks
    elif nnz > T2:
        classify as Medium Mapping // 多strip共享TC block
    else:
        classify as Short Mapping  // 映射到SpTC (2:4 sparsity)

// 步骤3: Long/Medium mapping → dense TC
// 将row strip内的非零元按列压缩(去除零值列)
// 对齐到TC block的B矩阵列位置
// 结果Y沿TC output matrix对角线输出
// 压缩后Ecomp = #nonzeros / (mma_m × mma_k)

// 步骤4: Short mapping → SpTC
// 非零元配对为2:4 pattern(每4位置最多2非零)
// 50%压缩+2-bit metadata编码位置
// 向量X按remapped CID重排到SpTC B矩阵
// SpTC硬件跳过零值计算→2x throughput
```

ArbitWeave 相比 DASP 的 m8n8k4 固定映射：在 A100/H100 上选择 m16n8k16（真TC硬件执行→避免ALU fallback），同时通过 Short Mapping 首次用 SpTC 计算极短行（DASP 中 short row 只能 fallback 到 CUDA Cores 计算）。

术语一般如何实现？如何使用？

ArbitWeave 实现为 CUDA C++ kernel 的预处理+运行时映射：预处理阶段扫描矩阵每个 row strip 的 NNZ 决定 Long/Medium/Short category→构建对应的 ZCF 索引结构（longPtr/mediumPtr/shortCid）→kernel launch 时 warp 根据 ZCF 指针索引正确的 TC block 数据→Long row 结果 intra-warp shuffle reduce + inter-warp reduction kernel 归约。Short Mapping 消融实验显示开启 SpTC short mapping 在 M12（econ_fwd500）上提速 1.54×、M17（cop20k_A）上提速 1.70×。

涉及论文标题：
- Exploiting Efficient Mapping and Pipelined Execution for Accelerating SpMV on Tensor Cores

## Zig-zag Chained Format (ZCF)

术语是什么？

Zig-zag Chained Format (ZCF) 是 Drawloom 为 Tensor Core 和 Sparse Tensor Core 加速设计的稀疏矩阵存储格式，替代 CSR 等传统格式。ZCF 按行 strip 的 NNZ 分类构建三种数据布局：(1) Long ZCF：longPtr（TC block 计数 per strip）+ longCid（列索引）+ longVal（非零值），Long row strip 独占完整 TC blocks；(2) Medium ZCF：mediumPtr（TC block 计数 per window）+ mediumCid + mediumVal，多个 medium row strip 聚合到一个 TC block；(3) Short ZCF：shortCid（remapped 列索引）+ shortVal（2:4 压缩值），直接兼容 SpTC 的 structured sparse 格式。ZCF 的核心特征是 zcf_value_stride 参数控制向量化访存粒度——FP16 stride=8, FP32/FP64 stride=4——使每 thread 可执行 128-bit vectorized load 对齐 GPU memory transaction 粒度。

从kernel调度角度拆解术语：

ZCF 的构建和访问伪代码：

```
// ZCF构建 (preprocessing)
Input: CSR matrix A (row_ptr, col_idx, values)
       TC shape: mma_m, mma_n, mma_k
       Thresholds: T1, T2

row_strip_width = mma_m / mma_n  // e.g., 2

// 按NNZ per row重排序矩阵行
sorted_rows = sort_rows_by_nnz(A)

for strip_id, (row_i, row_{i+1}) in enumerate(sorted_rows):
    nnz = count_nnz(strip)
    if nnz > T1:    // Long
        longVal.append(values[strip])
        longCid.append(col_idx[strip])
        longPtr.append(num_TC_blocks)
    elif nnz > T2:  // Medium  
        if window_full:
            mediumVal.append(values[window])
            mediumCid.append(col_idx[window])
            mediumPtr.append(num_TC_blocks)
    else:           // Short (SpTC)
        remap_2_4_sparsity(values, col_idx)  // 50% compress
        shortVal.append(compressed_values)
        shortCid.append(remapped_column_ids)

// ZCF kernel访问 (runtime)
// 向量化访存: zcf_value_stride对齐128-bit transaction
// zcf_value_stride = 8 (FP16) or 4 (FP32/FP64)
for each TC block in warp assignment:
    // 128-bit vectorized load (每个thread加载连续stride个元素)
    val_tile[i] = longVal[block_base + thread_id * stride]
    cid_tile[i] = longCid[block_base + thread_id * stride]
```

相比 DASP 的离散、非 coalesced 访存，ZCF 通过 zig-zag chain 布局保证 TC block 内数据连续→减少 IMAD（memory index 计算）指令 67.8%、branch 指令 50%、memory bandwidth 提升 48.3%。

术语一般如何实现？如何使用？

ZCF 实现为 CUDA C++ 预处理函数（CPU 端运行一次，overhead 可被多次 SpMV 迭代摊销）。预处理：读取 CSR→按 NNZ per row 排序→按 T1/T2 阈值分类→构建三种 ZCF 数组→输出为 Drawloom kernel 的输入。kernel 中 warp 通过 ZCF 的 ptr 数组索引正确的 Cid/Val 数据块执行 vectorized load（使用 `float4`/`double2` 等 128-bit 类型强制 coalesced global memory transaction）。FP16 场合 stride=8 使每 warp（32 thread）一轮访存 load 256 个值（32×8）。

涉及论文标题：
- Exploiting Efficient Mapping and Pipelined Execution for Accelerating SpMV on Tensor Cores

## Multi-stage Register Pipeline (for SpMV on Tensor Cores)

术语是什么？

Multi-stage Register Pipeline 是 Drawloom 为解决 SpMV 中 memory-compute 串行化导致的 warp stall 而提出的 GPU kernel 流水线设计。将 SpMV 的四大操作步骤——Fetch Sparse A (FS)、Fetch Column ID (FCid)、Load Vector X (LX)、TC Computation (Comp)——重构为五个流水线阶段：FillSMEM（async-copy 将 sparse A + CID 从 GMEM 异步拷贝到 SMEM）→ FillREG（从 SMEM 索引 prefetch vector X 到寄存器）→ Comp（TC MMA 计算）→ EmptySMEM（完成剩余 SMEM→REG 数据搬移）→ EmptyREG（完成剩余计算）。通过两个可调参数 delaySMEM 和 delayREG 控制流水线 overlap 深度：delaySMEM 控制 GMEM→SMEM 的 double buffering 阶段数，delayREG 控制 SMEM→REG + computation 的重叠阶段数。

从kernel调度角度拆解术语：

Multi-stage Register Pipeline 的伪代码执行流程：

```
// Pipeline参数
delaySMEM = 1  // double buffering: 2个SMEM buffer交替
delayREG = 2   // 3个REG set轮流使用

// 5阶段流水线
int stage = 0;

// 阶段1: FillSMEM — 填充SMEM buffer
while (stage < delaySMEM + 1):
    async_copy_g2s(FS[stage], &SMEM_A[stage % 2])  // sparse A
    async_copy_g2s(FCid[stage], &SMEM_CID[stage % 2])  // column IDs
    pipeline_commit()
    stage++

// 阶段2: FillREG — SMEM→REG + Load X
while (stage < delaySMEM + delayREG + 1):
    pipeline_wait_prior(stage - delaySMEM - 1)  // 等待SMEM就绪
    for tid in warp:
        cid = SMEM_CID[(stage-delaySMEM) % 2][tid]
        REG_X[stage % (delayREG+1)][tid] = LDG_instruction(X_base + cid)  // PTX LDG
    stage++

// 阶段3: Comp — TC Computation
while (stage < total_stages):
    pipeline_wait_prior(stage - delaySMEM - 1)
    for tid in warp:
        cid = SMEM_CID[(stage-delaySMEM) % 2][tid]
        REG_X[stage % (delayREG+1)][tid] = LDG_instruction(X_base + cid)
    
    // TC MMA: 使用前一阶段的REG数据
    TC_MMA(REG_A[stage-delaySMEM-delayREG], 
           REG_X[(stage-delayREG) % (delayREG+1)], 
           REG_C)
    stage++

// 阶段4-5: EmptySMEM & EmptyREG — 流水线排空
while pending_stages > 0:
    pipeline_wait_prior(...)
    TC_MMA(...)
    pending_stages--
```

2-stage pipeline（delaySMEM=1, delayREG=0）仅重叠GMEM→SMEM传输与后续操作；Multi-stage（delaySMEM+delayREG>1）进一步解耦SMEM→REG和计算，使多级寄存器集轮转消除 data dependency stall。

术语一般如何实现？如何使用？

实现依赖 A100+ 的 async-copy（`__pipeline_memcpy_async` + `__pipeline_wait_prior`）和 PTX LDG 指令（`ld.global.nc` 加载 X 向量绕过 L1 cache）。double buffering 通过两个 SMEM buffer 交替使用实现。REG prefetch 通过 delayREG 控制的多个寄存器集合轮转实现。在 Drawloom 的消融实验中，v4（+Multi-stage Pipeline）相比 v3 平均提速 1.46×（mip1 达 5.68×），warp stall 改善 3.02×-3.13×（在 majority representative matrices），memory throughput 提升 2.61×-2.75×。Pipeline 设计的关键限制是 SpMV 每次 memory access 只触发一个 TC 操作（与 SpMM 的大 tile 多 TC 操作不同），因此需要多级 REG pipeline 进一步重叠 short-latency TC computation 与 irregular X vector access。

涉及论文标题：
- Exploiting Efficient Mapping and Pipelined Execution for Accelerating SpMV on Tensor Cores

## TC Result Efficiency (Eres) and TC Computation Efficiency (Ecomp)

术语是什么？

Eres（TC Result Efficiency）和 Ecomp（TC Computation Efficiency）是 Drawloom 定义的两个衡量 Tensor Core 在 SpMV 计算中利用效率的指标。Eres = (# of valid output elements) / (mma_m × mma_n)，衡量 TC 指令产生的输出中有多少实际贡献给最终向量 Y。Ecomp = (# of non-zero elements in A) / (mma_m × mma_k)，衡量 TC block 内实际参与计算的非零元素比例。两者分别从输出有效性（Eres）和计算有效性（Ecomp）两个维度刻画 TC 在稀疏计算中的利用效率。高 Eres 低 Ecomp 意味着大量零值参与计算但输出位置合理；低 Eres 高 Ecomp 意味着计算高效但输出位置错配。两者需同时优化。

从kernel调度角度拆解术语：

Eres 和 Ecomp 的计算例子（以 m4n2k4 TC，V=2 为例）：

```
// 场景：row strip (2行) 含 5 个非零元
// TC shape: mma_m=4, mma_n=2, mma_k=4
// V = mma_m/mma_n = 2

// Ecomp计算：
// 输入：5 nonzeros → 填充到 4×4 TC A矩阵
// nnz_in_block = 5
// Ecomp = 5 / (4 × 4) = 5/16 = 31.25%

// Eres计算：
// 输出：TC输出 4×2=8 个结果位置
// 其中4个位置对应实际Y元素（valid outputs沿对角线）
// valid = 4
// Eres = 4 / (4 × 2) = 4/8 = 50%

// DASP m8n8k4: Ecomp≈60%, Eres 较高
// DASP naive m4n2k4(v=2): Ecomp 增加但 Eres 降至 25%
// Drawloom ArbitWeave: Ecomp≈60% 保持，Eres 显著提升
```

Drawloom 的 ArbitWeave 通过 column compression 保持高 Ecomp 同时利用大 TC shape 改善 Eres。SpMM naive approach SpMV 的 Ecomp 仅 11.78%（因稀疏导致大量零值填充到 TC block）——说明 SpMM 优化不能直接用于 SpMV。

术语一般如何实现？如何使用？

Eres 和 Ecomp 是分析性指标，用于指导 TC shape 选择和 mapping 策略优化。在实际实现中，preprocessing 阶段可基于这两个指标预估每种 TC shape 的理论效率，选择最优的 TC shape。Drawloom 在 FP16 下选择 m16n8k16（Ecomp 平均 60.15% vs DASP 61.20%，Eres 显著提升），FP64 下选择 H100 的 m16n8k16 FP64 TC。

涉及论文标题：
- Exploiting Efficient Mapping and Pipelined Execution for Accelerating SpMV on Tensor Cores

## Two-Level Load Balancing (for SpMV on TCs)

术语是什么？

Two-Level Load Balancing 是 Drawloom 为解决 SuiteSparse 数据集中稀疏矩阵行间 NNZ 分布不均导致的 GPU 线程负载不均衡而设计的两级负载均衡策略。(1) Matrix-level：根据 NNZ per row 用两个阈值 T1=(mma_m/V)×mma_k×WarpLoad 和 T2=mma_k 将矩阵行分为 Long/Medium/Short 三类，不同类采用不同 mapping 策略（Long独占TC block、Medium聚合、Short用SpTC）；(2) Warp-level：Long row 生成的 TC blocks 通过 WarpLoad 参数均匀分配至各 warp 执行，Long Mapping 的结果需 inter-warp reduction kernel 归约。

从kernel调度角度拆解术语：

Two-Level Load Balancing 伪代码：

```
// Matrix-level: 按NNZ分类
for row in sparse_matrix:
    nnz = nnz_per_row[row]
    if nnz > T1:
        row_category[row] = LONG
        tc_blocks = ceil(nnz / (mma_m * mma_k))  // 独占TC blocks
    elif nnz > T2:
        row_category[row] = MEDIUM  
        // 多row strip聚合到shared TC block
    else:
        row_category[row] = SHORT  // SpTC

// Warp-level: Long row 的TC blocks分配
// 每个thread block含4 warps (warp_size=32)
WarpLoad = tunable_parameter  // 每warp分配的TC block数
total_long_blocks = sum(tc_blocks for LONG rows)
blocks_per_warp = total_long_blocks / (num_warps)  // 均衡分配

for warp_id in range(num_warps):
    start_block = warp_id * WarpLoad
    end_block = start_block + WarpLoad
    for tc_block in [start_block, end_block):
        compute_tc_mma(tc_block)  // 每个warp处理相近数量的TC blocks

// Long结果归约
if row_category == LONG:
    warp_shuffle_reduce(partial_Y)  // warp内shuffle归约
    atomic_add(global_Y[row_id], partial_Y)  // 或 launch reduction kernel
```

T2=mma_k 的合理性：NNZ < mma_k 的行无法填满一个 TC block 的 k 维度，不适合用 dense TC（浪费算力），故分配给 SpTC Short Mapping。

术语一般如何实现？如何使用？

实现为 CUDA kernel 的 warp 分配逻辑和独立的 reduction kernel（Long row warp 间归约）。WarpLoad 是可调参数，论文中 set to 2。消融实验显示 ZCF 的 Two-Level Load Balancing 将 CSR 的 if-else thread selection 替换为向量化访存，branch 指令减少 50%。

涉及论文标题：
- Exploiting Efficient Mapping and Pipelined Execution for Accelerating SpMV on Tensor Cores

---

## Vector-driven (Column-major) SpMSpV Paradigm

术语是什么？

Vector-driven (向量驱动) SpMSpV 执行范式（也称 column-major 或 push paradigm）是一种仅遍历稀疏输入向量非零元的 SpMSpV 计算模式。与 row-major (matrix-driven) 遍历矩阵所有行不同，vector-driven 以 x 的每个非零元为起点，索引矩阵对应列，生成 partial products 并 scatter 到输出向量。流程分为两个阶段：**Fetch 阶段**——从 CSC 格式矩阵中加载非零列对应的 indices 和 values；**Write-back 阶段**——将 partial products (mat_val × vec_val) 按 row index scatter 累加到输出向量 y。Vector-driven 范式能充分利用向量稀疏性——计算量与 x 的非零元数成比例而非矩阵大小，在 x 极度稀疏时显著优于 row-major。

从kernel调度角度拆解术语：

Vector-driven SpMSpV 的 GPU kernel 伪代码：

```
// GPU Kernel: Vector-driven SpMSpV
Input: A in CSC (col_ptr, indices, values); x_sparse (idx[], val[]); n_active
Output: y[]

// === Fetch Phase (Load Balancing Variants) ===
// Variant 1: Direct-mapped — 每个活跃列分配给一个 CTA
for each active_idx i in [0, n_active) mapped to CTA tid:
    col = x_sparse.idx[i]
    load column segment: col_ptr[col] to col_ptr[col+1]

// Variant 2: Block-mapped (Gunrock-style) — 多个短列聚合到一个 CTA
CTA groups multiple short columns
for each group assigned to CTA:
    block_prefix_scan over column lengths
    load indices and values for the group

// Variant 3: Global-mapped (merge-based) — 全局 prefix-scan 按 NNZ 均匀分配
for each nonzero in x, record column length
global_inclusive_scan over lengths → total_NNZ
partition total_NNZ evenly across CTAs

// === Write-back Phase ===
for each (row_idx, mat_val) in fetched segment:
    partial = mat_val × x_sparse.val[col]
    // Strategy A: Atomic — 直接 global atomic
    atomicAdd(&y[row_idx], partial)
    // Strategy B: Sort-reduce — buffer + sort + reduce
    buffer.append(row_idx, partial)
    sort(buffer by row_idx); reduce duplicates
    // Strategy C: Hash aggregation (VDHA)
    hash_insert(shared_hash_table, row_idx, partial)
```

负载均衡策略对比：Direct-mapped 无 prefix-scan 开销但负载不均（NaiveSpMSpV）；Block-mapped 多短列聚合到 CTA（Gunrock）；Global-mapped 按 NNZ 均匀分配负载最均衡但有 prefix-scan 开销（merge-based SpMV）。

术语一般如何实现？如何使用？

GPU 实现要点：矩阵需以 CSC 格式存储（col_ptr、row_indices、values 数组）；输入向量以 sparse format (idx[], val[]) 存储；性能瓶颈在 write-back 阶段的 many-to-one scatter pattern 导致的 address contention 和 uncoalesced stores。Adaptive SpMSpV 框架根据矩阵特征和向量稀疏度在多种 vector-driven kernel 与 row-major kernel 间动态选择。适用于图分析（BFS, PageRank）、SNN spike propagation、科学计算中的稀疏线性代数。

涉及论文标题：
- VDHA: Vector-Driven Hash Aggregation for Sparse Matrix-Sparse Vector Multiplication on GPUs

---

## Hash-based Aggregation for SpMSpV Write-back

术语是什么？

Hash-based aggregation (基于哈希的聚合) 是 GPU SpMSpV 中 write-back 阶段的一种策略：用每个 CTA 私有的 shared-memory hash table 暂存和累积 partial products，在 hash table 接近满时批量 flush 到 global memory。相比 atomic write-back（每个 partial product 发 global atomicAdd）和 sort-based write-back（全局排序后 reduce），hash-based 方法通过 local aggregation 减少 global write conflicts，并通过 bucket-order flush 改善 memory coalescing。该策略受 SpGEMM 中 hash-based accumulation 启发——SpGEMM 每个 output row 维护小 hash table 消除 intra-row conflicts。但 SpMSpV 缺少 SpGEMM 的自然 row partitioning，所有 intermediate updates 汇聚到单一 output vector，因此 hash table 只能消除 intra-block conflicts，cross-block conflicts 仍需 global atomics。

从kernel调度角度拆解术语：

VDHA 的 hash-based insertion 伪代码：

```
function Insert(H, idx, val):
    h ← idx % TABLE_SIZE              // modulo hash，保留 row index 低位
    cnt ← 0
    while cnt < FALLBACK_ITER:
        old ← atomicCAS(&H.key[h], -1, idx)   // -1 = empty slot
        if old == -1 or old == idx:
            UpdateHash(H.val[h], val)          // accumulate value
            return
        h ← (h + STRIDE) % TABLE_SIZE          // linear probing with fixed stride
        cnt ← cnt + 1
    Fallback(idx, val)               // probe 超限 → global atomicAdd fallback
```

关键设计：
- **Modulo hash**: idx % TABLE_SIZE，保留低位使 bucket order flush 改善 coalescing
- **Linear probing with stride**: (h + C) % TABLE_SIZE，相比 (h+1) 降低 locally distributed nonzeros 碰撞概率
- **FALLBACK_ITER**: 限制 probing 次数，防止无限循环导致 warp divergence
- **Update 策略**: 同列 segment 直接用 `H.val[idx] += val`（无 atomic）；跨列需 atomicAdd
- **局部聚合率 ρ(T) = 1 - F(T)/N**：衡量 hash table 吸收的 update 比例。在 it-2004 上 column decomposition+reorder 后 ρ 从 51.0% 提升至 89.8% (T=2048, density=100%)
- **Table size**: 2048 entries (16KB)，平衡 aggregation 效果和 occupancy

术语一般如何实现？如何使用？

每个 CTA 维护独立 shared-memory hash table（不跨 CTA 共享）。`atomicCAS` 实现 lock-free concurrent insertion。Linear probing 提供良好 GPU cache locality。Fallback 保证正确性同时避免无限循环。有效性依赖矩阵结构：需 sufficient temporal locality (ρ) 使 aggregation 收益超过 hash overhead。VDHA 证明通过 column decomposition + reordering + fetch-compute-writeback pipeline，hash-based write-back 在 web graphs 上实现 1.41× geomean speedup。

涉及论文标题：
- VDHA: Vector-Driven Hash Aggregation for Sparse Matrix-Sparse Vector Multiplication on GPUs

---

## Column Decomposition with Reordering

术语是什么？

Column decomposition with reordering (列分解与重排序) 是 SpMSpV 预处理阶段的 locality enhancement 技术。真实世界图（社交网络、web graphs）的列长分布极度偏斜：少数 long column 包含大多数非零元。例如 it-2004 中 1.4% 的列（≥256 NNZ）占总 NNZ 的 70%+，平均长度 1403 vs 全局均值 27.9。Long column 内部 row indices 不重叠——hash table 无法在列内提供聚合收益。跨列 segments 可能共享重叠 row ranges。Column decomposition 将 long column 切分为固定大小 segments，Reordering 按 segment 首 row index 排序增强跨列 locality。

从kernel调度角度拆解术语：

VDHA Vector Processing 伪代码：

```
// Column classification + splitting + reordering
segments = []
for each active (col, vec_val) in x_sparse:
    start = col_ptr[col], len = col_ptr[col+1] - start
    if len < LEN_THRES:               // short column, LEN_THRES=128
        segments.append((start, len, vec_val))
    else:                              // long column → split
        for i in [0, ceil(len/SPLIT_SIZE)):   // SPLIT_SIZE=256
            seg_start = start + i * SPLIT_SIZE
            seg_len = min(len - i*SPLIT_SIZE, SPLIT_SIZE)
            segments.append((seg_start, seg_len, vec_val))

// Reorder: 仅排序 long-column segments 的 metadata
long_segs = [s for s in segments if from long column]
sort(long_segs by A.indices[s.start])     // O(S log S), S << N
short_segs = [s for s in segments if short column]
segment_queue = short_segs + long_segs    // block-mapped to CTAs
```

关键参数：LEN_THRES=128 区分 short/long column；SPLIT_SIZE=256 使每个 segment 适应 CTA 256 threads + hash table。Reordering 成本 O(S log S) 远小于排序所有 nonzeros O(N log N)。Short columns 不重排序——数量可能接近输入向量长度，排序开销过大。

在 it-2004 上的效果（T=2048, density=100%）：ρ 51.0%→89.8%，γ 0.744→2.607，atomic-unit utilization 22.99%→12.82%。

术语一般如何实现？如何使用？

CPU 预处理步骤（轻量级，可在线执行）。仅排序 segment metadata 而非 nonzeros 本身。受 RoDe (row decomposition-based SpMV) 启发——将类似思想从行应用到列并增加重排序。适用于偏斜列长分布的矩阵（power-law），对角线规则矩阵收益有限。

涉及论文标题：
- VDHA: Vector-Driven Hash Aggregation for Sparse Matrix-Sparse Vector Multiplication on GPUs

---

## Fetch-Compute-Writeback Pipeline for GPU Kernels

术语是什么？

Fetch-compute-writeback pipeline (取数-计算-写回流水线) 是一种 GPU kernel 执行重叠技术：将传统 fetch→writeback 两阶段重构为 fetch→compute (hash aggregation)→writeback 三阶段，通过 double buffering 和异步 copy (cp.async) 实现当前 segment 的 hash computation 与下一 segment 的 memory fetch 重叠，隐藏 hash computation latency。该技术利用 write-back 阶段的高 memory stall（>45% long scoreboard waits）——即使高 occupancy 也无法隐藏 uncoalesced memory access latency，因此可在 memory stall 期间执行有用计算。

从kernel调度角度拆解术语：

VDHA pipeline 伪代码：

```
Input: segments seg[]; SMEM buffers buf[0], buf[1]; hash table H

cp.async.Fetch(seg[0], buf[0])               // 异步加载首个 segment
__syncthreads()
for i = 0 to N_segs-1:
    if i != N_segs-1:
        cp.async.Fetch(seg[i+1], buf[(i+1)%2]) // 异步预取下一 segment
    indices, values ← buf[i%2]                  // 当前 segment 已在 SMEM
    for each (row, mat_val) in segment:
        hash_insert(H, row, mat_val × vec_val)  // hash aggregation
    __syncthreads()
    if hash_full(H) or i == N_segs-1:
        flush(H, y)                    // bucket-order flush
    cp.async.wait_group()               // 确保下一 segment 已就绪
```

Timeline 重叠示意：
```
Seg 0: [Fetch seg0][===== Hash Compute seg0 =====][Flush]
Seg 1:             [Fetch seg1][===== Hash Compute seg1 =====][Flush]
Seg 2:                         [Fetch seg2][===== Hash Comp... ]
                   ↑ overlap: fetch seg(i+1) || hash compute seg(i)
```

效果：stall ratio >45%→~15%，hash computation cost 16.7%→12.3%。

术语一般如何实现？如何使用？

需要 GPU 架构支持异步 copy（NVIDIA Ampere SM80+ cp.async，AMD asynchronous copy units）。Double buffering 需要足够 SMEM 容纳两个 buffer + hash table。效率取决于 compute 和 memory fetch 的耗时比例——memory stall 越严重 overlap 收益越大。类似技术广泛用于 FlashAttention async copy、SpGEMM kernel memory-compute overlap、general GPU kernel optimization。

涉及论文标题：
- VDHA: Vector-Driven Hash Aggregation for Sparse Matrix-Sparse Vector Multiplication on GPUs

---

## Write-back Strategies in GPU SpMSpV

术语是什么？

GPU SpMSpV 的 write-back 阶段有三种基本策略处理 partial products 到输出向量的 many-to-one scatter accumulation：

1. **Atomic write-back**: 每个 partial product 通过 `atomicAdd(&y[row_idx], partial)` 直接累加。优点：实现简单、无额外内存。缺点：many-to-one scatter 导致 severe address contention，uncoalesced stores 浪费带宽（A100 上 sparsity=0.1 时 ~270 GB/s，仅 17% peak），write-back 占 runtime >30%。

2. **Sort-based write-back** (FastSpMSpV): buffer 所有 (row_idx, val) pairs → global sort → sequential reduce。优点：完全避免 atomics。缺点：sort 阶段极慢（~43.3 GB/s），占 runtime >70%，需大临时 buffer。

3. **Hash-based write-back** (VDHA): CTA-private shared-memory hash table → local aggregation → bucket-order flush。优点：减少 conflicts（atomic-unit utilization 22.99%→12.82%），改善 coalescing（γ 0.744→2.607），hash cost 可通过 pipeline 隐藏。缺点：额外 SMEM 消耗，缺乏 locality 的矩阵收益有限。

4. **Segment-sum write-back** (Swift): warp-level shared-memory segment sum → 仅 segment 起点的线程做 atomicAdd。利用 positionIdx/offsetIdx 辅助索引，在 shared memory 中将同 row_idx 的 partial 预合并。优点：无需 hash probing（确定性索引，无 collision/fallback），segment sum 在 regular block 中极高效。缺点：依赖列排序预处理生成辅助索引；irregular block 不可用，仍需 direct atomicAdd。

从kernel调度角度拆解术语：

三种策略对比（it-2004, sparsity=0.1, A100）：

| 策略 | Bandwidth | Runtime占比 | Memory开销 |
|------|-----------|------------|-----------|
| Atomic | ~270 GB/s (A100, sparsity=0.1) | >30% | 无额外buffer |
| Sort | ~43 GB/s | >70% | 临时buffer存储全部pairs |
| Hash | improved | reduced | 16KB SMEM/CTA + flush buffer |
| Segment Sum | improved (确定性归约) | reduced (仅segment起点写回) | positionIdx/offsetIdx + SMEM buffer (~128B/warp) |

VDHA motivation benchmark：随着密度增加，atomic bandwidth 下降（sparsity 0.2→251 GB/s），sort bandwidth 始终~45 GB/s。Hash-based 通过 local aggregation 取两者之长：避免 sort 全局开销 + 减少 atomic conflict 次数。

术语一般如何实现？如何使用？

选择指导：Atomic 适用 conflict 少的矩阵或极度稀疏向量；Sort 适用需完全避免 atomics 场景但通常不如 hash；Hash 适用有 locality 的矩阵（small-world graphs），需足够 SMEM 和 fallback。Segment sum 适用列结构规整的矩阵（已做列排序+regular blocking），无需处理 hash collision 但依赖预处理。Adaptive SpMSpV 使用 ML-based kernel selector 在 atomic/sort/row-major 间选择；VDHA 提供 decision tree predictor（91.3% accuracy）判断 hash 是否更优。

涉及论文标题：
- VDHA: Vector-Driven Hash Aggregation for Sparse Matrix-Sparse Vector Multiplication on GPUs
- Swift: High-Performance Sparse-Dense Matrix Multiplication on GPUs

## Cross-precision GEMM（跨精度矩阵乘法）

术语是什么？通过联网搜索让回答具体和精准。

Cross-precision GEMM（跨精度通用矩阵乘法）是指在一次矩阵乘法C=A×B中，操作数A和B使用不同数值精度的计算模式。在RoMeo的混合精度量化场景中，weight和activation可能各自处于INT4或INT8精度，产生四种精度组合：W4A4（weight INT4 × activation INT4）、W4A8（weight INT4 × activation INT8）、W8A4（weight INT8 × activation INT4）、W8A8（weight INT8 × activation INT8）。这四种组合的计算吞吐量不同：INT4 Tensor Core提供~8×FP16峰值吞吐，INT8提供~2×FP16。Cross-precision GEMM使用INT32累加器防止混合精度累加时overflow。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

RoMeo中cross-precision GEMM的separate-kernels实现（以Qwen3-8B Down_proj [8192×4096×4096]为例，batch=64）：

```
// 四种separate kernel，各自操作dense uniform-precision矩阵
// 每个kernel使用CUTLASS实现，内部含software pipeline

Kernel W4A4: // 主体（~88%计算量），最高吞吐
  Input:  A_int4 [M_normal, K] packed, B_int4 [K, N_normal] packed
  // 使用m16n8k32 tiling, INT4 Tensor Core
  for tile_k in 0..K/32:
    cp.async A_tile[tile_k] → smem_A
    cp.async B_tile[tile_k] → smem_B  
    cp.async.wait(Nstage-1)
    mma.sync.aligned.m16n8k32 smem_A, smem_B → accum
  Output: C_W4A4 [M_normal, N_normal] FP32

Kernel W4A8: // activation outlier rows × normal weight cols
  Input: A_int8 [M_outlier, K], B_int4 [K, N_normal] packed
  // INT4×INT8 cross-precision: B在SMEM内cast到INT8
  for tile_k in 0..K/32:
    cp.async A_tile, B_tile → smem
    // INT4 → INT8 casting in shared memory
    // 使用两条binary arithmetic指令而非type conversion
    smem_B_int8 = cast_int4_to_int8(smem_B_int4)  
    cp.async.wait(Nstage-1)
    mma.sync.aligned.m16n8k16 smem_A_int8, smem_B_int8 → accum  
  Output: C_W4A8 [M_outlier, N_normal] FP32

Kernel W8A4: // normal activation rows × outlier weight cols
  // 类似W4A8，A在SMEM内cast

Kernel W8A8: // outlier交叉部分（~0.25%计算量，但需更高精度）
  Input: A_int8 [M_outlier, K], B_int8 [K, N_outlier]
  // 纯INT8 GEMM，使用INT8 Tensor Core
  // Shared memory需求为INT4-INT4 kernel的2倍
```

关键设计选择separate-kernels而非fused-kernel：
- INT4-INT4 kernel：shared memory小→compiler可用更多register做loop unrolling提升ILP
- INT8-INT8 kernel：shared memory大→occupancy由SMEM限制，compiler自动减少register使用
- Fused kernel无法为不同精度组合独立分配on-chip资源→suboptimal

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：
- CUTLASS模板库提供INT4/INT8 GEMM的参考实现，RoMeo在CUTLASS基础上customize supporting mixed input types
- 类型转换策略：在shared memory内用binary arithmetic（IADD3+LOP3）完成INT4→INT8 casting，避免昂贵的PTX type conversion指令
- Software pipeline（Algorithm 2 in RoMeo）：使用cp.async异步GMEM→SMEM加载→pipeline fill (Nstage iterations)→steady state (wait+mma+issue)→drain
- 异步并发执行：四种kernel在不同CUDA stream上并发执行→掩盖单独kernel的launch overhead和tall-and-skinny矩阵的SM underutilization
- 结果合并：低精度W4A4结果为基础，高精度outlier计算结果通过post-mul overwrite覆盖对应位置

涉及论文标题：
- RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

## Non-reduction Dimension Mixed Precision Computation（非归约维度混合精度计算）

术语是什么？通过联网搜索让回答具体和精准。

在矩阵乘法C[M,N] = A[M,K] × B[K,N]中，K为reduction dimension（归约维度），M和N为non-reduction dimension（非归约维度）。Mixed precision quantization可根据outlier所在的维度分为两类：(1) Reduction dimension（channel-wise）——outlier在K维度，可沿K分解GEMM为两个独立dense GEMM；(2) Non-reduction dimension（token-wise）——outlier在M维度，无法沿M简单分解，产生sparse computation pattern。RoMeo的token-wise mixed precision属于Non-reduction dimension类型，需要专门的permutation-free系统设计来高效实现。

从kernel调度角度拆解术语：

Non-reduction dimension mixed precision的本质挑战（以token-wise outlier为例）：

```
// Activation X [M, K], token-wise outliers在M维度
// M=4 tokens (t0-t3), K=4 channels, t0和t2是outlier tokens

// Channel-wise (reduction dim, 可分解):
//   X = [t0: INT8, t1: INT4, t2: INT8, t3: INT4] 每行精度不同
//   沿K分解: 将outlier columns抽出独立计算→dense GEMM
//   C = INT4_GEMM(X_[:, normal_cols], W_[normal_cols, :]) 
//     + INT8_GEMM(X_[:, outlier_cols], W_[outlier_cols, :])
//   ✓ 两个dense GEMM，Tensor Core兼容

// Token-wise (non-reduction dim, 不可简单分解):
//   沿M分解会导致sparse pattern:
//   C[outlier_rows, :] = INT8_dot(outlier_row, W_col)  // sparse access
//   C[normal_rows, :]  = INT4_dot(normal_row, W_col)   // sparse access
//   ✗ sparse computation, incompatible with Tensor Core dense tile requirement

// RoMeo的Permutation-free方案:
//   1. 复制outlier tokens到dedicated buffer（redundant computation代价）
//   2. 所有子矩阵uniform precision → dense GEMM
//   A_int4 = [M, K] (all INT4, 含outlier rows的INT4版本)
//   A_int8 = [k_a, K] (仅outlier rows, INT8)
//   W_int4 = [K, N_normal] (normal columns, INT4)
//   W_int8 = [K, N_outlier] (outlier columns, INT8)
//   → W4A4: A_int4[M, K] × W_int4[K, N_normal] = dense ✓
//   → W4A8: A_int4[M, K] × W_int8[K, N_outlier] = dense ✓
//   → W8A4: A_int8[k_a, K] × W_int4[K, N_normal] = dense ✓
//   → W8A8: A_int8[k_a, K] × W_int8[K, N_outlier] = dense ✓
```

Non-reduction dimension的sparse pattern可视化：
```
Thread Block workload partitioning (M=3 output tiles):
  Tile 0: 2 INT4 rows + 1 INT8 row → heterogeneous → conditional branches
  Tile 1: 1 INT4 row + 2 INT8 rows → heterogeneous → conditional branches  
  Tile 2: 3 INT4 rows             → homogeneous   → optimal Tensor Core use
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现策略对比：
1. **Permutation-based**：将同精度数据重排到相邻位置使tile内精度统一。但permutation引入非平凡index computation和in-place swap overhead（RoMeo实验表明常超过计算收益）。
2. **Permutation-free（RoMeo方案）**：tolerate redundant computation——outlier token同时参与INT4和INT8计算→所有矩阵为dense uniform-precision→无需permutation。代价为~5%额外计算（outlier比例），但换取Tensor Core高效dense执行。Post-mul overwrite将高精度结果覆盖到最终输出。
3. **Reduction-dimension方案（传统）**：沿K维度分解→天然的dense computation，无需处理non-reduction sparse pattern。这是channel-wise方法（MixQ/LLM.int8()）天然高效的原因。

涉及论文标题：
- RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

## Online Outlier Detection and Quantization Fused Kernel（在线异常值检测量化融合kernel）

术语是什么？通过联网搜索让回答具体和精准。

Online Outlier Detection and Quantization Fused Kernel是RoMeo中使用Triton实现的融合kernel，将per-token outlier detection（row-max reduction + top-k selection）、mixed precision quantization（INT4 + INT8）和INT4 data packing合并为单一GPU kernel，消除多次kernel launch和中间global memory round-trip的开销。由于token-wise outlier来源于input text的语言特征而非静态模型结构，必须在serving时在线检测。

从kernel调度角度拆解术语：

RoMeo的fused Triton kernel伪代码：

```
// Fused Triton Kernel: Outlier Detection + Quantization + Packing
// Grid: (M // BLOCK_M, 1), each block processes BLOCK_M token rows

Kernel FusedOutlierDetectQuantize:
  Input:  X_rot [M, K] (FP16, after Hadamard rotation)
  Output: X_Q_int4 [M, K//2] (packed INT4, 2 elements per byte)
          X_Q_int8 [k_a, K] (outlier buffer, INT8)
          scales_int4 [M] (FP16 per-token scaling factors)
          scales_int8 [k_a] (FP16 per-token scaling factors)
          outlier_mask [M] (bool)

  pid = program_id(0)
  row_start = pid * BLOCK_M
  row_end = min(row_start + BLOCK_M, M)

  // === Phase 1: Per-token row-max reduction ===
  // 每个program处理BLOCK_M个token rows
  shared_max [BLOCK_M]  // shared memory for row-wise max
  for i in row_start..row_end:
    local_max = 0.0
    for k_block in 0..K/BLOCK_K:
      tile = load(X_rot[i, k_block*BLOCK_K : (k_block+1)*BLOCK_K])
      local_max = max(local_max, max(|tile|))
    shared_max[i - row_start] = local_max

  // === Phase 2: Top-k outlier selection (within block) ===
  // 使用shared memory做block-local topk
  sorted_idx = argsort(shared_max, descending=True)
  for j in 0..min(k_a_per_block, BLOCK_M):
    outlier_mask[sorted_idx[j]] = True

  // === Phase 3: Mixed precision quantization ===
  for i in row_start..row_end:
    row = X_rot[i, :]
    if outlier_mask[i]:
      // INT8 quantization: range [-127, 127]
      scale = max(|row|) / 127.0
      X_Q_int8[outlier_idx] = round(row / scale)
      scales_int8[outlier_idx] = scale
      outlier_idx += 1
    // INT4 quantization: range [-7, 7] (always quantize for W4A4)
    scale = max(|row|) / 7.0
    X_Q_int4_packed[i] = pack_int4(round(row / scale))
    scales_int4[i] = scale

  // === Phase 4: INT4 data packing ===
  // 每2个INT4元素pack为1 byte: [elem0 | elem1<<4]
  // 压缩比 2:1, K个INT4元素→K/2 bytes
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 实现方式：Triton编写，融合了row-max reduction（parallel reduction over K dim）、topk selection、round-to-nearest quantization、scaling factor计算和INT4 packing。Triton的block-level programming model适合这种fused算子。
- 开销：RoMeo实测所有在线overhead（Hadamard + outlier detection + quantization + post-mul overwrite）共占layer latency约12%（batch=64时），其中outlier detection+quantization占约4%。
- 与offline outlier detection对比：channel-wise方法（MixQ）可用offline calibration static确定outlier→无在线detection开销。Token-wise方法（RoMeo）必须在线→fused kernel是minimize overhead的关键。
- INT4 packing格式：NVIDIA Tensor Core要求INT4以特定packed layout存储（每byte两个元素，even element在低4 bit、odd element在高4 bit），Triton的`tl.store`配合合适的block pointer可直接生成正确packing。
- 开源：https://github.com/thu-pacman/RoMeo

涉及论文标题：
- RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

## AU Arithmetic Intensity (ARI) for Operator-Level AU Selection

术语是什么？

ARI (Arithmetic Intensity for AU Selection) 是AUM论文提出的轻量级AU使用率判定指标，用于在operator级别选择最优AU (AMX vs AVX)。定义为：
- Prefill phase: ARI = 6(1/d + 3/BL)^(-1)，其中d=model hidden dimension, B=batch size, L=input sequence length
- Decode phase: ARI = 6(1/d + 3/B)^(-1)
ARI越高表示AU使用率U_AU越高——大d/大B/大L → higher ARI → 应使用AMX (TMUL compute-bound优势)；小d/小B/小L → lower ARI → 应使用AVX (避免AMX tile register配置overhead)。

从kernel调度角度拆解术语：

ARI-based AU selection的决策逻辑：
```
function select_au(phase, batch_size B, model_dim d, input_len L):
    if phase == "prefill":
        ARI = 6 * (1/d + 3/(B*L))^(-1)
        // d=4096, B=16, L=512 → ARI = 6/(0.000244+0.000366) = 9836
        // → High U_AU → AMX + High-AU region (2.1-2.5 GHz)
    elif phase == "decode":
        ARI = 6 * (1/d + 3/B)^(-1)
        // d=4096, B=16 → ARI = 6/(0.000244+0.1875) = 31.9
        // → Low U_AU → AVX + Low-AU region (2.8-3.1 GHz)
    
    return (U_AU, frequency_region)
    // U_AU determines:
    //   - AU choice: AMX vs AVX
    //   - Frequency region: C_H/C_L/C_N
    //   - Resource allocation: R_AU from AUV Model
```

术语一般如何实现？如何使用？

AUM Background Profiler offline: 对每个LLM operator计算ARI→判定U_AU→记录到AUV Model bucket。Runtime Controller online: 按bucket中U_AU分配频率region→查性能表P_a/P_t。ARI使AUM适应新模型仅需d, B, L参数（无需重profiling）。Paper基于先前研究推导的公式 [36][37]，与实测AMX cycle ratio (prefill 14.4%/decode 1.5%) 和 uop ratio (AMX FP ops: prefill 3.7%/decode 0.5%) 吻合。ARI阈值基于server-level AU usage distribution设定。

涉及论文标题：
- AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving

## Fully Integer Dequantization Datapath（全整数解量化数据路径）

术语是什么？

Fully Integer Dequantization是GyRot PE中使用的全整数解量化数据路径。传统group quantization的dequantization在浮点域执行（INT GEMM result → FP convert → FP scale × partial_sum → FP bias → FP accumulate），GyRot通过CoRFiG+HAP+重公式化非对称量化使scale factor (SX/SW)和zero-point (ZX)可用INT8精度表示，从而在PE内部以全整数完成dequantization——消除type conversion和FP arithmetic overhead。

从kernel调度角度拆解术语：

GyRot PE内fully integer dequantization的计算流程：

```
// 每个group (G=32) 的计算
Input:  X_Q[0:31]  // 32× INT4 activation
        W_Q[0:31]  // 32× INT4 weight
        SX         // INT8 activation scale factor
        ZX         // INT8 activation zero-point
        SW         // INT8 weight scale factor
        WSUM       // INT13 precomputed: Σ_{i=0}^{31} W_Q[i]

// Stage 1: INT4 dot product (1 cycle)
partial_sum = Σ_{i=0}^{31} X_Q[i] * W_Q[i]  // → 13-bit signed

// Stage 2: Integer dequantization (pipelined, 3 cycles)
// Step 2a: Multiply SX (INT8 × INT13 → INT21)
scaled = SX * partial_sum

// Step 2b: Multiply ZX × WSUM and subtract (INT8 × INT13 → INT21)
bias = ZX * WSUM
debiased = scaled - bias

// Step 2c: Multiply SW (INT8 × INT21 → INT29)
result = SW * debiased

// Stage 3: 32-bit integer accumulation (across groups)
accumulator += result  // 32-bit int, no FP conversion

// 最后: 32-bit int → FP16 (仅output writeback时转换一次)
output = fp16(accumulator)

// 对比传统FP dequantization:
// partial_sum → fp16_convert(partial_sum) → fp16_mul(SX_fp16) → 
// fp16_add(ZX_fp16 * WSUM_fp16) → fp16_mul(SW_fp16) → fp32_accumulate
// 每group需: 1× int→fp, 2× fp mul, 1× fp add, 1× fp accumulate
// GyRot: 0× type convert (all int), 2× int mul, 1× int sub, 1× int accumulate
```

术语一般如何实现？如何使用？

实现要素：(1) WSUM预计算：per-group weight sum Σŵ_i在weight加载时计算一次，存储在weight buffer的metadata bank，broadcast给整行PE共享——避免per-PE重复计算。(2) Multiplier设计：SX乘法器(INT8×INT13)、ZX×WSUM乘法器(INT8×INT13)、SW乘法器(INT8×INT21)均为定点整数乘法器，面积和功耗显著低于FP16乘法器。(3) 与重公式化非对称量化的配合：传统公式的dequantization顺序为SW·SX·(partial_sum − ZX·WSUM)，GyRot重公式化后变为SW·(SX·partial_sum − ZX·WSUM)——SX先乘partial_sum获得更大中间值范围，减少后续ZX减法引入的精度损失。(4) 硬件开销：GyRot-INT的dequantization+accumulation占PE area的4.2%、power的16.0%——远低于MANT（FP16 SF, G=64）和LightRot（FP16 SF+ZP, G=128）的FP dequantization unit。

涉及论文标题：
- GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

## GyRot PE (Integer Tensor Processing Element / 整数张量处理单元)

术语是什么？

GyRot PE是GyRot accelerator的基本计算单元，执行32-way INT4 dot product + fully integer dequantization。每个PE采用3D tensor组织（8×8×32），区别于传统2D systolic PE。PE支持INT4 weight × INT4 activation的32路并行点积，内嵌INT8 dequantization pipeline，输出32-bit整数累加器。

从kernel调度角度拆解术语：

```
// GyRot PE microarchitecture (Fig. 7b)
// 配置: 32-way INT4 dot product, minimum group size G=32

每个cycle的PE操作流程:
  ⃝1 32-way INT4 dot product:
    // 从input buffer读32× INT4 activation (X0~X31)
    // 从weight buffer读32× INT4 weight (W0~W31)
    // 32个4b×4b乘法 + adder tree → 13-bit partial sum

  ⃝2 Dequantization stage (pipelined):
    // SX (INT8) × partial_sum (INT13) → Multiply unit
    // ZX (INT8) × WSUM (INT13) → Multiply unit (parallel)
    // scaled - ZX×WSUM → Subtract unit
    // result × SW (INT8) → Multiply unit

  ⃝3 Integer accumulation:
    // 32-bit accumulator per PE
    // 跨group累加 (intra-group通过dot product, inter-group通过accumulator)
    // Output时转FP16写buffer

// PE array: 8×8×32 = 2048 parallel ops/cycle
// 8×8 systolic array, 每PE 32-way dot product
// output-stationary dataflow: partial sums stay in PE accumulator
```

GyRot PE与GPU上group quantization kernel的关键区别：
- GPU: Tensor Cores执行INT4 GEMM → CUDA cores做FP dequantization (INT→FP convert + FP scale/bias + FP accumulate) → mixed-precision path
- GyRot PE: 单一PE内integer domain完成全部计算 → fused INT datapath

术语一般如何实现？如何使用？

- RTL实现：SystemVerilog，Samsung 28nm工艺，Synopsys Design Compiler综合，1GHz目标频率
- PE Array: 8×8×32 tensor organization → output-stationary systolic dataflow
- 相比baseline PE：Tender (8-bit systolic, no group quant), MANT (G=64, FP16 SF), LightRot (G=128, FP16 SF+ZP)
- 面积/功耗：GyRot-INT PE相对Tender面积减65.2%、能耗减69.2%
- WSUM broadcast：WSUM unit用8×32-way adder-tree预计算per-group weight sum，整行8个PE共享，减少per-PE重复计算

涉及论文标题：
- GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

## Token-Centric PIM Partitioning (TCP)

术语是什么？通过联网搜索让回答具体和精准。

Token-Centric PIM Partitioning（TCP）是PIMphony提出的PIM workload映射策略，将长上下文LLM decoding中Attention（QK^T和SV）的并行维度从传统的head/batch维度转为token维度。在单个PIM module内，TCP沿token sequence方向切分：对于QK^T，每个channel处理一段token的Key cache，与同一query做部分dot-product，结果在module内经PIM HUB/EPU拼接后进入Softmax；对于SV，每个channel处理一段score/value的partial context，随后通过module内GPR-based inter-channel reduction得到完整context vector。TCP仅在单个PIM module内切token，不跨module，因此避免跨module同步开销（论文报告SV的module内reduction开销在LLM-7B 16K tokens下<0.2% attention latency）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

```
// TCP QK^T执行（per attention head, per PIM module with N channels）:
// Input: Query q[dh], Key cache K[T][dh], token length T
// Output: Score[T] per channel, concatenated in HUB

for each channel c in 0..N-1:
    // 每个channel处理 T/N 个token的Key segment
    t_start = c * (T / N)
    t_end = (c + 1) * (T / N) - 1

    // Step 1: WR-INP — broadcast query to GBuf
    WR_INP(GBuf[0], q[0:dh])

    // Step 2: MAC — per-token dot product with local Key segment
    for t in t_start..t_end:
        MAC(GBuf[0], K[t][0:dh], OBuf[t - t_start])
        // OBuf[t - t_start] += dot(q, K[t])

    // Step 3: RD-OUT — read partial scores
    for t in 0..(T/N - 1):
        RD_OUT(OBuf[t], Score_c[t])

// HUB: concatenate Score_0..Score_{N-1} → Score[0..T-1]
// EPU: Softmax(Score) → Score_norm[0..T-1]

// TCP SV执行:
for each channel c in 0..N-1:
    t_start = c * (T / N)
    t_end = (c + 1) * (T / N) - 1

    for t in t_start..t_end:
        // 用score scalar缩放V[t]并累加
        MAC(Score_norm[t], V[t][0:dh], OBuf_partial[c])
        // OBuf_partial[c] += Score_norm[t] * V[t]

// Inter-channel reduction in GPR:
// context = sum_{c=0}^{N-1} OBuf_partial[c]
```

TCP的关键特性：(1) 并行度来自token维度——长上下文token数量大（32K-1M），远多于head数（32-64），确保每个channel有充足工作；(2) 在16-channel/16-bank配置下，QK^T token length>256、SV token length>32即可full channel activation；(3) 不跨module同步——避免了分布式reduction的通信开销；(4) 与batch size解耦——即使batch=1也能利用所有channel。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

TCP由PIMphony的MLIR compiler在编译时自动生成partition scheme：compiler根据模型config（num_heads、head_dim、PIM module count、channel count）计算per-channel token segment range，在生成的PIM instruction sequences中嵌入token range metadata。Runtime IREE HAL根据当前请求的token length将对应segment的指令分发到各channel。TCP适合长上下文场景（token数大→并行度高），短上下文收益降低但论文仍报告256 tokens下有2.1× speedup。

涉及论文标题：
- PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System

## Dynamic PIM Command Scheduling (DCS)

术语是什么？通过联网搜索让回答具体和精准。

Dynamic PIM Command Scheduling（DCS）是PIMphony提出的PIM controller增强机制，将传统PIM的静态固定顺序命令发射替换为基于entry-level data dependency的乱序发射。传统PIM controller按固定时间间隔（tWR-INP、tMAC、tRD-OUT）串行发射WR-INP→MAC→RD-OUT命令序列，即使命令间无真实hazard也等待保守间隔，导致MAC pipeline大量idle。DCS在PIM HUB controller中增加Dependency Table（D-Table，记录每个GBuf/OBuf entry的最近访问命令ID）、Status Table（S-Table，记录每个命令的ID、完成时间和OBuf的is-MAC flag）和dependency-check unit。新命令到达时，controller查询D-Table/S-Table——仅当命令依赖的前序命令未完成时才等待，否则立即乱序发射，实现I/O数据搬运与MAC计算的overlap。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

```
// DCS Scheduling Logic (per command):
function dcs_schedule(cmd):
    // cmd has: type (WR_INP/MAC/RD_OUT),
    //           gbuf_entry (for WR_INP/MAC),
    //           obuf_entry (for MAC/RD_OUT)

    if cmd.type == WR_INP:
        // Check: any pending MAC reading same GBuf entry?
        pending = d_table[cmd.gbuf_entry].has_pending_read()
        if not pending:
            issue_now(cmd)
        else:
            wait_for(d_table[cmd.gbuf_entry].last_read_cmd)

    if cmd.type == MAC:
        // Check: WR_INP to same GBuf entry done?
        wr_done = s_table[d_table[cmd.gbuf_entry].last_write_cmd].completed
        // Check: previous MAC to same OBuf entry done?
        obuf_free = not s_table.has_pending_mac_to(cmd.obuf_entry)
        if wr_done and obuf_free:
            issue_now(cmd)
        else:
            wait_for(max(wr_done_cmd, obuf_mac_cmd))

    if cmd.type == RD_OUT:
        // Check: MAC writing to same OBuf entry done?
        mac_done = s_table[d_table[cmd.obuf_entry].last_mac_cmd].completed
        if mac_done:
            issue_now(cmd)
        else:
            wait_for(mac_done_cmd)

// 静态vs动态调度示例（FP16 GEMV）:
// 静态: W0→W1→W2→M3→M4→M5→R6（串行，34 cycles）
// 动态: W0→W1→M3(等W0done)→W2(与M3并行)→M4(等W1done)
//       →R6(等M3done)→M5(等W2done)→M7(与R6无冲突,提前)
//       （乱序重叠，22 cycles）
```

DCS的关键使能硬件：(1) dual-port OBuf——port A被MAC写入时port B可同时读出已完成结果（或反之），允许MAC和RD-OUT在不同OBuf entry上并行；(2) multi-entry GBuf——允许WR-INP预取下一批数据到其他GBuf entry，MAC消费当前entry时不受影响。在GQA row-reuse场景下，DCS利用dual-port在MAC消费当前GBuf query entry时预取下一批query/score，将row-reuse的KV复用转化为真实吞吐收益。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DCS在PIM HUB controller中以硬件逻辑实现：D-Table和S-Table为per-controller 576B metadata SRAM，dependency-check unit为组合逻辑（比较当前命令的operand entry与table中记录的最近命令ID/状态）。Compiler在生成PIM指令时嵌入dependency annotations（标识每条指令读/写哪个GBuf/OBuf entry），runtime DCS controller在issue前做轻量级查表和比较。Paper对比ping-pong buffering baseline：ping-pong因静态调度不知道entry级依赖，需等两个region均idle才能切换（hand-off pipeline stalls），DCS以同buffer size实现up to 1.4× higher compute-unit utilization。

涉及论文标题：
- PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System

## Head-First Partitioning (HFP) for PIM

术语是什么？通过联网搜索让回答具体和精准。

Head-First Partitioning（HFP）是现有PIM-based LLM加速器（CENT[16]、NeuPIMs[21]等）普遍采用的workload映射策略：将attention head与batch pair分配到PIM channel执行，每个channel负责特定(head, batch)组合的QK^T和SV计算。HFP隐含假设batch size或head数量足够多以填充所有channel。但在长上下文decoding中：(1) 单个request的KV cache足以占满一个channel的容量，压制batch size；(2) Tensor Parallelism下不同request token length差异导致channel执行时间不均衡；(3) Pipeline Parallelism下每stage只激活与当前layer相关的少数channel。PIMphony论文在32K context CENT分析中观察到HFP导致MAC utilization下降48%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

```
// HFP执行（PIM module with 4 channels, 2 heads, batch=2）:
// 分配: CH0→R(1,head=1), CH1→R(1,head=2), CH2→R(2,head=1), CH3→R(2,head=2)
//
// Decode step (batch=2, 每request需处理所有heads):
for each channel c:
    (r, h) = channel_assignment[c]  // 固定映射
    q = get_query(r, h)            // 从对应request和head取query
    K_cache = get_K_cache(r, h)    // 从该channel的KV cache读Key
    // QK^T: dot(q, K_cache[t]) for t in 0..T_r-1
    // SV: weighted sum of V_cache[t] by score[t]

// HFP问题示例 (TP=2, batch=2, 但request 1比request 2长得多):
// CH0: R(1,h1) — 处理128K tokens → 耗时很长
// CH1: R(2,h1) — 处理16K tokens  → 早早完成，之后idle
// CH2: R(1,h2) — 处理128K tokens → 耗时很长
// CH3: R(2,h2) — 处理16K tokens  → 早早完成，之后idle
// 总体MAC utilization = (128+16+128+16) / (128×4) ≈ 56%
// 极端情况batch=1时: 仅2/4 channel激活 → MAC util 50%
```

HFP的核心缺陷是并行维度（head/batch）在长上下文场景下稀缺且波动——head数固定（32-64），batch size被KV cache容量压缩，导致channel无法被充分填充。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

HFP在现有PIM系统中通过compiler或runtime在模型加载时静态分配实现——根据模型head数和PIM module/channel拓扑预先确定每个channel的(head, layer) assignment，推理过程中保持不变。其简单性和确定性使其易于实现，但在长上下文下的效率退化促使PIMphony提出TCP作为替代方案。

涉及论文标题：
- PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System

## PIM Primitive Pipeline (WR-INP, MAC, RD-OUT)

术语是什么？通过联网搜索让回答具体和精准。

PIM Primitive Pipeline（WR-INP / MAC / RD-OUT）是PIM系统中执行向量-矩阵运算的基本命令流水线。WR-INP（Write Input）：将32B input tile从host/PIM HUB写入PIM channel内的Global Buffer (GBuf)指定entry；MAC（Multiply-Accumulate）：从GBuf读取input tile和从DRAM bank读取weight/KV cache data，执行dot-product（GEMV）并将结果累加到Output Buffer (OBuf)对应entry；RD-OUT（Read Output）：从OBuf读出累加完成的结果返回PIM HUB/host。这一三阶段pipeline是PIM computation的基本构建块，一次完整的GEMV需要多轮WR-INP→MAC→RD-OUT序列（取决于维度大小和tile size）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

```
// PIM primitive pipeline执行（32B tile粒度，FP16 GEMV: q[1×4096] × K[4096×T]）:
// 每个MAC处理一个32B tile = 16个FP16元素

for tile_idx in 0..(4096/16 - 1):   // 256个tile
    // Phase 1: WR-INP
    //   将q[tile_idx*16 : (tile_idx+1)*16]写入GBuf指定entry
    //   耗时: tWR-INP (DRAM write-to-GBuf latency)
    WR_INP(GBuf[entry_a], q[tile_idx*16 : (tile_idx+1)*16])

    // Phase 2: MAC
    //   读取GBuf[entry_a]的input tile
    //   读取DRAM bank中K[tile_idx*16 : (tile_idx+1)*16, t]的对应16个weight
    //   执行16个乘法+adder tree → 累加到OBuf[result_entry]
    //   耗时: tMAC (bank read + multiply + accumulate latency)
    MAC(GBuf[entry_a], K_addr[tile_idx*16 : (tile_idx+1)*16], OBuf[result_entry])

    // Phase 3: RD-OUT
    //   当所有tile累加完成，从OBuf读出最终结果
    //   耗时: tRD-OUT (OBuf read-to-HUB latency)
RD_OUT(OBuf[result_entry], output)

// 静态调度问题:
// 传统controller按: WR-INP0→MAC0→RD-OUT→WR-INP1→MAC1→... 串行
// 即使WR-INP1与MAC0无依赖也不能提前发射
// DCS改进: WR-INP0→(WR-INP1可以并行)→MAC0(等GBuf0就绪)→(WR-INP2并行)→MAC1→...
```

PIM primitive的关键时序特性：(1) WR-INP和RD-OUT是I/O操作，占用PIM内部总线但不占用MAC unit；(2) MAC是计算操作，占用MAC unit但总线可用于其他channel的I/O；(3) Attention的GEMV因dh小（128）、tile数少（8个tile），I/O占比高，静态调度下MAC大量idle（论文中Attention MAC utilization低至14.7%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

PIM primitive以ISR（Instruction Set Register）指令编码，由compiler生成指令序列，PIM controller解码执行。在SK hynix AiM架构中，对应指令为MAC_ABK/MAC_SBK（All/Single-Bank MAC）、WR_ABK/WR_SBK和RD_ABK/RD_SBK。PIMphony通过MLIR compiler自动将attention subgraph映射为PIM primitive sequences，DCS controller在runtime做dependency-aware issue。

涉及论文标题：
- PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System

## Dynamic PIM Access (DPA)

术语是什么？通过联网搜索让回答具体和精准。

Dynamic PIM Access（DPA）是PIMphony提出的动态KV cache内存管理机制，通过两类动态PIM指令和on-module dispatcher实现运行时virtual-to-physical地址翻译，使KV cache可以按实际token length以1MB chunk lazy allocation而非按最大context length Tmax静态预留。Dyn-Loop指令的loop bound来自请求当前token index Tcur（而非编译期Tmax），Dyn-Modi指令在loop内按stride自动修改row/col等operand field形成逻辑virtual address。On-module dispatcher在PIM HUB内查询VA2PA table将virtual address翻译到已分配的物理KV cache chunk地址。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

```
// DPA执行流程:

// 1. Compiler生成DPA指令（编译时，不绑定物理地址）:
//    Dyn-Loop: for i in 0..Tcur:  // Tcur在runtime确定
//    Dyn-Modi: row = (Tcur / (nCH * nBank)) + i * stride_row
//              col = (Tcur % (nCH * nBank)) + i * stride_col
//    MAC(GBuf[0], K_virtual[row][col], OBuf[out])

// 2. Runtime VA→PA translation (on-module dispatcher):
function translate(va_row, va_col, req_id):
    chunk_id = va_row / CHUNK_SIZE_IN_ROWS  // 确定目标chunk
    pa_base = VA2PA[req_id][chunk_id]        // 查表得物理chunk基址
    if pa_base == NULL:
        // 需要新chunk → signal host
        signal_host_alloc(req_id, chunk_id)
        pa_base = wait_for_alloc()
    pa_row = (va_row % CHUNK_SIZE_IN_ROWS) + pa_base
    return pa_row, va_col  // col通常不变

// 3. Chunk lifecycle:
//    请求进入: host alloc chunk_0 → update VA2PA
//    请求增长: Tcur增加→需要新chunk → host alloc chunk_1 → update VA2PA
//    请求结束: host free all chunks of this request

// 对比静态分配:
// 静态: pre-alloc Tmax * dh * sizeof(FP16) → 大量浪费
// DPA:  alloc ceil(Tcur / CHUNK_CAPACITY) chunks → 仅最后chunk有碎片
```

DPA的capacity utilization提升：静态方案在不同workload上仅31.0%-40.5% utilization（因按Tmax预留），DPA达到平均75.6% utilization。关键trade-off：chunk粒度1MB——过大则碎片多，过小则VA2PA table entry多和host-PIM通信频繁。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DPA由compiler（生成Dyn-Loop/Dyn-Modi指令编码）、on-module dispatcher hardware（VA2PA lookup + address translation）和host runtime（chunk allocation/release + VA2PA table update）三层协同实现。Compiler不枚举物理地址，而是生成参数化指令；runtime在请求进入/增长/结束时更新mapping；dispatcher在每条指令decode时执行轻量级翻译。Paper在CENT和NeuPIMs simulator中建模dispatcher的VA2PA lookup延迟和chunk allocation通信开销。

涉及论文标题：
- PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System

## Arithmetic Intensity-Aware PIM-GPU Operator Scheduling（算术强度感知的PIM-GPU算子调度）

术语是什么？通过联网搜索让回答具体和精准。
Arithmetic Intensity-Aware Operator Scheduling是SADDLE提出的一种运行时operator-to-device动态映射机制，用于PIM+GPU异构系统上的speculative decoding。核心思想：speculative decoding中variable draft lengths和changing effective micro-batch sizes会动态改变operator的arithmetic intensity (FLOPs/Byte)，使离线静态mapping失效（如SpecPIM的offline genetic algorithm-based mapping）。SADDLE的Scheduler在运行时快速估算operator CI，与预标定的PIM compute-bound和GPU memory-bound ridge point比较，动态决定operator在PIM或GPU执行。初始固定映射：DLM attention → PIM（每iteration 1 token/request，算术强度极低）；TLM FC → GPU（Shared Pool聚合token后变为compute-intensive GEMM）。动态remap对象：DLM FC和TLM attention。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。
Scheduler的per-iteration决策伪代码：
```
// 预标定阶段（offline）
PIM_peak_compute = measure_pim_tflops()     // e.g. HBM-PIM PE array peak
PIM_peak_bandwidth = measure_pim_bw()       // internal bandwidth ~144 TB/s
GPU_peak_compute = measure_gpu_tflops()     // A100 Tensor Core peak
GPU_peak_bandwidth = measure_gpu_bw()       // HBM2e bandwidth ~1.5 TB/s

PIM_ridge = PIM_peak_compute / PIM_peak_bandwidth   // ~1 FLOP/Byte
GPU_ridge = GPU_peak_compute / GPU_peak_bandwidth    // ~100 FLOP/Byte

// === 每次prediction后: schedule DLM FC ===
function schedule_DLM_FC():
    active_reqs = count(H_t > τ for all requests)
    eff_bs = active_reqs
    
    // DLM FC: [eff_bs, d_model] × [d_model, d_model]
    FLOPs = 2 * eff_bs * d_model^2
    Bytes = (eff_bs * d_model + d_model^2) * 2  // FP16
    CI_dlm_fc = FLOPs / Bytes  // ≈ eff_bs (approx, when eff_bs << d_model)
    
    if CI_dlm_fc < PIM_ridge:   return "PIM"
    else:                        return "GPU"

// === 每次verification前: schedule TLM Attention ===
function schedule_TLM_attention():
    total_tokens = SharedPool.count()
    FLOPs_attn = 4 * total_tokens * d_head^2
    Bytes_attn = 2 * total_tokens * d_head * 2  // KV reads
    CI_attn = FLOPs_attn / Bytes_attn  // ≈ 2 * d_head (per-token)
    
    if CI_attn > GPU_ridge:  return "GPU"
    else:                     return "PIM"
```

关键动态：当micro-batch从12请求降至4请求（短draft请求先完成），DLM FC的CI从~12降到~4 → 从GPU bandwidth-bound转到PIM compute-bound → optimal target从GPU变为PIM。当draft length从1增至8，TLM attention CI提升 → GPU超越PIM（即使operator still memory-bound on GPU） → optimal target从PIM变为GPU。

术语一般如何实现？如何使用？
SADDLE Scheduler在Manager中实现：offline预标定每个device的peak compute和bandwidth（一次性），runtime用活跃请求数和Shared Pool token count快速估算CI→与预标定阈值比较→决定operator映射。与SpecPIM的offline genetic algorithm/MCTS（执行前一次性mapping，推理中不变）和PAPI的dynamic profiling形成对比：SADDLE用轻量CI估算替代完整profiling，仅需预标定阈值和简单代数运算。消融：动态scheduling使SADDLE吞吐再提升1.13×（over static mapping），PIM ops占比从9.51%升至14.89%、GPU从90.49%降至85.11%，整体吞吐提升1.21×。

涉及论文标题：
- Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems

## Hardware Instruction-Induced Low-Bit Layout for Tensor Cores (ldmatrix-based)

术语是什么？通过联网搜索让回答具体和精准。

Hardware Instruction-Induced Low-Bit Layout 是 BitDecoding 提出的方法，利用 ldmatrix PTX 指令的 thread-to-register 映射自动为低比特（INT4/INT2）量化数据生成 Tensor Cores 兼容的 packed layout。核心观察：ldmatrix 从 shared memory 加载数据到 register 时，按 Tensor Cores fragment 的 interleaved pattern 将数据分配到 warp 内各 thread。如果在 ldmatrix 加载 FP16 数据后立即由每个 thread 在 register 内做量化+pack，则写出到 global memory 的 packed low-bit data 天然保持 Tensor Cores 期望的 interleaved layout——后续用相同 ldmatrix 配置 unpack 时，值已处于正确位置可直接参与 mma。这消除了 Ladder/Marlin 等方法所需的重weight offline layout transformation kernel（prefill 58.02ms→0.06ms, decode 0.41ms→0.008ms overhead）。

从 kernel 调度角度拆解术语：

```
// === Hardware Instruction-Induced Layout (Offline zero-cost) ===
// Residual Kernel: 生成 layout-compatible packed KV cache
__global__ void residual_kernel(FP16* KV, INT16* KV_packed, ...) {
    // Step 1: ldmatrix加载FP16 KV到register（自动获得TC interleaved layout）
    ldmatrix.sync.aligned.m16n8k16.shared.b16 [...], [smem_addr];
    
    // Step 2: 执行mma计算（QK^T 或 PV）
    mma.sync.aligned.m16n8k16 [...];  // Tensor Cores
    
    // Step 3: 每thread在register内量化+pack
    // 关键：register中数据已按TC fragment layout排布
    FP16 val = reg_data[thread_local_idx];
    INT4  q   = quantize(val, scale, zero);  // FP16→INT4
    pack_bits(local_packed, q, bit_offset);   // 打包到INT16
    
    // Step 4: 写出packed data到global memory
    // layout已隐式正确——因步骤1的ldmatrix决定了pack后的排列
    KV_packed[global_idx] = local_packed;
}

// Packing Kernel: 消费时layout自动正确
__global__ void packing_kernel(INT16* KV_packed, FP16* Q, ...) {
    // 用与Residual Kernel相同的ldmatrix变体加载
    // 自动恢复正确的TC fragment layout
    ldmatrix.sync.aligned.m16n8k16.shared.b16 [...], [KV_packed_smem];
    
    // dequant后值已在正确register位置 → 可直接mma
    mma.sync.aligned.m16n8k16 [...];  // Tensor Cores，无layout mismatch
}
```

关键条件：(i) Packing Kernel 必须 mirror Residual Kernel 的 ldmatrix variant 和 mma variant；(ii) warp-tiling 配置必须一致；(iii) residual block size Nr=Pn×Wn×R 确保每个 TC fragment 被完整填充。

术语一般如何实现？如何使用？

实现通过统一 instruction configuration 协调 Residual 和 Packing kernel：根据 GPU 架构确定 ldmatrix 和 mma variant → 根据 bit-width（β=4或2）计算 packing ratio R=ω/β → 根据 Wn 和 Pn 计算 Nr。对于不同 GPU 世代（Ampere/Hopper/Blackwell），ldmatrix/mma variant 不同但原理通用。开源于 https://github.com/OpenBitSys/BitDecoding。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

## Residual KV Cache with Block Size Alignment (Nr = Pn × Wn × R)

术语是什么？通过联网搜索让回答具体和精准。

Residual KV Cache 是 BitDecoding 中为对齐 Tensor Cores warp-tiling 而引入的 KV cache 分区策略。将 total KV cache X ∈ R^{L×d} 分为两部分：X_pack = X[: L-Nr]（量化后 packed 存储）和 X_res = X[L-Nr:]（保留 FP16 精度的 residual cache）。Residual block size Nr = Pn × Wn × R，其中 Pn 是单个 warp tile 在 N 维度处理的元素数（如 mma.m16n8k16 下 Pn=8），Wn 是 N 维度 warp 数，R = ω/β 是 packing ratio（如 INT4→INT16, R=4）。Decoding 时新生成的 K/V tokens 先追加到 FP16 residual cache，当累计达 Nr 时触发 Residual Kernel 将其批量量化写入 packed cache。该设计保证每个 Tensor Cores fragment 被完整填充（无 underfill 导致的 compute waste），同时 residual cache overhead 极小（Nr 通常 <256，seq_len >> Nr 时仅占很小比例——paper Fig 14 显示 128K 下 overhead < 0.02ms）。

从 kernel 调度角度拆解术语：

```
// === Residual KV Cache 调度流程（Decode Step） ===
// 状态: residual_len 当前residual cache中FP16 token数

// Step 1: 新生成K/V追加到residual cache
FP16 K_new[d], V_new[d];  // 本轮decode新生成
residual_K[residual_len] = K_new;
residual_V[residual_len] = V_new;
residual_len++;

// Step 2: Packing Kernel执行attention
// 同时使用packed low-bit cache (L - residual_len tokens) 
// 和 residual FP16 cache (residual_len tokens)
packing_kernel_attention(Q, K_packed, V_packed, 
                         residual_K, residual_V, residual_len);

// Step 3: 若residual_len == Nr，触发Residual Kernel
if (residual_len == Nr):
    residual_kernel(residual_K, residual_V, Nr);  // 量化+pack → K_packed/V_packed
    residual_len = 0;  // 清空residual cache
```

Nr 计算示例（INT4, mma.m16n8k16, Wn=4）：Nr = 8 × 4 × 4 = 128。即每128个新token触发一次批量量化。

术语一般如何实现？如何使用？

Residual block size Nr 由 hardware instruction configuration 自动推导：根据 GPU 架构确定 mma variant（Ampere: m16n8k16, Hopper: wgmma.m64n64k16）→ 得到 Pn → 用户配置量化 bit-width β → 自动计算 R → 根据经验或 tuning 确定 Wn → 计算 Nr。Residual cache 存储为 pre-allocated FP16 buffer（size = Nr × d × 2 for K and V）。与 KIVI 的 per-token quantization（每 decode step 都量化，更高 overhead）和 continuous-packing baseline（每次 quantization 需 layout transform）相比，residual block 策略通过批量量化 amortize overhead 并用 ldmatrix 消除 layout transform。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

## lop3-based Layout Remapping for Low-Bit Dequantization (75316420 pattern)

术语是什么？通过联网搜索让回答具体和精准。

lop3-based Layout Remapping 是 BitDecoding 中用于高效 INT4/INT2→FP16 dequantization 的 register-level 位操作技术。虽然 ldmatrix-based layout induction 保证 packed data 与 Tensor Cores 兼容，但 naive INT→FP16 static_cast 在 GPU 上极慢（Kim et al., 2022 指出 low-bit cast 是已知的性能瓶颈）。BitDecoding 的方案：ldmatrix 加载 packed INT16 到 register → cast 为 INT32 → 用 NVIDIA PTX 的 lop3 指令（arbitrary 3-input lookup table / bitwise logic）按 75316420 pattern 重新排布 bits → 使得后续 FP16 转换高效。75316420 是经验确定的 bit permutation pattern，将 packed INT4 values 重新映射为符合 Tensor Cores interleaved register layout 且利于 FP16 exponent field manipulation 的顺序。相比 Marlin 的 layout transform kernel（独立 kernel, prefill 58ms overhead），lop3 remapping 完全在 register 内完成，zero extra memory traffic。

从 kernel 调度角度拆解术语：

```
// === lop3-based Layout Remapping（在Packing Kernel dequant阶段） ===
// 输入：INT16 packed data（8个INT4值，或16个INT2值）
//       已通过ldmatrix加载到register

// Step 1: Cast INT16 → INT32
uint32_t packed = (uint32_t)ldmatrix_result[thread_idx];

// Step 2: lop3 bit permutation (75316420 pattern)
// lop3: PTX bitwise logic instruction with arbitrary 3-input truth table
// 将bits按75316420模式重新排列
// 效果：
//   原始bits: b7 b6 b5 b4 b3 b2 b1 b0 b15 b14 b13 b12 b11 b10 b9 b8 ...
//   → 重排后更利于后续提取和FP16转换
uint32_t remapped;
asm volatile("lop3.b32 %0, %1, 0, 0, 0x88;"  // 具体imm根据pattern确定
             : "=r"(remapped) : "r"(packed));

// Step 3: Extract 4-bit values and convert to FP16
// pattern保证extract的4-bit value在FP16 mantissa field位置
uint32_t fp16_bits = (remapped_val << 10) | 0x6400;  // exponent=1024
FP16 deq = __int2float_rn(fp16_bits) - 1024.0f;       // 高效INT→FP16

// Step 4: Apply scale and zero-point
FP16 result = deq * scale + zero_point;
```

lop3 指令是 SM70+(Volta) 开始支持的通用 PTX 位操作指令，接收 3 个 32-bit 输入和一个 8-bit 真值表（lookup table, 256 entries），按真值表对输入逐 bit 输出。BitDecoding 利用其表达能力在一两条指令内完成复杂的 bit permutation，避免多条 shift/mask/or 指令。

术语一般如何实现？如何使用？

lop3 remapping 作为 CUDA inline PTX assembly 嵌入 Packing Kernel。75316420 pattern 根据 mma variant 和 bit-width 在 compile time 确定（不是 runtime 计算）。对于不同 GPU 世代和不同 bit-width (4-bit/2-bit)，pattern 可能不同，由 BitDecoding 的 unified instruction configuration 自动选择。Blackwell 架构上 lop3 remapping 被绕过（因原生 mxfp4 mma 不需要 software dequantization）。参考实现：https://github.com/OpenBitSys/BitDecoding。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

## Warp Parallelism Strategy for Low-Precision Decoding (Wm=1, Wn↑)

术语是什么？通过联网搜索让回答具体和精准。

Warp Parallelism Strategy for Low-Precision Decoding 是 BitDecoding 为解决低比特 dequantization 导致 warp stall 而提出的 warp 分配策略。核心思想：在 decode 阶段 Q length=1（极小的 M 维度），将 M 维度的 warp 数压缩到最小(Wm=1)，将释放的 warp 资源重新分配到 N 维度(Wn↑)。这样做的好处——多 warp 在 N 维度上并行处理 K/V 的不同 segment，每个 warp 的 dequantization（CUDA Cores）与 Tensor Cores mma 可以被 SM warp scheduler 自然地 overlap：当 warp_i 在 Tensor Cores 上执行 mma 时，warp_{i+1} 同时在 CUDA Cores 上做 dequantization。在 FlashAttention 原始 warp layout 下（Wn=1, Wm 较大），单个 warp 沿 N 维串行处理所有 tile，每次 dequant 都 stall 该 warp → Tensor Cores utilization 仅 10.91%（Table III）。BitDecoding 将 Wn 增至 4 后 TC utilization 提升至 19.66%。

从 kernel 调度角度拆解术语：

```
// === Warp Layout对比 ===
// 
// FlashAttention原始layout (register-level softmax):
//   Grid: Wm × Wn, 其中Wm>1 (沿M分割Q rows), Wn=1
//   Decode时Q_len=1 → M维度极小 → Wm个warp均严重underutilized
//   单个warp沿N串行: for each K_tile → dequant → mma → dequant stall
//
// BitDecoding layout:
//   Grid: Wm=1 × Wn (Wn≥4)
//   每个warp负责一段K/V: warp_0→K[0:Tn], warp_1→K[Tn:2Tn], ...
//   SM warp scheduler自动overlap:
//     cycle 0: warp_0发射mma, warp_1开始dequant
//     cycle 1: warp_0 still in mma, warp_1完成dequant→发射mma, warp_2开始dequant
//     ...
//   No warp idling — dequant latency hidden by parallel warp execution

// Dequantization stall消除原理:
// Original (Wn=1): Time = T_dequant + T_mma (串行)
// BitDecoding (Wn=4): Time ≈ max(T_dequant, T_mma) + T_dequant/Wn (parallel)
// 当T_mma >> T_dequant/Wn时dequant几乎零overhead
```

术语一般如何实现？如何使用？

在 CUDA kernel 中通过调整 grid/block 维度和 warp 分配实现：`blockDim = (32, Wn)` 或等效逻辑，query 沿 M 维度不拆分（Wm=1）。需要配套的 cooperative softmax（以 shared memory 替代 register-level softmax）处理 P 矩阵的跨 warp 聚合。此策略尤其适合 decode 场景（Q_len 极小，M 维度 compute 压力低），prefill 阶段（Q_len 大）不适用——prefill 下 M 维度本身可填满多个 warp。Paper Table III 验证：Wn=1→4 时 latency 从 3.746ms 降至 0.613ms（6.1× 改善），TCs utilization 从 10.91%升至 19.66%。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

## Cooperative Softmax with Cross-Warp Shared Memory Reduction

术语是什么？通过联网搜索让回答具体和精准。

Cooperative Softmax 是 BitDecoding 提出的跨 warp softmax 实现，用于配合 Wn>1 的 warp layout。FlashAttention 原始实现中 softmax 在两轮迭代间完全在 register 内完成（online softmax with running max/sum），依赖所有 KV tile 被同一 warp 串行处理。当 Wn>1（多 warp 并行处理不同 KV segments），每个 warp 仅持有部分 P 矩阵，register-level softmax 不再可行。BitDecoding 引入两个 shared memory buffer：sTMP ∈ R^{Wn}（跨 warp reduction 计算 row-wise max）和 sAcc ∈ R^{Tm×Tn}（暂存 P 矩阵用于后续 reload）。流程：各 warp 独立完成 QK^T→intra-warp register reduction for rowmax→store local max to sTMP→cross-warp shared memory reduction (parallel reduction tree)→broadcast global max→各 warp 计算 exp(P - global_max)→store P to sAcc (r2s)→reload P via ldmatrix from sAcc (s2r)→PV mma。Shared memory overhead 极小（Wn 通常 ≤8，sAcc 复用 sTMP pointer），仅引入 0.5% latency overhead（Table III）。

从 kernel 调度角度拆解术语：

```
// === Cooperative Softmax Algorithm (Algorithm 1 in paper) ===
// 输入: Qi ∈ RTm×d, Ki/Vi ∈ RTn×d (in REG)
// Shared Memory: sTMP ∈ RWn, sAcc ∈ RTm×Tn

// Step 1: QK^T mma (Tensor Cores)
Si = Qi × Kj^T;  // Si ∈ RTm×Tn, in TC registers

// Step 2: Cross-warp max reduction
// 2a: Intra-warp max (register-level shuffle reduction)
local_max = warp_reduce_max(Si);  // __shfl_xor_sync reduction

// 2b: Inter-warp max (shared memory)
if (lane_id == 0):
    sTMP[warp_id] = local_max;    // 每个warp写入shared mem
__syncwarp();
global_max = shared_mem_parallel_reduce(sTMP, Wn);  // log2(Wn)步

// Step 3: Online softmax update
mnew = max(m_old, global_max);
Pi = exp(Si - mnew);  // Pi ∈ RTm×Tn

// Step 4: Store P to shared memory (r2s)
sAcc[tile_row][tile_col] = Pi;  // tiled copy, register→shared

// Step 5: Reload P via ldmatrix (s2r) for proper TC alignment
ldmatrix.sync.aligned.m16n8k16.shared.b16 [...], [sAcc];

// Step 6: PV mma (Tensor Cores), using reloaded P
Onew = Pi_reloaded × Vj + diag(exp(m_old - mnew)) × O_old;
```

关键设计：(i) sAcc 复用 sTMP 的 shared memory 指针以最小化 memory overhead；(ii) s2r 通过 ldmatrix 重载 P 确保后续 mma 需要的 interleaved layout；(iii) Hopper 上 sAcc 可直接被 wgmma_SS 访问，省去 s2r step。

术语一般如何实现？如何使用？

Cooperative softmax 实现在 BitDecoding 的 Packing Kernel 中，约 200 行 CUDA PTX。Wn 的典型取值为 4 或 8（受限于 shared memory size 和 SM warp 数）。性能 trade-off：增加 Wn 提升 parallelism 但增加 cross-warp reduction overhead（O(log Wn) shared memory accesses）；paper 表明 Wn=4 在 A100 上接近最优。对于 prefill（Q_len 大），可回退到 register-level softmax（Wn=1 的 FlashAttention 模式）。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

## STSM-based Dequantized Data Pipeline for Hopper WGMMA

术语是什么？通过联网搜索让回答具体和精准。

STSM (Store to Shared Memory) 是 NVIDIA Hopper 架构新增的 PTX 指令，允许将 register 中的数据直接写入 shared memory 而不经过 global memory。BitDecoding 利用 STSM 为 Hopper 的 WGMMA 建立 dequantized data pipeline：WGMMA 的 wgmma_RS 模式要求 B 矩阵在 shared memory 中，而 dequantized K/V 通常在 register 中。BitDecoding 用 STSM 将 dequantized FP16 K/V 从 register store 到 shared memory → wgmma_RS 直接从 shared memory 读取 B 矩阵。由于 WGMMA 是异步执行的（non-blocking），STSM 的 shared memory 写入与 Tensor Cores 的 wgmma 计算可以自然 overlap——这一特性被 BitDecoding 用于实现 register→shared→wgmma 的无缝 data flow。

从 kernel 调度角度拆解术语：

```
// === STSM + WGMMA Pipeline (Hopper H100 decode step) ===
// 目标: 在Hopper上高效执行dequantized K × Q^T

// Step 1: ldmatrix加载packed low-bit K到register
ldmatrix.sync.aligned.m16n8k16.shared.b16 reg_K_packed, [smem_K_packed];

// Step 2: Dequantization in CUDA Cores (register)
for each thread:
    reg_K_fp16[thread_idx] = lop3_remap_dequant(reg_K_packed[thread_idx], scale, zero);

// Step 3: STSM将dequantized FP16 K写入shared memory
// (替代Ampere上需要的shared memory barrier + ldmatrix reload)
stsm [smem_K_fp16], reg_K_fp16;  // register → shared memory
__syncwarp();  // 保证STS完成

// Step 4: WGMMA读取shared memory作为B矩阵
// wgmma_RS: A=Q(register), B=K_fp16(shared memory), C=accumulator(register)
wgmma.fence;
wgmma.m64n64k16 C_acc, Q_reg, smem_K_fp16;
wgmma.commit_group;

// Step 5: 异步等待WGMMA完成
wgmma.wait_group 0;

// 关键：STSM与上一轮WGMMA可异步重叠
//  producer: STSM写下一块K_deq到shared memory
//  consumer: WGMMA消费当前shared memory中的K_deq
```

术语一般如何实现？如何使用？

STSM 通过 CUDA inline PTX 实现（`asm volatile("stsm ...")`）。在 BitDecoding 的 Hopper 版本中，Packing Kernel 内部通过 warp-specialized pipeline：部分 warps 负责 ldmatrix + dequantization + STSM，部分 warps 负责 wgmma computation。两者通过 shared memory 交替 ping-pong buffer 通信，无需 barrier（得益于 wgmma 的异步特性）。对比 Ampere 架构（无 WGMMA, 无 STSM）：dequantized data 保留在 register 直接参与 mma，无 STSM step——但 mma 是同步阻塞的，无异步 overlap 机会。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

## Query Transformation for Tensor Cores (MHA/MQA/GQA Unification)

术语是什么？通过联网搜索让回答具体和精准。

Query Transformation 是 BitDecoding 中使 MHA/MQA/GQA 在 Tensor Cores 上高效执行的统一方法。Decode 阶段 Q length 仅为 1 token，M 维度极小（低 arithmetic intensity），直接 QK^T 会严重 underfill Tensor Cores。BitDecoding 利用 GQA/MQA 的 KV head sharing 特性：GQA 下 gq = hq/hkv 个 query heads 共享同一组 K/V。Query Transformation 将 Q tensor 从 [1, (gq, hkv)] reshape 为 [gq, hkv]——在 batch 维度上将多个共享相同 KV head 的 query 合并为更大 GEMM block→TC fragment 被完整填充→warp occupancy 和 throughput 显著提升。MHA (gq=1) 下无 sharing 可合并，但 decode Q_len=1 时 M 维度仍小→BitDecoding 仍受益于 Wn↑ warp layout。MQA (hkv=1, gq=hq) 下 sharing 最大→收益最大。

从 kernel 调度角度拆解术语：

```
// === Query Transformation Example (GQA, gq=4, hkv=8) ===
// 原始: Q.shape = [1, 32]   (1 token, hq=32)
//       K.shape = [L, 8, d] (L tokens, hkv=8 heads)
// GQA: 每4个Q head共享1个KV head

// 原始decode实现: 
//   for each query head hq_i (0..31):
//       kv_head = hq_i / 4;  // 每4个Q head映射到同一KV head
//       score[hq_i] = Q[0, hq_i] × K[:, kv_head]^T  // 1×L GEMV
//   → 32个独立GEMV → TC严重underutilized (M=1)

// BitDecoding Query Transformation:
// 1. Reshape: Q: [1, 32] → [1, (4, 8)] → [4, 8]
//    - 4 = gq (grouped queries per KV head)
//    - 8 = hkv (number of KV heads)
// 2. 对每个KV head k (0..7):
//    Q_group = Q[:, k]  // shape [4, d_head] — 4个共享相同K的Q
//    K_kv   = K[:, k, :] // shape [L, d_head]
//    scores_k = Q_group × K_kv^T  // shape [4, L] — 4×L GEMM on TC
// 3. 结果: 8个[4, L] score矩阵

// 效果: M维度从1→4 → TC mma tile M=4更接近完整的M=16→更高occupancy
// MQA (hkv=1): Q reshape [1, hq] → [hq, 1] → M=hq完全利用TC
```

术语一般如何实现？如何使用？

Query Transformation 在 kernel launch 前通过 memory layout reshape 实现（zero data copy）。对于 MHA (gq=1)，no reshape needed——warp parallelism (Wn↑) 补偿。在 BitDecoding 的 query transformation module 中统一处理三种 attention 变体：通过 gq = hq/hkv 自动确定 reshape 逻辑。开源于 https://github.com/OpenBitSys/BitDecoding。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

## Page-Aware Windowed Clustering（页感知窗口化聚类）

术语是什么？通过联网搜索让回答具体和精准。
Page-Aware Windowed Clustering是AQPIM提出的算法-硬件协同设计技术，通过将PQ clustering的centroid分配限制在DRAM page/row大小内来实现高效的indirect lookup。核心原理：HBM-PIM每个bank有1KB row buffer（可存512个FP16 inner product values）。AQPIM将input sequence分割为多个window，每个window内的所有tokens被映射到不超过K=512个centroids——这意味着这些centroids对应的512个inner product values完全fit在单个DRAM row中。当attention lookup通过indices访问inner product values时，因为所有值已在同一row buffer中，只需1次row activation即可完成所有512次lookup。Window advance时，前window的centroids被复制到新page并更新以适应新tokens。

从kernel调度角度拆解术语：
```
// Page-Aware Windowed Clustering for Decode Attention Lookup
// 硬件约束: 1KB row buffer per bank = 512 FP16 values

window_size = L  // 初始: single window covers full sequence
if N_tokens generates > 512 unique centroid assignments:
    num_windows = ceil(N_tokens / W)
    
for each window w:
    // Step 1: Cluster within window
    centroids_w = weighted_kmeans(KV[w_start:w_end], K=512)
    
    // Step 2: Attention Inner Product computation
    IP_values[0:511] = query_sub × centroids_w^T  // stored in 1 row buffer
    
    // Step 3: Intra-row lookups (all within row buffer)
    ACTIVATE row_buffer  // single activation
    for idx in indices_w:
        partial_sum += COLUMN_SELECT(row_buffer, idx)
    
    // Step 4: Sum across subvectors and windows
    qKT[w_start:w_end] = partial_sum

// 优势: Row activations = num_windows (而非 N_indices)
```

术语一般如何实现？如何使用？
Implementation依赖于tight co-design：(a) Algorithm side: clustering restricted to fit each window's centroid count ≤512 (FP16) = row buffer size；(b) Hardware side: intra-row indirection (GRF→MUX→column decoder) executes lookup without additional row activations；(c) Window management: centroids copied from previous window as the window slides forward, updated incrementally for new tokens (not full reclustering). 对于大多数long-context scenarios，1个window映射整个sequence到512个centroids即足够（accuracy saturates at K=512, Table III）。当需要更多centroids时，多window方案扩展：每window最多512 centroids，window间independence使得lookup parallelism不受影响。复杂度和overhead: 每window 1 row activation → total activations = num_subvectors × num_windows (远小于 naive O(N) per lookup)。

涉及论文标题：
- AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization

## PQ-Based Attention Computation on PIM（PIM上的PQ注意力计算——查表+求和替代GEMV）

术语是什么？通过联网搜索让回答具体和精准。
PQ-Based Attention Computation是AQPIM提出的将传统attention GEMV (qK^T) 转换为codebook lookup + partial summation的kernel设计。核心transform：不存储和计算full key matrix K (N×d)，而是存储key codebook (d×K, K个FP16 centroid per subvector) + key indices (m×N, m个subvector各1个INT index per token)。Attention计算流程：query split成m subvectors → 各subvector与对应codebook做小规模matmul得inner product matrix (1×K) → 用key indices从IPM中lookup对应值 → m个值求和得qK^T approximation。该设计使attention compute complexity从O(N×d)降至O(K×d + m×N)，当K≪N时（512 vs 32K）大幅减少计算量。关键优势：(1) 无需explicit dequantization——直接操作compressed representation；(2) 复用现有FP16 MAC units（no INT ALU needed）；(3) 与GQA/MQA正交兼容。

从kernel调度角度拆解术语：
```
// PQ-Based Attention Kernel (Decode step, 1 new query token)
// Input: q ∈ R^{1×d}, key_codebook ∈ R^{m×d/m×K}, key_indices ∈ Z^{m×N}
//        value_codebook ∈ R^{m×K×d/m}, value_indices ∈ Z^{m×N}

// === BankPE: ATNK (Attention-Kernel Key) ===
q_sub[1..m] = split_into_m_subvectors(q)  // m=32, each [1, d/m]
for sub in 1..m:  // Parallel across banks
    // Inner Product Matrix: query_sub × key_codebook
    IPM[sub][0:K-1] = q_sub[sub] @ key_codebook[sub]^T  // [1,d/m] × [d/m,K] → [1,K]
    // IPM stored in row buffer (512 FP16 values = 1KB)

// === BufferPE: SFM (Softmax with lookup) ===
for sub in 1..m:  // Sequential or parallel
    IPM[sub] received from BankPE via PIM_MV_BA
// Lookup + Sum + Softmax:
for n in 1..N:  // For each token in sequence
    qKT[n] = 0
    for sub in 1..m:
        idx = key_indices[sub][n]  // ∈ [0, K-1]
        qKT[n] += IPM[sub][idx]    // intra-row indirection lookup

attn[0:N-1] = softmax(qKT / sqrt(d_head))  // BufferPE SFM unit

// === BankPE: ATNV (Attention-Kernel Value) ===
// attn weights distributed back to BankPE via PIM_MV_BF
for sub in 1..m:  // Parallel across banks
    // Reconstruct value vectors: Σ attn[n] × value_codebook[sub][index]
    output[sub][0:d/m-1] = 0
    for n in 1..N:
        idx = value_indices[sub][n]
        output[sub] += attn[n] × value_codebook[sub][idx][0:d/m-1]

output = concat(output[1..m])  // → GPU via PIM_RD
```

术语一般如何实现？如何使用？
PIM Command Sequence for one PQ-attention decode step:
1. PIM_SET_CONFIG: configure m=32, K=512, d_head dimensions
2. PIM_WR: write query vector to BankPE
3. PIM_MAC_AB: BankPE computes query×codebook → IPM in row buffer
4. PIM_MV_BA: transfer IPM to BufferPE
5. PIM_RET: intra-row indirection lookup (PIM_RET command) → read IPM values by indices
6. PIM_SFM: softmax on looked-up values
7. PIM_MV_BF: transfer attention weights back to BankPE
8. PIM_MAC_AB: value reconstruction (attn × value codebook)
9. PIM_RD: read final attention output → GPU

Performance: GEMV complexity O(N×d) → PQ attention O(K×d + m×N), K=512 ≪ N=32768 → matmul cost constant w.r.t. sequence length (maintained as K grows negligibly). 实测: decoding per-step latency AQPIM=0.12× vs GPU baseline at S_len=32768 (8.33× speedup, Fig.12).

涉及论文标题：
- AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization

## Similarity Gather

术语是什么？通过联网搜索让回答具体和精准。
Similarity Gather是Focus SIC中的streaming压缩操作，在systolic array的每个GEMM tile输出后立即执行，将m×n tile中的vectors做vector-level similarity检测和deduplication，最终仅将deduplicated vectors和similarity map写回DRAM。它是SIC的"压缩"阶段（Scatter是"恢复"阶段），两者成对构成gather-scatter循环贯穿所有FC层。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Similarity Gather的per-tile执行伪代码：
```
# Input: GEMM output tile O[m][n], m=1024, n=32 (a=n=32)
#        SEC offset encoding for position recovery
# Output: compact_vectors[p][n] (p <= m), similarity_map[m]

def similarity_gather(tile_O[m][n], offsets[m]):
    # Step 1: Position recovery + FHW layout
    positions = restore_fhw(offsets)  # (f, r, c) per vector
    vectors = reorder_to_fhw(tile_O, positions)

    # Step 2: L2-norm precompute (each vector = 32-dim)
    l2_norms[0..m-1] = [sqrt(sum(v[d]^2 for d in 0..31)) for v in vectors]

    # Step 3: Block-wise similarity matching
    similarity_map = [-1] * m  # -1 = unique, else points to representative
    compact_idx = 0
    compact_vectors = []

    for each 2×2×2 block (stride 1):
        # 8 vectors per block: 4 from frame A, 4 from frame B
        key_idx = max(block)  # highest-index vector as key
        key_vec = vectors[key_idx]
        key_norm = l2_norms[key_idx]

        if similarity_map[key_idx] == -1:  # key not yet matched
            similarity_map[key_idx] = compact_idx
            compact_vectors.append(key_vec)
            compact_idx += 1

        for other_idx in block \ {key_idx}:
            if similarity_map[other_idx] != -1: continue  # already processed
            dot_prod = sum(key_vec[d] * vectors[other_idx][d] for d in 0..31)
            cos_sim = dot_prod / (key_norm * l2_norms[other_idx])
            if cos_sim > 0.9:  # similarity threshold
                similarity_map[other_idx] = similarity_map[key_idx]
                # other_idx reuses key's compact index
            else:
                similarity_map[other_idx] = compact_idx
                compact_vectors.append(vectors[other_idx])
                compact_idx += 1

    # Step 4: Writeback
    write_to_dram(compact_vectors)  # p vectors, p <= 1024
    write_to_dram(similarity_map)   # 1 × m
```

Timing分析：matcher最多需要 `8 × m = 8 × 1024 = 8192 cycles` per tile，而GEMM需要 `K/b × m = 3584/32 × 1024 = 114,688 cycles`。Matcher overhead < 7% of GEMM time，不在critical path。仅当K < 256时matcher接近critical path，此时可部署多matcher并行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Similarity Gather在Focus中实现为hardware module：convolution-style layouter (地址生成+bank mapping) → L2-norm buffer (1×m FP32 values) → dot-product unit (32-cycle for 32-dim) → comparator (threshold 0.9) → compact output buffer (stores deduplicated vectors) + similarity map buffer (1×m int indices)。Gather operation在GEMM tile output streaming完成时触发，完成后触发DRAM write。下一层GEMM读取compact vectors + similarity map，执行Similarity Scatter恢复full output。开源实现含algorithm/simulator/rtl：https://github.com/dubcyfor3/Focus。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

## Similarity Scatter

术语是什么？通过联网搜索让回答具体和精准。
Similarity Scatter是Focus SIC中的streaming重建操作，与Similarity Gather配对构成gather-scatter循环。Scatter在GEMM对concentrated (deduplicated) vectors执行计算后，根据上一层Similarity Gather产生的similarity map将compact vectors的partial sums复制/分发回所有原始token位置，在output-stationary buffer中累加得到正确的full-size output tile。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Similarity Scatter的per-GEMM-iteration执行伪代码：
```
# Input: concentrated input tile: p vectors × K (p < 1024)
#        weight tile: k × n (k=32, n=32)
#        similarity_map[m] from previous Similarity Gather

def similarity_scatter(compact_input[p][K], weight[k][n], sim_map[m]):
    # GEMM execution on compact vectors (output-stationary outer loop)
    output_tile = zeros(m, n)  # m=1024, n=32 — full size accumulator
    for k_start in 0..K-1 step k:  # outer loop: ⌈K/k⌉ iterations
        # Inner loop: weight stationary GEMM
        partial_sum = compact_input[:, k_start:k_start+k] @ weight[k_start:k_start+k, :]
        # partial_sum: p × n (p compact vectors, each n=32 dims)

        # Scatter: replicate compact vector results to original positions
        for orig_idx in 0..m-1:
            compact_idx = sim_map[orig_idx]
            # accumulate: each original position gets its representative's partial
            output_tile[orig_idx, :] += partial_sum[compact_idx, :]

    # After all K iterations: output_tile contains correct full results
    # Then invoke Similarity Gather for next-layer compression
    return output_tile
```
Key: Scatter使用2a-wide accumulator (64 when a=n=32)，支持concurrent accumulation of reconstructed + streaming outputs。因为不同sub-tile可能有不同的compact vector subsets（每个subset代表多个original tokens），直接accumulation会因semantic aliasing产生错误——Scatter通过similarity map的index-based replication解决此问题。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Similarity Scatter在Focus中实现为hardware accumulation logic：2a-wide (64-wide) accumulator array → index mapper (lookup sim_map per original position) → replication logic (broadcast compact partial sum to mapped positions) → output-stationary buffer (512KB for full m×n tile)。Scatter的reconstruction overhead negligible（index lookup + parallel accumulation），不require additional memory allocation。Scatter与GEMM inner loop pipeline重叠：每个inner loop iteration生成partial sums后立即scatter→accumulate，不等待所有iterations完成。开源RTL见https://github.com/dubcyfor3/Focus。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

## Streaming Bubble Sorter

术语是什么？通过联网搜索让回答具体和精准。
Streaming Bubble Sorter是Focus SEC中用于top-k selection的轻量级硬件排序器。它将importance analyzer中的a个并行max units级联为a-way bubble sorter，以streaming方式逐步refine top-a tokens，最终以O(M·a·k) cycles完成top-k selection（M为候选数，k为保留数，a为并行度）。与传统full sorting (O(M log M))或global sorting不同，streaming bubble sorter利用bubble sort的局部比较特性实现pipelined、low-area的streaming top-k。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Streaming Bubble Sorter的硬件执行流程：
```
# a-way streaming bubble sort for top-k from M candidates
# a: number of parallel max units (e.g., a=4)
# k: number of top elements to retain (k < M)

top_a = [0] * a  # maintain top-a elements in sorted order (descending)
result = []  # final top-k output
remaining = M

while remaining > 0:
    # Stream in a new elements per cycle
    batch = stream_next_a_elements()  # a elements from importance vector
    for new_elem in batch:
        # Bubble insert: compare and swap to maintain descending order
        pos = a - 1
        while pos > 0 and top_a[pos-1] < new_elem:
            top_a[pos] = top_a[pos-1]
            pos -= 1
        top_a[pos] = new_elem
        # top_a now contains the a largest elements seen so far
    remaining -= a

# After all M elements processed, top_a contains the a largest values
# To get top-k (k may be larger than a): iterate with reduced scope
# Actually: chain multiple a-way sorters for k > a
# Complexity: M * a * k / a = M * k cycles (a cancels out partially)
# More precisely: M/a * k cycles for k <= a
```
论文证明SEC sorting与image attention GEMM完全重叠：sorting需要 M·a·k cycles，attention GEMM需要 M·(M+T)·h·n/(a·b) cycles，ratio = (M+T)·h·n/(k·b)。当h·n=3584, b=32, k<M+T时ratio ≫ 1，sorting远在attention GEMM完成前结束。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Streaming Bubble Sorter在Focus SEC中用SystemVerilog实现。它复用importance analyzer的并行max units（a个），通过级联和feedback path构成bubble chain。每个cycle接受a个新scores，与当前top-a比较并更新。Area开销计入SEC的1.9%总面积（含analyzer + sorter + encoder）。此设计可参数化a和k以适应不同accuracy/complexity trade-off。相比full sorting logic，bubble sorter的area和latency均远小。开源RTL见https://github.com/dubcyfor3/Focus。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

## GEMM Tiling with Streaming Concentration

术语是什么？通过联网搜索让回答具体和精准。
GEMM Tiling with Streaming Concentration是Focus的核心运行时调度策略，将systolic array的标准GEMM tiling（m=1024, n=32, k=32）与on-chip streaming concentration紧密对齐。每个GEMM tile的输出不直接写回DRAM，而是立即stream到SIC做vector-level similarity detection和deduplication，仅deduplicated vectors + similarity map写回DRAM。后续GEMM对compact vectors执行计算，通过similarity map做scatter reconstruction。这种tile-local compression-while-computing模式使compression与GEMM pipeline完全融合，消除global token-wise方法的DRAM往返。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GEMM Tiling with Streaming Concentration的完整tile-level pipeline：
```
# Focus accelerator GEMM tiling + concentration pipeline

for each layer:
    # === Attention layers: SEC semantic pruning ===
    if is_attention_layer:
        # GEMM: QK^T tiling (standard)
        S_tile = systolic_array(Q_tile, K_tile^T)  # output: m × (M+T)
        S_softmax = SFU.softmax(S_tile)  # special function unit

        # SEC streaming: importance + top-k (overlapped with next GEMM)
        importance = SEC.importance_analyzer(S_softmax[T:, :M])  # T×M → 1×M
        top_k_indices = SEC.streaming_bubble_sort(importance, k)

        # Pruned P×V GEMM: only load retained tokens
        P_tile = systolic_array(Q_retained, K_retained^T)  # smaller M
        V_output = systolic_array(P_tile, V_tile)

    # === FC layers: SIC scatter-gather ===
    if is_fc_layer:
        # Previous layer wrote: compact_vectors[p][n] + similarity_map[m]
        # Current layer GEMM on compact vectors:
        for k_start in 0..K-1 step k:  # outer loop (output stationary)
            # inner loop (weight stationary):
            partial = systolic_array(compact_input[:, k_start:k_start+k],
                                     weight[k_start:k_start+k, :])
            # SIC Scatter (in-place, concurrent with accumulation):
            for orig_idx in 0..m-1:
                compact_idx = sim_map[orig_idx]
                output_tile[orig_idx] += partial[compact_idx]
                # 2a-wide accumulator for concurrent scatter

        # After all K iterations: output_tile is full and correct
        # SIC Gather: compress output tile before DRAM writeback
        compact_next, sim_map_next = SIC.similarity_gather(output_tile)
        write_to_dram(compact_next, sim_map_next)  # for next FC layer
```
关键优势：reduced input vectors (p < 1024) → lower GEMM workload → 每次FC GEMM的compute savings累计可达~5.0×。DRAM traffic reduction：仅compact vectors + similarity map (small int metadata) 写回，vs baseline全量tokens，达4.9× reduction。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
此tiling+concentration pipeline在Focus的SCALEsim-v2-based cycle-accurate simulator中建模，接收PyTorch算法实现生成的layer-wise sparse traces（每tile的active/inactive indices, similarity map, compact vector count）作为输入。硬件RTL实现中，GEMM controller与SIC的gather/scatter logic通过ready/valid handshake同步：GEMM tile ready → trigger SIC gather → DRAM write → next layer's GEMM reads compact data → scatter during accumulation → after tile complete → trigger gather again。开源实现含完整simulator和RTL：https://github.com/dubcyfor3/Focus。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

## Similarity Map

术语是什么？通过联网搜索让回答具体和精准。
Similarity Map是Focus SIC中的核心metadata结构，大小为1×m（m=1024 per tile），记录每个original vector在compact buffer中的代表vector index。例如，若token 32与token 31的cosine similarity > 0.9，则similarity_map[32] = similarity_map[31] = index_of_token_31。该map使下游GEMM的Similarity Scatter能正确地将compact vector的partial sums复制回所有原始token位置，保证concentration后的功能正确性。Similarity Map是Focus实现lossless vector-level compression的关键enabler。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Similarity Map的生成和使用：
```
# Generation (in Similarity Gather):
similarity_map = [-1] * 1024  # -1 = unassigned
compact_idx = 0
for each 2×2×2 block:
    key_idx = max(block)
    if similarity_map[key_idx] == -1:
        similarity_map[key_idx] = compact_idx
        compact_idx += 1
    for other_idx in block \ {key_idx}:
        if cos_sim(key, vectors[other_idx]) > 0.9:
            similarity_map[other_idx] = similarity_map[key_idx]  # reuse
        else:
            similarity_map[other_idx] = compact_idx
            compact_idx += 1

# Usage (in Similarity Scatter):
# partial_sum[p][n]: results for p compact vectors
for orig_idx in 0..1023:
    compact_idx = similarity_map[orig_idx]
    output[orig_idx] += partial_sum[compact_idx]
```
Similarity Map存储为1×m的int array，每entry为representative index（最多log2(m) bits）。Memory overhead极小：1024 × 10 bits ≈ 1.28KB per tile。写回DRAM时与deduplicated vectors一同写入，读回时一同读出。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Similarity Map在Focus硬件中存储为1×m on-chip buffer（~1.28KB per tile）。在Similarity Gather阶段由similarity collection logic写入，在Similarity Scatter阶段由index mapper读出。Map的index lookup为single-cycle random access（地址=original_idx→读出representative_idx）。Scatter阶段用此index从compact partial sums中选取对应结果并广播到original output position进行累加。开源实现见https://github.com/dubcyfor3/Focus。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

## Offset Encoding

术语是什么？通过联网搜索让回答具体和精准。
Offset Encoding是Focus SEC中的一个紧凑位置编码方案，用于在semantic pruning后记录保留token的相对空间位置。由于SEC pruning破坏了token的连续空间结构（移除不重要的tokens），下游SIC需要知道每个retained token的原始(Frame, Height, Width)坐标以构建2×2×2时空block。Offset Encoder用sliding window为每对连续保留token记录一个小整数offset（相对于前一保留token的position delta），而非存储绝对坐标。这种compact encoding仅需lightweight registers，无需global memory access。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Offset Encoding的生成和使用：
```
# Generation (in SEC, after top-k selection):
# retained_indices: sorted indices of top-k tokens in original sequence
# Original token positions are in FHW order: (f, r, c) → linear index

prev_pos = retained_indices[0]
offset_stream = [prev_pos]  # first entry: absolute position of first retained token
for idx in retained_indices[1:]:
    delta = idx - prev_pos  # offset to previous retained token
    offset_stream.append(delta)
    prev_pos = idx

# offset_stream stored as compact ints, streamed alongside GEMM output

# Usage (in SIC layouter):
# Reconstruct absolute positions from offset stream:
positions = [offset_stream[0]]
for delta in offset_stream[1:]:
    positions.append(positions[-1] + delta)

# Map linear position to FHW coordinates:
for pos in positions:
    f = pos // (H * W)      # frame index
    r = (pos % (H * W)) // W  # row index
    c = pos % W               # column index
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Focus SEC中用lightweight registers实现：sliding window over retained token indices → compute delta → output as compact int。完全local和streaming，与SEC sorter同步输出。开源实现见https://github.com/dubcyfor3/Focus。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

---

## Early-Exit Sorting（提前退出排序）

术语是什么？通过联网搜索让回答具体和精准。

Early-Exit Sorting是一种排序优化策略，在排序过程中一旦累积结果满足退出条件即终止排序，避免对全部元素完整排序。在V-Rex的WTU（WiCSum Threshold Unit）中使用：对ScoreCluster per row的score按降序排列，从高分开始累积weighted sum，当累积值超过阈值Th_wics时立即停止，剩余低分元素不再排序或处理。关键原理：少量大score通常占weighted sum的大多数（V-Rex paper报告平均仅需处理16% scores即可达到threshold），从高分bucket开始可快速触发early exit。与完整排序（O(n log n)）相比，early-exit sorting的实际复杂度接近O(k log n)（k为实际处理的元素数），在k << n时显著降低latency和energy。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

WTU中Early-Exit Sorting的硬件pipeline：

```
// 硬件数据流：Bucket-based early-exit pipeline
// Input: ScoreCluster[row] [num_clusters], token_count [num_clusters]
//        Th_wics (precomputed: weighted_sum * 0.3)

// == Preprocess Step (parallel across all WTU cores) ==
weighted_sum = 0; min_score = INF; max_score = -INF
for j in 0..num_clusters-1:
    weighted_sum += ScoreCluster[j] * token_count[j]
    min_score = min(min_score, ScoreCluster[j])
    max_score = max(max_score, ScoreCluster[j])
Th_wics = weighted_sum * 0.3
score_range = max_score - min_score

// == Token Selection Step (early-exit pipeline) ==
num_buckets = 16  // WTU bucket count
curr_range_start = max_score
selected = []
acc_sum = 0

while curr_range_start >= min_score:
    range_end = curr_range_start - score_range/num_buckets

    // Bucket Sort (upper/lower sorters, parallel)
    in_range_bitmask = (ScoreCluster >= range_end) &
                       (ScoreCluster <= curr_range_start)
    // Sorters generate bitmask of scores in current range
    in_range_scores = ScoreCluster[in_range_bitmask]
    in_range_counts = token_count[in_range_bitmask]

    // Cumulative Sum Check
    for s, c in zip(in_range_scores, in_range_counts):
        acc_sum += s * c
        selected.append(index of cluster)
        if acc_sum > Th_wics:
            goto EXIT_EARLY  // ← Early Exit trigger

    curr_range_start = range_end

EXIT_EARLY:
// Output: selected cluster indices → map to token indices via HC table
// avg only 16% of total scores processed per row
```

GPU vs Hardware对比：GPU上早期退出需要global synchronization和conditional branching，与SIMT模型冲突导致warp divergence和underutilization。WTU专用硬件通过bucket sorters（upper + lower）和adder tree流水线化处理，无需synchronization，bitmask-based selection消除branching overhead。消融实验：AGX+ReSV (GPU)上KV prediction占48% latency → +KVPU (hardware)降至0.5%，其中WTU early-exit sorting贡献了significant reduction。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

硬件实现（V-Rex WTU）：每WTU core含upper bucket sorter、lower bucket sorter、multipliers、adder tree、bucket range updater。数据处理流程：preprocess预计算weighted sum和threshold→从高分bucket开始bucket sort（bitmask-based parallel sort）→multipliers+adder tree累积→比较器check→early exit。软件GPU模拟：使用PyTorch的`torch.sort(descending=True)`配合`torch.cumsum`和mask可实现功能等价版本，但缺少真正的early exit（仍需完整sort）。通用CPU实现可使用partial sort（`std::partial_sort`）或priority queue（`std::priority_queue`）配合cumulative sum check实现近似early exit。

涉及论文标题：
- V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval

## Stripe-based VMM (Vector-Matrix Multiplication) / 条纹式向量矩阵乘法

术语是什么？通过联网搜索让回答具体和精准。

Stripe-based VMM是RPU论文提出的一种将Vector-Matrix Multiplication (O=V×W, V∈R^{1×K}, W∈R^{K×N})分块执行的策略，专为低延迟LLM decode的分布式VMM设计。一个stripe定义为8个垂直堆叠的tile（一个tile=8列×K行weight子矩阵）跨所有weight shard列。VMM按stripe粒度执行：先处理一个stripe内所有列tile的tile-rows（column-first traversal），做完列级tree-sum reduction后写入local register，再进入下一stripe。Stripe-based执行在三种常见VMM遍历策略中取得平衡：(1) Column-first (inner-product风格)需要full activation vector on-chip；(2) Row-first (outer-product风格)产生高partial sum writeback bandwidth；(3) Stripe-based最小化on-chip bandwidth需求，并使计算与通信fine-grained overlap（处理当前stripe时network buffer收集下一stripe所需activation shard）。

从kernel调度角度拆解：

Stripe-based VMM的伪代码执行流程（以O = V × W, C个core column-sharded的分布式场景为例）：

```
// VMM Stripe-based Kernel
// W sharded across C cores: each core stores W_i ∈ R^{K×N/C}
// Core i computes O_i = V × W_i (output fragment)

// === Stripe Definition ===
// Stripe: 8 vertically stacked tiles spanning all columns of W_i
// Tile: 8 columns × K rows of W_i (actually: 8×8 weight sub-matrix per TMAC cycle)
// Activation shard per stripe: 64 BF16 values

// === For each stripe s in W_i ===
for stripe_s = 0 to (N/C / 64) - 1:
    // Phase 1: Wait for activation shard
    activation_shard = network_buffer.read_valid(64 BF16 values)
    write_to_activation_register_file(activation_shard)  // 64 entries, 8 per tile column

    // Phase 2: Column-first traversal within stripe
    // (Process tile-column 0 first, then tile-column 1, ..., tile-column 7)
    for tile_col in 0..7:
        // Phase 2a: Tile-row iteration (weight-streaming)
        for tile_row in 0..(K/8)-1:
            weight_tile = stream_decoder.decode(memory_buffer.read())
            // weight_tile is 8×8 BF16, broadcast to all TMAC columns
            TMAC.compute(activation_shard[tile_col], weight_tile)
            // 64 MACs/cycle: activation[col] broadcast across 8 columns,
            // weight elements stream in element-wise

        // Phase 2b: Column-wise tree-sum reduction
        // After all tile-rows in this column processed:
        partial_sums[0..7] = TMAC.accumulators[tile_col][0..7]
        output[tile_col] = tree_sum_3stage(partial_sums)  // 3-stage adder tree

    // Phase 3: Write back stripe results
    local_register_file.write(output[0..7])
    // Meanwhile, network_buffer asynchronously collects next stripe's activation

// === After all stripes: output fragment O_i ready ===
// O_i forwarded to downstream cores via Network DMA
```

Striping的关键特性：
1. **最小化on-chip bandwidth**：每次仅需64 BF16 activation on-chip（vs full K-length vector for inner-product）
2. **低writeback pressure**：仅stripe结束时writeback一次（8个output values），而不是每tile writeback
3. **通信-计算overlap**：处理当前stripe时network异步收集下一stripe activation
4. **高度适配weight-streaming dataflow**：weight tiles按stripe内行列顺序依次stream through TMAC

术语一般如何实现？如何使用？

Stripe-based VMM通过RPU compiler生成的ISA指令实现。Compiler在将torch.nn.Linear lowering为三阶段micro-kernel时，Looping阶段内嵌stripe iteration logic：Loading stage配置DMA从HBM-CO pre-read weight tiles for next stripe → Loop stage drive TMAC execute current stripe (column-first traversal + tree-sum) → Launch stage forward activation to next core。TMAC hardware直接支持stripe-based dataflow——64-entry activation register file、3-stage tree-sum reduction、per-column accumulator management。Pipeline Arbiter管理stripe间的activation和weight buffer entry同步。

涉及论文标题：
- RPU - A Reasoning Processing Unit

## Decoupled Memory-Compute-Network Pipelines（解耦的内存-计算-网络流水线）

术语是什么？通过联网搜索让回答具体和精准。

Decoupled Memory-Compute-Network Pipelines是RPU论文提出的一种微架构设计模式，将传统GPU中耦合的内存访问、计算和网络通信三条数据路径分离为独立、可异步前进的硬件pipeline。每pipeline有独立的DMA engine（Memory DMA、Compute DMA、Network DMA）和专用SRAM buffer。关键创新在于：pipeline之间通过Pipeline Arbiter在buffer entry粒度做data-driven同步（非global barrier），使得任一pipeline可以在其他pipeline stall时继续前进——例如memory pipeline在compute等待network activation时继续预取weights到on-chip buffer。这种设计解决了低batch LLM decode中因kernel launch、synchronization和small distributed matrices导致的memory bandwidth underutilization问题（H100在decode时仅达32.2% BW utilization）。

从kernel调度角度拆解：

Decoupled pipelines的kernel执行伪代码（以Llama3-8B, BS=1, 64-CU RPU为例）：

```
// === Decoupled Pipeline Execution for One Transformer Layer ===
// Each core autonomously executes its instruction stream
// Three pipelines operate concurrently with buffer-level synchronization

// --- Memory Pipeline (never stalls on compute/network) ---
memory_pipeline:
    for each weight_tile in layer.weights:
        // Prefetch weights from HBM-CO to memory buffer
        DMA_HBM_to_MemBuf(weight_addr, mem_buf_entry)
        mem_buf[mem_buf_entry].valid_count = consumercount  // set by Pipeline Arbiter
    for each kv_cache_tile:
        DMA_HBM_to_MemBuf(kv_addr, mem_buf_entry)
        mem_buf[mem_buf_entry].valid_count = consumercount

// --- Compute Pipeline ---
compute_pipeline:
    for each stripe in VMM:
        // Check-valid: stall if activation not yet in network buffer
        network_buf[act_entry].check_valid()
        activation = network_buf[act_entry].read()
        register_file.write(activation)  // 64 BF16

        for each tile_col in stripe:
            for each tile_row:
                // Check-valid: stall if weight tile not yet decoded
                mem_buf[weight_entry].check_valid()
                weight = mem_buf[weight_entry].read()
                decoded_weight = stream_decoder.decode(weight)  // on-the-fly dequant
                TMAC.compute(activation[col], decoded_weight)

            tree_sum_reduce(TMAC.accumulators[col])

        // Write output to local register
        local_reg.write(output)

// --- Network Pipeline ---
network_pipeline:
    // Receive activation fragments from upstream cores
    for each activation_fragment:
        DMA_Recv(upstream_core, net_buf_entry)
        net_buf[net_buf_entry].valid_count = 2  // consumed by: compute + forward
    // Forward activation fragments to downstream cores
    for each forward_fragment:
        net_buf[fwd_entry].check_valid()  // wait for compute to produce
        DMA_Send(downstream_core, fwd_entry)
```

解耦带来的关键执行行为（Fig.8 simulation trace）：
- **BS=1, wQKV阶段**: network latency-limited（activation broadcast across CUs），memory pipeline继续预取weights（提前~80KB ahead of compute）
- **BS=1, QK^T阶段**: 跨CU gather Q/K/V shards + distributed max/reduction → compute stalls → memory pipeline prefetches KV$ entries
- **BS=32, wUp/wGate阶段**: compute-bound（weight processing ~4× longer than memory read）→ memory pipeline prefetches deep ahead (~6MB/CU) → buffer absorbs phase imbalance
- **无decoupling对比**: 全局barrier同步会使用memory/compute/network互相等待，导致累计stall延迟增加至1.6×

术语一般如何实现？如何使用？

在RPU中，decoupled pipelines通过以下机制实现：1) 每个pipeline有独立DMA engine和专用address space；2) SRAM buffer entry粒度的valid counter（Pipeline Arbiter）；3) NUMA at all scales（每core独立NUMA domain，无shared memory，跨domain通信显式由software-programmable DMA管理）；4) RPU ISA指令embed Pipeline Arbiter flags（check-valid/valid-count set）。这种设计与GPU的host-driven offload + global barrier模式根本不同——GPU需要等所有thread blocks完成kernel后才释放barrier，而RPU的pipeline在数据ready时立即前进。

涉及论文标题：
- RPU - A Reasoning Processing Unit

## Weight-Streaming Output-Stationary Dataflow（权重流式输出驻留数据流）

术语是什么？通过联网搜索让回答具体和精准。

Weight-Streaming Output-Stationary Dataflow是RPU论文中TMAC（Tile Multiplier-Accumulator）采用的dataflow策略。在这种dataflow中：(1) **Weight-streaming**: 权重元素从off-chip memory→on-chip buffer→Stream Decoder→TMAC以streaming方式依次流入，每个weight element被消费后立即被下一个weight element替换（无on-chip weight reuse）；(2) **Output-stationary**: partial sum（output的partial accumulation）驻留在TMAC的local accumulator register中，跨多个weight tile累积直到一个stripe完成。Activation在streaming和output之间扮演中间角色——broadcast across TMAC columns但局限在一个stripe内（64 BF16 values），不跨stripe复用。这种dataflow是三种经典systolic-array dataflow（weight-stationary, output-stationary, input-stationary）的混合变体，专为LLM decode的VMM优化。

从kernel调度角度拆解：

Weight-Streaming Output-Stationary dataflow的伪代码（per TMAC, per stripe）：

```
// === TMAC Dataflow (8×8 MAC array) ===
// Activation: broadcast across columns (spatial reuse)
// Weight: stream through tiles (no on-chip reuse)
// Partial sum: stationary in accumulator (temporal accumulation)

// For one stripe (8 tile-columns, T tile-rows per column):
for col in 0..7:  // tile columns in stripe
    // === Activation loading (once per column) ===
    act_elements[0..7] = activation_register_file[col*8 : (col+1)*8]
    // act_elements stay in column broadcast registers
    
    // Reset column accumulators
    accumulators[col][0..7] = 0  // FP32
    
    for row in 0..T-1:  // tile rows
        // === Weight streaming (per tile) ===
        weight_tile = stream_decoder.decode_next()  // 8×8 BF16
        // weight_tile elements stream through MAC array:
        // - 8 columns × 8 rows = 64 MACs in parallel
        // - Each row receives different weight element
        // - All rows in same column receive same activation element
        
        // Parallel MAC (64 MACs/cycle):
        for r in 0..7, c in 0..7:
            accumulators[col][r] += act_elements[c] * weight_tile[r][c]
            // act_elements[c]: broadcast across column c
            // weight_tile[r][c]: element-wise streamed
            // accumulators[col][r]: output-stationary (stays in local reg)
    
    // === Column tree-sum reduction (after all tile-rows) ===
    for r in 0..7:
        output[col] = tree_sum_3stage(accumulators[col][0..7])
        // 3-stage reduction: 8→4→2→1
        // Stage 1: (A0+A1), (A2+A3), (A4+A5), (A6+A7)
        // Stage 2: (A01+A23), (A45+A67)
        // Stage 3: A0123 + A4567

// After all 8 columns: output[0..7] ready
```

Dataflow选择理由（vs alternatives）：
- **Weight-stationary**: weight需要保持在MAC array中跨多个activation复用——在decode VMM中每个weight仅被使用一次（BS=1），无复用机会
- **Input-stationary (activation stationary)**: 需要full activation vector on-chip → 大activation（K-dim~16K for Llama3-70B）无法全部on-chip
- **Output-stationary**: partial sum保持在local accumulator，最小化writeback bandwidth——decode VMM中output vector size（N/C）相对较小（per-core shard of N-dim），适合on-chip保持

术语一般如何实现？如何使用？

在RPU中，TMAC硬件直接实现weight-streaming output-stationary dataflow：1024-bit compute bus每cycle deliver一个8×8 BF16 weight tile（64 elements）到TMAC；activation register file保持64 BF16 activation values（8 columns × 8 rows broadcast）；每个MAC单元的FP32 accumulator独立保持partial sum，直至stripe结束tree-sum reduction到local register file。Dataflow由RPU ISA指令驱动compiler决定，运行时TMAC硬件自动执行fixed streaming schedule。

涉及论文标题：
- RPU - A Reasoning Processing Unit

## GPU Acquire/Release Synchronization and Cache Coherence Actions / GPU获取/释放同步与缓存一致性动作

术语是什么？

GPU的Acquire/Release同步是GPU编程模型中实现线程间同步和内存一致性的核心机制。在CUDA等GPU编程模型中，acquire（获取）和release（释放）操作标记了同步边界：acquire表示后续内存访问必须看到release之前的所有写入；release表示之前的所有内存写入对后续acquire可见。在硬件层面，传统GPU（monolithic）实现acquire/release同步的方式是：acquire操作invalidate所有本地私有cache（L1 cache），确保后续load从全局共享的LLC读取最新数据；release操作flush所有dirty data从本地cache写回LLC，确保之前的所有store对后续acquire可见。Atomic同步操作（atomicCAS、atomicAdd等）通常绕过本地cache直接在LLC中执行，以确保原子性和一致性。

在多chiplet GPU中，由于额外的cache层级（L1.5 cache），acquire/release的coherence action变得更加昂贵：(1) Acquire不仅需要invalidate L1，还需要invalidate整个L1.5 cache（更大容量、更多way、影响chiplet内所有SM），开销远大于单L1 invalidation；(2) Release若L1.5为write-back policy需flush L1.5 dirty data到LLC，若write-through则无flush开销但acquire仍需invalidate；(3) 跨chiplet atomic同步受inter-chiplet有限带宽约束，多个SM对同一地址的atomic操作（如spin lock的atomicCAS重试）产生大量跨chiplet流量。

从kernel调度角度拆解：

GPU同步操作的硬件执行流程（以lock-based synchronization + CUDA kernel为例）：

```
// CUDA kernel 伪代码
__global__ void kernel_with_lock() {
    // ... local computation ...
    
    // ACQUIRE phase: spin until lock acquired
    while (atomicCAS(&lock, 0, 1) != 0);
    __threadfence();  // acquire fence: invalidate local caches
    // Hardware action (MCM-GPU): invalidate L1 + L1.5 cache
    // Hardware action (LRM-GPU): query sync-val directory;
    //   if owner=local chiplet: skip L1.5 invalidation;
    //   if owner=remote: flush remote L1.5 + invalidate local L1.5
    
    // CRITICAL SECTION: access shared data
    val = shared_data;       // load (must see latest value)
    shared_data = new_val;   // store
    
    // RELEASE phase: release lock
    __threadfence();  // release fence: ensure stores visible
    atomicExch(&lock, 0);
    // Hardware action (MCM-GPU w/ write-through L1.5): no flush needed
    // Hardware action (LRM-GPU): write sync var to LLC, delay L1.5 actions
}
```

同步操作的coherence action组合（以4-chiplet GPU, write-through L1/L1.5为例）：
- **MCM-GPU acquire**: invalidate L1 (per-SM) + invalidate L1.5 (per-chiplet, affects ALL SMs!)
- **MCM-GPU release**: write-through L1/L1.5 → LLC is up-to-date, no flush
- **MCM-GPU atomic**: bypass L1/L1.5 → route to LLC → cross-chiplet via inter-chiplet network
- **LRM-GPU acquire**: query directory → local owner: LLC read only; remote owner: flush old L1.5 + update owner + invalidate new L1.5
- **LRM-GPU release**: write sync var to LLC → local owner: no extra actions; remote: flush L1.5 + update owner + invalidate L1.5
- **LRM-GPU atomic**: AMU in-network merge → combined request sent → multicast response

术语一般如何实现？如何使用？

GPU acquire/release同步的实现要素：(1) Memory fence指令——CUDA中`__threadfence()`（global fence）、`__threadfence_block()`（block-level fence），PTX中对应`membar.cta`（CTA scope）和`membar.gl`（GPU scope）指令。这些指令在硬件层面触发cache coherence actions。(2) Hardware coherence actions——传统GPU通常采用software-driven coherence而非硬件自动coherence protocol（如CPU的MESI）。acquire/release通过fence指令显式触发硬件invalidate/flush。(3) 同步原语——GPU支持多种同步模式：barrier (`__syncthreads`用于block内同步，global barrier需atomic+spin实现)、lock（atomicCAS实现spin lock）、semaphore（atomicAdd/Sub实现计数信号量）、atomic update（atomicAdd更新共享数据结构如histogram的bin）。每种模式对acquire/release coherence action的需求不同。(4) Multi-chiplet差异化——额外cache level迫使同步机制考虑多级cache incoherence问题。HMG使用完整coherence protocol（类似VI protocol）维护所有cache line状态；hLRC缓存同步变量在多级cache中但引入跨SM同步变量write-back等待和重试；LRM-GPU采用折中：同步变量不缓存但引入lightweight owner tracking减少L1.5 coherence action（仅在跨chiplet ownership迁移时触发）。

涉及论文标题：
- LRM-GPU: Alleviating Synchronization Overhead for Multi-Chiplet GPU Architecture

## In-Network Atomic Merge for GPU Synchronization / GPU同步的网络内原子合并

术语是什么？

In-Network Atomic Merge是一种在网络（NoC/interconnect）内部对跨chiplet的同步atomic请求进行合并处理的技术，由LRM-GPU提出。传统GPU中，每个SM发出的atomic同步操作独立通过网络传输到LLC执行，当多个SM竞争同一同步变量（如spin lock的atomicCAS）或更新同一共享地址（如atomicAdd更新histogram bin），这些atomic请求在网络中产生大量冗余流量，尤其在跨chiplet场景下inter-chiplet有限带宽成为瓶颈。In-network atomic merge在网络中嵌入专用的Synchronization Atomic Merge Unit (AMU)，检测去往同一地址的多个atomic请求，根据可合并性将其合并为单一的aggregated request发送，响应返回时再通过multicast向所有参与SM广播结果。这种方法减少跨chiplet atomic传输次数和LLC atomic执行次数，缓解inter-chiplet bandwidth pressure。

从kernel调度角度拆解：

AMU对atomic同步操作的合并调度流程：

```
// AMU request processing pipeline
function AMU_Process_Atomic_Request(req):
    if not is_cross_chiplet(req.target_addr):
        forward_directly(req)  // local chiplet requests bypass AMU
        return
    
    // CAM lookup: match status=VALID + mergeable opcode + same address
    entry = merge_table.cam_lookup(req.opcode, req.addr)
    
    if entry and entry.status == VALID and can_merge(entry, req):
        // HIT: merge with existing entry
        entry.data = alu_merge(entry.opcode, entry.data, req.data)
        entry.sm_list.append(req.sm_id)
        if entry.sm_list.size >= MAX_SM_LIST:  // SM list full → send
            send_merged_request(entry)
            entry.status = RESERVE
        return
    
    if has_free_entry(merge_table):
        // MISS + free entry: allocate new entry, start timer
        new_entry = alloc(status=VALID, opcode=req.opcode,
                          addr=req.addr, sm_list=[req.sm_id], data=req.data)
        start_countdown_timer(new_entry)
        return
    
    // No free entry: bypass AMU, send directly
    forward_directly(req)

// Timer callback or SM list full
function send_merged_request(entry):
    send_to_llc(entry.opcode, entry.addr, entry.data)
    entry.status = RESERVE  // block further merging until response

// Response processing
function AMU_Process_Response(rsp):
    entry = merge_table.lookup_by_addr(rsp.addr)
    if entry and entry.status == RESERVE:
        // Multicast broadcast to all participating SMs
        for sm_id in entry.sm_list:
            forward_to_sm(sm_id, rsp)
        release_entry(entry)  // entry → INVALID, reusable
    else:
        forward_directly(rsp)  // pass-through for non-merged responses
```

支持的atomic类型及合并规则：
- **atomicAdd/Sub/Min/Max/And/Or/Xor**: 可自由合并（commutative/unordered），如 atomicAdd(a, 1) + atomicAdd(a, 1) → atomicAdd(a, 2)
- **atomicCAS**: 仅在comparison data相同时合并 → 选一个作为combined request，其余等待返回fail结果
- **Cross-cache-line**: 同一coarse-grained地址区域内不同offset的请求按operation-mask合并

术语一般如何实现？如何使用？

In-network atomic merge的实现要点：(1) 网络内处理 vs 端点处理——传统atomic合并方案（ARC[11]在warp内、LAB[10]在SM内的atomic buffer、Atomic Cache[54]在cache内实现atomic）在请求进入网络前合并，受限于SM/warp局部范围。In-network merge (AMU) 在网络中合并，能看到跨多个SM的atomic请求，合并机会显著更大。(2) 合并窗口设计——通过countdown timer或SM list阈值控制合并窗口，需要在合并机会（窗口长）和latency overhead（窗口短）之间trade off。论文在4-chiplet系统中AMU贡献1.16×加速和12% traffic reduction。(3) 正确性保证——可交换/无序的atomic（Add/Sub/And/Or/Xor/Min/Max）天然可合并且结果等价于串行执行；atomicCAS需要comparison data相同才合并（否则不同比较值的atomicCAS不应互相影响）。(4) 与cache coherence的交互——AMU在LLC之前合并请求，不改变LLC对atomic操作的执行语义；multicast broadcast根据SM list向所有参与SM返回结果。AMU与LRC互补：LRC减少cache invalidation overhead，AMU减少inter-chiplet atomic bandwidth。(5) Broadcast efficiency——一个合并请求的响应替代多个单独请求的响应，进一步减少inter-chiplet返回流量和LLC响应端口压力。

涉及论文标题：
- LRM-GPU: Alleviating Synchronization Overhead for Multi-Chiplet GPU Architecture

## Coalesced Dual-Input SpMM Kernel

术语是什么？

Coalesced Dual-Input SpMM Kernel 是 Swift 论文提出的 GPU SpMM kernel 设计策略，核心目标是同时实现稀疏矩阵 A 和稠密矩阵 B 的 coalesced memory access。传统 GPU SpMM 方法（Sputnik/ASpT/RoDe/cuSPARSE）通常只优化稀疏矩阵侧的存储格式或线程负载均衡，未考虑 warp 内线程在访问 A 的稀疏索引后，用这些索引访问 dense B 时产生的地址跳跃（接近 warpSize 次 memory transaction）。Swift 通过 sparsity-based column sorting + dense row rearrangement + CSC format + warp-size blocking 的组合设计，让 warp 内相邻线程处理的稀疏列对应 B 中连续地址，使 A 和 B 的加载都具备高度合并访问。

从kernel调度角度拆解术语：

Swift 实现 coalesced dual-input 的 kernel 执行流程：

```
// Regular kernel: coalesced dual-input SpMM (1 warp per 32-column block)
// A: M×K sparse (CSC), B: K×N dense (column-major, rows rearranged)
// C: M×N dense output

// Thread block = 32×8 threads (8 warps)
warp_id = tid / 32
lane_id = tid % 32
block_col_start = blkColIdx[warp_id]         // 该 warp 处理的起始列号
block_offset = blkPtr[warp_id]               // value/rowIdx 数组起点

for offset = block_offset to next_block_offset:
    // Step 1: Coalesced sparse A read
    // 所有 lane 读取同一列 block 内不同行的非零元
    value = A_value[offset + lane_id]          // 32 threads → 128B aligned
    row_idx = A_rowIdx[offset + lane_id]       // 32 threads → 128B aligned

    // Step 2: Coalesced dense B read
    // col_idx = block_col_start + lane_id → 连续 32 列
    col_idx = block_col_start + lane_id
    // B 为 column-major，连续列索引 = 连续地址
    for j = 0 to N-1:
        b_val = B[col_idx * K + j]             // 32 threads → coalesced
        partial = value * b_val

    // Step 3: Shared memory store (coalesced per warp)
    smem[lane_id] = partial

    // Step 4: Segment sum in shared memory
    // positionIdx/offsetIdx guide intra-warp reduction (see Segment Sum term)
    ...

    // Step 5: Reduced atomicAdd write-back to C
    segment_sum = smem_reduce(row_idx)
    atomicAdd(&C[row_idx * N + j], segment_sum)
```

coalescing 效果量化：
- 论文实验观察：数据加载相关开销平均超过整体性能的 32%
- Regular part coalesced B access 相对 non-coalesced 版本：N=32 时 1.32×, N=128 时 1.38× 几何平均 speedup
- Column-major B layout 是 coalescing 的关键：warp 内 32 线程处理连续 32 列 → B 地址 = colIdx × K + j → 相邻线程 colIdx 差 1 → 地址差 K（连续）

术语一般如何实现？如何使用？

Coalesced dual-input 的实现要求：
1. **稀疏格式选择**：必须使用 CSC (Compressed Sparse Column) 而非 CSR，因为 column-major 方向使同一列的非零元 row_idx 连续，同 warp 内线程按列分配时访问 B 的列地址也连续。
2. **列排序预处理**：按每列 NNZ 升序排序 A 的列，并同步重排 B 的对应行。排序后相邻列 NNZ 相近 → warp 内各 lane 工作量相似 → 负载均衡改善。B 行重排是关键——不能只排序 A 而让 B 保持原位（否则列索引映射错误）。
3. **Warp-size 对齐**：按 warpSize=32 划分 column block。列宽恰好为 32 的进入 regular kernel，不满足的进入 irregular kernel。
4. **适用条件**：非零元分布较均匀、NNZ 不过大（NNZ > 10^6 时排序+blocking 预处理开销可能超过收益）。

涉及论文标题：
- Swift: High-Performance Sparse-Dense Matrix Multiplication on GPUs

## Segment Sum Shared-Memory Reduction for GPU SpMM

术语是什么？

Segment Sum Reduction 是 Swift 在 regular SpMM kernel 中使用的一种 shared-memory 局部规约技术，用于在 global atomicAdd 之前先将同 row_idx 的 partial product 在 warp/shared memory 内部合并，从而大幅减少 atomic 操作次数和地址冲突。Swift 维护 positionIdx（记录每个非零元在其列内的位置，用于判断是否为 segment 起点）和 offsetIdx（记录 segment 在 shared memory 中的偏移量）两套辅助索引，指导 warp 内线程对 rowIdx 相同的 partial sum 做 segment-level 局部求和。

从kernel调度角度拆解术语：

```
// Swift Regular Kernel 的 segment sum 流程
// 输入: smem[0..31] 含 32 个 (row_idx, partial) pairs
//       每个线程已将其 partial product 写入 smem[lane_id]

// positionIdx[i] = 该非零元在所在列内的位置偏移
//   若 positionIdx[i] == 0 → 该非零元是所在列的第一个非零元 → segment 起点
// offsetIdx[i] = 该非零元对应的 segment 在归约 buffer 中的偏移量

// Step 1: 识别 segment 边界
is_segment_start = (positionIdx[lane_id] == 0)

// Step 2: 将 segment start 标记写入 shared memory
__syncthreads()
smem_flag[lane_id] = is_segment_start

// Step 3: Warp-level segment sum
// 按 row_idx 分组: 将同 row_idx 的 partial 累加到 offsetIdx 指定的 buffer 位置
for step = 0 to warpSize-1:
    if lane_id == step:
        seg_offset = offsetIdx[step]
        smem_buf[seg_offset] += smem_partial[step]

__syncthreads()

// Step 4: 仅 segment 起点线程写回
if is_segment_start:
    seg_offset = offsetIdx[lane_id]
    row = rowIdx[lane_id]
    atomicAdd(&C[row * N + j], smem_buf[seg_offset])
```

对比 naive 方法（每个线程都做 atomicAdd）：
- Naive: 32 个 atomicAdd 调用 → 大量 address contention → 数据加载开销 >32%
- Segment sum: 仅 segment 起点线程做 atomicAdd（通常 << 32）→ 大幅减少 atomic 冲突
- 例如：32 个非零元中 16 个指向同 1 个 row → naive 需 32 atomics（16 个竞争同一地址），segment sum 仅 1 次 atomic（16 个 partial 已在 SMEM 合并）

术语一般如何实现？如何使用？

Segment sum 的实现要素：
1. **辅助索引**：positionIdx 和 offsetIdx 在 CPU 预处理阶段构建，基于排序后 A 的列结构计算。对于 regular block（32 列），每列内非零元的 positionIdx 从 0 递增；offsetIdx 由同 row_idx 的首次出现位置决定。
2. **Shared memory 使用**：regular kernel 的 thread block 需要额外 SMEM 用于 segment sum buffer（通常 32×sizeof(float) = 128B per warp + flag 数组）。
3. **同步点**：segment start 标记写入后需要 `__syncthreads()`；segment sum 完成后需要 `__syncthreads()` 再写回。这些同步开销需要被 coalesced 加载带来的收益抵消。
4. **适用性**：segment sum 在 regular part 最有效，因为 regular block 的列结构规整，positionIdx/offsetIdx 紧凑。irregular part 因为列长分布不规则，仍用 stride-based 遍历 + 直接 atomicAdd。
5. **与前序工作的关系**：此策略与 hash-based write-back (VDHA) 的 shared-memory hash aggregation 共享"先局部聚合再写回"的思想，但实现不同：Segment sum 用确定性索引（positionIdx/offsetIdx）而非 hash probing，无需处理 hash collision 和 fallback。

涉及论文标题：
- Swift: High-Performance Sparse-Dense Matrix Multiplication on GPUs

## Regular/Irregular Block Partition for GPU SpMM

术语是什么？

Regular/Irregular Block Partition 是 Swift 提出的 GPU SpMM 预处理策略。在按 NNZ 排序稀疏矩阵 A 的列之后，Swift 将 A 按 warpSize=32 划分为两类 block：(1) Regular block：列宽恰好等于 32 的连续列组，每个 block 由 1 个 warp (32 threads) 处理；(2) Irregular block：不足 32 列的短列组或超过 warpSize 的长列残留元素。两类 block 用不同的 kernel 路径处理——regular kernel 优先追求 memory coalescing（通过 segment sum 减少 atomic），irregular kernel 优先追求 load balancing（通过 sub-column 拆分避免单 warp 阻塞）。

从kernel调度角度拆解术语：

```
// Swift 预处理: blocking 阶段
// 输入: A 已按列 NNZ 升序排序的 CSC 格式
// 输出: regular 和 irregular 两套索引结构

block_start = 0
regular_count = 0
irregular_count = 0

while block_start < K:  // K = A 的列数
    // 尝试构成 width=32 的 regular block
    if block_start + 32 <= K:
        // Regular block: 32 个连续列
        blkPtr[regular_count] = nnz_offset            // block 在 value/rowIdx 中的起点
        blkColIdx[regular_count] = block_start        // block 起始列号
        // 为 block 内非零元生成 positionIdx, offsetIdx (用于 segment sum)
        regular_count++
        block_start += 32
    else:
        // Irregular part: 剩余不足 32 列
        break

// 剩余列 + 未进入 regular 的长列残块 → irregular part
for col = block_start to K-1:
    nnz_in_col = colPtr[col+1] - colPtr[col]
    if nnz_in_col > SPLIT_THRESH:  // 长列拆分
        num_subcols = ceil(nnz_in_col / SPLIT_SIZE)
        for s = 0 to num_subcols-1:
            irrPtr[irregular_count] = colPtr[col] + s * SPLIT_SIZE
            colIdxIndex[irregular_count] = col       // 记录原列号
            blkStart[irregular_count] = s * SPLIT_SIZE
            blkStop[irregular_count] = min((s+1)*SPLIT_SIZE, nnz_in_col)
            irregular_count++
    else:  // 短列直接作为一个 block
        irrPtr[irregular_count] = colPtr[col]
        irregular_count++
```

两类 kernel 的调度差异：
- **Regular kernel**: thread block = 32×8 (8 warps)，每 warp 负责 1 个 regular block (32 列)。执行路径固定：coalesced 加载 → shared memory segment sum → 少量 atomicAdd。
- **Irregular kernel**: warp 动态调度——通过 colIdxIndex 判断当前任务是独立短列还是长列子块 → blkStart/blkStop 定位范围 → lane 以 stride=32 遍历范围 → 对每个非零元循环访问 B 的 N 列 → atomicAdd 写回。

术语一般如何实现？如何使用？

Blocking 的实现要点：
1. **warpSize 耦合**：regular block width=32 与 CUDA warp size (32) 严格对齐，保证每线程一列的 1:1 映射。
2. **索引结构**：Regular 需要 blkPtr、blkColIdx、positionIdx、offsetIdx。Irregular 需要 irrPtr、irrValue、irrRowIdx、colIdxIndex、blkStart、blkStop。
3. **预处理开销**：blocking 需要在 CPU 端完成，包括列分类和索引生成。NNZ > 10^6 时预处理成本可能超过或接近 Sputnik/RoDe 的开销。
4. **消融收益**：irregular part 的 load-balancing 优化（相对 naive 单 warp 处理长列）在 N=32 和 N=128 下分别带来 2.26× 和 2.69× 几何平均 speedup。
5. **适用条件**：非零元分布较均匀时 blocking 效果最好；当非零元集中在少数 32×32 block 中时收益下降，此时 ASpT 的 adaptive tiling 可能更优。

涉及论文标题：
- Swift: High-Performance Sparse-Dense Matrix Multiplication on GPUs

## UWMMA (Unified Warp-level Matrix Multiply-Accumulate)

术语是什么？

UWMMA 是 Uni-STC 论文提出的统一 warp 级矩阵乘累加指令序列接口，用于替代传统 GPU dense tensor core 的 MMA 指令。UWMMA 由三类指令组成——stc.load（同步收集 BBC 格式的 metadata 和数值）、stc.task（异步触发 TMS/DPG 硬件生成 task queue）、stc.numeric（检查 READY/BUSY 状态驱动 SDPU 执行）——将稀疏计算的控制流从"gather data"转为"gather tasks"模式。与传统 MMA 指令要求 A/B 数据在调用前完全就绪不同，UWMMA 的异步 task generation 允许 task 分解与 SM 后续操作重叠执行。

从kernel调度角度拆解术语：

UWMMA 指令序列的伪代码（以 SpMM 为例）：
```
// 一个 warp 执行 SpMM C[i_row_batch][j_col] = A[i_row_batch][k] × B[k][j_col]
// 假设 A 以 BBC 格式存储，B 以 BBC 格式存储

warp:
  // Phase 1: Load
  stc.load.meta rA_meta, [A_bbc_base + row_meta_off]   // load A 的 top-level CSR tile metadata
  stc.load.val  rA_vals, [A_bbc_base + val_off]         // load A 的 tile values
  stc.load.meta rB_meta, [B_bbc_base + col_meta_off]    // load B 的 tile metadata
  stc.load.val  rB_vals, [B_bbc_base + val_off]         // load B 的 tile values
  stc.load.meta rC_meta, [C_bbc_base + row_meta_off]    // load C 的 tile metadata (用于 T4 code 中编码 C 写入目标)

  // Phase 2: Task Generation (async)
  stc.task.gen.mm                                      // 触发 TMS → DPG task generation pipeline
  // TMS: 从 Meta Buffer 读 top-level bitmap → 沿 K 维匹配 A/B tile
  //      → 拆分 16×16×16 T1 为多个 4×4×4 T3 → 写入 Tile queue
  // DPG: 从 Tile queue 弹 T3 → 读 bottom-level bitmap → overlay A/B bitmap
  //      → 生成 1×1×4 T4 task code → 以 Z-shaped 顺序入 Dot-product queue

  // Phase 3: Numeric Execution
  check_ready:
    stc.poll.ready r_status                              // 检查 Dot-product queue 是否有 READY T4 task
    if r_status == BUSY: goto check_ready               // task generation in progress, stall

  stc.numeric.mm                                        // 驱动 SDPU 执行
  // SDPU: 弹出 T4 task code → 解码 A/B 操作数地址
  //      → 执行 1×1×4 segmented dot-product → 累加到 accumulator
  //      → merge-forward 合并 partial products → 写 C

  // Phase 4: Write-back
  stc.writeback rC_vals, [C_bbc_base + val_off]         // 更新 C 的 BBC value 区
```

指令设计的关键特性：
- **异步 task generation**：stc.task 提交后 warp 可继续执行其他指令，通过 stc.poll.ready 检查状态
- **统一 opcode**：同一套 load/task/numeric 指令覆盖 SpMV、SpMSpV、SpMM、SpGEMM，仅在 `.gen.mm`/`.gen.mv`/`.gen.spspv`/`.gen.gemm` 后缀上有区分
- **软件控制 BBC 数据路径**：ValPtr_Lv2 直接提供给 Uni-STC，使 TMS 能控制 tile 内数据转发而无需复杂硬件解码

术语一般如何实现？如何使用？

UWMMA 需要 GPU SM 侧扩展：instruction decoder 新增 opcode 解析、warp scheduler 支持 UWMMA 指令分发。软件编译时，程序员通过类似 CUDA PTX inline assembly 的方式嵌入 UWMMA 指令序列。数据路径要求 register file 提供足够 operand port（Ampere 类需每线程每周期最多 16 个 FP64 source + 4 个 FP64 destination operands）。BBC 格式矩阵的 ValPtr_Lv2 指针在 stc.load 时直接传入 Uni-STC 硬件 buffer，后续 task generation 通过指针索引而非逐元素搬运。

涉及论文标题：
- Uni-STC: Unified Sparse Tensor Core

## Segmented Dot-Product

术语是什么？

Segmented Dot-Product 是 Uni-STC 的 SDPU 硬件执行单元采用的核心计算原语，将多个短向量点积（1×1×4 T4 tasks）按"相同 C 写入目标"的原则拼接成一个连续的执行 segment，在局部累积 partial products 后在段尾统一写回 C。与传统的"一个 task→一次 multiply-accumulate→一次 write C"模式不同，segmented dot-product 允许最多四个相邻 T4 task 的 partial products 在 SDPU 内部的累加器中被预合并（merge-forward），然后再写入 C tile 的 accumulator buffer，从而减少 C 网络上的写回流量和数据 conflict。

从kernel调度角度拆解术语：

Segmented dot-product 的执行伪代码：
```
// SDPU 执行一个 segment（包含多个连续性 T4 task）
// T4 task = {A_idx, B_idx, C_target, K_mask(4-bit)}

SDPU_execute_segment(tasks[T4_start..T4_end]):
    acc_cur = {target: -1, partial: 0}   // 当前段的部分和
    for t4 in tasks[T4_start..T4_end]:
        if t4.C_target != acc_cur.target and acc_cur.target != -1:
            // 段边界：预合并完成，写回 accumulator
            accumulator[acc_cur.target] += acc_cur.partial
            acc_cur = {target: t4.C_target, partial: 0}
        
        // 执行 1×1×4 dot-product: 对 K 维度的 4 个位置做乘加
        for k in 0..3:
            if t4.K_mask & (1 << k):
                acc_cur.partial += A[t4.A_idx + k] × B[t4.B_idx + k]
    
    // 写回最后一段
    if acc_cur.target != -1:
        accumulator[acc_cur.target] += acc_cur.partial

    // merge-forward: 最多合并 4 个不同 C_target 的 partial products 后
    // 通过 C network 写回 C tile
```

与传统 dot-product 的对比：
- **传统方法（RM-STC/DS-STC 类）**：每个 outer-product 或 row-row 任务独立计算→独立通过大 network 搬运中间乘积→独立写 C，带宽浪费大
- **Segmented dot-product（Uni-STC）**：T4 code 编码 C_target 信息→SDPU 在 pop T4 时识别段边界→同 C_target 的 T4 在内部累加→仅段结束时才触发 C 写入，网络带宽需求降低且 conflict 减少

术语一般如何实现？如何使用？

Segmented dot-product 由 SDPU 硬件实现，其内部包含：1KB accumulator buffer（存 C tile 的部分和）、merge-forward 逻辑（比较相邻 T4 的 C_target，最多合并 4 个 partial products 后写回）、Benes/MUX network（将 1-4 个 partial products 路由到正确 accumulator 槽位）。Z-shaped T4 task ordering 由 DPG 在写入 Dot-product queue 时选择，目标是将共享 A 或 B tile 的 T4 聚集以减少 A/B buffer 的重复读取。SDPU 的 merge-forward 深度（4）是 hardware cost vs. write traffic reduction 的 trade-off——更深可进一步减少 C traffic 但增加 merge-forward logic 面积和 critical path。

涉及论文标题：
- Uni-STC: Unified Sparse Tensor Core

## Gather Tasks Paradigm

术语是什么？

"Gather Tasks"（任务聚合/拼接）是 Uni-STC 提出的稀疏张量核调度范式，核心思想是将稀疏计算的控制流从"gather data"（将稀疏数据 gather 成固定形状任务后送入 MAC array）转为"gather tasks"（生成轻量控制 task，将多个低负载任务灵活拼接后统一执行）。传统稀疏 TC（如 DS-STC、RM-STC）的 gather data 模式遇到长行、窄行或双边稀疏时，固定形状任务导致大量 MAC 利用率不足的周期。Gather tasks 则通过 BBC 格式的两级 bitmap + TMS/DPG 的任务拆分 + SDPU 的 segmented dot-product 拼接，使每周期 MAC 利用率更高且中间数据传输量更小。

从kernel调度角度拆解术语：

Gather data vs Gather tasks 的调度流程对比：

**Gather data 模式（RM-STC / DS-STC）：**
```
for each row in A (warp-level):
    // Step 1: 从 DRAM gather 该行和对应 B 列的数值到 SMEM/register
    load A_row_vals = gather(A[row_indices], DRAM)
    load B_col_vals = gather(B[col_indices], DRAM)
    
    // Step 2: 将数值组织成固定形状的 T2/T3 任务
    task = pack_to_fixed_shape(A_row_vals, B_col_vals, task_shape)
    // 问题：如果 A_row 非零少（短行），task_shape 填不满 → MAC 空转
    //      如果 A_row 非零多（长行），一个 T2/T3 装不下 → 拆成多个独立任务
    //      多个独立任务无法跨 K 维拼接 → 增加无效访存和 network traffic
    
    // Step 3: 送入 MAC array 执行
    result = MAC_array.execute(task)
```

**Gather tasks 模式（Uni-STC）：**
```
// 软件侧：BBC 格式已在预处理时完成 tile 组织和 bitmap 编码
stc.load.meta...   // 仅加载轻量 bitmap metadata 和 values 到硬件 buffer

// 硬件侧自动执行 task generation + concatenation:
// TMS: 从 top-level bitmap → 动态判断 A/B tile 沿 K 维的重叠情况 → 
//      选择拼接策略（outer-product vs row-major ordering）
//      → 生成灵活形状的 T3 (4×4×4) tasks → 入 Tile queue
// DPG: 弹 T3 → 解析 bottom-level bitmap → 生成 T4 (1×1×4) task code
//      → 以 Z-shaped order 入 Dot-product queue → 
//      低负载 task 自然拼接
// SDPU: 弹 T4 → 执行 segmented dot-product → 
//       同 C_target 自动预合并 → 减少写回

// 关键：queue 中存的是 task code（8-bit）而非中间乘积 → 低带宽
```

量化效果：RM-STC 和 DS-STC 分别有 62.78% 和 61.68% 的周期 MAC utilization < 50%，Uni-STC 的低利用率周期比例显著降低。在 SuiteSparse 全量矩阵上，Uni-STC 对 DS-STC/RM-STC 的几何平均 speedup 为 3.35x/2.21x，energy reduction 为 1.97x/1.27x。

术语一般如何实现？如何使用？

Gather tasks 范式要求软硬件协同：(1) 软件侧用 BBC 格式存储稀疏矩阵，bitmap 结构直接供硬件解析；(2) 硬件侧实现 TMS（task merge & split）、DPG（dot-product generation）和 SDPU（segmented dot-product）三级流水线；(3) 指令接口（UWMMA）支持异步 task generation。关键 trade-off 是新增硬件面积和 BBC 预处理成本 vs. gain（MAC utilization 提升 + 中间产品网络带宽节省）。Uni-STC 论文评估额外面积约 2.12%，预处理时间可被迭代应用摊销。

涉及论文标题：
- Uni-STC: Unified Sparse Tensor Core

## Wavefront Specialization（波前专业化）

术语是什么？通过联网搜索让回答具体和精准。

Wavefront Specialization（也称warp specialization）是GPU kernel编程优化技术，将workgroup内的wavefront（NVIDIA术语warp）分配不同角色，而非所有wavefront执行相同指令（传统SIMT模式）。典型producer-consumer分工：一个或多个dedicated producer wavefront专门执行memory transfer（如ATT load从global memory搬运tile到LDS），其余consumer wavefronts专门执行computation。这种分工引入compute heterogeneity但enable fine-grained overlap of data movement and computation——producer wavefront在后台异步搬运数据时，consumer wavefronts在前台同时计算前一个tile。该技术要求精确同步（custom barriers）防止data race和死锁。Wavefront specialization最初由Bauer等人（PPoPP'14, Singe）提出，后在NVIDIA Hopper架构上因TMA+WGMMA的异步能力成为主流优化模式（CUTLASS 3.x, FlashAttention-3均采用）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

QuCo论文中wavefront specialization + Operand Queue的kernel执行伪代码（以Matrix-Matrix为例）：

```
// ===== Kernel Dispatch =====
// Workgroup = N wavefronts: WF 0 是 Producer, WF 1..N-1 是 Consumers

// ===== Producer Wavefront (WF 0) =====
if (wf_id == 0):
    for each tile_i in K_dimension:
        queue.Push(tile_i)          // 发起 ATT async tile load → LDS
        async_barrier.arrive()      // non-blocking: 通知数据就绪
        queue.Wait_For_Push()       // 确保 ATT 写入 LDS 完成
    async_barrier.exit_producer()

// ===== Consumer Wavefronts (WF 1..N-1) =====
else:
    while has_work:
        async_barrier.wait()        // 阻塞直到 producer arrive
        tile_data = queue.Peek(idx) // 只读访问 LDS 中 tile
        partial += compute(tile_data) // 矩阵乘法等计算
        queue.Pop(idx)              // 释放 LDS slot 供下一 tile
```

关键参数决策（QuCo自动化的内容）：
1. **tile size**：决定每次ATT transfer数据量→影响memory bandwidth utilization和LDS占用→需与kernel compute intensity匹配
2. **queue slots数量**：决定多少tile可同时in-flight→slots过少无法充分overlap（pipeline bubble），slots过多浪费LDS且增加memory contention
3. **consumer wavefront数量**：决定compute parallelism→更多consumer更快消费tile但增加scheduling pressure

传统手动wavefront specialization的痛点：同一kernel在不同GPU上的最优配置不同（R9 Nano→MI-100差1.4×），不同kernel间不能复用（可达1.2× degradation），需exhaustive DSE（Matrix-Matrix需2.6×10^14次kernel launch逐一尝试）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

现代GPU上wavefront specialization实现方式：
1. **CUDA/CUTLASS**：通过thread block内thread ID判断角色——if (threadIdx.x == 0)执行TMA load，其余线程执行MMA compute。CUTLASS 3.x使用warp group specialization（4 warps per warp group, producer vs consumer groups）
2. **Triton语言**：Tawa项目（arXiv 2510.14719）在Triton IR层面引入asynchronous references (aref)自动将程序分区为producer/consumer warps
3. **AMD GPU (ROCm/HIP)**：使用wavefront ID区分角色（__builtin_amdgcn_workitem_id_x() / warpSize）
4. **关键硬件前提**：需async memory transfer（ATT/TMA/cp.async）+ async barrier（mbarrier或等价机制）+ per-warp register allocation（如setmaxnreg控制register file占用）

涉及论文标题：
- QuCo: Efficient and Flexible Hardware-Driven Automatic Configuration of Tile Transfers in GPUs

## Asynchronous Transaction Barriers（异步事务屏障）

术语是什么？通过联网搜索让回答具体和精准。

Asynchronous Transaction Barriers是配合ATT异步tile transfer使用的两阶段同步原语，协调producer wavefront（执行memory transfer）和consumer wavefronts（执行computation）之间的数据依赖。与传统的__syncthreads()（所有线程同步阻塞）不同，async barrier拆分同步为两个独立阶段：(1) arrive：producer执行non-blocking arrive→通知数据已就绪→自身可继续执行其他工作不stall；(2) wait：consumers需要数据时执行blocking wait→barrier在arrive count到达expected count时硬件自动trip→consumers继续。这种两阶段设计让early threads利用idle cycles做额外工作，避免busy-wait和pipeline bubble。在QuCo论文中，async barrier是Operand Queue同步的核心机制：每queue需1-2个barrier（一个producer internal sync确认ATT写入完成，一个producer-consumer sync通知数据可消费）。NVIDIA H100的硬件实现称为mbarrier（managed barrier），SM硬件accelerated，shared-memory-based。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// ===== Barrier Lifecycle =====
// Uninit → init(arrive_count, tx_bytes) → Ready → 
//   arrive()/wait() cycles → inval() → Invalidated

// ===== QuCo 中 async barrier 使用 =====
// QuCo 为每个 queue 分配 barrier index 并写入 LDS metadata

// Producer-ATT Sync (Wait_For_Push):
barrier.arrive_expect_tx(byte_count = tile_size * element_size)
// ATT engine 后台传输 tile → LDS, 完成后更新 barrier
barrier.wait()   // 等待 ATT engine 完成所有字节传输

// Producer-Consumer Sync:
// Producer:
async_barrier.arrive()  // non-blocking: "LDS中数据已可用"

// Consumers:
async_barrier.wait()    // blocking: 等待 producer 通知
tile = queue.Peek(idx)  // 安全读取 LDS 数据
compute(tile)
queue.Pop(idx)          // 释放 LDS slot, 可能触发下一 arrive
```

传统barrier vs async barrier的关键差异：
- 传统__syncthreads()：所有线程必须到达→全部阻塞→全部释放；无法区分producer/consumer角色
- Async barrier：arrive和wait解耦→producer arrive后立即继续（如发起下一tile load）→consumer在wait时只阻塞consumer线程；支持byte-level transaction tracking（TX count）确保持久化数据完整

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式（以NVIDIA H100 mbarrier为例）：
1. **硬件资源**：每SM ~32个mbarrier hardware resources，shared memory中分配mbarrier object（需alignment）
2. **初始化**：mbarrier_init(&barrier, arrive_count) + fence_proxy_async_shared_cta()确保TMA engine可见barrier
3. **Arrive with TX**：mbarrier_arrive_expect_tx(&barrier, byte_count)——TMA copy前调用，告知barrier期望传输字节数
4. **Wait**：mbarrier_try_wait(&barrier)或mbarrier_wait(&barrier)——阻塞直到expected_arrivals和expected_bytes均满足
5. **Invalidate**：mbarrier_inval(&barrier)——释放barrier硬件resource以便复用
6. **AMD等价物**：尚无硬件mbarrier；QuCo在MGPUSim中建模功能等价behavior

在FlashFuser中，mbarrier被用于实现dsm_comm原语的many-to-many synchronization——论文明确指出这不同于CUTLASS默认的all-to-one cluster-sync。FlashFuser的dsm_shuffle（ring communication）和dsm_reduce_scatter需要仅同步参与特定操作的CTA子集（而非cluster内所有CTA），mbarrier的many-to-many机制使其能精确同步shuffle group/reduce group内的CTA。prologue阶段通过extended semaphore initialization准备DSM barrier资源；mainloop中DSM mul/shuffle操作前后使用mbarrier arrive/wait协调producer-consumer CTA间的数据依赖；epilogue的DSM reduce操作前同步所有参与reduce的CTA。

涉及论文标题：
- QuCo: Efficient and Flexible Hardware-Driven Automatic Configuration of Tile Transfers in GPUs
- FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection

## Operand Queue（操作数队列）

术语是什么？通过联网搜索让回答具体和精准。

Operand Queue是QuCo论文提出的高层ATT（Asynchronous Tile Transfer）编程抽象，封装ATT descriptors管理、LDS buffer分配和producer-consumer同步，将底层异步数据传输暴露为对程序员透明的队列接口。每个Operand Queue由以下参数组成（由QuCo自动配置）：tile size（每slot元素数）、number of slots（queue深度，决定pipeline overlap程度）、queue type（streaming——数据单次消费后释放 vs stationary——数据持久化供多次消费）、LDS base address和barrier indices。灵感来自NVIDIA cuda::pipeline API（单/多阶段pipeline包装TMA操作）、CUTLASS3+CuTe的TMA pipelines和ThunderKittens的asynchronous I/O抽象。核心差异：这些高级框架仍要求程序员手动选择tile sizes, stages, descriptor参数，而Operand Queue的所有参数由QuCo hardware在kernel launch时自动计算。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// ===== Queue 配置（QuCo 自动完成） =====
// Host: driver.RegisterQueue(K_dim=2048, elem=4B, TYPE_STREAMING)
//  QuCo → tile_size=512, slots=2, LDS[offset:offset+4096B]

// ===== 2-Slot Streaming Queue 时间轴 =====
// t0: Producer Push(slot_0) → ATT engine load tile_0
// t1: [ATT transfer ████████]  Consumer idle (waiting)
// t2: [ATT transfer ████████]  Wait_For_Push → arrive()
// t3: Producer Push(slot_1) → ATT load tile_1, 
//     Consumer wait() → Peek(slot_0) → compute
// t4: [ATT transfer ████████]  Consumer compute(slot_0)
// t5: [ATT transfer ████████]  Consumer Pop(slot_0)
//     Producer Wait_For_Push → arrive()
// t6: Producer Push(slot_0) → ATT load tile_2,
//     Consumer wait() → Peek(slot_1) → compute
// ...pipeline continues with 2-slot double buffering

// ===== Queue API =====
queue.Push(tile_idx)        // Producer: 发起 ATT async load → slot[tile_idx % slots]
queue.Wait_For_Push()       // Producer: 等待 ATT engine 完成写入 LDS
data = queue.Peek(idx)      // Consumer: 只读访问 LDS 中 tile（不释放）
queue.Pop(idx)              // Consumer: 标记 slot 为 free
```

Streaming vs Stationary queues：
- **Streaming**：tile单次消费后释放，适合无数据复用的kernel（Elementwise, Dot-Product）。QuCo用Little's Law（slots = memory_transfer_time / compute_time）计算slot数
- **Stationary**：tile在LDS中持久化供多次访问，适合有数据复用的kernel（Matrix-Matrix weight tile）。slot数由剩余LDS容量均分

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

当前实现方式（类比框架）：
1. **NVIDIA cuda::pipeline**：cuda::pipeline_shared_state + memcpy_async + pipeline_commit_wait，支持单/多阶段pipeline。但stages数和tile size需程序员手动选择
2. **CUTLASS 3.x pipelines**：MainloopPipeline/TileSchedulerPipeline等，内部管理TMA descriptor pass、shared memory tile buffer allocation、warp group barrier synchronization
3. **ThunderKittens async I/O**：提供高层次的asynchronous copy抽象，封装底层TMA/cp.async路径
4. **QuCo Operand Queue**：仍为学术proposal，核心创新在于queue参数（tile size, slots, LDS layout）由硬件自动计算，而非程序员指定

涉及论文标题：
- QuCo: Efficient and Flexible Hardware-Driven Automatic Configuration of Tile Transfers in GPUs

## Expert Parallelism (EP / 专家并行)

术语是什么？
Expert Parallelism (EP) 是MoE模型推理中的分布式并行策略：将不同expert的完整权重分配到不同GPU，而非像Tensor Parallelism (TP)将每个expert权重切分到多个GPU。当gate network选择某expert时，token hidden state被发送到持有该expert的GPU执行计算，然后结果汇总。EP的核心优势是减少TP中每GPU都参与每token计算的通信开销（只有被选中expert所在的GPU接收token），但expert负载不均会导致GPU闲置。FineMoE在6×RTX 3090上使用EP，通过hash map和round-robin将expert分布到GPU，结合expert offloading进一步管理每GPU显存。

从kernel调度角度拆解术语：
```
# Expert Parallelism执行流程（Mixtral-8x7B, 2GPUs, 每GPU 4 experts）：
# GPU0: Experts E0-E3, GPU1: Experts E4-E7

# 对某token在第l层MoE block：
h = hidden_state[l]              # GPU0上的hidden state
gate_logits = h @ W_g            # Gate Network计算（每GPU复制W_g）
probs = Softmax(gate_logits)     # [E0:0.6, ..., E7:0.01]
selected = TopK(probs, 2)        # 假设选到E0(GPU0)和E5(GPU1)

# GPU0计算本地expert：
output_gpu0 = probs[E0] * FFN_E0(h)  # 本地计算，无跨GPU通信

# GPU1计算远程expert：
send(h, GPU0→GPU1)                   # NVLink传输hidden state
output_gpu1 = probs[E5] * FFN_E5(h)  # 远端计算
send(output_gpu1, GPU1→GPU0)         # 传回结果

output = output_gpu0 + output_gpu1   # 汇总
```
EP与Expert Offloading结合时（FineMoE）：GPU cache管理仅限于本地expert，每GPU独立执行prefetch/cache/evict决策。On-demand loading也按GPU本地expert执行，跨GPU的expert miss通过NVLink all-to-all通信处理。GPU侧task pool + 异步线程调度expert prefetching和on-demand loading。

术语一般如何实现？
TensorRT-LLM支持TP+EP混合并行（`moe_tp_size` + `moe_ep_size`）。Megatron-Core推荐Mixtral-8x7B在64GPUs上TP=1, EP=8, PP=4。FineMoE用MoE-Infinity的expert parallelism实现：hash map做expert→GPU映射，round-robin分配确保GPU间expert数均匀。跨GPU传输通过NVLink（RTX 3090 pairwise NVLinks），CPU→GPU传输通过PCIe 4.0（32GB/s）。EP主要挑战是load imbalance和通信开销。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

## Stacked Co-location (堆叠共置)

术语是什么？

Stacked Co-location（堆叠共置）是指NVIDIA GPU硬件调度器的默认行为：当一个CUDA kernel被launch后，其多个blocks被顺序调度到SM cores，同一kernel的blocks因具有相同的资源需求而被"堆叠"在相同的SM内。由于kernel的blocksize通常远小于SM的总thread容量（如A40: 1536 threads, 默认blocksize 512-1024），一个SM可以同时容纳多个同kernel blocks，但这些blocks的hardware resource需求完全相同，导致SM内仅一种硬件资源被充分利用，其余资源闲置。"Stacked"一词强调同kernel blocks在SM内的垂直堆叠关系。在μShare的实验中，vectorized kernel (dominant LDST, blocksize 1024) 和 roll kernel (dominant INT32, blocksize 512) 虽在不同CUDA stream上并发launch，但vectorized kernel的所有blocks先占满所有SM并执行完毕，roll kernel的blocks才被调度，无法在SM内实现不同kernel的并行。

从kernel调度角度拆解术语：

Stacked co-location的执行过程（以PyTorch launch two kernels on NVIDIA A40为例）：

```
// Stage 1: Launch — PyTorch launch两个kernel到不同CUDA stream
cudaLaunchKernel(vectorized_kernel, gridDim, 1024, ..., stream1);
cudaLaunchKernel(roll_kernel, gridDim, 512, ..., stream2);

// Stage 2: Dispatch — GPU hardware dispatch unit按顺序调度block
// vectorized_kernel: 每个block 1024 threads
//   SM0: block_0(1024 threads) — 剩余512 threads
//   SM0: block_1(1024 threads) — 无法，因为 512 < 1024
//   但可以：SM0: block_0, SM1: block_1, ...
//   当线程数超过总thread容量时：SM0: block_0 + block_84, SM1: block_1 + block_85, ...
//   → 每个SM内都放着同kernel的两个block

// roll_kernel的blocks全部排队等待，直到vectorized_kernel blocks接近完成
// 原因：当kernel总线程数 > GPU总thread容量(129024)，需等待前序blocks释放SM资源

// Stage 3: Execution — SM内
// vectorized_kernel blocks在SM0内：
//   LDST: 58.02% active (layer normalization的主要操作)
//   FP32: 13.43%, INT32: ≈0%, FP64: 0%, SFU: 11.03%, Tensor: 0%
//   → "1 more, 5 less" 模式
```

μShare的实验数据显示：在max batch下，61.85%的kernel执行时线程数超过GPU总容量（129024 threads），这些kernel占70.83%的总执行时间，因此stacked co-location是主要瓶颈。6802次kernel执行的统计显示NVIDIA-SMI报告81.16%利用率，但Nsight Compute仅报告9.28% low-level hardware利用率。

术语一般如何实现？如何使用？

Stacked co-location是NVIDIA GPU硬件调度器的固有行为，由闭源的hardware dispatch unit实现。GPU的left-over scheduling策略：只要SM的剩余thread capacity ≥ blocksize，block就可以被调度到该SM。由于同一kernel的所有block有相同blocksize和资源需求，硬件scheduler自然倾向将它们填入相同的SM集合。现有系统（INFless、Orion、MPS、MIG）在inter-SM层面做spatial/temporal sharing，但无法解决intra-SM的stacked co-location问题。避免stacked co-location的方法包括：kernel fusion（Tacker/T3/Rammer/COMBO，将互补kernel合并为一个）和persistent kernel（ISPA/Plasticine/Elastic kernel，空核驻留），但这些都需要侵入性修改。μShare选择非侵入路径：通过修改kernel launch参数（blocksize）间接影响硬件scheduler的decision，实现从stacked到scattered co-location的转变。

涉及论文标题：
- μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs

## Scattered Co-location (散布共置)

术语是什么？

Scattered Co-location（散布共置）是μShare提出的替代stacked co-location的intra-SM kernel调度策略：通过将部分kernel的blocksize设置为略超半SM thread容量的值（half-plus blocksize），迫使同kernel的blocks散布到不同SM（因为同一SM无法容纳两个超过半容量的block），剩余threads可被其他kernel的小block占用，实现不同kernel的blocks在同一SM内的交叉分布（"scattered"）。与stacked co-location中同kernel blocks独占SM不同，scattered co-location使SM内同时运行不同kernel的blocks，各block使用不同的dominant hardware resource，互补利用SM内的多种hardware unit。

从kernel调度角度拆解术语：

Scattered co-location的调度流程（以μShare on A40, 1536 threads/SM, half-plus blocksize=800为例）：

```
// Kernel set O被分为X（half-plus shaped）和Y（time-shifted launch）
// X: vectorized_kernel (LDST dominant), blocksize=800 (half-plus)
// Y: roll_kernel (INT32 dominant), blocksize=512 (default)

// Dispatch决策：
// SM0空闲，剩余thread = 1536
//   T1: dispatch unit取vectorized_kernel block_0 (800 threads)
//       → 剩余thread = 1536 - 800 = 736
//   T2: dispatch unit取vectorized_kernel block_1 (800 threads)
//       → 800 > 736 → 无法放入SM0 → 分配到SM1
//   T3: dispatch unit取roll_kernel block_0 (512 threads)
//       → 512 ≤ 736 → 可以放入SM0
//   → SM0: vectorized block_0 (800 threads, LDST 58%) + roll block_0 (512 threads, INT32 33%)
//     共800+512=1312 threads ≤ 1536
//   → SM0内同时执行LDST-heavy和INT32-heavy计算
//   → 6种HW utilization: LDST ~58% + INT32 ~33% + SFU ~11%(vec) + ~25%(roll) + ...
//     相比stacked时只用LDST 58%或INT32 33%单独一种
//     avg low-level HW utilization: 15.10% (scattered) vs 10.90% (stacked INFless)
```

关键约束：
- Half-plus blocksize = thalf + α, 其中 thalf = SM_thread_capacity / 2 = 768 (A40), α最小为warp size (32)，所以最小800
- 对A800/A100/H200 (2048 threads/SM)，改用1/3-plus: blocksize = 2048/3 + α ≈ 704，允许同kernel两个1/3-plus block入1SM (704×2=1408, 剩余640)
- α动态调整: slack positive → α=32 (最小); slack negative → α逐步+32加速kernel

术语一般如何实现？如何使用？

Scattered co-location的实现需要：
1. **Kernel拦截**：通过LD_PRELOAD劫持CUDA kernel launch函数（cudaLaunchKernel/cublasSgemm），读取/修改blocksize参数
2. **Blocksize塑形**：根据SM thread capacity计算half-plus值 → 写入modified blocksize → 调用原始launch函数
3. **时移启动**：对unmodifiable kernel（cuDNN/cuBLAS/tiling kernel），检查其6种hardware resource utilization + shared memory/registers与当前SM中active kernel的combined utilization是否≤100% → 满足则立即launch → 不满足则delay β=10μs后重检
4. **资源互补判断**：基于offline profiled per-kernel 9-tuple {rFP32, rFP64, rINT32, rLDST, rSFU, rTensor, rmem, rreg, tLaunch}选择co-location pairing。实验显示：dominant resource不同的kernel配对时half-plus提升throughput 19.94%，相同>10.37%下降

涉及论文标题：
- μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs

## Half-Plus Blocksize Shaping (半超blocksize塑形)

术语是什么？

Half-Plus Blocksize Shaping（半超blocksize塑形）是μShare的核心技术，通过将CUDA kernel的blocksize参数设置为略大于SM总thread容量一半的值，间接操控NVIDIA GPU硬件调度器的block placement决策，实现intra-SM scattered co-location。原理：当blocksize > SM_thread_capacity/2时，同一SM无法容纳两个该kernel的block（2×blocksize > SM_thread_capacity），迫使硬件scheduler将同kernel blocks散布到不同SM，剩余threads可分配给其他kernel的小block。"half-plus"中的"half"指SM thread capacity的一半（如A40: 1536/2=768），"plus"指在此基础上加一个小偏移α（最小为warp size=32，避免thread fragmentation），最终blocksize=800(A40)或704(A800)。

从kernel调度角度拆解术语：

Half-plus blocksize shaping的数学定义：

```
// GPU: A40 (1536 threads/SM, CUDA max blocksize=1024)
// half-plus blocksize b 的定义域：
//   {b | 768 < b ≤ 1024, b ≡ 0 (mod 32)}
//   最小: 768 + 32 = 800
//   可选: 800, 832, 864, 896, 928, 960, 992, 1024

// 选择策略：
//   slack s_k > 0: b = 800 (最小half-plus，减少thread waste)
//   slack s_k < 0: b = previous + 32（逐步增加加速kernel执行）

// GPU: A800/A100/H200 (2048 threads/SM)
// 1/3-plus blocksize b 的定义域：
//   {b | 683 < b ≤ 1024, b ≡ 0 (mod 32)}
//   最小: 704

// 调度的间接控制原理：
//   Condition for block scheduling to SM:
//     SM.available_threads >= b
//   
//   Half-plus guarantee:
//     2 × b_min > SM_thread_capacity
//     2 × 800 = 1600 > 1536  ✓  同kernel两个block不能在同一SM
//     剩余threads = 1536 - 800 = 736 < 800 → 不能放第三个half-plus block
//     但 736 ≥ 512 (default roll block) → 可以放小block
```

μShare的消融实验证明：
- half-plus shaping贡献最大：μShare w/o shape（无blocksize调整）→ throughput下降30.95%，SLO violation增加6.33%
- fixed blocksize (1024) vs dynamic half-plus：固定1024时throughput下降3.36%，因为无法根据slack调整α控制加速力度
- 动态调整的必要性源自static preset blocksize在co-location下不再optimal：roll kernel exclusive执行时最优512，co-locate with vectorized kernel时最优变为1024（1.98× improvement）

术语一般如何实现？如何使用？

实现步骤：
1. Offline profiling：确定SM thread capacity → 计算half-plus/1/3-plus阈值
2. Kernel interceptor通过mmap共享内存读取kernel launch slack（s_k = tLaunch - tIntercept）
3. Block shaper根据slack sign和kernel类型决定α：
   - GPUs with 1536 threads/SM (A40, RTX 4090, RTX 3080 Ti): b_min = 800, 最大1024
   - GPUs with 2048 threads/SM (A100, A800, H200): b_min = 704, 1/3-plus替代half-plus（因为max blocksize=1024 < 2048/2, 两个1024 block可stacked co-locate）
4. 将修改后的blocksize写入共享内存 → dlsym获取的原始cudaLaunchKernel被调用时使用新blocksize
5. α值遵循warp对齐原则（32倍数），避免thread资源碎片

涉及论文标题：
- μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs

## Time-Shifted Launching (时移启动)

术语是什么？

Time-Shifted Launching（时移启动）是μShare针对blocksize不可修改的CUDA kernel（如cuDNN/cuBLAS wrapper函数和tiling kernel如Conv2d）设计的互补调度技术。由于这些kernel的blocksize隐藏在闭源代码中或修改后产生CUDA internal error（如Conv2d tiling匹配破坏），无法通过half-plus shaping进行scattered co-location。Time-shifted launching通过控制这些kernel的启动时机（relaunch time），延迟其launch直到SM上正在执行的half-plus kernel与待启动kernel的资源需求互补（combined hardware utilization ≤ 100%且shared memory/registers足够），此时才释放kernel进行co-location。如果等待导致kernel launch slack变为negative（接近SLO violation），则将其升级为half-plus shaping。

从kernel调度角度拆解术语：

Time-shifted launching的调度伪代码：

```
// 输入：kernel set O = X ∪ Y, X=half-plus shaped, Y=time-shifted
// 对每个kernel kj ∈ Y：
function time_shifted_launch(kj):
    while kj not launched:
        // 1. 检查资源互补条件
        can_colocate = true
        for each hw_resource in {FP32, FP64, INT32, LDST, SFU, Tensor}:
            if current_SM_active_util[hw_resource] + kj.util[hw_resource] > 100%:
                can_colocate = false
                break
        if kj.smem + current_SM_active_smem > SM_total_smem:
            can_colocate = false
        if kj.reg + current_SM_active_reg > SM_total_reg:
            can_colocate = false
        
        // 2. 满足条件 → 立即launch (用default blocksize，不修改)
        if can_colocate:
            cudaLaunchKernel(kj.func, kj.gridDim, kj.default_blocksize, ...)
            return
        
        // 3. 不满足 → delay β=10μs 后重检
        usleep(β)
        
        // 4. 更新slack = tLaunch - current_time
        kj.slack = kj.tLaunch - now()
        
        // 5. 重新排序整个kernel set O
        O_sorted = sort_ascending_by_slack(O)
        
        // 6. 若kj进入top-x（|X| = min x s.t. Σ^{x}_{i=1} blocks_i.count > num_SMs）
        if rank(kj, O_sorted) <= x:
            // 升级为half-plus shaping
            kj.blocksize = half_plus_sm_threads()
            cudaLaunchKernel(kj.func, kj.gridDim, kj.blocksize, ...)
            return
```

术语一般如何实现？如何使用？

实现要点：
1. **资源互补判断**：依赖offline profiled per-kernel 9-tuple resource profile，profiler使用NVIDIA Nsight Compute记录6种low-level hardware utilization + Nsight Systems记录launch timing
2. **延迟参数β**：论文通过多次实验确定最优β=10μs。β太小导致频繁重检overhead，β太大浪费co-location窗口
3. **Slack动态管理**：随着等待时间增长，kj的slack递减，可能进入sorted set的top-x位置 → 升级为half-plus。这形成一种优雅的退化路径：worst-case（所有kernel都unmodifiable）→ 所有kernel by time-shifted launching拉满 → 等效于resource-coupled co-location（INFless级别性能）
4. **适用性**：μShare的10模型分析中unmodifiable kernel占48.37%（3290/6802次执行），time-shifted launching是必要但非主要的补充机制。实验显示unmodifiable kernel比例从100%降至48.37%时throughput从47.59单调提升至58.81

涉及论文标题：
- μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs

## Left-over Scheduling Strategy (剩余调度策略)

术语是什么？

Left-over Scheduling（剩余调度策略）是NVIDIA GPU hardware dispatch unit采用的block-to-SM调度策略：当一个新的block需要被调度时，dispatch unit检查所有SM的剩余资源（主要是available threads），选择第一个满足资源需求的SM将block分配进去。这种策略简单、硬件实现成本低，但它是导致stacked co-location的直接原因——因为同kernel的所有block需求相同，它们会被连续分配到相同的SM集合。"Left-over"一词指调度决策仅基于SM的剩余容量（left-over capacity），不考虑block之间的资源互补性。μShare正是利用了这一策略的可预测行为：通过设置half-plus blocksize使SM的"left-over capacity"对同kernel的第二个block不满足条件，从而"欺骗"硬件scheduler将blocks散布。

从kernel调度角度拆解术语：

Left-over scheduling的简化逻辑：

```
// GPU dispatch unit的block分配逻辑 (simplified from observation)
function dispatch_block(block, blocksize):
    for each SM in GPU_SMs:
        if SM.available_threads >= blocksize
            AND SM.available_shared_memory >= block.smem
            AND SM.available_registers >= block.regs:
            assign block to SM
            SM.available_threads -= blocksize
            return SM.id
    // 如果没有SM有足够资源 → block进入pending queue
    // 当某个SM的active block完成释放资源后 → retry
    return PENDING

// 单kernel场景 (blocksize=512, A40 SM=1536 threads):
//   SM0: 512 → 1024 left → block_1: 512 → 512 left → block_2: 512 → 0 left
//   SM1: 512 → 1024 left → ...
//   → stacked co-location (同kernel的3个block在SM0内)

// Half-plus场景 (blocksize=800):
//   SM0: 800 → 736 left → block_1: 800 > 736 → skip SM0 → SM1
//   SM1: 800 → 736 left → block_2: 800 > 736 → skip SM0,SM1 → SM2
//   → scattered: 每个SM仅1个half-plus block
//   → 736 left in SM0 可放 roll block (512) → co-location
```

术语一般如何实现？如何使用？

Left-over scheduling是NVIDIA GPU的硬件实现，闭源不可修改。μShare的贡献在于反向利用这一策略：
1. **正向利用**：设置half-plus blocksize → 使同kernel blocks不能满足left-over条件 → forced scattering
2. **Timer-shifted launching补充**：对unmodifiable kernel，通过延迟其launch timing使SM的left-over capacity刚好被互补kernel block利用
3. **局限性**：left-over scheduling缺乏全局优化（不考虑block间资源互补、不进行负载均衡），μShare只能在launch stage间接影响，无法在dispatch stage直接干预

涉及论文标题：
- μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs

## dsm_comm Primitive（DSM通信原语）

术语是什么？通过联网搜索让回答具体和精准。

dsm_comm primitive 是 FlashFuser 论文提出的高层 DSM 通信抽象，用于在 fused GEMM kernel 中描述 cluster 内 SM 之间的数据交换模式。它将 H100 Thread Block Cluster 的 SM 划分、数据流方向和通信模式（reduce/shuffle/multiply）统一编码为可组合的原语。整个原语体系包括四种基本操作：(1) dsm_all_exchange——cluster 内沿 K 维 All-Reduce 聚合 partial sum 以产生完整中间 tile；(2) dsm_shuffle——Shuffle Group 内 ring communication 交换中间 tensor 切片；(3) dsm_reduce_scatter——cluster 内 scatter-reduce 聚合 partial output；(4) inter_cluster_reduce——基于 TMA cp.reduce.async.bulk 的跨 cluster 原子归约。对 Gated FFN，dsm_all_exchange 从 Add 变为 Mul 操作。primitive 的核心参数由 cluster size 四维参数 (clsm, clsn, clsk, clsl) 决定，派生两个关键变量：clsshuffle = clsl / clsk（参与 shuffle 的 block 数）和 clsreduce = clsn / clsshuffle（参与 reduce 的 shuffle group 数）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

以标准 FFN 两段 GEMM (A×B=C, C×D=E) 的 cluster size (2,4,2,4) 为例：

```
// ===== Cluster 配置 =====
clsm=2, clsn=4, clsk=2, clsl=4
clsshuffle = clsl/clsk = 4/2 = 2  (每组 2 个 block 参与 shuffle)
clsreduce = clsn/clsshuffle = 4/2 = 2  (2 个 shuffle group 参与 reduce)

// ===== GEMM0 Phase (A[M×K] × B[K×N] → C[M×N]) =====
// clsk=2 → K 维 spatial partition 到 2 个平行 block
// Block(0,0): C_0,0(0) = Σ(A_0,i × B_i,0) for i=0..K/2
// Block(0,1): C_0,0(1) = Σ(A_0,i × B_i,0) for i=K/2..K

// ===== dsm_all_exchange (沿 K 维 All-Reduce) =====
// 同一 cluster 内参与 K 维 partition 的 block 交换 partial sum
dsm_all_exchange(group=[Block(0,0), Block(0,1)], op=Add)
  → C_0,0 = C_0,0(0) + C_0,0(1)
// C_0,0 留在 DSM 中，不写 global memory

// ===== GEMM1 Phase (C[M×N] × D[N×L] → E[M×L]) =====
// dsm_shuffle: Shuffle Group 内交换 C 切片
// Shuffle Group 0: {Block(0,0), Block(0,1)} — 共享 C row 0
// Shuffle Group 1: {Block(1,0), Block(1,1)}
dsm_shuffle(group=ShuffleGroup_0, pattern=ring_communication)
  // 每个 Block 需要完整 C row 才能与 D 的不同 tile 相乘
  Block(0,0): C_0,0,C_0,2,... → 用于计算 E_0,0
  Block(0,1): C_0,0,C_0,1,... → 用于计算 E_0,1

// ===== GEMM1 计算 =====
Block(0,0): E_0,0(0) = C_0,0 × D_0,0
Block(0,1): E_0,1(0) = C_0,0 × D_0,1

// ===== Store Phase =====
// dsm_reduce_scatter: cluster 内 scatter-reduce
dsm_reduce_scatter(group=ReduceGroup, op=Add)
  → Block(0,0): 负责写回 E_0,0 = E_0,0(0) + E_0,0(1)
  → Block(0,1): 负责写回 E_0,1 = E_0,1(0) + E_0,1(1)

// inter_cluster_reduce: 跨 cluster 原子归约
if (多 cluster 贡献同一输出 E tile):
  inter_cluster_reduce(E_tile, op=Add)
  // 通过 TMA cp.reduce.async.bulk 异步原子 reduce
```

Gated FFN 变体：两个 Up-FFN GEMM 分支并行执行→dsm_all_exchange 从 Add 变为 Mul→将 SiLU 分支和另一 GEMM 分支的结果 element-wise 乘。可空间划分(clsk=2, 两分支到不同 block group)最大化并行或顺序执行最小化 DSM 通信。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

dsm_comm primitive 的实现基于 NVIDIA H100 的以下硬件/软件机制：
1. **数据移动**：通过 TMA（Tensor Memory Accelerator）实现 SM-to-SM 的细粒度数据交换。TMA 的 cluster 内地址空间支持 `shared::cluster` 修饰符直接访问其他 SM 的 shared memory
2. **同步**：使用 mbarrier many-to-many synchronization 而非 CUTLASS 默认的 all-to-one cluster-sync。mbarrier 可只同步参与特定 shuffle/reduce 的 CTA 子集
3. **代码生成**：FlashFuser 后端在 CUTLASS kernel 结构的三个位置插入 dsm_comm 操作——prologue 初始化 DSM semaphore/barrier；mainloop 插入 DSM mul/shuffle（GEMM accumulation 完成后）；epilogue 执行 DSM reduce + global memory store
4. **Ring Communication**：SHUFFLE 使用 ring communication 模式——各 CTA 发送本 CTA 的 C tile 切片给下一个 CTA，同时从上一个 CTA 接收需要的切片
5. **配置灵活性**：cluster size (clsm, clsn, clsk, clsl) 可配置以适应不同 problem size（尤其是小尺寸或不可整除 case）

涉及论文标题：
- FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection

## Row Dataflow / OP Dataflow for Attention

术语是什么？
Row Dataflow 和 OP (Output-stationary) Dataflow 是 VAR-Turbo 论文中 Unified Attention Core 支持的两种可重构 attention 执行数据流。Row Dataflow 将 PE Cluster 按 attention head 分配，每个 Cluster 独立处理一个 head 的完整 K×V 矩阵乘加，适合大矩阵的 Big Attention（全局建模）。OP Dataflow 将 PE Cell/Node 切换为 output-stationary 模式，每个 PE 固定负责特定的输出位置，输入数据流经 PE array，适合小矩阵的 Small Attention（local window 内 token 聚合）。两种 dataflow 通过 Snooper+Fat Tree 动态切换。

从kernel调度角度拆解术语：
Row Dataflow 执行 Big Attention 的计算过程：
```
// PE Cluster 按 attention head 分配
for each head h:
  Q_h, K_h, V_h = get_qkv(h)  // 分配到 PE Cluster[h]
  Scores = Q_h × K_h^T        // PE Cell 间按 row 并行
  Attn = softmax(Scores)       // Row-wise normalization
  Out_h = Attn × V_h           // PE Cell 间按 row 并行
```

OP Dataflow 执行 Small Attention 的计算过程：
```
// PE Cell/Node 固定输出位置，输入数据流动
for each local window w:
  Q_w, K_w, V_w = get_qkv(w)  // 小矩阵（如 window_size=2, d=128）
  // K_w^T 广播到所有 PE Cell → Q_w 沿 PE array 流动
  Scores = Q_w × K_w^T         // output-stationary: 每个 PE 固定输出元素
  Coeff = mean(softmax(Scores), axis=0)  // 平均 attention coefficient
  Rep = Coeff × V_w            // 聚合为 representative token
```

Dataflow 切换机制：
- Snooper 读取当前层类型（Learning Region vs Inert Region）→配置 PE Cell 的 packet ID 映射
- Fat Tree 根据 packet ID 路由数据到对应 PE Cell→实现单 cycle 内 Row ↔ OP dataflow 切换
- Row Dataflow 优势：高吞吐（大数据矩阵），适合 Big Attention
- OP Dataflow 优势：低延迟（小数据矩阵）、减少中间数据移动，适合 Small Attention

术语一般如何实现？如何使用？
作为 VAR-Turbo accelerator 的 Unified Attention Core 的一部分以 SystemVerilog RTL 实现，TSMC 28nm 综合。Row+OP MAC 通过 Divide-and-Conquer 技术将大矩阵乘法分解为子块，Fluid Zone Detection 动态调整 FP 累加精度边界，shared FP accumulator 降低面积和功耗。Row/OP dataflow 的设计思路也可应用于其他需同时处理异构 attention pattern 的 accelerator。

涉及论文标题：
- VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

## Four-Stage Radix Select TopK

术语是什么？
Four-Stage Radix Select TopK 是 VAR-Turbo 论文中 Radix Sort Core 采用的大 K TopK 选择 dataflow。将 TopK 问题从通用排序方案（如 Bitonic Sort + Merge Sort，在大 K 上需 O(N log² N) 比较和全局数据重排）转化为固定 4 阶段流水线，利用 radix 分解（按 digit 分桶）避免全局比较和重排。特别适用于 VAR 场景中大 K（如 N=4096 时 K=1936）confidence/importance selection。

从kernel调度角度拆解术语：
Four-Stage Radix Select 的计算过程（以 N=4096, K=1936, 8-bit radix 为例）：
```
// Stage 1: CountBin
bin_counts[256] = {0}
for each element i in [0, N):
  digit = (confidence[i] >> (radix_shift * 8)) & 0xFF
  bin_counts[digit]++

// Stage 2: PrefixSum
prefix_sum[0] = 0
for b in [1, 256):
  prefix_sum[b] = prefix_sum[b-1] + bin_counts[b-1]

// Stage 3: SelectBin
// 从最高 radix digit 开始搜索
for b in [255, 0]:
  if prefix_sum[b] <= K < prefix_sum[b] + bin_counts[b]:
    candidate_bin = b
    K_offset = K - prefix_sum[b]
    break

// Stage 4: Filter
candidates = []
for each element i in [0, N):
  digit = extract_radix_digit(confidence[i])
  if digit == candidate_bin:
    candidates.append((i, confidence[i]))
sort(candidates by confidence descending)
selected_indices = candidates[0:K_offset]
```
多轮 radix（从最高 digit 到最低 digit）迭代后精确选出 TopK。Locality-aware Scheduling 扩展：维护 history table（mask map）标记已解码区域，PE 分组在不同空间区域并行执行各自的 Radix Select，优先处理靠近已解码 token 的高置信行/block。

术语一般如何实现？如何使用？
在 VAR-Turbo accelerator 的 Radix Sort Core 中以 SystemVerilog RTL 实现（TSMC 28nm）。关键优势：相比 Bitonic Sort（O(N log² N) 比较器和 log² N 级流水线）和 Merge Sort（需全局 data shuffle），Radix Select 的 4 阶段流水线是固定深度的（与 K 和 N 无关），仅需 bin counter、prefix sum adder、comparator 和 filter mux。Locality-aware Scheduling 利用 2D 图像 token 的空间局部性进一步提升吞吐。该 dataflow 可推广至其他需在线大 K TopK 的场景（如 attention sparsification、KV cache pruning、MoE gating）。

涉及论文标题：
- VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

## CUDA Graph for Draft Model Decoding（Draft模型解码的CUDA Graph优化）

术语是什么？通过联网搜索让回答具体和精准。
CUDA Graph 是 NVIDIA CUDA 提供的机制，允许将一系列 kernel launch 和 memory copy 操作预录制为有向无环图（DAG），后续通过单次 graph launch 回放整个图，消除逐个 kernel launch 的 CPU-GPU 同步和 driver overhead。AdaServe 将 CUDA Graph 用于 draft model 的 speculative decoding steps：从第二个 speculation step 到第 d 步，若活跃请求数相同，则每步计算形状一致，可复用预捕获的 CUDA graph。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
AdaServe 中 CUDA Graph 的使用流程：
```
// 预捕获阶段（系统启动时或活跃请求数变化时）
for num_requests in possible_range:
    graph[num_requests] = cudaGraphCreate()
    cudaStreamBeginCapture(stream)
    // 录制 d-1 步 draft decoding（step 2 到 step d）
    for step = 2 to d:
        draft_model.forward(tokens, num_requests)  // 固定 batch size
    cudaStreamEndCapture(stream, graph[num_requests])
    executable[num_requests] = cudaGraphInstantiate(graph[num_requests])

// 运行时（每轮 iteration）
// Step 1: 首次 draft forward（形状可能变化，单独 launch）
draft_model.forward_first(tokens, current_num_requests)

// Steps 2-d: 复用 CUDA Graph
if current_num_requests == prev_num_requests:
    cudaGraphLaunch(executable[current_num_requests], stream)  // 单次 launch
else:
    // 活跃请求数变化 → 重新捕获或用普通 launch
    fallback_individual_launch()
```
关键优化点：(1) 从第二步开始的 draft decoding 有 d-1 步形状相同→可录制成单图；(2) 单次 graph launch 替代 d-1 次独立 kernel launch → 显著减少 CPU-GPU sync overhead；(3) 活跃请求数变化时才重新捕获 graph。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 AdaServe 中实现于 FlexFlow Serve 的 execution engine 中。CUDA Graph 优化使 draft model 的重复 decoding steps 的 kernel launch overhead 大幅降低。实验显示 CPU selection overhead 仅占总 serving time 的 0.41%/0.31%，说明 draft decoding 的 GPU 执行 overhead 已被有效控制。

涉及论文标题：
- AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding


---

## Load-Evict Overlap with CUDA Stream Pipeline（CUDA流管线的KV Cache加载-驱逐重叠）

术语是什么？

Load-Evict Overlap是TokenFlow Hierarchical KV Cache Manager中的CUDA stream调度技术，专用于请求抢占和恢复场景下的KV cache在GPU HBM和CPU DRAM之间的传输优化。核心思想是利用CUDA stream的异步并发能力，将"被抢占请求KV cache的GPU显存释放（evict）"与"恢复请求KV cache的CPU→GPU加载（load）"在时间上重叠执行，避免两者串行执行时的I/O stall。

从kernel调度角度拆解术语：

Load-Evict Overlap的执行流程（以请求R1被抢占、R3恢复为例）：

```
// Step 1: Scheduler决定抢占R1、恢复R3
pending_evict = get_unsynced_chunks(R1)  // write-through已同步大部分，剩余少量chunk
pending_load = get_kv_chunks(R3)         // R3的KV cache在CPU DRAM中

// Step 2: 创建/复用CUDA streams
compute_stream = get_cuda_stream("compute")
load_stream = get_cuda_stream("load")  
evict_stream = get_cuda_stream("evict")

// Step 3: 记录compute stream当前进度
cudaEventRecord(compute_done, compute_stream)

// Step 4: 使load和evict stream等待compute完成
cudaStreamWaitEvent(load_stream, compute_done)
cudaStreamWaitEvent(evict_stream, compute_done)

// Step 5: 并行发射load和evict操作
// load_stream: H2D传输 R3的KV chunk
for each chunk in pending_load:
    cudaMemcpyAsync(gpu_kv[chunk.dst], cpu_kv[chunk.src], 
                    chunk.size, cudaMemcpyHostToDevice, load_stream)

// evict_stream: 释放R1未同步chunk的GPU显存
// 已同步的chunk直接free(无需等待)，未同步的chunk先D2H再free
for each chunk in pending_evict:
    if chunk.is_synced:
        free_gpu_block(chunk.gpu_addr)  // 即时释放
    else:
        cudaMemcpyAsync(cpu_kv[chunk.dst], gpu_kv[chunk.src],
                        chunk.size, cudaMemcpyDeviceToHost, evict_stream)
        // 传输完成后释放
        cudaFreeAsync(chunk.gpu_addr, evict_stream)

// Step 6: 同步barrier
cudaEventRecord(load_done, load_stream)
cudaEventRecord(evict_done, evict_stream)

// Step 7: compute stream等待load完成
cudaStreamWaitEvent(compute_stream, load_done)

// Step 8: 恢复compute——R3的decode开始
launch_decode_kernel(R3, compute_stream)
```

关键设计点：
1. **动态Chunk Sizing**：`chunk.size`不是在启动时固定。estimator根据预估的下一轮compute duration自适应选择chunk size，确保`transfer_time(chunk) ≤ compute_duration`，最大化compute-I/O overlap。
2. **Batched Transfer**：多个请求的KV chunk合并为单次cudaMemcpyAsync调用→减少DMA engine setup次数→提升PCIe带宽实际利用率。
3. **CUDA Event同步**：使用cudaEventRecord/cudaStreamWaitEvent实现细粒度stream间依赖（而非全局cudaDeviceSynchronize），避免阻塞无关stream。
4. **Write-Through预同步**：大部分KV cache已在正常decode过程中通过write-through同步到host→抢占时pending_evict通常极小（仅最近1-2次decode iteration的增量）→evict延迟远小于load延迟→load成为瓶颈→load-evict overlap的价值在于让evict不阻塞load。

术语一般如何实现？如何使用？

在TokenFlow中通过PyTorch CUDA stream API（torch.cuda.Stream）实现：系统维护三类persistent CUDA stream对象（而非每次创建/destroy），Python threading为每类stream分配独立控制线程。CUDA events在compute→load/evict→compute的依赖链上建立同步点。消融实验：去掉evict-load overlap后完成时间明显增加。TokenFlow未开源。

与普通KV cache offload方案的区别：普通方案通常在显存压力下被动触发evict（reactive），且evict/load串行；TokenFlow的load-evict overlap主动（proactive，write-through提前准备）+重叠（overlap，CUDA stream并发），将KV cache搬移从系统瓶颈转化为可与计算重叠的后台操作。

涉及论文标题：
- TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling

## Recompute-Decode Fused Kernel (K1+K2 Fusion)

术语是什么？通过联网搜索让回答具体和精准。

Recompute-Decode Fused Kernel（K1+K2融合kernel）是eLLM系统在layer-level execution中提出的CUDA kernel优化技术。它将两个原本串行执行的kernel融合为单一kernel launch：(1) K1——Recomputation kernel，为layer i+1的uncached旧token执行KV投影（GEMM操作，compute-intensive）；(2) K2——Decode kernel，用layer i的完整历史KV（cached+recomputed）对当前新token执行decode attention（memory-intensive）。融合后减少kernel launch overhead，提高SM utilization，并使两类操作的计算资源可以在thread block级别动态分配。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

K1+K2 Fused Kernel的调度伪代码（以Llama2-13B MHA decode为例）：

```
// === Pre-compilation ===
// 预编译多组.so: 32种thread配置 (32..1024 step 32)
for num_threads in range(32, 1024+1, 32):
    compile_fused_variant(num_threads)
    // 每个variant包含K1和K2子区域，thread block数按计算量比分

// === Runtime per-layer execution ===
function fused_k1_k2_launch(layer_i, batch_requests, b, r):
    // 计算K1和K2的工作量
    num_uncached_tokens = sum(req.seq_len * req.r for req in batch_requests)
    // K1 FLOPs: GEMM QKV projection for old tokens
    flops_K1 = 2 * hidden_size * head_dim * num_heads * num_uncached_tokens
    // K2 FLOPs: attention decode for current token  
    flops_K2 = hidden_size * (num_cached_tokens + num_uncached_tokens)
    
    // 计算FLOP ratio
    ratio = flops_K1 / (flops_K1 + flops_K2)  // e.g. 0.7 → K1占70%
    
    // 选择最接近的预编译variant
    target_threads = int(ratio * max_threads)  // e.g. 0.7*1024 = 717
    target_threads = round_to_multiple(target_threads, 32)  // 对齐warp
    fused_kernel = load_library(target_threads)
    
    // Stream A: 异步host↔GPU KV传输
    cudaStream_t stream_A = streams["transfer"]
    for token in uncached_tokens:
        cudaMemcpyAsync(gpu_kv_buf, host_kv_buf, ..., stream_A)
    
    // Stream B: fused K1+K2在单一kernel launch中执行
    cudaStream_t stream_B = streams["compute"]
    fused_kernel<<<grid, block, 0, stream_B>>>(
        // K1子kernel参数
        K1_Q, K1_K, K1_V,        // old token QKV (query投影可选)
        K1_uncached_token_ids,    // 需要重算的token indices
        K1_output_KV,             // 输出：重算的KV (存入临时workspace)
        
        // K2子kernel参数  
        K2_Q_current,             // 当前新token的query
        K2_KV_cached,             // 已缓存的KV (从GPU显存)
        K2_KV_recomputed,         // K1刚产生的KV (从临时workspace)
        K2_output,                // 输出：当前token的attention output
        
        thread_ratio              // K1/K2 thread block分配比例
    )
    
    // K1完成后释放临时KV workspace
    // 同步两条stream
    cudaStreamSynchronize(stream_A)
    cudaStreamSynchronize(stream_B)

// === Fused Kernel CUDA内部 ===
__global__ void fused_k1_k2_kernel(...):
    block_id = blockIdx.x
    total_blocks = gridDim.x
    
    if block_id < K1_blocks:  // 前ratio*N个block执行K1
        // K1: GEMM for KV recomputation
        token_idx = K1_uncached_token_ids[block_id % num_uncached_tokens]
        // X[token_idx] · W_K → computed_K
        // X[token_idx] · W_V → computed_V  
        computed_K[...] = matmul(X[token_idx], W_K)
        computed_V[...] = matmul(X[token_idx], W_V)
        __syncthreads()
        // 写入临时workspace供K2使用
        K1_output_KV[token_idx] = {computed_K, computed_V}
        
    else:  // 剩余(1-ratio)*N个block执行K2
        // K2: attention decode
        head_idx = (block_id - K1_blocks) % num_heads
        // concat(cached_KV, recomputed_KV)
        full_K = concat(K2_KV_cached.K, K2_KV_recomputed.K)
        full_V = concat(K2_KV_cached.V, K2_KV_recomputed.V)
        // scaled dot-product attention
        Q_head = K2_Q_current[head_idx]
        scores = Q_head @ full_K^T / sqrt(head_dim)
        attn_weights = softmax(scores)
        K2_output[head_idx] = attn_weights @ full_V
```

**关键设计要点**：
1. **thread block分区**：不按warp而是按thread block划分K1/K2——K1和K2使用独立的thread block组，避免warp divergence。grid中的前`ratio * total_blocks`个block执行K1 GEMM，剩余block执行K2 attention。
2. **临时KV workspace**：K1产生的旧token KV暂存于临时workspace buffer（约1 layer KV大小），K2读取后立即释放，避免持久占用显存。
3. **线程数对齐**：总线程数对齐到32的倍数（warp granularity），确保无idle warp lane。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在eLLM中的实现方式：
1. **离线编译**：对Llama2-13B MHA和Llama2-70B GQA分别预编译32组fused kernel variant（tuned for 32..1024 threads），每组为独立的CUDA .so shared library。
2. **运行时加载**：layer-level scheduler在每层执行前根据当前b×r估算K1/K2 FLOP ratio → 在32组中选最接近的thread配置 → 动态加载对应.so → 配置grid/block dim → launch。
3. **workspace管理**：为K1临时KV输出预分配workspace buffer（大小 = max_uncached_tokens × per_token_kv_size_in_layer_group），每层复用。
4. **与Comm-Com Overlap协同**：K1+K2在Stream B上执行期间，Stream A异步传输下一层的host-GPU cached KV（对swapped token），两者通过cudaEvent同步。
5. **消融验证**：论文禁用Kernel Fusion后TPOT和throughput退化。融合的代价是额外workspace显存约1 layer KV，通过closed-loop adaptation中的Mo参数反馈控制。

术语的通用性：虽然K1+K2 fusion是eLLM的具体实现，但这种"将不同latency-bound/compute-bound特性的kernel融合以减少launch overhead并共享SM"的思路在其他LLM serving kernel优化中也有应用（如FlashAttention的forward+backward融合、FlashInfer的prefill+decode融合）。

涉及论文标题：
- High Throughput and Low Latency LLM Serving via Adaptive KV Caching

## Attention Fusion for Decode（解码阶段的注意力融合）

术语是什么？通过联网搜索让回答具体和精准。
Attention Fusion是MFS论文提出的一种GPU kernel级优化技术，用于提升multi-tier serving中小batch decode阶段的GPU并行度。在LLM decode阶段，每个请求每次只生成一个token，导致attention计算中query矩阵极小（batch_size=1或很小），GPU的SMs利用率低。MFS的Attention Fusion将来自group batch中不同请求的Q、K、V矩阵按batch维度拼接为一个更大的联合QKV矩阵，执行一次联合attention计算，以少量冗余attention计算（跨请求token之间本不需要计算的attention）换取更高的GPU SM利用率和计算吞吐。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Attention Fusion的kernel执行流（以3个请求的group batch为例）：

传统decode attention（无fusion）：
```
for each request in batch:
    Q = hidden[req] @ W_q  # shape: (1, num_heads, head_dim)
    K = hidden[req] @ W_k  # shape: (1, num_heads, head_dim)
    V = hidden[req] @ W_v  # shape: (1, num_heads, head_dim)
    # 小Q→低GPU occupancy
    attn_out = flash_attention(Q, K_cache[req], V_cache[req])
```

Attention Fusion（MFS方案）：
```
# Step 1: 分别计算各请求的QKV
Q_list, K_list, V_list = [], [], []
for req in group_batch:  # e.g., 3 requests
    Q_list.append(hidden[req] @ W_q)  # each (1, nh, hd)
    K_new = hidden[req] @ W_k
    V_new = hidden[req] @ W_v
    # 将新KV追加到各自cache
    K_cache[req] = concat([K_cache[req], K_new])
    V_cache[req] = concat([V_cache[req], V_new])

# Step 2: 拼接各请求的Q、K、V
Q_fused = concat(Q_list, dim=0)  # shape: (3, nh, hd)→GPU parallelism ↑
K_fused = concat([K_cache[r] for r in group_batch], dim=0)  # (3 * seq_len, nh, hd)
V_fused = concat([V_cache[r] for r in group_batch], dim=0)  # (3 * seq_len, nh, hd)

# Step 3: 单次联合attention（含冗余cross-request attention）
attn_fused = flash_attention(Q_fused, K_fused, V_fused)
# 注意：R1的query会attend to R2和R3的KV→冗余计算
# 论文认为GPU并行能力可掩盖这部分额外时间

# Step 4: 拆分结果回各请求
attn_out[req_1], attn_out[req_2], attn_out[req_3] = split(attn_fused, dim=0)
```

关键trade-off：
- Compute overhead：Attention Fusion引入了跨请求token之间的冗余attention计算（如R1的query与R2的KV做attention——这些结果会被丢弃或不影响最终输出）。在标准attention实现中无法mask掉这些跨请求attention（因为不同请求的sequence可能不等长且causal mask不同）。
- GPU parallelism gain：更大的QKV矩阵→更多thread blocks→更高SM occupancy→decode阶段GPU compute bound转变为更充分利用。论文认为在GPU compute能力充足时，parallelism gain > redundant compute cost。
- 适用场景：仅在小batch decode阶段有效（batch size小时GPU utilization低）。prefill阶段本身计算量已足够大，通常不需要fusion。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
论文未详细说明Attention Fusion的具体kernel实现。根据论文描述，Attention Fusion使用标准的attention实现（如FlashAttention-style kernel），但输入是拼接后的QKV矩阵。论文未提出新的CUDA kernel，而是在标准attention kernel的调用层面通过改变输入张量的组织方式来提升GPU利用率。该技术与flash attention的batch mode类似但增加了跨请求冗余计算。论文未开源实现代码。Attention Fusion在MFS系统中的典型收益：配合group batching，在speculative sampling场景中将GPU utilization从Orca的约23.9%提升到约59.8%（提升主要来自group batching + attention fusion的联合效果，论文未单独消融attention fusion的贡献）。

涉及论文标题：
- MFS: An Efficient Model Family Serving System for LLMs

## Hardware-aware Tile Quantization for NPU GEMM

术语是什么？通过联网搜索让回答具体和精准。

Hardware-aware tile quantization是一种将fine-grained group quantization的weight layout重新排列以对齐矩阵乘法单元（如HMX）tile-level数据布局的技术。核心思想：先按矩阵单元期望的tile memory order重排权重，再在memory order上做group quantization，使quantization group的连续性与硬件tile的连续性一致。对HMX FP16：tile layout为32×32 column-major + tile内2-row permutation；在此order上做group size=32的4-bit量化等价于以2×16 tile片段为量化组。量化后通过group coalescing将8个group的INT4值合并为128-byte super-group（恰好填满一个HVX 1024-bit register），scale连续存放。相比conventional column-major group layout需要runtime scatter到TCM（开销极大），tile quantization使weight tile可连续写入TCM。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Offline: Weight preparation
W = load_hf_weights()                   // [K, N], FP16
W_tile = permute_to_hmx_layout(W)       // tile级column-major + tile内2-row perm
for group g in range(num_groups):       // group size 32, 在tile memory order连续
    scale = max(abs(W_tile[g*32:(g+1)*32])) / 7.0
    W_q[g*32:(g+1)*32] = round(W_tile[...]/scale)

// Post-quant: Coalesce 8 groups → super-block
// 8 × 32 INT4 = 256 INT4 = 128B = 1 HVX register; scales 8 × 2B = 16B contiguous
for super_g in range(num_groups // 8):
    dst_q[super_g*128:(super_g+1)*128] = concat(groups 0..7 quantized)
    dst_s[super_g*16:(super_g+1)*16] = concat(groups 0..7 scales)

// Runtime: Dequantized GEMM kernel
for tile_idx in range(num_tiles):
    dma_load(W_q[tile_idx], TCM)       // DMA搬weight+activation tile入TCM
    dma_load(A_tile, TCM)
    for super_g in tile:
        W_fp16 = vlut16(W_q[super_g], LUT_INT4_TO_FP16)  // INT4→FP16
        scales = vlut16(const_indices, scale_LUT)          // broadcast 4 scales
        W_fp16 *= scales
    O_tile += hmx_mma(W_fp16, A_tile)  // HMX 32×32 tile MM, FP32 accum
    dma_store(O_tile, DDR)
```

相比conventional layout (column-major group quant + runtime scatter)，加速9.65-19.04×；相比仅HMX layout无coalesce，再加速1.82-3.45×；仅比no-dequantization上界慢~27%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：(1) 离线转换脚本：HuggingFace weight→HMX tile layout GGUF→llama-quantize (Q4_0/Q8_0)；(2) 不同NPU的tile layout不同，需为每种硬件定制permutation；(3) 通用性：AMX (Intel)、SME (ARM)等CPU矩阵单元也有类似tile layout，核心思想可迁移；(4) 精度：tile-group quant与conventional group quant的WinoGrande/MMLU/Wiki PPL差异远小于量化本身损失。开源实现见llama.cpp-npu。

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

## LUT-based Dequantization and Softmax on NPU Vector Units

术语是什么？通过联网搜索让回答具体和精准。

LUT-based computation在NPU上使用向量单元的查表指令替代复杂数学计算，克服NPU向量单元通用计算能力弱、缺乏专用数学函数单元的瓶颈。论文两项LUT技术：(1) LUT dequantization：HVX的vlut16指令将4-bit INT4直接查表映射为FP16，避免传统mask-unpack-convert多指令序列；(2) LUT Softmax：HVX的vgather指令从预计算64KiB FP16 exp LUT收集值，替代多项式exp2展开，消除VLIW顺序依赖并减少指令数。两者利用NPU向量单元的硬件LUT指令以低延迟完成原本昂贵计算。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

**LUT Softmax (safe softmax, FP16):**
```
// 预计算: 64KiB exp LUT in TCM, 仅x≤0的32768 entries
// LUT[half_bits] = exp(FP16_val) for FP16_val ≤ 0
// safe softmax: S - rowmax ensures all inputs ≤ 0

LUT_Exp(S_sub_max):  // all ≤ 0
    off = (S_sub_max << 1) & 0xFFFE   // 忽略MSB (sign=1), left-shift for 2B offset
    return vgather(LUT_base + off)     // 一次收集64个FP16

// FlashAttention on-chip softmax (Algorithm 1):
for KV_tile j:
    S_i = HMX_MatMul(Q_i, K_j^T, Acc=FP32)  // [Bq, Bkv] FP16
    m_new = max(m_old, HVX_rowmax(S_i))
    P_i = LUT_Exp(S_i - m_new)               // vgather
    l_new = exp(m_old-m_new)*l_old + HVX_rowsum(P_i, Acc=FP32)
    O_i = diag(exp(m_old-m_new))*O_old + HMX_MatMul(P_i, V_j)
```

LUT Softmax vs F32 exp: 1.26-2.19× speedup; vs F16 exp: up to 1.60×。

**LUT Dequantization (vlut16):**
```
// vlut16: 每个8-bit index查16-entry table, 输出16-bit; 生成一对128B registers
LUT_INT4_TO_FP16 = [FP16(-8),..., FP16(7)]  // 16 entries, 32B
// Scale广播: 将4组scales放LUT content, constant indices查表
// 一次vlut16完成4组scale广播到全register

dequant(W_q, W_s):
    W_fp16 = vlut16(W_q_high, LUT4_TO_FP16)  // INT4→FP16, 无qfloat overhead
    scales = vlut16(indices, scale_LUT)        // broadcast
    return W_fp16 * scales
```

vlut16直接产生IEEE-754 FP16 (vs HVX默认qfloat格式需额外转换)，进一步减少指令。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LUT在系统初始化时预计算（64KiB exp LUT占TCM 0.8%），无运行时overhead。通用性：(1) LUT dequant支持任意4-bit编码(FP4, NF4, IQ4_NL)，仅换table content；(2) LUT Softmax依赖safe softmax保证输入≤0，否则需128KiB全范围LUT；(3) vgather延迟较高(24-48 VLIW packets on V75)，需与无关指令交错编排；(4) 类似思想可用于ARM SVE/SSVE、x86 AVX-512等有LUT指令的向量ISA。

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

## HMX FP16 Tile Memory Layout

术语是什么？通过联网搜索让回答具体和精准。

HMX FP16 Tile Memory Layout是Hexagon NPU的HMX矩阵单元对FP16操作数要求的内存排列格式。每个tile为32×32 FP16 (2KiB)。Layout分两级：(1) Level-1 tile内部：每两行做permutation——[a0..a31; b0..b31]排列为[a0,b0,a1,b1,...,a31,b31]，等价于转置后的2×32子矩阵；(2) Level-2 tile间：weight tiles按column-major排列（HMX执行tile-level inner product，weight column-major匹配accumulation order）。Activation tile为row-major排列。所有HMX指令只能读取TCM内数据，因此runtime需将数据按此layout放入TCM。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// GEMM C[M,N] = A[M,K] × B[K,N] with HMX tiles
// 所有矩阵reshape为 [dim1/32, dim2/32, 32, 32] tiles
// 每tile内部: 2-row permutation memory order

for n in 0..N/32:        // Column tiles of B and C
    for m in 0..M/32:    // Row tiles of A and C
        C[m][n] = 0
        for k in 0..K/32:
            C[m][n] += hmx_mma(A[m][k], B[k][n])
            // hmx_mma: load A tile & B tile from TCM → 32×32×32 inner product
            // internally FP32 accumulate → output FP16 tile
        // Post: per-column scale + bias on output tile (HMX native)
```

Decode阶段观察：activation A shape [B, hidden_dim]，B通常=1。A reshape为[B/32, hidden_dim/32, 32, 32]，B=1时仅1个activation tile行有有效数据，31行空闲——这是HMX decode underutilization的核心。TTS增加B（如B=8），填充更多tile行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

权重tile layout转换离线完成（模型转换脚本，一次转换永久存储）。Activation runtime转换：HVX cross-lane shuffle指令对每两行做permutation，可与dequant重叠。不同量化格式(IQ4_NL等)通过LUT vlut16直接映射到HMX-compatible FP16。开源实现见llama.cpp-npu的GGUF conversion和htp-ops-lib的kernel代码。

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

## Pipelined Parameter Restoration

术语是什么？通过联网搜索让回答具体和精准。

Pipelined Parameter Restoration是TZ-LLM提出的TEE内LLM推理的运行时调度机制，将LLM推理由传统串行流程（先完整加载/解密参数→再计算）改造为allocation→loading→decryption→computation四种operator的DAG流水线。核心理念：利用LLM computation graph的确定性拓扑顺序（parameters按layer顺序访问），将参数restoration操作（CMA allocation、flash I/O、AES decryption）与CPU/NPU computation重叠执行，把restoration latency隐藏到computation latency下。配合priority-based greedy scheduling、preemptive micro-operator scheduling和partial parameter caching，使按需动态扩展secure memory的overhead不再完全暴露在TTFT critical path上。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Pipelined Parameter Restoration的调度算法：

```
// 扩展computation graph: 在每个computation operator前插入restoration operators
for layer_idx in topological_order:
    comp_op = original_graph[layer_idx]
    alloc_op = new AllocOp(layer_idx, param_size[layer_idx])
    load_op  = new LoadOp(layer_idx, param_size[layer_idx])
    decr_op  = new DecrOp(layer_idx, param_size[layer_idx])
    // 依赖: alloc → load → decr → comp
    // 跨层无依赖: comp[layer_i] 和 {alloc,load,decr}[layer_{i+1}] 可重叠

// Priority-based Greedy Scheduler
ready_queue = PriorityQueue()
while has_pending_operators():
    ready_queue.update()  // 加入依赖已满足的operators
    if ready_queue.has(COMPUTE):
        op = ready_queue.pop_highest_priority()  // 优先computation
    else:
        earliest_comp = find_earliest_computation_operator()
        op = get_associated_restoration(earliest_comp)
    execute(op)

// Preemptive micro-operator: alloc/decrypt切为~64KB micro-ops
// computation就绪时抢占当前micro-op，减少pipeline bubble
```

具体执行时序（Llama-3-8B 512-token prompt）：
```
Layer0: [Alloc0][Load0][Decrypt0]======[Compute0]==========
Layer1:          [Alloc1][Load1][Decrypt1]======[Compute1]==
Layer2:                   [Alloc2][Load2][Decrypt2]=========
```
Restoration latency被隐藏到computation latency下，strawman中串行的~11.6s restoration overhead被pipeline化。

关键调度策略：
1. **Priority Policy**：CPU computation operator优先级最高（避免NPU/CPU idle），其次与earliest computation关联的restoration
2. **Preemptive Scheduling**：large alloc/decrypt切为~64KB micro-operator，computation就绪时抢占。实验：preemptive进一步降低TTFT最多16.2%
3. **Partial Parameter Caching**：REE memory pressure允许时缓存早期prefill参数（按topological order），按reverse order释放。cache比例阈值前TTFT近似线性下降

Pipeline-aware extend/shrink：利用LLM参数first-in-last-out模式保证TZASC连续物理内存。性能：pipeline scheduling距critical path lower bound仅0.01%-9.9%，vs strawman TTFT降低77.1%-91.1%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

TZ-LLM基于llama.cpp实现（~1.2K LoC扩展）：提取computation graph拓扑顺序→插入restoration operator→priority queue scheduler（C++ std::priority_queue）+ preemptive micro-operator机制。CMA allocation通过REE Linux CMA API，flash I/O通过REE异步IO，decryption通过OpenSSL AES。依赖LLM参数访问顺序确定性（MoE/early-exit可能预取未使用参数）。开源：Zenodo artifact DOI 10.5281/zenodo.17213486。

涉及论文标题：
- TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone

## Shadow Job / Secure NPU Job Scheduling

术语是什么？通过联网搜索让回答具体和精准。

Shadow Job是TZ-LLM co-driver NPU架构中的核心调度机制，用于在TEE和REE之间安全调度NPU job。Shadow Job是TEE TA在REE NPU调度器队列中放置的轻量占位job——仅包含job ID和元数据（优先级、estimated duration），不含实际NPU command/register序列。当REE scheduler选中此shadow job时，通过SMC通知TEE data plane driver，由TEE driver执行实际secure NPU job launch。Shadow Job使REE统一管理REE NN job和TEE secure job的队列调度，同时REE无法窥探或篡改secure job内容。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Shadow Job调度流程伪代码：

```
// === TEE侧: 提交Secure NPU Job ===
def submit_secure_npu_job(cmd_buf, input_buf, output_buf):
    secure_ctx = SecureExecutionContext(
        cmd_seq = cmd_buf,
        io_pt = build_iopt(),
        seq_num = next_seq_num,      // monotonic递增
        state = INITIALIZED          // 已初始化,未发行
    )
    shadow = ShadowJob(job_id, priority)
    smc_call(SUBMIT_SHADOW_JOB, shadow)

// === REE侧: NPU Scheduler ===
class NPUScheduler:
    job_queue = UnifiedQueue()  // REE NN + Shadow jobs统一队列
    
    def schedule_loop():
        while True:
            job = job_queue.pop_next()
            if isinstance(job, ShadowJob):
                smc_call(SECURE_NPU_LAUNCH, job.job_id)
                job.mark_submitted()
            else:
                npu_mmio_write(LAUNCH_REG, job.cmd_buf)
                wait_npu_completion()

// === TEE侧: Secure Launch (被SMC触发) ===
def handle_secure_npu_launch(job_id):
    ctx = secure_ctxs[job_id]
    // 安全校验
    assert ctx.state == INITIALIZED and not ctx.issued
    assert ctx.seq_num == expected_seq  // 防重放/重排序
    
    // 硬件配置
    tzpc_set(NPU, SECURE_ONLY)     // 阻止REE访问NPU MMIO
    gic_set_irq(NPU_IRQ, SECURE)   // 中断路由到TEE
    wait_current_nonsecure_job()
    tzasc_allow_npu(SECURE_MEM)    // NPU可DMA访问secure memory
    
    // Launch
    ctx.issued = True
    npu_mmio_write(LAUNCH_REG, ctx.cmd_seq)
    smc_return()  // 不等完成——中断异步通知

// TEE ISR: completion处理 → 恢复TZPC/TZASC/GIC → SMC通知REE
```

安全保证机制：
1. **防窥探**：Shadow job不含实际命令数据，REE仅见job ID/priority
2. **防重放**：monotonic sequence number——TEE验证seq_num == expected_seq
3. **防重排序**：状态机INITIALIZED→ISSUED→COMPLETED，ISSUED后不能再次launch
4. **防任意启动**：未INITIALIZED的job无法进入launch（state校验）
5. **硬件隔离**：TZPC+TZASC+GIC确保secure job执行期间REE无NPU访问权

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

TZ-LLM在RK3588实现：REE NPU driver +167 LoC（shadow job scheduling + UnifiedQueue管理 + SMC转发）；TEE侧~1K LoC NPU data plane driver（secure launch + completion handling）。Shadow job overhead：每job额外SMC往返，但TZPC/TZASC/GIC配置微秒级（vs driver detach-attach ~32ms）。NPU time-sharing对REE NN应用slowdown <=3.8%。开源：Zenodo artifact DOI 10.5281/zenodo.17213486。

涉及论文标题：
- TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone

## CMA (Contiguous Memory Allocator) for TEE Secure Memory

术语是什么？通过联网搜索让回答具体和精准。

CMA（Contiguous Memory Allocator）是Linux内核的连续物理内存分配器，用于为大块连续物理内存需求（设备DMA缓冲区、GPU/NPU framebuffer等）提供服务。在TZ-LLM中，CMA被用来为TZASC分配secure memory——TZASC要求被保护的DRAM区域必须是连续物理地址（region descriptor以物理地址起始+大小为参数），而Linux buddy allocator在系统运行后很难提供GB级连续物理页。CMA在系统启动时预留一块大连续物理内存区域（cmdline cma=配置），运行时从此区域分配；非CMA分配时这些页可作为movable页被page cache/用户页使用。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

TZ-LLM中CMA用于TEE secure memory的分配/释放流程：

```
// REE侧: CMA分配 (TZ driver, +197 LoC)
def extend_allocated(op_id, size):
    pages = cma_alloc(size, align=4KB)
    // cma_alloc内部: 
    //   1. CMA bitmap中查找足够连续空闲页
    //   2. 有movable页占用时调用migrate_pages()搬走
    //   3. 返回连续物理页的起始PFN
    return pages  // [pa_start, pa_end)，此时Non-Secure

// TEE侧: 升级为Secure
def extend_protected(op_id, pa_start, size):
    tzasc_add_region(pa_start, size, SECURE_ONLY)
    tee_map_pages(ta_vm_space, pa_start, size)
    // 此后REE无法访问此内存

// 释放流程 (inference后按reverse topological order)
def shrink_allocated(op_id, pa_start, size):
    tzasc_remove_region(pa_start, size)
    tee_unmap_pages(ta_vm_space, pa_start, size)
    cma_release(pa_start, size)  // REE侧归还CMA pool
```

CMA分配性能特性（与TZ-LLM pipeline的关系）：
- **分配延迟**：Llama-3-8B (7.9GB) CMA allocation约4.182s（memory stress下），取决于碎片和需migrate的page量
- **Pipeline隐藏**：allocation被pipeline化——operator 0的CMA完成后立即开始computation，后续operator的CMA在computation后台完成
- **Memory pressure**：stress-ng模拟REE memory pressure（四个模型对应13/11/10/6GB压力），增加CMA migration开销
- **碎片影响**：pipeline-aware extend/shrink利用LLM参数first-in-last-out模式减缓碎片（顺序分配逆序释放减少碎片空洞）

Buddy vs CMA：Buddy 4KB页粒度无法保证GB级连续（碎片化后）；CMA可保证GB级连续但需预留区域且migrate有开销。TZASC的连续内存需求是移动TEE保护大模型的根本限制——长期运行碎片严重时即使是CMA也可能无法分配数GB连续内存。Geekbench评估CMA分配对REE应用性能干扰最高约6.7%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

CMA是Linux kernel标准特性（CONFIG_CMA），通过cmdline（cma=size）或device tree配置。TZ-LLM通过REE TZ driver封装cma_alloc/cma_release供TEE远程调用。TZ-LLM通过transient allocation + pipeline overlapping缓解CMA延迟和碎片问题。

涉及论文标题：
- TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone

## Bubble-free DP Block Selection（无气泡动态规划块选择）

术语是什么？通过联网搜索让回答具体和精准。

Bubble-free DP Block Selection是FlashPS提出的用于扩散模型mask-aware serving pipeline的运行时决策算法。在FlashPS中，每个transformer block可以选择"加载cached unmasked activation + 仅计算masked tokens"或"全量计算所有tokens"。仅使用cache的问题在于cache loading（从host memory通过PCIe到GPU HBM）可能比直接全量计算更慢，导致GPU computation stream等待cache load stream，产生pipeline bubble。Bubble-free DP通过动态规划在O(N)时间内为N个transformer block决定每个block的最优策略：比较每个block在"加载cache后计算masked tokens"的完成时间与"全量计算"的完成时间，并考虑相邻block间的pipeline overlap，选择总latency最小且无bubble的块级执行方案。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Bubble-free DP的伪代码：

```
输入: N个transformer blocks, mask_ratio m, 每个block的:
      compute_full[i] - block i全量计算的GPU time
      compute_masked[i] - block i仅计算masked tokens的GPU time (≈m * compute_full[i])
      load_cache[i] - block i从host memory加载cached Y到HBM的PCIe time
输出: use_cache[1..N] - 每个block是否使用cache

// DP状态定义
dp[i][0] = min total time for blocks 1..i when block i uses FULL compute
dp[i][1] = min total time for blocks 1..i when block i uses CACHE

// 初始化
dp[0][0] = dp[0][1] = 0

// DP递推
for i = 1 to N:
    // Case A: block i用全量计算
    // 前一个block可以是任何状态，计算时间独立
    dp[i][0] = max(dp[i-1][0], dp[i-1][1]) + compute_full[i]
    
    // Case B: block i用cache
    // 需要考虑cache loading与上一个block计算的重叠
    // block i的cache loading可以与block i-1的计算重叠
    if i == 1:
        dp[i][1] = max(load_cache[i], compute_masked[i])  // 第一个block无前置重叠
    else:
        // cache loading可以与前置block的计算并行
        dp[i][1] = max(
            dp[i-1][0] + load_cache[i],                    // loading在prev block完成后才开始
            max(dp[i-1][0], 0) + load_cache[i]             // 重叠: loading可与prev compute并行
        )
        // 取min后再加masked compute time（与loading串行）
        dp[i][1] = max(dp[i][1], load_cache[i]) + compute_masked[i]

// 回溯
use_cache[N] = argmin(dp[N][0], dp[N][1])
for i = N-1 downto 1:
    use_cache[i] = backtrack(dp, use_cache[i+1])

return use_cache[1..N]
```

核心洞察：
1. **Bubble产生条件**：当`load_cache[i] > compute_full[i]`时，GPU computation stream将等待cache load完成才继续，产生bubble。
2. **DP的作用**：并非所有block都值得用cache。对于compute-intensive block（如attention），全量计算可能很快，cache loading反而成为瓶颈。DP确保只在"loading+masked计算 < 全量计算"的block使用cache。
3. **Pipeline overlap建模**：DP状态转移中，block i的cache loading可与block i-1的计算重叠。DP选取总completion time最小的策略，等同于最小化bubble时间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现细节：
1. **离线profiling**：对每个block，在目标GPU上profiling得到`compute_full[i]`、`compute_masked[i]`（作为mask ratio m的函数）、`load_cache[i]`（作为unmasked token数的函数）。FlashPS用线性模型估算。
2. **在线DP执行**：每个请求到达时，scheduler根据其mask ratio m执行O(N) DP（N通常为~20-30个transformer blocks for SDXL/Flux），DP耗时可忽略（微秒级）。
3. **CUDA Stream编排**：DP输出`use_cache[i]`后，对use_cache[i]=true的block，提前在cache load stream上发起`cudaMemcpyAsync`；computation stream在完成block i-1后直接对masked tokens执行attention/FFN。两个stream间的同步通过CUDA event实现——computation stream在block合并点等待cache load stream完成。
4. **与传统kernel fusion的区别**：Bubble-free DP不是kernel fusion——它不合并kernel，而是在更高层级（transformer block）选择执行路径。它是pipeline级别的调度优化。

涉及论文标题：
- FlashPS: Efficient Generative Image Editing with Mask-aware Caching and Scheduling

## Asynchronous Wavefront Optimization (G(u))

术语是什么？通过联网搜索让回答具体和精准。
Asynchronous Wavefront Optimization是Infera TEU SelectKernels中第一阶段data block selection的优化目标：在data dependency DAG中选择data blocks时，不是简单选择所有zero in-degree节点，而是通过递归定义的G(u) metric估计每个候选节点对未来asynchronous wavefront（互不依赖、可并行执行的data block数量）的贡献，优先选择能最大化未来并行度的节点。G(u)越大，说明选择该节点后能释放更多下游独立可执行节点。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
G(u)的递归定义和计算过程：
```
// G(u): 节点u的asynchronous wavefront增益
// Γ⁺(u): u的出邻居集合
// d⁻(v): v的入度

G(u) = (∑_{v ∈ Γ⁺(u)} G(v) + 2) / |Γ⁺(u)| - 1

// 递归终止: 在特定深度时 G(v) = -1

// 选择策略: 对所有zero in-degree节点计算G(u), 选最大值
```

计算例子（简化3层DAG）：
```
    A (completed)     B (completed)
   / \               / \
  C   D             E   F
   \ /               \ /
    G (in-degree=2)   H (in-degree=2)
     \               /
      I (in-degree=2)
```
- 节点C: out-neighbor G (d⁻(G)=2) → 假设depth limit, G(G)=-1 → G(C) = ( (-1) + 2 ) / 1 - 1 = 0
- 节点D: out-neighbor G (d⁻(G)=2) → G(D) = ( (-1) + 2 ) / 1 - 1 = 0
- 如果 C 和 D 都完成: G的in-degree变为0 → 释放G为可执行

递归特性：transitive dependency propagation使得子节点的asynchrony gain均匀传播到所有父节点。终止深度防止长链过度影响G(u)值。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Infera在SelectKernels阶段实现G(u)计算：data dependency DAG在编译期通过computation graph静态分析构建，推理时动态维护节点状态（completed/pending/running）。G(u)的计算overhead低，因为大部分在编译期预计算（DAG拓扑、d⁻值），推理时仅对受影响的节点增量更新。选择G(u)最高的zero in-degree data blocks后，再进入第二阶段kernel selection（min #inst/IPC s.t. TLP≥4）。

涉及论文标题：
- Automated End-to-End Model Serving with Cooperative Compilation and Scheduling

## GDRCopy-Based Kernel Transfer

术语是什么？通过联网搜索让回答具体和精准。
GDRCopy (GPU Direct RDMA Copy) 是NVIDIA提供的一种低延迟host-device数据传输库，允许CPU直接通过PCIe BAR (Base Address Register) mapping读写GPU显存，bypasses传统CUDA API的DMA引擎和driver stack。Infera使用GDRCopy的gdr_copy_to_mapping函数将fused kernel的binary code和arguments从host端HKQ传输到device端kernel slots和DKQ，实现<100ns的小数据延迟和<5μs的典型kernel传输延迟。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GDRCopy在Infera kernel launch pipeline中的位置：
```
Host side:
  fused kernel binary + args in host memory (cudaHostAllocWriteCombined)
        ↓
  gdr_copy_to_mapping(gdr, dev_ptr, host_ptr, size)
        ↓ (direct PCIe BAR write, <5µs for typical kernel ~KB size)
  Device kernel slot (placeholder overwrite) + global memory (args)
        ↓
  Kernel pointer + arg pointer → DKQ
        ↓
  Daemon kernel → cudaLaunchDevice (fire-and-forget)
```

对比传统路径：
```
传统: cuLaunchKernel/cuModuleLoad
  → CUDA driver stack → GPU work queue → DMA engine → GPU execution
  → ~10-100µs overhead + global host-device synchronization

Infera: GDRCopy + daemon kernel CDP
  → GDRCopy BAR write → DKQ enqueue → device-side launch
  → <10µs total, no host-device sync, no HoL blocking
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GDRCopy需要：(1) GPU支持GPUDirect RDMA（需NVIDIA数据中心GPU + BAR1 mapping enabled）；(2) 安装nvidia-peermem和gdrcopy kernel module。Infera中，compiled kernel binary和arguments放置在cudaHostAllocWriteCombined标记的host memory中（优化PCIe write combining），通过GDRCopy直接写入GPU显存的预留kernel slot area和argument area。Kernel slot覆盖使用driver-level修改（绕过cuModuleLoad），因为cuModuleLoad会触发global host-device同步破坏低延迟目标。Host launcher维护fused kernel cache pool在host memory中复用。

涉及论文标题：
- Automated End-to-End Model Serving with Cooperative Compilation and Scheduling

## Driver-Level Placeholder Kernel Slots

术语是什么？通过联网搜索让回答具体和精准。
Driver-Level Placeholder Kernel Slots是Infera绕过CUDA driver kernel module加载限制的技术：NVIDIA GPU执行kernel前需通过cuModuleLoad将CUDA binary加载到module，该函数会触发global host-device synchronization，不适用于推理时的低延迟动态kernel发射。Infera在GPU memory中预留placeholder kernels（空壳kernel），运行时通过driver-level修改直接覆盖其code section为fused kernel的实际binary code，从而跳过cuModuleLoad，实现低延迟动态kernel加载。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Placeholder kernel slot的工作流程：
```
编译期（离线）:
1. 预留N个placeholder CUDA kernels
2. 编译为标准CUDA binary
3. cuModuleLoad加载到GPU → 分配kernel memory slots
4. 记录每个slot的device address和size

推理期（在线）:
1. FuseKernels生成fused kernel binary
2. 选择size足够的空闲placeholder slot
3. GDRCopy gdr_copy_to_mapping(slot_addr, fused_binary, size)
   → 直接覆盖slot的code section
4. 更新launch config (gridDim, blockDim, sharedMem)
5. DKQ入队(kernel_ptr=slot_addr, args_ptr, launch_config)
6. Daemon kernel cudaLaunchDevice(slot_addr, ...)
   → GPU执行覆盖后的fused kernel code
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Infera利用NVIDIA GPU kernel memory管理的特性：cuModuleLoad后kernel code section在GPU memory中固定，不会动态重定位。通过driver-level API（论文引用[21]但未详述具体API）获取slot物理地址并直接写入。关键约束：(1) fused kernel code size ≤ placeholder slot size；(2) slot数量需足够容纳并发fused kernel的峰值；(3) 覆盖操作需与daemon kernel launch正确同步（slot被占用时不可覆盖）。论文未说明driver-level修改的具体API名称，仅标注引用[21]。

涉及论文标题：
- Automated End-to-End Model Serving with Cooperative Compilation and Scheduling
