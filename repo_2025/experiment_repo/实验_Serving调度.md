## 82-COMET- Towards Practical W4A4KV4 LLMs Serving.pdf

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：COMET 推理框架——首个实用的 W4A4KV4 LLM Serving 系统，集成 FMPQ 量化算法和 COMET-W4Ax kernel。基于 TensorRT-LLM 和 CUTLASS 构建（新增约7000行 C++/CUDA），提供 Python 接口（pybind）和 C++ API。核心修改：(1) 集成 W4Ax kernel 替代标准 FP16/INT8 GEMM，支持混合精度矩阵乘法 O = W × X（W 为 INT4 权重，X 为混合精度 INT4/INT8 激活）；(2) 采用 vLLM 的 PagedAttention KV cache 管理优化 [23]，支持 4-bit KV cache 以减少显存占用；(3) Kernel 编译为独立 .so 动态库，可无缝集成到现有推理系统（TensorRT-LLM、llama.cpp、DeepSpeed），通过 pybind 绑定到 Python 以支持 PyTorch/HuggingFace。
  - 实验比较：(1) End-to-end throughput (input/output=1024/512 和 128/128)：COMET vs TRT-LLM-FP16 / TRT-LLM-W4A16 / TRT-LLM-W8A8 / Qserve，覆盖 Mistral-7B、LLaMA-2-7B/13B、LLaMA-3-8B、LLaMA-1-30B、LLaMA-3-70B、Qwen2-72B；(2) 跨 batch size 吞吐 (BS=4/8/16/32/64): TRT-LLM-FP16 / TRT-LLM-W4A16 / TRT-LLM-W8A8 / COMET，LLaMA-3-8B；(3) Batch size=4 展开各 LLM 吞吐对比；(4) Ablation (end-to-end): weight-activation only / KV cache only / COMET 全配置；(5) Kernel 和 end-to-end 的优化逐步消融实验。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA A100-80GB-SXM4 (单卡, 80GB HBM2e, 2.0TB/s bandwidth)
  - CUDA 12.1
  - 性能测量：NVIDIA Nsight Systems [38]（端到端）、NVIDIA Nsight Compute [37]（kernel级）

- 开源Serving框架是什么。修改了什么。
  - 基础框架：TensorRT-LLM v0.10.0 [40] + CUTLASS [36]，集成 vLLM PagedAttention KV cache 管理
  - 修改点：(1) 新增 COMET-W4Ax kernel（7000行 C++/CUDA），替换标准 GEMM kernel；(2) 实现混合精度 tile descriptor，调度 INT4/INT8 混合计算 tile；(3) 实现 4-bit KV cache 量化集成（channel-wise INT4）；(4) pybind 封装 Python 接口 + C++ API；(5) 支持编译为独立 .so 动态库，可热加载到现有推理系统

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：https://github.com/rhmaaa/COMET-LLM
  - 框架执行全过程（以 LLaMA-3-70B on A100-80G-SXM4, batch_size=64, input/output=1024/512 为例）：
    1. **模型加载与量化**（离线）：加载预训练 LLaMA-3-70B FP16 权重 → FMPQ 算法执行：(a) OmniQuant 对所有权重做 4-bit 量化；(b) 校准数据采样，定位各层 outlier channels，执行 channel permutation；(c) Block-wise (k=128) 激活量化，大部分 block 为 INT4（~84%），少数含 outlier block 为 INT8（~16%）；(d) KV cache 配置为 channel-wise INT4 量化。量化后模型权重和量化参数（scale、permutation index）写入磁盘。
    2. **COMET 框架初始化**：加载量化模型参数到 HBM → 初始化 COMET-W4Ax kernel（.so 动态库）→ 初始化 PagedAttention KV cache（4-bit，vLLM 风格 page 管理）。
    3. **Prefill Phase（GPU 执行）**：对输入 prompt [B=64, L=1024, H=8192]，逐层执行：
       - LayerNorm → QKV Proj：COMET-W4Ax GEMM。激活 tensor 按 k=128 分块，52 个 INT4 block + 12 个 INT8 block 混合计算。tile remapping + task-stealing 均衡 SM 负载。
       - Attention：Q×K^T (activation-activation，memory-bound) + Softmax + S×V。生成 KV cache（channel-wise INT4 量化后存入 PagedAttention page）。
       - O Proj + FFN (Gate/Up/Down Proj)：COMET-W4Ax GEMM，同上。
    4. **Decode Phase（GPU 执行）**：对每步生成 [B=64, H=8192]：
       - QKV Proj：COMET-W4Ax GEMM（小矩阵，memory-bound），batch 并行是关键。
       - Attention：从 PagedAttention page 加载 4-bit KV cache，dequant 后计算。因 4-bit KV cache 显著减少显存占用，可支持更大 batch。
       - O Proj + FFN：COMET-W4Ax GEMM。
    5. **显存效益**：INT4 权重 (vs FP16) 节省 4×，INT4 KV cache 节省 4×，输入激活 W4Ax 减少计算数据量。在 A100 80GB 上可支持 LLaMA-3-70B 的大 batch（如 BS=64），而 FP16 baseline 可能 OOM。
    6. **吞吐输出**：tokens/s = total output tokens / wall-clock time。COMET 在 1024/512 设置下平均 2.02× over TRT-LLM baseline。

## 81-Klotski- Efficient Mixture-of-Expert Inference via Expert-Aware Multi-Batch Pipeline .pdf

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Klotski 是基于 PyTorch 和 Hugging Face Transformers 实现的 MoE 推理引擎（约3000行 Python），构建在 FlexGen 之上。核心修改包括：(1) expert-aware multi-batch pipeline paradigm——同时考虑多个 batch 的计算，以 expert 而非 batch 为维度组织 MoE layer 的计算图，prefetch 时仅加载 gate 和 hot experts 的权重而非整个 MoE layer，并在 expert 计算顺序上优先执行 hot experts 以给 cold experts 的 I/O 传输留出时间；(2) constraint-sensitive I/O-compute planner——先测量当前硬件环境下各层的计算时间和传输时间，然后通过不等式约束组求解最优 batch group 数量 n，使 pipeline bubbles 最小化；(3) correlation-aware expert prefetcher——离线阶段通过预跑推理记录 expert selection 生成 expert correlation table（JSON格式），在线阶段基于此表和当前 token 的历史 expert 激活路径确定 hot experts 进行 prefetch；(4) adaptive tensor placement——根据当前环境的 GPU/CPU/Disk 三层异构内存容量，智能分配各类 tensor（expert/gate/attention/KV cache/activation）的存储位置，支持 layer 粒度分布；(5) 使用四个 CUDA stream 实现 I/O 与计算 overlap：weight prefetch stream、expert transfer stream、KV cache prefetch stream、KV cache store stream。
  - 实验比较：(1) 端到端 throughput 对比：Klotski vs Accelerate / DeepSpeed-FastGen / FlexGen / MoE-Infinity / Fiddler，在 Mixtral-8×7B（Env1: RTX 3090 + Intel Xeon Gold 5318Y + SSD）和 Mixtral-8×22B（Env1 / Env2: H800 + Intel Xeon Platinum 8470）上，batch sizes 4-64，序列输入长度512输出长度32；(2) throughput-latency trade-off 对比；(3) GPU memory usage 在 prefill 阶段的变化；(4) ablation study：simple pipeline → +multi batches → +only prefetch hot experts → Klotski（+adjust expert computation order）→ Klotski(q) 的逐步 throughput 提升；(5) prefetch accuracy per layer（prefetched hot experts 是否参与计算 / 是否真的是 hot experts）；(6) n（batch group 数量）和 batch size 对 throughput 的影响。

- 硬件平台是什么，配置是什么。
  - Environment 1：NVIDIA RTX 3090 (24GB VRAM) + Intel Xeon Gold 5318Y (256GB DRAM) + SSD (2TB, 1GB/s read) + PCIe 4.0 x16
  - Environment 2：NVIDIA H800 (80GB VRAM) + Intel Xeon Platinum 8470 (800GB DRAM) + SSD (1TB) + PCIe 5.0 x16

- 开源Serving框架是什么。修改了什么。
  - 基础框架：PyTorch + Hugging Face Transformers + FlexGen。Klotski 在 FlexGen 的 zig-zag block schedule 基础上进行了针对 MoE 的扩展性修改。
  - 修改点：(1) 将 FlexGen 的逐层全量 prefetch 策略改为 expert-aware 的部分 prefetch（仅 gate + K hot experts）；(2) 将 FlexGen 的 batch-by-batch 执行模式修改为 expert-by-expert 执行模式（MoE layer 内按 expert 维度而非 batch 维度组织计算）；(3) 新增 expert computation order 调整逻辑，hot experts 优先计算；(4) 新增 correlation-aware expert prefetcher 模块，基于 expert correlation table 动态确定 prefetch target；(5) 新增 constraint-sensitive I/O-compute planner，自动计算最优 n；(6) 新增 adaptive tensor placement 模块，支持 layer 粒度三层存储分配；(7) 新增 HQQ 量化（可选）和 StreamingLLM sparse attention（可选）集成。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文未明确说明代码仓库链接。实现基于 PyTorch、Hugging Face Transformers 和 FlexGen（均为开源框架）。
  - 框架执行全过程：
    1. **离线阶段**：adaptive tensor placement 感知当前环境的 GPU/CPU/Disk 内存容量，将 MoE 模型各层 tensor 分配到三级异构内存。同时，使用 wikitext-2 随机采样数据（batch size=8, sequence length=512）进行预跑，记录每层每个 token 的 expert 选择，生成 expert correlation table（JSON 格式）。
    2. **在线阶段输入**：用户请求 batch 输入（wikitext-103 采样），constraint-sensitive I/O-compute planner 根据当前硬件约束测量各层计算/传输时间，求解不等式组确定最优 batch group 数量 n。
    3. **执行pipeline**：Inference thread 和 I/O thread 并行工作。对每个 MoE block：(a) 在 attention layer 对 n 个 batch 进行竖向多 batch 计算的同时，I/O thread 仅 prefetch gate 和 K 个 hot experts（由 correlation-aware expert prefetcher 根据 expert correlation table 确定）；(b) gate 计算完成后检查每个 token 选中的 expert 是否已 prefetched，未命中则立即发起 transfer；(c) MoE expert layer 按 expert 维度（而非 batch 维度）组织计算，hot experts 优先执行，cold experts 按传输完成顺序执行，充分利用 hot experts 的高计算需求/cold experts 的高 I/O 需求的互补关系；(d) 每个 expert 的所有 token 计算完成后立即 offload 其权重，不等整个 layer 完成；(e) 非 expert layer（attention/normalization）仍按 batch 顺序竖向执行。
    4. **I/O管理**：四个 CUDA stream 异步执行。当 CPU 内存不足需使用 disk 时，维护固定 L 层的 CPU 缓存窗口：GPU 从 CPU memory prefetch layer i+1 的同时，CPU 从 disk prefetch layer i+L。

## 76-InstAttention_In-Storage_Attention_Offloading_for_Cost-Effective_Long-Context_LLM_Inference.pdf

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：InstAttention 基于 FlexGen 框架修改，将 Computational Storage Drive (CSD) 集成为异构计算设备，实现 decoding-phase attention 和 KV cache 联合 offload 到 CSD。关键修改：(1) 将 FlexGen 中原用于 KV cache offload 的 TorchDisk 对象升级为 TorchDevice——一个具备计算能力的存储设备，通过相同 API 无缝集成 CSD 到 FlexGen 的 GPU-CPU 异构计算框架中；(2) 实现 GPU-CSD 的 P2P DMA 直接数据传输，绕过 host memory 和 filesystem，在 NVMe 命令的 DWord10 字段嵌入自定义 32-bit 逻辑地址（含 batch/layer/token/head/channel 字段）；(3) layer-wise pipeline 传输——prefilling 阶段第 i 层生成的 KV cache 在 GPU 计算第 i+1 层时并发传输至 CSD，隐藏传输延迟；(4) Host 端 InstAttention Scheduler 负责调度推理任务、编排 GPU-CSD 间数据传输（发起 DMA transaction between CSD 逻辑地址和 GPU VRAM 映射地址）、以及在 CSD 空闲时触发 GC（垃圾回收）；(5) 支持 multi-CSD 扩展——利用 MHA 的 head 级并行性，将不同 attention head 分配到多个 CSD 并发处理（nhead≫n_csd 时可线性扩展），各 CSD 输出传回 GPU 后 concatenate。
  - 实验比较：(1) 1-SSD(CSD) 端到端 throughput (tokens/s): InstA（无 SparF）和 InstA-SparF（完整） vs DeepSpeed / FlexGen / FlexGen-GDS / FlexGen-SparQ / Recomp，OPT-13B on A6000, batch sizes 4-256；(2) 2-SSD(CSD) 扩展性 throughput；(3) OPT-30B 和 Llama-2-13B 上的 throughput；(4) Multi-CSD scalability：1-20 CSDs 下 dense 和 sparse 吞吐变化；(5) 压缩比 sensitivity：不同 SparF 压缩比（1, 0.8, 0.6, 0.4, 0.2, 0.125）下 1 CSD 和 2 CSDs 的吞吐。

- 硬件平台是什么，配置是什么。
  - GPU: NVIDIA A6000 48GB VRAM, PCIe Gen4x16 (32GB/s)
  - CPU: Intel Xeon 5320 2.2GHz, 96GB DDR4
  - SSD: Samsung 980 Pro 2TB
  - InstCSD: Daisyplus OpenSSD with Xilinx ZU17EG MPSoC (FPGA + 4-core ARM, 2GB DRAM, PCIe 3.0x4); 虚拟化 InstCSD via NVMeVirt (8 channels, 1.4GB/s per channel, PCIe 4.0x4 7GB/s); FPGA原型: Xilinx Zynq7045 (更经济的FPGA SoC for edge)

- 开源Serving框架是什么。修改了什么。
  - 框架: FlexGen (https://github.com/FMInference/FlexGen)
  - 修改: (1) TorchDisk→TorchDevice: 使 CSD 从被动存储变为主动计算设备，可接收 attention 计算请求；(2) 移除 host filesystem 依赖——KV cache 完全由 InstCSD 内部 FTL 管理，GPU 直接通过 NVMe 自定义逻辑地址访问；(3) 新增 NVMe 控制面命令：config()（设置模型超参数如 head 数、hidden dim）、attend()（触发 attention 计算）、reclaim()（发起 GC 擦除过期 KV cache block）；(4) 实现 P2P DMA 传输路径（GPU↔CSD 直通 PCIe，不经过 host DRAM）；(5) 添加 layer-wise KV cache 传输 pipeline 调度； (6) 扩展 FlexGen 支持 Llama-2 系列模型；(7) 添加 multi-CSD head 级并行调度。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源: https://github.com/ChaseLab-PKU/InstAttention
  - 框架使用全过程（以 OPT-13B on 1×A6000 + 1×InstCSD, batch_size=64, input/output=1024 tokens 为例）：
    1. **请求到达与 Batch 构造**：Host 端 InstAttention Scheduler 接收用户推理请求，将所有请求的 prompt tokens 打包为 batch（bs=64, 1024 tokens each），送入 InstGPU（A6000）。
    2. **Prefilling Phase（全在 GPU）**：Layer 0 的 QKV Proj. 计算（GeMM）→ 生成 Q,K,V tensors（每 token 的 K,V 即初始 KV cache）。K,V tensors 作为 KV cache → InstGPU driver 发起 P2P DMA：将 Layer 0 的 KV cache 从 GPU VRAM 传输到 InstCSD 的自定义逻辑地址（包含 batch=0~63, layer=0, token=0~1023 段）。传输与 Layer 1 计算叠加（pipeline）。Attention (Logit+Attend)、O Proj.、FFN 仍在 GPU 执行。循环 N 层完成 prefilling 并输出第一个 token。
    3. **Decoding Phase（GPU-CSD 协作）**：GPU 执行 QKV Proj.（仅新 token 的 Q,K,V，轻量）→ 将新的 q,k,v 向量通过 P2P DMA 发送至 InstCSD → InstCSD 执行 SparF Attention（q vector→argtopk→双步加载 K[:,i]/K[j,:]/V[j,:] from flash→GeMV+Softmax 计算注意力→输出 attention out）→ 将 attention output 传回 GPU → GPU 继续 O Proj. + FFN → 生成新 token。KV cache 保持在 InstCSD flash 中直到 GC。
    4. **KV Cache Writing（InstCSD 内部）**：新生成的 k,v 向量（每 token 128 FP16×40 heads = ~10KB）写入 InstCSD DRAM group buffer。Buffer 中累积足够 heads 合并满一个 flash block（如 40 heads × 128 dim = 5120 FP16 = 10KB/token, 需多个 token 凑满 4KB page × 数百 pages/block）后，后台批量写回 flash。Batched writing 避免小写导致的 write amplification。Token-indexed mapping 将同 token 的不同 head group 写入同一 block 的不同 page。
    5. **Multi-CSD 扩展（若有）**：若配置 4 个 InstCSD，40 个 attention heads 分配为 10 heads/CSD。GPU 将各 head 的 q,k,v 分别路由到对应 CSD，各 CSD 独立执行 SparF Attention → 各自 attention output 传回 GPU → GPU concatenate 全部 40 heads → 继续后续计算。
    6. **Garbage Collection**：Host Scheduler 维护 job list（含 token number 和是否 outdated 标记）。InstCSD idle 时（available page budget < 阈值），Scheduler 发送 reclaim(task_id) 命令 → InstCSD FTL 通过 token-indexed L2P table 查找该 task 的所有 KV cache 物理地址 → 以 LRU 顺序 erase flash block。由于 KV cache 为 append-only 顺序写入（从不 in-place 修改），无数据碎片，GC 仅需擦除无前台干扰。
    7. **Performance 输出**：端到端 throughput = total decoded tokens / wall-clock time。InstA-SparF (bs=256) 相比 FlexGen (bs=64) 吞吐提升 up to 11.1×。InstA (dense, bs=256) 相比 FlexGen (bs=32) 提升 10.5×（2 CSDs 场景）。

## 72-ALISA_Accelerating_Large_Language_Model_Inference_via_Sparsity-Aware_KV_Caching.pdf

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：ALISA在FlexGen基础上实现三阶段token级动态KV缓存调度。Phase I (GPU Caching): KV tensors全部fit在GPU内存中，仅GPU缓存无CPU访问。Phase II (GPU-CPU Caching): KV tensors超出GPU容量，按token粒度split——局部静态token的KV保留在GPU（减少CPU访问因局部token固定），全局动态token的旧KV offload到CPU并按需reload。Phase III (Recomputation-Caching): 序列长度超出某阈值后，最旧的KV tensors从CPU删除，仅在GPU核需要时重新计算（因recompute时间<CPU→GPU reload时间）。Phase transition由offline优化问题求解：最小化总执行时间 T = ΣT_j^c(compute) + ΣT_j^m(α)(GPU-CPU transfer) + ΣT_j^r(β)(recompute)，决策变量为{α(of fload ratio), β(recompute ratio), p1(p hase I→II步数), p2(p hase II→III步数)}。求解分两子问题：数据传输问题（由硬件容量/带宽约束求解）和计算问题（offline profiling + greedy search）。调度结束offline进行，无在线开销。
  - 实验比较：(1) 端到端throughput (tokens/sec): ALISA (80% KV sparsity) vs DeepSpeed-ZeRO / HuggingFace Accelerate / FlexGen / vLLM，模型OPT-6.7B/13B/30B、LLaMA-7B/13B/33B on V100 GPU和H100 GPU，batch size 4–64，输入128/输出512；(2) LLM推理分解：ALISA vs FlexGen各阶段执行时间和GPU/CPU内存使用对比；(3) 重计算影响（Phase III）: recomputation on/off对执行时间的影响；(4) Ablation study: SWA alone / SWA+DynamicScheduling / SWA+DS+KVCompression的吞吐量对比。

- 硬件平台是什么，配置是什么。
  - GPU: NVIDIA Tesla V100 (16/32 GB HBM) 和 NVIDIA H100 (80 GB HBM)。30B级别模型仅H100。
  - CPU: 2.60 GHz Intel Xeon, 128 GB DRAM。PCIe CPU-GPU带宽20 GB/s。

- 开源Serving框架是什么。修改了什么。
  - 框架: FlexGen (https://github.com/FMInference/FlexGen) 作为底层offload框架 + HuggingFace Transformers。
  - 修改: (1) 将FlexGen的head-level静态KV tensor offload策略替换为token-level动态三阶段调度（原FlexGen静态offload比例求解offline linear programming）；(2) 以layer-wise方式管理KV tensor的GPU/CPU内存分配；(3) 集成SWA的稀疏token索引选择，在Phase II将局部静态token固定保留在GPU、全局动态旧token offload到CPU；(4) 添加Phase III的KV tensor删除+GPU重计算逻辑；(5) 集成INT8 channel-wise KV compression（quantize: x_quant=round(x/λ+z), dequant: x=λ(x_quant-z)）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源: 论文未提供独立开源链接。基于FlexGen和HuggingFace Transformers实现。
  - 框架使用全过程（以OPT-30B on 1×H100 80GB, batch_size=64, input=128, output=512, 80% KV sparsity为例）：
    1. **Offline Phase**: 对OPT-30B的每个attention layer，profiling不同配置下的compute和recompute时间，结合硬件约束求解优化问题得到{α, β, p1, p2}参数。
    2. **Prefill阶段**: 一次性处理全部128个input tokens，生成初始KV tensors并缓存到GPU。序列长度s=128, GPU memory中存储完整dense KV。
    3. **Phase I (decoding step 1 to p1=~80)**: KV tensors总量≤GPU memory容量。每步：从GPU memory load KV→计算SWA attention（gather稀疏token indices→QK^T→softmax→×V）→存储新生成的K_j/V_j到GPU。无CPU访问。
    4. **Phase II (step p1 to p2=~240)**: KV总量超出GPU容量。每步：CPU→GPU load全局动态token对应的旧KV（按SWA indices）→GPU→core load所有所需KV→SWA attention compute→GPU store新K_j/V_j→将K_j^α/V_j^α（of fload ratio α部分）从GPU store到CPU（局部静态token保留GPU）。全局动态token需从CPU频繁reload，有I/O开销。
    5. **Phase III (step p2 to 512)**: 序列更长后reload CPU的KV开销>recompute开销。每步：CPU中删除最旧的K_j^β/V_j^β → 无需CPU→GPU reload这些token → 改为GPU重计算这些token的K/V值→ SWA attention。虽增加compute但省去大量CPU I/O。
    6. **KV Compression**: 所有offloaded KV tensors经INT8 channel-wise量化（λ=(max-min)/255, z=round(-2^8·min/(max-min))），在CPU内存中存储INT8格式（大小减半），GPU计算前dequantize到FP16。
    7. **Throughput计算**: end-to-end execution time / total generated tokens。ALISA相比FlexGen加速1.4~3.0×（batch越大加速越显著），相比vLLM加速最高1.9×（大batch大模型时资源受限场景）。

## 64-POD-Attention- Unlocking Full Prefill-Decode Overlap for Faster LLM Inference.pdf

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：POD-Attention 集成到 Sarathi-Serve（基于 vLLM 的 hybrid batching LLM serving 系统），实现端到端的 prefill+decode attention 融合推理。Sarathi-Serve 原生使用 hybrid batching（每 iteration 含一个 prefill chunk + 多个 decodes），但 linear 操作已融合而 attention 仍用独立 prefill/decode kernel 串行执行。POD-Attention 替换 Sarathi-Serve 的 attention backend，使 prefill 和 decode attention 在 GPU SM 内并发计算，complete the fusion stack（linear + attention 均融合），提升 GPU 资源利用率和端到端 serving 性能。
  - 实验比较：(1) **Offline inference throughput**：Sarathi+POD vs Sarathi（FlashAttention kernel） vs vLLM original（prefill-prioritized scheduling），模型 Yi-6B/Llama-2-7B/Llama-3-8B，context 16K tokens；(2) **Online inference latency**：TTFT、TBT、end-to-end request latency（P50/P99），以及 generation stall 比例（200ms/500ms threshold），Llama-3-8B 在两个 workload（internal enterprise workload ~10.5K mean CL、arXiv-Summarization ~9.5K mean CL）上，QPS=1.1–1.2；(3) **Chunk size sensitivity**：Sarathi+POD 在 chunk size 1024/1536/2048 下的 TTFT-TBT tradeoff；(4) **P:D ratio sensitivity**：prefill:decode token ratio 从 8 到 24 变化时 Sarathi vs Sarathi+POD throughput 对比。

- 硬件平台是什么，配置是什么。
  - GPU：Yi-6B on 1×NVIDIA A100 80GB；Llama-2-7B 和 Llama-3-8B on 2×NVIDIA A100 80GB（tensor parallelism）。
  - x86 机器，Ubuntu 22.04。

- 开源Serving框架是什么。修改了什么。
  - 框架：Sarathi-Serve (https://github.com/microsoft/sarathi-serve)，基于 vLLM (https://github.com/vllm-project/vllm)。
  - 修改：(1) 将 Sarathi-Serve 的 attention backend 从默认的 FlashInfer batched attention（FI_Batched，用 prefill kernel 同时处理 prefill+decode）替换为 POD-Attention kernel；(2) POD-Attention kernel 在 Sarathi-Serve 的每个 iteration（含一个 prefill chunk + 若干个 decodes）中被调用，替代原有的 prefill attention + decode attention 两次独立 kernel 调用。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：POD-Attention kernel 代码位于 https://github.com/microsoft/vattention/tree/main/pod_attn；Sarathi-Serve 开源。Docker: rnp1910/pod_attention:asplos_25_pytorch_run。
  - 框架使用全过程（以 Llama-3-8B on 2×A100，arXiv-Summarization workload 为例）：
    1. **请求到达**：Client 发送请求（arXiv 摘要生成，输入 ~9.5K tokens mean）。Sarathi-Serve 的 scheduler 将请求的 prefill tokens 划分为 chunk_size=1024 的多个 chunk。
    2. **Hybrid Batch 构造**：每个 iteration，scheduler 选取 1 个新的 prefill chunk（1024 tokens）与当前所有 ongoing decode requests（batch size ~60–80，每个 decode 处理当前上下文 1 token）组合为一个 hybrid batch。
    3. **Linear 操作（已融合）**：QKV projection、output projection、FFN 等线性层对 hybrid batch 内的 prefill + decode inputs 融合计算——模型权重一次加载至 SM，同时应用于 prefill 和 decode tokens（减少 HBM 访问）。
    4. **Attention 操作（POD-Attention kernel）**：调用 POD-Attention fused kernel 替代原本分两次的 prefill/decode attention kernel 调用。Kernel 内部 SM-aware CTA scheduling 将 prefill（chunk_size=1024，compute-heavy）和 decode（batch_size ~60–80，memory-heavy）CTAs 按 proportional policy 分配至各 SM 并发执行。关键：prefill tile 使用大 QSL tile（128）+ limited KV splits（≤2 waves）；decode tile 使用小 QSL tile（16）最小化 tensor core 占用；virtual decode CTA 平衡 shared memory。
    5. **KV Cache 管理**：vAttention [47] 动态管理 KV cache（替代 PagedAttention），避免 vLLM 的 page table 开销。
    6. **Iteration 循环**：完成一 iteration 的 linear + attention 计算后，decode 分各 request 采样生成 1 个新 token → 更新 KV cache → 检查 completion（EOS/reach max tokens）→ 下一 iteration 构造新 hybrid batch。
    7. **性能指标采集**：TTFT = 请求第一个 token 生成的端到端时间（含排队 + 全部 prefill iterations）；TBT = 相邻 decode token 间的时间（反映每 iteration latency）；Request Latency = 请求从到达到完整输出完成的总时间；Throughput = requests/minute（offline）或 QPS（online）。
    8. **执行命令**：`make figure12`（offline throughput 实验，~9 hours），`make table6`（online latency experiment，~4 hours）。输出 CSV 到 `output/` 目录。

## 58-ExeGPT- Constraint-Aware Resource Scheduling for LLM Inference.pdf

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：ExeGPT在FasterTransformer (FT) 基础上构建约束感知的LLM推理调度系统，包含四个组件：(1) XProfiler——对单层encoder/decoder分别测量attention kernel和其他层的执行时间，sweep所有batch size×sequence length和tensor-parallel degree组合，同时测量tensor/pipeline parallelism的同步开销；(2) XSimulator——利用输入/输出序列长度概率分布PE(S)和PD(S)，结合profiling结果构建执行timeline，估计给定配置下的throughput和latency；(3) XScheduler——基于branch-and-bound的单调优化算法，在四个控制变量（batch size BE/BD、decoder micro-batch Bm、partial tensor parallelism TP、encoding frequency FE）的搜索空间中寻找满足latency bound且最大化throughput的最优配置；(4) XRunner——扩展FT实现early-termination（completed queries的key/value cache compact）和WAA Scheduling的跨GPU KV cache transfer（GPU→CPU→GPU memory）。两种调度策略：RRA（Round-Robin Allocation，GPUs轮询分配encoder/decoder层，周期性运行encoding后固定ND次decoding迭代）和WAA（Workload-Aware Allocation，按计算量/内存比例分配GPU给encoder和decoder，异步解耦执行；子类型WAA-C按计算时间、WAA-M按内存消耗分配）。运行时动态调整encoder batch size保持workload consistency。
  - 实验比较：(1) 端到端throughput：ExeGPT (RRA/WAA中较优者) vs FT，small/mid LLMs (T5 11B, OPT 13B, GPT-3 39B/101B) 三个task×四个latency bound；large LLMs (GPT-3 101B/175B/341B) 三个task×四个latency bound；(2) 基线系统横向比较：FT vs DSI vs ORCA(vLLM iteration-level调度) vs vLLM，OPT 13B on 4×A40；(3) WAA memory overhead：FT vs WAA encoder/decoder GPUs memory usage对比；(4) real-world datasets (WMT/Alpaca/CNN) 评估；(5) 序列分布变化下的robustness：改变avg/std/skewness后non-adjusted vs optimal schedule的throughput/latency；(6) profiling/scheduling/(re-)deploying cost；(7) monotonicity验证和case study of trade-off。

- 硬件平台是什么，配置是什么。
  - A100 cluster: 16 GPUs (2×NDm A100 v4 VMs on Azure, 每VM 8×A100 80GB), NVLink 3.0 intra-node, 1.6Tb InfiniBand (8×200Gbps Mellanox HDR) inter-node。
  - A40 cluster: 48 GPUs (6 machines, 每machine 8×A40 48GB), PCIe 4.0×16 intra-node, 100Gb InfiniBand inter-node。
  - CPU: AMD EPYC 7313 (profiling用)。

- 开源Serving框架是什么。修改了什么。
  - 框架: NVIDIA FasterTransformer [23] (开源，https://github.com/NVIDIA/FasterTransformer)。修改：
    1. **Early-termination + KV cache compaction**: 在FT的decoding iteration中检测completed queries，提前终止其计算，compact其KV cache entries（移除已完成的entry使后续batch更紧凑）。
    2. **WAA跨GPU KV cache transfer**: 实现从encoding GPU memory→CPU memory→decoding GPU memory的KV cache拷贝路径，以支持WAA调度中encoder和decoder在不同GPU上的解耦执行。
    3. **Pipeline parallelism的computation/communication overlap优化**。
    4. **XRunner调度执行引擎**: 读取XScheduler产出的optimal schedule配置参数（batch sizes、TP degree、ND等），在运行时强制enforce该schedule。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源: 论文未提供开源链接（ASPLOS '24论文，截至评估时未发现公开代码仓库）。
  - 框架使用全过程（以GPT-3 175B on 16×A100, WAA Scheduling, 翻译任务为例）:
    1. **离线Profiling (XProfiler)**: 对GPT-3的单层decoder，sweep attention kernel和其余层的执行时间——batch size [1, 2, 4, ..., 256]×sequence length [1, ..., 2048]，同时sweep tensor-parallel degree [1, 2, 4, 8]。记录各配置下的kernel执行时间以及all-reduce同步开销。一次profiling per model+GPU cluster，耗时<2小时。
    2. **序列分布获取**: 从WMT翻译数据集采样，拟合输入/输出sequence length的截断正态分布参数——PE(S): (μ=128, σ=81), PD(S): (μ=128, σ=68)。
    3. **调度优化 (XScheduler)**: 输入latency bound LB（如99th pctl序列长度对应的latency），在控制变量空间（BE ∈ [4,256], BD = BE·SD, Bm ∈ [1,BD], TP degree ∈ [1,8], applied GPU count ∈ [1,16]）执行branch-and-bound搜索。算法从初始block B0=[(amin, bmin), (amax, bmax)]开始，每次split block为两个子block，计算子block边界点的(XSimulator estimated) latency/throughput，剪枝不满足LB的block，保留有希望block进priority queue。WAA policy按计算时间比例 CE/(CE+CD) 分配GPU给encoder/decoder。XScheduler分别对RRA和WAA运行算法，选throughput更高者。scheduling耗时1-5分钟。
    4. **模型部署 (XRunner)**: 按schedule从SSD加载模型到GPU（WAA需在encoding和decoding GPU上各存一份模型副本——memory overhead ~29% for GPT-3）。若encoder和decoder分配在不同GPU，通过InfiniBand连接不同pipeline stage。
    5. **在线推理执行**: 
       - Client发送翻译请求（如英译德），预处理为输入token序列。
       - Encoder GPU: 以BE为batch收集BE个输入序列，执行encoding（attention kernel + feedforward），产出encoder hidden states。WAA动态调整encoder batch size使workload (Σinput lengths) 保持在average workload阈值内。
       - KV cache transfer: encoder hidden states从encoding GPU memory→CPU pinned memory→decoding GPU memory。
       - Decoder GPU: 以BD=BE·SD为batch合并新encoding结果和已有decoding batch，执行decoding迭代（self-attention with KV cache + cross-attention on encoder outputs + feedforward）。Early-termination: 每个decoding iteration后检查completed queries，compact其KV cache。Decoder micro-batch机制将BD拆为Bm个micro-batch以减少pipeline bubble（如BD=32, Bm=8, 4 micro-batches of 8 each pipeline-overlapped）。
       - Partial tensor parallelism: 若TP=4 applied to 8 GPUs，则前8 GPUs以TP=4执行（每层split到4 GPUs并行，2次all-reduce sync），剩余GPUs以TP=1执行。
       - Decoding持续至所有queries completed（max output length）或达到ND次迭代（RRA情况下触发新一轮encoding）。
       - 输出tokens经detokenization得到翻译文本，返回client。

## 52-Pimba- A Processing-in-Memory Acceleration for Post-Transformer Large Language Model Serving..pdf

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Pimba系统将LLM serving分为prefill和generation两阶段。Prefill阶段所有操作在GPU执行（可重组为compute-intensive GEMM）。Generation阶段将state update和attention操作offload到PIM执行，其余操作（FFN、causal conv、discretization等）保留在GPU。软件栈基于HBM-PIM [40]设计：(1) Pimba device driver分配物理连续内存块以支持PIM操作；(2) 自定义GPU kernel为每个Pimba操作issue必要的PIM命令并计算PIM内存地址；(3) CUDA编程模型扩展API以issue Pimba的5条自定义DRAM命令（ACT4/REG_WRITE/COMP/RESULT_READ/PRECHARGES）；(4) 编译时API lowering为自定义DRAM命令；(5) kernel注册为PyTorch自定义操作，用户通过标准API调用。GPU和PIM之间以blocked方式交替执行（数据依赖），PIM操作具有确定性时序，GPU可连续issue多条PIM命令。intra-Pimba通信通过GPU片外内存直连；inter-Pimba通信通过NVLink支持pipeline/tensor parallelism。
  - 实验比较：(1) 端到端生成吞吐：Pimba vs GPU vs GPU+Q vs GPU+PIM，batch 32/64/128，small scale (2.7B/7B) 和 large scale (70B)；(2) 延迟分解：GPU/GPU+Q/GPU+PIM/Pimba各操作的归一化延迟（state update/attention/discretization/causal conv/GEMM/communication/others）；(3) 能耗分解；(4) 与NeuPIMs对比延迟和内存使用；(5) H100平台适配验证。

- 硬件平台是什么，配置是什么。
  - Small scale: 1×NVIDIA A100 80GB, 40 HBM2E PIM stacks @1,512MHz。Large scale: 8×A100 via NVLink3 600GB/s, tensor parallelism。
  - 也验证H100: 40 HBM3 PIM stacks @2.626GHz, SPU @657MHz, NVLink4 900GB/s。
  - HBM时序参数: tRP=14, tRAS=34, tCCD_S=2, tCCD_L=4, tWR=16, tRTP_S=4, tRTP_L=6, tREFI=3900, tFAW=30。

- 开源Serving框架是什么。修改了什么。
  - 框架: PyTorch + CUDA扩展。修改: 新增自定义GPU kernel（PIM kernel）issue Pimba DRAM命令，扩展CUDA API支持ACT4/REG_WRITE/COMP/RESULT_READ/PRECHARGES命令，注册为PyTorch自定义操作。PIM device driver管理物理连续内存分配。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源: https://github.com/casys-kaist/pimba, uv + gcc编译, 需要Ampere+ GPU ≥24GB。
  - 框架使用全过程:
    1. **输入**: 用户通过PyTorch调用模型推理，如model.generate(input_ids)。
    2. **Prefill**: GPU执行QKV生成、state update（重组为GEMM）、FFN等全操作。此阶段PIM不参与。
    3. **Generation (每个token)**:
       - GPU执行QKV生成、causal conv、discretization等，产出operands (d_t, k_t, v_t, q_t)。
       - GPU kernel调用PIM API——issue REG_WRITE命令将operands以MX8格式写入PIM寄存器（若host不支持MX8，memory controller中的Quantization Unit转换）。
       - GPU issue ACT4命令激活state数据所在DRAM row（4 banks一起激活，遵守tFAW约束）。在ACT4的空闲间隙插入REG_WRITE传输operands。
       - GPU issue COMP命令——PIM内SPU执行4-stage流水线：Stage1读取state sub-chunk→Stage2并行decay+outer product→Stage3 state update→Stage4 dot product + writeback。
       - 因数据依赖，GPU阻塞等待PIM完成。GPU issue RESULT_READ在PRECHARGES的tRP间隙内取回结果。
       - GPU收到partial sums后执行accumulation得到最终output vector。
       - GPU继续执行FFN等剩余层操作。
    4. **Attention操作**: 分两阶段——score阶段PIM做q·K dot product→GPU softmax→attend阶段PIM做score·V乘法累加。
    5. **输出**: GPU汇总所有层输出，经LM head得到下一个token。

## 24-Accelerating LLM Serving for Multi-turn Dialogues with Efficient Resource Management.pdf (FlashGen)

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：FlashGen在vLLM之上构建两个核心技术——FlashGen-Cache（多级KV cache: GPU→CPU→SSD）和FlashGen-Sched（请求重排序调度），加速多轮对话LLM推理。FlashGen-Cache的核心设计包括：(i) 推理过程中主动将生成的attention KV异步拷贝到host memory（write-back机制），GPU端的已完成请求KV标记为reclaimable但保留为cache；(ii) 三层缓存命中路径：GPU hit→直接使用；GPU miss→从host memory按layer-by-layer pipeline恢复（传输第L+1层KV时并行执行第L层计算）；host miss→从SSD staging到host memory再传输，或动态选择recomputation（当SSD检索延迟大于重算时）；(iii) SSD staging策略：用户请求到达时预加载KV到host memory（Waiting状态），利用前序请求执行时间隐藏SSD加载延迟；(iv) batch-aware KV restoration：当batch仅含generation phase时，排除KV未完全加载的请求，避免拖慢整个batch。FlashGen-Sched的核心设计：(i) 贪婪reordering——当队首请求因GPU内存不足无法调度时（head-of-line blocking），跳过它提取后续可调度的请求（promoted requests）；(ii) starvation-free——track promoted request占用的内存，一旦前序请求完成释放内存+promoted内存之和足以容纳deferred request，就preempt promoted request、调度deferred request，随后利用FlashGen-Cache从host memory恢复preempted request的KV（避免重算）——本质上将promoted request的内存视为"可回收的free space"。
  - 实验比较：以vLLM为基线，消融比较FlashGen-Sched、FlashGen-Cache、FlashGen三个设计选项，同时与CachedAttention（USENIX ATC '24）对比。评估指标：throughput (tokens/s) vs normalized latency (ms/token)、P95 TTFT（time to first token）、TPOT（time per output token）CDF、KV cache hit rate分解、GPU memory utilization分解、prompt phase时间分解（重算 vs KV传输）。

- 硬件平台是什么，配置是什么。
  - Azure Standard_NC48ads_A100_v4：2× NVIDIA A100 80GB GPU（tensor parallelism用于30B/70B模型）
  - Host: 440GB DRAM（默认分配~50%即224GB用于KV cache）
  - Storage: 2× 960GB NVMe SSD（RAID-0，提升读写带宽）
  - 单GPU用于OPT 13B、Llama-2 13B；双GPU tensor parallelism用于OPT 30B、Llama-2 70B

- 开源Serving框架是什么。修改了什么。
  - 基于vLLM（开源，https://github.com/vllm-project/vllm），已集成PagedAttention和iteration-level scheduling。
  - FlashGen的修改包括：
    1. **KV Cache管理系统（FlashGen-Cache）**：新增multi-level cache hierarchy manager——GPU memory层（reclaimable/completed KV cache）、host memory层（write-back + retrieval controller）、SSD层（background async archive + staging）。实现proactive write-back：每层decoder layer生成新KV后异步拷贝到host memory。实现pipelined restoration：从host到GPU按layer-by-layer传输，利用当前layer计算时间隐藏下一layer KV传输。实现SSD staging controller：请求到达时提前从SSD加载KV到host memory staging区（标记non-reclaimable），等待调度时传输到GPU。
    2. **Scheduler修改（FlashGen-Sched）**：替换vLLM的FCFS调度为reordering调度——新增promoted/deferred请求状态机，preemption逻辑，memory tracking（将promoted请求占用内存计入free space计算）。Starvation prevention: 当deferred请求变为可调度时，preempt promoted请求并调度deferred请求。
    3. **Attention Kernel修改**：修改Flash-Attention（prefill）和Flash-Decoding（generation）kernel，支持非连续物理内存中的KV block——因为PagedAttention中history KV的物理page可能不连续。修改方式与FlashInfer库类似。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文未明确说明FlashGen代码是否开源，未提供开源链接。vLLM为开源框架（Apache 2.0）。
  - Serving框架输入到硬件执行全过程（以OPT 30B on 2×A100, ShareGPT多轮对话为例）：
    1. **输入**：客户端（模拟多轮对话session）发送第N轮请求——prompt包含当前轮用户输入+前N-1轮的完整对话历史（prompt和generation）。Tokenizer将文本转为token ids。FlashGen Scheduler接收请求，根据GPU空闲内存决定是否立即调度。
    2. **KV Cache Lookup（FlashGen-Cache）**：按session ID查找前序轮的history KV——L1: GPU cache（completed requests的reclaimable KV）→ L2: host memory（write-back copy）→ L3: SSD（archived KV）。若L3 miss则fallback到recomputation。
    3. **SSD→Host Staging（若需）**：若KV仅在SSD，KV manager启动后台DMA从RAID-0 NVMe SSD读取KV到host memory staging区，标记为non-reclaimable。该操作与当前running request的推理并行。
    4. **Prefill Phase（Prompt Processing）**：当前轮prompt tokens + history KV送入模型。若history KV在host memory：Pipeline restoration——layer 1 KV从host→GPU传输时，layer 1开始执行attention（用已传输的KV）；layer 2 KV同时在后台传输。使用修改后的Flash-Attention kernel（支持non-contiguous KV blocks，类似FlashInfer的分段attention实现）。若GPU memory中已有（cache hit），直接使用。若需要recomputation（fallback），重算所有history tokens的KV。
    5. **KV Write-back**：每层decoder layer生成的new KV在写入GPU KV cache的同时，异步DMA拷贝一份到host memory（write-back bit标记是否需要进一步写到SSD）。GPU端该请求完成后KV保留为reclaimable cache。
    6. **Generation Phase（Autoregressive Decoding）**：每step生成1个token，使用修改后的Flash-Decoding kernel。若当前batch仅含generation phase且某请求的KV未完成加载，Scheduler将该请求移出当前batch单独等待，避免拖慢整体TPOT。
    7. **Scheduling with Reordering（FlashGen-Sched）**：若GPU内存不足以容纳队首请求的长prompt（head-of-line blocking），Scheduler跳过它提取后续短prompt请求执行（promoted request），闲置内存得以利用。当队首请求的前序请求完成（释放内存），且累积free space（含promoted request可回收空间）足以容纳deferred request时，Scheduler preempt promoted request（其KV已backup到host memory），调度deferred request。Promoted request稍后从host memory恢复KV后resume执行，无需重算。
    8. **SSD Background Archive**：Host memory中的KV按FIFO策略周期性异步写入SSD（background thread，非关键路径）。Write-back bit标记的KV在host memory不足时evict到SSD。
    9. **输出**：生成的token经detokenizer转为文本返回客户端。Session完成后GPU/CPU/SSD中的KV可保留供后续turn使用。SSD缓存使session间隔较长的多轮对话也能避免重算。

## 86-Oaken- Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization.pdf

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Oaken作为LLM serving加速方案，通过KV cache量化扩展serving系统的有效内存容量和带宽，支持更大batch size和更长context length。Oaken集成page-based memory management（类似vLLM的PagedAttention），KV cache以page为单位管理，量化后的KV cache占用更少物理内存（平均4.4-bit vs FP16 16-bit），使系统能在相同加速器内存下服务更多并发请求。
  - Oaken在serving层面的核心贡献：(i) 通过4.4-bit average KV cache，内存容量有效扩展约3.6×；(ii) attention操作从bandwidth-bound变为compute-bound（因KV cache数据量减少），提升GPU/加速器利用率；(iii) 支持online per-token量化，量化操作与token生成pipeline重叠；(iv) MMU管理量化page的元数据（thresholds, scales），支持page-level的量化/反量化状态追踪。
  - 实验比较：vLLM（FP16）、KIVI（GPU KV量化）、QServe（GPU KV量化）。评估指标：throughput (token/sec) at context length 2K，sweep batch size 16-256。

- 硬件平台是什么，配置是什么。
  - NVIDIA A100（312 FP16 TFLOPS, HBM 80/160GB, 2.0 TB/s）
  - Oaken-HBM加速器（270 TFLOPS, HBM 80GB, 2.0 TB/s, Quant/Dequant Engine integrated）
  - Oaken-LPDDR加速器（270 TFLOPS, LPDDR 256GB, 1.1 TB/s）

- 开源Serving框架是什么。修改了什么。
  - Oaken基于自研加速器（LPU类架构），非修改vLLM等开源GPU serving框架。Oaken serving系统的核心修改：(i) page-based KV cache memory management with quantization-aware page allocation——每page存储encoded KV数据及量化元数据；(ii) 集成Quant Engine和Dequant Engine硬件模块到memory data path——KV cache写入时经Quant Engine量化，读取时经Dequant Engine反量化；(iii) MMU（Memory Management Unit）with management table追踪每page的量化状态（group thresholds, scale factors, encoding format）。
  - 开源accuracy evaluation代码（https://github.com/casys-kaist/oaken）在GPU上模拟Oaken量化算法的精度行为，不包含完整serving系统和硬件模拟。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：accuracy evaluation code开源（https://github.com/casys-kaist/oaken），serving系统硬件实现未开源。Oaken的商业化由HyperAccel公司推进（LPU加速器产品）。
  - Serving框架输入到硬件执行全过程（以Llama2-7B处理batch=64, context=2K请求为例）：
    1. **输入**：多个用户请求到达→tokenizer将prompt转为token ids→Oaken serving scheduler将请求按page-based memory管理分组为batch。Scheduler根据当前可用KV cache pages数量决定是否接纳新请求。
    2. **Prefill阶段**：batch中所有prompt tokens通过模型forward——每层decoder layer：MPU计算Q·K^T·V attention→VPU计算FFN→输出hidden states。同时，K/V值流经Quant Engine：Threshold Comparator按offline预计算的T_lo/T_hi分组→Scale Calculator计算per-group scale→Quantizer执行INT4/INT5量化→Dense-and-Sparse Encoder融合为8-bit对齐编码→通过DMA写入Device Memory的KV cache pages。MMU更新management table记录每page的{thresholds, scales, group_config}。
    3. **Generation阶段**：每步生成1个new token→计算attention时，MMU查management table获取所需pages的量化参数→DMA从Device Memory读取对应pages的encoded KV→Dequant Engine：Decomposer按group flag拆分dense/sparse→Min&Max恢复scale→Dequantizer还原FP16→送入MPU计算attention score。同时new token的K/V经Quant Engine量化后追加写入KV cache pages。
    4. **输出**：生成的token ids经detokenizer还原为文本返回用户。当请求完成（生成EOS或达到max length），对应KV cache pages被释放回page pool。

## 38-SpecInfer- Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification.pdf

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：SpecInfer是一个基于tree-based speculative inference和verification的LLM serving系统。核心设计包括：(1) Learning-based Speculator——使用一个或多个Small Speculative Models (SSMs)预测LLM输出，通过expansion-based（单个SSM的top-k展开）和merge-based（多个SSM的adaptive boosting调优后合并）两种方式构造token tree；token tree中每个节点代表一条候选token序列；(2) Token Tree Verifier——将LLM用作token tree verifier而非incremental decoder，通过tree-based parallel decoding机制在一次LLM decoding step中并行验证token tree中所有候选序列的正确性，同时利用topology-aware causal mask保证因果关系；(3) Request Manager——CPU端实现request scheduling（adapt自Orca的iteration-level调度 + continuous batching）、token tree merge、以及verification逻辑；(4) 分布式LLM serving——SSM使用data parallelism分布到多GPU，LLM使用tensor model parallelism + pipeline model parallelism（即Megatron-LM的hybrid parallelization策略）。加速原理：通过token tree提高每步验证tokens数，减少LLM参数memory access次数和端到端推理步数。
  - 实验比较：与vLLM、HuggingFace Text Generation Inference (TGI)、FasterTransformer对比分布式LLM inference延迟；与FlexGen对比offloading-based LLM inference延迟。同时与SpecInfer自身的incremental decoding模式和sequence-based speculative inference模式做消融实验。评估指标：per-token latency (ms/token)。

- 硬件平台是什么，配置是什么。
  - 2× AWS g5.12xlarge instances，每台配备4× NVIDIA A10 24GB GPU、48 CPU cores、192 GB DRAM。节点间通过100 Gbps以太网连接。
  - 单节点内GPU使用tensor model parallelism；跨节点使用pipeline model parallelism。
  - 单GPU场景：1× A10 24GB用于LLaMA-7B；多GPU场景：4× A10用于OPT-30B（1节点）；8× A10用于LLaMA-65B（2节点）。
  - Offloading场景：单A10 24GB GPU，模型参数offload到CPU DRAM（192GB），按需加载到GPU HBM。

- 开源Serving框架是什么。修改了什么。
  - 基于FlexFlow（开源，https://github.com/flexflow/FlexFlow），一个分布式多GPU DNN计算runtime。SpecInfer artifact开源在 https://github.com/goliaro/specinfer-ae（Apache License v2.0）。
  - 主要修改：
    1. **Speculator模块（新增）**：实现expansion-based token tree construction——根据预定义的expansion configuration（如⟨1,1,3,1,1,1,1,1⟩，在第3步展开3个token），从SSM采样top-k tokens构造token tree。实现merge-based token tree construction——使用adaptive boosting对多个SSM做无监督collective boost-tuning，将多个SSM的输出合并为统一token tree。实现multi-step speculative sampling (MSS) verification算法，支持stochastic decoding下provably preserving LLM的generative performance。
    2. **Token Tree Verifier模块（新增）**：实现tree-based parallel decoding——将tree attention计算fused到单个CUDA kernel，使用topology-aware causal mask（非传统sequence topology的causal mask）。使用depth-first search order遍历token tree更新shared KV-cache，避免多序列各自维护独立KV-cache的冗余。
    3. **Request Manager（新增）**：CPU端实现iteration-level request scheduling（adapt自Orca的continuous batching）、token tree merge、token verification逻辑。
    4. **CUDA Kernel（修改FasterTransformer）**：基于FasterTransformer的attention kernel修改为支持tree-based parallel decoding。每个thread block计算单个request单个attention head，query tensor加载到shared memory，各线程计算query/key product的一段。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源仓库：https://github.com/goliaro/specinfer-ae（论文artifact，含完整源码、脚本、conda环境）
  - 框架输入到硬件执行全过程（以expansion-based speculation + greedy decoding为例）：
    1. **输入**：用户发送prompt tokens序列 ℐ 到Request Manager。
    2. **Request Scheduling**：Request Manager按iteration-level调度选择pending requests，将prompt tokens分发到SSM GPU workers和LLM GPU workers。
    3. **SSM Speculation**（GPU workers）：每个SSM GPU worker加载对应的SSM（如LLaMA-68M on GPU1），以incremental decoding方式对prompt生成下一步的top-k tokens。按expansion configuration（如⟨1,1,3,1,1,1,1,1⟩）构造token tree：step0取top-1→step1取top-1→step2取top-3（展开为3个分支）→后续各取top-1。SSM-generated tokens送回Request Manager。
    4. **Token Tree Merge**（CPU）：Request Manager将多个SSM的输出token tree进行merge（Definition 3.2: token tree merge），生成统一的speculated token tree 𝒩。
    5. **Tree-based Parallel Decoding**（LLM GPU workers）：LLM参数按tensor model parallelism分布在各GPU内、pipeline model parallelism跨节点。GPU workers执行tree-based parallel decoding——将所有token tree节点tokens与新到达的请求tokens一起batch处理，使用topology-aware causal mask确保每个token只attend到其tree祖先节点。使用depth-first search order维护shared KV-cache的一致性。FasterTransformer-based CUDA kernel将整个tree attention fused到单次kernel launch。
    6. **Verification**（CPU）：LLM-generated tokens 𝒪 送回Request Manager。对于greedy decoding，VerifyGreedy从tree root开始逐节点检查child token是否匹配LLM输出；匹配则继续前进，不匹配则附加LLM输出后终止。对于stochastic decoding，VerifyStochastic（MSS算法）按tree分支依次以probability min(1, P_LLM/P_SSM)接受token，拒绝则normalize residual distribution继续采样下一SSM分支，所有SSM失败则从residual distribution直接采样（Theorem 4.2保证输出分布等价于incremental decoding）。
    7. **输出**：Verified tokens序列 𝒱 追加到response并返回客户端。若token为<EOS>则请求完成。

## 41-Splitwise_Efficient_Generative_LLM_Inference_Using_Phase_Splitting.pdf

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Splitwise将生成式LLM推理的两个阶段（prompt computation和token generation）拆分到分离的机器池上运行，实现phase-specific资源管理。核心设计包括三个组件：
    1. **两级层次化调度器**：(i) Cluster-Level Scheduler (CLS)——负责机器池管理（prompt pool / token pool / mixed pool），以Join-the-Shortest-Queue (JSQ)策略为每个请求同时分配一对prompt机器和token机器，监视各机器内存容量和pending queue变化以动态迁移机器池归属；(ii) Machine-Level Scheduler (MLS)——运行在各机器上，负责GPU内存利用率跟踪、pending queue维护和每iteration batching决策。Prompt MLS使用FCFS调度并限制总batch prompt tokens≤2048（因为超过此后throughput下降）；Token MLS使用FCFS并尽可能增大batch直至GPU内存满；Mixed MLS优先调度prompts（满足TTFT SLO），必要时preempt tokens（带age-based priority和per-request preemption上限防止starvation）。
    2. **KV-cache跨机传输机制**：prompt机器将生成的KV-cache通过InfiniBand传输到token机器。Splitwise实现两种传输策略：(i) serialized transfer——prompt phase结束后一次性传输所有层KV-cache，适用于小prompt (<512 tokens on H100)；(ii) per-layer async transfer——每层prompt计算生成该层KV-cache后立即异步传输，与下一层prompt计算重叠，隐藏传输延迟。使用MSCCL++的zero-copy one-sided put原语，prompt机器主动推送KV-cache数据，token机器通过semaphore同步（同一InfiniBand连接）。vLLM中KV-cache block-by-block传输，考虑contiguity减少传输次数。
    3. **动态机器池管理**：Mixed pool按需从prompt/token池借调机器，使用mixed continuous batching。当请求pending queue超过阈值时CLS从mixed pool或opposite pool分配机器，队列排空后归还。负载分布变化较大时支持coarse-grained机器重新划归（re-purposing）。
  - 实验比较：
    - Baseline对比：Baseline-A100（统一DGX-A100机池，mixed continuous batching）、Baseline-H100（统一DGX-H100机池，mixed continuous batching）
    - Splitwise变体：Splitwise-AA（A100 prompt + A100 token）、Splitwise-HH（H100 prompt + H100 token）、Splitwise-HA（H100 prompt + A100 token）、Splitwise-HHcap（H100 prompt + H100 token with 70% power cap per GPU）
    - 评估场景：(i) iso-power throughput-optimized集群——固定总功率budget，最大化throughput；(ii) iso-cost throughput-optimized——固定总cost，最大化throughput；(iii) iso-throughput power-optimized——固定throughput target，最小化power；(iv) iso-throughput cost-optimized——固定throughput target，最小化cost
    - 评估指标：TTFT P50/P90/P99、TBT P50/P90/P99、E2E latency P50/P90/P99、throughput (RPS)、per-token throughput、cost、power、space (机器数)
    - Robustness测试：conversation trace跑在coding-optimized集群上；Llama-70B跑在BLOOM-176B-optimized集群上

- 硬件平台是什么，配置是什么。
  - DGX-A100：8× NVIDIA A100 80GB GPU，19.5 TFLOPs，HBM 80GB，HBM bandwidth 2039GB/s，TDP 400W/GPU，NVLink 50Gbps per GPU pair，InfiniBand 200Gbps per machine
  - DGX-H100：8× NVIDIA H100 80GB GPU，66.9 TFLOPs，HBM 80GB，HBM bandwidth 3352GB/s，TDP 700W/GPU，NVLink 100Gbps per GPU pair，InfiniBand 400Gbps per machine
  - 实验环境：Microsoft Azure VMs (2× DGX-A100 + 2× DGX-H100)，InfiniBand互联
  - 大规模集群评估通过SplitwiseSim事件驱动模拟器，以piece-wise linear performance model（MAPE<3%）和communication model驱动
  - 成本参考：DGX-A100 $17.6/hr，DGX-H100 $38/hr (CoreWeave pricing)

- 开源Serving框架是什么。修改了什么。
  - 基于vLLM（开源，https://github.com/vllm-project/vllm），已集成PagedAttention和continuous batching。
  - Splitwise的修改包括：
    1. **KV-cache Transfer实现**：在vLLM中新增KV-cache跨机传输机制。使用MSCCL++（https://github.com/microsoft/mscclpp）GPU-driven communication library实现serialized和per-layer两种传输模式。Per-layer模式在各transformer layer的KV-cache计算完成后立即触发异步传输（one-sided put via InfiniBand），使用semaphore实现跨机同步。
    2. **Mixed Continuous Batching**：原版vLLM仅支持continuous batching with token preemption（会导致较高TBT），Splitwise实现state-of-the-art mixed continuous batching——允许prompt和token phase在同一forward pass中混跑，减少TBT影响。
    3. **SplitwiseSim模拟器**（新增）：独立的事件驱动集群模拟器，建模Splitwise机器池、两级调度器、MLS内存/队列、KV-cache传输延迟。输入：请求trace、SLO、performance model、集群配置。输出：per-request TTFT/TBT/E2E、机器利用率。
  - 开源情况：
    - KV-cache transfer prototype: https://github.com/vllm-project/vllm/pull/2809（PR to vLLM）
    - SplitwiseSim模拟器: https://github.com/Mutinifni/splitwise-sim
    - Production traces: https://github.com/Azure/AzurePublicDataset (AzureLLMInferenceDataset2023)
    - Artifact完整包: https://doi.org/10.5281/zenodo.11003049
    - 代码license: MIT; 数据license: CC-BY

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - Serving框架输入到硬件执行全过程（以Splitwise-HH on BLOOM-176B，coding trace为例）：
    1. **输入与路由**：客户端向推理服务发送prompt文本请求。Tokenizer将文本转为token ids。CLS收到请求后，以JSQ策略从prompt pool选出一台prompt machine（如DGX-H100 #P3），从token pool选出一台token machine（如DGX-H100 #T7），同时分配给该请求。
    2. **Prompt Phase（Prompt Machine MLP）**：Prompt machine的MLS将多个并发请求的prompt tokens按FCFS batch（总tokens≤2048限制），执行单次forward pass。BLOOM-176B的70层transformer通过tensor parallelism分布在8×H100 GPU上（每GPU内NVLink 100Gbps互联）。每层decoder依次计算self-attention（Q·K^T·V）和FFN，生成该层的KV-cache。对于大prompt（≥512 tokens），每层KV-cache计算完成后立即通过MSCCL++的one-sided put原语经InfiniBand（400Gbps）异步推送到token machine #T7；对于小prompt（<512 tokens），所有层计算完成后一次性serialized传输。Prompt machine生成第一个token返回，同时semaphore通知token machine所有层KV-cache传输完成。
    3. **KV-cache传输**（跨机）：每层KV-cache数据从prompt machine GPU HBM经GPU RDMA→NIC→InfiniBand fabric→token machine NIC→GPU RDMA→GPU HBM。Per-layer模式：layer L的KV在传输时，layer L+1的prompt计算同时进行（计算-通信重叠）。传输使用MSCCL++的zero-copy one-sided put，接收方仅等待semaphore信号无需主动recv。vLLM的KV-cache按block传输，MSCCL++对连续block合并传输减少次数。
    4. **Token Generation Phase（Token Machine MLS）**：Token machine等待KV-cache传输完成后开始autoregressive token generation。每iteration：MLS从pending queue按FCFS选择可容纳的请求（受GPU HBM 80GB限制），计算当前step的attention（用KV-cache+最新token），生成下一个token。Token MLS尽可能将更多请求batch到一起（token phase batching对TBT影响极小——batch=64仅2× TBT增长），直至GPU memory满。每生成新token后更新KV-cache并返回token。
    5. **Mixed Pool介入（高负载时）**：当pending queue超过阈值，CLS从mixed pool（或从opposite pool借调）分配机器执行mixed batching——允许prompt和token phase在同一forward pass混跑。Mixed MLS优先调度prompt（满足TTFT SLO），必要时preempt已有token phase（age-based priority防starvation）。请求完成后机器归还原池。
    6. **输出**：EOS token生成后请求完成，token machine释放该请求的KV-cache内存。所有生成token经detokenizer转为文本返回客户端。Splitwise不改变模型精度（lossless KV-cache transfer，无随机化）。

## 47-SpecEE- Accelerating Large Language Model Inference with Speculative Early Exiting

- 属于Serving调度的实现是什么？实验比较什么？
  - SpecEE 提出 **Two-level heuristic predictor scheduling engine**（两级启发式预测器调度引擎），通过离线+在线调度动态决定哪些 transformer layers 需要整合 predictor（early exiting 预测器），取代 baseline early exiting 系统中每层均部署 predictor 的做法（Llama2-7B 32 层全部署带来 ~20% inference overhead）。实验比较 SpecEE 集成到 HuggingFace、vllm、AWQ、llama.cpp 等 Serving 框架后的 speedup 和 throughput。Cloud 场景：SpecEE+HF 比 HF 平均 speedup 1.27×-1.43×（Llama2-7B/13B/70B on A100, RTX 4090），SpecEE+vllm 比 vllm 1.12×-1.14×，SpecEE+AWQ 比 AWQ 1.09×-1.14×。PC 场景：SpecEE+llama.cpp 比 llama.cpp 1.25×，SpecEE+PowerInfer 比 PowerInfer 1.15×。

- 硬件平台是什么，配置是什么。
  - Cloud：NVIDIA Tesla A100-80GB（CUDA 12.1，Intel Xeon Platinum 8358 2.60GHz）；NVIDIA RTX 4090 24GB（CUDA 12.1，AMD EPYC 7542 2.90GHz）
  - PC：Lenovo Legion Y7000，NVIDIA RTX 4060 Laptop 8GB（CUDA 12.6），Intel i7-13650HX 2.6GHz

- 开源Serving框架是什么。修改了什么。
  - Cloud 场景：基于 HuggingFace Transformers（Pytorch front-end + C++/CUDA backend）、vllm（PagedAttention）、AWQ（量化 inference engine）
  - PC 场景：基于 llama.cpp（C++/CUDA backend for NVIDIA GPU + Intel CPU）、PowerInfer（sparse activation + GPU-CPU hybrid inference）
  - 修改内容：
    1. 在每个 decoder layer 后插入 predictor 调用点（feature extraction + MLP inference + verification algorithm），按 heuristic scheduling engine 动态激活
    2. Offline scheduling：预先用大量 prompt 跑 full-predictor 推理，统计各层 exit 频率，按频率排序生成 predictor 激活优先级列表，存入模型配置文件（per-model，仅执行一次）
    3. Online scheduling：运行时维护长度为 N（如 5）的环形队列记录最近 N 个 token 的 exit layer；维护长度为 L（总层数）的数组统计最近 N 个 exit layer 的 ±2 邻域命中次数。最终激活的 predictor 集合 = offline 高频子集 ∪ online 邻域子集，平均只需 ~10.2 个 predictor（vs 原始 31 个）即实现 ~68% predictor reduction，~1.21× 端到端加速

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：https://github.com/infinigence/SpecEE，Zenodo DOI: https://doi.org/10.5281/zenodo.15102802，MIT License
  - SpecEE on HuggingFace 推理全过程（Llama2-7B on A100，autoregressive decoding）：
    1. **输入与预处理**：用户 prompt 文本 → HuggingFace tokenizer → token ids [t₁,t₂,...,tₘ]，长度 m
    2. **Speculative Token 生成**：EAGLE DLM（~0.9GB memory on GPU）基于 prompt 生成 4 个 speculative tokens [T₁,...,T₄]，提取 speculative_lm_head = lm_head[:, spec_indices]（4096×4 矩阵，存于 GPU global memory）
    3. **Heuristic Scheduling Engine 激活**：Offline scheduling 已预计算 predictor 优先级列表（如 layers [15,16,17,18,19,20,21,22,23,24]）。Online scheduling 检查环形队列中最近 5 个 token 的 exit layer → 若上一 token exit at layer 20 → 激活 layers [18,19,20,21,22] 的 predictor。最终 Union 集合 ≈ 10 predictors
    4. **Decoder Layer Forward（HuggingFace）**：
       - 对每层 i = 0..31：执行 self-attention（Q,K,V projection → FlashAttention → O projection）→ FFN（gate_proj, up_proj, down_proj）。若该层 predictor 在激活集合中，则执行 feature extraction（hidden_states[1×4096] × speculative_lm_head[4096×4] + softmax + prob_diff → 12-dim feat）→ MLP predictor forward（12×512 + 512×1 matmuls on GPU tensor cores）→ 若 pred>0.5 则 verification（hidden_states × lm_head → global argmax 是否在 spec set），是则 early exit，否则继续
    5. **输出**：early exit 层输出的 token → 经 HuggingFace tokenizer detokenize → 文本。若未 exit，最终在 layer 31 通过 lm_head 输出
  - 关键：SpecEE 不修改原始 LLM 参数，predictor overhead ~0.0009s/token（vs 端到端 ~0.016s/token ≈ 5.6%）；heuristic scheduling 将 predictor 数量从 31 降至 ~10.2，predictor 总 overhead 约 ~1.87% inference latency

## 59-Proteus- A High-Throughput Inference-Serving System with Accuracy Scaling.pdf

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Proteus是一个基于accuracy scaling的高吞吐推理serving系统，核心包含两个模块：(1) **Resource Manager**——通过MILP（Mixed Integer Linear Programming）联合优化三个子问题：Model Selection（选择哪个模型变体及每个变体的副本数）、Model Placement（将每个模型变体放在哪个异构设备上）、Query Assignment（各query type的请求按什么比例路由到各个设备）。目标函数：最大化effective accuracy（所有成功服务请求的平均准确率），约束：每设备最多1个模型变体、每query type路由比≤1、模型变体必须支持对应query type、设备服务吞吐不能超过峰值、所有到达请求必须被服务。MILP求解使用Gurobi Python接口，在macro-scale需求变化时异步触发（默认每30秒或负载突变时），不与query serving竞争关键路径。(2) **Adaptive Batching**——基于proactive、non-work-conserving方法的自适应batching算法：对队列中q个query，计算T_max_wait(q+1) = T_exp(1) - T_process(q+1)（第一个query过期时间减去处理q+1个query所需时间），若在此时间内第q+1个请求未到达则立即以batch=q执行；若到达则更新T_max_wait(q+2)继续等待，循环直到必须执行。该算法确保不违反任何请求的latency SLO，同时最大化batching吞吐。
  - 系统架构：Controller（Resource Manager + Model Registry + Model Profiler + Statistics Collector）→ Load Balancer（Request Router + Monitoring Daemon，每种query type一个）→ Workers（Adaptive Batching + Hardware Executor，每个worker运行一个模型变体）。Controller与Load Balancer分离使Resource Manager异步于推理关键路径。
  - 实验比较：(1) 端到端性能：Proteus vs Clipper-HT（静态配置高吞吐版）/Clipper-HA（静态配置高精度版）/Sommelier（部分动态accuracy scaling）/INFaaS-Accuracy（全动态但heuristic-based），Twitter真实trace+多种模型家族；(2) Bursty workload响应性：所有baseline在合成burst trace上的throughput/accuracy/SLO violations时序列对比；(3) Adaptive Batching单独评估：Proteus batching vs Clipper AIMD batching vs Nexus early-drop batching on Uniform/Poisson/Gamma分布trace；(4) 消融实验：移除Model Placement/MS/Query Assignment/Adaptive Batching各组件后的性能；(5) SLO敏感性：1×-3.5× SLO multiplier扫描；(6) 各模型家族性能分解；(7) MILP求解开销随device/model variant/query type数量扩展。

- 硬件平台是什么，配置是什么。
  - 集群：20× Intel Xeon Gold 6126 @ 2.60GHz CPU workers + 10× NVIDIA GTX 1080 Ti GPU workers + 10× NVIDIA V100 GPU workers
  - 软件栈：Kubernetes + Docker容器部署，ONNX Runtime（CUDA Execution Provider用于GPU，CPU Execution Provider用于CPU）
  - Simulator：∼6000行Python代码，基于事件队列实现，使用profiling的平均推理延迟作为处理时间

- 开源Serving框架是什么。修改了什么。
  - Proteus为自研serving系统，非基于Clopper/vLLM/TFServing等已有开源serving框架的直接修改。但基准中对比的框架包括Clipper（静态资源分配+AIMD batching）、Sommelier（单设备accuracy scaling）、INFaaS（动态model selection/placement，greedy heuristic）。
  - Proteus核心修改（相对于无accuracy scaling的静态serving系统）：
    1. **Resource Manager**：新增MILP-based模型选择、放置、查询分配联合优化，使用Gurobi MILP solver异步求解。
    2. **Controller-Load Balancer解耦架构**：将资源分配从推理关键路径分离。
    3. **Adaptive Batching模块**：替换标准static batching或work-conserving batching为proactive non-work-conserving算法。
    4. **Model Profiler**：新增per-(model variant, device type, batch size)三元组的延迟/吞吐profiling，存储到O(1)查找的in-memory key-value store。
    5. **Statistics Collector**：从各Load Balancer的Monitoring Daemon收集QPS统计，用于触发Resource Manager重新分配。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：https://github.com/UMass-LIDS/Proteus, DOI: https://doi.org/10.5281/zenodo.10428550。Simulator代码约6000行Python，包含配置文件和trace数据。提供Docker和本地安装两种方式，Docker方式无需Gurobi license即可评估。
  - Serving框架输入到硬件执行全过程（以聚类部署，EfficientNet分类任务，Twitter trace workload为例）：
    1. **开发者注册**：开发者通过Controller注册application（如EfficientNet image classification）及其所有model variants（B0-B7，各变体对应不同accuracy/throughput trade-off）。Controller的Model Profiler在每个worker设备上profile各model variant在不同batch size下的推理延迟/吞吐，存入in-memory KV store（key: (model variant, device type, batch size)）。
    2. **工作负载到达与监控**：Twitter trace生成QPS时间序列→每秒的请求按Poisson过程分配inter-arrival times→Load Balancer的Monitoring Daemon记录到达QPS并定期上报Controller的Statistics Collector。
    3. **Resouce Manager MILP求解**：当macro-scale QPS变化超过阈值或周期30秒触发→Resource Manager构建MILP：变量{x_d,m}（model selection+placement，bool，device d是否host model m）、{y_d,q}（query assignment，[0,1]，query type q路由到device d的比例）。Obj: max Σ_q a_q（各query type的accuracy权重和，a_q = Σ_m A_m · (Σ_d x_d,m · z_d,q · s_q)）。约束：(i) Σ_m x_d,m ≤ 1每设备最多一个模型；(ii) Σ_d y_d,q ≤ 1路由比例和≤1；(iii) Σ_m Σ_d B_m,q · x_d,m · y_d,q = Σ_d y_d,q路由必须到支持该type的设备；(iv) Σ_q z_d,q ≤ Σ_q y_d,q · s_q服务量不超过分配量；(v) z_d,q ≤ Σ_m P_d,m,q · x_d,m每设备服务不超过其峰值吞吐；(vi) Σ_d z_d,q = s_q所有需求必须被服务。使用Gurobi求解（平均4.2秒），产出optimal {x_d,m}, {y_d,q}。
    4. **资源重新分配**：Controller按MILP解terminate不再需要的模型变体实例（Docker container），启动新模型变体实例→新模型变体的ONNX模型加载到worker的GPU/CPU设备→Request Router更新路由表（entry per model variant per accelerator，O(D×M×Q)空间，<1ms查找延迟）。
    5. **推理请求处理（Adaptive Batching）**：每个Worker设备的Adaptive Batching模块按队列状态动态决定batch size——设当前队列有q个请求，第一个请求在T_exp(1)过期，处理q+1个请求耗时T_process(q+1)→计算T_max_wait(q+1)=T_exp(1)-T_process(q+1)，在此时点前若第q+1个请求到达则更新为T_max_wait(q+2)继续等待；若超时则立即以batch size=q执行。T_process通过Model Profiler的profiled latency查表（考虑batch size scaling）。最大batch size受SLO（推断延迟≤SLO/2，因最坏情况请求在batch开始后到达）和设备内存限制。
    6. **硬件执行**：ONNX Runtime（CUDA) Execution Provider执行batching inference→NVIDIA V100或1080 Ti GPU上运行模型forward pass→结果通过ONNX Runtime返回Host→Load Balancer返回给客户端。

- 属于Serving调度的实现是什么？实验比较什么？
  - **实现**：针对batched RALM inference系统提出 **Selective Batching + Early Generation** 调度策略。核心设计：(1) **Selective Batching**：扩展Orca [99]的continuous batching思路到RALM——每个request在完成自己的retrieval后立即恢复token generation（而非等待batch内所有request完成retrieval），retrieval completion作为异步事件触发generation batch的组建；(2) **Early Generation**：在某个request的retrieval尚未完成时，其他已就绪的request可提前生成额外token（early tokens），利用MNM dual row buffer实现PIM-based generation与PNM-based retrieval的物理并发；(3) **Request Manager**：Runtime测量每token generation时间T_gen和最大retrieval延迟T_ret,max→计算N_ret=⌊T_gen/T_ret,max⌋（限制同时并发retrieval的request数，bound early generation防精度衰减）→形成Request Group (Gr)→维护Retrieval Group Queue（FIFO+动态重排）；(4) **updateQueue()**：根据各Group已积累的early generated tokens数重排retrieval queue，优先服务已提前生成较多token的Group；(5) **Database分布**：同一retrieval configuration的requests归入同一batch（避免heterogeneous latency skew），database clusters均匀分布到所有MNM HBM stack（避免hot cluster导致PNM负载不均）。
  - **比较目标**：(a) Baseline GPU batched RALM（FAISS-GPU sequential retrieval per batch, 4-request同步）；(b) PipeRAG [35]（co-execution: generation/retrieval pipelined overlap，但受限于causal order和latency matching）；(c) GPU + Early Gen（GPU上纯软件调度，无PIM/PNM加速→early token过多导致perplexity升高）；(d) 不同batch size（4→256）和nprobe（32→512）下的accuracy/perplexity对比。

- 硬件平台是什么，配置是什么。
  - **GPU**：NVIDIA H100 NVL 94GB HBM3, 132 SMs @ 1GHz, L2 60MB, NVLink 4.0。
  - **MNM HBM**：6× MNM HBM3 stacks (每stack 8-Hi 16GB, 1024-bit I/F @5.2Gb/s/pin, 16Ch organization) + 6× standard HBM3 stacks = 12 HBM total。
  - **Host CPU**：Intel Xeon Gold 6526Y。
  - **Interconnect**：GPU↔MNM Controller via high-BW interconnect (PCIe 5.0 or CXL)；GPU↔GPU via NVLink 4.0 (for 2×GPU baseline)；GB200 via NVLink 5.0 (scalability)。

- 开源Serving框架是什么。修改了什么。
  - **Serving框架**：论文并未基于现有开源Serving框架（如vLLM/TGI/Orca的直接代码）实现，而是基于RETRO [7]的开源实现（RETRO TensorFlow/model code）和FAISS-GPU [37]构建了自定义的RALM inference runtime。修改的核心：
    - **原有RETRO执行流**：Sequential generation-retrieval loop——生成retrieval_interval个token→调用FAISS IVF-PQ检索（batch内所有requests同步等待）→encode retrieved data→feed到CCA层→继续生成→重复。Batch内requests等待最慢的retrieval完成才能继续。
    - **MNM调度修改**：
      1. **Selective Batching**：将batch内requests的retrieval与generation解耦——retrieval completion event-driven→每个request独立状态机（GENERATING/RETRIEVING/READY）→在任何时刻，所有READY requests可batched execution PIM-based generation，RETRIEVING requests的PNM-based retrieval在独立硬件路径上执行→无须等待。
      2. **Early Generation**：在GPT-style autoregressive loop中插入early token预算检查——request可在retrieval pending期间生成最多N_early个token（由T_gen/T_ret,max bound限制）→防止过长的retrieval延迟导致generation资源闲置→当pnretrieval完成后立即插入retrieved data到CCA→继续正常generation。
      3. **Request Group formation**：启动时按retrieval configuration分组（Config 1-5分离，避免混batch的latency imbalance）→每组内按N_ret=⌊T_gen/T_ret,max⌋切分为retrieval groups→Request Manager维护Retrieval Group Queue（circular queue with dynamic reordering）→GPU-Generation API: MNM::Gen(group)→GPU-Retrieval API: MNM::Ret(group)。
      4. **Database distribution**：140M/700M/32B vectors database的clusters均匀shard到6个MNM HBM stack→每个stack负责nlist/6个cluster的PNM retrieval→GPU merge各stack输出的partial top-k→global top-k。
    - **Perplexity-aware scheduling**：评估early generation对perplexity的影响（图8）→batch size越大/nprobe越大→GPU early gen perplexity显著升高（因retrieval延迟长→更多tokens在无最新检索数据下生成）→MNM early gen因PNM加速retrieval延迟→early tokens少→perplexity degradation可控。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **论文未开源scheduling runtime**。RETRO开源模型代码和FAISS-GPU [37]为公开资源，但MNM调度器（Request Manager/Retrieval Group Queue/Selective Batching+Early Generation逻辑）未找到公开仓库。MNM调度与vLLM的continuous batching [99]思路相似但面向RALM的retrieval-generation two-phase场景。
  - **Serving框架从输入到硬件执行全过程（Batch 32, Config 5, RETRO-7.5B）**：
    1. **Admission**：32个requests到达→按retrieval configuration分组（同Config 5的requests归入同一batch）→database clusters均匀shard到6个MNM stack（load balancing）。
    2. **Initial measurement**：Warmup: 执行第1个token generation并测量T_gen（per-token PIM-based generation latency）→执行第1次retrieval并测量T_ret,max（最大cluster PNM retrieval latency）→计算N_ret=⌊T_gen/T_ret,max⌋=假设3→形成Retrieval Groups: Gr0{Req0,1,2}, Gr1{Req3,4,5}, ...Gr10{Req30,31}。
    3. **Retrieval Group Queue组建**：Request Manager将各Group插入Retrieval Group Queue（初始按顺序）→设置On-Ret Gr=Gr0→发起Gr0的PNM retrieval（MNM::Ret(Gr0)）→同时Gr1-Gr10的requests可执行early generation（MNM::Gen(Gr1∪Gr2∪...∪Gr10)）——生成early tokens并记录各Group的early token count。
    4. **Retrieval→Generation转换**：Gr0 PNM retrieval完成→GPU读取top-k IDs from MNM→发送给Host CPU→Host通过ID mapping table查询raw data IDs→从main memory database fetch raw text→编码为chunk embeddings→送回GPU。GPU发起MNM::Gen(Gr0∪其他READY requests)→Gr0 requests的CCA层用新检索数据更新K/V→PIM-based MHA generation→各request生成tokens。
    5. **Request Manager更新**：On-Ret Gr从Gr0切换到Gr1→发起MNM::Ret(Gr1)的PNM retrieval→Request Manager检查各Gr的early generated token count→Gr3积累了4个early tokens（已超过retrieval_interval=8的一半但未到）→Gr1 retrieval期间Gr3 continue generation→Gr3生成满8 tokens后自动进入RETRIEVING状态（无需等待Gr1完成，因为下一个retrieval slot可分配给Gr3）。
    6. **updateQueue()**：所有Groups完成一轮retrieval后（all Grs通过retrieval cycle）→Host调用updateQueue()→按各Gr的early generated token count降序重排Retrieval Group Queue→Gr3（4 early tokens）排到队列最前→Gr2（1 early token）排后→形成下一轮retrieval顺序：Gr3→Gr0→Gr5→...。确保已积累最多early tokens的Group优先获得下一次retrieval slot。
    7. **全过程持续**→E2E latency = 从第1个request admitted到最后1个request完成generation的时间。MNM调度与baseline GPU的关键差异：GPU需要batch内所有requests在每次retrieval后同步→大量idle cycles等待最慢request→MNM通过selective batching（异步retrieval）和early generation（生成预支）将retrieval延迟的大部分隐藏→GPU utilization更高。

## 60-SpotServe- Serving Generative Large Language Models on Preemptible Instances.pdf

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：SpotServe是第一个在可抢占实例上运行的分布式LLM serving系统，由四个核心组件组成：(1) Parallelization Controller——自适应配置优化器，以配置C=(D,P,M,B)（数据/流水线/张量模型并行度+最大batch size）表示并行化策略，根据实例可用性变化和请求到达率α_t，动态选择最小化端到端延迟l_req(C)且维持吞吐φ(C)≥α_t的配置，考虑吞吐、延迟和货币成本的trade-off；当峰值吞吐无法满足到达率时最大化吞吐；(2) Device Mapper——将设备映射建模为二分图匹配问题，GPU设备为左侧节点，新配置的pipeline-stage-shard位置为右侧节点，边权重为可复用模型参数和KV cache量，用Kuhn-Munkres (KM)算法求最大权重匹配，最小化context迁移的数据传输量；(3) Migration Planner——提出渐进式迁移计划：优先迁移KV cache context，然后按内存优化顺序迁移各层权重context；优先完成第一阶段Stage的迁移使其尽早开始服务，与后续Stage迁移overlap；(4) Interruption Arranger (Stateful Inference Recovery)——利用grace period在token粒度（而非request粒度）commit推理进度，通过JIT Arrangement决定何时停止decoding和启动迁移：preemption前最大化arranged iterations（S_t = argmax{lexe(S|C_t) < T^- - Tmig}），acquisition前最小化（S_t = argmin{lexe(S|C_t) ≥ T^+}）。
  - 实验比较：(1) 端到端推理延迟：SpotServe vs Reparallelization vs Rerouting，三个模型（OPT-6.7B/GPT-20B/LLaMA-30B）×四个trace（AS/BS/AS+O/BS+O）×多个延迟指标（average/P50/P90/P95/P99/P99.9）；(2) 货币成本对比：GPT-20B上per-token cost和延迟的trade-off，含on-demand only baseline；(3) 波动负载评估：MAF production trace上三系统对比，含per-request latency时间序列分析；(4) Ablation study：逐步禁用Parallelization Controller→Migration Planner→Interruption Arranger→Device Mapper，测P99 tail latency和average latency变化。

- 硬件平台是什么，配置是什么。
  - AWS g4dn.12xlarge spot instances，每instance 4×NVIDIA Tesla T4 GPU (16GB)，inter-instance网络带宽50Gbps，x86_64 CPU。实验使用最多12个instances（48 GPUs total）。Grace period: 30秒（AWS spot instances标准）。CPU推理服务器部署在单独的on-demand CPU instance上。

- 开源Serving框架是什么。修改了什么。
  - 框架: NVIDIA FasterTransformer [5] (开源，https://github.com/NVIDIA/FasterTransformer)，基于CUDA/cuBLAS/C++的高性能Transformer推理框架。修改：
    1. **Context Daemon**: 新增独立进程，管理GPU上的model context（模型参数）和cache context（中间激活/KV cache）。推理引擎通过proxy访问这些context；实例被抢占时Context Daemon进程存活，避免重新加载context到GPU。
    2. **Memory allocation替换**: 将FasterTransformer中model context和cache context的原始内存分配替换为从Context Daemon获取对应GPU tensor。
    3. **Context Migration**: 实现基于NCCL batched asynchronous send/recv primitives的context迁移操作，需要GPU memory中额外的通信buffer空间（动态分配和释放）。
    4. **CUDA IPC**: Context Daemon和FasterTransformer属于不同进程，使用CUDA Inter-Process Communications (IPC)共享context指针。
    5. **Mutex Lock**: 为每个context tensor使用mutex lock，在迁移未完成前阻塞推理计算，以支持渐进式迁移与推理的overlap。
    6. **Inference Server**: 5.6K LoC C++ + 2.2K LoC Python，包含三个常驻进程：Request Manager（接收请求、动态batching、分发到推理实例、收集输出）、Instance Manager（与云平台交互，接收抢占/获取通知，分配/释放实例）、Meta-Context Manager（管理并行配置调整，发送context migration指令到GPU instances）。
    7. **Cost Model + Offline Profiler**: 估计推理延迟、系统吞吐和context migration开销，使用piece-wise linear functions建模。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源: https://github.com/Hsword/SpotServe (Apache-2.0 License)，artifact DOI: https://doi.org/10.5281/zenodo.10558752。包含Global Server (Python)、ParamsClient/Context Daemon (C++)、modified FasterTransformer (C++)。构建需要CUDA≥10.2、NCCL≥2.10、MPI、CMake≥3.8。每instance需约600GB磁盘空间。
  - Serving框架从输入到硬件执行全过程（以GPT-20B on trace AS，12 instances，每instance 4×T4 GPU为例）：
    1. **系统初始化**: Inference Server部署在on-demand CPU instance。12个g4dn.12xlarge spot instances各部署Inference Engine + Context Daemon。初始并行配置C_0=(D=1,P=3,M=4)，模型参数通过Context Daemon加载到各GPU。Offline Profiler预先完成各(D,P,M,B)配置的延迟和吞吐估计。
    2. **请求到达与分发**: 用户请求以Gamma arrival process (CV=6)到达→Request Manager接收→按当前配置C_t的batch size B动态打包→分发到inference pipelines→各pipeline的FasterTransformer engine执行autoregressive decoding（S_in=512, S_out=128）。初始phase处理prompt tokens（并行），incremental decoding每iteration生成1个output token（KV cache加速）。
    3. **Instance Preemption处理**（以t=120s时1个instance被preempt为例）: Cloud发送preemption notification（含30s grace period）→Instance Manager接收→通知Meta-Context Manager。Parallelization Controller执行Algorithm 1: 当前N_t=44 GPUs (11 instances×4)，C_t=(D=1,P=3,M=4)，α_t=0.35 req/s→计算是否存在C使得φ(C)≥α_t且云有足够instances→选择满足条件且最小化l_req(C)的配置C_{t+1}→Instance Manager同时尝试分配spot和on-demand instances。
    4. **Device Mapping**: Device Mapper构建二分图G=(V_a, V_t, E)，左侧为可用的GPU devices（含grace period内的），右侧为C_{t+1}的pipeline-stage-shard位置（D×P×M个位置）→计算每条边的权重（源GPU与目标位置共享的model parameters + KV cache量）→KM算法求最大权重匹配→产出最优device mapping，最大化context复用、最小化NCCL数据传输。
    5. **Interruption Arranger JIT决策**: 每个要被preempt的GPU instance上，Interruption Arranger接收grace period开始通知→计算S_t = argmax{lexe(S|C_t) < T^- - Tmig}→在grace period内尽可能多生成tokens→临界点到达后触发context migration。
    6. **Migration Planner执行**: Algorithm 2生成migration plan→优先迁移KV cache context→MemOptMigPlanner按内存buffer上限U_max确定layer weight迁移顺序（skip超过buffer上限的layer→对剩余layer解min-max问题选使max buffer usage最小化的layer→顺序迁移）→Progressive Migration: 第一阶段Stage的所有context迁移完成后立即<start>该stage开始服务，与后续stage迁移overlap。
    7. **Context Migration执行**: 通过NCCL batched async send/recv在inter-instance 50Gbps网络上传输model parameters和KV cache→Context Daemon的mutex lock在迁移期间阻塞推理→迁移完成后释放lock→新的推理pipeline继续之前interrupted requests的decoding（从committed token state恢复，无需recomputation）。
    8. **持续运行**: SpotServe维护额外2个instance作为candidate pool以平滑instance替换。当检测到workload下降或instance acquisition完成→重新触发ConfigOptimizer→调整到更优配置（如从高吞吐配置切换回低延迟配置）→释放多余的on-demand instances（on-demand first释放以节省cost）。全过程config optimizer在线运行overhead<1s。

## 61-Duplex- A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching.pdf

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Duplex实现了一套serving调度系统，包含两个层面：(1) **Continuous Batching Scheduler**：自定义的serving scheduler，实现stage-level continuous batching[56]，将LLM inference分解为prefill和decoding stages并在stage级别batching。支持两种stage类型：mixed stage（新到达request的prefill与现有request的decoding batch在一起）和decoding-only stage（所有request都在decoding阶段）。Batch size由stage中request数量决定。Request注入按Poisson distribution。(2) **Expert Co-processing调度**：在MoE层中，gate确定每token选择的experts后，统计每个expert处理的token数。基于lookup table（预存不同token数下xPU和Logic-PIM的处理时间估计），将处理token数少的experts分配给Logic-PIM执行，其余分配给xPU。决策时间为negligible。通过tensor parallelism for experts（Duplex+PE+ET）增加每device处理的expert数量以增强co-processing效果。(3) **Attention Co-processing调度**：在mixed stage中，prefilling sequences的attention分配给xPU（高Op/B），decoding sequences的attention分配给Logic-PIM（低Op/B）。(4) **Memory Allocation管理**：将device memory space按bank bundle index分为4个section。Expert FFNs round-robin分配在这4个space上。KV cache交替分配在3个space上，第4个space存储prefilling sequences的Q/K/V矩阵。Co-processing时按memory space选择xPU或Logic-PIM处理，避免bank bundle冲突。(5) **Prefill阶段的K/V迁移**：prefill结束后，xPU将prefilling sequences的K/V矩阵从第4个memory space迁移到KV cache bank bundle，overhead negligible（仅执行一次）。
  - 实验比较：(1) Throughput：Duplex vs GPU vs 2xGPU，以及Duplex vs Duplex+PE vs Duplex+PE+ET消融分析；(2) 各种latency (TBT/T2FT/E2E, p50/p90/p99) 的对比；(3) 不同QPS load下的latency scalability；(4) Split Prefill/Decoding node vs non-split的吞吐对比。Batch sizes 32/64/128，Lin/Lout从(256,256)到(4096,4096)。

- 硬件平台是什么，配置是什么。
  - Baseline GPU: NVIDIA H100 [35] (80GB HBM3 per device, 16GB per HBM stack, 8-hi 2-ranks per stack)。xPU等同H100规格。
  - Duplex配置: xPU + Logic-PIM (4x HBM bandwidth, 8 Op/B peak computing, 每stack 21.3 TFLOPS)。
  - System配置: ≤8 devices使用HGX NVLink 900GB/s bidirectional；>8 devices通过8-device节点 + InfiniBand 400GB/s互联。
  - Default配置: Mixtral/OPT/Llama3: 1 node 4 devices; GLaM: 1 node 8 devices; Grok1: 2 nodes 8 devices each。

- 开源Serving框架是什么。修改了什么。
  - 框架：论文自研的serving scheduler + Ramulator 2.0 cycle-accurate simulator。未修改任何开源serving框架（如vLLM/FasterTransformer）。
  - 修改/实现：在simulator中实现了continuous batching scheduler，管理ongoing inference requests——每stage生成request信息（prefilling/decoding、sequence length）发送到cluster。实现expert co-processing的lookup table决策机制和attention co-processing的request-based分配逻辑。实现memory allocation的bank bundle-aware策略。
  - 模型分布方法[46]: tensor parallelism for non-expert weights（node内）+ expert parallelism for MoE layers（expert FFNs分布到不同GPU）+ data parallelism across nodes。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文未明确说明Duplex serving scheduler代码是否开源。Simulator基于Ramulator 2.0 (开源: https://github.com/CMU-SAFARI/ramulator2) 扩展。
  - **Serving框架从输入到硬件执行全过程**（以Mixtral 47B on 4-Duplex devices, batch size=64, Lin=2048, Lout=1024为例）：
    1. **请求到达与Scheduling**: New requests按Poisson分布到达→Serving scheduler检查当前stage类型：(a) 无新request→decoding-only stage；(b) 有新request→mixed stage（新request的prefill与现有request的decoding batch在一起）。Scheduler生成每stage的request信息（每个request的prefilling/decoding状态、当前sequence length、KV cache位置）发送到cluster。
    2. **模型分布与Device Assignment**: Cluster按tensor parallelism将non-expert weights（QKV generation、Projection、Token embedding、LM head）按列/行slice分布到4个Duplex devices。Expert FFNs通过expert parallelism分配（8 experts distributed: 每device 2 experts）。Duplex+PE+ET配置下使用tensor parallelism for experts——将每个expert的weight列切分到所有4 devices。
    3. **Decoding-only Stage执行** (dominant stage):
       - **QKV Generation/Projection/FC层**: 全在xPU上执行（高Op/B GEMM操作）。
       - **Attention (Decoding)**: Logic-PIM处理decoding sequences的attention。每个request的KV cache独立存储在HBM中。Logic-PIM的GEMM module从HBM读取Q vector (1×D)和KV matrices→执行attention operation（Q×K→softmax→×V）。Request parallelism: 64个decoding requests完全独立可并行。Head parallelism: 32 heads的Q/K/V slices独立处理。由于GQA (deggrp=4)，每group共享K/V，attention表现为narrow GEMM。
       - **MoE Layer (Decoding)**: Gate计算token-to-expert assignment→expert co-processing决策：gate后统计每expert的token分布→查lookup table→分配。假设64 tokens distributed across 8 experts, expert 0/3各处理15 tokens（xPU），expert 1/2/4/5/6/7各处理5-6 tokens（Logic-PIM）。Logic-PIM从HBM读取对应expert的weight matrices→执行FC1(Gate-proj)→Gated Activation (SiLU)→FC2(Up-proj)→All-reduce across 4 Logic-PIM stacks by xPU→FC3(Down-proj)。xPU同步处理分配给它的experts。
       - **LM Head**: xPU执行最后的token prediction。
    4. **Mixed Stage执行** (new request arrives):
       - New request prefill (Lin=2048 tokens): 全在xPU上执行（高Op/B——2048 tokens sharing same weights）。
       - Decoding sequences attention: Logic-PIM处理（同decoding-only stage）。
       - Prefill attention: xPU处理（2048 Q slices vs same K/V per head，高Op/B GEMM）。
       - MoE Layer: gate处理所有tokens (2048+64=2112 tokens)。Op/B大幅提高→大部分experts由xPU处理，Logic-PIM仅处理token数极少的experts。
       - Prefill完成后：xPU将新生成的K/V矩阵从prefill memory space迁移到KV cache bank bundle。
    5. **迭代**: Decoding stages持续迭代Lout=1024次→每step生成1 token per request→直至所有requests完成。
    6. **效果**: Duplex exploits Logic-PIM高带宽处理低Op/B的decoding-only MoE和attention→median TBT降低58.3% vs GPU。Duplex+PE+ET通过tensor parallelism for experts增强expert co-processing→throughput提升至2.67x vs GPU。

## 66-Chameleon- Adaptive Caching and Scheduling for Many-Adapter LLM Inference Environments..pdf

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Chameleon 在 S-LoRA 开源 LLM serving 平台上增加两大机制：(1) **Chameleon Adapter Cache**——利用空闲 GPU 内存缓存 LoRA adapter weights，避免每次请求从 host 通过 PCIe 加载 adapter 的开销。Cache 动态调整大小以适配 KV cache 和 running request 的内存需求波动，使用复合淘汰策略（Score = F×Frequency + R×Recency + S×Size, F=0.45, R=0.10, S=0.45）考虑 adapter 的频率、访问时间局部性和 adapter size（rank）来决定淘汰对象（优先淘汰小 adapter 降低 miss cost）。Cache Manager 维护 reference counter 保护运行中的 adapter 不被淘汰。(2) **Chameleon Scheduler**——非抢占式多级队列（MLQ）调度器，将请求按照 Weighted Request Size（WRS = A×(InputSize/MaxInputSize) + B×(OutputSize/MaxOutputSize) × (AdapterSize/MaxAdapterSize)，A=0.4, B=0.6）分类为 small/medium/large 三档，分别进入不同队列。各队列有 resource quota（token 数），每 decode iteration 按两阶段（Phase 1: 按队列 quota 依次从 small→large 队列选取请求进 batch；Phase 2: 未用完的 spare resources 按同样顺序重新分配给有 pending request 的队列）。队列数量通过 K-Means 聚类动态决定（max K=4），per-queue cutoff 取相邻 centroid 中点，每 T_refresh=5min 重计算。Queue quota 用 M/M/1 队列理论公式 T_{ok}^{min} ≥ S×D×(1/SLO + λ) 分配以满足 SLO。包含 Opportunistic Bypass 机制：当队列头部请求因内存不足无法入 batch 时，跳过该请求选取后面能入的请求（带 squash 重执行保护）。
  - 实验比较：(1) **Tail Latency (P99 TTFT)** vs S-LoRA baseline under varying loads (5–13 RPS)；(2) **Performance breakdown**：Chameleon vs ChameleonNoCache（仅调度） vs ChameleonNoSched（仅缓存）；(3) **Scheduling policy comparison**：Chameleon Scheduler vs S-LoRA default FIFO vs μServe SJF，记录时间轴上 P99 TTFT 变化和各类请求（small/medium/large）的平均排队延迟；(4) **Cache eviction policy comparison**：Chameleon 复合淘汰 vs LRU vs FairShare（三项等权重），按 adapter rank 和总量对比 P99 TTFT；(5) **Prefetching**：Chameleon + Histogram-based prefetch vs Chameleon vs S-LoRA；(6) **Sensitivity analysis**：output length predictor 准确率（60%/80%/100%）、adapter rank 和 adapter 流行度分布（U-U, U-P, P-P）、总 adapter 数量（10–200）、不同 trace（Splitwise/WildChat/LMSYS-Chat-1M）、Static vs Dynamic 队列组织；(7) **Scalability**：Llama-7B/13B/30B 不同模型大小、24GB/48GB/80GB 不同 GPU 内存、A40 vs A100 计算能力；(8) **Multi-GPU**：TP1/TP2/TP4 with 4×A100-80GB，不同 load 下 P99 TTFT。

- 硬件平台是什么，配置是什么。
  - 主要平台：1×NVIDIA A40 GPU (48GB memory)，AMD EPYC 9454 CPU (48 cores, 377GB main memory)。
  - Scalability 实验：1×NVIDIA A100 GPU (24GB/48GB/80GB 可配置 memory)。
  - Multi-GPU 实验：4×NVIDIA A100 GPU (80GB each)，tensor parallelism (TP2/TP4)。
  - 模型：Llama-7B 为主要，Llama-13B/Llama-30B 用于 scalability。也测试了 Falcon、OPT、Mixtral（论文称 trend 相似）。
  - Workload：基于 Azure production trace (Splitwise [41])，input/output length scaled down to match testbed memory。Poisson 分布 control inter-arrival time。Na=100 不同 adapters，rank 取 {8,16,32,64,128} 均匀分布 rank 流行度，power-law adapter 流行度 within a rank。也测试 WildChat-1M 和 LMSYS-Chat-1M traces。

- 开源Serving框架是什么。修改了什么。
  - 框架：S-LoRA (https://github.com/S-LoRA/S-LoRA)，state-of-the-art multi-adapter LLM serving platform。S-LoRA 原生采用 FIFO iteration-level scheduling + continuous batching，adapter weights 默认存储在 host memory，请求到来时异步 prefetch 到 GPU memory 并在请求结束后丢弃。
  - Chameleon 修改：(1) 在 S-LoRA 的 GPU memory management 中加入 Adapter Cache 层——Cache Manager 管理 adapter 的 fetch/evict/prefetch/resize；(2) 替换 S-LoRA 的 FIFO scheduler 为 Chameleon Multi-Queue Scheduler——增加 request classification（output length predictor + WRS 计算）、multi-queue admission、queue quota management、K-Means queue reorganization 和 Opportunistic Bypass；(3) 不需要修改 CUDA kernel、硬件或操作系统。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：Chameleon 论文未给出独立开源仓库链接。基于 open-source S-LoRA 实现。S-LoRA 开源地址 https://github.com/S-LoRA/S-LoRA。Output length predictor 使用 open-source BERT-based proxy model (来自 μServe [46])。
  - 框架使用全过程（以 Llama-7B on 1×A40 48GB, Splitwise trace, Na=100 adapters, 9 RPS 为例）：
    1. **初始化**: Base Llama-7B model weights 从 host memory 传输至 GPU memory。GPU memory 分区：Base Model Weights (~14GB) + KV Cache + Activations + 空闲区域→Adapter Cache。Scheduler 初始化 multi-queue 结构（初始基于 K-Means clustering 分析 WRS 分布得出 K=3 queues，per-queue cutoffs 和 quotas 计算完成），Cache Manager 初始化 metadata 表（Adapter ID, Rank, Last Used Timestamp, Usage Frequency, Reference Counter）。
    2. **请求到达与分类**: Inference request 到达→Frontend 提取 {input tokens, required adapter ID, adapter rank}→BERT-based output length predictor 估计 output token count→Chameleon Scheduler 计算 WRS = 0.4×(InputSize/MaxInputSize) + 0.6×(OutputSize/MaxOutputSize)×(AdapterSize/MaxAdapterSize)→根据 WRS 与 per-queue cutoffs 将请求分入 small/medium/large 对应队列。
    3. **Batch 构造（每 decode iteration 执行）**: Algorithm 1 两阶段——Phase 1: 从 small queue 开始按 quota tokens 选取请求（如 small queue quota=5000 tokens，取 3 个请求：req1 input 512 + output 320 token-eq, req2 200+150, req3 100+80，共 1362 tokens <5000），逐个入 batch。Medium queue（quota=3000 tokens）：取到 1 个请求消耗 800 tokens 后用尽。Large queue（quota=2000 tokens）：1 个请求消耗 1800 tokens。Phase 2: Medium queue 未用完的 2200 tokens→收集入 Total Spare Resources→从 small queue 开始重新尝试入 pending 请求→small queue 额外入 2 请求消耗 1200 tokens→remaining 1000 tokens 分配给 medium queue 入 1 个 pending 请求→batch 最终含 8 个请求。
    4. **Adapter Cache 管理**: Scheduler 将 batch 所需的 adapter IDs 发送给 Cache Manager→Manager 逐 adapter 检查：已在 Chameleon Cache 中→hit, RC++；miss→从 host memory 通过 PCIe load adapter weights 至 GPU→写入 Cache。若空闲 GPU memory 不足以容纳新 adapter→Manager 计算所有 RC=0 且不在 queued requests 中的 adapter 的 eviction score = 0.45×Frequency + 0.10×Recency + 0.45×Size→淘汰最低分 adapter(s) 至足够空间→load 新 adapter。同时 Cache Manager 检查 queued requests 的 adapter→若有空闲 memory 则 prefetch 入 cache。
    5. **GPU 推理执行（Prefill + Decode）**: batch 发至 GPU→Prefill 阶段：所有请求的 input tokens 并行过 base Llama-7B layers + 各自 LoRA adapter 两层 matrix multiplications（用 MBGMM kernel from S-LoRA）→生成 KV cache entries 和 first output tokens。Decode 阶段：iteration-level continuous batching→每 iteration，completed requests 释放 resource quota（tokens 还回队列）、KV cache 和 adapter RC--。新产生的 decode tokens 聚合。当请求的 RC=0 且 Cache Manager 决定保留 adapter→adapter 从 "in-use" 转为 "cached"。若 GPU memory 压力高→Cache Manager shrink cache。
    6. **Dynamic Reorganization**: 每 T_refresh=5min→Scheduler 收集最近请求的 WRS→重新 K-Means clustering (K=1..Kmax=4)→更新 queue 数量、per-queue cutoffs 和 quotas→平顺过渡至新配置。
    7. **循环**: Requests 持续到达、分类、入队→每 iteration 执行 Algorithm 1→Cache Manager 协同管理→GPU 执行 inference→completed requests 退出→直至所有请求完成。P99 TTFT 从 S-LoRA 的 8s 降至 ~1.5s (9 RPS)。

## 67-ELORA- Efficient LoRA and KV Cache Management for Multi-LoRA LLM Serving.pdf

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：FAST LIBRA（论文提出的 Multi-LoRA 推理缓存系统）基于 vLLM 实现，新增 8324 行 Python + 1644 行 C++ 代码。包含两个核心模块：(1) **Dependency-aware Cache Manager**——基于 Trie 树构建 LoRA 与 KV cache 之间的 usage dependency tree（根节点为 virtual root，第二层为 LoRA 节点，第三层以下为各 LoRA 对应的历史 KV cache 子树），维护统一的 HBM/主存 caching pool，使 LoRAs 和 KVs 可共享同一块 memory blocks（打破 vLLM 静态分区），并支持异步 swap-in/out（基于 Torch Stream）；(2) **Performance-driven Cache Swapper**——每 100ms monitor interval 评估 HBM 使用率（upper/lower threshold: 95%/70%），基于统一 cost 模型 Eval_i = LoRA_Eval_i × Retain_Eval_i 决定 swap-in/out 哪些 LoRA 和 KV cache。其中 Retain_Eval_i = cost_i × prob_i × (1 − sigmoid(t_i))（考虑传输代价、访问频率、LRU 时间衰减），LoRA_Eval_i = max(1, Low_lora / NowLoRA)（鼓励加载足够 LoRA 数量以避免 LoRA cold-start），Low_lora = Σ[1 − (1 − prob_i)^BS]（基于使用概率和近期 batch size 估计所需的 LoRA 数量）。Swap 规则：HBM 满时从叶子节点开始 swap-out，HBM 空闲时从根节点开始 swap-in，保证 dependency tree 始终连通、HBM 内 KV cache 全部 valid。
  - 实验比较：(1) **三种 Multi-LoRA 应用场景下的 latency 和 throughput**——Chatbots（LMSYS-33K [52]）、Multi-language Translations（Opus-100 [47]，用 Microsoft Azure Function trace [35,50] 采样时序）、Personal Agents（Google Taskmaster [5]，同 MAFT 采样），对比 vLLM（静态 HBM 分区+LRU policy）和 S-LoRA（统一 caching pool 但不保留历史 KV cache）；(2) **不同模型和 LoRA 数量的扩展性**——Llama-7B/13B/34B，LoRA 数量 20/50/100，rank 32/64 随机；(3) **TTFT breakdown 分析**——将 TTFT 分解为 queue、LoRA cold-start、KV cold-start 三部分 latency；(4) **HBM 利用率和 cache hit rate**——对比 HBM space 利用率和 KV cache/LoRA 命中率随时间变化；(5) **Ablation study：三个 variant 消融实验**——FAST LIBRA-WOM（去除 dependency maintenance）、FAST LIBRA-WOS（用 LRU 替代 cost model）、FAST LIBRA-WOL（去除 LoRA quantity reward）；(6) **大规模 LoRA 场景**——Llama-7B 下 LoRA 数量 1000/2000，三种分布（Uniform/Distinct/Skewed-0.1/Skewed-0.3），验证通用性。

- 硬件平台是什么，配置是什么。
  - CPU：Arm CPU（192 cores），256GB main memory。
  - 加速器：4× High-performance NPU，每个 256 TFLOPS FP16、64GB HBM，通过 PCIe × 16 Gen4.0 连接 host。
  - 模型部署：Llama-7B→1×NPU，Llama-13B→2×NPU，Llama-34B→4×NPU（按参数量 scaling）。

- 开源Serving框架是什么。修改了什么。
  - 框架：vLLM (https://github.com/vllm-project/vllm)。
  - 修改：(1) **Unified Caching Pool**——扩展 vLLM 的 BlockManager，将 HBM 和 main memory 统一划分为相同大小的 memory blocks；LoRA weights 按 rank 维度 block-wise 分区，对齐 KV cache block 大小，消除内存碎片；二者可共享同一 HBM 区域（废除 vLLM 的静态 HBM 分区，原 vLLM 分配 LoRA:HBM = 0.2）；(2) **Usage Dependency Tree**——新增基于 Trie 树的依赖结构，逻辑上记录 memory block 的地址和 token sequence/LoRA ID，node 存储 visit frequency、LRU time、node size；每 query 匹配时 DFS 遍历树（先匹配第二层 LoRA→再按 token order 匹配 KV cache 子树）；匹配成功后新 token 的 KV cache 插入对应 LoRA 分支的叶子；(3) **Asynchronous Swap-in/out**——基于 Torch Stream [33] 实现，query 等待所需 LoRA/KV 从主存 swap-in 时，其他已就绪 queries 继续执行，计算与数据传输 overlap；(4) **Cache Swapper 决策逻辑**——每 100ms 监控 HBM 使用率→>95% 触发 swap-out（从叶子 node 开始）→<70% 触发 swap-in（从根 node 开始），使用 greedy 算法按 Eval_i 排序逐个 swap 直至 HBM 平衡状态。与 baseline vLLM 代码差异：vLLM 使用静态 HBM 分区（empirically LoRA ratio=0.2）+ LRU 策略分别管理 LoRA 和 KV 内存区域，FAST LIBRA 将二者合并为统一 tree-based 结构并引入 cost-model 驱动 swap。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：FAST LIBRA 论文未给出独立开源仓库链接。截至搜索时未发现公开 GitHub 仓库。实现基于开源 vLLM。
  - 框架使用全过程（以 Llama-7B on 1×NPU 64GB HBM，Chatbot 场景，100 LoRAs，rank=32/64 随机，2 queries/sec sending rate 为例）：
    1. **初始化阶段**：加载 Llama-7B base model weights 至 NPU HBM。FAST LIBRA 扩展 vLLM BlockManager，将 HBM（64GB）和 main memory（256GB）统一划分为相同大小的 memory blocks。所有 100 个 LoRA adapter weights 按 rank 维度 block-wise 分区（与 KV cache block 大小对齐，避免内存碎片）。创建 virtual root node → 第二层挂载 100 个 LoRA 节点（label=LoRA ID），每个 LoRA 下的子树初始为空。HBM threshold 设为 upper 95%/lower 70%。
    2. **请求到达与 LoRA/KV 匹配**：Query Q 携带 {dialogue history tokens, target LoRA ID} 到达。Frontend 将请求送入 FAST LIBRA Cache Manager。Cache Manager 在 dependency tree 第二层 DFS 匹配 LoRA 节点——若 LoRA 在 HBM 中（hit）→继续在对应 LoRA 子树中 DFS 匹配历史 KV cache（按 token sequence 逐个匹配，如 "How are you" 先匹配 "How"→再匹配 "are"→再匹配 "you"）。若 LoRA 在 main memory 中（miss）→异步 swap-in（Torch Stream 通过 PCIe × 16 Gen4.0 传输），同时 query 排队等待。KV cache 匹配过程：逐个 token 沿 trie 路径走→hit 则复用已有 KV（prefix caching 生效）→miss 则到达 trie 叶子，后续由 prefill 计算新 token 的 KV cache。匹配到的 KV caches 必须是 valid 的——即其上方所有 ancestor nodes（含 LoRA node）均在 HBM 中，保证 dependency tree 在 HBM 内的部分连通。
    3. **HBM 满时的 Swap-out 决策（每 100ms 周期）**：Monitor 检测 HBM usage > 95%→触发 swap-out。Cache Swapper 从 Cache Manager 获取所有 HBM 内叶子节点作为候选（叶子 swap-out 不影响子树上层依赖）。对每个候选 node i 计算 cost model Eval_i：(a) Retain_Eval_i = transfer_cost_i × prob_i × (1 − sigmoid(t_i))，其中 transfer_cost_i = node_size / PCIe_bandwidth（LoRA node 按 rank 分块的 block 数×block 大小，KV node 为单个 token 的 KV block），prob_i = node visit frequency（从 dependency tree 记录的访问计数归一化），(1 − sigmoid(t_i)) 是 LSTM forget-gate 风格的时间衰减（t_i = 当前时间 − LRU time）；(b) LoRA_Eval_i = max(1, Low_lora / NowLoRA)，其中 Low_lora = Σ[1 − (1 − prob_i)^BS]（BS 取最近 5 秒平均 batch size）；(c) Eval_i = LoRA_Eval_i × Retain_Eval_i。对叶子候选按 Eval_i 递增排序（最小 benefit 的先 evict）→greedy 逐 node 通过 PCIe swap-out 到 main memory→回收 memory blocks 至 unified pool→直至 HBM usage 降至 70%−95% 之间。
    4. **HBM 空闲时的 Swap-in 决策（每 100ms 周期）**：Monitor 检测 HBM usage < 70%→触发 prefetch swap-in。候选为 main memory 中各 LoRA 子树根的 children nodes（从根开始 swap-in 保持 dependency 连通）。同样计算 Eval_i，按降序排序（最大 benefit 的先加载）→greedy 逐 node 通过 PCIe swap-in 入 HBM→直至 HBM usage 回到 70%−95%。这实现 proactive prefetch：低负载时提前将将来可能高频率访问的 LoRA/KV 调入 HBM，避免后续 cold-start。
    5. **推理执行（Prefill + Decode）**：当 query Q 所需的 LoRA adapter 和所有可匹配 prefix KV caches 都在 HBM 中后，query 进入 inference 阶段。Prefill 阶段：剩余不匹配的 prompt tokens（必须从第一个 missing prefix token 起）通过 base model forward + LoRA branch 计算——每层 W' = W + A_t B_t（LoRA 适配，A_t、B_t 为低秩矩阵，rank=32 或 64）→生成新 KV cache entries（KV_Cache_q,t = W_{k,v} q + A_t B_t q）。这些新 KV caches 被插入 dependency tree 对应 LoRA 分支的叶子位置（如 prefix "To be" 下已匹配 token "To" 和 "be" 的 KV 在 trie 中→新 token "or" 的 KV 插入 "be" node 下方），且直接在 HBM 保留（无需决策置于 HBM 还是 main memory）。Decode 阶段：每 iteration 生成 1 个新 token→new KV cache 插入 trie 叶子。若 decode 过程 HBM 压力升高→Cache Swapper 可能 swap-out 低 Eval_i 的历史 KV caches 为 running KV caches 让位。
    6. **SGMV Batching**：多个使用不同 LoRA 的 queries 通过 Segmented Gather Matrix-Vector multiplication (SGMV) [6,37] 打包为 single batch inference——一次 base model weights load 服务 batch 内所有 queries，各 query 通过各自 LoRA branch (W + A_t B_t) 修正输出。FAST LIBRA 不改动 SGMV kernel 本身，仅通过 cache manager 保证 batch 内所有 queries 所需的 LoRAs 均已 cached 在 HBM 中（消除 LoRA cold-start 导致的 batch 拆分或等待）。
    7. **请求完成**：Query 生成 EOS 或达到 max tokens→请求完成。其 KV caches 保留在 dependency tree 中供后续同 LoRA 的 query 复用（prefix caching）。当 HBM 压力高→Cache Swapper 根据 Eval_i 淘汰低价值历史 KV 为新高价值 KV 让位。
    8. **端到端效果**：以 2 QPS chatbot 场景为例，vLLM baseline TTFT=1032.4ms（含 queue + LoRA cold-start + KV cold-start），FAST LIBRA 将 TTFT 降至 ~380ms（63.4% reduction），TPOT 降低 40.1%，peak throughput 提升 35.2%。Cache hit rate 从 vLLM 的 ~60% 提升至 ~80%（1.3×），HBM 利用率从 vLLM 的 ~60% 提升至 ~72%（1.2×），消除 vLLM 中高达 48.1% 的 invalid KV caches（KV 在 HBM 但其 LoRA 已被 swap-out）。

## 71-Pre-gated_MoE_An_Algorithm-System_Co-Design_for_Fast_and_Scalable_Mixture-of-Expert_Inference.pdf

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Pre-gated MoE 系统将所有稀疏 MoE expert 参数 offload 到 CPU DDR4，仅将非 MoE dense 参数（self-attention、layer norm 等）常驻 GPU HBM。利用 pre-gate function 提前确定下一个 MoE block 需要哪些 experts，在当前 MoE block 执行 expert computation 期间异步从 CPU→GPU 通过 PCIe 迁移仅被激活的 experts（而非全部 experts），实现 expert 迁移延迟与 expert 计算的完全重叠。第一个 MoE block 的 expert selection→execution 仍需串行（无前置 pre-gate），但从第二个 block 起实现 compute（current block expert execution）+ communication（next block expert migration）流水线化。系统基于 NVIDIA FasterTransformer 实现。
  - 实验比较：(1) MoE block 平均延迟对比 Pre-gated MoE vs GPU-only vs MoE-OnDemand vs MoE-Prefetch，在 Switch-Base (8/64/128 experts) 和 Switch-Large (128 experts) 上；(2) 端到端推理吞吐（tokens/sec）；(3) Peak GPU memory usage（normalized to GPU-only），包括 Switch-Base 256 experts 扩展实验；(4) SSD offloading 场景的吞吐对比；(5) Expert caching（LIFO/LFU/LRU, 1%/10%/20% cache ratio）对 Pre-gated MoE 和 MoE-OnDemand 吞吐的影响。

- 硬件平台是什么，配置是什么。
  - GPU：单 NVIDIA A100 80GB HBM。
  - CPU：AMD EPYC 7V12 64-Core + 1.8TB DDR4 memory。
  - 互联：PCIe Gen4，实测 32 GB/sec 单向带宽。
  - GPU-only baseline：单 A100 80GB（Switch-Large 128 experts 105.6GB 会 OOM）。

- 开源Serving框架是什么。修改了什么。
  - 框架：NVIDIA FasterTransformer (https://github.com/NVIDIA/FasterTransformer)，state-of-the-art CUDA/C++ 推理库。
  - 修改：
    1. 修改 MoE block 的 gate function 为 pre-gate function——gate 计算输出 target 从当前 block 改为下一个 block 的 expert indices。
    2. 实现 CPU→GPU 异步 expert 迁移：使用 CUDA Stream 的异步 memory copy（cudaMemcpyAsync），在当前 block 的 expert FFN kernel 执行期间并发进行 PCIe 传输。
    3. GPU memory 管理：非 MoE 参数常驻 GPU；expert 参数在 GPU 中分配两块 buffer——一块为当前 block 激活的 experts，一块为下一个 block 预取的 experts（双缓冲）。
    4. 第一/最后 MoE block 特殊处理：第一个 block 额外保留传统 gate（选当前 block expert），最后一个 block 无 pre-gate。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：GitHub https://github.com/ranggihwang/Pregated_MoE；Zenodo https://doi.org/10.5281/zenodo.10976343；Docker: nvcr.io/nvidia/pytorch:22.09-py3。
  - Serving 框架使用全过程（Switch-Base 128 experts, top-1 activation, 单 batch 推理）：
    1. **模型加载**：非 MoE 参数（self-attention Q/K/V/O projection, layer norm, embedding）加载到 GPU HBM 常驻。所有 expert FFN 参数加载到 CPU DDR4（~30GB for Switch-Base 128 experts）。GPU 分配两块 expert buffer，每块可容纳 2×(top-k) experts 参数（当前 + 下一个 block 最多各 2 experts for top-2）。
    2. **请求到达（Prefill 阶段）**：输入 prompt token sequence 到达。FasterTransformer 按 layer 顺序执行。
    3. **Non-MoE Layer 执行**：Self-attention（QKV projection + MHA + output projection）在 GPU 上正常执行，参数已在 HBM。Layer norm 同。
    4. **第一个 MoE Block 执行（特殊处理）**：
       - First gate: 对 hidden states 做 gate forward → softmax → top-k 选当前 block experts（e.g., expert #2）。Expert #2 参数通过 cudaMemcpyAsync 从 CPU DDR4→GPU HBM 迁移（PCIe Gen4, ~120MB/expert for Switch-Base-128）。等迁移完成后，执行 expert #2 的 FFN (W1, W2, bias)。
       - 同时 Pre-gate: 对同一 hidden states 做 pre-gate forward → 得到下一个 block 的 expert indices（e.g., expert #5）。立即启动 cudaMemcpyAsync 将 expert #5 迁移到 GPU 第二块 buffer——此传输与 expert #2 的 FFN 计算并发（在不同 CUDA stream 上）。
    5. **中间 MoE Block 执行（全流水线化）**：
       - Block 1: Expert #5 参数已在 GPU ready → 直接执行 FFN kernel。Pre-gate 输出 block 2 的 expert indices → 异步迁移 block 2 expert 参数。FFN compute (green) 与 PCIe comm (blue) 完全重叠。
       - Block 2, 3, ..., N-1: 同上流水线。
    6. **最后一个 MoE Block 执行**：无 pre-gate，直接执行 expert FFN → 输出经 self-attention + layer norm → 采样出第一个 token。
    7. **Decode 阶段（逐 token 生成）**：每次 decoder iteration 生成一个 token，整个 MoE block 序列重新执行（与 prefill 相同流水线）。每 iteration 的 expert 选择可能不同（input-dependent）。
    8. **性能对比**：
       - GPU-only: 全部参数在 GPU → 无 PCIe 传输 → 最快但 OOM for 大模型（>80GB）。
       - MoE-OnDemand: last gate 选 expert → 迁移 → 执行 → 串行（无法重叠）→ 延迟最高 1.7× vs Pre-gated。
       - MoE-Prefetch: 预取整个 MoE block 全部 experts（~15GB for 128 experts）→ PCIe 传输本身成为瓶颈 → 延迟最高 42× vs Pre-gated。
       - Pre-gated MoE: 仅迁移 1–2 个 activated expert (~120–240MB) → PCIe 传输完全隐藏于 FFN compute 内 → 仅比 GPU-only 慢 19%。
    9. **构建与运行**：
       ```
       # Docker
       docker run -ti --gpus all --shm-size 5g --name pregated \
         -v ${DATA_PATH}:/data nvcr.io/nvidia/pytorch:22.09-py3 bash
       # Build FasterTransformer
       mkdir -p FasterTransformer/build && cd FasterTransformer/build
       cmake -DSM=80 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
       make -j
       pip install -r ../examples/pytorch/t5/requirement.txt
       # Run evaluation
       ./scripts/convert.sh   # 下载 SwitchTransformer models
       python scripts/eval_all.py  # 输出 block_lats.csv, throughputs.csv, peak_mems.csv
       ```

## 77-Make_LLM_Inference_Affordable_to_Everyone_Augmenting_GPU_Memory_with_NDP-DIMM.pdf

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Hermes系统在host CPU上实现了完整的LLM推理调度器，管理consumer-grade GPU和多个NDP-DIMM的异构计算。调度器包括四个关键组件：（1）**Monitor**：运行时采集neuron激活信息，记录各DIMM和GPU的计算负载；（2）**Lightweight Predictor**：基于token-wise similarity（4-bit neuron state table + finite state machine更新，借鉴分支预测的两级自适应策略）和layer-wise correlation（offline采样的neuron correlation table），在线预测下一token的activated neurons，预测准确率98%，内存<1MB；（3）**Neuron Mapper**：基于预测结果实时调整hot/cold neuron分区——neuron state > Th(10)判定为hot→从DIMM拷贝到GPU memory，GPU memory中state最低的cold neuron被覆盖换出；（4）**Window-based Online Scheduling**（Algorithm 1）：每5个连续token为一个window，统计window内各DIMM的激活neuron总数Z_j，按Z_j降序排列，配对最繁忙和最空闲的DIMM，从高负载DIMM remap最活跃neuron到低负载DIMM（通过DIMM-link @25GB/s），实现load balance。调度器还管理指令队列：GPU侧通过cudaLaunchKernel触发GEMM/GEMV kernel，NDP-DIMM侧通过memory command interface发送MAC/softmax命令。使用PIM-SYCL编程模型编译异构平台，Unified Memory编程实现GPU-NDP-DIMM间隐式数据传输。
  - 实验比较：（1）Hermes vs Huggingface Accelerate, FlexGen, Deja Vu（OPT-13B/30B/66B, bs=1）：Hermes平均578.42× over Accelerate, 247.25× over FlexGen, 75.24× over Deja Vu；（2）Hermes vs Hermes-host（CPU offloading替代NDP-DIMM）：4.79×-7.75× speedup；（3）Hermes vs Hermes-base（无激活稀疏性NDP-DIMM扩展）：大模型（Falcon-40B, LLaMA2-70B）上平均5.17× speedup；（4）Batching性能（bs=1-16）：Hermes平均148.98× over FlexGen, 75.24× over Deja Vu, 7.17× over Hermes-host；（5）Hermes vs TensorRT-LLM（LLaMA2-70B）：bs=1时达79.1%性能，bs=16时24.4%（Hermes成本$2,500 vs TensorRT-LLM $50,000）。

- 硬件平台是什么，配置是什么。
  - GPU: 单张NVIDIA RTX 4090 24GB GDDR6, 82.6 TFLOPS, 1321 Tensor TOPS (FP16), 936 GB/s带宽
  - NDP-DIMM: 8× DDR4-3200 DIMM（32GB/DIMM, 总256GB），每DIMM含1个NDP core（256 multipliers @1GHz GEMV unit + Activation unit + DIMM-link 25GB/s）
  - 互联: PCIe 4.0 64GB/s
  - Host CPU: Intel i9-13900K（运行scheduler: predictor, mapper, monitor）；Hermes-host baseline用i9-13900K计算cold neurons（89.6 GB/s内存带宽）

- 开源Serving框架是什么。修改了什么。
  - 论文未基于开源Serving框架（如vLLM, TensorRT-LLM）构建，而是以Huggingface Accelerate [22]和FlexGen [50]为主要对比baseline。Hermes调度器完全自研，关键设计：（1）将传统的GPU-CPU offloading扩展为GPU-NDP-DIMM异构计算，用NDP-DIMM的GEMV unit替代CPU作为cold neuron计算单元；（2）用lightweight predictor（token-wise + layer-wise）替代Deja Vu中昂贵的per-layer MLP-based predictor；（3）用window-based greedy scheduling替代FlexGen的offline zig-zag scheduling；（4）用DIMM-link（25GB/s）实现inter-DIMM直连通信替代宿主机中转（加速62×）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - Hermes系统未开源。以下为论文描述的执行流程。
  - 框架输入到硬件执行全过程（以LLaMA2-70B, bs=1, 输出第k个token为例）：
    1. **输入**: 已生成的token序列 [B, 1, H]（B=1, H=hidden_dim），scheduler的monitor持有前k-1个token的neuron state table。
    2. **预测阶段（Host CPU scheduler）**: Predictor读取neuron state table（4-bit per neuron），更新预测结果：token-wise（当前token的激活模式更新state）+ layer-wise（查前一层correlation table获取本层高概率激活neuron）。输出：每个layer的predicted activated neuron set。Neuron mapper据此调整hot/cold分区——neuron state > 10的新hot neuron从DIMM复制到GPU memory，state最低的GPU冷neuron被覆盖。同时更新neuron mapping table。
    3. **指令分发（Host CPU scheduler）**: Instruction queue为每层生成指令序列。GPU侧：cudaLaunchKernel触发各层的QKV generation（GEMM on hot neurons）和projection（GEMM）。NDP-DIMM侧：通过memory command interface发送MAC命令（cold neuron GEMV计算）和softmax命令（attention）。指令间插入同步barrier。
    4. **GPU执行（RTX 4090）**: 从GDDR6加载hot neuron权重（约占20%），Tensor Cores执行GEMM。QKV generation部分结果通过PCIe 4.0 (64GB/s)传输到DIMM。Projection全在GPU。GPU完成当前层hot neuron计算后进入下层的同步等待。
    5. **NDP-DIMM执行**: GEMV unit通过center buffer从DRAM bank读取cold neuron权重（内部带宽远高于PCIe），256 multipliers bit-serial处理FP16数据。Activation unit执行softmax（256 exp + 256 add + 256 mul + comparator tree + adder tree + divider）。Attention在DIMM侧执行（利用高内部带宽，KV cache存在DIMM中不用传回GPU）。Merge kernel在DIMM侧汇总GPU传来的hot neuron结果和本地cold neuron结果。Projection期间DIMM idle时执行online adjustment（neuron remap）。
    6. **Window-based remap**: 每5个token触发一次。Monitor统计各DIMM的window内Z_j（激活neuron数），按Z_j排序，配对max Z和min Z的DIMM，通过DIMM-link迁移最活跃neuron（约几KB），耗时<0.2%总时间。
    7. **输出**: 最后一个transformer layer完成后，LM head输出next token logits→采样→输出token。进入下一轮token generation。

## 80-CoServe- Efficient Collaboration-of-Experts (CoE) Model Inference with Limited Memory.pdf

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：CoServe 是一个面向有限内存设备的高效 CoE（Collaboration-of-Experts）模型推理 serving 系统，采用 PyTorch 实现，在异构 CPU-GPU 上运行。核心实现包含三大技术：(1) **Dependency-aware request scheduling（§4.2）**——预测每个 executor 队列添加新请求后的额外推理延迟（execution latency + expert switching latency），将请求分配到使总推理时间最短的队列；在队列内将使用相同 expert 的请求排列在一起（request arranging）；batch splitter 根据当前可用内存和最大可执行 batch size 将请求切分为多个 batch。(2) **Dependency-aware expert management（§4.3）**——两阶段 expert eviction 策略：第一阶段优先驱逐无 preliminary dependency 的 subsequent expert（按 memory footprint 降序），第二阶段按使用概率升序驱逐其他 expert。(3) **Offline profiler（§4.4-4.5）**——通过 microbenchmarks 获取各 expert 的最大 batch size、执行延迟、内存占用等性能矩阵，利用滑动衰减窗口在 CDF 上搜索最优 expert 加载数量，自动确定 GPU/CPU 的 memory allocation 和 executor 数量。
  - 实验比较：(1) 端到端 Throughput (img/s)：CoServe Best / CoServe Casual vs Samba-CoE / Samba-CoE FIFO / Samba-CoE Parallel，四个 task（Task A1/A2/B1/B2），NUMA (RTX 3080Ti) 和 UMA (Apple M2) 平台；(2) Expert switch 次数比较：CoServe vs 各 baseline 在四 task 上的 expert switching count；(3) Ablation study：CoServe None / CoServe EM（仅 expert management）/ CoServe EM+RA（+ request arranging）/ 完整 CoServe 的 throughput 和 expert switch count 分解；(4) Executor 数量 sensitivity：1G+1C 到 5G+1C 和 3G/4G+2C 配置下的 throughput；(5) Memory allocation 性能——滑动衰减窗口法在不同 expert 加载数量下的 throughput 曲线；(6) Scheduling overhead——请求调度延迟 vs 推理延迟 vs 预调度推理延迟对比；(7) Expert management overhead——占总时间的比例分析。

- 硬件平台是什么，配置是什么。
  - NUMA: GPU NVIDIA RTX 3080Ti 12GB, CPU Intel Xeon Silver 4214R, 16GB CPU Memory, SSD MTFD-DAK480TDS (530 MB/s read bandwidth)
  - UMA: GPU Apple M2, CPU Apple M2, 24GB unified memory, SSD APPLE AP0512Z (~3000 MB/s read bandwidth)

- 开源Serving框架是什么。修改了什么。
  - 框架: CoServe 基于 PyTorch 自研实现，baseline 对比 Samba-CoE [MICRO 2024]。
  - 修改/实现: (1) **Dependency-aware Request Scheduler**：基于 offline profiled 的执行延迟（K, B 常数）预测额外推理延迟，计算 `latency = K × (number_of_requests_in_batch) + B`，将新请求分配到 total inference time 最短的队列，请求在队列内排列到使用相同 expert 的请求之后。(2) **Batch Splitter**：实际 batch size = min(当前内存可容纳的最大 batch, offline profiled 的最大 batch size)。(3) **Expert Manager** 两阶段驱逐：Stage 1 驱逐无 preliminary 依赖的 subsequent experts（按 memory footprint 降序，逐出最少数量满足内存需求），Stage 2 按 usage probability 升序驱逐其余 experts。(4) **Offline Profiler**：运行 microbenchmarks 获取每个 expert 的 max batch size、execution latency (K, B)、memory footprint、loading latency，生成 CDF 曲线后通过滑动衰减窗口（`decay_factor = 1 - initial_window_value/100`）和线性预测（`f(N) = kN + b`）搜索最优 expert 加载数量。(5) **Expert Initializer**：按 usage probability 降序 round-robin 分配 experts 到各 executor 直至内存用完。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源: 论文未提供开源链接（ASPLOS '25 论文，截至评估时未发现公开代码仓库）。基于 PyTorch 实现。
  - 框架使用全过程（以 Circuit Board A inspection, Task A1, 2,500 requests, NUMA RTX 3080Ti, 3 GPU executors + 1 CPU executor 为例）：
    1. **Offline Phase（一次性）**：对每种 expert 架构（ResNet101, YOLOv5m, YOLOv5l），在 GPU/CPU 上分别运行 microbenchmarks——sweep batch sizes 测量执行延迟和内存占用 → 得到 K, B 常数（`latency = K × batch_size + B`）、max batch size（平均延迟 plateau 点）、loading latency、memory footprint。基于 routing rules 和组件分布计算各 expert 的 usage probability。运行滑动衰减窗口搜索最优 GPU expert 加载数量（如 Task A: 35 experts loaded in GPU memory）。
    2. **System Initialization**：Offline profiler 确定配置：GPU memory 分配（expert 加载 vs batch inference）、CPU memory 分配、executor 数量（3 GPU + 1 CPU）。Executor Creator 创建 3 个 GPU inference executors + 1 个 CPU executor。Expert Initializer 按 usage probability 降序加载 experts 到各 executor model pool（round-robin 分配到各 executor）。
    3. **Online Request 到达**：电路板组件图片以每 4ms 一张的速率持续输入。每个请求对应一个检测任务（classification + optional object detection）。
    4. **Dependency-aware Request Scheduling**：Scheduler 遍历所有 executor 的 request queue，对每个 queue 预测添加新请求后的额外推理延迟（`Δlatency = execution_latency + expert_switching_latency`）。选择使 `max(queue_total_time)` 最小化的 executor queue。若多个 queue 效果相同，选择 Δlatency 最小的。请求被分配到对应 queue 后，在队列中被安排到使用相同 expert 的已有请求之后（request arranging——group same-expert requests together）。
    5. **Request Splitting**：当 executor 准备好处理时，Batch Splitter 从 queue 头部取请求，若连续请求使用相同 expert 且累计 batch size 未超过 `min(available_memory_capacity, max_batch_size)`，则打包为一个 batch。一个 queue 可能拆分为多个 batch（如 {Req 0,1,3,6} → Batch 1, {Req 2,4,5,7} → Batch 2）。
    6. **Inference (expert 命中)**：若所需 expert 已在 model pool 中（GPU memory 内），直接执行 batched inference：加载 expert weights 到 GPU → 执行 ResNet101/YOLO 的 forward pass → 生成检测结果（defect/no-defect，或进入下一步 object detection）。
    7. **Inference (expert miss → Expert Switching)**：若所需 expert 不在 model pool 中，触发 expert switching。Expert Manager 检查可用内存。若内存不足：Stage 1——扫描 model pool 中的 subsequent experts（无 preliminary dependency 的），按 memory footprint 降序逐出，直到释放足够内存；若仍不足：Stage 2——按 usage probability 升序逐出剩余 experts。然后从 CPU memory 或 SSD 加载所需 expert 到 GPU memory，执行推理。Loading latency 已由 offline profiler 记录并用于调度预测。
    8. **CPU Executor 并行运行**：CPU executor 独立处理被分配到的请求（使用 CPU-optimized batch size），其 usage probability 较低、batch size 较小，与 GPU executors 并行工作以平衡负载。
    9. **Throughput 输出**：系统吞吐 = total_processed_images / wall_clock_time (img/s)。CoServe 在 Task A1 (NUMA) 上达到 28.7 img/s throughput，相比 Samba-CoE (3.0 img/s) 提升 9.4×；expert switch 从 513 次降至 68 次。
    10. **关键公式**：`additional_inference_latency = K × batch_size + B + expert_switching_latency`（同 expert 批处理时 expert_switching_latency = 0）。`decay_factor = 1 - initial_window_value / 100`。`f(N) = kN + b`（线性拟合预测 throughput 趋势）。窗口滑动终止条件：`|f(N+1) - actual_result| / f(N+1) > error_margin`。

## 8-SambaNova_SN40L_Scaling_the_AI_Memory_Wall_with_Dataflow_and_Composition_of_Experts.pdf

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Samba-CoE runtime，一个面向Composition of Experts (CoE) 模型部署的serving级runtime，核心机制包括：
    - **Dynamic Expert Activation**：150个独立编译的expert models（均源自Llama2-7B）的weights全部存储在DDR中。Router常驻HBM。当请求到达时，Router确定目标expert → CoE runtime检查expert是否已在HBM中 → 若不在，发起DDR→HBM DMA copy "activate" expert → transfer control到compiled kernel执行推理。类似传统动态链接器/加载器的设计。
    - **LRU Expert Eviction**：HBM中维护active experts的LRU cache。若新expert超出HBM容量，evict最旧的expert。编译器标注read-only weights（如model parameters），runtime跳过将其copy回DDR（无需回写只读数据，节省带宽）。
    - **Independent Expert Lifecycle**：每个expert独立开发、编译、训练、微调、量化、维护和serving——类似软件模块化。CoE runtime通过已编译model binary的metadata（precisely how much HBM and DDR space required）在运行时动态链接和切换experts。
    - **Batch CoE Execution**：Router以batch运行（BS=8）确定各prompt所需expert → 不同prompts可能需要不同experts → 所需experts批量从DDR→HBM → 每个(prompt, expert) pair顺序执行（prompt间无依赖）。
  - 实验比较：(1) CoE延迟对比（SN40L Node vs DGX A100 vs DGX H100）：BS=8/BS=1, 20/200 output tokens, 50-200+ experts；(2) 模型切换时间：SN40L DDR→HBM >1TB/s vs DGX A100 32GB/s (15× faster) vs DGX H100 64GB/s (16× faster)；(3) 机器footprint对比：随expert数量增加(10→850+)，维持TP8延迟所需的node数量（SN40L 1 node vs DGX 最多19 nodes）；(4) >150 experts时DGX OOM，SN40L可支持最多850 experts。

- 硬件平台是什么，配置是什么。
  - SN40L Node：8×SN40L RDU sockets + 1 host x86 CPU。Per socket: 638 BF16 TFLOPS, 64GB HBM, 1.5TB DDR。8 sockets aggregate: ~5.1 PFLOPS BF16, 512GB HBM, 12TB DDR。
  - 对比平台：DGX A100 (8×A100 80GB HBM) 和 DGX H100 (8×H100 80GB HBM)。DGX延迟为基于已发布Llama2-7B推理延迟[57]和DGX规格[58]-[61]的乐观估计（假设全部HBM和host memory可用于weights和KV-cache）。

- 开源Serving框架是什么。修改了什么。
  - Samba-CoE runtime为SambaNova自研闭源serving runtime。非基于开源serving框架修改。
  - 核心Serving设计：(1) CoE runtime运行在host CPU上，管理DDR/HBM allocation via low-level device driver；(2) 每个compiled model binary内含precise HBM/DDR memory requirements → CoE runtime动态分配DDR space；(3) Expert activation: copy HBM-intended memory segments from DDR→HBM → transfer control to compiled application binary → 执行 → return to CoE runtime waiting for next request；(4) LRU eviction: 尽量keep最多models active in HBM → capacity limit hit时evict oldest → compiler read-only annotations跳过weight write-back。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - Samba-CoE runtime为闭源商业软件。Samba-CoE (150 experts, ~1T params) 已部署于SambaNova生产系统。
  - **CoE Serving全流程（以Chatbot use case: 1 prompt → 20 output tokens, 8-socket SN40L Node为例）**：
    1. **System Initialization**：150个Llama2-7B experts + 1 Router的compiled binaries加载 → CoE runtime在DDR中为每个expert分配memory（包含compiler指定的HBM-intended segments）→ Router weights loaded into HBM → HBM中预留expert region。
    2. **Prompt Arrival**：User prompt到达host CPU → CoE runtime接收request。
    3. **Router Execution**：Router model (Llama2-7B, in HBM, TP=8) 执行推理 → 输出expert selection（如"math expert"）。
    4. **Expert Activation**：CoE runtime检查目标expert是否已在HBM LRU cache中。**Cache Hit**：立即transfer control到expert kernel, zero switching overhead。**Cache Miss**：若HBM有空间→ DDR→HBM DMA copy (~10ms for 7B model via >1TB/s aggregate bandwidth)。若HBM满→ LRU evict oldest expert (skip read-only weight write-back) → load new expert。
    5. **Expert Execution**：Activated expert以fused streaming dataflow kernel执行——decoder layer在PCUs/PMUs上pipeline → autoregressive decoding loop (20 iterations) → 每iteration从HBM stream weights和KV cache → PMU buffers → PCU compute → output token。利用expert weights在decoding loop中多次读取的temporal locality在HBM中缓存。
    6. **Response Return**：20个output tokens生成完毕 → control返回CoE runtime → runtime等待下一request。
    7. **Batch CoE Scenario (BS=8)**：8个prompts batch到达 → Router BS=8执行得到8个expert selections → 可能触发多个不同experts的DDR→HBM copy（sequential per unique expert）→ 各(prompt, expert) pair顺序执行解码 → 所有prompts完成后返回。
    8. **作用**：使系统在单8-socket node上serve 150+ experts的万亿参数CoE，模型切换延迟仅~10ms（vs DGX ~150-310ms），machine footprint降低最多19×，overall CoE推理速度提升3.7× (vs DGX H100) 到 6.6× (vs DGX A100)。

## 90-AUM- Unleasing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving.pdf

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：AUM——AU-aware资源管理器，基于xFasterTransformer构建（Python实现），作为系统组件运行。包含两个协作组件：(1) Background AU Profiler：离线将AU Usage Pattern（以算术强度ARI表征）、Frequency Interference（High-AU/Low-AU/None-AU三区域划分）和Resource Bound（L2/LLC/Memory Bandwidth最小需求）信息合成为离散AUV Model；(2) Runtime AU Controller：作为系统守护进程在线决策，包含三个stage——Slack-aware SLO Analyzer（FCFS调度prefill + LAG-based延迟分析decode）、Efficiency-aware Core Switcher（加权性能/功耗优化核心划分）、Collision-aware Allocation Tuner（监控AU性能并动态调整资源分配）。核心修改：(1) 新增AU使用的选择机制（根据ARI决定AMX vs AVX）；(2) 处理器核心三区域划分（CH/CL/CN分别对应高AU使用+低频率/低AU使用+中频率/无AU+高频率）；(3) 基于Intel RDT（CAT+MBA）动态调整LLC way分配和内存带宽分区。
  - 实验比较：(1) CPU性能功耗比效率（Perf-per-Watt）：AUM vs ALL-AU（exclusive）/SMT-AU/RP-AU，覆盖chatbot/code completion/summarization三种场景+Compute/OLAP/SPECjbb三种共享负载；(2) 不同硬件平台效率（GenA/GenB/GenC）with SPECjbb；(3) AU应用SLO保证（TTFT prefll + TPOT decode）：AUM vs ALL-AU/SMT-AU/RP-AU；(4) AUM消融实验（AU-UP/AU-FI/AU-RB variants）；(5) 资源分配CDF对比（LLC/Memory Bandwidth分配决策）。

- 硬件平台是什么，配置是什么。
  - GenA: Intel Sapphire Rapids Xeon 8475B, 48 cores/2 sockets, 25.6 TFLOPS AVX-512 / 206.4 TFLOPS AMX, 2.7 GHz base, 32KB L1-I/48KB L1-D/2MB L2 per core, 97.5MB LLC/socket, DDR5 1TB (233.8 GB/s)
  - GenB: Intel Sapphire Rapids Xeon Max 9468, 48 cores/2 sockets, 25.6/206.4 TFLOPS, 2.1 GHz base, 32KB/48KB/2MB, 105MB LLC, HBM 128GB (588 GB/s)
  - GenC: Intel Granite Rapids Xeon 6982P-C, 120 cores/1 socket, 32/344 TFLOPS, 2.8 GHz base, 64KB/48KB/2MB, 504MB LLC, MCR 768GB (600 GB/s)

- 开源Serving框架是什么。修改了什么。
  - 基础框架：Intel xFasterTransformer (xft) [21] with Intel AMX support + Intel oneDNN [29]
  - 修改点：(1) 新增Background AU Profiler（Python），记录AU使用模式（ARI）、频率降低规律（三区域划分）、资源需求（CAT/MBA profiling）；(2) 新增Runtime AU Controller（Python daemon），实现SLO分析 + 核心切换 + 资源分配调优三阶段决策；(3) 集成Intel RDT接口（pqos、CAT、MBA），实现动态LLC way和Memory Bandwidth分区；(4) 利用LAG指标量化decode阶段token生成进度，自适应调整AU配置

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：论文未明确说明开源链接（基于xFasterTransformer开源框架构建）
  - AUM Serving全流程（以chatbot场景, llama2-7b BF16, GenA SPR, SPECjbb共享为例）：
    1. **离线Profiling（部署前）**：Background AU Profiler在专用节点上运行重复实验：(a) Usage Profiling：遍历不同batch size下的prefill/decode阶段，测量AMX cycle ratio/µop ratio，计算ARI=6(1/d+3/BL)^{-1}(prefill)/6(1/d+3/B)^{-1}(decode)，确定High-AU/Low-AU阈值；(b) Frequency Profiling：测量不同核心数启用AMX时的核心频率下降，记录[CH,FH,CL,FL,CN,FN]三元组；(c) Resource Profiling：通过CAT/MBA接口测量不同LLC way和Memory BW分配下的AU性能，确定最小资源需求RAU。AUV Model按AU Bucket机制离散化——3 division × 3 sharing × 5 resource config × 10 repetitions共450次执行收敛。
    2. **Runtime初始化**：xft加载llama2-7b BF16量化模型到DDR5内存 → AUM Runtime Controller作为daemon启动 → 加载AUV Model到内存（~15MB）→ 设定SLO: dTTFT=250ms, dTPOT=100ms → AU应用性能价格α=1.8 (prefill), β=0.2 (decode)，共享应用价格γ=3e-5 (SPECjbb)。
    3. **Prefill Phase**：用户prompt到达（ShareGPT, avg input=755 tokens）→ SLO Analyzer采用FCFS调度prompt → SLOH=dTTFT - twait → Core Switcher分配High-AU区域核心（CH,启用AMX,频率~2.5GHz）→ xft执行GEMM操作（8192×4096×22016, AMX TMUL, 40.57 TFLOPS）→ Allocation Tuner监控TTFT，若P^m < SLOH则aggressive harvest资源（使用P^a平均性能），否则conservative返回资源（使用P^t tail性能）。
    4. **Decode Phase**：生成token → SLO Analyzer用LAG_i(token, Ti(t))=Σ(dTPOT - e_token)量化请求进度（LAG<0=>落后，LAG>0=>超前）→ SLOL=dTPOT + LAG_i → Core Switcher分配Low-AU区域核心（CL,启用AVX,频率~3.1GHz）→ xft执行GEMV操作（16×4096×22016, AVX, 3.87 TFLOPS）→ Allocation Tuner根据TPOT SLO动态调整LLC way和Memory BW分配。
    5. **共享负载执行**：SPECjbb在None-AU区域核心（CN,无AU,频率~3.2GHz全turbo）上运行 → Allocation Tuner优先harvest对AU性能影响最小的LLC资源给共享应用，其次调整高亲和度的Memory BW → 监控δ_AU = U_AU × P^m / SLO，若超过threshold(2)则触发Core Switcher重新划分核心区域。
    6. **效率输出**：E_CPU = (α×P_H + β×P_L + γ×P_N) / W_CPU，AUM vs ALL-AU实现平均8.8%效率提升，SLO violation减少7-11%，Runtime Controller决策延迟<1ms（查表），perf-per-dollar达GPU方案的88%。