## MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends

- baseline方法是什么？
  Baseline方法分为三类：(1) **手写专家库**：FlashAttention-2/3 for Softmax Attention（~2.7k行CUDA，针对标准headdim相同、softmax归一化、H100/A100优化）、FlashSigmoid for Sigmoid Attention（~1.9k行CUDA）、FlashMLA for DeepSeek MLA（~1.7k行CUDA、blockSize=64）、Mamba2 chunk kernel for Mamba2 SSM（~3k行Triton）、Flash-Linear-Attention v0.2.0 for Gated Retention/RetNet Recurrent（~0.4k行Triton）。全栈执行例子：算法层标准Softmax Attention Q×K^T→softmax→×V→Serving/系统框架层FlashAttention-3 kernel在H100上通过TMA异步加载、Tensor Cores MMA、register-level pipelining→编译框架层手写CUDA kernel无自动生成→kernel调度层固定tiling/pipeline策略→硬件架构层NVIDIA H100/A100 GPU。缺陷：(a) 仅支持特定attention变体——Gated-RetNet和ReLU-Attention上现有库表现极差或不支持（Fig.2：FLOPS utilization仅2.9%-7.1% vs Softmax-Attention的65.3%）；(b) 硬件移植成本高——FlashAttention v2在A100达70% peak throughput但H100仅30%，需register-level pipelining和ping-pong kernel design才能发挥H100性能；(c) 非标准shape不支持——如DeepSeek MLA的dimqk≠dimv、RetNet的非标准embedding维度、query seqlen=1解码场景；(d) 开发成本高——每种新attention变体需要专家手写上干行kernel代码（Table 5）。

  (2) **通用DL compiler**（Torch Inductor、TVM、Ansor-AF、Welder、Alcop、TensorRT）：将attention作为不透明算子序列做operator fusion。全栈执行例子：算法层attention展开为matmul+softmax+matmul→编译框架层Torch Inductor/TVM识别算子图并尝试fusion→kernel调度层生成fused kernel→硬件架构层GPU。缺陷：(a) 不理解attention语义——无法自动推导online softmax、chunk parallelism、memory-efficient pipelining等attention特有优化；(b) 在attention workload上性能远低于手写库（Fig.2：Torch Inductor在Softmax-Attention仅14.1% FLOPS utilization vs FlashAttention-3的65.3%）；(c) Ansor-style auto-tuning编译时间长。

  (3) **模板/接口型方法**（FlexAttention、FlashInfer）：预定义大部分attention计算，开放有限参数或user code injection。全栈执行例子：算法层parallel attention→编译框架层FlexAttention/FlashInfer暴露score_mod等有限callback→kernel调度层预编译fixed-pattern kernel→硬件架构层GPU。缺陷：(a) 仅支持parallel pattern——无法覆盖recurrent/linear attention（Mamba2、RetNet、YOCO等）；(b) 接口灵活性有限——非标准tensor shape（dimqk≠dimv、MLA head/head_kv、chunk-based state维护等）超出接口能力。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**MetaAttention**：统一attention抽象框架，通过relevance scoring + aggregation两个核心操作+ customizable functions模板，结合IntermediateTensor-based两层调度策略自动生成跨硬件后端优化kernel。

  论文方法全栈执行例子（以RetNet Parallel Attention在H100上为例）：
  - 算法层：用户选择Parallel Pattern→声明Q[batch,head,seq_len,256]/K[batch,head,seq_len_kv,256]/V[batch,head,seq_len_kv,512]→定义scores_Mod（scores*mask缩放）+scores_RowNorm（reduceAbsSum-based row normalization）→MetaAttention将RetNet的retention机制映射为relevance scoring（Q×K^T matmul）+Mod+RowNorm+aggregation（scores×V matmul）
  - 编译框架层：frontend trace customizable functions→tensor DAG（elementwise节点→SIMT fusion，row-reduce节点→intra-warp reduction）→scheduler enumerates output tile sizes（如[64,128]）→propagate tiles to Q/K/V/scores/state→TileResourceScheduling分配memory tier和pipeline stage→H100 DeviceConfig约束（basetile 128×128，register 65536×32bit，shared mem 228KB）→生成scheduling plan
  - kernel调度层：runtime选择Parallel Pattern kernel template→inline customized Mod/RowNorm online（online_prologue初始化row_sum_wo_clamp=0,row_sum=0→online_forward逐tile计算reduceAbsSum+更新全局row_sum+对前一tile rescale）→TMA异步加载K/V tile到shared memory→Tensor Cores执行Q×K^T MMA→on-chip执行customized functions→Tensor Cores执行scores×V MMA→遍历所有KV tiles
  - 硬件架构层：NVIDIA H100 SXM5（CUDA 12.4），利用TMA硬件单元异步加载、Tensor Cores做MMA、shared memory做tile缓存

  以Mamba2 SSM Recurrent Attention在H100上为例：
  - 算法层：用户选择Recurrent Pattern→声明Q/K/V shape、headv=80,dimqk=128,dimv=64→relevance scoring为Q×hidden_state matmul→aggregation为hidden_state += K[i]×V[i]
  - 编译框架层：scheduler将hidden_state、Q/K/V tile、临时Mod结果建模为IntermediateTensor→搜索chunk parallelism的chunk size和state tile memory placement→scheduling time约82秒
  - kernel调度层：runtime使用chunk parallelism将长序列切为并行chunk→chunk内维护recurrent state→elementwise+reduction逻辑融合到recurrent kernel→single fused kernel执行

  对应解决Baseline缺陷：
  (1) **手写库仅支持特定attention变体** → unified attention abstraction覆盖Parallel和Recurrent两种pattern，customizable functions（Mod/RowNorm/RowNorm online）表达Softmax/Sigmoid/ReLU/RetNet/Mamba2/MLA/Sparse GQA等变体。Table 3展示10种attention mechanism均可在MetaAttention中实现。自定义attention仅需22-90行代码（vs手写库0.4k-3k行）。
  
  (2) **手写库硬件移植成本高** → DeviceConfig + IntermediateTensor scheduling自动适应硬件差异。H100上使用TMA+Tensor Cores+CUTE/TileLang，AMD MI250上使用Matrix Cores+TileLang。编译时间控制在分钟级（46-89秒），无需为每种新GPU手写不同kernel。MI250上avg 3.3× forward/2.0× backward speedup over baseline证明跨后端能力。
  
  (3) **手写库不支持非标准shape** → MetaAttention不要求headdim_qk=headdim_v或特定head layout。Diff-Transformer-3B（dimqk=128≠dimv=256）上相对FlashAttention-3平均1.61× speedup（FA3需padding到同维度）。MLA（head=128, head_kv=1, dimqk=576≠dimv=512, query seqlen=1）上性能接近FlashMLA且比MLA Triton快4.6×。
  
  (4) **通用compiler不理解attention语义** → MetaAttention显式建模relevance scoring/aggregation/online RowNorm/recurrent state语义，支持online softmax（RowNorm online接口：prologue/forward/epilogue三段式）、chunk parallelism（recurrent pattern自动分块）、on-chip fusion（IntermediateTensor memory tier控制）。H100上相对PyTorch Inductor获得大幅speedup。
  
  (5) **模板库仅支持parallel pattern** → MetaAttention并行支持parallel和recurrent两种pattern。Mamba2 SSM forward/backward平均1.66×/1.78× over Flash-Linear-Attention。RetNet Recurrent、YOCO-13B、RFA-Big等recurrent变体均被覆盖，而FlexAttention/FlashInfer不支持。
  
  (6) **开发效率低** → attention-specific programming interface（pattern选择+shape声明+Mod/RowNorm/RowNorm online函数定义）将开发代码量从数百至数千行降至数十行（Table 5：MLA 90 vs 1700 LoC，Mamba2 27 vs 3000 LoC）。

  Trade-off：(a) MetaAttention用受约束的attention-specific model换取可优化性——不是任意Python/Tensor IR compiler，假设attention可分解为relevance scoring+aggregation+有限类型customizable functions；(b) scheduling time为分钟级（46-89s），短于Ansor等传统auto-tuning compiler，但仍不是零成本即时编译；(c) 在已有高度优化手写库且shape完全匹配的场景，目标为comparable performance，主要优势来自变体/非标准shape/跨后端/开发效率。
