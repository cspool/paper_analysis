## Accelerating Sparse Transformer Inference on GPU (STOF)

- baseline方法是什么？
==mask的每列对应K的token向量（KT的列向量），Q的token向量（Q的行向量）。==
![[Pasted image 20260519141319.png]]
Baseline分为两类：
- (1) **MHA fused kernel baseline**：QK-HBM-SFX-HBM-PV
	- FA2/FA3仅支持causal和sliding window等连续mask pattern，无法处理discrete/unstructured分布
		- 如random attention：每个q的局部+全局随机key/token/block
		- Bigbird：sliding+random+Global行；
	- FlashMask使用column-wise representation(mask稀疏分布的张量表示和QK计算)，但仅支持columns上element连续的mask，无法表示discrete分布；
	- FlexAttention支持任意mask但constrained to fixed optimizations仅达到suboptimal性能（**自定义稀疏算子框架baseline**）；
	- ByteTransformer自定义kernel限制max seq_len=1024；
	- SPLAT专注regular sparse kernels (R-SDDMM：计算稀疏QK。R-SpMM：计算稀疏PV)，kernel特定优化，不直接兼容MHA的融合优化；
	- PyTorch Native/MCFuser/ByteTransformer不支持sparse mask，需先做mask subtraction（减inf）再做full GEMM，无法减少QK的计算量（保底通用措施）。

- (2) **Operator fusion baseline**：考虑MHA之间的中间数据传输优化
	- PyTorch Compile/AStitch仅fuse MI operators（访存密集MI+MI），CI operators（计算密集）分开用vendor library；
	- Welder/DNNFusion fuse CI+MI但受限于category-based规则；
		```
		`for segment in graph:`
			`if all(op.is_memory_intensive() for op in segment):`
				`fuse_with_MI_template(segment)`
			`elif has_GEMM(segment) and has_elementwise_or_norm(segment):`
				`fuse_with_CI_MI_template(segment)`
			`elif is_gemm_chain(segment):`
				`fuse_with_CI_CI_template(segment)`
			`else:`
				`leave_to_vendor_library(segment)`
		```
	- Chimera/MCFuser fuse CI+CI chain但忽略硬件细节（bank conflict等），long sequence下性能差；
	- Bolt基于CUTLASS template（gemm+add+LN、gemm+gemm+act、add+ln等模板）但fusion range扩展困难。
	- 所有baseline的共性缺陷：
		- 固定operator fusion scheme（算子级规则GEMM+LN），无法根据tensor dimensions（batch size/seq_len/hidden_dim）自适应——例如GEMM+Layernorm fusion在hidden_dim=512时最高加速39.1×，但在hidden_dim=1024时反而显著减速。
		- 此外，individual operator tuning的最优参数配置无法直接迁移到fused operator（因为search space fundamentally不同），operator-by-operator sequential tuning（单算子分别tuning后直接应用）会使GEMM+Layernorm在A100上平均仅达2.4×而非post-fusion tuning的10.1×。

  全栈execution例子（以FA2在A100上跑BERT-Base + Bigbird mask、batch=16、seq_len=4096为例）：
  - 算法层：Bigbird mask (80.8% sparsity, unstructured) → FA2不支持Bigbird mask，PyTorch Native fallback为先GEMM得full score matrix→mask subtraction（将mask位置score设为-inf）→Softmax→GEMM，无计算量节省
  - 系统框架层：PyTorch fx graph capture MHA subgraph，但因mask不支持fused kernel（**无法直接调用fused kernel**），subgraph被拆分为fine-grained meta operators逐op执行
  - 编译框架层：PyTorch Compile对meta operators做通用compilation optimization（constant folding + instruction scheduling），CI operators用cuBLAS library，MI operators做通用fusion（**底层细粒度fusion影响局部但通用**）
  - kernel调度层：FA2 kernel仅支持causal/sliding window mask → fallback到cuBLAS GEMM + Softmax + cuBLAS GEMM多kernel launch，每次intermediate result write back to HBM再reload
  - 硬件架构层：NVIDIA A100 (108 SM, 80GB HBM2e)，大量HBM读写intermediate results → memory bandwidth成为瓶颈

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**STOF (Sparse Transformer Optimized Framework)**，通过customized MHA kernels + adaptive operator fusion实现sparse Transformer inference在GPU上的全栈优化。

  论文方法全栈execution例子（以STOF在A100上跑BERT-Base + Bigbird mask、batch=16、seq_len=4096为例）：
  - **算法层**：Bigbird mask (80.8% sparsity, unstructured) → **two-level storage format**表示为full OTs (BSR) + part OTs (64×uint64 bitmap_mask)，任意mask pattern统一表示。Mask sparsity直接转化为计算量跳过（仅加载/计算valid OTs）
![[Pasted image 20260519160511.png]]
  - **系统框架层**：STOF将模型分为MHA structure（sparse，custom kernel处理）和downstream operators（dense，template-based fusion处理）→ torch/cpp_extension将MHA kernel封装为PyTorch native function→ fx.GraphModule operate downstream graph
==分别处理MHA和其他算子的计算图优化，MHA优化不同mask的fusion，其他算子优化张量尺寸自适应的fusion。==
  - **编译框架层**：neural hashing发现频繁子图→predefined rules生成initial fusion scheme→hash encoding转为binary array→numerical decoding映射到Triton/TileLang compilation template→two-stage search engine确定optimal fusion scheme + kernel parameter
![[Pasted image 20260519165408.png]]
![[Pasted image 20260519165428.png]]
  - **kernel调度层**：Block-wise kernel（因seq_len=4096，valid OT ratio适中）：Q_i在register中resident→仅加载valid OTs的K_Tj/V_j→cp.async异步加载V_j与GEMM重叠→part OTs用bitmap_mask精确mask 8×8 IT→Softmax with scaling factor α做跨OT reduction→最终write back HBM。OT行主序（Softmax迭代）、IT列主序（bank conflict-free）、Q register resident（避免SMEM重复读写）
  - **硬件架构层**：NVIDIA A100 (108 SM, 80GB HBM2e) → block-wise kernel利用Tensor Cores (8×8 IT对齐mma.m16n8k16)，SMEM double buffering重叠memory access与compute → 端到端相对FA2加速~4.8×（(16,4096)时）
![[Pasted image 20260519164552.png]]

  对应解决Baseline缺陷：
==MHA的不同mask的统一fusion kernel设计，支持bigbird mask==
  **(1) 现有MHA fused kernel无法表示任意mask pattern → two-level storage (BSR+bitmap)**：FlashMask仅支持column-continuous mask（4-array column-range表示无法处理discrete分布），FlexAttention fixed optimizations suboptimal。STOF通过OT/IT两层抽象：OT级BSR（full_row_ptr/full_col_idx + part_row_ptr/part_col_idx）表示全局skip blocks，IT级64×uint64 bitmap_mask表示block内精确element分布。causal (50% sparsity)到Bigbird unstructured (80.8% sparsity)均可统一表示，且存储格式本身直接驱动kernel skip逻辑（load_row_ptr/load_col_idx告知加载哪些OT、bitmap_mask告知IT内哪些element需mask）
  **(2) MHA kernel缺乏灵活kernel选择 → analytical model驱动的kernel selection**：row-wise kernel（小seq_len+高稀疏→row-sliced Q并行+warp内shuffle无sync overhead）vs block-wise kernel（大seq_len通用场景→partition Q/K/V到SMEM利用memory hierarchy）。公式1基于valid OT ratio和seq_len计算threshold（log penalty压制extreme sparse长seq），自动选择最优kernel

==从CFG提取高频子图（conv+hash聚类）、考虑张量尺寸来将子图匹配到初始fusion模式和初始segment划分、将初始segment划分进行二进制编码（优化空间和起点）、每个可能fusion对应一个template kernel（triton、TileLang）、对优化空间（二进制编码）进行二阶段搜索/优化（基于规则的相邻segment合并+固定fusion下的参数搜索）。==

  **(3) Fixed operator fusion scheme无法适应diverse input scales → hash encoding + two-stage search**：Fixed category-based fusion（MI-only/CI+MI/CI+CI）在hidden_dim从512变为1024时可能从16.5×加速变为slowdown。STOF用hash encoding将fusion scheme表达为searchable binary expression（任意scheme可表示），two-stage search（fusion expansion through expand/seize/compete rules→parameter sampling with reward-based allocation）自动发现per-input-scale最优fusion方案。Tuning time比MCFuser快6.7×、比Bolt快6.9×  
  **(4) Individual tuning参数不transfer到fused operator → co-tuning fusion scheme + kernel parameters**：直接复用individual operator最优参数到fused operator导致Bias+Layernorm仅2.4×而post-fusion tuning达10.1×（A100）。STOF的two-stage search在每次fusion expansion时对pre-fusion和post-fusion分别采样参数比较，performance cache复用避免重复，确保fusion scheme和kernel parameters在hierarchical space中co-optimized


==triton、TileLang的template优化==
  **(5) Downstream operator fusion忽略硬件细节 → compilation template with tile-level optimization**：Chimera/MCFuser的loop-based construction忽略bank conflict等GPU硬件细节。STOF的Triton/TileLang template内部：tile decomposition最大化data reuse、warp-level primitives做高效reduction、multi-stage pipeline重叠memory/compute、仅暴露关键参数（block_size/num_stages/num_warps/blkM/blkN/blkK）作为search space，既保证硬件效率又限制搜索复杂度

==MHA的kernel优化==
  **(6) Long sequence下baseline OOM → MHA compute skipping + memory saving**：seq_len 32k时所有baseline (PyTorch Compile/ByteTransformer/MCFuser)均OOM，STOF运行正常（64k才OOM）。seq_len 16k时STOF相对PyTorch Compile加速16.8×
