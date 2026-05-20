## Accelerating Sparse Transformer Inference on GPU (STOF)

- 属于编译框架的实现是什么？实验比较什么？
  提出Operator Fusion Module，实现sparse Transformer下游dense operator的自适应编译融合。关键设计：(1) Fusion Scheme Converter：用hash encoding将抽象fusion pattern量化为binary numerical expression——通过convolutional subgraph analysis (neural hashing)发现频繁子图→predefined rules提取初始fusion scheme→binary hash code表示（每位对应一个operator，相同数字表示属于同一fused segment，不同数字表示fusion boundary），可转换为hexadecimal压缩格式；numerical decoding将binary code映射到Triton/TileLang compilation template（如template_gemm_layernorm、template_gemm_gemm、template_add_layernorm），template内部分解tensor操作为tile级操作、利用warp-level primitives做高效reduction、应用multi-stage pipelining重叠memory access和计算；(2) Two-stage Search Engine：Stage 1 Fusion Expansion基于expand/seize/compete三条规则用DFS逐步扩大fusion边界（最多每个segment两个CI operator），每次扩展后随机采样参数配置，比较pre-fusion vs post-fusion性能，有gain则保留否则回退，performance cache避免重复尝试；Stage 2 Parameter Sampling用reward-based算法动态分配各segment采样数——首轮各segment等量采样，后续对带来最高overall gain的segment增加采样配额；(3) Template实现：基于Triton和TileLang，对每个fused operator pattern选择性能更优的backend。实验比较：ablation study区分MHA module vs fusion module贡献（A100，Bigbird mask）——小输入scale (1,128)时fusion module贡献更高（比仅MHA module高19.5% speedup），大输入scale (16,2048)时MHA module主导（仅MHA module平均2.0×）。end-to-end tuning time对比MCFuser和Bolt（A100）——STOF平均快6.7×（MCFuser）和6.9×（Bolt）。

- 硬件平台是什么，配置是什么。
  NVIDIA RTX 4090 (Ada Lovelace, 24GB)、NVIDIA A100 (Ampere, 80GB)。Ubuntu 22.04, CUDA v12.6, PyTorch 2.7.0。

- 开源编译框架是什么。修改了什么。
  基于Triton和TileLang两种high-level programming interface实现compilation template。两者均提供tile-level编程抽象，便于template derivation以支持更广泛的fusion范围。修改：STOF在Triton/TileLang之上实现：(a) compilation template library——针对CI+MI (GEMM+Layernorm)、CI+CI (GEMM+GEMM chain)、MI+MI (Add+Layernorm)等fusion pattern手写template，template内部使用tile decomposition+data reuse+warp reduction+multi-stage pipeline优化；(b) template与hash code的双向映射——numerical decoding解析binary expression确定使用哪个template；(c) PyTorch FX graph manipulation——通过操作fx.GraphModule对象实现graph capture和node replacement（将fused nodes替换为compilation template调用）；(d) 整体兼容torch.compile，可复用其compilation optimizations。论文无对Triton/TileLang compiler本身的修改。

- 开源情况。编译框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：Zenodo artifact DOI: 10.5281/zenodo.17705801。STOF编译融合使用流程：
  1. Compilation template定义：对每种fusion pattern（如GEMM+Layernorm MI+CI fusion）手写Triton/TileLang template kernel，template.Config暴露block_size/num_stages/num_warps等关键参数作为tuning search space
  2. Fusion scheme表达：遍历PyTorch FX computational graph→neural hashing (F_hash∘F_conv(G)) 发现频繁子图→predefined rules（如小batch/short seq下优先fuse GEMM chain）生成initial scheme→hash encoding将scheme转为binary array [0 1 0 0 0 0 0 1 1 0 1 1 1 0 0]，其中相同bit值表示同一fused segment
  3. Numerical decoding：解析binary array→识别各segment的operator组成→映射到对应compilation template（如segment [1 1 1]→template_gemm_layernorm）
  4. Two-stage tuning：(a) Fusion expansion：DFS遍历expand/seize/compete规则扩展每个segment边界→每次扩展后采样N个参数配置（pre-fusion和post-fusion各N个）→取best比较→有gain保留否则回退→cache记录已尝试scheme避免重复；(b) Parameter sampling：固定总采样数→首轮等分→后续对贡献最大gain的segment增加配额→cache避免重复参数配置
  5. 例如BERT-Base在RTX 4090上(batch=1, seq_len=128)Bigbird mask：fusion module主导加速（贡献约19.5% more speedup than仅MHA module），因为小workload下CI operator fusion（如GEMM+GEMM chain）特别友好，而MHA compute不是瓶颈

