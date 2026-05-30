## FastTree Optimizing Attention Kernel and Runtime for Tree-Structured LLM Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  FastTree 的 GPU attention kernel 和 tree structure-adaptive runtime 属于 kernel 调度/运行时计算优化。实现包括：(i) tree-structured attention kernel——将按 radix tree 共享 KV cache 的 queries 聚合为 context-queries groups，在单个 kernel 中并行处理不同 group，每个 group 内部以 Flash-Attn 风格 tile-by-tile 执行 attention，Q tile 跨 block 并行化、KV tile 在 block 内串行迭代，利用 online softmax 存储中间结果于 shared memory；(ii) greedy heuristic runtime——以 BFS 遍历 radix tree，对每条边比较 SplitKVCost（分离）和 SplitQCost（拼接）的估计开销（padding cost + intermediate result cost），贪心选择开销更小的边赋值，生成 virtual tree 后做 node-centric query aggregation；(iii) multi-phase tiling——靠近 root 的节点聚合 query 多，用大 tile size 最大化 KV 复用；靠近 leaf 的节点 query 少，用小 tile size 避免 shared memory 浪费、提升 SM occupancy；(iv) GPU-efficient long context splitting——当 group 级并行度不足或存在极长 tail context 时，split 超阈值长度的 context 提升 GPU occupancy，虽引入 intermediate result reduction overhead 但被 SM utilization 提升所抵消；(v) 最后 launch 一个 lightweight reduce kernel，利用 LogSumExp vectors 做跨 group 的中间结果 rescaling。
  实验比较：(i) kernel benchmark——FastTree vs FlashAttention v2.6.3、FlashInfer v0.1.6、SGLang Triton kernels、DeFT、Multi-Level Cascade Attention（CascadeAttn），测量 attention kernel execution time，覆盖多种 tree shape（N=node number per level, C=context length per level）和 GQA ratio（1/4/16），head dim=128；(ii) end-to-end——FastTree+SGLang vs SGLang-Triton vs SGLang-FlashInfer，在 Llama-2-7B 和 Mistral-7B 上测量四个 benchmark（multi-level system prompt、multiple few-shot learning、multi-chain reasoning、multi-document QA）的 throughput；(iii) 消融——无 greedy heuristic 的 naive aggregation vs full FastTree、long context splitting 的单独影响。

- 后端平台是什么，配置是什么。
  NVIDIA H100 GPU (80GB)，CUDA 12.2。论文当前实现未使用 Hopper 特有特性（如 TMA）。Triton 3.0.0。FlashInfer v0.1.6。SGLang v0.2.13。

- 评估性能的软件/脚本是什么。修改了什么。
  - Kernel benchmark：自编 Python 脚本，用 N（每层 node 数）和 C（每层 per-node context length）两个数组生成不同 tree shape 的 KV cache，如 N=[1,2,4], C=[128,32,32] 表示 3 层共 7 个 node 的 tree。测量各配置下 attention kernel 执行时间（ms），结果以 normalized speedup 展示。
  - End-to-end：在 SGLang v0.2.13 上集成 FastTree，替换原有 attention backend（Triton/FlashInfer），测量 batch=128、gen=256 tokens 的 throughput（tokens/s），breakdown 分析 decoding latency 和 GPU kernel execution time。
  修改：(i) 实现 FastTree attention kernel（Triton），支持 Flash-Attn 风格的 tiled attention + query aggregation；(ii) 实现 tree structure-adaptive runtime（Python/C++），包括 BFS greedy heuristic 和 long context splitting；(iii) 集成到 SGLang 的 attention 路径，decode 阶段使用 FastTree kernel，prefill 阶段沿用 FlashInfer。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  已开源：https://github.com/PanZaifeng/FastTree-Artifact（Apache-2.0）。提供 Docker 环境，含 CUDA 12.2、PyTorch 2.4.0、Triton 3.0.0、FlashAttention 2.6.3、FlashInfer 0.1.6、SGLang 0.2.13。运行 `kernel_bench/run.sh` 可一键复现 kernel benchmark 结果，生成 `kernel_bench/norm_perf.pdf`。

  评估原理与流程（以 N=[1,2,4], C=[128,32,32], GQA=1 的 kernel benchmark 为例）：
  1. **输入**：Tree shape (N, C) 定义 radix tree 结构——N=[1,2,4] 表示 3 层分别有 1、2、4 个 node；C=[128,32,32] 表示各层 context length。总 node 数=1+2+4=7。每个 leaf node 关联若干 queries（模拟不同请求），leaf 间共享路径上的祖先节点 KV cache。
  2. **Runtime 处理**：FastTree runtime 读取 radix tree → BFS 遍历各节点 → 对每条 parent→child 边贪心决策：
     a. SplitKVCost (edge=0, 分离): 计算 parent node v 的 padding cost C_P(nQ_curr, len_v) + child node l 的 padding cost C_P(nQ_l, len_l) + intermediate result cost γ·nQ_l·d
     b. SplitQCost (edge=1, 拼接): 计算分离后 v 的 padding cost C_P(nQ_curr - nQ_l, len_v) + 拼接后 l 的 padding cost C_P(nQ_l, len_v + len_l)
     c. 比较两 cost，选更小的赋值。若拼接则更新 nQ_curr -= nQ_l 和 L[l] += len_v。
  3. **Virtual tree 生成**：根据边赋值生成 virtual tree。拼接边对应的 node 被"复制"后拼接到不同 leaf。最终通过 node-centric query aggregation 生成 (context, {queries}) groups list。
  4. **Attention kernel 执行**：单 kernel launch 并行处理所有 groups。每个 group 分配到一个 block set：
     a. Tile division：Q 矩阵按 tile size 沿 query dim 划分 → 不同 block
     b. For each block：copy Q tile + 首 KV tile from HBM → shared memory
     c. Loop over KV tiles：每次 load 下一个 KV tile → BMM1(QK^T) on tensor core → online softmax (max update + exp + rowsum) → BMM2(PV) on tensor core → 更新 partial O + L (in shared memory)
     d. Write partial results (O, L) to HBM
  5. **Reduction kernel**：对每个 query，跨 groups 的 partial O_i 用 L_i rescale 后累加得到 final attention output。
  6. **Performance measurement**：CUDA event 记录 kernel launch 到 completion 的 wall time。Compare FastTree vs baselines（FlashAttention/FlashInfer/SGLang Triton/DeFT/CascadeAttn）。FlashAttention 因 decode 阶段 GEMV 仅 <1% effective computation；FastTree 通过 query aggregation 将 GEMV 转 GEMM，tensor core utilization 大幅提升，且共享 KV tile 仅在 shared memory 加载一次（非 HBM 重复加载）。
  7. **输出**：Normalized speedup plot（Figure 9），FastTree 平均 5.1× over FlashAttention, 4.2× over FlashInfer, 10.6× over DeFT。

  Long context splitting 评估原理：在 N=[1,10], C=[4000,400] 等 GPU 欠饱和配置下，开启 splitting 可获 up to 1.9× speedup。Splitting 将超长 context node 沿 context dim 切分，增加 group 级并行度以填满 SM，虽引入中间结果 reduction 开销但被 occupancy 改善抵消。
