## MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出MetaAttention attention runtime，将用户定义的attention变体自动生成跨后端优化kernel。核心kernel/runtime设计：(1) Parallel Pattern kernel：将relevance scoring实现为Q×K^T matmul、aggregation实现为scores×V matmul，采用online row-wise normalization technique（类似FlashAttention的online softmax）将score tile保持在on-chip memory，通过tiling遍历K/V sequence避免写出完整attention matrix到global memory；(2) Recurrent Pattern kernel：relevance scoring实现为Q×state matmul、aggregation实现为state += K[i]×V[i]状态更新，采用chunk parallelism将长序列切为可并行处理的chunk，在chunk内维护recurrent state，elementwise/reduction逻辑融合到recurrent kernel内；(3) RowNorm online接口：将行归一化拆为online_prologue（初始化row_max/row_sum状态）→online_forward（逐tile更新局部reduce结果、全局状态和对前tile rescale）→online_epilogue（完成最终输出），runtime在遍历K/V tile时同步更新归一化状态；(4) kernel templates实现：包含global→shared、global→register、shared→register三级数据搬移，以及输入位于shared/register时的matrix multiplication，调度计划确定后选择合适template并inline已lowering的customizable functions。实验比较：operator-level normalized latency，覆盖10种attention mechanism（Softmax Attention/DeepSeek-V2-Lite/LLAMA-3.1-8B/DiffTransformer-3B、Sigmoid Attention、ReLU Attention、Retention Parallel、Mamba2 SSM、Retention Recurrent、Gated Retention/RFA-Big/YOCO-13B、Multi-head Latent Attention/DeepSeek-V3、Sparse GQA/SeerAttention）。H100上：Softmax attention相对FlashAttention-3在Diff-Transformer-3B forward平均1.61× speedup（headdim_qk≠headdim_v无需padding）；customized parallel attention平均3.6× speedup（1.1×∼10.4× over FlashSigmoid/PyTorch）；recurrent attention forward/backward平均1.66×/1.78× over Flash-Linear-Attention；MLA性能接近FlashMLA且比MLA Triton快4.6×；Sparse GQA平均1.71× over SeerAttention Triton kernel。MI250上Softmax/ReLU/Mamba2/RetNet Recurrent subset forward 3.3×、backward 2.0× over baseline。

- 后端平台是什么，配置是什么。
  NVIDIA H100 SXM5（CUDA 12.4, Triton 2.3.1）：使用Tensor Memory Accelerator (TMA)异步加载、Tensor Cores MMA，基于TileLang和CUTE两种backend framework实现。AMD Instinct MI250（ROCm 6.2.4, Triton 3.1.0）：使用Matrix Cores + asynchronous copy units，基于TileLang backend实现。也支持MI300X。

- 评估性能的软件/脚本是什么。修改了什么。
  Operator-level benchmark脚本执行forward/backward latency测量。模型使用PyTorch/Transformers实现。end-to-end推理基于Transformers替换attention operator（H100单卡）。end-to-end训练基于TRL（H100单卡，seqlen 8k）。benchmark覆盖batch size 1/8，sequence length 2K/4K/8K，解码配置query seqlen=1+不同KV cache length。代码修改：MetaAttention runtime实现的attention kernel替代原有PyTorch/flash-attn等library的attention调用。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：GitHub https://github.com/SJTU-IPADS/MetaAttention，Zenodo DOI: 10.5281/zenodo.17701680。kernel使用流程：
  1. 用户定义attention模板（pattern+shape+customizable functions），MetaAttention frontend trace生成computation graph
  2. Scheduler搜索IntermediateTensor tile/mem/pipeline stage配置，生成scheduling plan
  3. Runtime根据plan选择kernel template（parallel/recurrent pattern），inline customized functions，生成后端代码
  4. 以MLA解码为例：Parallel Pattern，query seqlen=1，head=128，head_kv=1，dimqk=576，dimv=512，KV cache length 2048-8192→每个tile加载Q+一段K/V cache→计算relevance score→on-chip执行custom Mod/RowNorm online→聚合V得output→遍历后续KV tile。全部计算在单一fused kernel内完成

