## PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  PAT实现了一套prefix-aware decode attention kernel，遵循pack-forward-merge执行范式：(1) Pack阶段：pack scheduler将vLLM block table转为prefix tree，用memory-centric profit model比较split/merge策略，生成CTA partition以最小化global KV cache访问，lazy update机制让调度结果在block table未变化时跨continuous-batching iterations复用并与pre-attention tasks重叠；(2) Forward阶段：multi-tile kernel为每个CTA从offline求解的可行tile set中选择合适的(m,n) tile配置（m为Q tile size，n为KV tile size），runtime tile selector以constant-time lookup选m（round-up规则覆盖当前CTA query数）和n（根据KV length和execution bubble trade-off选择），multi-stream forward为不同tile配置创建独立CUDA stream并行执行，long-KV split将KV超长CTA沿KV维拆分；(3) Merge阶段：lightweight merge kernel用online softmax合并每个query被多个CTA计算出的partial max score、log-sum-exp accumulator和partial value-weighted sum为最终输出。Kernel约3k行Cutlass/CuTe+C++实现，数据搬运使用cp_async+double buffering。实验比较attention latency和memory read/write量，对比FlashAttention v2.5.9（query-centric, tile固定(64,128)）、FlashInfer v0.2.5（query-centric, dynamic CTA partitioning, tile(16,128)）、FastTree（KV-centric, compute-oriented cost model, 两种tile configs）、RelayAttention（KV-centric, pack first-level prefix）、RelayAttention++（扩展版使用vLLM-style KV-cache reuse+L2 cache）、DeFT（KV-centric, fixed(32,16) tile）、Cascade Inference（KV-centric, fixed packing）。Synthetic batch下PAT相对FlashAttention最高21.5×、相对FlashInfer最高11.7×、相对FastTree最高3.2×加速，平均降低attention latency 53.5%。每配置重复20次取平均。

- 后端平台是什么，配置是什么。
  NVIDIA A100-SXM4-80GB (108 SM, 40MB L2, 80GB HBM)，NVIDIA H100-SXM4-80GB (132 SM)。CUDA 12.4, PyTorch 2.7.0, vLLM v0.9.0。Head configs: (#heads, #kv_heads) = (64,8), (32,8), (16,8), (32,32)，head dimension=128, FP16。端到端模型：Qwen3-8B、Llama-3-8B（A100单卡）；Qwen2.5-72B-Instruct（4×A100 TP=2/PP=2）；Qwen3-30B-A3B（单卡A100）。

- 评估性能的软件/脚本是什么。修改了什么。
  NCU profiling测量FlashAttention KV cache traffic（比理论最小值高4.3-8.7×）；PTX profiling分析CTA execution pipeline和execution bubble。Kernel benchmark脚本构造synthetic decode batch：B定义prefix tree结构和leaf数（如B=[1,4,16]表示两级shared prefix和16 leaves），L定义各层KV长度（如L=[128,256,1024]），共20种(𝐵,𝐿)配置。Multi-tile kernel基于Cutlass/CuTe实现MMA，tile配置通过offline solver基于三个约束导出：① register/shared memory约束（上界：m*h*b + n*h*b + 中间结果 ≤ S_smem，offline编译获取R_thr和R_CTA），② bandwidth lower bound（n ≥ LB/(S*C*h*b)确保inflight data覆盖memory latency），③ CUTLASS constraint（m,n为2的幂且≥16）。A100上得到11组可行(m,n)配置，H100上12组。Tile selector运行时constant-time lookup：m用round-up规则（选≥CTA query数的最小可行m避免维度padding），n根据KV length做profiled piecewise决策（长KV偏大n降低concurrency减少tail bubble，短KV偏小n避免compute bubble）。Multi-stream forward为每种active tile配置创建独立CUDA stream。Long-KV split在CTA KV length超batch均值时沿KV维拆分。Ablation对比：PAT-compute（FastTree compute-oriented cost model）、PAT-naive（简单每node独立pack）、PAT-fixed（固定(64,128) tile）、PAT-serial（串行multi-kernel执行）。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  已开源：https://github.com/flashserve/PAT（MIT License），Zenodo DOI: 10.5281/zenodo.18217189，Docker镜像 flashserve/pat:ae。使用流程：
  1. 环境：x86-64 Linux, ≥64GB RAM, 200GB disk, A100-80GB, CUDA driver ≥550
  2. 拉取镜像：`docker pull flashserve/pat:ae`
  3. 启动容器：`docker run -it --gpus all --shm-size=64g -v ${PWD}/PAT:/workspace/PAT -w /workspace/PAT flashserve/pat:ae /bin/bash`
  4. Kernel benchmark：`cd /workspace/PAT/benchmark && bash ./run_kernel_bench.sh`（约1.5小时，复现Figure 11）
  5. 端到端实验：`bash ./run_e2e_bench_part.sh`（快速验证，8-10 GPU-hours）或`bash ./run_e2e_bench_full.sh`（完整实验，>60 GPU-hours）
  6. 生成图：`cd /workspace/PAT/plot && python eval_kernel_perf.py --log-file ../benchmark/kernel_perf.json`
  在vLLM中启用PAT仅需设置环境变量：`VLLM_ATTENTION_BACKEND=PAT`（开源版README示例为`VLLM_ATTENTION_BACKEND="PREFIX_ATTN"`，接口名可能存在小幅演进）。Multi-tile kernel移植到新GPU需重新基于shared memory/register/bandwidth约束和CUTLASS requirement推导等价tile set（论文在H100上验证了该procedure的通用性）。

