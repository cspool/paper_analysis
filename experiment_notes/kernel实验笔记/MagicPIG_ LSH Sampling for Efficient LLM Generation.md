## MagicPIG: LSH Sampling for Efficient LLM Generation

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  GPU-CPU异构计算调度：GPU负责compute-bound的线性投影和随机投影（HashEncode），CPU负责memory-bound的KV cache哈希表查询和稀疏注意力计算。具体运行时流程：GPU上PyTorch执行所有线性层和LSH随机投影，然后将hash code和新KV传输到CPU；CPU上运行FBGEMM bfloat16内核执行稀疏qK^T和weighted V求和；CPU结果通过recursive attention与GPU上的on-device cache结果合并。实验比较了不同(K,L)超参数下的延迟和吞吐量（Table 7），以及三种硬件场景的系统性能（Figure 8）。

- 后端平台是什么，配置是什么。
  GPU: NVIDIA A100 (80GB HBM), L20 (48GB, 864GB/s带宽), 模拟RTX 4090 (24GB, ~1TB/s带宽)。CPU: Intel Platinum 8480+ / Intel 8563C。CPU DRAM带宽按150GB/s估算（论文实测group query attention size=4下的经验带宽）。

- 评估性能的软件/脚本是什么。修改了什么。
  GPU端：原生PyTorch (Paszke et al., 2019)。CPU端：FBGEMM (Khudia et al., 2021) bfloat16精度。修改内容：(1) 新增GPU端HashEncode kernel：q_code = Sign(q @ W)，将d维向量投影到K×L bit哈希码，所有attention head共享W，内存开销400KB~825KB；(2) 新增CPU端LSH哈希表数据结构：每个KV head L张哈希表，存储所有key的索引和哈希码，内存开销随context length和head数线性增长（如Llama-3.1-8B 96K context: 14GB for (10,150)）；(3) CPU端稀疏注意力计算：FBGEMM执行q·K_S^T的内积计算和加权求和。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源链接: https://github.com/Infini-AI-Lab/MagicPIG。评估原理和流程：
  评估原理：测量端到端LLM解码的wall-clock性能指标（token间延迟TBT、最大吞吐量tokens/sec、200ms延迟约束下的吞吐量Throughput200ms），量化不同(K,L)配置下的计算开销(Cost_2 = 采样后稀疏注意力FLOPs / 全注意力FLOPs)与系统性能的关系。
  Kernel输入到性能输出全过程：
    输入：prefill后的KV cache (n×d)、模型参数、LSH超参数(K,L)、随机投影矩阵W
    GPU端：每个decode step执行
      - 线性层投影 (compute-bound, GPU利用率高)
      - HashEncode: q_code = Sign(q @ W) → 传输到CPU
    CPU端：
      - HashTable查询：L次查找，收集S = {i | collision_count_i ≥ 2}
      - 稀疏注意力：加载K_S, V_S → FBGEMM计算q·K_S^T → softmax(· - log(u)) → Σ w_i·v_i
      - 结果传回GPU → recursive attention合并 → 输出
    性能输出：TBT (time between tokens, ms)、最大吞吐量(可容纳的最大batch size × 每秒tokens)、Throughput200ms (200ms延迟SLO下的最大吞吐量)
    关键发现：(K,L)=(10,150)下TBT=18.31ms、最大吞吐53.78 tokens/s、Throughput200ms=48.89 tokens/s；KV cache offload使batch size可达baseline的12×以上。
