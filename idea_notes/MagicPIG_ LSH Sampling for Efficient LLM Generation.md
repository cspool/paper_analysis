## MagicPIG: LSH Sampling for Efficient LLM Generation

- baseline方法是什么？
  Baseline是TopK attention及其搜索近似变体（如Quest的block-level dynamic sparsity）。TopK attention仅选择attention scores最高的K个key-value对参与注意力计算，本质上是一个有偏估计（biased estimator）。其全栈执行例子：
  - 算法层：Quest使用page-level分块，计算q与每个page summary的内积近似TopK选择，page_size=16时Cost_1=1/16(搜索开销)+Cost_2=手动控制(稀疏计算开销)
  - 系统框架层：全注意力在GPU上执行，KV cache全部驻留在GPU HBM，FlashAttention / FlashDecoding进行IO-aware加速
  - Kernel层：GPU执行标准Softmax(qK^T/√d)V，memory-bound瓶颈
  - 缺陷：(1) TopK丢弃了低attention score tokens中大量有效信息(长尾分布下Top20% token仅覆盖70-80% attention score)，在聚合任务(CWE, FWE)中准确率显著下降；(2) 搜索TopK本身开销大(IVF需访问30% key才能获得精确TopK)；(3) KV cache仍全量驻留GPU显存，限制了最大batch size和context length。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MagicPIG提出基于LSH的采样方法来估计attention输出，替代TopK的确定性选择。核心思想是将attention output视为从attention score分布w中采样的期望值o=E_{i~w}[v_i]，通过Self-normalized Importance Sampling + LSH SimHash实现高效的无偏/近似无偏估计。全栈执行例子：
  - 算法层：q在GPU上计算K×L bit SimHash码 → CPU上查询L张哈希表，收集至少2表中碰撞的key集合S → 计算每个采样key的碰撞概率u_i = 1-(1-p_i^K)^L-L·p_i^K·(1-p_i^K)^{L-1} → 注意力估计ō=Softmax(qK_S^T/√d - log(u))·V_S。关键创新：(a) centering预处理解决q和k方向几乎相反导致LSH失效的问题，(b) 至少2表碰撞机制提升采样质量，(c) on-device cache保留sink+local tokens避免丢失关键信息。
  - 系统框架层：GPU执行compute-bound的线性投影和HashEncode(Cost_1≈0)，CPU执行memory-bound的哈希表查询和稀疏注意力(Cost_2=2%~5%全注意力FLOPs)，KV cache完整offload到CPU DRAM
  - Kernel层：GPU PyTorch执行线性层+随机投影(3.8%~8.5%额外计算)，CPU FBGEMM bfloat16执行稀疏qK^T和weighted sum
  - 对应解决：(a) 采样比TopK更准确——oracle sampling减少4×估计误差，在CWE和FWE任务上MagicPIG甚至超过exact TopK 3-8%；(b) LSH采样Cost_1≈0，远低于TopK搜索的3-6%；(c) KV cache offload到CPU DRAM使batch size达baseline的12×，突破GPU显存限制。
