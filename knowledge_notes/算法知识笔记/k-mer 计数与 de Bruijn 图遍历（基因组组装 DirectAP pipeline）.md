## k-mer 计数与 de Bruijn 图遍历（基因组组装 DirectAP pipeline）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
de novo 基因组组装的两步核心：k-mer 计数——把 reads 切成全部长度 k 子串并统计频数（用于错误校正与图中节点权重/过滤低深度 k-mer）；de Bruijn 图构建与简化遍历——节点=k-mer、边=相邻 k-mer 重叠 k−1 个碱基，简化掉 tips/bulges、把非分叉路径合并成 unitig，最终拼接出 contig。SPAdes（Bankevich 2012，web：https://github.com/ablab/spades）是多 k 迭代 + 错误校正（BayesHammer）的代表实现，BAAP 用它作多核 CPU 参考实现。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 基因组组装 pipeline（论文 §IV-E，BAAP 在 UPMEM DIMM 上）
for k in K:                            # 多 k 扫描（BAAP 单 k 演示）
  counts = {}
  for read in reads:                   # host 阶段：k-mer 计数
    for i in 0..len(read)-k:
      counts[read[i:i+k]] += 1
  # BAAP DirectAP：计数改为 bank 内穷举 CAM 匹配（ap_regex 序列）
  G = deBruijn(counts, k)              # 节点=k-mer、边=重叠 k-1
  G.remove_tips(); G.remove_bulges()
  unitigs = G.collapse_nonbranching()  # 合并非分叉路径
  traversal_order = AP-BFS(G, start)   # BAAP 算法 1：tag 编码前沿的图遍历
```
BAAP 映射（论文 §IV-E）：UPMEM 基线 k-mer 计数 = 线性扫 bank + 哈希表更新，遍历 = DPU 标量前沿管理；DirectAP 把两步都变成 WRAM 上的关联查询——计数 2–38×（随 k）、遍历 1.1–2.8×；k>21 时 2^k 搜索空间爆炸、中间图可占满整条 DIMM，回退 16 核 host。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
工具链：SPAdes（多 k 迭代、--isolate/--careful 模式）、KMC（k-mer 计数）。数据集：BAAP 用真实 A. thaliana（SRR29124148）。适用：错误校正、宏基因组学、群体研究（小 k 值场景同样常见）。硬件映射要点：k-mer 匹配是"重复成员查询于大而稀疏状态"的典型 DirectAP 形态；跨 DPU 前沿交换必须经 host 中转，是 k 增大时的主要瓶颈。

涉及论文标题：
- BAAP: Coupling Compute-in-SRAM with DRAM Banks for Near-Memory Processing
