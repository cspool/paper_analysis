## TAGT: An Efficient Graph Transformer Accelerator with Topology-aware Sparsification and Merging

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现为 Topology Dependency Subgraph (TDS) 拓扑感知稀疏化与合并方法：把图 Transformer 的全局 O(N²) 注意力近似为在稀疏子图 TDS 上的注意力，把每个顶点 attend 的边数从 O(N) 降到平均 O(m·log_m N)（m=2 时即 O(log N)），总边数 O(m·N·log_m N)=O(N log N)。TDS 由三类边构成：(1) original edges——保留输入图的原始局部邻域结构；(2) fusion edges——自底向上的分层聚合边：沿原生 1D 输入顺序每次递归合并 m 个内存连续顶点形成 fusion 顶点（约 log_m N 层直到单根），fusion 顶点持有其全部子顶点的聚合特征，作为"高阶代理顶点"保留远程/全局上下文；(3) association edges——指向目标顶点的逐层递归边：每层从目标顶点左右两侧各取 m 个关联顶点（起始下标 p_{l+1}=parent(p_l+m)，若 p_l+m-1 为奇数则再纳入下一个顶点且 p_{l+1}=parent(p_l+m+1) 保证集合互斥），覆盖多粒度远程上下文。TDS 总顶点数约 N + N/(m-1)。构造保证任意两个原始顶点通过 fusion+association 边最多 2-hop 可达，使每个目标顶点的 1-hop 注意力邻域同时含局部邻居、多粒度上下文与全局根顶点，一次稀疏注意力即达到多跳 message passing 的全局效果。
  - 实验比较：(1) 软件层——TAGT-S（用 TDS 方法修改 DGL v2.4.0 的软件实现，跑在 A100 GPU）对比 DGL-CPU（保留 O(N²) 全局注意力的 CPU baseline）与 TorchGT（SOTA GT 训练框架）：TAGT-S 比 TorchGT 快 1.8×–2.5×（TorchGT 依赖 Hamiltonian path 前提，现实图不满足时回退 O(N²) 全局注意力），不同序列长度下带宽利用率 >60% vs TorchGT；(2) 准确率（Table VI，序列长度固定 16K、100 次独立运行平均）——TAGT vs DGL-CPU 全注意力参考：全部数据集/模型上准确率下降 <1pp（GT 0.11–0.91pp、Graphormer 0.03–0.55pp、UGformer 0.22–0.84pp、EGformer 0.08–0.88pp），且 TAGT 准确率高于 TorchGT；(3) 训练效率——TAGT 相对全局注意力 baseline 有 >3× 训练与收敛加速（Fig.4）。
- 硬件平台是什么，配置是什么。
  - TAGT-S 软件实现跑在 NVIDIA Tesla A100 GPU（6,912 cores、80GB HBM）；DGL-CPU baseline 跑在 32 核 Intel Xeon Platinum 8357B @2.6GHz、503GB DDR4 RAM、16 内存通道。TAGT 硬件实现：Xilinx Alveo U280 FPGA（见 硬件架构 层条目）。
- 模型是什么。数据集和bench分别是什么。
  - 模型（Table III，按原论文超参）：Graph Transformer (GT) [Dwivedi & Bresson, 2020]（4 层、hidden 128、12 head）、Graphormer [Ying et al., 2021]（4 层、hidden 768、8 head）、UGformer [Nguyen et al., 2022]（4 层、hidden 384、4 head）、Edge Transformer / EGformer [Bergen et al., 2021]（8 层、hidden 200、4 head）。数据集（Table II）：Yelp (YP)（716,847 顶点、13,954,819 边、300 维特征、100 类分类）、Reddit (RD)（232,965 顶点、114,615,892 边、602 维、41 类分类）、Ogbn-Arxiv (OA)（169,343 顶点、1,166,243 边、128 维、40 类分类）、Ogbn-Products (OP)（2,449,029 顶点、61,859,140 边、100 维、47 类分类）、Ogbn-Papers100M (PM)（111,059,956 顶点、1,615,685,872 边、128 维、172 类分类）。任务均为节点分类。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文未给出 TAGT/TAGT-S 开源链接；联网搜索（2026-08）未找到官方仓库，无法确认是否开源。baseline 开源：DGL（https://github.com/dmlc/dgl，v2.4.0）、TorchGT（https://github.com/hengruizhang98/torchgt）、Graphormer（https://github.com/microsoft/Graphormer）。
  - TDS 构造伪代码（m=2，沿 1D 输入顺序）：
    ```
    # 输入：顶点特征 x[0..N-1]（1D 内存顺序），合并基数 m，原始边集 E_orig
    # 阶段1：自底向上分层聚合——生成 fusion 顶点及其特征
    cur = list(range(N))                      # 基层：原始顶点为 leaves
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), m):       # 每 m 个内存连续顶点合并一次
            fus = 新建 fusion 顶点
            feature(fus) = aggregate(feature(c) for c in cur[i:i+m])   # 如 mean/sum 聚合
            for c in cur[i:i+m]: 添加 fusion 边 c -> fus               # 底向上有向边
            nxt.append(fus)
        cur = nxt                             # 上一层（约 log_m N 层到根）
    # 阶段2：为目标顶点构造 association 边（逐层左右各取 m 个关联顶点）
    for 目标顶点 v_k (k 为 1D 下标):
        for 右侧（左侧对称，用 k-1 递减）:
            p = k + 1
            for l in range(log_m N):
                在层 l 取下标 p..p+m-1 的 m 个顶点，添加 association 边 v_k -> 它们
                if (p + m - 1) 为奇数:  p = parent(p + m + 1)   # 互斥机制
                else:                   p = parent(p + m)
    # 阶段3：目标顶点注意力——只在 TDS 1-hop 邻域（original + fusion + association 边）上做
    ```
  - 张量计算示例（一个目标顶点 v 的一层更新，式1）：H^v = concat({h_u^l | u ∈ N_TDS(v)})，其中 N_TDS(v) 为 TDS 上 v 的 1-hop 邻域（K = O(m·log_m N) 个顶点，远小于 N）；注意力输出 h̄_v^{l+1} = softmax( h_v^l·W_Q · (H^v·W_K)^T / √d_K ) · (H^v·W_V)；随后 h_v^{l+1} = FFN(h̄_v^{l+1}) + h̄_v^{l+1}（残差）。Graphormer 类结构编码可融入初始嵌入（h_v^{(0)} = x_v + z^-_{deg^-(v)} + z^+_{deg^+(v)}）与注意力偏置（+ bias_{φ(v,u)}，φ 为最短路径距离）。总注意力边数 O(m·N·log_m N)，m=2 时即 O(N log N)。理论误差界（式2-4）：‖Δh_i‖₂ ≤ L‖V‖₂·Σ_{j∉T_i(m)} α_ij + ε_fus(m)，在注意力量重尾衰减 α_{i,(k)} ≤ c·k^{-β}（β>1）假设下 ≤ O((m·log_m N)^{1-β}) + ε_fus(m)——误差由结构截断与 fusion 粗粒度双因素决定，m 控制保真度-效率权衡，m=N 时退化为精确 O(N²) 全局注意力，m=2 为准确率最优（Fig.15d）。
