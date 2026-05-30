## GraphMETRO Mitigating Complex Graph Distribution Shifts via Mixture of Aligned Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 GraphMETRO：一种基于 Mixture-of-Experts (MoE) 架构的 GNN 泛化方法，将未知分布偏移（distribution shift）分解为多个 shift components，通过 gating model + multiple expert models 生成 referentially invariant representations。核心实现：(1) Gating model ϕ（GNN encoder）输入图数据输出权重向量 w ∈ R^{K+1}，预测各 shift component 的贡献；(2) K+1 个 Expert models {ξ_i}，每个对应一个 shift component，其中 ξ_0 为 reference model，其他 expert 对其分配的 shift component 产生 referentially invariant representations（定义：ξ_0(G) ≈ ξ_i(τ_i(G))，∀G ∈ supp(D_s)）；(3) 对 expert 输出进行 softmax 加权聚合 h(G) = Softmax(w) · [z_0, ..., z_K]^T；(4) 训练目标 L = L1 + L2：L1 为 BCE loss 优化 gating 预测混合成分，L2 为 CE + λ·Frobenius distance loss 优化 expert 分类和与 reference model 的对齐（L2 不反向传播到 gating model）。
  - 实验比较：(1) 真实数据集（GOOD benchmark）：WebKB、Twitch、Twitter、GraphSST2 上 vs ERM、DANN、IRM、VREx、GroupDRO、Deep Coral、SRGNN、EERM、OODGAT、DIR、G-Mixup、GSAT、CIGA 的分类准确率/ROC-AUC；(2) 合成数据集（DBLP、CiteSeer、IMDB-MULTI、REDDIT-BINARY）上 vs ERM 和 ERM-Aug 在不同单/多 shift component 组合（14种环境）下的准确率；(3) Ablation：移除 L1 loss、shared encoder vs independent encoder、移除 alignment term (λ=0)、不同数量和类型的 transform functions（2-7个）；(4) Invariance matrix 可视化（验证每个 expert 对其对应 shift component 的不变性）；(5) Distribution shift discovery（gating model 输出的 mixture 揭示目标分布的 shift 类型，如 WebKB 以 "add_edge" 为主导，Twitch 以 "noisy_node_feat" 和 "drop_node" 为主）。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA GPU（论文 checklist 声明提供了 GPU 信息，但正文未列出具体 GPU 型号）。论文未明确说明具体 GPU 型号或数量。

- 模型是什么。数据集和bench分别是什么。
  - 模型：
    - 真实实验 backbone：Graph Convolutional Network (GCN) [25]（node-level：WebKB, Twitch），Graph Isomorphism Network (GIN) with Virtual Node [70, 18]（graph-level：Twitter, SST2）
    - 合成实验 backbone：Graph Attention Networks (GAT) [63]
    - GraphMETRO 架构：gating model ϕ（一个 GNN encoder），K+1 个 expert models ξ_i（每个为独立 GNN encoder，或共享 GNN + 独立 MLP），classifier μ（MLP + softmax）
    - 激活函数：ReLU（真实）/ PReLU（合成）
  - 数据集：
    - 真实数据集（来自 GOOD benchmark [20]）：
      - WebKB：5-class 节点分类，按大学域名划分 train/test split（natural covariate shift）
      - Twitch：二分类节点分类（预测是否 streaming mature content），按用户语言域划分 split，metric=ROC-AUC
      - Twitter：grammar tree graph 分类（不同 domain 的句子长度和语言风格不同）
      - GraphSST2：sentiment tree graph 分类，metric=accuracy
    - 合成数据集：
      - 节点分类：DBLP [16]、CiteSeer [73]
      - 图分类：IMDB-MULTI、REDDIT-BINARY [46]
  - Benchmark：GOOD (Graph Out-of-Distribution) benchmark [20]

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源链接：https://github.com/Wuyxin/GraphMETRO
  - 基于 PyG (PyTorch Geometric)：https://github.com/pyg-team/pytorch_geometric
  - GOOD benchmark：https://github.com/divelab/GOOD/tree/GOODv1
  - 算法 pipeline 伪代码：

```
# 输入：图 G，源分布 Ds，K 个 stochastic transform functions {τ_i}_{i=1}^K
# 模块定义：
#   ϕ: Gating GNN，输出 w ∈ R^{K+1}（各 shift component 的权重）
#   {ξ_i}_{i=0}^K: Expert GNNs (独立 GNN encoder)
#   μ: Classifier (MLP + softmax)

# ===== 训练阶段（每次梯度步） =====
for (G, y) in Ds:
    # 1. 采样 joint stochastic transform τ^{(k)}（k 个 transform 的组合）
    τ^{(k)} = sample_k_transforms({τ_1, ..., τ_K})
    G_transformed = τ^{(k)}(G)
    
    # 2. Gating Loss L1（预测 mixture）
    w = ϕ(G_transformed)  # w ∈ R^{K+1}
    Y_gt[i] = 1 if τ_i in τ^{(k)} else 0  # ground truth binary vector
    L1 = BCE(w, Y_gt)
    
    # 3. Expert Loss L2（分类 + 对齐，不反向传播到 ϕ）
    z_i = ξ_i(G_transformed) for i = 0..K  # 每个 expert 生成表示 [z_i ∈ R^v]
    z_0_ref = ξ_0(G)  # reference model 在原图上的表示
    
    # Softmax 加权聚合
    w_norm = Softmax(w)
    h = Σ_{i=0}^{K} w_norm[i] · z_i  # h ∈ R^v
    
    # 分类 + Frobenius 距离对齐
    y_pred = μ(h)  # classifier 输出
    d = (1/n) · ||h - z_0_ref||_F  # Frobenius norm, λ=1
    
    L2 = CE(y_pred, y) + λ · d  # d 为 referential alignment 项
    
    # 总 loss
    L_total = L1 + L2
    # 梯度更新：L1 → ϕ; L2 → {ξ_i}, μ (不更新 ϕ)

# ===== 推理阶段 =====
for G_test in D_test:
    w = ϕ(G_test)  # gating 预测 shift mixture
    z_i = ξ_i(G_test) for i = 0..K
    h = Softmax(w) · [z_0, ..., z_K]^T  # weighted sum aggregation
    y_pred = μ(h)  # 最终预测
```

  - **Stochastic Transform Functions**（基于 PyG 构建，共 11 种，论文实验中用了 5 种）：
    ```
    {mask_edge_feat, noisy_edge_feat, edge_feat_shift,
     mask_node_feat, noisy_node_feat, node_feat_shift,
     add_edge, drop_edge, drop_node, drop_path, random_subgraph}
    ```
    每个函数允许一或多个超参数控制变换程度（如 Bernoulli drop probability 在 [0.3, 0.5]），保留随机性确保多样性。
  - **关键超参数**：Adam optimizer, weight_decay=0, hidden_dim=64(CiteSeer:32)/128(graph)/300(Twitter/SST2), num_layers=2(graph)/3(node), dropout=0.0(合成)/0.5(真实), learning_rate=1e-3 或 1e-2（GraphMETRO 使用更高的 lr 做 grid search）, epoch=100-200, batch_size=32(graph)/NA(node)。
  - **训练计算复杂度**：前向 O(K) encoder passes，训练 O(K²|Ds|)（因 extrapolation 将数据集扩大 K+1 倍）。
