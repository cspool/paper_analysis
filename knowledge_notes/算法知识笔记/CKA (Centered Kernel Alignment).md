## CKA (Centered Kernel Alignment)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CKA (Centered Kernel Alignment) 是 Kornblith et al. (ICML 2019) 提出的神经网络表示相似度度量方法。通过比较两个模型在相同输入上的**样本间相似结构**（而非直接比较特征向量）来衡量表示相似度。公式：CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) · HSIC(L, L))，HSIC(K, L) = Tr(K H L H)，K/L 是激活值的核矩阵（如线性核 K=XX^T），H = I - (1/n)1·1^T 是中心化矩阵。值∈[0,1]。关键特性：(1) 对正交变换不变；(2) 对同向缩放不变；(3) **不对任意可逆线性变换不变**——使其能捕捉有意义的结构差异，而 CCA 等方法在低样本/高维度下失效；(4) 可比较不同 shape 的表示矩阵（p1≠p2），传统 cosine similarity 无法做到。

从算法pipeline角度拆解术语，给出具体例子。
Mordal 中 CKA 的两步聚类流程：
```
// Step 1: Vision Encoder CKA
for each pair (VE_A, VE_B):
    act_A = VE_A(images)  // [N, d_ve_A]
    act_B = VE_B(images)  // [N, d_ve_B]
    K = act_A @ act_A.T; L = act_B @ act_B.T
    H = I - 1/N * ones(N,N)
    cka = Tr(K@H@L@H) / sqrt(Tr(K@H@K@H) * Tr(L@H@L@H))
    dist = 1 - cka
C_ve = HierarchicalClustering(dist, t_ve=0.7)

// Step 2: LLM CKA (per VE cluster)
medoid_ve = PickMostCentral(VE_cluster)
fixed_output = WarmupProjector(medoid_ve(images))  // 统一shape
for each pair (LLM_A, LLM_B):
    rep_A = LLM_A.last_hidden_state(fixed_output)
    rep_B = LLM_B.last_hidden_state(fixed_output)
    cka = CKA(rep_A, rep_B)
// → LLM clusters → Cartesian product → VLM candidate clusters
```
Mordal 选择 CKA 的两个关键原因：(1) 不同 VE 输出维度不同（d_CLIP≠d_SigLIP），CKA 通过核矩阵投影回避维度对齐；(2) MLP Feature Projector 的变换不影响 CKA 的鲁棒性。

术语一般如何实现？如何使用？
Mordal 使用 MinibatchCKA (Nguyen et al. 2020, Raghu et al. 2021) 支持大数据集，`scipy.cluster.hierarchy` 进行层次聚类。PyPI 包：`cka`。论文验证：ScienceQA 和 VizWiz 上相似 CKA 表示的 VE 产生相似性能。适用场景：模型调试（检测冗余层）、迁移学习（识别 freeze 层）、架构比较。局限：对异常值敏感，不满足三角不等式。

涉及论文标题：
- Mordal: Automated Pretrained Model Selection for Vision Language Models

---
