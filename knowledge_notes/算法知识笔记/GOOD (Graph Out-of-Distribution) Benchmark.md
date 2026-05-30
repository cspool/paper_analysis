## GOOD (Graph Out-of-Distribution) Benchmark

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

GOOD (Graph Out-of-Distribution) Benchmark 是 NeurIPS 2022 提出的 GNN 分布外泛化基准（Gui et al. 2022）。区别于标准 ML benchmark 的随机数据划分，GOOD 提供基于真实 covariate shift 的数据集划分（如按大学域名、用户语言域、分子 scaffold 划分 train/val/test），模拟真实部署中的自然分布偏移。GraphMETRO 使用四个 GOOD 数据集：WebKB（5-class 节点分类，按大学域名划分，target=Washington domain）、Twitch（二分类，按用户语言域划分，metric=ROC-AUC）、Twitter（grammar tree graph 分类，不同 domain 的句子结构形成 shift）、GraphSST2（sentiment tree graph 分类）。GOOD 提供标准化 encoder/classifier 和评估协议。代码：https://github.com/divelab/GOOD/tree/GOODv1。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# GOOD benchmark 使用 PyG 接口
from GOOD import get_dataset
dataset = get_dataset(root='./data', dataset_name='WebKB', domain='domain')
# train: Cornell/Wisconsin/Texas, val/test: Washington (natural domain shift)
# 评估不提供 domain 标签（GraphMETRO 不需要 domain info）
# GraphMETRO 使用 GOOD 的统一架构：GCN for node tasks, GIN+VirtualNode for graph tasks
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GOOD 提供统一的数据加载、encoder、classifier 和评估协议，确保不同方法间的公平比较。GraphMETRO 使用 GOOD 的标准 GCN/GIN 作为 backbone，与 ERM、IRM、EERM、DIR、GSAT 等 baseline 在同一框架下比较。训练/验证/测试划分固定，减少划分随机性对结果的影响。

涉及论文标题：
- GraphMETRO Mitigating Complex Graph Distribution Shifts via Mixture of Aligned Experts
