## Domain Generalization (DG / 域泛化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Domain Generalization（DG，域泛化）是一种机器学习范式，目标是从多个源域（source domains）学习一个模型，使其能在未见过的目标域（target domain）上良好泛化。与 Domain Adaptation（需要目标域未标注数据）不同，DG 在训练期间完全无法访问目标域数据。DG 的核心挑战是防止模型对源域过拟合——模型可能学到源域特有的虚假相关性（spurious correlations，如纹理、背景、光照），而非真正的类别判别特征。DG 的标准评估协议是 DomainBed（Gulrajani & Lopez-Paz, ICLR 2021）：每个域轮流作为目标域，其余作为源域，源域中 20% 作为验证集用于模型选择，在目标域上评估并报告平均准确率和标准差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DG 训练以 DomainBed 协议 + ResNet-50 为例：对每个目标域（leave-one-domain-out），源域 80% 训练 20% 验证，ResNet-50 ImageNet 预训练初始化，Adam (lr=5e-5, no weight decay)，batch_size=32 per-domain，训练 5000 步（DomainNet 15000 步），每 300 步验证选模。主要 DG 方法类别：(1) Domain Alignment——对齐源域特征分布（CORAL, DANN, MMD）；(2) Regularization——正则化抑制源域特有特征（IRM, VREx）；(3) Weight Averaging——平均权重以找平坦极小值（SWAD, DiWA）；(4) Ensembling——集成多模型（EoA）；(5) CLIP-based——利用大规模多模态预训练。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DomainBed 是 DG 的标准评测框架（GitHub: facebookresearch/DomainBed），提供统一的数据加载和评估协议，支持 PACS、VLCS、OfficeHome、TerraIncognita、DomainNet 五大数据集。QT-DoG 在该框架上引入 QAT 作为隐式正则化，单模型 7-bit 达 66.2% DomainBed 平均（ERM 63.8%），体积减少 4.6×。

涉及论文标题：
- QT-DoG Quantization-Aware Training for Domain Generalization
