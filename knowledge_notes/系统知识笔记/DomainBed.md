## DomainBed

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DomainBed 是 Gulrajani & Lopez-Paz (ICLR 2021) 提出的域泛化（Domain Generalization）标准评测框架和基准测试套件。它是一个 PyTorch 测试床（testbed），统一了 DG 方法的实现、训练协议和评估流程，解决了此前 DG 研究评估标准不一致（不同论文使用不同超参、数据划分、模型选择协议）的问题。DomainBed 的关键贡献：(1) 标准化 leave-one-domain-out 评估协议——每个域轮流作为目标域测试，其余为源域，源域中 20% 随机划分作为验证集用于模型选择；(2) 统一下游实现——所有方法使用相同的 ResNet-50 ImageNet 预训练骨干、Adam optimizer (lr=5e-5, no weight decay)、batch_size=32 per-domain；(3) 包含 ERM、IRM、GroupDRO、Mixup、MLDG、CORAL、MMD、DANN、CDANN、MTL、SagNet、ARM、VREx、RSC 等 14 种 DG 算法的统一实现；(4) 覆盖 PACS、VLCS、OfficeHome、TerraIncognita、DomainNet 五个图像分类数据集。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
DomainBed 作为 DG 评测系统架构的设计：
```
训练框架：
  算法注册表 (algorithms.py) → 统一接口: def update(alg, minibatches, device)
  数据加载器 (datasets.py) → 按域加载 + 20% 验证集划分
  训练循环 (train.py) → 
    for step in range(total_steps):
        minibatches = sample from each source domain
        step_vals = alg.update(minibatches, device)
        if step % checkpoint_freq == 0:
            val_acc = evaluate(alg, val_loader)
            记录结果
  结果收集 (collect_results.py) → 多 trial 平均 + 标准误
```
QT-DoG 完全遵循 DomainBed 协议：使用 DomainBed 默认超参和训练循环，仅在 ERM 算法中嵌入 QAT（LSQ 量化模块），训练步数与其他方法对齐（5000 步大部分数据集，15000 步 DomainNet），量化在 2000/8000 步启动。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DomainBed 的使用：(1) 安装：`pip install domainbed` 或从 GitHub（facebookresearch/DomainBed）clone；(2) 训练：`python -m domainbed.scripts.train --algorithm ERM --dataset PACS --test_env 0`；(3) 结果收集：`python -m domainbed.scripts.collect_results --input_dir results/`；(4) 新算法集成：在 algorithms.py 中实现 `class Algorithm(torch.nn.Module)`，覆盖 `update()` 和 `predict()` 方法。DomainBed 的局限：(a) 仅支持图像分类任务；(b) 默认仅 ResNet-50 骨干；(c) 模型选择协议（训练域验证）与实际场景（无目标域标签）对齐但偏乐观。

涉及论文标题：
- QT-DoG Quantization-Aware Training for Domain Generalization
