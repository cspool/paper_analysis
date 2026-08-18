## DLRM 训练（Deep Learning Recommendation Model，训练视角）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DLRM（Deep Learning Recommendation Model）是 Meta 的推荐模型架构：dense 特征（如用户年龄）过 MLP，sparse 特征（如 post ID）查 embedding 表（table-batched embedding，TBE），经 dense interaction 层连接后进最终 MLP 输出预测（点击率/参与度）。训练视角（MTIA 300，ISCA'26）：DLRM 训练与 GenAI 训练不同——FLOPS 需求中等（单样本 ~3 GFLOPs）但 HBM 容量/带宽与网络带宽需求大、collective 通信频繁，常致加速器利用率低。MTIA 300 的生产 DLRM 训练模型约 150B 参数（99% 在稀疏侧，embedding 表常超单卡容量故需混合并行分片），用 TorchRec 实现 + TorchInductor 全图编译 + 分布式 Shampoo 优化器 + 分布式数据并行。性能：40 卡 local batch 6144 时通信超 H100 3.9×、端到端 Perf/TCO 1.42×（Table IV）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
一次 DLRM 训练迭代的算法 pipeline（MTIA 300）：
```python
# 前向
for u in batch:                                    # 批量用户
    dense_vecs = MLP(dense_features[u])            # dense 特征 → MLP（PE 的 DPE）
    emb = TBE_forward(sparse_idx[u], tables)       # 稀疏特征 → 查 embedding 表
    out = interaction(dense_vecs, emb)             # dense interaction 层
    yhat = final_MLP(out)
loss = CE(yhat, label); loss.backward()            # AOTAutograd 生成反向图
# 反向: embedding 索引 radix-sort 重排 + TBE_backward + dense 梯度
# 优化: 分布式 Shampoo（AllGather 阶段）+ AllReduce 梯度同步 + AllToAllv 特征交换
# 通信画像（40 卡）: AllReduce 1.6 GB / AllGather 2.1 GB / 35×AllToAllv(1KB-1GB)
```
MTIA 300 侧 co-design：关闭 row-wise FP8 量化通信（+4.4%）、Shampoo 特征分解 offload host CPU（1:1）、local batch 10240（24 卡、+2%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：TorchRec（PyTorch 推荐域库）建模 + TorchInductor 编译（MTIA 与 H100 同栈对比）；嵌入表按 table-wise/row-wise 混合并行分片；Shampoo 分布式优化。使用场景：Meta 广告/短视频/好友流推荐训练；DLRM 与 GenAI（LLM）训练的系统需求差异是 MTIA 300 硬件设计（内置 NIC/ME/NMC、高 HBM bytes-to-FLOPS）的直接动机。演进：GenAI 影响下 DLRM 采用更大 dense 组件与 Transformer 结构（MTIA 400 提高 FLOPS 应对）。信息缺口：论文未给出具体 DLRM 网络结构（层数/维度）。

涉及论文标题：
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
