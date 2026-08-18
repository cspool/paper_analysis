## Activation Rematerialization（激活重物化）与 ILP 图调度（训练内存优化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Activation Rematerialization（激活重物化 / 梯度检查点 gradient checkpointing）是训练内存优化：不保存所有前向激活，反向时按需重算部分层激活，以计算换内存。ILP 图调度（整数线性规划）是 MTIA 300 编译器的图调度器启发式：把算子调度到训练迭代中，使每迭代峰值内存最小。二者是 MTIA 300 训练内存优化套件的两部分（论文软件栈节）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
MTIA 300 中的运转流程：
```python
# 图调度器（ILP 启发式）: 在计算 DAG 上做调度/内存分配
#   目标: min(峰值内存) 约束: 算子执行序、live-range、LS/HBM 容量
schedule = ilp_scheduler(graph, mem_budget)
# Activation rematerialization: 选择部分算子重算而非保存
for node in schedule:
    if node.activation in checkpoint_set:   # 编译器选中的重算点
        save_none(node)                      # 反向时重算 node 的前向
    else:
        save_activation(node)
# 效果: 更大有效 batch（10240 vs 6144）+ 更高 HBM 利用
```
对比 H100：MTIA 300 用大 HBM（216 GB）+ 这些优化支撑 local batch 10240（24 卡、Perf/TCO 1.42×），H100 只能 6144（40 卡）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：rematerialization 是标准 checkpoint 技术（PyTorch 有 activation checkpointing API，MTIA 编译器在图中自动选择重算点）；ILP 调度为编译期离线求解（论文未给 ILP 模型细节）。使用场景：DLRM 训练（内存压力大：150B 参数 + 优化器状态 + 激活）与任何大 batch 训练。信息缺口：论文未给出 ILP 变量/约束/求解器与重算点选择策略细节。

涉及论文标题：
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
