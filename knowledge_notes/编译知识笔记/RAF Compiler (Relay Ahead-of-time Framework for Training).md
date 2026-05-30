## RAF Compiler (Relay Ahead-of-time Framework for Training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

RAF (Relay Ahead-of-time Framework) 是 AWS 开发的开源深度学习训练编译器（https://github.com/awslabs/raf），扩展自 Apache TVM（Chen et al., OSDI 2018）。与 TVM 主要面向推理不同，RAF 专注于训练工作负载的全栈编译优化。核心特点：(1) 接受前向模型，内部自动生成完整训练图（包括反向传播），使编译器可跨前向/反向边界全局优化；(2) Operator Dialect 机制无缝集成手写 kernel（cuDNN）、张量编译器（TVM Ansor/AutoTVM）和 vendor 库；(3) 原生支持自动微分、自动混合精度（AMP）、图优化和分布式训练。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

RAF 编译流程（Lancet 集成方式）：

```
Input: PyTorch 前向模型定义
  ↓
1. RAF Frontend: 模型 → Relay IR
  ↓
2. AutoDiff: 自动生成反向传播 IR → 完整训练图
  ↓
3. Optimization Pass Pipeline (Pass Manager):
  ├── Graph-level optimizations (算子融合、死代码消除、内存规划)
  ├── [Lancet] Pass 1: Weight Gradient Computation Schedule Pass
  ├── [Lancet] Pass 2: Operator Partition Pass
  ├── Auto Mixed Precision (AMP)
  └── Memory optimization
  ↓
4. CodeGen: IR → 可执行代码
  ├── 手写kernel → Operator Dialect 直接调用
  ├── 编译器生成kernel → TVM Ansor/AutoTVM 自动调优
  └── 通信算子 → NCCL
  ↓
5. Runtime Execution: 优化后模型在 GPU 集群上训练
```

Lancet 在 RAF 上新增约 13K LoC C++，作为两个 IR Pass 实现。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

RAF 通过在 TVM 基础上增加训练专用抽象（自动求导、AMP、分布式训练）区别于 TVM。Lancet 选择 RAF 而非直接使用 TVM 的关键原因：RAF 提供完整的训练图 IR（含前向和反向），编译器可跨 pass 边界进行 dW 调度和算子分区优化。用户通过 RAF Pass Manager 配置启用 Lancet 优化，无需修改 Python 训练代码。开源代码包含 profile/opt/benchmark 脚本：`python run_exp_configs.py --lancet-profile`（profile）、`python run_exp_configs.py --lancet-opt`（优化运行）、`python run_exp_configs.py`（baseline）。

涉及论文标题：
- Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping
